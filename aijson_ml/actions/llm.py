import json
import logging
import os
import tempfile
import base64
from collections import defaultdict
from json import JSONDecodeError
from typing import Optional, AsyncIterator, Any

import aiohttp
import openai
import pydantic
import tenacity

from aijson.models.config.action import ActionInvocation
from aijson.models.io import (
    DefaultModelInputs,
    DefaultOutputOutputs,
)
from aijson import StreamingAction, Field


from aijson_ml.utils.prompt_context import (
    RoleElement,
    PromptElement,
    QuoteStyle,
    TextElement,
)
from aijson.models.config.model import OptionalModelConfig, ModelConfig

import litellm

from aijson.models.json_schema import JsonSchemaObject
from aijson.utils.async_utils import Timer, measure_async_iterator
from aijson.utils.json_schema_utils import jsonschema_to_pydantic
from aijson.utils.secret_utils import get_secret
from aijson.utils.singleton_utils import SingletonContext

# for some reason if this is imported later it hangs consistently
try:
    import vertexai  # noqa
except:  # noqa
    pass

litellm.telemetry = False
litellm.drop_params = True
# litellm.set_verbose = True

# disable litellm logger
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.ERROR)

retry_errors = (
    aiohttp.ClientError,  # connection errors
    openai.OpenAIError,  # includes litellm errors
)

try:
    import anthropic

    retry_errors += (anthropic.AnthropicError,)  # anthropic errors
except ImportError:
    pass


class PromptEnvContext(SingletonContext):
    # push anthropic API key into env if not there, and
    # inject the GCP credentials from the base64 encoded environment variable
    # into an Application Default Credentials file,
    # using a temporary file

    def __init__(self):
        super().__init__()
        self.anthropic_env_var_bak = None
        self.gcp_env_var_bak = None
        self.file = None

    def enter(self):
        anthropic_api_key = get_secret("ANTHROPIC_API_KEY")
        if anthropic_api_key is not None:
            self.anthropic_env_var_bak = os.environ.get("ANTHROPIC_API_KEY")
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

        base64_encoded_credentials = get_secret("GCP_CREDENTIALS_64")
        if base64_encoded_credentials is not None:
            credentials_string = base64.b64decode(base64_encoded_credentials).decode(
                "ascii"
            )
            self.file = tempfile.NamedTemporaryFile(mode="w")
            self.file.write(credentials_string)
            self.file.flush()
            self.gcp_env_var_bak = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.file.name

    def exit(self, *args):
        if self.file is not None:
            self.file.close()

        if self.anthropic_env_var_bak is not None:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_env_var_bak
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        if self.gcp_env_var_bak is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_env_var_bak
        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]


prompt_env_context_singleton = PromptEnvContext()


class Inputs(DefaultModelInputs):
    model: Optional[OptionalModelConfig] = None
    quote_style: Optional[QuoteStyle] = Field(
        default=None,
        description="The quote style to use for the prompt. "
        "Defaults to XML-style quotes for Claude models and backticks for others.",
    )
    prompt: str | list[PromptElement] = Field(
        description="""
The prompt to send to the language model.  
Consists of multiple elements like text, roles, variables, links, and more.

See [prompting in-depth](https://github.com/asynchronous-flows/aijson?tab=readme-ov-file#prompting-in-depth) for more information.
""",
    )
    output_schema: None | dict[str, JsonSchemaObject] = Field(
        default=None,
        description="""
Optionally, a JSON schema forcing the language model to output structured data adhering to it.

The schema adheres to the standard, except for the `required` field – if it is not provided, all fields are required.

💡 WARNING: You must instruct the language model to generate JSON in your prompt.  
Some model providers (like Ollama) do not properly support this feature, and will not guarantee
adherence to the schema, but will still generate JSON.
""",
    )


class Outputs(DefaultOutputOutputs):
    _default_output = "response"

    result: str = Field(
        description="""
Use `my_prompt` or `my_prompt.response` instead of `my_prompt.result`.  
Alternatively, use `my_prompt.data` if `output_schema` input is specified.
""",
        deprecated=True,
    )
    response: str = Field(
        description="""
Text response given by the LLM.  
If `output_schema` input is specified, this is a JSON string. Use `my_prompt.data` for structured access instead.
""",
    )
    data: Any | None = Field(
        description="Structured data, abides by the JSON schema specified in `output_schema` input. `None` otherwise.",
    )


class Prompt(StreamingAction[Inputs, Outputs]):
    name = "llm"

    description = """
    Prompt the LLM with a message and receive a response.
    """

    version = 0

    @classmethod
    def construct_model_from_schema(cls, schema: dict) -> pydantic.BaseModel | None:
        try:
            # TODO support parital inference; for `var:` and `link:` in schema
            #  honestly rewrite `jsonschema_to_pydantic` to work on dict instead of this obj
            schema_object = JsonSchemaObject(
                type="object",
                properties=schema,
            )
        except ValueError:
            return None

        return jsonschema_to_pydantic(schema_object)

    @classmethod
    def narrow_outputs_type(
        cls, action_invocation: ActionInvocation
    ) -> type[Outputs] | None:
        class OutputsWithoutSchema(Outputs):
            data: None  # pyright: ignore[reportGeneralTypeIssues]

        if not hasattr(action_invocation, "output_schema"):
            return OutputsWithoutSchema
        schema = getattr(action_invocation, "output_schema")
        if schema is None:
            return OutputsWithoutSchema

        model = cls.construct_model_from_schema(schema)
        if model is None:
            return None

        class OutputsWithSchema(Outputs):
            data: model  # type: ignore

        return OutputsWithSchema

    def build_messages(
        self,
        message_config: str | list[PromptElement],
        model_config: ModelConfig,
        quote_style: None | QuoteStyle,
    ) -> list[dict[str, str]]:
        if quote_style is None:
            if "claude" in model_config.model:
                quote_style = QuoteStyle.XML
            else:
                quote_style = QuoteStyle.BACKTICKS

        messages = []
        current_role = "user"
        current_message_elements = []

        def deposit_messages(new_role: str):
            nonlocal messages
            nonlocal current_role
            nonlocal current_message_elements

            if current_message_elements:
                messages.append(
                    {
                        "role": current_role,
                        "content": "\n\n".join(current_message_elements),
                    }
                )
            current_message_elements = []
            current_role = new_role

        if isinstance(message_config, str):
            messages += [
                # TODO should we append a standard system message? "You are a helpful assistant"?
                {
                    "role": "user",
                    "content": message_config,
                }
            ]
        else:
            for prompt_element in message_config:
                if isinstance(prompt_element, RoleElement):
                    deposit_messages(prompt_element.role)
                    continue
                elif isinstance(prompt_element, TextElement):
                    role = prompt_element.role
                    if role is not None:
                        deposit_messages(role)

                if isinstance(prompt_element, str):
                    as_str = prompt_element
                else:
                    as_str = prompt_element.as_string(quote_style)
                current_message_elements.append(as_str)
            if current_message_elements:
                messages.append(
                    {
                        "role": current_role,
                        "content": "\n\n".join(current_message_elements),
                    }
                )

        token_count = litellm.token_counter(  # pyright: ignore[reportPrivateImportUsage]
            model=model_config.model,
            messages=messages,
        )
        max_prompt_tokens = model_config.max_prompt_tokens
        if token_count > max_prompt_tokens:
            self.log.warning(
                "Trimming messages",
                token_count=token_count,
                max_prompt_tokens=max_prompt_tokens,
            )
            trimmed_messages: None | list[dict[str, str]] = litellm.utils.trim_messages(
                messages=messages,
                max_tokens=max_prompt_tokens,
                model=model_config.model,
                trim_ratio=1,
            )  # litellm is badly typed  # type: ignore
            if trimmed_messages is None:
                self.log.error(
                    "Failed to trim messages",
                    token_count=token_count,
                    max_prompt_tokens=max_prompt_tokens,
                )
                raise ValueError("Failed to trim messages")
            messages = trimmed_messages

        return messages

    async def _invoke_anthropic(
        self,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
    ) -> AsyncIterator[tuple[str, int | None]]:
        from anthropic import AsyncAnthropic
        from anthropic.types import MessageParam
        from anthropic import NOT_GIVEN

        system_messages = [
            message for message in messages if message["role"] == "system"
        ]
        system_prompt = "\n\n".join(message["content"] for message in system_messages)

        compatible_messages = [
            message for message in messages if message["role"] in ("user", "assistant")
        ]
        anthropic_messages = [
            MessageParam(
                role=message["role"],
                content=message["content"],
            )
            for message in compatible_messages
            if message["role"] in ("user", "assistant")  # for typing
        ]

        outstanding_messages = [
            message
            for message in messages
            if message not in system_messages and message not in compatible_messages
        ]
        if outstanding_messages:
            self.log.warning(
                "Some messages were not included in the prompt",
                messages=outstanding_messages,
            )

        if model_config.api_base is not None:
            self.log.warning("Ignoring api_base for Claude models")

        client = AsyncAnthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
        async with client.messages.stream(
            max_tokens=model_config.max_output_tokens,
            system=system_prompt,
            messages=anthropic_messages,
            model=model_config.model,
            temperature=model_config.temperature
            if model_config.temperature is not None
            else NOT_GIVEN,
            top_p=model_config.top_p if model_config.top_p is not None else NOT_GIVEN,
        ) as stream:
            async for completion in stream.text_stream:
                yield completion, None

    async def _invoke_ollama(
        self,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
    ) -> AsyncIterator[tuple[str, int | None]]:
        api_base = (
            model_config.api_base
            or get_secret("OLLAMA_API_BASE")
            or "http://localhost:11434"
        )

        api_url = f"{api_base}/api/chat"

        model_name = model_config.model.removeprefix("ollama/")

        headers = {}
        if model_config.auth_token is not None:
            headers["Authorization"] = f"Bearer {model_config.auth_token}"

        options = {}

        if model_config.temperature is not None:
            options["temperature"] = model_config.temperature
        if model_config.max_output_tokens is not None:
            options["num_predict"] = model_config.max_output_tokens

        # TODO support the other optional parameters
        # if model_config.top_p is not None:
        #     options["top_p"] = model_config.top_p
        # if model_config.frequency_penalty is not None:
        #     options["frequency_penalty"] = model_config.frequency_penalty

        data = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        def process_completion(completion):
            if not completion:
                return None
            try:
                data = json.loads(completion)
            except json.JSONDecodeError:
                return None
            if (
                not isinstance(data, dict)
                or "message" not in data
                or not isinstance(data["message"], dict)
                or "content" not in data["message"]
            ):
                return None
            delta = data["message"]["content"]
            if delta:
                return delta

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                json=data,
                headers=headers,
            ) as response:
                response.raise_for_status()

                # can't use `response.json` cus of unexpected mimetype: application/x-ndjson
                buffer = ""
                async for completion in response.content.iter_any():
                    buffer += completion.decode()
                    while "\n" in buffer:
                        json_, buffer = buffer.split("\n", 1)
                        completion = process_completion(json_)
                        if completion is not None:
                            yield completion, None
                if buffer:
                    completion = process_completion(buffer)
                    if completion is not None:
                        yield completion, None

    async def _invoke_litellm(
        self,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        schema: None | JsonSchemaObject,
    ) -> AsyncIterator[tuple[str, int | None]]:
        openai_api_key = get_secret("OPENAI_API_KEY")
        if openai_api_key is None and "gpt" in model_config.model:
            self.log.warning("OpenAI API key not set")

        headers = {}
        if model_config.auth_token is not None:
            headers["Authorization"] = f"Bearer {model_config.auth_token}"

        schema_kwargs = {}
        use_tool_calling = "gpt" not in model_config.model
        if schema is not None:
            json_schema = schema.model_dump(exclude_unset=True)
            # Use structured data generation for OpenAI,
            # tool calling otherwise (does not guarantee schema compliance)
            if not use_tool_calling:
                schema_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "function",
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            else:
                schema_kwargs["tool_choice"] = "required"
                schema_kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "function",
                            # "description": "Function to be called",
                            "parameters": json_schema,
                        },
                    }
                ]

        client = None
        tool_index = None
        try:
            if model_config.model.startswith("gpt"):
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=openai_api_key)

            completion: litellm.ModelResponse  # pyright: ignore[reportPrivateImportUsage]
            with prompt_env_context_singleton:
                async for completion in await litellm.acompletion(  # type: ignore
                    stream=True,
                    messages=messages,
                    client=client,
                    model=model_config.model,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_output_tokens,
                    top_p=model_config.top_p,
                    frequency_penalty=model_config.frequency_penalty,
                    presence_penalty=model_config.presence_penalty,
                    base_url=model_config.api_base,
                    extra_headers=headers,
                    **schema_kwargs,
                    # **model_config.model_dump(),
                ):
                    delta_obj = completion.choices[0].delta  # type: ignore
                    if schema is not None and use_tool_calling:
                        if delta_obj.tool_calls is not None:
                            tool_call = delta_obj.tool_calls[0]
                            tool_index = tool_call.index
                            delta_string = tool_call.function.arguments
                        else:
                            delta_string = None
                    else:
                        delta_string = delta_obj.content
                    if delta_string is None:
                        break
                    yield delta_string, tool_index
        finally:
            if client is not None:
                await client.close()

    async def invoke_llm(
        self,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        schema: None | JsonSchemaObject,
    ) -> AsyncIterator[tuple[str, int | None]]:
        # this function returns (delta, tool_index); when filling functions (currently only via litellm),
        # it returns one argument at a time (incrementing tool_index)
        if model_config.model.startswith("claude") and schema is None:
            iterator = self._invoke_anthropic(
                messages=messages,
                model_config=model_config,
            )
        elif model_config.model.startswith("ollama/") and schema is None:
            iterator = self._invoke_ollama(
                messages=messages,
                model_config=model_config,
            )
        else:
            iterator = self._invoke_litellm(
                messages=messages,
                model_config=model_config,
                schema=schema,
            )

        timer = Timer()
        first_completion_received = False
        async for completion in measure_async_iterator(
            self.log,
            iterator,
            timer,
        ):
            if not first_completion_received:
                self.log.info(
                    "First completion received",
                    seconds=timer.wall_time,
                )
                first_completion_received = True
            yield completion
        self.log.info("Invoked LLM", blocking_time=timer.blocking_time)

    def estimate_cost(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        completion: str,
    ) -> float:
        return litellm.completion_cost(  # pyright: ignore[reportPrivateImportUsage]
            model=model.model,
            messages=messages,
            completion=completion,
        )

    def parse_structured_response(
        self, tool_responses: dict[int, str], output: str, schema: dict
    ) -> dict | None:
        data = {}

        if tool_responses:
            try:
                for tool_response in tool_responses.values():
                    data |= json.loads(tool_response)
            except json.JSONDecodeError:
                self.log.exception(
                    "Failed to parse JSON response", tool_responses=tool_responses
                )
                return None
        else:
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                self.log.exception("Failed to parse JSON response", output=output)
                return None

        model = self.construct_model_from_schema(schema)
        if model is None:
            return None

        try:
            model.model_validate(data)
        except pydantic.ValidationError:
            self.log.exception("Generated data does not adhere to schema", data=data)
            return None

        return data

    async def iterate_invoke_llm(
        self,
        messages: list[dict[str, str]],
        model_config: ModelConfig,
        schema: None | JsonSchemaObject,
    ) -> AsyncIterator[tuple[str, dict[int, str], dict | None]]:
        output = ""
        tool_responses = defaultdict(str)
        partial_data = {}
        async for partial_output, tool_index in self.invoke_llm(
            messages=messages,
            model_config=model_config,
            schema=schema,
        ):
            output += partial_output
            if tool_index is not None:
                tool_responses[tool_index] += partial_output
                try:
                    loaded_output = json.loads(tool_responses[tool_index])
                    partial_data |= loaded_output
                except JSONDecodeError:
                    pass
            yield output, tool_responses, partial_data

    async def run(self, inputs: Inputs) -> AsyncIterator[Outputs]:
        if inputs.model is None:
            resolved_model = inputs._default_model
        else:
            override_attrs = inputs.model.model_dump(exclude_defaults=True)
            resolved_model = inputs._default_model.model_copy(update=override_attrs)

        messages = self.build_messages(
            inputs.prompt,
            resolved_model,
            inputs.quote_style,
        )

        # resolve schema
        if inputs.output_schema is None:
            schema = None
        else:
            schema = JsonSchemaObject(
                type="object",
                properties=inputs.output_schema,
            )

        output = ""
        tool_responses = defaultdict(str)
        async for attempt in tenacity.AsyncRetrying(
            wait=tenacity.wait_random_exponential(min=1, max=60),
            # let it timeout via `action_timeout`
            # stop=tenacity.stop_after_attempt(10),
            retry=tenacity.retry_if_exception_type(retry_errors),
            before_sleep=tenacity.before_sleep_log(
                self.log,  # pyright: ignore [reportArgumentType]
                logging.WARNING,
            ),
        ):
            with attempt:
                async for (
                    output,
                    tool_responses,
                    partial_data,
                ) in self.iterate_invoke_llm(
                    messages=messages,
                    model_config=resolved_model,
                    schema=schema,
                ):
                    yield Outputs(
                        result=output,
                        response=output,
                        data=partial_data or None,
                    )

        try:
            estimated_cost_usd = self.estimate_cost(
                model=resolved_model,
                messages=messages,
                completion=output,
            )
        except litellm.utils.NotFoundError:  # pyright: ignore[reportPrivateImportUsage]
            self.log.warning("Failed to estimate cost", model=resolved_model.model)
            estimated_cost_usd = None

        self.log.info(
            "Prompt completed",
            messages=messages,
            result=output,
            estimated_cost_usd=estimated_cost_usd,
            model=resolved_model.model_dump(),
        )

        # validate and yield final data
        if inputs.output_schema is not None:
            data = self.parse_structured_response(
                tool_responses, output, inputs.output_schema
            )
            if data is None:
                raise RuntimeError(
                    "Data returned by LLM does not adhere to output schema"
                )
            yield Outputs(
                result=output,
                response=output,
                data=data,
            )


# if __name__ == "__main__":
#     from aijson.tests.utils import run_action_manually
#
#     inputs = Inputs(
#         prompt=PromptConfig(
#             text="What should I make a fruit salad with?",
#             context=PromptContext([
#                 PromptContextEntry(
#                     heading="What I have in my kitchen",
#                     value="Apples, bananas, oranges, potatoes, and onions.",
#                 )
#             ]),
#         ),
#         instructions=PromptConfig(
#             text="No chattering, be as concise as possible."
#         ),
#     )
#     asyncio.run(run_action_manually(action=PromptString, inputs=inputs))
