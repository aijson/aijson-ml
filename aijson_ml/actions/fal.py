from aijson import register_action


@register_action(
    cache=False,
)
async def fal(
    prompt: str,
    model: str = "fal-ai/flux-pro",
    safety_tolerance: int = 5,
) -> str:
    import fal_client

    handler = await fal_client.submit_async(
        model,
        arguments={"prompt": prompt, "safety_tolerance": str(safety_tolerance)},
    )

    result = await handler.get()
    return result["images"][0]["url"]
