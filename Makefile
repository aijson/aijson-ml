type:
	pyright aijson_ml

test:
	pytest aijson_ml

test-no-skip:
	pytest --disallow-skip

test-fast:
	pytest -m "not slow" aijson_ml

lint:
	ruff check --fix

format:
	ruff format

all: format lint type test-fast
