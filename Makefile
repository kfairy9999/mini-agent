.PHONY: test eval demo install

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

eval:
	mini-agent-eval

demo:
	mini-agent --mock "What is (17 * 23) + 5?"
