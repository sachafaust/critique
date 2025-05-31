.PHONY: install install-dev test lint format clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=llm_critique --cov-report=html

lint:
	flake8 llm_critique tests
	mypy llm_critique tests
	black --check llm_critique tests
	isort --check-only llm_critique tests

format:
	black llm_critique tests
	isort llm_critique tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 