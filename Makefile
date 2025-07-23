.PHONY: help install install-dev test test-cov lint format clean build publish docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install the package in development mode"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build the package"
	@echo "  publish      - Publish to PyPI (requires credentials)"
	@echo "  docs         - Build documentation"
	@echo "  pre-commit   - Install pre-commit hooks"

# Install the package in development mode
install:
	pip install -e .

# Install development dependencies
install-dev:
	pip install -e ".[dev]"
	pip install pre-commit

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=memorix --cov-report=html --cov-report=term-missing

# Run linting checks
lint:
	flake8 memorix/ tests/ examples/
	mypy memorix/
	black --check --diff memorix/ tests/ examples/
	isort --check-only --diff memorix/ tests/ examples/

# Format code
format:
	black memorix/ tests/ examples/
	isort memorix/ tests/ examples/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build the package
build:
	python -m build

# Publish to PyPI (requires credentials)
publish:
	twine upload dist/*

# Build documentation
docs:
	mkdocs build

# Install pre-commit hooks
pre-commit:
	pre-commit install

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files

# Security checks
security:
	bandit -r memorix/
	safety check

# Type checking
type-check:
	mypy memorix/

# Run all quality checks
quality: lint type-check security

# Development setup
setup: install-dev pre-commit
	@echo "Development environment setup complete!"

# Quick development cycle
dev: format lint test
	@echo "Development cycle complete!"

# Release preparation
release: clean build test-cov lint security
	@echo "Release preparation complete!" 