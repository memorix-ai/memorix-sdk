# Contributing to Memorix SDK

Thank you for your interest in contributing to Memorix SDK! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 🐛 Reporting Bugs

- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Provide clear steps to reproduce the issue
- Include your environment details (OS, Python version, etc.)
- Add error messages and stack traces if applicable

### ✨ Suggesting Features

- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the problem you're trying to solve
- Provide use cases and examples
- Consider implementation complexity

### 📚 Improving Documentation

- Use the [Documentation Request template](.github/ISSUE_TEMPLATE/documentation_request.md)
- Suggest specific improvements
- Provide examples of what you'd like to see

### 💻 Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/memorix-ai/memorix-sdk.git
cd memorix-sdk

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=memorix --cov-report=html

# Run specific test file
pytest tests/test_memory_api.py

# Run with verbose output
pytest -v
```

### Code Quality Tools

```bash
# Format code with black
black memorix/ tests/ examples/

# Sort imports with isort
isort memorix/ tests/ examples/

# Lint with flake8
flake8 memorix/ tests/ examples/

# Type checking with mypy
mypy memorix/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically format and lint your code:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files
pre-commit run --all-files
```

## 📋 Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Documentation Style

- Use Google-style docstrings
- Include examples in docstrings
- Keep documentation up to date with code changes
- Use clear and concise language

### Test Style

- Write tests for all new functionality
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Test both success and failure cases

## 🏗️ Project Structure

```
memorix-sdk/
├── memorix/                 # Core package
│   ├── __init__.py         # Package exports
│   ├── memory_api.py       # Main API
│   ├── config.py           # Configuration
│   ├── vector_store.py     # Vector store interfaces
│   ├── embedder.py         # Embedding interfaces
│   └── metadata_store.py   # Metadata storage
├── examples/               # Usage examples
├── tests/                  # Test suite
├── docs/                   # Documentation
└── .github/               # GitHub templates and workflows
```

## 🔧 Adding New Features

### Vector Store Backends

1. Create a new class implementing `VectorStoreInterface`
2. Add the new backend to the factory in `VectorStore._create_store()`
3. Add configuration options to `Config._get_default_config()`
4. Write comprehensive tests
5. Update documentation

### Embedding Models

1. Create a new class implementing `EmbedderInterface`
2. Add the new model to the factory in `Embedder._create_embedder()`
3. Add configuration options to `Config._get_default_config()`
4. Write comprehensive tests
5. Update documentation

### Metadata Stores

1. Create a new class implementing `MetadataStoreInterface`
2. Add the new store to the factory in `MetadataStore._create_store()`
3. Add configuration options to `Config._get_default_config()`
4. Write comprehensive tests
5. Update documentation

## 🧪 Testing Guidelines

### Test Structure

- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance tests for critical paths

### Test Naming

- Use descriptive test names
- Follow the pattern: `test_<method>_<scenario>_<expected_result>`
- Group related tests in classes

### Test Data

- Use fixtures for common test data
- Create realistic test scenarios
- Clean up test data after tests

### Mocking

- Mock external API calls
- Mock file system operations
- Mock time-dependent operations

## 📦 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Changelog is updated
- [ ] Version is bumped
- [ ] Release notes are written
- [ ] Package is built and tested
- [ ] Release is published to PyPI

## 🐛 Debugging

### Common Issues

1. **Import Errors**: Ensure you're in the correct virtual environment
2. **Test Failures**: Check that all dependencies are installed
3. **Linting Errors**: Run `black` and `isort` to format code
4. **Type Errors**: Run `mypy` to check type annotations

### Getting Help

- Check existing issues and pull requests
- Search the documentation
- Ask questions in GitHub Discussions
- Contact the maintainers

## 📄 License

By contributing to Memorix SDK, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Acknowledgments

Thank you to all contributors who have helped make Memorix SDK better!

---

For more information, see our [README](README.md) and [Documentation](https://docs.memorix.ai). 