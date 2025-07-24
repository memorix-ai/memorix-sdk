# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Memorix SDK
- Core memory management API
- YAML-based configuration system
- Vector store plug-in architecture (FAISS, Qdrant)
- Embedding model support (OpenAI, Gemini, Sentence Transformers)
- Metadata storage backends (SQLite, In-Memory, JSON)
- Comprehensive test suite
- Documentation and examples

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Memorix SDK
- Core memory management API with store, retrieve, update, and delete operations
- YAML-based configuration system with environment variable support
- Vector store plug-in architecture with FAISS and Qdrant implementations
- Embedding model support for OpenAI, Google Gemini, and Sentence Transformers
- Metadata storage backends including SQLite, In-Memory, and JSON file storage
- Comprehensive test suite with unit and integration tests
- Documentation with API reference and usage examples
- Development tools configuration (black, isort, mypy, pytest)
- CI/CD workflows for testing, linting, and deployment
- Issue and pull request templates
- Contributing guidelines

### Features
- **MemoryAPI**: Main interface for memory operations
- **Config**: Flexible configuration management
- **VectorStore**: Pluggable vector store architecture
- **Embedder**: Multiple embedding model support
- **MetadataStore**: Optional metadata handling
- **Examples**: Basic usage examples
- **Tests**: Comprehensive test coverage

### Documentation
- README with installation and usage instructions
- API documentation with examples
- Configuration guide
- Contributing guidelines
- Issue and PR templates

---

## Version History

- **0.1.0**: Initial release with core functionality

## Release Process

1. Update version in `pyproject.toml`
2. Update this changelog
3. Create a release tag
4. Publish to PyPI
5. Create GitHub release

## Contributing to Changelog

When adding entries to the changelog, please follow these guidelines:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues and pull requests liberally
- Consider starting with a verb (Add, Change, Deprecate, Remove, Fix, Security)
- Group changes by type and scope
- Include breaking changes prominently 