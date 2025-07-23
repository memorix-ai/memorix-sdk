# 🧠 Memorix SDK

[![PyPI version](https://badge.fury.io/py/memorix-sdk.svg)](https://badge.fury.io/py/memorix-sdk)
[![Python versions](https://img.shields.io/pypi/pyversions/memorix-sdk.svg)](https://pypi.org/project/memorix-sdk/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/memorix-ai/memorix-sdk/workflows/CI/badge.svg)](https://github.com/memorix-ai/memorix-sdk/actions)
[![Codecov](https://codecov.io/gh/memorix-ai/memorix-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/memorix-ai/memorix-sdk)

> **A flexible memory management system for AI applications** with plug-in support for various vector stores and embedding models.

## ✨ Features

- 🗄️ **Flexible Vector Stores**: FAISS, Qdrant, and custom implementations
- 🤖 **Multiple Embedding Models**: OpenAI, Google Gemini, Sentence Transformers
- 📊 **Metadata Management**: Optional metadata storage with multiple backends
- ⚙️ **YAML Configuration**: Easy configuration management
- 🚀 **Simple API**: Clean and intuitive interface
- 🧪 **Comprehensive Testing**: Full test coverage and CI/CD
- 📚 **Rich Documentation**: API docs, examples, and guides

## 🚀 Quick Start

### Installation

```bash
pip install memorix-sdk
```

### Basic Usage

```python
from memorix import MemoryAPI, Config

# Initialize with configuration
config = Config('memorix.yaml')
memory = MemoryAPI(config)

# Store a memory
memory_id = memory.store(
    "Python is a high-level programming language.",
    metadata={"topic": "programming", "language": "python"}
)

# Retrieve relevant memories
results = memory.retrieve("programming languages", top_k=5)
for result in results:
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Metadata: {result['metadata']}")
```

## ⚙️ Configuration

Create a `memorix.yaml` file:

```yaml
vector_store:
  type: faiss
  index_path: ./memorix_index
  dimension: 1536

embedder:
  type: openai
  model: text-embedding-ada-002
  api_key: ${OPENAI_API_KEY}

metadata_store:
  type: sqlite
  database_path: ./memorix_metadata.db

settings:
  max_memories: 10000
  similarity_threshold: 0.7
```

## 🔌 Supported Components

### Vector Stores
- **FAISS**: Fast similarity search with CPU/GPU support
- **Qdrant**: Vector database with advanced features
- **Custom**: Implement your own vector store

### Embedding Models
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small, etc.
- **Google Gemini**: models/embedding-001
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.

### Metadata Stores
- **SQLite**: Persistent storage with SQL database
- **In-Memory**: Fast temporary storage
- **JSON File**: Simple file-based storage

## 📖 API Reference

### MemoryAPI

| Method | Description |
|--------|-------------|
| `store(content, metadata=None)` | Store content with optional metadata |
| `retrieve(query, top_k=5)` | Retrieve relevant memories based on query |
| `update(memory_id, content, metadata=None)` | Update an existing memory |
| `delete(memory_id)` | Delete a memory by ID |
| `list_memories(limit=100)` | List all memories with basic info |

## 🎯 Examples

### Core Examples

See the `examples/` directory for basic usage examples:

- `basic_usage.py`: Minimal usage example

### 🚀 Comprehensive Examples & Demos

For comprehensive examples, interactive demos, and real-world applications, check out our dedicated examples repository:

**[📚 Memorix Examples Repository](https://github.com/memorix-ai/memorix-examples)**

#### Featured Demos:
- **[Basic Usage Demo](https://github.com/memorix-ai/memorix-examples/tree/main/demos/01_basic_usage.py)** - Get started in 5 minutes
- **[Chatbot with Memory](https://github.com/memorix-ai/memorix-examples/tree/main/demos/02_chatbot_memory.py)** - Build conversational AI with persistent memory
- **[Vector Store Comparison](https://github.com/memorix-ai/memorix-examples/tree/main/demos/04_vector_store_comparison.py)** - Compare FAISS vs Qdrant performance

#### Interactive Applications:
- **[Streamlit Web App](https://github.com/memorix-ai/memorix-examples/tree/main/streamlit_app/)** - Interactive memory management interface
- **[Gradio Demo](https://github.com/memorix-ai/memorix-examples/tree/main/gradio_app/)** - Quick prototyping interface

#### Ready-to-Use Templates:
- **[Chatbot Template](https://github.com/memorix-ai/memorix-examples/tree/main/templates/chatbot/)** - Production-ready chatbot with memory

#### Quick Start:
```bash
# Clone the examples repository
git clone https://github.com/memorix-ai/memorix-examples.git
cd memorix-examples

# Install dependencies
pip install -r requirements.txt

# Run a demo
python demos/01_basic_usage.py
```

## 🛠️ Development

### Setup

```bash
# Clone the repository
git clone https://github.com/memorix-ai/memorix-sdk.git
cd memorix-sdk

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Common Commands

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Run linting
make lint

# Build package
make build

# See all available commands
make help
```

### Running Tests

```bash
python -m pytest tests/
```

### Running Examples

```bash
python examples/basic_usage.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [ ] Add more vector store backends (Pinecone, Weaviate, etc.)
- [ ] Add more embedding models (Cohere, Hugging Face, etc.)
- [ ] Add memory compression and summarization
- [ ] Add batch operations
- [ ] Add memory versioning
- [ ] Add memory expiration
- [ ] Add memory categories and tags
- [ ] Add memory search filters
- [ ] Add memory export/import
- [ ] Add memory analytics and insights

## 📞 Support

- 📚 **Documentation**: [docs.memorix.ai](https://docs.memorix.ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/memorix-ai/memorix-sdk/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/memorix-ai/memorix-sdk/discussions)
- 📧 **Email**: support@memorix.ai

---

<div align="center">

**Made with ❤️ by the Memorix Team**

[![GitHub stars](https://img.shields.io/github/stars/memorix-ai/memorix-sdk?style=social)](https://github.com/memorix-ai/memorix-sdk)
[![GitHub forks](https://img.shields.io/github/forks/memorix-ai/memorix-sdk?style=social)](https://github.com/memorix-ai/memorix-sdk)

</div> 