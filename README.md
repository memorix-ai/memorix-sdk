# Memorix SDK

A flexible memory management system for AI applications with plug-in support for various vector stores and embedding models.

## Features

- **Flexible Vector Stores**: Support for FAISS, Qdrant, and more
- **Multiple Embedding Models**: OpenAI, Google Gemini, Sentence Transformers
- **Metadata Management**: Optional metadata storage with multiple backends
- **YAML Configuration**: Easy configuration management
- **Simple API**: Clean and intuitive interface

## Installation

```bash
pip install memorix-sdk
```

## Quick Start

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
    print(f"Similarity: {result['similarity']}")
    print(f"Metadata: {result['metadata']}")
```

## Configuration

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

## Supported Components

### Vector Stores
- **FAISS**: Fast similarity search
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

## API Reference

### MemoryAPI

#### `store(content, metadata=None)`
Store content with optional metadata.

#### `retrieve(query, top_k=5)`
Retrieve relevant memories based on query.

#### `update(memory_id, content, metadata=None)`
Update an existing memory.

#### `delete(memory_id)`
Delete a memory by ID.

#### `list_memories(limit=100)`
List all memories with basic info.

## Examples

See the `examples/` directory for usage examples:

- `basic_usage.py`: Minimal usage example
- More examples coming soon...

## Development

### Setup

```bash
git clone https://github.com/your-org/memorix-sdk.git
cd memorix-sdk
pip install -e .
```

### Running Tests

```bash
python -m pytest tests/
```

### Running Examples

```bash
python examples/basic_usage.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

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

## Support

- Documentation: [docs.memorix.ai](https://docs.memorix.ai)
- Issues: [GitHub Issues](https://github.com/your-org/memorix-sdk/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/memorix-sdk/discussions) 