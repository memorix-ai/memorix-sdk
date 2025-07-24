# 🚀 Memorix SDK Quick Reference

## Installation

```bash
pip install memorix-ai
```

## Basic Usage

```python
from memorix import MemoryAPI, Config

# Setup
config = Config('memorix.yaml')
memory = MemoryAPI(config)

# Core Operations
memory_id = memory.store("content", metadata={"key": "value"})
results = memory.retrieve("query", top_k=5)
success = memory.update(memory_id, "new content", metadata={"updated": True})
success = memory.delete(memory_id)
memories = memory.list_memories(limit=100)
```

## Configuration (memorix.yaml)

```yaml
vector_store:
  type: faiss                    # faiss, qdrant
  index_path: ./memorix_index
  dimension: 1536

embedder:
  type: openai                   # openai, gemini, sentence_transformers
  model: text-embedding-ada-002
  api_key: ${OPENAI_API_KEY}

metadata_store:
  type: sqlite                   # sqlite, memory, json
  database_path: ./metadata.db

settings:
  max_memories: 10000
  similarity_threshold: 0.7
```

## API Reference

| Method | Description | Returns |
|--------|-------------|---------|
| `store(content, metadata=None)` | Store memory | `memory_id` |
| `retrieve(query, top_k=5)` | Search memories | `List[Dict]` |
| `update(memory_id, content, metadata=None)` | Update memory | `bool` |
| `delete(memory_id)` | Delete memory | `bool` |
| `list_memories(limit=100)` | List memories | `List[Dict]` |

## Result Format

```python
{
    'memory_id': 'uuid-string',
    'content': 'original text content',
    'similarity': 0.85,  # float 0-1
    'metadata': {
        'timestamp': '2024-01-01T12:00:00',
        'content_length': 42,
        'custom_key': 'custom_value'
    }
}
```

## Common Patterns

### Chatbot with Memory
```python
def chat(user_input):
    # Store user input
    memory_id = memory.store(user_input, {"type": "user_input"})
    
    # Get context
    context = memory.retrieve(user_input, top_k=3)
    
    # Generate response
    response = generate_response(user_input, context)
    
    # Store response
    memory.store(response, {"type": "bot_response"})
    
    return response
```

### Document Search
```python
def add_document(doc_id, content):
    memory.store(content, {
        "doc_id": doc_id,
        "type": "document",
        "added_at": datetime.now().isoformat()
    })

def search_documents(query):
    results = memory.retrieve(query, top_k=10)
    return [r for r in results if r['metadata'].get('type') == 'document']
```

### Recommendation System
```python
def add_preference(user_id, item_id, rating):
    content = f"User {user_id} rated {item_id} with {rating} stars"
    memory.store(content, {
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "type": "preference"
    })

def get_recommendations(user_id):
    return memory.retrieve(f"User {user_id} preferences", top_k=5)
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-gemini-key"
```

## Error Handling

```python
try:
    memory_id = memory.store("content")
except Exception as e:
    print(f"Store failed: {e}")
    # Handle error appropriately
```

## Performance Tips

- Use batch operations for multiple items
- Set appropriate `top_k` values
- Use meaningful metadata for filtering
- Consider similarity thresholds
- Monitor operation timing

## Support

- 📚 [Documentation](https://docs.memorix.ai)
- 🐛 [Issues](https://github.com/memorix-ai/memorix-ai/issues)
- 💬 [Discussions](https://github.com/memorix-ai/memorix-ai/discussions)
- �� support@memorix.ai 