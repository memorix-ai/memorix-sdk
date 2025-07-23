# 🚀 Usage Guide

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
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Metadata: {result['metadata']}")
```

## Basic Operations

### 📝 Storing Memories

```python
# Store with metadata
memory_id = memory.store(
    "Machine learning is a subset of AI.",
    metadata={
        "topic": "AI",
        "category": "machine learning",
        "source": "textbook",
        "importance": "high"
    }
)

# Store without metadata
memory_id = memory.store("Simple text content")

# Store multiple memories
texts = [
    "Python is great for data science.",
    "JavaScript is popular for web development.",
    "Rust provides memory safety."
]

for text in texts:
    memory.store(text, metadata={"category": "programming"})
```

### 🔍 Retrieving Memories

```python
# Basic retrieval
results = memory.retrieve("data science", top_k=3)

# Retrieval with similarity threshold
results = memory.retrieve("web development", top_k=5)
filtered_results = [r for r in results if r['similarity'] > 0.8]

# Retrieval with metadata filtering
results = memory.retrieve("programming", top_k=10)
python_results = [r for r in results if r['metadata'].get('language') == 'python']

# Print results
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Content: {result['content']}")
    print(f"  Similarity: {result['similarity']:.3f}")
    print(f"  Memory ID: {result['memory_id']}")
    print(f"  Metadata: {result['metadata']}")
```

### ✏️ Updating Memories

```python
# Update content and metadata
success = memory.update(
    memory_id,
    "Python is excellent for data science and machine learning.",
    metadata={
        "topic": "programming",
        "language": "python",
        "category": "data science",
        "updated": True
    }
)

# Update only content
success = memory.update(memory_id, "Updated content")

# Update only metadata
success = memory.update(memory_id, metadata={"priority": "high"})
```

### 🗑️ Deleting Memories

```python
# Delete by memory ID
success = memory.delete(memory_id)

# Delete multiple memories
memory_ids = ["id1", "id2", "id3"]
for memory_id in memory_ids:
    memory.delete(memory_id)
```

### 📋 Listing Memories

```python
# List all memories
memories = memory.list_memories(limit=100)

# List with custom limit
recent_memories = memory.list_memories(limit=10)

# Process memories
for memory in memories:
    print(f"ID: {memory['memory_id']}")
    print(f"Content: {memory['content'][:100]}...")
    print("---")
```

## Advanced Usage

### 🔧 Custom Configuration

```python
from memorix import Config

# Create configuration programmatically
config = Config()
config.set('vector_store.type', 'faiss')
config.set('vector_store.index_path', './my_index')
config.set('embedder.type', 'openai')
config.set('embedder.api_key', 'your-api-key')
config.set('metadata_store.type', 'sqlite')
config.set('metadata_store.database_path', './my_metadata.db')

# Initialize with custom config
memory = MemoryAPI(config)
```

### 🗄️ Different Vector Stores

```python
# FAISS (default)
config.set('vector_store.type', 'faiss')
config.set('vector_store.index_path', './faiss_index')

# Qdrant
config.set('vector_store.type', 'qdrant')
config.set('vector_store.url', 'http://localhost:6333')
config.set('vector_store.collection_name', 'memories')
```

### 🤖 Different Embedding Models

```python
# OpenAI
config.set('embedder.type', 'openai')
config.set('embedder.model', 'text-embedding-ada-002')
config.set('embedder.api_key', 'your-openai-key')

# Google Gemini
config.set('embedder.type', 'gemini')
config.set('embedder.model', 'models/embedding-001')
config.set('embedder.api_key', 'your-gemini-key')

# Sentence Transformers (local)
config.set('embedder.type', 'sentence_transformers')
config.set('embedder.model', 'all-MiniLM-L6-v2')
```

### 📊 Different Metadata Stores

```python
# SQLite (persistent)
config.set('metadata_store.type', 'sqlite')
config.set('metadata_store.database_path', './metadata.db')

# In-Memory (temporary)
config.set('metadata_store.type', 'memory')

# JSON File
config.set('metadata_store.type', 'json')
config.set('metadata_store.file_path', './metadata.json')
```

## Real-World Examples

### 🤖 Chatbot with Memory

```python
class Chatbot:
    def __init__(self):
        self.memory = MemoryAPI(Config())
        self.conversation_history = []
    
    def chat(self, user_input):
        # Store user input
        memory_id = self.memory.store(
            user_input,
            metadata={
                "type": "user_input",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Retrieve relevant context
        context = self.memory.retrieve(user_input, top_k=3)
        
        # Generate response (simplified)
        response = self.generate_response(user_input, context)
        
        # Store response
        self.memory.store(
            response,
            metadata={
                "type": "bot_response",
                "timestamp": datetime.now().isoformat(),
                "related_input": memory_id
            }
        )
        
        return response
    
    def generate_response(self, user_input, context):
        # Simplified response generation
        context_text = "\n".join([r['content'] for r in context])
        return f"Based on context: {context_text[:100]}... Response to: {user_input}"
```

### 📚 Document Search System

```python
class DocumentSearch:
    def __init__(self):
        self.memory = MemoryAPI(Config())
    
    def add_document(self, doc_id, content, metadata=None):
        """Add a document to the search index."""
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "doc_id": doc_id,
            "type": "document",
            "added_at": datetime.now().isoformat()
        })
        
        memory_id = self.memory.store(content, metadata)
        return memory_id
    
    def search_documents(self, query, top_k=10):
        """Search documents by query."""
        results = self.memory.retrieve(query, top_k=top_k)
        
        # Filter for documents only
        documents = [r for r in results if r['metadata'].get('type') == 'document']
        
        return documents
    
    def get_document(self, doc_id):
        """Get a specific document by ID."""
        # This would require additional indexing for efficient lookup
        memories = self.memory.list_memories(limit=1000)
        for memory in memories:
            if memory['metadata'].get('doc_id') == doc_id:
                return memory
        return None
```

### 🎯 Recommendation System

```python
class RecommendationSystem:
    def __init__(self):
        self.memory = MemoryAPI(Config())
    
    def add_user_preference(self, user_id, item_id, rating, category):
        """Store user preference."""
        content = f"User {user_id} rated {item_id} with {rating} stars"
        metadata = {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "category": category,
            "type": "preference"
        }
        
        self.memory.store(content, metadata)
    
    def get_recommendations(self, user_id, category=None, top_k=5):
        """Get recommendations for a user."""
        query = f"User {user_id} preferences"
        if category:
            query += f" {category}"
        
        results = self.memory.retrieve(query, top_k=top_k)
        
        # Filter and process recommendations
        preferences = [r for r in results if r['metadata'].get('type') == 'preference']
        
        return preferences
```

## Performance Optimization

### 📦 Batch Operations

```python
# Batch store
texts = ["Text 1", "Text 2", "Text 3", ...]
metadata_list = [{"id": i} for i in range(len(texts))]

memory_ids = []
for text, metadata in zip(texts, metadata_list):
    memory_id = memory.store(text, metadata)
    memory_ids.append(memory_id)

# Batch retrieve
queries = ["query 1", "query 2", "query 3"]
all_results = []
for query in queries:
    results = memory.retrieve(query, top_k=5)
    all_results.extend(results)
```

### 🔄 Connection Management

```python
# For production applications, consider connection pooling
import contextlib

@contextlib.contextmanager
def memory_session():
    """Context manager for memory operations."""
    memory = MemoryAPI(Config())
    try:
        yield memory
    finally:
        # Clean up connections if needed
        pass

# Usage
with memory_session() as memory:
    memory.store("Important data")
    results = memory.retrieve("query")
```

## Error Handling

```python
import logging
from memorix import MemoryAPI, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_memory_operation(func):
    """Decorator for safe memory operations."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Memory operation failed: {e}")
            return None
    return wrapper

@safe_memory_operation
def store_with_retry(memory, content, metadata=None, max_retries=3):
    """Store content with retry logic."""
    for attempt in range(max_retries):
        try:
            return memory.store(content, metadata)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Store attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Best Practices

### 📝 Memory Management

```python
# Use meaningful metadata
metadata = {
    "source": "user_input",
    "timestamp": datetime.now().isoformat(),
    "user_id": "user123",
    "session_id": "session456",
    "importance": "high",
    "category": "feedback"
}

# Regular cleanup
def cleanup_old_memories(memory, days_old=30):
    """Remove memories older than specified days."""
    cutoff_date = datetime.now() - timedelta(days=days_old)
    memories = memory.list_memories(limit=1000)
    
    for memory_item in memories:
        timestamp = memory_item['metadata'].get('timestamp')
        if timestamp:
            memory_date = datetime.fromisoformat(timestamp)
            if memory_date < cutoff_date:
                memory.delete(memory_item['memory_id'])
```

### 🔍 Query Optimization

```python
# Use specific queries
good_query = "Python machine learning libraries"
bad_query = "stuff about programming"

# Combine multiple queries for better results
queries = ["Python", "machine learning", "libraries"]
all_results = []
for query in queries:
    results = memory.retrieve(query, top_k=3)
    all_results.extend(results)

# Remove duplicates
unique_results = {r['memory_id']: r for r in all_results}.values()
```

### 📊 Monitoring and Logging

```python
import time
from functools import wraps

def monitor_memory_operations(func):
    """Decorator to monitor memory operation performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

# Apply to memory operations
memory.store = monitor_memory_operations(memory.store)
memory.retrieve = monitor_memory_operations(memory.retrieve)
```

## Next Steps

1. **Explore [Examples](../examples/) for more use cases**
2. **Check [Configuration Options](../memorix.yaml) for advanced setup**
3. **Read [Architecture Documentation](ARCHITECTURE.md) for deep dive**
4. **Join [Community Discussions](https://github.com/memorix-ai/memorix-sdk/discussions)** 