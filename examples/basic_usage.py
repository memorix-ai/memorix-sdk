"""
Basic usage example for Memorix AI.
"""

from memorix import MemoryAPI, Config


def main():
    """Demonstrate basic usage of Memorix AI."""
    
    # Initialize configuration
    config = Config()
    
    # Set API key for OpenAI (you would set this in your config file)
    config.set('embedder.api_key', 'your-openai-api-key-here')
    
    # Initialize memory API
    memory = MemoryAPI(config)
    
    # Store some memories
    print("Storing memories...")
    
    memory_id1 = memory.store(
        "Python is a high-level programming language known for its simplicity and readability.",
        metadata={"topic": "programming", "language": "python"}
    )
    
    memory_id2 = memory.store(
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        metadata={"topic": "AI", "category": "machine learning"}
    )
    
    memory_id3 = memory.store(
        "Data structures are ways of organizing and storing data for efficient access and modification.",
        metadata={"topic": "computer science", "category": "data structures"}
    )
    
    print(f"Stored memories with IDs: {memory_id1}, {memory_id2}, {memory_id3}")
    
    # Retrieve relevant memories
    print("\nSearching for programming-related content...")
    results = memory.retrieve("programming languages", top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Memory ID: {result['memory_id']}")
        print(f"  Content: {result['content']}")
        print(f"  Similarity: {result['similarity']:.3f}")
        print(f"  Metadata: {result['metadata']}")
    
    # Search for AI-related content
    print("\nSearching for AI-related content...")
    results = memory.retrieve("artificial intelligence", top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Memory ID: {result['memory_id']}")
        print(f"  Content: {result['content']}")
        print(f"  Similarity: {result['similarity']:.3f}")
        print(f"  Metadata: {result['metadata']}")
    
    # List all memories
    print("\nListing all memories...")
    memories = memory.list_memories(limit=10)
    
    for i, mem in enumerate(memories, 1):
        print(f"  {i}. {mem['content']}")
    
    # Update a memory
    print(f"\nUpdating memory {memory_id1}...")
    success = memory.update(
        memory_id1,
        "Python is a high-level, interpreted programming language known for its simplicity, readability, and extensive standard library.",
        metadata={"topic": "programming", "language": "python", "updated": True}
    )
    
    if success:
        print("Memory updated successfully!")
        
        # Retrieve the updated memory
        results = memory.retrieve("python programming", top_k=1)
        if results:
            print(f"Updated content: {results[0]['content']}")
    
    # Delete a memory
    print(f"\nDeleting memory {memory_id3}...")
    success = memory.delete(memory_id3)
    
    if success:
        print("Memory deleted successfully!")
        
        # Verify deletion
        memories = memory.list_memories(limit=10)
        print(f"Remaining memories: {len(memories)}")


if __name__ == "__main__":
    main() 