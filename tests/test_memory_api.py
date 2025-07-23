"""
Tests for Memorix SDK Memory API.
"""

import unittest
from memorix import MemoryAPI, Config


class TestMemoryAPI(unittest.TestCase):
    """Test cases for MemoryAPI class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        # Use in-memory storage for tests
        self.config.set("vector_store.type", "faiss")
        self.config.set("metadata_store.type", "memory")
        self.config.set("embedder.type", "sentence_transformers")
        self.config.set("embedder.model", "all-MiniLM-L6-v2")

        self.memory = MemoryAPI(self.config)

    def test_store_and_retrieve(self):
        """Test storing and retrieving memories."""
        # Store a memory
        content = "Python is a programming language"
        metadata = {"topic": "programming", "language": "python"}

        memory_id = self.memory.store(content, metadata)

        # Verify memory was stored
        self.assertIsNotNone(memory_id)

        # Retrieve the memory
        results = self.memory.retrieve("programming language", top_k=1)

        # Verify retrieval
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], content)
        self.assertEqual(results[0]["metadata"]["topic"], "programming")

    def test_retrieve_multiple(self):
        """Test retrieving multiple memories."""
        # Store multiple memories
        self.memory.store("Python is a programming language", {"topic": "programming"})
        self.memory.store(
            "Java is another programming language", {"topic": "programming"}
        )
        self.memory.store("Machine learning is a subset of AI", {"topic": "AI"})

        # Retrieve memories
        results = self.memory.retrieve("programming", top_k=3)

        # Should return some results
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        # All results should have metadata
        for result in results:
            self.assertIn("metadata", result)
            self.assertIn("topic", result["metadata"])

    def test_update_memory(self):
        """Test updating a memory."""
        # Store initial memory
        memory_id = self.memory.store("Initial content", {"version": 1})

        # Update the memory
        new_content = "Updated content"
        new_metadata = {"version": 2, "updated": True}

        success = self.memory.update(memory_id, new_content, new_metadata)

        # Verify update was successful
        self.assertTrue(success)

        # Retrieve and verify updated content
        results = self.memory.retrieve("content", top_k=1)
        self.assertEqual(results[0]["content"], new_content)
        self.assertEqual(results[0]["metadata"]["version"], 2)

    def test_delete_memory(self):
        """Test deleting a memory."""
        # Store a memory
        memory_id = self.memory.store("Content to delete", {"test": True})

        # Verify it exists
        results = self.memory.retrieve("delete", top_k=1)
        self.assertEqual(len(results), 1)

        # Delete the memory
        success = self.memory.delete(memory_id)

        # Verify deletion was successful
        self.assertTrue(success)

        # Verify it no longer exists
        results = self.memory.retrieve("delete", top_k=1)
        self.assertEqual(len(results), 0)

    def test_list_memories(self):
        """Test listing memories."""
        # Store multiple memories
        self.memory.store("Memory 1", {"id": 1})
        self.memory.store("Memory 2", {"id": 2})
        self.memory.store("Memory 3", {"id": 3})

        # List memories with limit
        memories = self.memory.list_memories(limit=2)

        # Should return 2 memories
        self.assertEqual(len(memories), 2)

        # Each should have content
        for memory in memories:
            self.assertIn("content", memory)
            self.assertIn("memory_id", memory)

    def test_metadata_handling(self):
        """Test metadata handling."""
        # Store memory without metadata
        memory_id1 = self.memory.store("Content without metadata")

        # Store memory with metadata
        metadata = {"topic": "test", "priority": "high"}
        memory_id2 = self.memory.store("Content with metadata", metadata)

        # Retrieve both
        results = self.memory.retrieve("content", top_k=2)

        # Find the memories
        memory1 = None
        memory2 = None

        for result in results:
            if result["memory_id"] == memory_id1:
                memory1 = result
            elif result["memory_id"] == memory_id2:
                memory2 = result

        # Verify metadata handling
        self.assertIsNotNone(memory1)
        self.assertIsNotNone(memory2)

        # Memory without explicit metadata should have auto-generated metadata
        self.assertIn("timestamp", memory1["metadata"])
        self.assertIn("content_length", memory1["metadata"])

        # Memory with explicit metadata should have both
        self.assertEqual(memory2["metadata"]["topic"], "test")
        self.assertEqual(memory2["metadata"]["priority"], "high")
        self.assertIn("timestamp", memory2["metadata"])


if __name__ == "__main__":
    unittest.main()
