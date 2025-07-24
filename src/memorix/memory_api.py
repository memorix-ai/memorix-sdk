"""
Main memory interface for the Memorix SDK.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import Config
from .embedder import Embedder
from .metadata_store import MetadataStore
from .vector_store import VectorStore


class MemoryAPI:
    """
    Main interface for memory operations.
    """

    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(config)
        self.embedder = Embedder(config)
        self.metadata_store = MetadataStore(config)

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store content in memory with optional metadata.

        Args:
            content: The content to store
            metadata: Optional metadata dictionary

        Returns:
            Memory ID of the stored content
        """  # noqa: E501
        memory_id = str(uuid.uuid4())

        # Generate embedding
        embedding = self.embedder.embed(content)

        # Store in vector store
        self.vector_store.store(memory_id, embedding, content)

        # Always store metadata (with auto-generated fields)
        if metadata is None:
            metadata = {}

        metadata["timestamp"] = datetime.now().isoformat()
        metadata["content_length"] = len(content)
        self.metadata_store.store(memory_id, metadata)

        return memory_id

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant memories with content and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)

        # Enrich with metadata
        enriched_results = []
        for result in results:
            memory_id = result["memory_id"]
            metadata = self.metadata_store.get(memory_id)

            enriched_results.append(
                {
                    "memory_id": memory_id,
                    "content": result["content"],
                    "similarity": result["similarity"],
                    "metadata": metadata or {},
                }  # noqa: E501
            )

        return enriched_results

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete(memory_id)
            self.metadata_store.delete(memory_id)
            return True
        except Exception:
            return False

    def update(
        self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing memory.

        Args:
            memory_id: ID of the memory to update
            content: New content
            metadata: Updated metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate new embedding
            embedding = self.embedder.embed(content)

            # Update vector store
            self.vector_store.update(memory_id, embedding, content)

            # Update metadata if provided
            if metadata:
                self.metadata_store.update(memory_id, metadata)

            return True
        except Exception:
            return False

    def list_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all memories with basic info.

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of memory summaries
        """
        return self.vector_store.list(limit)
