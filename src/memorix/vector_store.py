"""
Vector store interface with plug-in support for FAISS, Qdrant, etc.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np


class VectorStoreInterface(ABC):
    """
    Abstract interface for vector stores.
    """
    
    @abstractmethod
    def store(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Store a memory with its embedding."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        pass
    
    @abstractmethod
    def update(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Update an existing memory."""
        pass
    
    @abstractmethod
    def list(self, limit: int) -> List[Dict[str, Any]]:
        """List all memories."""
        pass


class FAISSVectorStore(VectorStoreInterface):
    """
    FAISS-based vector store implementation.
    """
    
    def __init__(self, config: 'Config'):
        self.config = config
        self.index_path = config.get('vector_store.index_path', './memorix_index')
        self.dimension = config.get('vector_store.dimension', 1536)
        
        # In-memory storage for demo purposes
        self.embeddings = {}
        self.contents = {}
        self.ids = []
        
    def store(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Store a memory with its embedding."""
        self.embeddings[memory_id] = embedding
        self.contents[memory_id] = content
        self.ids.append(memory_id)
    
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar memories using cosine similarity."""
        if not self.embeddings:
            return []
            
        similarities = []
        for memory_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((memory_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for memory_id, similarity in similarities[:top_k]:
            results.append({
                'memory_id': memory_id,
                'content': self.contents[memory_id],
                'similarity': similarity
            })
            
        return results
    
    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        if memory_id in self.embeddings:
            del self.embeddings[memory_id]
            del self.contents[memory_id]
            if memory_id in self.ids:
                self.ids.remove(memory_id)
    
    def update(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Update an existing memory."""
        if memory_id in self.embeddings:
            self.embeddings[memory_id] = embedding
            self.contents[memory_id] = content
    
    def list(self, limit: int) -> List[Dict[str, Any]]:
        """List all memories."""
        results = []
        for memory_id in self.ids[:limit]:
            results.append({
                'memory_id': memory_id,
                'content': self.contents[memory_id][:100] + '...' if len(self.contents[memory_id]) > 100 else self.contents[memory_id]
            })
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)


class QdrantVectorStore(VectorStoreInterface):
    """
    Qdrant-based vector store implementation.
    """
    
    def __init__(self, config: 'Config'):
        self.config = config
        # Placeholder for Qdrant implementation
        self.embeddings = {}
        self.contents = {}
        self.ids = []
    
    def store(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Store a memory with its embedding."""
        self.embeddings[memory_id] = embedding
        self.contents[memory_id] = content
        self.ids.append(memory_id)
    
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        # Placeholder implementation
        return []
    
    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        if memory_id in self.embeddings:
            del self.embeddings[memory_id]
            del self.contents[memory_id]
            if memory_id in self.ids:
                self.ids.remove(memory_id)
    
    def update(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Update an existing memory."""
        if memory_id in self.embeddings:
            self.embeddings[memory_id] = embedding
            self.contents[memory_id] = content
    
    def list(self, limit: int) -> List[Dict[str, Any]]:
        """List all memories."""
        results = []
        for memory_id in self.ids[:limit]:
            results.append({
                'memory_id': memory_id,
                'content': self.contents[memory_id][:100] + '...' if len(self.contents[memory_id]) > 100 else self.contents[memory_id]
            })
        return results


class VectorStore:
    """
    Vector store factory and interface.
    """
    
    def __init__(self, config: 'Config'):
        self.config = config
        self.store_type = config.get('vector_store.type', 'faiss')
        self._store = self._create_store()
    
    def _create_store(self) -> VectorStoreInterface:
        """Create vector store based on configuration."""
        if self.store_type == 'faiss':
            return FAISSVectorStore(self.config)
        elif self.store_type == 'qdrant':
            return QdrantVectorStore(self.config)
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def store(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Store a memory with its embedding."""
        self._store.store(memory_id, embedding, content)
    
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        return self._store.search(query_embedding, top_k)
    
    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._store.delete(memory_id)
    
    def update(self, memory_id: str, embedding: List[float], content: str) -> None:
        """Update an existing memory."""
        self._store.update(memory_id, embedding, content)
    
    def list(self, limit: int) -> List[Dict[str, Any]]:
        """List all memories."""
        return self._store.list(limit) 