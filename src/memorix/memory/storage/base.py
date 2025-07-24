"""
Copyright 2025 Memorix AI Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Standard Plugin Interfaces for Memory Storage

This module defines the standard plugin interfaces for:
- VectorStoreInterface: Vector storage and similarity search
- MetadataStoreInterface: Metadata storage and retrieval
- StoragePlugin: Base interface for all storage plugins
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class StorageConfig:
    """Configuration for storage plugins."""
    plugin_type: str
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 0  # Higher priority plugins are used first


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    memory_id: str
    content: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None
    tier: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class StorageMetrics:
    """Metrics for storage operations."""
    operation_type: str
    memory_id: str
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StoragePlugin(ABC):
    """Base interface for all storage plugins."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metrics: List[StorageMetrics] = []
        self.max_metrics = 1000
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the storage plugin."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the storage plugin is available and ready."""
        pass
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about the plugin."""
        pass
    
    def record_metric(self, operation_type: str, memory_id: str, 
                     duration_ms: float, success: bool, 
                     error_message: Optional[str] = None) -> None:
        """Record a storage operation metric."""
        metric = StorageMetrics(
            operation_type=operation_type,
            memory_id=memory_id,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )
        
        self.metrics.append(metric)
        
        # Trim metrics if too many
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_metrics(self, operation_type: Optional[str] = None) -> List[StorageMetrics]:
        """Get storage metrics with optional filtering."""
        if operation_type:
            return [m for m in self.metrics if m.operation_type == operation_type]
        return self.metrics.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.metrics:
            return {"total_operations": 0, "success_rate": 0.0}
        
        total_ops = len(self.metrics)
        successful_ops = len([m for m in self.metrics if m.success])
        success_rate = successful_ops / total_ops if total_ops > 0 else 0.0
        
        # Calculate average duration
        durations = [m.duration_ms for m in self.metrics if m.success]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Operation type breakdown
        op_types = {}
        for metric in self.metrics:
            op_type = metric.operation_type
            if op_type not in op_types:
                op_types[op_type] = {"total": 0, "success": 0}
            op_types[op_type]["total"] += 1
            if metric.success:
                op_types[op_type]["success"] += 1
        
        return {
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration,
            "operation_types": op_types,
            "plugin_info": self.get_plugin_info()
        }


class VectorStoreInterface(StoragePlugin):
    """Interface for vector storage and similarity search."""
    
    @abstractmethod
    def store_vector(self, memory_id: str, embedding: np.ndarray, 
                    content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a vector with associated content and metadata."""
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10, 
                      similarity_threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def update_vector(self, memory_id: str, embedding: np.ndarray, 
                     content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing vector."""
        pass
    
    @abstractmethod
    def delete_vector(self, memory_id: str) -> bool:
        """Delete a vector by memory ID."""
        pass
    
    @abstractmethod
    def get_vector(self, memory_id: str) -> Optional[SearchResult]:
        """Get a specific vector by memory ID."""
        pass
    
    @abstractmethod
    def list_vectors(self, limit: int = 100, offset: int = 0) -> List[SearchResult]:
        """List vectors with pagination."""
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """Get the total number of stored vectors."""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all stored vectors."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension supported by this store."""
        pass
    
    def batch_store(self, vectors: List[Tuple[str, np.ndarray, str, Optional[Dict[str, Any]]]]) -> List[bool]:
        """Store multiple vectors in batch."""
        results = []
        for memory_id, embedding, content, metadata in vectors:
            try:
                success = self.store_vector(memory_id, embedding, content, metadata)
                results.append(success)
            except Exception as e:
                self.logger.error(f"Failed to store vector {memory_id}: {e}")
                results.append(False)
        return results
    
    def batch_delete(self, memory_ids: List[str]) -> List[bool]:
        """Delete multiple vectors in batch."""
        results = []
        for memory_id in memory_ids:
            try:
                success = self.delete_vector(memory_id)
                results.append(success)
            except Exception as e:
                self.logger.error(f"Failed to delete vector {memory_id}: {e}")
                results.append(False)
        return results


class MetadataStoreInterface(StoragePlugin):
    """Interface for metadata storage and retrieval."""
    
    @abstractmethod
    def store_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        """Store metadata for a memory."""
        pass
    
    @abstractmethod
    def get_metadata(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a memory."""
        pass
    
    @abstractmethod
    def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a memory."""
        pass
    
    @abstractmethod
    def delete_metadata(self, memory_id: str) -> bool:
        """Delete metadata for a memory."""
        pass
    
    @abstractmethod
    def list_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """List all metadata."""
        pass
    
    @abstractmethod
    def search_metadata(self, query: Dict[str, Any]) -> List[str]:
        """Search for memory IDs based on metadata criteria."""
        pass
    
    @abstractmethod
    def get_metadata_count(self) -> int:
        """Get the total number of metadata entries."""
        pass
    
    @abstractmethod
    def clear_all_metadata(self) -> bool:
        """Clear all metadata."""
        pass
    
    def batch_store_metadata(self, metadata_batch: List[Tuple[str, Dict[str, Any]]]) -> List[bool]:
        """Store multiple metadata entries in batch."""
        results = []
        for memory_id, metadata in metadata_batch:
            try:
                success = self.store_metadata(memory_id, metadata)
                results.append(success)
            except Exception as e:
                self.logger.error(f"Failed to store metadata for {memory_id}: {e}")
                results.append(False)
        return results
    
    def batch_update_metadata(self, metadata_batch: List[Tuple[str, Dict[str, Any]]]) -> List[bool]:
        """Update multiple metadata entries in batch."""
        results = []
        for memory_id, metadata in metadata_batch:
            try:
                success = self.update_metadata(memory_id, metadata)
                results.append(success)
            except Exception as e:
                self.logger.error(f"Failed to update metadata for {memory_id}: {e}")
                results.append(False)
        return results
    
    def batch_delete_metadata(self, memory_ids: List[str]) -> List[bool]:
        """Delete multiple metadata entries in batch."""
        results = []
        for memory_id in memory_ids:
            try:
                success = self.delete_metadata(memory_id)
                results.append(success)
            except Exception as e:
                self.logger.error(f"Failed to delete metadata for {memory_id}: {e}")
                results.append(False)
        return results


class StoragePluginRegistry:
    """Registry for managing storage plugins."""
    
    def __init__(self):
        self.vector_stores: Dict[str, VectorStoreInterface] = {}
        self.metadata_stores: Dict[str, MetadataStoreInterface] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_vector_store(self, name: str, plugin: VectorStoreInterface) -> None:
        """Register a vector store plugin."""
        if not isinstance(plugin, VectorStoreInterface):
            raise ValueError(f"Plugin must implement VectorStoreInterface")
        
        self.vector_stores[name] = plugin
        self.logger.info(f"Registered vector store plugin: {name}")
    
    def register_metadata_store(self, name: str, plugin: MetadataStoreInterface) -> None:
        """Register a metadata store plugin."""
        if not isinstance(plugin, MetadataStoreInterface):
            raise ValueError(f"Plugin must implement MetadataStoreInterface")
        
        self.metadata_stores[name] = plugin
        self.logger.info(f"Registered metadata store plugin: {name}")
    
    def get_vector_store(self, name: str) -> Optional[VectorStoreInterface]:
        """Get a vector store plugin by name."""
        return self.vector_stores.get(name)
    
    def get_metadata_store(self, name: str) -> Optional[MetadataStoreInterface]:
        """Get a metadata store plugin by name."""
        return self.metadata_stores.get(name)
    
    def list_vector_stores(self) -> List[str]:
        """List all registered vector store names."""
        return list(self.vector_stores.keys())
    
    def list_metadata_stores(self) -> List[str]:
        """List all registered metadata store names."""
        return list(self.metadata_stores.keys())
    
    def unregister_vector_store(self, name: str) -> bool:
        """Unregister a vector store plugin."""
        if name in self.vector_stores:
            del self.vector_stores[name]
            self.logger.info(f"Unregistered vector store plugin: {name}")
            return True
        return False
    
    def unregister_metadata_store(self, name: str) -> bool:
        """Unregister a metadata store plugin."""
        if name in self.metadata_stores:
            del self.metadata_stores[name]
            self.logger.info(f"Unregistered metadata store plugin: {name}")
            return True
        return False
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get statistics for all registered plugins."""
        stats = {
            "vector_stores": {},
            "metadata_stores": {},
            "total_plugins": len(self.vector_stores) + len(self.metadata_stores)
        }
        
        for name, plugin in self.vector_stores.items():
            stats["vector_stores"][name] = plugin.get_statistics()
        
        for name, plugin in self.metadata_stores.items():
            stats["metadata_stores"][name] = plugin.get_statistics()
        
        return stats


# Global plugin registry
plugin_registry = StoragePluginRegistry() 