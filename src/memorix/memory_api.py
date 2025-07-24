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
Enhanced Memory API for Memorix SDK with hierarchical memory support.

This module provides the main interface for memory operations with:
- Hierarchical memory model (STM → MTM → LTM)
- Declarative and scoped recall
- Comprehensive tracing and timeline tracking
- Configurable decay and scoring strategies
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .config import ConfigManager
from .embedder import Embedder
from .metadata_store import MetadataStore
from .vector_store import VectorStore
from .memory_hierarchy import (
    MemoryTierManager, MemoryEntry, MemoryTier, 
    DecayStrategy, ScoringStrategy
)
from .recall import (
    RecallEngine, DeclarativeQuery, RecallResult, 
    RecallScope, TimeRange, FilterCondition, FilterOperator
)
from .tracing import (
    TracingManager, OperationType, TraceLevel
)


class MemoryAPI:
    """
    Enhanced memory interface with hierarchical memory support.
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vector_store = VectorStore(config)
        self.embedder = Embedder(config)
        self.metadata_store = MetadataStore(config)
        
        # Initialize hierarchical memory components
        self.tier_manager = MemoryTierManager(config.config.dict())
        self.recall_engine = RecallEngine(
            self.tier_manager, self.embedder, self.vector_store, self.metadata_store
        )
        self.tracing_manager = TracingManager(config.config.dict())
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("MemoryAPI initialized with hierarchical memory support")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging_config = self.config.get_logging_config()
        logging.basicConfig(
            level=getattr(logging, logging_config.get("level", "INFO")),
            format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
              importance_score: float = 1.0, decay_strategy: Optional[DecayStrategy] = None,
              scoring_strategy: Optional[ScoringStrategy] = None) -> str:
        """
        Store content in hierarchical memory with enhanced metadata.

        Args:
            content: The content to store
            metadata: Optional metadata dictionary
            importance_score: Importance score (0.0 to 1.0)
            decay_strategy: Decay strategy for this memory
            scoring_strategy: Scoring strategy for this memory

        Returns:
            Memory ID of the stored content
        """
        with self.tracing_manager.trace_operation(OperationType.STORE) as event_id:
            memory_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedder.embed(content)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content),
                "importance_score": importance_score,
                "decay_strategy": decay_strategy.value if decay_strategy else self.config.config.memory.default_decay_strategy.value,
                "scoring_strategy": scoring_strategy.value if scoring_strategy else self.config.config.memory.default_scoring_strategy.value
            })
            
            # Create memory entry
            entry = MemoryEntry(
                memory_id=memory_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
                importance_score=importance_score,
                decay_strategy=decay_strategy or self.config.config.memory.default_decay_strategy,
                scoring_strategy=scoring_strategy or self.config.config.memory.default_scoring_strategy
            )
            
            # Store in vector store
            self.vector_store.store(memory_id, embedding, content)
            
            # Store metadata
            self.metadata_store.store(memory_id, metadata)
            
            # Add to hierarchical memory
            self.tier_manager.add_entry(entry)
            
            # Trace the operation
            self.tracing_manager.trace_memory_operation(
                OperationType.STORE,
                memory_id,
                content,
                metadata,
                tier=entry.tier.value,
                importance_score=importance_score
            )
            
            self.logger.info(f"Stored memory {memory_id} in {entry.tier.value}")
            return memory_id

    def retrieve(self, query: str, top_k: int = 5, scope: RecallScope = RecallScope.ALL) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using enhanced recall system.

        Args:
            query: Search query
            top_k: Number of results to return
            scope: Recall scope (which memory tiers to search)

        Returns:
            List of relevant memories with content and metadata
        """
        with self.tracing_manager.trace_operation(OperationType.RETRIEVE) as event_id:
            # Use declarative query for retrieval
            declarative_query = DeclarativeQuery(
                query_text=query,
                scope=scope,
                limit=top_k
            )
            
            results = self.recall_engine.recall(declarative_query)
            
            # Convert to legacy format for backward compatibility
            legacy_results = []
            for result in results:
                legacy_results.append({
                    "memory_id": result.memory_id,
                    "content": result.content,
                    "similarity": result.similarity,
                    "metadata": result.metadata,
                    "tier": result.tier.value,
                    "importance_score": result.importance_score,
                    "access_count": result.access_count
                })
            
            # Trace the operation
            self.tracing_manager.trace_recall_operation(
                query, len(results), scope.value
            )
            
            return legacy_results

    def recall(self, query: DeclarativeQuery) -> List[RecallResult]:
        """
        Perform declarative memory recall.

        Args:
            query: Declarative query specification

        Returns:
            List of recall results
        """
        return self.recall_engine.recall(query)

    def recall_by_tags(self, tags: List[str], scope: RecallScope = RecallScope.ALL, limit: int = 10) -> List[RecallResult]:
        """
        Recall memories by tags.

        Args:
            tags: List of required tags
            scope: Recall scope
            limit: Maximum number of results

        Returns:
            List of recall results
        """
        return self.recall_engine.recall_by_tags(tags, scope, limit)

    def recall_by_time_range(self, time_range: TimeRange, scope: RecallScope = RecallScope.ALL, limit: int = 10) -> List[RecallResult]:
        """
        Recall memories within a time range.

        Args:
            time_range: Time range filter
            scope: Recall scope
            limit: Maximum number of results

        Returns:
            List of recall results
        """
        return self.recall_engine.recall_by_time_range(time_range, scope, limit)

    def recall_by_agent(self, agent_ids: List[str], scope: RecallScope = RecallScope.ALL, limit: int = 10) -> List[RecallResult]:
        """
        Recall memories by agent IDs.

        Args:
            agent_ids: List of agent IDs
            scope: Recall scope
            limit: Maximum number of results

        Returns:
            List of recall results
        """
        return self.recall_engine.recall_by_agent(agent_ids, scope, limit)

    def recall_similar(self, content: str, scope: RecallScope = RecallScope.ALL, limit: int = 10, min_similarity: float = 0.7) -> List[RecallResult]:
        """
        Recall similar memories.

        Args:
            content: Content to find similar memories for
            scope: Recall scope
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of recall results
        """
        return self.recall_engine.recall_similar(content, scope, limit, min_similarity)

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if successful, False otherwise
        """
        with self.tracing_manager.trace_operation(OperationType.DELETE, memory_id) as event_id:
            try:
                # Remove from vector store
                self.vector_store.delete(memory_id)
                
                # Remove from metadata store
                self.metadata_store.delete(memory_id)
                
                # Remove from hierarchical memory (find and remove from all tiers)
                entry = self.tier_manager._find_entry(memory_id)
                if entry:
                    if entry.tier == MemoryTier.STM:
                        self.tier_manager.stm_entries.pop(memory_id, None)
                    elif entry.tier == MemoryTier.MTM:
                        self.tier_manager.mtm_entries.pop(memory_id, None)
                    elif entry.tier == MemoryTier.LTM:
                        self.tier_manager.ltm_entries.pop(memory_id, None)
                
                # Trace the operation
                self.tracing_manager.trace_memory_operation(
                    OperationType.DELETE,
                    memory_id
                )
                
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False

    def update(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None,
               importance_score: Optional[float] = None) -> bool:
        """
        Update an existing memory with enhanced capabilities.

        Args:
            memory_id: ID of the memory to update
            content: New content
            metadata: Updated metadata
            importance_score: New importance score

        Returns:
            True if successful, False otherwise
        """
        with self.tracing_manager.trace_operation(OperationType.UPDATE, memory_id) as event_id:
            try:
                # Generate new embedding
                embedding = self.embedder.embed(content)
                
                # Update vector store
                self.vector_store.update(memory_id, embedding, content)
                
                # Update metadata if provided
                if metadata:
                    self.metadata_store.update(memory_id, metadata)
                
                # Update hierarchical memory entry
                entry = self.tier_manager._find_entry(memory_id)
                if entry:
                    entry.content = content
                    entry.embedding = embedding
                    if metadata:
                        entry.metadata.update(metadata)
                    if importance_score is not None:
                        entry.importance_score = importance_score
                    
                    # Recalculate scores and check for tier migration
                    self.tier_manager._update_scores(entry, datetime.now())
                    self.tier_manager._check_tier_migration(entry, datetime.now())
                
                # Trace the operation
                self.tracing_manager.trace_memory_operation(
                    OperationType.UPDATE,
                    memory_id,
                    content,
                    metadata,
                    importance_score=importance_score
                )
                
                return True
            except Exception as e:
                self.logger.error(f"Failed to update memory {memory_id}: {e}")
                return False

    def access(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Access a memory and update its access properties.

        Args:
            memory_id: ID of the memory to access

        Returns:
            Memory data if found, None otherwise
        """
        # Access the memory in hierarchical system
        entry = self.tier_manager.access_entry(memory_id)
        if not entry:
            return None
        
        # Get metadata
        metadata = self.metadata_store.get(memory_id) or {}
        
        return {
            "memory_id": memory_id,
            "content": entry.content,
            "metadata": metadata,
            "tier": entry.tier.value,
            "importance_score": entry.importance_score,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed.isoformat(),
            "created_at": entry.created_at.isoformat(),
            "hybrid_score": entry.hybrid_score
        }

    def list_memories(self, limit: int = 100, tier: Optional[MemoryTier] = None) -> List[Dict[str, Any]]:
        """
        List memories with optional tier filtering.

        Args:
            limit: Maximum number of memories to return
            tier: Optional tier filter

        Returns:
            List of memory summaries
        """
        if tier:
            entries = self.tier_manager.get_entries_by_tier(tier)
        else:
            # Combine all tiers
            entries = {}
            entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.STM))
            entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.MTM))
            entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.LTM))
        
        # Convert to list format
        memories = []
        for memory_id, entry in list(entries.items())[:limit]:
            memories.append({
                "memory_id": memory_id,
                "content": entry.content,
                "tier": entry.tier.value,
                "importance_score": entry.importance_score,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat(),
                "created_at": entry.created_at.isoformat(),
                "hybrid_score": entry.hybrid_score
            })
        
        return memories

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary containing memory statistics
        """
        # Get hierarchical statistics
        tier_stats = self.tier_manager.get_statistics()
        
        # Get tracing statistics
        tracing_stats = self.tracing_manager.get_statistics()
        
        # Get metadata statistics
        all_metadata = self.metadata_store.list_all()
        metadata_stats = {
            "total_metadata_entries": len(all_metadata),
            "metadata_keys": list(set().union(*[set(md.keys()) for md in all_metadata.values()]))
        }
        
        return {
            "hierarchical_memory": tier_stats,
            "tracing": tracing_stats,
            "metadata": metadata_stats,
            "vector_store": {
                "total_vectors": len(self.vector_store.list(10000))  # Approximate
            }
        }

    def cleanup_expired_memories(self) -> int:
        """
        Clean up expired memories from all tiers.

        Returns:
            Number of memories cleaned up
        """
        with self.tracing_manager.trace_operation(OperationType.CLEANUP) as event_id:
            initial_count = (
                len(self.tier_manager.stm_entries) +
                len(self.tier_manager.mtm_entries) +
                len(self.tier_manager.ltm_entries)
            )
            
            self.tier_manager.cleanup_expired_entries()
            
            final_count = (
                len(self.tier_manager.stm_entries) +
                len(self.tier_manager.mtm_entries) +
                len(self.tier_manager.ltm_entries)
            )
            
            cleaned_count = initial_count - final_count
            
            self.logger.info(f"Cleaned up {cleaned_count} expired memories")
            return cleaned_count

    def export_timeline(self, file_path: str, format: str = "json") -> None:
        """
        Export timeline data to file.

        Args:
            file_path: Path to export file
            format: Export format (currently only "json" supported)
        """
        self.tracing_manager.export_timeline(file_path, format)

    def get_trace_events(self, operation_type: Optional[OperationType] = None, 
                        level: Optional[TraceLevel] = None, limit: int = 100) -> List[Any]:
        """
        Get trace events with optional filtering.

        Args:
            operation_type: Optional operation type filter
            level: Optional trace level filter
            limit: Maximum number of events to return

        Returns:
            List of trace events
        """
        return self.tracing_manager.get_events(operation_type, level, limit)

    def get_timeline_entries(self, memory_id: Optional[str] = None, 
                           operation_type: Optional[OperationType] = None, 
                           limit: int = 100) -> List[Any]:
        """
        Get timeline entries with optional filtering.

        Args:
            memory_id: Optional memory ID filter
            operation_type: Optional operation type filter
            limit: Maximum number of entries to return

        Returns:
            List of timeline entries
        """
        return self.tracing_manager.get_timeline_entries(memory_id, operation_type, limit)

    def add_timeline_hook(self, hook) -> None:
        """
        Add a timeline hook for external monitoring.

        Args:
            hook: Timeline hook instance
        """
        self.tracing_manager.add_hook(hook)

    def remove_timeline_hook(self, hook) -> None:
        """
        Remove a timeline hook.

        Args:
            hook: Timeline hook instance to remove
        """
        self.tracing_manager.remove_hook(hook)
