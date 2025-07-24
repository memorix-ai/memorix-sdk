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
Enhanced Recall System for Memorix SDK

This module implements declarative and scoped memory recall with support for:
- Tag-based filtering
- Time-based filtering
- Agent-based filtering
- Semantic similarity search
- Hierarchical tier-based recall
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from .memory_hierarchy import MemoryTier, MemoryEntry, MemoryTierManager


class RecallScope(Enum):
    """Recall scope options."""
    ALL = "all"
    STM_ONLY = "stm_only"
    MTM_ONLY = "mtm_only"
    LTM_ONLY = "ltm_only"
    STM_MTM = "stm_mtm"
    MTM_LTM = "mtm_ltm"


class FilterOperator(Enum):
    """Filter operators for declarative queries."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


@dataclass
class FilterCondition:
    """Represents a filter condition for declarative queries."""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = True


@dataclass
class TimeRange:
    """Represents a time range for filtering."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within the time range."""
        if self.start_time and timestamp < self.start_time:
            return False
        if self.end_time and timestamp > self.end_time:
            return False
        return True


class DeclarativeQuery(BaseModel):
    """Declarative query for memory recall."""
    
    # Basic query
    query_text: Optional[str] = Field(default=None, description="Semantic search query")
    
    # Scope and limits
    scope: RecallScope = Field(default=RecallScope.ALL, description="Recall scope")
    limit: int = Field(default=10, description="Maximum number of results")
    offset: int = Field(default=0, description="Result offset")
    
    # Filtering
    tags: Optional[List[str]] = Field(default=None, description="Required tags")
    exclude_tags: Optional[List[str]] = Field(default=None, description="Excluded tags")
    agent_ids: Optional[List[str]] = Field(default=None, description="Filter by agent IDs")
    time_range: Optional[TimeRange] = Field(default=None, description="Time range filter")
    
    # Advanced filtering
    filters: List[FilterCondition] = Field(default_factory=list, description="Custom filters")
    
    # Scoring and ranking
    min_similarity: float = Field(default=0.0, description="Minimum similarity threshold")
    min_importance: float = Field(default=0.0, description="Minimum importance score")
    sort_by: str = Field(default="similarity", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order (asc/desc)")
    
    # Hierarchical options
    include_tier_info: bool = Field(default=True, description="Include tier information")
    tier_weights: Dict[str, float] = Field(
        default={"stm": 1.0, "mtm": 0.8, "ltm": 0.6},
        description="Weights for different tiers"
    )


class RecallResult(BaseModel):
    """Result of a memory recall operation."""
    
    memory_id: str
    content: str
    similarity: float
    tier: MemoryTier
    metadata: Dict[str, Any]
    
    # Additional information
    importance_score: float
    access_count: int
    last_accessed: datetime
    created_at: datetime
    
    # Hierarchical information
    tier_score: float
    final_score: float


class FilterEvaluator(ABC):
    """Abstract base class for filter evaluation."""
    
    @abstractmethod
    def evaluate(self, entry: MemoryEntry, condition: FilterCondition) -> bool:
        """Evaluate a filter condition against a memory entry."""
        pass


class MetadataFilterEvaluator(FilterEvaluator):
    """Evaluates filters against metadata fields."""
    
    def evaluate(self, entry: MemoryEntry, condition: FilterCondition) -> bool:
        """Evaluate metadata filter condition."""
        field_value = entry.metadata.get(condition.field)
        
        if condition.operator == FilterOperator.EXISTS:
            return field_value is not None
        elif condition.operator == FilterOperator.NOT_EXISTS:
            return field_value is None
        
        if field_value is None:
            return False
        
        return self._compare_values(field_value, condition.value, condition.operator, condition.case_sensitive)
    
    def _compare_values(self, field_value: Any, condition_value: Any, operator: FilterOperator, case_sensitive: bool) -> bool:
        """Compare values based on operator."""
        if not case_sensitive and isinstance(field_value, str) and isinstance(condition_value, str):
            field_value = field_value.lower()
            condition_value = condition_value.lower()
        
        if operator == FilterOperator.EQUALS:
            return field_value == condition_value
        elif operator == FilterOperator.NOT_EQUALS:
            return field_value != condition_value
        elif operator == FilterOperator.GREATER_THAN:
            return field_value > condition_value
        elif operator == FilterOperator.LESS_THAN:
            return field_value < condition_value
        elif operator == FilterOperator.GREATER_EQUAL:
            return field_value >= condition_value
        elif operator == FilterOperator.LESS_EQUAL:
            return field_value <= condition_value
        elif operator == FilterOperator.CONTAINS:
            if isinstance(field_value, str) and isinstance(condition_value, str):
                return condition_value in field_value
            elif isinstance(field_value, list):
                return condition_value in field_value
            return False
        elif operator == FilterOperator.NOT_CONTAINS:
            if isinstance(field_value, str) and isinstance(condition_value, str):
                return condition_value not in field_value
            elif isinstance(field_value, list):
                return condition_value not in field_value
            return True
        elif operator == FilterOperator.IN:
            if isinstance(condition_value, list):
                return field_value in condition_value
            return False
        elif operator == FilterOperator.NOT_IN:
            if isinstance(condition_value, list):
                return field_value not in condition_value
            return True
        
        return False


class RecallEngine:
    """Enhanced recall engine with declarative and scoped capabilities."""
    
    def __init__(self, tier_manager: MemoryTierManager, embedder, vector_store, metadata_store):
        self.tier_manager = tier_manager
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.logger = logging.getLogger(__name__)
        
        self.filter_evaluator = MetadataFilterEvaluator()
    
    def recall(self, query: DeclarativeQuery) -> List[RecallResult]:
        """
        Perform declarative memory recall.
        
        Args:
            query: Declarative query specification
            
        Returns:
            List of recall results
        """
        self.logger.info(f"Executing recall query: {query.query_text}")
        
        # Get entries based on scope
        entries = self._get_entries_by_scope(query.scope)
        
        # Apply filters
        filtered_entries = self._apply_filters(entries, query)
        
        # Perform semantic search if query text provided
        if query.query_text:
            filtered_entries = self._apply_semantic_search(filtered_entries, query)
        
        # Score and rank results
        scored_entries = self._score_entries(filtered_entries, query)
        
        # Sort and limit results
        final_results = self._sort_and_limit(scored_entries, query)
        
        self.logger.info(f"Recall returned {len(final_results)} results")
        return final_results
    
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
        query = DeclarativeQuery(
            tags=tags,
            scope=scope,
            limit=limit
        )
        return self.recall(query)
    
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
        query = DeclarativeQuery(
            time_range=time_range,
            scope=scope,
            limit=limit
        )
        return self.recall(query)
    
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
        query = DeclarativeQuery(
            agent_ids=agent_ids,
            scope=scope,
            limit=limit
        )
        return self.recall(query)
    
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
        query = DeclarativeQuery(
            query_text=content,
            scope=scope,
            limit=limit,
            min_similarity=min_similarity
        )
        return self.recall(query)
    
    def _get_entries_by_scope(self, scope: RecallScope) -> Dict[str, MemoryEntry]:
        """Get memory entries based on scope."""
        if scope == RecallScope.ALL:
            # Combine all tiers
            all_entries = {}
            all_entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.STM))
            all_entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.MTM))
            all_entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.LTM))
            return all_entries
        elif scope == RecallScope.STM_ONLY:
            return self.tier_manager.get_entries_by_tier(MemoryTier.STM)
        elif scope == RecallScope.MTM_ONLY:
            return self.tier_manager.get_entries_by_tier(MemoryTier.MTM)
        elif scope == RecallScope.LTM_ONLY:
            return self.tier_manager.get_entries_by_tier(MemoryTier.LTM)
        elif scope == RecallScope.STM_MTM:
            entries = {}
            entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.STM))
            entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.MTM))
            return entries
        elif scope == RecallScope.MTM_LTM:
            entries = {}
            entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.MTM))
            entries.update(self.tier_manager.get_entries_by_tier(MemoryTier.LTM))
            return entries
        else:
            raise ValueError(f"Unknown scope: {scope}")
    
    def _apply_filters(self, entries: Dict[str, MemoryEntry], query: DeclarativeQuery) -> Dict[str, MemoryEntry]:
        """Apply all filters to memory entries."""
        filtered_entries = entries.copy()
        
        # Tag filtering
        if query.tags:
            filtered_entries = self._filter_by_tags(filtered_entries, query.tags, include=True)
        
        if query.exclude_tags:
            filtered_entries = self._filter_by_tags(filtered_entries, query.exclude_tags, include=False)
        
        # Agent filtering
        if query.agent_ids:
            filtered_entries = self._filter_by_agents(filtered_entries, query.agent_ids)
        
        # Time range filtering
        if query.time_range:
            filtered_entries = self._filter_by_time_range(filtered_entries, query.time_range)
        
        # Custom filters
        if query.filters:
            filtered_entries = self._apply_custom_filters(filtered_entries, query.filters)
        
        return filtered_entries
    
    def _filter_by_tags(self, entries: Dict[str, MemoryEntry], tags: List[str], include: bool) -> Dict[str, MemoryEntry]:
        """Filter entries by tags."""
        filtered = {}
        
        for memory_id, entry in entries.items():
            entry_tags = entry.metadata.get("tags", [])
            if isinstance(entry_tags, str):
                entry_tags = [entry_tags]
            
            has_required_tags = all(tag in entry_tags for tag in tags)
            
            if (include and has_required_tags) or (not include and not has_required_tags):
                filtered[memory_id] = entry
        
        return filtered
    
    def _filter_by_agents(self, entries: Dict[str, MemoryEntry], agent_ids: List[str]) -> Dict[str, MemoryEntry]:
        """Filter entries by agent IDs."""
        filtered = {}
        
        for memory_id, entry in entries.items():
            entry_agent = entry.metadata.get("agent_id")
            if entry_agent in agent_ids:
                filtered[memory_id] = entry
        
        return filtered
    
    def _filter_by_time_range(self, entries: Dict[str, MemoryEntry], time_range: TimeRange) -> Dict[str, MemoryEntry]:
        """Filter entries by time range."""
        filtered = {}
        
        for memory_id, entry in entries.items():
            created_time = entry.created_at
            if time_range.contains(created_time):
                filtered[memory_id] = entry
        
        return filtered
    
    def _apply_custom_filters(self, entries: Dict[str, MemoryEntry], filters: List[FilterCondition]) -> Dict[str, MemoryEntry]:
        """Apply custom filters to entries."""
        filtered = {}
        
        for memory_id, entry in entries.items():
            passes_all_filters = True
            
            for condition in filters:
                if not self.filter_evaluator.evaluate(entry, condition):
                    passes_all_filters = False
                    break
            
            if passes_all_filters:
                filtered[memory_id] = entry
        
        return filtered
    
    def _apply_semantic_search(self, entries: Dict[str, MemoryEntry], query: DeclarativeQuery) -> Dict[str, MemoryEntry]:
        """Apply semantic search to filter entries."""
        if not query.query_text:
            return entries
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query.query_text)
        
        # Calculate similarities
        similarities = {}
        for memory_id, entry in entries.items():
            similarity = self._calculate_similarity(query_embedding, entry.embedding)
            if similarity >= query.min_similarity:
                similarities[memory_id] = similarity
        
        # Sort by similarity and limit
        sorted_ids = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)
        limited_ids = sorted_ids[:query.limit]
        
        return {memory_id: entries[memory_id] for memory_id in limited_ids if memory_id in entries}
    
    def _calculate_similarity(self, query_embedding: np.ndarray, entry_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        dot_product = np.dot(query_embedding, entry_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_entry = np.linalg.norm(entry_embedding)
        
        if norm_query == 0 or norm_entry == 0:
            return 0.0
        
        return dot_product / (norm_query * norm_entry)
    
    def _score_entries(self, entries: Dict[str, MemoryEntry], query: DeclarativeQuery) -> List[tuple]:
        """Score entries based on query parameters."""
        scored_entries = []
        
        for memory_id, entry in entries.items():
            # Calculate tier score
            tier_weight = query.tier_weights.get(entry.tier.value, 1.0)
            tier_score = entry.hybrid_score * tier_weight
            
            # Calculate final score
            final_score = tier_score
            
            # Apply importance filter
            if entry.importance_score < query.min_importance:
                continue
            
            scored_entries.append((memory_id, entry, tier_score, final_score))
        
        return scored_entries
    
    def _sort_and_limit(self, scored_entries: List[tuple], query: DeclarativeQuery) -> List[RecallResult]:
        """Sort and limit results."""
        # Sort by specified field
        reverse = query.sort_order.lower() == "desc"
        
        if query.sort_by == "similarity":
            scored_entries.sort(key=lambda x: x[3], reverse=reverse)  # final_score
        elif query.sort_by == "importance":
            scored_entries.sort(key=lambda x: x[1].importance_score, reverse=reverse)
        elif query.sort_by == "recency":
            scored_entries.sort(key=lambda x: x[1].last_accessed, reverse=reverse)
        elif query.sort_by == "created":
            scored_entries.sort(key=lambda x: x[1].created_at, reverse=reverse)
        else:
            # Default to similarity
            scored_entries.sort(key=lambda x: x[3], reverse=reverse)
        
        # Apply offset and limit
        start_idx = query.offset
        end_idx = start_idx + query.limit
        limited_entries = scored_entries[start_idx:end_idx]
        
        # Convert to RecallResult objects
        results = []
        for memory_id, entry, tier_score, final_score in limited_entries:
            result = RecallResult(
                memory_id=memory_id,
                content=entry.content,
                similarity=final_score,
                tier=entry.tier,
                metadata=entry.metadata,
                importance_score=entry.importance_score,
                access_count=entry.access_count,
                last_accessed=entry.last_accessed,
                created_at=entry.created_at,
                tier_score=tier_score,
                final_score=final_score
            )
            results.append(result)
        
        return results 