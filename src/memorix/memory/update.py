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
Memory Update System with FIFO Flow and Adaptive Decay Scoring

This module implements the adaptive memory update system that manages:
- FIFO (First-In-First-Out) flow between memory tiers
- Adaptive decay scoring based on access patterns
- Automatic tier migration triggers
- Memory consolidation and optimization
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from ..memory_hierarchy import (
    MemoryEntry, MemoryTier, DecayStrategy, ScoringStrategy,
    MemoryTierManager
)


@dataclass
class UpdateMetrics:
    """Metrics for memory update operations."""
    memory_id: str
    tier_from: MemoryTier
    tier_to: MemoryTier
    migration_reason: str
    score_before: float
    score_after: float
    access_count: int
    time_in_tier: float
    decay_factor: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FIFOBuffer:
    """FIFO buffer for managing memory flow between tiers."""
    max_size: int
    buffer: deque = field(default_factory=deque)
    
    def add(self, memory_id: str) -> Optional[str]:
        """Add memory ID to buffer, return evicted ID if buffer is full."""
        if len(self.buffer) >= self.max_size:
            evicted = self.buffer.popleft()
            self.buffer.append(memory_id)
            return evicted
        else:
            self.buffer.append(memory_id)
            return None
    
    def remove(self, memory_id: str) -> bool:
        """Remove memory ID from buffer."""
        try:
            self.buffer.remove(memory_id)
            return True
        except ValueError:
            return False
    
    def contains(self, memory_id: str) -> bool:
        """Check if memory ID is in buffer."""
        return memory_id in self.buffer
    
    def get_oldest(self) -> Optional[str]:
        """Get oldest memory ID in buffer."""
        return self.buffer[0] if self.buffer else None
    
    def get_newest(self) -> Optional[str]:
        """Get newest memory ID in buffer."""
        return self.buffer[-1] if self.buffer else None
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class AdaptiveDecayCalculator(ABC):
    """Abstract base class for adaptive decay calculations."""
    
    @abstractmethod
    def calculate_adaptive_decay(self, entry: MemoryEntry, 
                               access_pattern: Dict[str, Any],
                               current_time: datetime) -> float:
        """Calculate adaptive decay factor based on access patterns."""
        pass


class FrequencyBasedDecay(AdaptiveDecayCalculator):
    """Frequency-based adaptive decay calculator."""
    
    def __init__(self, base_decay_rate: float = 0.1, frequency_boost: float = 0.2):
        self.base_decay_rate = base_decay_rate
        self.frequency_boost = frequency_boost
    
    def calculate_adaptive_decay(self, entry: MemoryEntry, 
                               access_pattern: Dict[str, Any],
                               current_time: datetime) -> float:
        """Calculate decay based on access frequency."""
        time_alive = (current_time - entry.created_at).total_seconds()
        if time_alive <= 0:
            return 1.0
        
        # Calculate frequency factor
        frequency_factor = entry.access_count / time_alive
        
        # Adaptive decay rate: higher frequency = slower decay
        adaptive_rate = self.base_decay_rate * (1.0 - (frequency_factor * self.frequency_boost))
        adaptive_rate = max(0.01, min(0.5, adaptive_rate))  # Clamp between 0.01 and 0.5
        
        # Apply decay based on strategy
        if entry.decay_strategy == DecayStrategy.EXPONENTIAL:
            return np.exp(-adaptive_rate * time_alive)
        elif entry.decay_strategy == DecayStrategy.LINEAR:
            return max(0.0, 1.0 - (adaptive_rate * time_alive))
        elif entry.decay_strategy == DecayStrategy.STEP:
            steps = int(time_alive / (1.0 / adaptive_rate))
            return max(0.0, 1.0 - (steps * 0.1))
        else:  # NoDecay
            return 1.0


class ContextAwareDecay(AdaptiveDecayCalculator):
    """Context-aware adaptive decay calculator."""
    
    def __init__(self, base_decay_rate: float = 0.1, context_weights: Dict[str, float] = None):
        self.base_decay_rate = base_decay_rate
        self.context_weights = context_weights or {
            "importance": 0.4,
            "recency": 0.3,
            "frequency": 0.2,
            "relevance": 0.1
        }
    
    def calculate_adaptive_decay(self, entry: MemoryEntry, 
                               access_pattern: Dict[str, Any],
                               current_time: datetime) -> float:
        """Calculate decay based on multiple contextual factors."""
        time_alive = (current_time - entry.created_at).total_seconds()
        if time_alive <= 0:
            return 1.0
        
        # Calculate contextual factors
        importance_factor = entry.importance_score
        recency_factor = self._calculate_recency_factor(entry, current_time)
        frequency_factor = self._calculate_frequency_factor(entry, time_alive)
        relevance_factor = self._calculate_relevance_factor(entry, access_pattern)
        
        # Combine factors
        context_score = (
            self.context_weights["importance"] * importance_factor +
            self.context_weights["recency"] * recency_factor +
            self.context_weights["frequency"] * frequency_factor +
            self.context_weights["relevance"] * relevance_factor
        )
        
        # Adaptive decay rate based on context
        adaptive_rate = self.base_decay_rate * (1.0 - context_score)
        adaptive_rate = max(0.01, min(0.5, adaptive_rate))
        
        # Apply decay
        if entry.decay_strategy == DecayStrategy.EXPONENTIAL:
            return np.exp(-adaptive_rate * time_alive)
        elif entry.decay_strategy == DecayStrategy.LINEAR:
            return max(0.0, 1.0 - (adaptive_rate * time_alive))
        elif entry.decay_strategy == DecayStrategy.STEP:
            steps = int(time_alive / (1.0 / adaptive_rate))
            return max(0.0, 1.0 - (steps * 0.1))
        else:  # NoDecay
            return 1.0
    
    def _calculate_recency_factor(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate recency factor."""
        time_since_access = (current_time - entry.last_accessed).total_seconds()
        return np.exp(-time_since_access / 3600)  # 1 hour half-life
    
    def _calculate_frequency_factor(self, entry: MemoryEntry, time_alive: float) -> float:
        """Calculate frequency factor."""
        if time_alive <= 0:
            return 0.0
        frequency = entry.access_count / time_alive
        return min(1.0, frequency)  # Normalize to 0-1
    
    def _calculate_relevance_factor(self, entry: MemoryEntry, access_pattern: Dict[str, Any]) -> float:
        """Calculate relevance factor based on access pattern."""
        # Simple relevance based on metadata tags and access context
        if "relevance_score" in entry.metadata:
            return entry.metadata["relevance_score"]
        return 0.5  # Default relevance


class MemoryUpdateManager:
    """Manages memory updates with FIFO flow and adaptive decay."""
    
    def __init__(self, tier_manager: MemoryTierManager, config: Dict[str, Any]):
        self.tier_manager = tier_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # FIFO buffers for each tier
        self.stm_buffer = FIFOBuffer(config.get("memory.stm_capacity", 100))
        self.mtm_buffer = FIFOBuffer(config.get("memory.mtm_capacity", 1000))
        self.ltm_buffer = FIFOBuffer(config.get("memory.ltm_capacity", 100000))
        
        # Adaptive decay calculator
        decay_type = config.get("memory.adaptive_decay_type", "frequency")
        if decay_type == "context":
            self.decay_calculator = ContextAwareDecay()
        else:
            self.decay_calculator = FrequencyBasedDecay()
        
        # Update metrics tracking
        self.update_metrics: List[UpdateMetrics] = []
        self.max_metrics = config.get("memory.max_update_metrics", 10000)
        
        # Migration thresholds
        self.migration_thresholds = {
            MemoryTier.STM: {
                "promote": 0.7,  # Score threshold for STM → MTM
                "demote": 0.1    # Score threshold for STM → deletion
            },
            MemoryTier.MTM: {
                "promote": 0.8,  # Score threshold for MTM → LTM
                "demote": 0.3    # Score threshold for MTM → STM
            },
            MemoryTier.LTM: {
                "demote": 0.5    # Score threshold for LTM → MTM
            }
        }
    
    def update_memory(self, memory_id: str, access_context: Dict[str, Any] = None) -> UpdateMetrics:
        """
        Update memory with adaptive decay and potential tier migration.
        
        Args:
            memory_id: ID of the memory to update
            access_context: Context information about the access
            
        Returns:
            UpdateMetrics object with migration details
        """
        # Find the memory entry
        entry = self.tier_manager._find_entry(memory_id)
        if not entry:
            raise ValueError(f"Memory {memory_id} not found")
        
        old_tier = entry.tier
        old_score = entry.hybrid_score
        
        # Update access properties
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        # Calculate adaptive decay
        access_pattern = access_context or {}
        decay_factor = self.decay_calculator.calculate_adaptive_decay(
            entry, access_pattern, datetime.now()
        )
        
        # Update scores with adaptive decay
        self._update_scores_with_decay(entry, decay_factor)
        
        # Check for tier migration
        new_tier = self._determine_tier_migration(entry)
        
        # Perform migration if needed
        if new_tier != old_tier:
            self._perform_migration(entry, old_tier, new_tier)
        
        # Create update metrics
        metrics = UpdateMetrics(
            memory_id=memory_id,
            tier_from=old_tier,
            tier_to=new_tier,
            migration_reason=self._get_migration_reason(entry, old_tier, new_tier),
            score_before=old_score,
            score_after=entry.hybrid_score,
            access_count=entry.access_count,
            time_in_tier=(datetime.now() - entry.created_at).total_seconds(),
            decay_factor=decay_factor
        )
        
        # Store metrics
        self._store_metrics(metrics)
        
        self.logger.debug(f"Updated memory {memory_id}: {old_tier.value} → {new_tier.value}")
        return metrics
    
    def batch_update(self, memory_ids: List[str], 
                    access_contexts: Optional[List[Dict[str, Any]]] = None) -> List[UpdateMetrics]:
        """
        Update multiple memories in batch.
        
        Args:
            memory_ids: List of memory IDs to update
            access_contexts: Optional list of access contexts
            
        Returns:
            List of UpdateMetrics objects
        """
        if access_contexts is None:
            access_contexts = [{}] * len(memory_ids)
        
        metrics = []
        for memory_id, context in zip(memory_ids, access_contexts):
            try:
                metric = self.update_memory(memory_id, context)
                metrics.append(metric)
            except Exception as e:
                self.logger.error(f"Failed to update memory {memory_id}: {e}")
        
        return metrics
    
    def optimize_memory_distribution(self) -> Dict[str, int]:
        """
        Optimize memory distribution across tiers.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            "stm_optimized": 0,
            "mtm_optimized": 0,
            "ltm_optimized": 0,
            "total_migrations": 0
        }
        
        # Optimize STM (promote high-value memories)
        stm_entries = self.tier_manager.get_entries_by_tier(MemoryTier.STM)
        for memory_id, entry in stm_entries.items():
            if entry.hybrid_score >= self.migration_thresholds[MemoryTier.STM]["promote"]:
                self._perform_migration(entry, MemoryTier.STM, MemoryTier.MTM)
                stats["stm_optimized"] += 1
                stats["total_migrations"] += 1
        
        # Optimize MTM (promote to LTM, demote to STM)
        mtm_entries = self.tier_manager.get_entries_by_tier(MemoryTier.MTM)
        for memory_id, entry in mtm_entries.items():
            if entry.hybrid_score >= self.migration_thresholds[MemoryTier.MTM]["promote"]:
                self._perform_migration(entry, MemoryTier.MTM, MemoryTier.LTM)
                stats["mtm_optimized"] += 1
                stats["total_migrations"] += 1
            elif entry.hybrid_score <= self.migration_thresholds[MemoryTier.MTM]["demote"]:
                self._perform_migration(entry, MemoryTier.MTM, MemoryTier.STM)
                stats["mtm_optimized"] += 1
                stats["total_migrations"] += 1
        
        # Optimize LTM (demote low-value memories)
        ltm_entries = self.tier_manager.get_entries_by_tier(MemoryTier.LTM)
        for memory_id, entry in ltm_entries.items():
            if entry.hybrid_score <= self.migration_thresholds[MemoryTier.LTM]["demote"]:
                self._perform_migration(entry, MemoryTier.LTM, MemoryTier.MTM)
                stats["ltm_optimized"] += 1
                stats["total_migrations"] += 1
        
        self.logger.info(f"Memory optimization completed: {stats}")
        return stats
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """Get update statistics and metrics."""
        if not self.update_metrics:
            return {"total_updates": 0, "migrations": 0}
        
        migrations = [m for m in self.update_metrics if m.tier_from != m.tier_to]
        
        return {
            "total_updates": len(self.update_metrics),
            "migrations": len(migrations),
            "avg_score_change": np.mean([m.score_after - m.score_before for m in self.update_metrics]),
            "migration_rate": len(migrations) / len(self.update_metrics) if self.update_metrics else 0,
            "avg_decay_factor": np.mean([m.decay_factor for m in self.update_metrics]),
            "recent_migrations": [
                {
                    "memory_id": m.memory_id,
                    "from_tier": m.tier_from.value,
                    "to_tier": m.tier_to.value,
                    "reason": m.migration_reason,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in migrations[-10:]  # Last 10 migrations
            ]
        }
    
    def _update_scores_with_decay(self, entry: MemoryEntry, decay_factor: float) -> None:
        """Update memory scores with decay factor."""
        # Update hybrid score with decay
        entry.hybrid_score *= decay_factor
        
        # Ensure score stays within bounds
        entry.hybrid_score = max(0.0, min(1.0, entry.hybrid_score))
    
    def _determine_tier_migration(self, entry: MemoryEntry) -> MemoryTier:
        """Determine if memory should migrate to a different tier."""
        current_tier = entry.tier
        score = entry.hybrid_score
        
        if current_tier == MemoryTier.STM:
            if score >= self.migration_thresholds[MemoryTier.STM]["promote"]:
                return MemoryTier.MTM
            elif score <= self.migration_thresholds[MemoryTier.STM]["demote"]:
                return MemoryTier.STM  # Will be deleted by cleanup
        
        elif current_tier == MemoryTier.MTM:
            if score >= self.migration_thresholds[MemoryTier.MTM]["promote"]:
                return MemoryTier.LTM
            elif score <= self.migration_thresholds[MemoryTier.MTM]["demote"]:
                return MemoryTier.STM
        
        elif current_tier == MemoryTier.LTM:
            if score <= self.migration_thresholds[MemoryTier.LTM]["demote"]:
                return MemoryTier.MTM
        
        return current_tier  # No migration
    
    def _perform_migration(self, entry: MemoryEntry, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Perform memory migration between tiers."""
        # Update FIFO buffers
        self._update_fifo_buffers(entry.memory_id, from_tier, to_tier)
        
        # Update entry tier
        entry.tier = to_tier
        
        # Let tier manager handle the actual migration
        self.tier_manager._migrate_entry(entry, from_tier, to_tier)
    
    def _update_fifo_buffers(self, memory_id: str, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Update FIFO buffers during migration."""
        # Remove from source tier buffer
        if from_tier == MemoryTier.STM:
            self.stm_buffer.remove(memory_id)
        elif from_tier == MemoryTier.MTM:
            self.mtm_buffer.remove(memory_id)
        elif from_tier == MemoryTier.LTM:
            self.ltm_buffer.remove(memory_id)
        
        # Add to destination tier buffer
        if to_tier == MemoryTier.STM:
            evicted = self.stm_buffer.add(memory_id)
            if evicted:
                self.logger.debug(f"FIFO eviction from STM: {evicted}")
        elif to_tier == MemoryTier.MTM:
            evicted = self.mtm_buffer.add(memory_id)
            if evicted:
                self.logger.debug(f"FIFO eviction from MTM: {evicted}")
        elif to_tier == MemoryTier.LTM:
            evicted = self.ltm_buffer.add(memory_id)
            if evicted:
                self.logger.debug(f"FIFO eviction from LTM: {evicted}")
    
    def _get_migration_reason(self, entry: MemoryEntry, from_tier: MemoryTier, to_tier: MemoryTier) -> str:
        """Get human-readable reason for migration."""
        if from_tier == to_tier:
            return "no_migration"
        
        score = entry.hybrid_score
        
        if from_tier == MemoryTier.STM and to_tier == MemoryTier.MTM:
            return f"promoted_to_mtm_score_{score:.3f}"
        elif from_tier == MemoryTier.MTM and to_tier == MemoryTier.LTM:
            return f"promoted_to_ltm_score_{score:.3f}"
        elif from_tier == MemoryTier.MTM and to_tier == MemoryTier.STM:
            return f"demoted_to_stm_score_{score:.3f}"
        elif from_tier == MemoryTier.LTM and to_tier == MemoryTier.MTM:
            return f"demoted_to_mtm_score_{score:.3f}"
        else:
            return f"migration_{from_tier.value}_to_{to_tier.value}"
    
    def _store_metrics(self, metrics: UpdateMetrics) -> None:
        """Store update metrics with size limit."""
        self.update_metrics.append(metrics)
        
        # Trim if too many metrics
        if len(self.update_metrics) > self.max_metrics:
            self.update_metrics = self.update_metrics[-self.max_metrics:] 