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
Hierarchical Memory Model Implementation

This module implements the three-tier memory hierarchy:
- Short-Term Memory (STM): Fast access, limited capacity, rapid decay
- Medium-Term Memory (MTM): Intermediate storage, moderate decay
- Long-Term Memory (LTM): Persistent storage, slow decay, high capacity
"""

import logging
import math
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field


class MemoryTier(Enum):
    """Memory tiers in the hierarchical model."""
    STM = "short_term"
    MTM = "medium_term"
    LTM = "long_term"


class DecayStrategy(Enum):
    """Available decay strategies."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


class ScoringStrategy(Enum):
    """Available scoring strategies."""
    FREQUENCY = "frequency"
    RECENCY = "recency"
    IMPORTANCE = "importance"
    HYBRID = "hybrid"


@dataclass
class MemoryEntry:
    """Represents a memory entry with hierarchical properties."""
    memory_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Hierarchical properties
    tier: MemoryTier = MemoryTier.STM
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    importance_score: float = 1.0
    
    # Decay properties
    decay_rate: float = 0.1
    decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    
    # Scoring properties
    scoring_strategy: ScoringStrategy = ScoringStrategy.HYBRID
    frequency_score: float = 0.0
    recency_score: float = 1.0
    hybrid_score: float = 1.0


class DecayCalculator(ABC):
    """Abstract base class for decay calculations."""
    
    @abstractmethod
    def calculate_decay(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate decay factor for a memory entry."""
        pass


class ExponentialDecay(DecayCalculator):
    """Exponential decay implementation."""
    
    def calculate_decay(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate exponential decay: e^(-decay_rate * time_delta)."""
        time_delta = (current_time - entry.last_accessed).total_seconds()
        return math.exp(-entry.decay_rate * time_delta)


class LinearDecay(DecayCalculator):
    """Linear decay implementation."""
    
    def calculate_decay(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate linear decay: 1 - (decay_rate * time_delta)."""
        time_delta = (current_time - entry.last_accessed).total_seconds()
        decay_factor = entry.decay_rate * time_delta
        return max(0.0, 1.0 - decay_factor)


class StepDecay(DecayCalculator):
    """Step decay implementation."""
    
    def calculate_decay(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate step decay based on time intervals."""
        time_delta = (current_time - entry.last_accessed).total_seconds()
        steps = int(time_delta / (1.0 / entry.decay_rate))
        return max(0.0, 1.0 - (steps * 0.1))


class NoDecay(DecayCalculator):
    """No decay implementation."""
    
    def calculate_decay(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Return 1.0 (no decay)."""
        return 1.0


class ScoreCalculator(ABC):
    """Abstract base class for score calculations."""
    
    @abstractmethod
    def calculate_score(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate score for a memory entry."""
        pass


class FrequencyScore(ScoreCalculator):
    """Frequency-based scoring."""
    
    def calculate_score(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate score based on access frequency."""
        time_alive = (current_time - entry.created_at).total_seconds()
        if time_alive > 0:
            return entry.access_count / time_alive
        return 0.0


class RecencyScore(ScoreCalculator):
    """Recency-based scoring."""
    
    def calculate_score(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate score based on recency of access."""
        time_since_access = (current_time - entry.last_accessed).total_seconds()
        return math.exp(-time_since_access / 3600)  # 1 hour half-life


class ImportanceScore(ScoreCalculator):
    """Importance-based scoring."""
    
    def calculate_score(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate score based on importance."""
        return entry.importance_score


class HybridScore(ScoreCalculator):
    """Hybrid scoring combining frequency, recency, and importance."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "frequency": 0.3,
            "recency": 0.4,
            "importance": 0.3
        }
        self.frequency_calc = FrequencyScore()
        self.recency_calc = RecencyScore()
        self.importance_calc = ImportanceScore()
    
    def calculate_score(self, entry: MemoryEntry, current_time: datetime) -> float:
        """Calculate hybrid score."""
        freq_score = self.frequency_calc.calculate_score(entry, current_time)
        recency_score = self.recency_calc.calculate_score(entry, current_time)
        importance_score = self.importance_calc.calculate_score(entry, current_time)
        
        # Normalize scores to 0-1 range
        freq_score = min(1.0, freq_score)
        recency_score = min(1.0, recency_score)
        importance_score = min(1.0, importance_score)
        
        return (
            self.weights["frequency"] * freq_score +
            self.weights["recency"] * recency_score +
            self.weights["importance"] * importance_score
        )


class MemoryTierManager:
    """Manages memory entries across different tiers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tier capacities
        self.stm_capacity = config.get("memory.stm_capacity", 100)
        self.mtm_capacity = config.get("memory.mtm_capacity", 1000)
        self.ltm_capacity = config.get("memory.ltm_capacity", 100000)
        
        # Decay calculators
        self.decay_calculators = {
            DecayStrategy.EXPONENTIAL: ExponentialDecay(),
            DecayStrategy.LINEAR: LinearDecay(),
            DecayStrategy.STEP: StepDecay(),
            DecayStrategy.NONE: NoDecay()
        }
        
        # Score calculators
        self.score_calculators = {
            ScoringStrategy.FREQUENCY: FrequencyScore(),
            ScoringStrategy.RECENCY: RecencyScore(),
            ScoringStrategy.IMPORTANCE: ImportanceScore(),
            ScoringStrategy.HYBRID: HybridScore()
        }
        
        # Memory storage
        self.stm_entries: Dict[str, MemoryEntry] = {}
        self.mtm_entries: Dict[str, MemoryEntry] = {}
        self.ltm_entries: Dict[str, MemoryEntry] = {}
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to the appropriate tier."""
        current_time = datetime.now()
        
        # Calculate initial scores
        self._update_scores(entry, current_time)
        
        # Determine initial tier based on importance
        if entry.importance_score >= 0.8:
            entry.tier = MemoryTier.LTM
        elif entry.importance_score >= 0.5:
            entry.tier = MemoryTier.MTM
        else:
            entry.tier = MemoryTier.STM
        
        # Add to appropriate tier
        self._add_to_tier(entry)
        
        self.logger.debug(f"Added memory {entry.memory_id} to {entry.tier.value}")
    
    def access_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Access a memory entry and update its properties."""
        current_time = datetime.now()
        
        # Find entry in any tier
        entry = self._find_entry(memory_id)
        if not entry:
            return None
        
        # Update access properties
        entry.access_count += 1
        entry.last_accessed = current_time
        
        # Recalculate scores
        self._update_scores(entry, current_time)
        
        # Check for tier promotion/demotion
        self._check_tier_migration(entry, current_time)
        
        self.logger.debug(f"Accessed memory {memory_id} in {entry.tier.value}")
        return entry
    
    def get_entries_by_tier(self, tier: MemoryTier) -> Dict[str, MemoryEntry]:
        """Get all entries in a specific tier."""
        if tier == MemoryTier.STM:
            return self.stm_entries.copy()
        elif tier == MemoryTier.MTM:
            return self.mtm_entries.copy()
        elif tier == MemoryTier.LTM:
            return self.ltm_entries.copy()
        else:
            raise ValueError(f"Unknown tier: {tier}")
    
    def cleanup_expired_entries(self) -> None:
        """Remove expired entries from all tiers."""
        current_time = datetime.now()
        
        # Clean STM (most aggressive)
        self._cleanup_tier(self.stm_entries, current_time, threshold=0.1)
        
        # Clean MTM (moderate)
        self._cleanup_tier(self.mtm_entries, current_time, threshold=0.3)
        
        # Clean LTM (conservative)
        self._cleanup_tier(self.ltm_entries, current_time, threshold=0.5)
    
    def _add_to_tier(self, entry: MemoryEntry) -> None:
        """Add entry to its designated tier."""
        if entry.tier == MemoryTier.STM:
            if len(self.stm_entries) >= self.stm_capacity:
                self._evict_from_stm()
            self.stm_entries[entry.memory_id] = entry
        elif entry.tier == MemoryTier.MTM:
            if len(self.mtm_entries) >= self.mtm_capacity:
                self._evict_from_mtm()
            self.mtm_entries[entry.memory_id] = entry
        elif entry.tier == MemoryTier.LTM:
            if len(self.ltm_entries) >= self.ltm_capacity:
                self._evict_from_ltm()
            self.ltm_entries[entry.memory_id] = entry
    
    def _find_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Find entry in any tier."""
        return (
            self.stm_entries.get(memory_id) or
            self.mtm_entries.get(memory_id) or
            self.ltm_entries.get(memory_id)
        )
    
    def _update_scores(self, entry: MemoryEntry, current_time: datetime) -> None:
        """Update all scores for an entry."""
        # Calculate decay
        decay_calc = self.decay_calculators[entry.decay_strategy]
        decay_factor = decay_calc.calculate_decay(entry, current_time)
        
        # Calculate scores
        score_calc = self.score_calculators[entry.scoring_strategy]
        entry.hybrid_score = score_calc.calculate_score(entry, current_time)
        
        # Apply decay to scores
        entry.hybrid_score *= decay_factor
    
    def _check_tier_migration(self, entry: MemoryEntry, current_time: datetime) -> None:
        """Check if entry should migrate between tiers."""
        old_tier = entry.tier
        
        # Determine new tier based on updated scores
        if entry.hybrid_score >= 0.8 and entry.tier != MemoryTier.LTM:
            entry.tier = MemoryTier.LTM
        elif entry.hybrid_score >= 0.5 and entry.tier == MemoryTier.STM:
            entry.tier = MemoryTier.MTM
        elif entry.hybrid_score < 0.3 and entry.tier == MemoryTier.LTM:
            entry.tier = MemoryTier.MTM
        elif entry.hybrid_score < 0.1 and entry.tier == MemoryTier.MTM:
            entry.tier = MemoryTier.STM
        
        # Migrate if tier changed
        if entry.tier != old_tier:
            self._migrate_entry(entry, old_tier, entry.tier)
    
    def _migrate_entry(self, entry: MemoryEntry, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Migrate entry between tiers."""
        # Remove from old tier
        if from_tier == MemoryTier.STM:
            self.stm_entries.pop(entry.memory_id, None)
        elif from_tier == MemoryTier.MTM:
            self.mtm_entries.pop(entry.memory_id, None)
        elif from_tier == MemoryTier.LTM:
            self.ltm_entries.pop(entry.memory_id, None)
        
        # Add to new tier
        self._add_to_tier(entry)
        
        self.logger.info(f"Migrated memory {entry.memory_id} from {from_tier.value} to {to_tier.value}")
    
    def _evict_from_stm(self) -> None:
        """Evict least valuable entry from STM."""
        if not self.stm_entries:
            return
        
        # Find entry with lowest score
        worst_entry = min(self.stm_entries.values(), key=lambda e: e.hybrid_score)
        del self.stm_entries[worst_entry.memory_id]
        
        self.logger.debug(f"Evicted memory {worst_entry.memory_id} from STM")
    
    def _evict_from_mtm(self) -> None:
        """Evict least valuable entry from MTM."""
        if not self.mtm_entries:
            return
        
        # Find entry with lowest score
        worst_entry = min(self.mtm_entries.values(), key=lambda e: e.hybrid_score)
        del self.mtm_entries[worst_entry.memory_id]
        
        self.logger.debug(f"Evicted memory {worst_entry.memory_id} from MTM")
    
    def _evict_from_ltm(self) -> None:
        """Evict least valuable entry from LTM."""
        if not self.ltm_entries:
            return
        
        # Find entry with lowest score
        worst_entry = min(self.ltm_entries.values(), key=lambda e: e.hybrid_score)
        del self.ltm_entries[worst_entry.memory_id]
        
        self.logger.debug(f"Evicted memory {worst_entry.memory_id} from LTM")
    
    def _cleanup_tier(self, entries: Dict[str, MemoryEntry], current_time: datetime, threshold: float) -> None:
        """Clean up expired entries from a tier."""
        expired_ids = []
        
        for memory_id, entry in entries.items():
            decay_calc = self.decay_calculators[entry.decay_strategy]
            decay_factor = decay_calc.calculate_decay(entry, current_time)
            
            if decay_factor < threshold:
                expired_ids.append(memory_id)
        
        for memory_id in expired_ids:
            del entries[memory_id]
            self.logger.debug(f"Cleaned up expired memory {memory_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory hierarchy statistics."""
        current_time = datetime.now()
        
        def calculate_tier_stats(entries: Dict[str, MemoryEntry]) -> Dict[str, Any]:
            if not entries:
                return {"count": 0, "avg_score": 0.0, "avg_age": 0.0}
            
            scores = [e.hybrid_score for e in entries.values()]
            ages = [(current_time - e.created_at).total_seconds() for e in entries.values()]
            
            return {
                "count": len(entries),
                "avg_score": sum(scores) / len(scores),
                "avg_age": sum(ages) / len(ages)
            }
        
        return {
            "stm": calculate_tier_stats(self.stm_entries),
            "mtm": calculate_tier_stats(self.mtm_entries),
            "ltm": calculate_tier_stats(self.ltm_entries),
            "total_entries": len(self.stm_entries) + len(self.mtm_entries) + len(self.ltm_entries)
        } 