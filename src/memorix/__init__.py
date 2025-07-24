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

Memorix SDK - Hierarchical Memory System for AI Agents

A comprehensive memory layer that implements a three-tier hierarchical model:
- Short-Term Memory (STM): Fast access, limited capacity, rapid decay
- Medium-Term Memory (MTM): Intermediate storage, moderate decay  
- Long-Term Memory (LTM): Persistent storage, slow decay, high capacity

Features:
- Configurable decay and scoring strategies
- Declarative and scoped recall operations
- Comprehensive tracing and timeline tracking
- YAML-based configuration
- Support for multiple vector stores and embedders
"""

from .memory_api import MemoryAPI
from .config import ConfigManager, Config
from .embedder import Embedder
from .metadata_store import MetadataStore
from .vector_store import VectorStore

# Hierarchical memory components
from .memory_hierarchy import (
    MemoryTierManager,
    MemoryEntry,
    MemoryTier,
    DecayStrategy,
    ScoringStrategy,
    DecayCalculator,
    ExponentialDecay,
    LinearDecay,
    StepDecay,
    NoDecay,
    ScoreCalculator,
    FrequencyScore,
    RecencyScore,
    ImportanceScore,
    HybridScore
)

# Enhanced recall system
from .recall import (
    RecallEngine,
    DeclarativeQuery,
    RecallResult,
    RecallScope,
    TimeRange,
    FilterCondition,
    FilterOperator,
    FilterEvaluator,
    MetadataFilterEvaluator
)

# Tracing and timeline system
from .tracing import (
    TracingManager,
    TraceEvent,
    TimelineEntry,
    TimelineHook,
    LoggingHook,
    FileHook,
    MetricsHook,
    TraceLevel,
    OperationType
)

# Memory update system
from .memory.update import (
    MemoryUpdateManager,
    UpdateMetrics,
    FIFOBuffer,
    AdaptiveDecayCalculator,
    FrequencyBasedDecay,
    ContextAwareDecay
)

# Storage plugin interfaces
from .memory.storage.base import (
    StoragePlugin,
    VectorStoreInterface,
    MetadataStoreInterface,
    StorageConfig,
    SearchResult,
    StorageMetrics,
    StoragePluginRegistry,
    plugin_registry
)

# Privacy and consent management
from .privacy.consent import (
    ConsentStatus,
    DataCategory,
    ProcessingPurpose,
    ConsentRecord,
    PrivacyPolicy,
    PrivacyAuditEvent,
    ConsentValidator,
    GDPRConsentValidator,
    PrivacyPolicyLoader
)

from .privacy.erase import (
    ErasureMethod,
    ErasureStatus,
    ErasureRequest,
    ErasureOperation,
    ErasureVerification,
    DataEraser,
    SecureDataEraser
)

# Audit and logging system
from .audit.log import (
    AuditLevel,
    AuditEventType,
    AuditCategory,
    AuditEvent,
    AuditFilter,
    AuditStatistics,
    AuditLogger,
    SQLiteAuditLogger,
    AuditManager
)

__version__ = "0.3.0"
__author__ = "Memorix AI"
__email__ = "founder@memorix.ai"

# Main exports for backward compatibility
__all__ = [
    # Core API
    "MemoryAPI",
    "Config",
    "ConfigManager",
    "Embedder", 
    "MetadataStore",
    "VectorStore",
    
    # Hierarchical memory
    "MemoryTierManager",
    "MemoryEntry", 
    "MemoryTier",
    "DecayStrategy",
    "ScoringStrategy",
    "DecayCalculator",
    "ExponentialDecay",
    "LinearDecay", 
    "StepDecay",
    "NoDecay",
    "ScoreCalculator",
    "FrequencyScore",
    "RecencyScore",
    "ImportanceScore", 
    "HybridScore",
    
    # Enhanced recall
    "RecallEngine",
    "DeclarativeQuery",
    "RecallResult",
    "RecallScope",
    "TimeRange",
    "FilterCondition",
    "FilterOperator",
    "FilterEvaluator",
    "MetadataFilterEvaluator",
    
    # Tracing and timeline
    "TracingManager",
    "TraceEvent",
    "TimelineEntry", 
    "TimelineHook",
    "LoggingHook",
    "FileHook",
    "MetricsHook",
    "TraceLevel",
    "OperationType",
    
    # Memory update system
    "MemoryUpdateManager",
    "UpdateMetrics",
    "FIFOBuffer",
    "AdaptiveDecayCalculator",
    "FrequencyBasedDecay",
    "ContextAwareDecay",
    
    # Storage plugin interfaces
    "StoragePlugin",
    "VectorStoreInterface",
    "MetadataStoreInterface",
    "StorageConfig",
    "SearchResult",
    "StorageMetrics",
    "StoragePluginRegistry",
    "plugin_registry",
    
    # Privacy and consent management
    "ConsentStatus",
    "DataCategory",
    "ProcessingPurpose",
    "ConsentRecord",
    "PrivacyPolicy",
    "PrivacyAuditEvent",
    "ConsentValidator",
    "GDPRConsentValidator",
    "PrivacyPolicyLoader",
    "ErasureMethod",
    "ErasureStatus",
    "ErasureRequest",
    "ErasureOperation",
    "ErasureVerification",
    "DataEraser",
    "SecureDataEraser",
    
    # Audit and logging system
    "AuditLevel",
    "AuditEventType",
    "AuditCategory",
    "AuditEvent",
    "AuditFilter",
    "AuditStatistics",
    "AuditLogger",
    "SQLiteAuditLogger",
    "AuditManager"
]
