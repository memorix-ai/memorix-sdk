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

#!/usr/bin/env python3
"""
Hierarchical Memory System Demo

This example demonstrates the enhanced Memorix SDK with:
- Hierarchical memory model (STM → MTM → LTM)
- Configurable decay and scoring strategies
- Declarative and scoped recall operations
- Comprehensive tracing and timeline tracking
- YAML-based configuration
"""

import os
import time
from datetime import datetime, timedelta
from typing import List

from memorix import (
    MemoryAPI, ConfigManager, MemoryTier, DecayStrategy, ScoringStrategy,
    RecallScope, TimeRange, DeclarativeQuery, FilterCondition, FilterOperator,
    OperationType, TraceLevel, FileHook
)


def setup_config() -> ConfigManager:
    """Setup configuration for the demo."""
    # Create a custom configuration
    config = ConfigManager()
    
    # Configure hierarchical memory
    config.set("memory.stm_capacity", 50)
    config.set("memory.mtm_capacity", 200)
    config.set("memory.ltm_capacity", 1000)
    config.set("memory.default_decay_strategy", "exponential")
    config.set("memory.default_decay_rate", 0.1)
    config.set("memory.default_scoring_strategy", "hybrid")
    
    # Configure logging and tracing
    config.set("logging.level", "INFO")
    config.set("logging.enable_tracing", True)
    config.set("logging.trace_memory_operations", True)
    config.set("logging.trace_tier_migrations", True)
    
    return config


def demo_basic_operations(memory_api: MemoryAPI):
    """Demonstrate basic memory operations."""
    print("\n=== Basic Memory Operations ===")
    
    # Store memories with different importance scores
    memory_ids = []
    
    # High importance memory (likely to go to LTM)
    memory_id = memory_api.store(
        content="The user's name is Alice and they prefer chocolate ice cream.",
        metadata={"tags": ["user_info", "preferences"], "agent_id": "assistant_1"},
        importance_score=0.9,
        decay_strategy=DecayStrategy.LINEAR,
        scoring_strategy=ScoringStrategy.IMPORTANCE
    )
    memory_ids.append(memory_id)
    print(f"Stored high-importance memory: {memory_id}")
    
    # Medium importance memory (likely to go to MTM)
    memory_id = memory_api.store(
        content="The weather today is sunny with a temperature of 75°F.",
        metadata={"tags": ["weather", "current"], "agent_id": "weather_bot"},
        importance_score=0.6,
        decay_strategy=DecayStrategy.EXPONENTIAL,
        scoring_strategy=ScoringStrategy.RECENCY
    )
    memory_ids.append(memory_id)
    print(f"Stored medium-importance memory: {memory_id}")
    
    # Low importance memory (likely to stay in STM)
    memory_id = memory_api.store(
        content="The user clicked the submit button.",
        metadata={"tags": ["ui_event", "temporary"], "agent_id": "ui_tracker"},
        importance_score=0.2,
        decay_strategy=DecayStrategy.STEP,
        scoring_strategy=ScoringStrategy.FREQUENCY
    )
    memory_ids.append(memory_id)
    print(f"Stored low-importance memory: {memory_id}")
    
    return memory_ids


def demo_hierarchical_behavior(memory_api: MemoryAPI, memory_ids: List[str]):
    """Demonstrate hierarchical memory behavior."""
    print("\n=== Hierarchical Memory Behavior ===")
    
    # Access memories multiple times to trigger tier migrations
    for i, memory_id in enumerate(memory_ids):
        print(f"\nAccessing memory {memory_id} multiple times...")
        
        for access_count in range(5):
            memory_data = memory_api.access(memory_id)
            if memory_data:
                print(f"  Access {access_count + 1}: Tier={memory_data['tier']}, "
                      f"Score={memory_data['hybrid_score']:.3f}, "
                      f"Access Count={memory_data['access_count']}")
            time.sleep(0.1)  # Small delay to simulate real usage
    
    # Show memory statistics
    stats = memory_api.get_memory_statistics()
    print(f"\nMemory Statistics:")
    print(f"  STM: {stats['hierarchical_memory']['stm']['count']} entries")
    print(f"  MTM: {stats['hierarchical_memory']['mtm']['count']} entries")
    print(f"  LTM: {stats['hierarchical_memory']['ltm']['count']} entries")


def demo_declarative_recall(memory_api: MemoryAPI):
    """Demonstrate declarative recall operations."""
    print("\n=== Declarative Recall Operations ===")
    
    # Store some additional memories for recall testing
    memory_api.store(
        content="The user asked about Python programming best practices.",
        metadata={"tags": ["programming", "python", "help"], "agent_id": "code_assistant"},
        importance_score=0.8
    )
    
    memory_api.store(
        content="The user mentioned they are working on a machine learning project.",
        metadata={"tags": ["ml", "project", "work"], "agent_id": "code_assistant"},
        importance_score=0.7
    )
    
    memory_api.store(
        content="The user's favorite programming language is Python.",
        metadata={"tags": ["preferences", "programming"], "agent_id": "assistant_1"},
        importance_score=0.6
    )
    
    # 1. Recall by tags
    print("\n1. Recall by tags (programming):")
    results = memory_api.recall_by_tags(
        tags=["programming"], 
        scope=RecallScope.ALL, 
        limit=5
    )
    for result in results:
        print(f"  - {result.content[:50]}... (Tier: {result.tier.value})")
    
    # 2. Recall by agent
    print("\n2. Recall by agent (code_assistant):")
    results = memory_api.recall_by_agent(
        agent_ids=["code_assistant"], 
        scope=RecallScope.ALL, 
        limit=5
    )
    for result in results:
        print(f"  - {result.content[:50]}... (Tier: {result.tier.value})")
    
    # 3. Recall by time range
    print("\n3. Recall by time range (last hour):")
    time_range = TimeRange(
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now()
    )
    results = memory_api.recall_by_time_range(time_range, scope=RecallScope.ALL, limit=5)
    for result in results:
        print(f"  - {result.content[:50]}... (Created: {result.created_at})")
    
    # 4. Declarative query with complex filters
    print("\n4. Complex declarative query:")
    query = DeclarativeQuery(
        query_text="user preferences",
        scope=RecallScope.ALL,
        limit=3,
        tags=["preferences"],
        min_importance=0.5,
        sort_by="importance",
        sort_order="desc"
    )
    results = memory_api.recall(query)
    for result in results:
        print(f"  - {result.content[:50]}... (Importance: {result.importance_score:.2f})")


def demo_tier_scoped_recall(memory_api: MemoryAPI):
    """Demonstrate tier-scoped recall operations."""
    print("\n=== Tier-Scoped Recall Operations ===")
    
    # Recall from specific tiers
    scopes = [
        (RecallScope.STM_ONLY, "Short-Term Memory Only"),
        (RecallScope.MTM_ONLY, "Medium-Term Memory Only"),
        (RecallScope.LTM_ONLY, "Long-Term Memory Only"),
        (RecallScope.STM_MTM, "STM + MTM"),
        (RecallScope.MTM_LTM, "MTM + LTM"),
        (RecallScope.ALL, "All Tiers")
    ]
    
    for scope, description in scopes:
        print(f"\n{description}:")
        results = memory_api.retrieve("user", top_k=3, scope=scope)
        print(f"  Found {len(results)} results")
        for result in results:
            print(f"    - {result['content'][:40]}... (Tier: {result['tier']})")


def demo_tracing_and_timeline(memory_api: MemoryAPI):
    """Demonstrate tracing and timeline features."""
    print("\n=== Tracing and Timeline Features ===")
    
    # Add a file hook for timeline export
    file_hook = FileHook("timeline_demo.json")
    memory_api.add_timeline_hook(file_hook)
    
    # Perform some operations to generate trace events
    memory_id = memory_api.store(
        content="This is a test memory for tracing demonstration.",
        metadata={"tags": ["test", "tracing"], "agent_id": "demo_agent"},
        importance_score=0.5
    )
    
    # Access the memory
    memory_api.access(memory_id)
    
    # Update the memory
    memory_api.update(memory_id, "This is an updated test memory for tracing demonstration.")
    
    # Get trace events
    events = memory_api.get_trace_events(limit=10)
    print(f"\nRecent trace events ({len(events)}):")
    for event in events:
        print(f"  [{event.operation_type.value}] {event.message}")
    
    # Get timeline entries
    entries = memory_api.get_timeline_entries(limit=5)
    print(f"\nRecent timeline entries ({len(entries)}):")
    for entry in entries:
        print(f"  [{entry.operation.value}] {entry.memory_id} in {entry.tier}")
    
    # Export timeline
    memory_api.export_timeline("timeline_export.json")
    print(f"\nTimeline exported to timeline_export.json")
    
    # Show tracing statistics
    stats = memory_api.get_memory_statistics()
    tracing_stats = stats['tracing']
    print(f"\nTracing Statistics:")
    print(f"  Total events: {tracing_stats['total_events']}")
    print(f"  Total timeline entries: {tracing_stats['total_timeline_entries']}")
    print(f"  Operation counts: {tracing_stats['operation_counts']}")


def demo_cleanup_and_maintenance(memory_api: MemoryAPI):
    """Demonstrate cleanup and maintenance operations."""
    print("\n=== Cleanup and Maintenance ===")
    
    # Store some temporary memories that will decay quickly
    for i in range(10):
        memory_api.store(
            content=f"Temporary memory {i} that will decay quickly.",
            metadata={"tags": ["temporary"], "agent_id": "cleanup_demo"},
            importance_score=0.1,
            decay_strategy=DecayStrategy.EXPONENTIAL,
            decay_rate=0.5  # High decay rate
        )
    
    print(f"Stored 10 temporary memories")
    
    # Show initial statistics
    stats = memory_api.get_memory_statistics()
    total_initial = stats['hierarchical_memory']['total_entries']
    print(f"Initial total memories: {total_initial}")
    
    # Wait a bit for some decay to occur
    print("Waiting for decay to occur...")
    time.sleep(2)
    
    # Perform cleanup
    cleaned_count = memory_api.cleanup_expired_memories()
    print(f"Cleaned up {cleaned_count} expired memories")
    
    # Show final statistics
    stats = memory_api.get_memory_statistics()
    total_final = stats['hierarchical_memory']['total_entries']
    print(f"Final total memories: {total_final}")


def main():
    """Main demo function."""
    print("Memorix SDK - Hierarchical Memory System Demo")
    print("=" * 50)
    
    # Setup configuration
    config = setup_config()
    
    # Initialize memory API
    memory_api = MemoryAPI(config)
    
    try:
        # Run demos
        memory_ids = demo_basic_operations(memory_api)
        demo_hierarchical_behavior(memory_api, memory_ids)
        demo_declarative_recall(memory_api)
        demo_tier_scoped_recall(memory_api)
        demo_tracing_and_timeline(memory_api)
        demo_cleanup_and_maintenance(memory_api)
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Check the generated files:")
        print("  - timeline_demo.json: Timeline data from file hook")
        print("  - timeline_export.json: Exported timeline data")
        print("  - memorix_metadata.db: SQLite metadata store")
        print("  - memorix_index/: FAISS vector store")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 