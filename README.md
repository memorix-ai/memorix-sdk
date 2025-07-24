# Memorix SDK - Hierarchical Memory System for AI Agents

A comprehensive memory layer that implements a three-tier hierarchical model inspired by human memory systems. Memorix provides configurable decay strategies, declarative recall operations, and comprehensive tracing for AI applications.

## 🧠 Hierarchical Memory Model

Memorix implements a three-tier memory hierarchy:

- **Short-Term Memory (STM)**: Fast access, limited capacity (default: 100 entries), rapid decay
- **Medium-Term Memory (MTM)**: Intermediate storage (default: 1,000 entries), moderate decay  
- **Long-Term Memory (LTM)**: Persistent storage (default: 100,000 entries), slow decay, high capacity

Memories automatically migrate between tiers based on importance, access frequency, and recency.

## ✨ Key Features

### 🎯 Hierarchical Memory Management
- **Automatic tier migration** based on scoring algorithms
- **Configurable capacities** for each memory tier
- **Intelligent eviction** policies to manage memory limits
- **Tier-specific decay rates** and cleanup thresholds

### 🔄 Configurable Decay & Scoring Strategies
- **Decay Strategies**: Exponential, Linear, Step, None
- **Scoring Strategies**: Frequency, Recency, Importance, Hybrid
- **Customizable weights** for hybrid scoring
- **Per-memory configuration** of decay and scoring

### 🔍 Declarative & Scoped Recall
- **Tag-based filtering** for semantic organization
- **Time-range queries** for temporal filtering
- **Agent-based filtering** for multi-agent systems
- **Tier-scoped recall** (STM only, MTM only, LTM only, etc.)
- **Complex declarative queries** with multiple filters

### 📊 Comprehensive Tracing & Timeline
- **Operation tracing** for all memory operations
- **Tier migration tracking** with reasons and metrics
- **Performance monitoring** with detailed statistics
- **Timeline hooks** for external monitoring systems
- **Export capabilities** for analysis and debugging

### ⚙️ YAML-Based Configuration
- **Hierarchical configuration** with nested settings
- **Environment-specific** configurations
- **Agent-specific** memory policies
- **Validation and merging** capabilities

## 🚀 Quick Start

### Installation

```bash
pip install memorix-ai
```

### Basic Usage

```python
from memorix import MemoryAPI, ConfigManager, DecayStrategy, ScoringStrategy

# Initialize with configuration
config = ConfigManager("config.yaml")
memory_api = MemoryAPI(config)

# Store a memory with custom settings
memory_id = memory_api.store(
    content="The user's name is Alice and they prefer chocolate ice cream.",
    metadata={"tags": ["user_info", "preferences"], "agent_id": "assistant_1"},
    importance_score=0.9,
    decay_strategy=DecayStrategy.LINEAR,
    scoring_strategy=ScoringStrategy.IMPORTANCE
)

# Retrieve similar memories
results = memory_api.retrieve("user preferences", top_k=5)
```

### Hierarchical Memory Operations

```python
from memorix import RecallScope, TimeRange, DeclarativeQuery

# Access a memory (updates access count and may trigger tier migration)
memory_data = memory_api.access(memory_id)

# Recall from specific tiers
stm_results = memory_api.retrieve("query", scope=RecallScope.STM_ONLY)
ltm_results = memory_api.retrieve("query", scope=RecallScope.LTM_ONLY)

# Declarative recall with filters
query = DeclarativeQuery(
    query_text="user preferences",
    scope=RecallScope.ALL,
    tags=["preferences"],
    min_importance=0.5,
    sort_by="importance"
)
results = memory_api.recall(query)
```

### Advanced Recall Operations

```python
from memorix import RecallScope, TimeRange
from datetime import datetime, timedelta

# Recall by tags
tag_results = memory_api.recall_by_tags(
    tags=["programming", "python"], 
    scope=RecallScope.ALL
)

# Recall by time range
time_range = TimeRange(
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now()
)
recent_results = memory_api.recall_by_time_range(time_range)

# Recall by agent
agent_results = memory_api.recall_by_agent(
    agent_ids=["code_assistant", "help_bot"]
)
```

### Tracing and Monitoring

```python
from memorix import FileHook, OperationType

# Add timeline hook for external monitoring
file_hook = FileHook("timeline.json")
memory_api.add_timeline_hook(file_hook)

# Get memory statistics
stats = memory_api.get_memory_statistics()
print(f"STM: {stats['hierarchical_memory']['stm']['count']} entries")
print(f"MTM: {stats['hierarchical_memory']['mtm']['count']} entries")
print(f"LTM: {stats['hierarchical_memory']['ltm']['count']} entries")

# Get trace events
events = memory_api.get_trace_events(operation_type=OperationType.MIGRATE)
for event in events:
    print(f"Migration: {event.memory_id} from {event.tier_from} to {event.tier_to}")

# Export timeline
memory_api.export_timeline("timeline_export.json")
```

## 📋 Configuration

### YAML Configuration File

```yaml
# Hierarchical Memory Configuration
memory:
  stm_capacity: 100          # Short-term memory capacity
  mtm_capacity: 1000         # Medium-term memory capacity  
  ltm_capacity: 100000       # Long-term memory capacity
  
  default_decay_strategy: "exponential"  # Options: exponential, linear, step, none
  default_decay_rate: 0.1                # Decay rate (0.0 to 1.0)
  default_scoring_strategy: "hybrid"     # Options: frequency, recency, importance, hybrid
  
  hybrid_weights:
    frequency: 0.3           # Weight for frequency-based scoring
    recency: 0.4             # Weight for recency-based scoring
    importance: 0.3          # Weight for importance-based scoring
  
  stm_cleanup_threshold: 0.1  # STM cleanup threshold (most aggressive)
  mtm_cleanup_threshold: 0.3  # MTM cleanup threshold (moderate)
  ltm_cleanup_threshold: 0.5  # LTM cleanup threshold (conservative)

# Vector Store Configuration
vector_store:
  type: "faiss"              # Options: faiss, qdrant, chroma
  index_path: "./memorix_index"
  dimension: 1536            # Embedding dimension
  similarity_threshold: 0.7   # Minimum similarity threshold
  max_results: 100           # Maximum number of results

# Embedder Configuration
embedder:
  type: "openai"             # Options: openai, gemini, sentence-transformers
  model: "text-embedding-ada-002"
  api_key: null              # Set your API key here or use environment variable
  batch_size: 32             # Batch size for embeddings
  max_retries: 3             # Maximum retry attempts
  timeout: 30                # Request timeout in seconds

# Metadata Store Configuration
metadata_store:
  type: "sqlite"             # Options: sqlite, memory, json
  database_path: "./memorix_metadata.db"
  auto_backup: true          # Enable automatic backups
  backup_interval: 3600      # Backup interval in seconds

# Logging and Tracing Configuration
logging:
  level: "INFO"              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  enable_tracing: true       # Enable operation tracing
  trace_memory_operations: true    # Trace memory operations
  trace_tier_migrations: true      # Trace tier migrations
  trace_decay_calculations: false  # Trace decay calculations (verbose)
  
  timeline_hooks:
    - "store"
    - "retrieve" 
    - "update"
    - "delete"
    - "migrate"
    - "recall"
  
  max_events: 10000          # Maximum number of trace events to store
  max_timeline_entries: 10000 # Maximum number of timeline entries to store

# Global Settings
max_memories: 10000          # Maximum total memories across all tiers
enable_hierarchical_memory: true  # Enable hierarchical memory system
auto_cleanup: true           # Enable automatic cleanup of expired memories
cleanup_interval: 300        # Cleanup interval in seconds
```

### Programmatic Configuration

```python
from memorix import ConfigManager

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

# Save configuration
config.save("my_config.yaml")
```

## 🔧 Decay Strategies

### Exponential Decay
```python
from memorix import DecayStrategy

# Exponential decay: e^(-decay_rate * time_delta)
memory_id = memory_api.store(
    content="Memory with exponential decay",
    decay_strategy=DecayStrategy.EXPONENTIAL,
    decay_rate=0.1
)
```

### Linear Decay
```python
# Linear decay: 1 - (decay_rate * time_delta)
memory_id = memory_api.store(
    content="Memory with linear decay",
    decay_strategy=DecayStrategy.LINEAR,
    decay_rate=0.05
)
```

### Step Decay
```python
# Step decay based on time intervals
memory_id = memory_api.store(
    content="Memory with step decay",
    decay_strategy=DecayStrategy.STEP,
    decay_rate=0.2
)
```

### No Decay
```python
# No decay - memory stays at full strength
memory_id = memory_api.store(
    content="Permanent memory",
    decay_strategy=DecayStrategy.NONE
)
```

## 📊 Scoring Strategies

### Frequency Scoring
```python
from memorix import ScoringStrategy

# Score based on access frequency
memory_id = memory_api.store(
    content="Frequently accessed memory",
    scoring_strategy=ScoringStrategy.FREQUENCY
)
```

### Recency Scoring
```python
# Score based on recency of access
memory_id = memory_api.store(
    content="Recently accessed memory",
    scoring_strategy=ScoringStrategy.RECENCY
)
```

### Importance Scoring
```python
# Score based on importance
memory_id = memory_api.store(
    content="Important memory",
    scoring_strategy=ScoringStrategy.IMPORTANCE,
    importance_score=0.9
)
```

### Hybrid Scoring
```python
# Hybrid scoring combining frequency, recency, and importance
memory_id = memory_api.store(
    content="Hybrid scored memory",
    scoring_strategy=ScoringStrategy.HYBRID,
    importance_score=0.7
)
```

## 🔍 Advanced Recall Examples

### Complex Declarative Queries

```python
from memorix import DeclarativeQuery, FilterCondition, FilterOperator, RecallScope

# Complex query with multiple filters
query = DeclarativeQuery(
    query_text="machine learning project",
    scope=RecallScope.ALL,
    limit=10,
    tags=["ml", "project"],
    exclude_tags=["deprecated"],
    agent_ids=["code_assistant"],
    min_importance=0.6,
    min_similarity=0.7,
    sort_by="importance",
    sort_order="desc",
    filters=[
        FilterCondition("priority", FilterOperator.GREATER_THAN, 5),
        FilterCondition("status", FilterOperator.EQUALS, "active")
    ]
)

results = memory_api.recall(query)
```

### Time-Based Queries

```python
from datetime import datetime, timedelta

# Recent memories from last week
time_range = TimeRange(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

recent_results = memory_api.recall_by_time_range(
    time_range=time_range,
    scope=RecallScope.MTM_LTM,  # Only medium and long-term memories
    limit=20
)
```

### Tier-Specific Operations

```python
# List all memories in STM
stm_memories = memory_api.list_memories(tier=MemoryTier.STM)

# Get statistics for specific tier
stats = memory_api.get_memory_statistics()
stm_stats = stats['hierarchical_memory']['stm']
print(f"STM entries: {stm_stats['count']}")
print(f"Average score: {stm_stats['avg_score']:.3f}")
print(f"Average age: {stm_stats['avg_age']:.1f} seconds")
```

## 🛠️ Maintenance Operations

### Cleanup Expired Memories

```python
# Manual cleanup
cleaned_count = memory_api.cleanup_expired_memories()
print(f"Cleaned up {cleaned_count} expired memories")

# Automatic cleanup (configured in YAML)
# auto_cleanup: true
# cleanup_interval: 300  # Every 5 minutes
```

### Memory Statistics

```python
# Get comprehensive statistics
stats = memory_api.get_memory_statistics()

# Hierarchical memory stats
hierarchical_stats = stats['hierarchical_memory']
print(f"Total entries: {hierarchical_stats['total_entries']}")
print(f"STM: {hierarchical_stats['stm']['count']} entries")
print(f"MTM: {hierarchical_stats['mtm']['count']} entries")
print(f"LTM: {hierarchical_stats['ltm']['count']} entries")

# Tracing stats
tracing_stats = stats['tracing']
print(f"Total events: {tracing_stats['total_events']}")
print(f"Operation counts: {tracing_stats['operation_counts']}")

# Performance stats
if 'avg_duration_ms' in tracing_stats:
    print(f"Average operation duration: {tracing_stats['avg_duration_ms']:.2f}ms")
```

## 🔌 Timeline Hooks

### Custom Timeline Hook

```python
from memorix import TimelineHook, TraceEvent, TimelineEntry

class CustomHook(TimelineHook):
    def on_event(self, event: TraceEvent):
        # Handle trace events
        if event.operation_type == OperationType.MIGRATE:
            print(f"Memory {event.memory_id} migrated from {event.tier_from} to {event.tier_to}")
    
    def on_timeline_entry(self, entry: TimelineEntry):
        # Handle timeline entries
        print(f"Timeline: {entry.operation.value} for {entry.memory_id}")

# Add custom hook
custom_hook = CustomHook()
memory_api.add_timeline_hook(custom_hook)
```

### File-Based Timeline Export

```python
from memorix import FileHook

# Add file hook for persistent timeline
file_hook = FileHook("timeline.json", max_entries=10000)
memory_api.add_timeline_hook(file_hook)

# Export timeline data
memory_api.export_timeline("timeline_export.json")
```

## 📚 Examples

See the `examples/` directory for comprehensive examples:

- `hierarchical_memory_demo.py` - Complete demonstration of all features
- `basic_usage.py` - Simple usage examples
- `advanced_recall.py` - Advanced recall operations
- `tracing_demo.py` - Tracing and monitoring examples

## 🏗️ Architecture

### Core Components

1. **MemoryTierManager**: Manages the three-tier memory hierarchy
2. **RecallEngine**: Handles declarative and scoped recall operations
3. **TracingManager**: Provides comprehensive tracing and timeline tracking
4. **ConfigManager**: Manages YAML-based configuration
5. **VectorStore**: Handles vector storage and similarity search
6. **Embedder**: Manages text embeddings
7. **MetadataStore**: Stores metadata and hierarchical properties

### Memory Flow

1. **Store**: Memory is created and placed in appropriate tier based on importance
2. **Access**: Memory access updates scores and may trigger tier migration
3. **Decay**: Memories decay over time based on configured strategy
4. **Migration**: Memories move between tiers based on updated scores
5. **Cleanup**: Expired memories are automatically removed
6. **Recall**: Declarative queries retrieve memories with filtering and scoring

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://docs.memorix.ai)
- [Examples Repository](https://github.com/memorix-ai/memorix-examples)
- [Issues](https://github.com/memorix-ai/memorix-sdk/issues)
- [Discussions](https://github.com/memorix-ai/memorix-sdk/discussions) 