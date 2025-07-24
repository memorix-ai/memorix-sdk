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
Tracing and Timeline System for Memorix SDK

This module implements comprehensive tracing and timeline tracking for:
- Memory operations (store, retrieve, update, delete)
- Tier migrations
- Decay calculations
- Performance metrics
- Timeline hooks for external systems
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager

import numpy as np
from pydantic import BaseModel, Field


class TraceLevel(Enum):
    """Trace levels for different types of operations."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OperationType(Enum):
    """Types of memory operations."""
    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    MIGRATE = "migrate"
    DECAY = "decay"
    CLEANUP = "cleanup"
    RECALL = "recall"


@dataclass
class TraceEvent:
    """Represents a trace event."""
    event_id: str
    operation_type: OperationType
    timestamp: datetime
    level: TraceLevel
    message: str
    memory_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    tier_from: Optional[str] = None
    tier_to: Optional[str] = None
    similarity_score: Optional[float] = None
    importance_score: Optional[float] = None
    access_count: Optional[int] = None


@dataclass
class TimelineEntry:
    """Represents a timeline entry for memory operations."""
    entry_id: str
    memory_id: str
    operation: OperationType
    timestamp: datetime
    tier: str
    content_preview: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class TimelineHook(ABC):
    """Abstract base class for timeline hooks."""
    
    @abstractmethod
    def on_event(self, event: TraceEvent) -> None:
        """Handle a trace event."""
        pass
    
    @abstractmethod
    def on_timeline_entry(self, entry: TimelineEntry) -> None:
        """Handle a timeline entry."""
        pass


class LoggingHook(TimelineHook):
    """Logging-based timeline hook."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def on_event(self, event: TraceEvent) -> None:
        """Log trace event."""
        level = getattr(logging, event.level.value.upper())
        self.logger.log(level, f"[{event.operation_type.value}] {event.message}", extra=asdict(event))
    
    def on_timeline_entry(self, entry: TimelineEntry) -> None:
        """Log timeline entry."""
        self.logger.info(f"Timeline: {entry.operation.value} for {entry.memory_id} in {entry.tier}")


class FileHook(TimelineHook):
    """File-based timeline hook."""
    
    def __init__(self, file_path: str, max_entries: int = 10000):
        self.file_path = file_path
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = []
    
    def on_event(self, event: TraceEvent) -> None:
        """Write event to file."""
        event_dict = asdict(event)
        event_dict['timestamp'] = event_dict['timestamp'].isoformat()
        
        self.entries.append(event_dict)
        self._trim_entries()
        self._write_to_file()
    
    def on_timeline_entry(self, entry: TimelineEntry) -> None:
        """Write timeline entry to file."""
        entry_dict = asdict(entry)
        entry_dict['timestamp'] = entry_dict['timestamp'].isoformat()
        
        self.entries.append(entry_dict)
        self._trim_entries()
        self._write_to_file()
    
    def _trim_entries(self) -> None:
        """Trim entries to max_entries."""
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def _write_to_file(self) -> None:
        """Write entries to file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.entries, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to write to timeline file: {e}")


class MetricsHook(TimelineHook):
    """Metrics collection timeline hook."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'operation_durations': [],
            'similarity_scores': [],
            'importance_scores': [],
            'tier_migrations': [],
            'error_count': 0
        }
    
    def on_event(self, event: TraceEvent) -> None:
        """Collect metrics from event."""
        if event.duration_ms is not None:
            self.metrics['operation_durations'].append(event.duration_ms)
        
        if event.similarity_score is not None:
            self.metrics['similarity_scores'].append(event.similarity_score)
        
        if event.importance_score is not None:
            self.metrics['importance_scores'].append(event.importance_score)
        
        if event.tier_from and event.tier_to:
            self.metrics['tier_migrations'].append(time.time())
        
        if event.error:
            self.metrics['error_count'] += 1
    
    def on_timeline_entry(self, entry: TimelineEntry) -> None:
        """Collect metrics from timeline entry."""
        if entry.performance_metrics:
            for metric_name, value in entry.performance_metrics.items():
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = []
                self.metrics[metric_name].append(value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collected statistics."""
        stats = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                if isinstance(values, list):
                    stats[f"{metric_name}_count"] = len(values)
                    stats[f"{metric_name}_mean"] = np.mean(values)
                    stats[f"{metric_name}_std"] = np.std(values)
                    stats[f"{metric_name}_min"] = np.min(values)
                    stats[f"{metric_name}_max"] = np.max(values)
                else:
                    stats[metric_name] = values
        
        return stats


class TracingManager:
    """Manages tracing and timeline tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.enable_tracing = config.get("logging.enable_tracing", True)
        self.trace_memory_operations = config.get("logging.trace_memory_operations", True)
        self.trace_tier_migrations = config.get("logging.trace_tier_migrations", True)
        self.trace_decay_calculations = config.get("logging.trace_decay_calculations", False)
        
        # Timeline hooks
        self.timeline_hooks: List[TimelineHook] = []
        self._setup_default_hooks()
        
        # Event storage
        self.events: List[TraceEvent] = []
        self.timeline_entries: List[TimelineEntry] = []
        self.max_events = config.get("logging.max_events", 10000)
        self.max_timeline_entries = config.get("logging.max_timeline_entries", 10000)
    
    def _setup_default_hooks(self) -> None:
        """Setup default timeline hooks."""
        if self.enable_tracing:
            # Add logging hook
            self.add_hook(LoggingHook())
            
            # Add metrics hook
            self.add_hook(MetricsHook())
    
    def add_hook(self, hook: TimelineHook) -> None:
        """Add a timeline hook."""
        self.timeline_hooks.append(hook)
    
    def remove_hook(self, hook: TimelineHook) -> None:
        """Remove a timeline hook."""
        if hook in self.timeline_hooks:
            self.timeline_hooks.remove(hook)
    
    @contextmanager
    def trace_operation(self, operation_type: OperationType, memory_id: Optional[str] = None, **kwargs):
        """Context manager for tracing operations."""
        if not self.enable_tracing:
            yield
            return
        
        start_time = time.time()
        event_id = str(uuid.uuid4())
        
        try:
            # Create start event
            start_event = TraceEvent(
                event_id=event_id,
                operation_type=operation_type,
                timestamp=datetime.now(),
                level=TraceLevel.INFO,
                message=f"Starting {operation_type.value} operation",
                memory_id=memory_id,
                metadata=kwargs
            )
            self._emit_event(start_event)
            
            yield event_id
            
            # Create success event
            duration_ms = (time.time() - start_time) * 1000
            success_event = TraceEvent(
                event_id=event_id,
                operation_type=operation_type,
                timestamp=datetime.now(),
                level=TraceLevel.INFO,
                message=f"Completed {operation_type.value} operation",
                memory_id=memory_id,
                metadata=kwargs,
                duration_ms=duration_ms
            )
            self._emit_event(success_event)
            
        except Exception as e:
            # Create error event
            duration_ms = (time.time() - start_time) * 1000
            error_event = TraceEvent(
                event_id=event_id,
                operation_type=operation_type,
                timestamp=datetime.now(),
                level=TraceLevel.ERROR,
                message=f"Failed {operation_type.value} operation: {str(e)}",
                memory_id=memory_id,
                metadata=kwargs,
                duration_ms=duration_ms,
                error=str(e)
            )
            self._emit_event(error_event)
            raise
    
    def trace_memory_operation(self, operation_type: OperationType, memory_id: str, content: str = "", 
                              metadata: Dict[str, Any] = None, **kwargs) -> None:
        """Trace a memory operation."""
        if not self.trace_memory_operations:
            return
        
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            operation_type=operation_type,
            timestamp=datetime.now(),
            level=TraceLevel.INFO,
            message=f"Memory {operation_type.value}: {memory_id}",
            memory_id=memory_id,
            metadata=metadata or {},
            **kwargs
        )
        self._emit_event(event)
        
        # Create timeline entry
        timeline_entry = TimelineEntry(
            entry_id=str(uuid.uuid4()),
            memory_id=memory_id,
            operation=operation_type,
            timestamp=datetime.now(),
            tier=kwargs.get('tier', 'unknown'),
            content_preview=content[:100] + "..." if len(content) > 100 else content,
            metadata=metadata or {},
            performance_metrics=kwargs.get('performance_metrics', {})
        )
        self._emit_timeline_entry(timeline_entry)
    
    def trace_tier_migration(self, memory_id: str, tier_from: str, tier_to: str, 
                           reason: str = "", **kwargs) -> None:
        """Trace a tier migration."""
        if not self.trace_tier_migrations:
            return
        
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            operation_type=OperationType.MIGRATE,
            timestamp=datetime.now(),
            level=TraceLevel.INFO,
            message=f"Tier migration: {memory_id} from {tier_from} to {tier_to}",
            memory_id=memory_id,
            tier_from=tier_from,
            tier_to=tier_to,
            metadata={'reason': reason, **kwargs}
        )
        self._emit_event(event)
    
    def trace_decay_calculation(self, memory_id: str, decay_factor: float, 
                              decay_strategy: str, **kwargs) -> None:
        """Trace a decay calculation."""
        if not self.trace_decay_calculations:
            return
        
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            operation_type=OperationType.DECAY,
            timestamp=datetime.now(),
            level=TraceLevel.DEBUG,
            message=f"Decay calculation: {memory_id} = {decay_factor:.4f}",
            memory_id=memory_id,
            metadata={
                'decay_factor': decay_factor,
                'decay_strategy': decay_strategy,
                **kwargs
            }
        )
        self._emit_event(event)
    
    def trace_recall_operation(self, query: str, results_count: int, 
                             scope: str = "all", **kwargs) -> None:
        """Trace a recall operation."""
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            operation_type=OperationType.RECALL,
            timestamp=datetime.now(),
            level=TraceLevel.INFO,
            message=f"Recall: {results_count} results for query",
            metadata={
                'query': query,
                'results_count': results_count,
                'scope': scope,
                **kwargs
            }
        )
        self._emit_event(event)
    
    def _emit_event(self, event: TraceEvent) -> None:
        """Emit a trace event to all hooks."""
        self.events.append(event)
        self._trim_events()
        
        for hook in self.timeline_hooks:
            try:
                hook.on_event(event)
            except Exception as e:
                self.logger.error(f"Error in timeline hook: {e}")
    
    def _emit_timeline_entry(self, entry: TimelineEntry) -> None:
        """Emit a timeline entry to all hooks."""
        self.timeline_entries.append(entry)
        self._trim_timeline_entries()
        
        for hook in self.timeline_hooks:
            try:
                hook.on_timeline_entry(entry)
            except Exception as e:
                self.logger.error(f"Error in timeline hook: {e}")
    
    def _trim_events(self) -> None:
        """Trim events to max_events."""
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def _trim_timeline_entries(self) -> None:
        """Trim timeline entries to max_timeline_entries."""
        if len(self.timeline_entries) > self.max_timeline_entries:
            self.timeline_entries = self.timeline_entries[-self.max_timeline_entries:]
    
    def get_events(self, operation_type: Optional[OperationType] = None, 
                  level: Optional[TraceLevel] = None, limit: int = 100) -> List[TraceEvent]:
        """Get trace events with optional filtering."""
        events = self.events
        
        if operation_type:
            events = [e for e in events if e.operation_type == operation_type]
        
        if level:
            events = [e for e in events if e.level == level]
        
        return events[-limit:]
    
    def get_timeline_entries(self, memory_id: Optional[str] = None, 
                           operation_type: Optional[OperationType] = None, 
                           limit: int = 100) -> List[TimelineEntry]:
        """Get timeline entries with optional filtering."""
        entries = self.timeline_entries
        
        if memory_id:
            entries = [e for e in entries if e.memory_id == memory_id]
        
        if operation_type:
            entries = [e for e in entries if e.operation == operation_type]
        
        return entries[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        stats = {
            'total_events': len(self.events),
            'total_timeline_entries': len(self.timeline_entries),
            'hooks_count': len(self.timeline_hooks)
        }
        
        # Operation type statistics
        operation_counts = {}
        for event in self.events:
            op_type = event.operation_type.value
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        stats['operation_counts'] = operation_counts
        
        # Level statistics
        level_counts = {}
        for event in self.events:
            level = event.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        stats['level_counts'] = level_counts
        
        # Performance statistics
        durations = [e.duration_ms for e in self.events if e.duration_ms is not None]
        if durations:
            stats['avg_duration_ms'] = np.mean(durations)
            stats['max_duration_ms'] = np.max(durations)
            stats['min_duration_ms'] = np.min(durations)
        
        return stats
    
    def export_timeline(self, file_path: str, format: str = "json") -> None:
        """Export timeline to file."""
        if format.lower() == "json":
            timeline_data = {
                'events': [asdict(e) for e in self.events],
                'timeline_entries': [asdict(e) for e in self.timeline_entries],
                'statistics': self.get_statistics()
            }
            
            # Convert datetime objects to strings
            for event in timeline_data['events']:
                event['timestamp'] = event['timestamp'].isoformat()
            
            for entry in timeline_data['timeline_entries']:
                entry['timestamp'] = entry['timestamp'].isoformat()
            
            with open(file_path, 'w') as f:
                json.dump(timeline_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_events(self) -> None:
        """Clear all events."""
        self.events.clear()
    
    def clear_timeline_entries(self) -> None:
        """Clear all timeline entries."""
        self.timeline_entries.clear() 