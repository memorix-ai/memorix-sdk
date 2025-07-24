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
Memory Access Audit Trail and Logging System

This module implements:
- Comprehensive memory access audit trails
- GDPR-compliant logging
- Performance monitoring
- Security event tracking
- Audit report generation
"""

import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


class AuditLevel(Enum):
    """Audit log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """Types of audit events."""
    # Memory operations
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_ACCESS = "memory_access"
    
    # Privacy operations
    CONSENT_GRANTED = "consent_granted"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    DATA_ERASED = "data_erased"
    PRIVACY_CHECK = "privacy_check"
    
    # System operations
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGE = "config_change"
    ERROR_OCCURRED = "error_occurred"
    
    # Security events
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_DENIED = "access_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AuditCategory(Enum):
    """Audit event categories."""
    MEMORY = "memory"
    PRIVACY = "privacy"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    category: AuditCategory
    level: AuditLevel
    user_id: Optional[str]
    agent_id: Optional[str]
    memory_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_signature: Optional[str] = None
    
    def __post_init__(self):
        """Generate hash signature for integrity."""
        if not self.hash_signature:
            self.hash_signature = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate hash signature for audit integrity."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "memory_id": self.memory_id,
            "description": self.description
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class AuditFilter:
    """Filter for audit queries."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    categories: Optional[List[AuditCategory]] = None
    levels: Optional[List[AuditLevel]] = None
    user_ids: Optional[List[str]] = None
    agent_ids: Optional[List[str]] = None
    memory_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    ip_addresses: Optional[List[str]] = None


@dataclass
class AuditStatistics:
    """Audit statistics."""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_category: Dict[str, int]
    events_by_level: Dict[str, int]
    events_by_user: Dict[str, int]
    events_by_agent: Dict[str, int]
    time_distribution: Dict[str, int]
    error_rate: float
    average_response_time: float


class AuditLogger(ABC):
    """Abstract base class for audit logging."""
    
    @abstractmethod
    def log_event(self, event: AuditEvent) -> bool:
        """Log an audit event."""
        pass
    
    @abstractmethod
    def get_events(self, filter_criteria: Optional[AuditFilter] = None, 
                  limit: int = 1000, offset: int = 0) -> List[AuditEvent]:
        """Retrieve audit events with optional filtering."""
        pass
    
    @abstractmethod
    def get_statistics(self, filter_criteria: Optional[AuditFilter] = None) -> AuditStatistics:
        """Get audit statistics."""
        pass


class SQLiteAuditLogger(AuditLogger):
    """SQLite-based audit logger with GDPR compliance."""
    
    def __init__(self, db_path: str, config: Dict[str, Any]):
        self.db_path = db_path
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.retention_days = config.get("audit.retention_days", 2555)  # 7 years
        self.max_events = config.get("audit.max_events", 1000000)
        self.enable_encryption = config.get("audit.enable_encryption", False)
        self.compression_enabled = config.get("audit.compression_enabled", True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the audit database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create audit events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        level TEXT NOT NULL,
                        user_id TEXT,
                        agent_id TEXT,
                        memory_id TEXT,
                        session_id TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        description TEXT NOT NULL,
                        details TEXT,
                        metadata TEXT,
                        hash_signature TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON audit_events(agent_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_id ON audit_events(memory_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_level ON audit_events(level)")
                
                # Create statistics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        level TEXT NOT NULL,
                        count INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize audit database: {e}")
            raise
    
    def log_event(self, event: AuditEvent) -> bool:
        """Log an audit event to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, category, level,
                        user_id, agent_id, memory_id, session_id, ip_address,
                        user_agent, description, details, metadata, hash_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.category.value,
                    event.level.value,
                    event.user_id,
                    event.agent_id,
                    event.memory_id,
                    event.session_id,
                    event.ip_address,
                    event.user_agent,
                    event.description,
                    json.dumps(event.details),
                    json.dumps(event.metadata),
                    event.hash_signature
                ))
                
                conn.commit()
                
                # Update statistics
                self._update_statistics(event)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            return False
    
    def get_events(self, filter_criteria: Optional[AuditFilter] = None, 
                  limit: int = 1000, offset: int = 0) -> List[AuditEvent]:
        """Retrieve audit events with optional filtering."""
        try:
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if filter_criteria:
                if filter_criteria.start_time:
                    query += " AND timestamp >= ?"
                    params.append(filter_criteria.start_time.isoformat())
                
                if filter_criteria.end_time:
                    query += " AND timestamp <= ?"
                    params.append(filter_criteria.end_time.isoformat())
                
                if filter_criteria.event_types:
                    placeholders = ",".join(["?"] * len(filter_criteria.event_types))
                    query += f" AND event_type IN ({placeholders})"
                    params.extend([et.value for et in filter_criteria.event_types])
                
                if filter_criteria.categories:
                    placeholders = ",".join(["?"] * len(filter_criteria.categories))
                    query += f" AND category IN ({placeholders})"
                    params.extend([cat.value for cat in filter_criteria.categories])
                
                if filter_criteria.levels:
                    placeholders = ",".join(["?"] * len(filter_criteria.levels))
                    query += f" AND level IN ({placeholders})"
                    params.extend([level.value for level in filter_criteria.levels])
                
                if filter_criteria.user_ids:
                    placeholders = ",".join(["?"] * len(filter_criteria.user_ids))
                    query += f" AND user_id IN ({placeholders})"
                    params.extend(filter_criteria.user_ids)
                
                if filter_criteria.agent_ids:
                    placeholders = ",".join(["?"] * len(filter_criteria.agent_ids))
                    query += f" AND agent_id IN ({placeholders})"
                    params.extend(filter_criteria.agent_ids)
                
                if filter_criteria.memory_ids:
                    placeholders = ",".join(["?"] * len(filter_criteria.memory_ids))
                    query += f" AND memory_id IN ({placeholders})"
                    params.extend(filter_criteria.memory_ids)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = self._row_to_event(row)
                    events.append(event)
                
                return events
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve audit events: {e}")
            return []
    
    def get_statistics(self, filter_criteria: Optional[AuditFilter] = None) -> AuditStatistics:
        """Get audit statistics."""
        try:
            # Build base query
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if filter_criteria:
                if filter_criteria.start_time:
                    query += " AND timestamp >= ?"
                    params.append(filter_criteria.start_time.isoformat())
                
                if filter_criteria.end_time:
                    query += " AND timestamp <= ?"
                    params.append(filter_criteria.end_time.isoformat())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total events
                cursor.execute(f"SELECT COUNT(*) FROM ({query})", params)
                total_events = cursor.fetchone()[0]
                
                # Get events by type
                cursor.execute(f"SELECT event_type, COUNT(*) FROM ({query}) GROUP BY event_type", params)
                events_by_type = dict(cursor.fetchall())
                
                # Get events by category
                cursor.execute(f"SELECT category, COUNT(*) FROM ({query}) GROUP BY category", params)
                events_by_category = dict(cursor.fetchall())
                
                # Get events by level
                cursor.execute(f"SELECT level, COUNT(*) FROM ({query}) GROUP BY level", params)
                events_by_level = dict(cursor.fetchall())
                
                # Get events by user
                cursor.execute(f"SELECT user_id, COUNT(*) FROM ({query}) WHERE user_id IS NOT NULL GROUP BY user_id", params)
                events_by_user = dict(cursor.fetchall())
                
                # Get events by agent
                cursor.execute(f"SELECT agent_id, COUNT(*) FROM ({query}) WHERE agent_id IS NOT NULL GROUP BY agent_id", params)
                events_by_agent = dict(cursor.fetchall())
                
                # Calculate error rate
                cursor.execute(f"SELECT COUNT(*) FROM ({query}) WHERE level IN ('error', 'critical')", params)
                error_count = cursor.fetchone()[0]
                error_rate = error_count / total_events if total_events > 0 else 0.0
                
                # Get time distribution (by hour)
                cursor.execute(f"SELECT strftime('%H', timestamp) as hour, COUNT(*) FROM ({query}) GROUP BY hour", params)
                time_distribution = dict(cursor.fetchall())
                
                return AuditStatistics(
                    total_events=total_events,
                    events_by_type=events_by_type,
                    events_by_category=events_by_category,
                    events_by_level=events_by_level,
                    events_by_user=events_by_user,
                    events_by_agent=events_by_agent,
                    time_distribution=time_distribution,
                    error_rate=error_rate,
                    average_response_time=0.0  # Would need additional tracking
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get audit statistics: {e}")
            return AuditStatistics(
                total_events=0,
                events_by_type={},
                events_by_category={},
                events_by_level={},
                events_by_user={},
                events_by_agent={},
                time_distribution={},
                error_rate=0.0,
                average_response_time=0.0
            )
    
    def export_audit_log(self, file_path: str, filter_criteria: Optional[AuditFilter] = None,
                        format: str = "json") -> bool:
        """Export audit log to file."""
        try:
            events = self.get_events(filter_criteria, limit=100000)  # Large limit for export
            
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump([asdict(event) for event in events], f, indent=2, default=str)
            
            elif format.lower() == "csv":
                import csv
                with open(file_path, 'w', newline='') as f:
                    if events:
                        writer = csv.DictWriter(f, fieldnames=asdict(events[0]).keys())
                        writer.writeheader()
                        for event in events:
                            writer.writerow(asdict(event))
            
            elif format.lower() == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump([asdict(event) for event in events], f, default_flow_style=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Audit log exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export audit log: {e}")
            return False
    
    def cleanup_old_events(self) -> int:
        """Clean up old audit events based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old events
                cursor.execute("DELETE FROM audit_events WHERE timestamp < ?", 
                             (cutoff_date.isoformat(),))
                deleted_count = cursor.rowcount
                
                # Vacuum database to reclaim space
                cursor.execute("VACUUM")
                
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old audit events")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old events: {e}")
            return 0
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of audit logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for events with missing hash signatures
                cursor.execute("SELECT COUNT(*) FROM audit_events WHERE hash_signature IS NULL")
                missing_hashes = cursor.fetchone()[0]
                
                # Check for duplicate event IDs
                cursor.execute("SELECT event_id, COUNT(*) FROM audit_events GROUP BY event_id HAVING COUNT(*) > 1")
                duplicates = cursor.fetchall()
                
                # Check for events with invalid timestamps
                cursor.execute("SELECT COUNT(*) FROM audit_events WHERE timestamp IS NULL OR timestamp = ''")
                invalid_timestamps = cursor.fetchone()[0]
                
                # Verify hash signatures
                cursor.execute("SELECT event_id, hash_signature FROM audit_events WHERE hash_signature IS NOT NULL")
                hash_verifications = []
                
                for event_id, stored_hash in cursor.fetchall():
                    # Recalculate hash and compare
                    cursor.execute("SELECT * FROM audit_events WHERE event_id = ?", (event_id,))
                    row = cursor.fetchone()
                    if row:
                        event = self._row_to_event(row)
                        calculated_hash = event._generate_hash()
                        hash_verifications.append({
                            "event_id": event_id,
                            "stored_hash": stored_hash,
                            "calculated_hash": calculated_hash,
                            "valid": stored_hash == calculated_hash
                        })
                
                invalid_hashes = len([v for v in hash_verifications if not v["valid"]])
                
                return {
                    "total_events": self.get_statistics().total_events,
                    "missing_hashes": missing_hashes,
                    "duplicate_events": len(duplicates),
                    "invalid_timestamps": invalid_timestamps,
                    "invalid_hashes": invalid_hashes,
                    "integrity_score": self._calculate_integrity_score(
                        missing_hashes, len(duplicates), invalid_timestamps, invalid_hashes
                    )
                }
                
        except Exception as e:
            self.logger.error(f"Failed to verify audit integrity: {e}")
            return {"error": str(e)}
    
    def _row_to_event(self, row) -> AuditEvent:
        """Convert database row to AuditEvent."""
        return AuditEvent(
            event_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            event_type=AuditEventType(row[2]),
            category=AuditCategory(row[3]),
            level=AuditLevel(row[4]),
            user_id=row[5],
            agent_id=row[6],
            memory_id=row[7],
            session_id=row[8],
            ip_address=row[9],
            user_agent=row[10],
            description=row[11],
            details=json.loads(row[12]) if row[12] else {},
            metadata=json.loads(row[13]) if row[13] else {},
            hash_signature=row[14]
        )
    
    def _update_statistics(self, event: AuditEvent) -> None:
        """Update audit statistics."""
        try:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update or insert statistics
                cursor.execute("""
                    INSERT OR REPLACE INTO audit_statistics 
                    (date, event_type, category, level, count)
                    VALUES (?, ?, ?, ?, 
                        COALESCE((SELECT count FROM audit_statistics 
                                 WHERE date = ? AND event_type = ? AND category = ? AND level = ?), 0) + 1)
                """, (
                    date_str, event.event_type.value, event.category.value, event.level.value,
                    date_str, event.event_type.value, event.category.value, event.level.value
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update audit statistics: {e}")
    
    def _calculate_integrity_score(self, missing_hashes: int, duplicates: int, 
                                  invalid_timestamps: int, invalid_hashes: int) -> float:
        """Calculate audit integrity score (0-100)."""
        total_events = self.get_statistics().total_events
        if total_events == 0:
            return 100.0
        
        issues = missing_hashes + duplicates + invalid_timestamps + invalid_hashes
        score = max(0.0, 100.0 - (issues / total_events) * 100.0)
        return round(score, 2)


class AuditManager:
    """High-level audit manager for the memory system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize audit logger
        db_path = config.get("audit.database_path", "audit_log.db")
        self.audit_logger = SQLiteAuditLogger(db_path, config)
        
        # Audit configuration
        self.enable_audit = config.get("audit.enable_audit", True)
        self.audit_level = AuditLevel(config.get("audit.level", "info"))
        self.include_metadata = config.get("audit.include_metadata", True)
        self.include_ip_address = config.get("audit.include_ip_address", True)
        self.include_user_agent = config.get("audit.include_user_agent", True)
    
    def log_memory_operation(self, operation_type: AuditEventType, user_id: str, 
                           agent_id: str, memory_id: str, description: str,
                           details: Optional[Dict[str, Any]] = None,
                           level: AuditLevel = AuditLevel.INFO,
                           session_id: Optional[str] = None,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> bool:
        """Log a memory operation."""
        if not self.enable_audit:
            return True
        
        if level.value < self.audit_level.value:
            return True
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(),
            event_type=operation_type,
            category=AuditCategory.MEMORY,
            level=level,
            user_id=user_id,
            agent_id=agent_id,
            memory_id=memory_id,
            session_id=session_id,
            ip_address=ip_address if self.include_ip_address else None,
            user_agent=user_agent if self.include_user_agent else None,
            description=description,
            details=details or {},
            metadata=self._get_metadata() if self.include_metadata else {}
        )
        
        return self.audit_logger.log_event(event)
    
    def log_privacy_operation(self, operation_type: AuditEventType, user_id: str,
                            agent_id: str, description: str,
                            details: Optional[Dict[str, Any]] = None,
                            level: AuditLevel = AuditLevel.INFO) -> bool:
        """Log a privacy operation."""
        if not self.enable_audit:
            return True
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(),
            event_type=operation_type,
            category=AuditCategory.PRIVACY,
            level=level,
            user_id=user_id,
            agent_id=agent_id,
            description=description,
            details=details or {}
        )
        
        return self.audit_logger.log_event(event)
    
    def log_security_event(self, event_type: AuditEventType, user_id: Optional[str],
                          agent_id: Optional[str], description: str,
                          details: Optional[Dict[str, Any]] = None,
                          level: AuditLevel = AuditLevel.WARNING,
                          ip_address: Optional[str] = None) -> bool:
        """Log a security event."""
        if not self.enable_audit:
            return True
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(),
            event_type=event_type,
            category=AuditCategory.SECURITY,
            level=level,
            user_id=user_id,
            agent_id=agent_id,
            ip_address=ip_address,
            description=description,
            details=details or {}
        )
        
        return self.audit_logger.log_event(event)
    
    def get_audit_report(self, filter_criteria: Optional[AuditFilter] = None,
                        format: str = "json") -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        try:
            statistics = self.audit_logger.get_statistics(filter_criteria)
            integrity = self.audit_logger.verify_audit_integrity()
            
            report = {
                "generated_at": datetime.now().isoformat(),
                "filter_criteria": asdict(filter_criteria) if filter_criteria else None,
                "statistics": asdict(statistics),
                "integrity": integrity,
                "summary": {
                    "total_events": statistics.total_events,
                    "error_rate": f"{statistics.error_rate:.2%}",
                    "integrity_score": integrity.get("integrity_score", 0),
                    "most_active_user": max(statistics.events_by_user.items(), 
                                          key=lambda x: x[1])[0] if statistics.events_by_user else None,
                    "most_active_agent": max(statistics.events_by_agent.items(), 
                                           key=lambda x: x[1])[0] if statistics.events_by_agent else None
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate audit report: {e}")
            return {"error": str(e)}
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"audit_{datetime.now().timestamp()}_{hash(str(datetime.now()))}"
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get system metadata for audit events."""
        return {
            "system_version": "1.0.0",
            "python_version": "3.8+",
            "platform": "memorix-sdk"
        } 