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
Privacy and Consent Management for GDPR Compliance

This module implements:
- GDPR consent validation and tracking
- Privacy policy compliance checks
- Data processing consent management
- Right to be forgotten support
- Privacy audit trail
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import yaml


class ConsentStatus(Enum):
    """Consent status enumeration."""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


class DataCategory(Enum):
    """Data categories for GDPR compliance."""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_DATA = "sensitive_data"
    BEHAVIORAL_DATA = "behavioral_data"
    TECHNICAL_DATA = "technical_data"
    ANALYTICS_DATA = "analytics_data"


class ProcessingPurpose(Enum):
    """Data processing purposes."""
    MEMORY_STORAGE = "memory_storage"
    SIMILARITY_SEARCH = "similarity_search"
    ANALYTICS = "analytics"
    IMPROVEMENT = "improvement"
    CUSTOMIZATION = "customization"


@dataclass_json
@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    consent_id: str
    user_id: str
    agent_id: str
    consent_status: ConsentStatus
    data_categories: Set[DataCategory]
    processing_purposes: Set[ProcessingPurpose]
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_version: str = "1.0"
    terms_accepted: bool = False
    gdpr_compliant: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.consent_status in [ConsentStatus.DENIED, ConsentStatus.WITHDRAWN]:
            return False
        
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        
        return True
    
    def has_expired(self) -> bool:
        """Check if consent has expired."""
        return self.expires_at and datetime.now() > self.expires_at
    
    def can_process_data(self, category: DataCategory, purpose: ProcessingPurpose) -> bool:
        """Check if data can be processed for given category and purpose."""
        if not self.is_valid():
            return False
        
        return (category in self.data_categories and 
                purpose in self.processing_purposes)


@dataclass
class PrivacyPolicy:
    """Privacy policy configuration."""
    policy_id: str
    version: str
    effective_date: datetime
    data_categories: Set[DataCategory]
    processing_purposes: Set[ProcessingPurpose]
    retention_period_days: int
    data_sharing: bool
    international_transfers: bool
    automated_decision_making: bool
    contact_email: str
    policy_url: str
    gdpr_compliant: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyAuditEvent:
    """Privacy audit event."""
    event_id: str
    timestamp: datetime
    user_id: str
    agent_id: str
    event_type: str
    data_category: Optional[DataCategory]
    processing_purpose: Optional[ProcessingPurpose]
    consent_id: Optional[str]
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsentValidator(ABC):
    """Abstract base class for consent validation."""
    
    @abstractmethod
    def validate_consent(self, user_id: str, agent_id: str, 
                        data_category: DataCategory, 
                        processing_purpose: ProcessingPurpose) -> bool:
        """Validate if consent exists and is valid for the given parameters."""
        pass
    
    @abstractmethod
    def get_consent_record(self, user_id: str, agent_id: str) -> Optional[ConsentRecord]:
        """Get the current consent record for a user and agent."""
        pass


class GDPRConsentValidator(ConsentValidator):
    """GDPR-compliant consent validator."""
    
    def __init__(self, privacy_policy: PrivacyPolicy):
        self.privacy_policy = privacy_policy
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.audit_events: List[PrivacyAuditEvent] = []
        self.logger = logging.getLogger(__name__)
    
    def validate_consent(self, user_id: str, agent_id: str, 
                        data_category: DataCategory, 
                        processing_purpose: ProcessingPurpose) -> bool:
        """Validate GDPR consent for data processing."""
        consent_record = self.get_consent_record(user_id, agent_id)
        
        if not consent_record:
            self._record_audit_event(
                user_id, agent_id, "consent_validation_failed",
                data_category, processing_purpose, None, False,
                "No consent record found"
            )
            return False
        
        if not consent_record.is_valid():
            self._record_audit_event(
                user_id, agent_id, "consent_validation_failed",
                data_category, processing_purpose, consent_record.consent_id, False,
                f"Consent is not valid: {consent_record.consent_status.value}"
            )
            return False
        
        if not consent_record.can_process_data(data_category, processing_purpose):
            self._record_audit_event(
                user_id, agent_id, "consent_validation_failed",
                data_category, processing_purpose, consent_record.consent_id, False,
                f"Consent does not cover {data_category.value} for {processing_purpose.value}"
            )
            return False
        
        # Record successful validation
        self._record_audit_event(
            user_id, agent_id, "consent_validation_success",
            data_category, processing_purpose, consent_record.consent_id, True
        )
        
        return True
    
    def get_consent_record(self, user_id: str, agent_id: str) -> Optional[ConsentRecord]:
        """Get consent record for user and agent."""
        key = f"{user_id}:{agent_id}"
        return self.consent_records.get(key)
    
    def grant_consent(self, user_id: str, agent_id: str, 
                     data_categories: Set[DataCategory],
                     processing_purposes: Set[ProcessingPurpose],
                     expires_at: Optional[datetime] = None,
                     ip_address: Optional[str] = None,
                     user_agent: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> ConsentRecord:
        """Grant consent for data processing."""
        consent_id = self._generate_consent_id(user_id, agent_id)
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            agent_id=agent_id,
            consent_status=ConsentStatus.GRANTED,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            granted_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            terms_accepted=True,
            gdpr_compliant=True,
            metadata=metadata or {}
        )
        
        key = f"{user_id}:{agent_id}"
        self.consent_records[key] = consent_record
        
        self._record_audit_event(
            user_id, agent_id, "consent_granted",
            None, None, consent_id, True,
            metadata={"data_categories": [c.value for c in data_categories],
                     "processing_purposes": [p.value for p in processing_purposes]}
        )
        
        self.logger.info(f"Consent granted for user {user_id}, agent {agent_id}")
        return consent_record
    
    def withdraw_consent(self, user_id: str, agent_id: str) -> bool:
        """Withdraw consent for data processing."""
        consent_record = self.get_consent_record(user_id, agent_id)
        
        if not consent_record:
            return False
        
        consent_record.consent_status = ConsentStatus.WITHDRAWN
        consent_record.withdrawn_at = datetime.now()
        
        self._record_audit_event(
            user_id, agent_id, "consent_withdrawn",
            None, None, consent_record.consent_id, True
        )
        
        self.logger.info(f"Consent withdrawn for user {user_id}, agent {agent_id}")
        return True
    
    def update_consent(self, user_id: str, agent_id: str,
                      data_categories: Optional[Set[DataCategory]] = None,
                      processing_purposes: Optional[Set[ProcessingPurpose]] = None,
                      expires_at: Optional[datetime] = None) -> bool:
        """Update existing consent."""
        consent_record = self.get_consent_record(user_id, agent_id)
        
        if not consent_record:
            return False
        
        if data_categories:
            consent_record.data_categories = data_categories
        if processing_purposes:
            consent_record.processing_purposes = processing_purposes
        if expires_at:
            consent_record.expires_at = expires_at
        
        self._record_audit_event(
            user_id, agent_id, "consent_updated",
            None, None, consent_record.consent_id, True
        )
        
        self.logger.info(f"Consent updated for user {user_id}, agent {agent_id}")
        return True
    
    def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records."""
        expired_count = 0
        current_time = datetime.now()
        
        for key, consent_record in list(self.consent_records.items()):
            if consent_record.has_expired():
                del self.consent_records[key]
                expired_count += 1
                
                self._record_audit_event(
                    consent_record.user_id, consent_record.agent_id, "consent_expired",
                    None, None, consent_record.consent_id, True
                )
        
        self.logger.info(f"Cleaned up {expired_count} expired consent records")
        return expired_count
    
    def get_consent_statistics(self) -> Dict[str, Any]:
        """Get consent statistics."""
        total_consents = len(self.consent_records)
        valid_consents = len([c for c in self.consent_records.values() if c.is_valid()])
        expired_consents = len([c for c in self.consent_records.values() if c.has_expired()])
        withdrawn_consents = len([c for c in self.consent_records.values() 
                                if c.consent_status == ConsentStatus.WITHDRAWN])
        
        return {
            "total_consents": total_consents,
            "valid_consents": valid_consents,
            "expired_consents": expired_consents,
            "withdrawn_consents": withdrawn_consents,
            "gdpr_compliant_rate": valid_consents / total_consents if total_consents > 0 else 0.0,
            "audit_events_count": len(self.audit_events)
        }
    
    def export_consent_data(self, user_id: str, agent_id: str) -> Dict[str, Any]:
        """Export consent data for GDPR right to access."""
        consent_record = self.get_consent_record(user_id, agent_id)
        
        if not consent_record:
            return {"error": "No consent record found"}
        
        # Get audit events for this user/agent
        user_audit_events = [
            event for event in self.audit_events
            if event.user_id == user_id and event.agent_id == agent_id
        ]
        
        return {
            "consent_record": consent_record.to_dict(),
            "audit_events": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "data_category": event.data_category.value if event.data_category else None,
                    "processing_purpose": event.processing_purpose.value if event.processing_purpose else None,
                    "success": event.success,
                    "error_message": event.error_message,
                    "metadata": event.metadata
                }
                for event in user_audit_events
            ],
            "privacy_policy": {
                "policy_id": self.privacy_policy.policy_id,
                "version": self.privacy_policy.version,
                "effective_date": self.privacy_policy.effective_date.isoformat(),
                "data_categories": [c.value for c in self.privacy_policy.data_categories],
                "processing_purposes": [p.value for c in self.privacy_policy.processing_purposes],
                "retention_period_days": self.privacy_policy.retention_period_days,
                "contact_email": self.privacy_policy.contact_email,
                "policy_url": self.privacy_policy.policy_url
            }
        }
    
    def _generate_consent_id(self, user_id: str, agent_id: str) -> str:
        """Generate a unique consent ID."""
        data = f"{user_id}:{agent_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _record_audit_event(self, user_id: str, agent_id: str, event_type: str,
                           data_category: Optional[DataCategory],
                           processing_purpose: Optional[ProcessingPurpose],
                           consent_id: Optional[str], success: bool,
                           error_message: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a privacy audit event."""
        event = PrivacyAuditEvent(
            event_id=f"audit_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            user_id=user_id,
            agent_id=agent_id,
            event_type=event_type,
            data_category=data_category,
            processing_purpose=processing_purpose,
            consent_id=consent_id,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        self.audit_events.append(event)
        
        # Keep only last 10000 audit events
        if len(self.audit_events) > 10000:
            self.audit_events = self.audit_events[-10000:]


class PrivacyPolicyLoader:
    """Loader for privacy policy configuration."""
    
    @staticmethod
    def load_from_yaml(file_path: str) -> PrivacyPolicy:
        """Load privacy policy from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return PrivacyPolicy(
            policy_id=data["policy_id"],
            version=data["version"],
            effective_date=datetime.fromisoformat(data["effective_date"]),
            data_categories={DataCategory(cat) for cat in data["data_categories"]},
            processing_purposes={ProcessingPurpose(purpose) for purpose in data["processing_purposes"]},
            retention_period_days=data["retention_period_days"],
            data_sharing=data["data_sharing"],
            international_transfers=data["international_transfers"],
            automated_decision_making=data["automated_decision_making"],
            contact_email=data["contact_email"],
            policy_url=data["policy_url"],
            gdpr_compliant=data.get("gdpr_compliant", True),
            metadata=data.get("metadata", {})
        )
    
    @staticmethod
    def create_default_policy() -> PrivacyPolicy:
        """Create a default GDPR-compliant privacy policy."""
        return PrivacyPolicy(
            policy_id="default_gdpr_policy",
            version="1.0",
            effective_date=datetime.now(),
            data_categories={
                DataCategory.PERSONAL_DATA,
                DataCategory.TECHNICAL_DATA,
                DataCategory.BEHAVIORAL_DATA
            },
            processing_purposes={
                ProcessingPurpose.MEMORY_STORAGE,
                ProcessingPurpose.SIMILARITY_SEARCH,
                ProcessingPurpose.CUSTOMIZATION
            },
            retention_period_days=365,
            data_sharing=False,
            international_transfers=False,
            automated_decision_making=False,
            contact_email="privacy@memorix.ai",
            policy_url="https://memorix.ai/privacy",
            gdpr_compliant=True
        ) 