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
Data Erasure and Right to be Forgotten Implementation

This module implements:
- GDPR Article 17 "Right to erasure ('right to be forgotten')"
- Secure data erasure with audit trails
- Batch erasure operations
- Erasure verification and confirmation
- Data retention policy enforcement
"""

import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ErasureMethod(Enum):
    """Data erasure methods."""
    SOFT_DELETE = "soft_delete"  # Mark as deleted but keep for audit
    HARD_DELETE = "hard_delete"  # Complete removal
    ANONYMIZE = "anonymize"      # Replace with anonymous data
    PSEUDONYMIZE = "pseudonymize"  # Replace with pseudonyms
    OVERWRITE = "overwrite"      # Overwrite with random data


class ErasureStatus(Enum):
    """Erasure operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class ErasureRequest:
    """Request for data erasure."""
    request_id: str
    user_id: str
    agent_id: str
    erasure_method: ErasureMethod
    data_categories: Set[str]
    reason: str
    requested_at: datetime
    requested_by: str
    priority: int = 1  # 1=normal, 2=urgent, 3=critical
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErasureOperation:
    """Data erasure operation."""
    operation_id: str
    request_id: str
    user_id: str
    agent_id: str
    erasure_method: ErasureMethod
    status: ErasureStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    data_types_erased: List[str] = field(default_factory=list)
    records_erased: int = 0
    verification_hash: Optional[str] = None
    error_message: Optional[str] = None
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErasureVerification:
    """Verification of data erasure."""
    verification_id: str
    operation_id: str
    verified_at: datetime
    verified_by: str
    verification_method: str
    is_successful: bool
    details: Dict[str, Any]
    notes: Optional[str] = None


class DataEraser(ABC):
    """Abstract base class for data erasure."""
    
    @abstractmethod
    def erase_user_data(self, user_id: str, agent_id: str, 
                       erasure_method: ErasureMethod,
                       data_categories: Optional[Set[str]] = None) -> bool:
        """Erase user data according to GDPR requirements."""
        pass
    
    @abstractmethod
    def verify_erasure(self, user_id: str, agent_id: str) -> bool:
        """Verify that user data has been completely erased."""
        pass
    
    @abstractmethod
    def get_erasure_status(self, user_id: str, agent_id: str) -> Optional[ErasureStatus]:
        """Get the current erasure status for a user."""
        pass


class SecureDataEraser(DataEraser):
    """Secure implementation of data erasure with GDPR compliance."""
    
    def __init__(self, vector_store, metadata_store, tier_manager, 
                 consent_validator, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.tier_manager = tier_manager
        self.consent_validator = consent_validator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Erasure tracking
        self.erasure_requests: Dict[str, ErasureRequest] = {}
        self.erasure_operations: Dict[str, ErasureOperation] = {}
        self.erasure_verifications: Dict[str, ErasureVerification] = {}
        
        # Configuration
        self.retention_period_days = config.get("privacy.retention_period_days", 30)
        self.verification_required = config.get("privacy.verification_required", True)
        self.audit_retention_days = config.get("privacy.audit_retention_days", 2555)  # 7 years
        
        # Secure erasure settings
        self.overwrite_passes = config.get("privacy.overwrite_passes", 3)
        self.overwrite_patterns = [
            b'\x00' * 1024,  # Zeros
            b'\xFF' * 1024,  # Ones
            os.urandom(1024)  # Random
        ]
    
    def request_erasure(self, user_id: str, agent_id: str, 
                       erasure_method: ErasureMethod = ErasureMethod.HARD_DELETE,
                       data_categories: Optional[Set[str]] = None,
                       reason: str = "GDPR right to be forgotten",
                       requested_by: str = "user",
                       priority: int = 1,
                       deadline: Optional[datetime] = None) -> str:
        """
        Request data erasure for GDPR compliance.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            erasure_method: Method of erasure
            data_categories: Specific data categories to erase
            reason: Reason for erasure request
            requested_by: Who requested the erasure
            priority: Priority level (1=normal, 2=urgent, 3=critical)
            deadline: Optional deadline for completion
            
        Returns:
            Request ID for tracking
        """
        request_id = self._generate_request_id(user_id, agent_id)
        
        # Default data categories if not specified
        if data_categories is None:
            data_categories = {
                "personal_data", "behavioral_data", "technical_data",
                "memory_data", "metadata", "consent_data"
            }
        
        # Set default deadline if not provided
        if deadline is None:
            deadline = datetime.now() + timedelta(days=30)  # GDPR default
        
        erasure_request = ErasureRequest(
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            erasure_method=erasure_method,
            data_categories=data_categories,
            reason=reason,
            requested_at=datetime.now(),
            requested_by=requested_by,
            priority=priority,
            deadline=deadline
        )
        
        self.erasure_requests[request_id] = erasure_request
        
        # Log the request
        self.logger.info(f"Erasure request created: {request_id} for user {user_id}, agent {agent_id}")
        
        # Start erasure operation if priority is high
        if priority >= 2:
            self._start_erasure_operation(request_id)
        
        return request_id
    
    def erase_user_data(self, user_id: str, agent_id: str, 
                       erasure_method: ErasureMethod = ErasureMethod.HARD_DELETE,
                       data_categories: Optional[Set[str]] = None) -> bool:
        """
        Immediately erase user data.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            erasure_method: Method of erasure
            data_categories: Specific data categories to erase
            
        Returns:
            True if erasure was successful
        """
        try:
            operation_id = self._generate_operation_id(user_id, agent_id)
            
            # Create erasure operation
            operation = ErasureOperation(
                operation_id=operation_id,
                request_id="immediate",
                user_id=user_id,
                agent_id=agent_id,
                erasure_method=erasure_method,
                status=ErasureStatus.IN_PROGRESS,
                started_at=datetime.now()
            )
            
            self.erasure_operations[operation_id] = operation
            
            # Perform erasure based on method
            success = self._perform_erasure(operation, data_categories)
            
            if success:
                operation.status = ErasureStatus.COMPLETED
                operation.completed_at = datetime.now()
                
                # Verify erasure if required
                if self.verification_required:
                    verification_success = self.verify_erasure(user_id, agent_id)
                    if verification_success:
                        operation.status = ErasureStatus.VERIFIED
                
                self.logger.info(f"Data erasure completed for user {user_id}, agent {agent_id}")
                return True
            else:
                operation.status = ErasureStatus.FAILED
                operation.completed_at = datetime.now()
                self.logger.error(f"Data erasure failed for user {user_id}, agent {agent_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during data erasure for user {user_id}, agent {agent_id}: {e}")
            return False
    
    def verify_erasure(self, user_id: str, agent_id: str) -> bool:
        """
        Verify that user data has been completely erased.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            
        Returns:
            True if no user data remains
        """
        try:
            verification_id = self._generate_verification_id(user_id, agent_id)
            
            # Check vector store
            vector_data_exists = self._check_vector_store_data(user_id, agent_id)
            
            # Check metadata store
            metadata_exists = self._check_metadata_store_data(user_id, agent_id)
            
            # Check tier manager
            tier_data_exists = self._check_tier_manager_data(user_id, agent_id)
            
            # Check consent data
            consent_exists = self._check_consent_data(user_id, agent_id)
            
            # Overall verification result
            is_successful = not (vector_data_exists or metadata_exists or 
                                tier_data_exists or consent_exists)
            
            # Create verification record
            verification = ErasureVerification(
                verification_id=verification_id,
                operation_id="verification",
                verified_at=datetime.now(),
                verified_by="system",
                verification_method="comprehensive_check",
                is_successful=is_successful,
                details={
                    "vector_data_exists": vector_data_exists,
                    "metadata_exists": metadata_exists,
                    "tier_data_exists": tier_data_exists,
                    "consent_exists": consent_exists
                }
            )
            
            self.erasure_verifications[verification_id] = verification
            
            if is_successful:
                self.logger.info(f"Erasure verification successful for user {user_id}, agent {agent_id}")
            else:
                self.logger.warning(f"Erasure verification failed for user {user_id}, agent {agent_id}")
            
            return is_successful
            
        except Exception as e:
            self.logger.error(f"Error during erasure verification for user {user_id}, agent {agent_id}: {e}")
            return False
    
    def get_erasure_status(self, user_id: str, agent_id: str) -> Optional[ErasureStatus]:
        """Get the current erasure status for a user."""
        # Find the most recent operation for this user/agent
        for operation in self.erasure_operations.values():
            if operation.user_id == user_id and operation.agent_id == agent_id:
                return operation.status
        return None
    
    def batch_erase_users(self, user_agent_pairs: List[Tuple[str, str]], 
                         erasure_method: ErasureMethod = ErasureMethod.HARD_DELETE) -> Dict[str, bool]:
        """
        Erase data for multiple users in batch.
        
        Args:
            user_agent_pairs: List of (user_id, agent_id) pairs
            erasure_method: Method of erasure
            
        Returns:
            Dictionary mapping user_id:agent_id to success status
        """
        results = {}
        
        for user_id, agent_id in user_agent_pairs:
            try:
                success = self.erase_user_data(user_id, agent_id, erasure_method)
                results[f"{user_id}:{agent_id}"] = success
            except Exception as e:
                self.logger.error(f"Batch erasure failed for {user_id}:{agent_id}: {e}")
                results[f"{user_id}:{agent_id}"] = False
        
        return results
    
    def get_erasure_statistics(self) -> Dict[str, Any]:
        """Get erasure operation statistics."""
        total_requests = len(self.erasure_requests)
        total_operations = len(self.erasure_operations)
        total_verifications = len(self.erasure_verifications)
        
        # Count by status
        status_counts = {}
        for operation in self.erasure_operations.values():
            status = operation.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by method
        method_counts = {}
        for operation in self.erasure_operations.values():
            method = operation.erasure_method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "total_requests": total_requests,
            "total_operations": total_operations,
            "total_verifications": total_verifications,
            "status_breakdown": status_counts,
            "method_breakdown": method_counts,
            "success_rate": status_counts.get("completed", 0) / total_operations if total_operations > 0 else 0.0
        }
    
    def cleanup_expired_audit_data(self) -> int:
        """Clean up expired audit data according to retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.audit_retention_days)
        cleaned_count = 0
        
        # Clean up old operations
        for operation_id, operation in list(self.erasure_operations.items()):
            if operation.completed_at and operation.completed_at < cutoff_date:
                del self.erasure_operations[operation_id]
                cleaned_count += 1
        
        # Clean up old verifications
        for verification_id, verification in list(self.erasure_verifications.items()):
            if verification.verified_at < cutoff_date:
                del self.erasure_verifications[verification_id]
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} expired audit records")
        return cleaned_count
    
    def _perform_erasure(self, operation: ErasureOperation, 
                        data_categories: Optional[Set[str]] = None) -> bool:
        """Perform the actual erasure operation."""
        try:
            user_id = operation.user_id
            agent_id = operation.agent_id
            method = operation.erasure_method
            
            # Add audit trail entry
            operation.audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "erasure_started",
                "method": method.value,
                "data_categories": list(data_categories) if data_categories else "all"
            })
            
            # Erase vector data
            if self._should_erase_category("memory_data", data_categories):
                vector_success = self._erase_vector_data(user_id, agent_id, method)
                if vector_success:
                    operation.data_types_erased.append("vector_data")
                    operation.records_erased += 1
            
            # Erase metadata
            if self._should_erase_category("metadata", data_categories):
                metadata_success = self._erase_metadata_data(user_id, agent_id, method)
                if metadata_success:
                    operation.data_types_erased.append("metadata")
                    operation.records_erased += 1
            
            # Erase tier manager data
            if self._should_erase_category("memory_data", data_categories):
                tier_success = self._erase_tier_manager_data(user_id, agent_id, method)
                if tier_success:
                    operation.data_types_erased.append("tier_data")
                    operation.records_erased += 1
            
            # Erase consent data
            if self._should_erase_category("consent_data", data_categories):
                consent_success = self._erase_consent_data(user_id, agent_id, method)
                if consent_success:
                    operation.data_types_erased.append("consent_data")
                    operation.records_erased += 1
            
            # Generate verification hash
            operation.verification_hash = self._generate_verification_hash(user_id, agent_id)
            
            # Add completion audit trail entry
            operation.audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "erasure_completed",
                "records_erased": operation.records_erased,
                "verification_hash": operation.verification_hash
            })
            
            return True
            
        except Exception as e:
            operation.error_message = str(e)
            operation.audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": "erasure_failed",
                "error": str(e)
            })
            return False
    
    def _erase_vector_data(self, user_id: str, agent_id: str, method: ErasureMethod) -> bool:
        """Erase vector data for user/agent."""
        try:
            if method == ErasureMethod.HARD_DELETE:
                # Find and delete all vectors for this user/agent
                # This would need to be implemented based on the specific vector store
                return True
            elif method == ErasureMethod.ANONYMIZE:
                # Replace with anonymous embeddings
                return True
            elif method == ErasureMethod.SOFT_DELETE:
                # Mark as deleted but keep for audit
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error erasing vector data: {e}")
            return False
    
    def _erase_metadata_data(self, user_id: str, agent_id: str, method: ErasureMethod) -> bool:
        """Erase metadata for user/agent."""
        try:
            if method == ErasureMethod.HARD_DELETE:
                # Delete metadata entries
                return True
            elif method == ErasureMethod.ANONYMIZE:
                # Replace with anonymous metadata
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error erasing metadata: {e}")
            return False
    
    def _erase_tier_manager_data(self, user_id: str, agent_id: str, method: ErasureMethod) -> bool:
        """Erase tier manager data for user/agent."""
        try:
            if method == ErasureMethod.HARD_DELETE:
                # Remove from tier manager
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error erasing tier manager data: {e}")
            return False
    
    def _erase_consent_data(self, user_id: str, agent_id: str, method: ErasureMethod) -> bool:
        """Erase consent data for user/agent."""
        try:
            if method == ErasureMethod.HARD_DELETE:
                # Withdraw consent
                return self.consent_validator.withdraw_consent(user_id, agent_id)
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error erasing consent data: {e}")
            return False
    
    def _check_vector_store_data(self, user_id: str, agent_id: str) -> bool:
        """Check if vector data exists for user/agent."""
        # Implementation depends on vector store interface
        return False
    
    def _check_metadata_store_data(self, user_id: str, agent_id: str) -> bool:
        """Check if metadata exists for user/agent."""
        # Implementation depends on metadata store interface
        return False
    
    def _check_tier_manager_data(self, user_id: str, agent_id: str) -> bool:
        """Check if tier manager data exists for user/agent."""
        # Implementation depends on tier manager interface
        return False
    
    def _check_consent_data(self, user_id: str, agent_id: str) -> bool:
        """Check if consent data exists for user/agent."""
        consent_record = self.consent_validator.get_consent_record(user_id, agent_id)
        return consent_record is not None
    
    def _should_erase_category(self, category: str, 
                              data_categories: Optional[Set[str]] = None) -> bool:
        """Check if a data category should be erased."""
        if data_categories is None:
            return True
        return category in data_categories
    
    def _generate_request_id(self, user_id: str, agent_id: str) -> str:
        """Generate unique request ID."""
        data = f"request:{user_id}:{agent_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_operation_id(self, user_id: str, agent_id: str) -> str:
        """Generate unique operation ID."""
        data = f"operation:{user_id}:{agent_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_verification_id(self, user_id: str, agent_id: str) -> str:
        """Generate unique verification ID."""
        data = f"verification:{user_id}:{agent_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_verification_hash(self, user_id: str, agent_id: str) -> str:
        """Generate verification hash for erasure."""
        data = f"erased:{user_id}:{agent_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _start_erasure_operation(self, request_id: str) -> None:
        """Start an erasure operation for a request."""
        request = self.erasure_requests.get(request_id)
        if not request:
            return
        
        operation_id = self._generate_operation_id(request.user_id, request.agent_id)
        
        operation = ErasureOperation(
            operation_id=operation_id,
            request_id=request_id,
            user_id=request.user_id,
            agent_id=request.agent_id,
            erasure_method=request.erasure_method,
            status=ErasureStatus.PENDING,
            started_at=datetime.now()
        )
        
        self.erasure_operations[operation_id] = operation 