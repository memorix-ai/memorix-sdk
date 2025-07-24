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
Privacy-Compliant AI Agent Example

This example demonstrates a complete privacy-compliant AI agent that:
1. Manages user consent for GDPR compliance
2. Implements hierarchical memory with FIFO flow
3. Uses secure data erasure capabilities
4. Maintains comprehensive audit trails
5. Supports pluggable storage backends
6. Implements adaptive decay and scoring

This example showcases the complete Memorix SDK v0.3.0 capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from memorix import (
    # Core components
    MemoryAPI, ConfigManager,
    
    # Hierarchical memory
    MemoryTier, DecayStrategy, ScoringStrategy,
    
    # Memory update system
    MemoryUpdateManager, UpdateMetrics, FIFOBuffer,
    FrequencyBasedDecay, ContextAwareDecay,
    
    # Storage plugins
    StoragePlugin, VectorStoreInterface, MetadataStoreInterface,
    StorageConfig, StoragePluginRegistry, plugin_registry,
    
    # Privacy and consent
    ConsentStatus, DataCategory, ProcessingPurpose,
    ConsentRecord, PrivacyPolicy, GDPRConsentValidator,
    PrivacyPolicyLoader, ErasureMethod, SecureDataEraser,
    
    # Audit and logging
    AuditLevel, AuditEventType, AuditCategory,
    AuditManager, SQLiteAuditLogger
)


class PrivacyCompliantAgent:
    """
    A privacy-compliant AI agent that demonstrates all Memorix SDK features.
    """
    
    def __init__(self, config_path: str = "privacy_agent_config.yaml"):
        """Initialize the privacy-compliant agent."""
        self.config = ConfigManager(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.memory_api = MemoryAPI(self.config)
        
        # Initialize privacy components
        self._setup_privacy_system()
        
        # Initialize audit system
        self._setup_audit_system()
        
        # Initialize memory update system
        self._setup_memory_update_system()
        
        # Initialize storage plugins
        self._setup_storage_plugins()
        
        self.logger.info("Privacy-compliant agent initialized successfully")
    
    def _setup_privacy_system(self):
        """Setup privacy and consent management system."""
        # Create default privacy policy
        self.privacy_policy = PrivacyPolicyLoader.create_default_policy()
        
        # Initialize GDPR consent validator
        self.consent_validator = GDPRConsentValidator(self.privacy_policy)
        
        # Initialize secure data eraser
        self.data_eraser = SecureDataEraser(
            vector_store=self.memory_api.vector_store,
            metadata_store=self.memory_api.metadata_store,
            tier_manager=self.memory_api.tier_manager,
            consent_validator=self.consent_validator,
            config=self.config.get_all()
        )
        
        self.logger.info("Privacy system initialized")
    
    def _setup_audit_system(self):
        """Setup audit and logging system."""
        self.audit_manager = AuditManager(self.config.get_all())
        
        # Log system startup
        self.audit_manager.log_security_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            description="Privacy-compliant agent started",
            level=AuditLevel.INFO
        )
        
        self.logger.info("Audit system initialized")
    
    def _setup_memory_update_system(self):
        """Setup memory update system with FIFO flow."""
        self.update_manager = MemoryUpdateManager(
            tier_manager=self.memory_api.tier_manager,
            config=self.config.get_all()
        )
        
        self.logger.info("Memory update system initialized")
    
    def _setup_storage_plugins(self):
        """Setup storage plugin registry."""
        # Register default plugins (these would be actual implementations)
        # For demo purposes, we'll just show the structure
        
        self.logger.info("Storage plugin registry initialized")
    
    def request_consent(self, user_id: str, agent_id: str, 
                       data_categories: List[str] = None,
                       processing_purposes: List[str] = None) -> bool:
        """
        Request user consent for data processing.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            data_categories: List of data categories to process
            processing_purposes: List of processing purposes
            
        Returns:
            True if consent was granted
        """
        if data_categories is None:
            data_categories = [
                DataCategory.PERSONAL_DATA.value,
                DataCategory.BEHAVIORAL_DATA.value,
                DataCategory.TECHNICAL_DATA.value
            ]
        
        if processing_purposes is None:
            processing_purposes = [
                ProcessingPurpose.MEMORY_STORAGE.value,
                ProcessingPurpose.SIMILARITY_SEARCH.value,
                ProcessingPurpose.CUSTOMIZATION.value
            ]
        
        # Convert to enums
        data_cat_enums = {DataCategory(cat) for cat in data_categories}
        purpose_enums = {ProcessingPurpose(purpose) for purpose in processing_purposes}
        
        # Grant consent (in real app, this would be user interaction)
        consent_record = self.consent_validator.grant_consent(
            user_id=user_id,
            agent_id=agent_id,
            data_categories=data_cat_enums,
            processing_purposes=purpose_enums,
            expires_at=datetime.now() + timedelta(days=365),
            ip_address="127.0.0.1",
            user_agent="PrivacyCompliantAgent/1.0"
        )
        
        # Log consent event
        self.audit_manager.log_privacy_operation(
            operation_type=AuditEventType.CONSENT_GRANTED,
            user_id=user_id,
            agent_id=agent_id,
            description=f"Consent granted for {len(data_categories)} data categories",
            details={
                "data_categories": data_categories,
                "processing_purposes": processing_purposes,
                "consent_id": consent_record.consent_id
            }
        )
        
        self.logger.info(f"Consent granted for user {user_id}, agent {agent_id}")
        return True
    
    def store_memory(self, user_id: str, agent_id: str, content: str,
                    metadata: Optional[Dict] = None) -> str:
        """
        Store memory with privacy compliance checks.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Memory ID if successful
        """
        # Check consent before processing
        if not self.consent_validator.validate_consent(
            user_id, agent_id, 
            DataCategory.PERSONAL_DATA, 
            ProcessingPurpose.MEMORY_STORAGE
        ):
            self.audit_manager.log_security_event(
                event_type=AuditEventType.ACCESS_DENIED,
                user_id=user_id,
                agent_id=agent_id,
                description="Memory storage denied - no valid consent",
                level=AuditLevel.WARNING
            )
            raise PermissionError("No valid consent for memory storage")
        
        # Store memory
        memory_id = self.memory_api.store(
            content=content,
            metadata=metadata or {},
            importance_score=0.8,
            decay_strategy=DecayStrategy.EXPONENTIAL,
            scoring_strategy=ScoringStrategy.HYBRID
        )
        
        # Log memory operation
        self.audit_manager.log_memory_operation(
            operation_type=AuditEventType.MEMORY_STORE,
            user_id=user_id,
            agent_id=agent_id,
            memory_id=memory_id,
            description=f"Memory stored: {content[:50]}...",
            details={"content_length": len(content), "metadata": metadata}
        )
        
        # Update memory with adaptive decay
        self.update_manager.update_memory(
            memory_id=memory_id,
            access_context={"user_id": user_id, "agent_id": agent_id}
        )
        
        self.logger.info(f"Memory stored successfully: {memory_id}")
        return memory_id
    
    def retrieve_memory(self, user_id: str, agent_id: str, query: str,
                       limit: int = 5) -> List[Dict]:
        """
        Retrieve memories with privacy compliance checks.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of memory results
        """
        # Check consent before retrieval
        if not self.consent_validator.validate_consent(
            user_id, agent_id,
            DataCategory.PERSONAL_DATA,
            ProcessingPurpose.SIMILARITY_SEARCH
        ):
            self.audit_manager.log_security_event(
                event_type=AuditEventType.ACCESS_DENIED,
                user_id=user_id,
                agent_id=agent_id,
                description="Memory retrieval denied - no valid consent",
                level=AuditLevel.WARNING
            )
            raise PermissionError("No valid consent for memory retrieval")
        
        # Retrieve memories
        results = self.memory_api.retrieve(query, limit=limit)
        
        # Log retrieval operation
        self.audit_manager.log_memory_operation(
            operation_type=AuditEventType.MEMORY_RETRIEVE,
            user_id=user_id,
            agent_id=agent_id,
            memory_id="query",
            description=f"Memory retrieved for query: {query}",
            details={"query": query, "results_count": len(results)}
        )
        
        # Update access patterns for adaptive decay
        for result in results:
            if "memory_id" in result:
                self.update_manager.update_memory(
                    memory_id=result["memory_id"],
                    access_context={"user_id": user_id, "agent_id": agent_id, "query": query}
                )
        
        self.logger.info(f"Retrieved {len(results)} memories for query: {query}")
        return results
    
    def erase_user_data(self, user_id: str, agent_id: str,
                       erasure_method: ErasureMethod = ErasureMethod.HARD_DELETE) -> bool:
        """
        Erase all user data (GDPR right to be forgotten).
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            erasure_method: Method of erasure
            
        Returns:
            True if erasure was successful
        """
        # Log erasure request
        self.audit_manager.log_privacy_operation(
            operation_type=AuditEventType.DATA_ERASED,
            user_id=user_id,
            agent_id=agent_id,
            description=f"Data erasure requested using {erasure_method.value}",
            level=AuditLevel.INFO
        )
        
        # Perform erasure
        success = self.data_eraser.erase_user_data(
            user_id=user_id,
            agent_id=agent_id,
            erasure_method=erasure_method
        )
        
        if success:
            # Withdraw consent
            self.consent_validator.withdraw_consent(user_id, agent_id)
            
            self.logger.info(f"User data erased successfully for {user_id}:{agent_id}")
        else:
            self.logger.error(f"Failed to erase user data for {user_id}:{agent_id}")
        
        return success
    
    def get_memory_statistics(self) -> Dict:
        """Get comprehensive memory statistics."""
        stats = {
            "memory_api": self.memory_api.get_memory_statistics(),
            "update_manager": self.update_manager.get_update_statistics(),
            "consent": self.consent_validator.get_consent_statistics(),
            "erasure": self.data_eraser.get_erasure_statistics(),
            "audit": self.audit_manager.get_audit_report()
        }
        
        return stats
    
    def optimize_memory(self) -> Dict:
        """Optimize memory distribution across tiers."""
        # Optimize memory distribution
        optimization_stats = self.update_manager.optimize_memory_distribution()
        
        # Clean up expired data
        expired_consents = self.consent_validator.cleanup_expired_consents()
        expired_audit = self.audit_manager.audit_logger.cleanup_old_events()
        
        return {
            "memory_optimization": optimization_stats,
            "expired_consents_cleaned": expired_consents,
            "expired_audit_events_cleaned": expired_audit
        }
    
    def export_audit_report(self, file_path: str, format: str = "json") -> bool:
        """Export comprehensive audit report."""
        return self.audit_manager.audit_logger.export_audit_log(
            file_path=file_path,
            format=format
        )
    
    def shutdown(self):
        """Gracefully shutdown the agent."""
        # Log system shutdown
        self.audit_manager.log_security_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            description="Privacy-compliant agent shutting down",
            level=AuditLevel.INFO
        )
        
        # Optimize memory before shutdown
        self.optimize_memory()
        
        self.logger.info("Privacy-compliant agent shutdown complete")


def create_privacy_config() -> str:
    """Create a privacy-compliant configuration file."""
    config = {
        "memory": {
            "stm_capacity": 100,
            "mtm_capacity": 1000,
            "ltm_capacity": 100000,
            "default_decay_strategy": "exponential",
            "default_scoring_strategy": "hybrid",
            "adaptive_decay_type": "context",
            "max_update_metrics": 10000
        },
        "privacy": {
            "retention_period_days": 30,
            "verification_required": True,
            "audit_retention_days": 2555,
            "overwrite_passes": 3
        },
        "audit": {
            "enable_audit": True,
            "level": "info",
            "include_metadata": True,
            "include_ip_address": True,
            "include_user_agent": True,
            "database_path": "privacy_agent_audit.db",
            "retention_days": 2555,
            "max_events": 1000000
        },
        "vector_store": {
            "type": "faiss",
            "config": {
                "index_type": "Flat",
                "dimension": 1536
            }
        },
        "embedder": {
            "type": "openai",
            "config": {
                "model": "text-embedding-ada-002",
                "api_key": "${OPENAI_API_KEY}"
            }
        },
        "metadata_store": {
            "type": "sqlite",
            "config": {
                "database_path": "privacy_agent_metadata.db"
            }
        }
    }
    
    config_path = "privacy_agent_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def main():
    """Main demonstration function."""
    print("🧠 Privacy-Compliant AI Agent Demo")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config_path = create_privacy_config()
    print(f"✅ Created configuration: {config_path}")
    
    # Initialize agent
    agent = PrivacyCompliantAgent(config_path)
    print("✅ Privacy-compliant agent initialized")
    
    # Demo user interaction
    user_id = "user_123"
    agent_id = "assistant_456"
    
    print(f"\n👤 User: {user_id}")
    print(f"🤖 Agent: {agent_id}")
    
    # Request consent
    print("\n📋 Requesting user consent...")
    consent_granted = agent.request_consent(user_id, agent_id)
    if consent_granted:
        print("✅ Consent granted")
    else:
        print("❌ Consent denied")
        return
    
    # Store memories
    print("\n💾 Storing memories...")
    memories = [
        "User prefers dark mode interface",
        "User frequently asks about weather",
        "User is interested in machine learning",
        "User's favorite programming language is Python",
        "User works as a software engineer"
    ]
    
    memory_ids = []
    for i, memory in enumerate(memories):
        memory_id = agent.store_memory(
            user_id=user_id,
            agent_id=agent_id,
            content=memory,
            metadata={"category": "preference", "priority": i + 1}
        )
        memory_ids.append(memory_id)
        print(f"  ✅ Stored: {memory[:30]}...")
    
    # Retrieve memories
    print("\n🔍 Retrieving memories...")
    query = "user preferences"
    results = agent.retrieve_memory(user_id, agent_id, query, limit=3)
    
    print(f"Query: '{query}'")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.get('content', 'N/A')[:50]}...")
    
    # Show statistics
    print("\n📊 Memory Statistics:")
    stats = agent.get_memory_statistics()
    print(f"  Total memories: {stats['memory_api']['total_memories']}")
    print(f"  STM: {stats['memory_api']['tier_distribution']['stm']}")
    print(f"  MTM: {stats['memory_api']['tier_distribution']['mtm']}")
    print(f"  LTM: {stats['memory_api']['tier_distribution']['ltm']}")
    print(f"  Valid consents: {stats['consent']['valid_consents']}")
    print(f"  Audit events: {stats['audit']['statistics']['total_events']}")
    
    # Optimize memory
    print("\n⚡ Optimizing memory...")
    optimization = agent.optimize_memory()
    print(f"  Memory migrations: {optimization['memory_optimization']['total_migrations']}")
    print(f"  Expired consents cleaned: {optimization['expired_consents_cleaned']}")
    
    # Export audit report
    print("\n📄 Exporting audit report...")
    agent.export_audit_report("privacy_agent_audit_report.json")
    print("✅ Audit report exported")
    
    # Demonstrate data erasure (GDPR right to be forgotten)
    print("\n🗑️ Demonstrating data erasure (GDPR right to be forgotten)...")
    erase_success = agent.erase_user_data(user_id, agent_id)
    if erase_success:
        print("✅ User data erased successfully")
    else:
        print("❌ Failed to erase user data")
    
    # Final statistics
    print("\n📊 Final Statistics:")
    final_stats = agent.get_memory_statistics()
    print(f"  Remaining memories: {final_stats['memory_api']['total_memories']}")
    print(f"  Valid consents: {final_stats['consent']['valid_consents']}")
    
    # Shutdown
    print("\n🔄 Shutting down...")
    agent.shutdown()
    print("✅ Demo completed successfully!")
    
    print("\n🎉 Privacy-Compliant Agent Demo Complete!")
    print("This demo showcased:")
    print("  • GDPR consent management")
    print("  • Hierarchical memory with FIFO flow")
    print("  • Adaptive decay and scoring")
    print("  • Comprehensive audit trails")
    print("  • Secure data erasure")
    print("  • Privacy-compliant operations")


if __name__ == "__main__":
    main() 