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
Privacy module for Memorix SDK.

This module contains GDPR-compliant privacy and consent management.
"""

from .consent import (
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

from .erase import (
    ErasureMethod,
    ErasureStatus,
    ErasureRequest,
    ErasureOperation,
    ErasureVerification,
    DataEraser,
    SecureDataEraser
)

__all__ = [
    # Consent management
    "ConsentStatus",
    "DataCategory", 
    "ProcessingPurpose",
    "ConsentRecord",
    "PrivacyPolicy",
    "PrivacyAuditEvent",
    "ConsentValidator",
    "GDPRConsentValidator",
    "PrivacyPolicyLoader",
    
    # Data erasure
    "ErasureMethod",
    "ErasureStatus",
    "ErasureRequest", 
    "ErasureOperation",
    "ErasureVerification",
    "DataEraser",
    "SecureDataEraser"
] 