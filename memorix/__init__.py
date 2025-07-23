"""
Memorix SDK - A flexible memory management system for AI applications.
"""

__version__ = "0.1.0"
__author__ = "Memorix Team"

from .memory_api import MemoryAPI
from .config import Config
from .vector_store import VectorStore
from .embedder import Embedder
from .metadata_store import MetadataStore

__all__ = [
    "MemoryAPI",
    "Config", 
    "VectorStore",
    "Embedder",
    "MetadataStore"
] 