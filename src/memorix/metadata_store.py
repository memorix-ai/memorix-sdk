"""
Optional metadata handling for Memorix SDK.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .config import Config

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class MetadataStoreInterface(ABC):
    """
    Abstract interface for metadata stores.
    """

    @abstractmethod
    def store(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a memory."""
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a memory."""
        pass

    @abstractmethod
    def update(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a memory."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete metadata for a memory."""
        pass

    @abstractmethod
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all metadata."""
        pass


class SQLiteMetadataStore(MetadataStoreInterface):
    """
    SQLite-based metadata store implementation.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.database_path = config.get(
            "metadata_store.database_path", "./memorix_metadata.db"
        )
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                memory_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def store(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a memory."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        metadata_json = json.dumps(metadata)

        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (memory_id, metadata_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """,
            (memory_id, metadata_json),
        )

        conn.commit()
        conn.close()

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a memory."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT metadata_json FROM metadata WHERE memory_id = ?", (memory_id,)
        )
        result = cursor.fetchone()

        conn.close()

        if result:
            return json.loads(result[0])
        return None

    def update(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a memory."""
        existing = self.get(memory_id)
        if existing:
            # Merge with existing metadata
            existing.update(metadata)
            self.store(memory_id, existing)
        else:
            self.store(memory_id, metadata)

    def delete(self, memory_id: str) -> None:
        """Delete metadata for a memory."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM metadata WHERE memory_id = ?", (memory_id,))

        conn.commit()
        conn.close()

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all metadata."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute("SELECT memory_id, metadata_json FROM metadata")
        results = cursor.fetchall()

        conn.close()

        metadata_dict = {}
        for memory_id, metadata_json in results:
            metadata_dict[memory_id] = json.loads(metadata_json)

        return metadata_dict


class InMemoryMetadataStore(MetadataStoreInterface):
    """
    In-memory metadata store implementation.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.metadata = {}

    def store(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a memory."""
        self.metadata[memory_id] = metadata.copy()

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a memory."""
        return self.metadata.get(memory_id)

    def update(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a memory."""
        if memory_id in self.metadata:
            self.metadata[memory_id].update(metadata)
        else:
            self.metadata[memory_id] = metadata.copy()

    def delete(self, memory_id: str) -> None:
        """Delete metadata for a memory."""
        if memory_id in self.metadata:
            del self.metadata[memory_id]

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all metadata."""
        return self.metadata.copy()


class JSONFileMetadataStore(MetadataStoreInterface):
    """
    JSON file-based metadata store implementation.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.file_path = config.get(
            "metadata_store.file_path", "./memorix_metadata.json"
        )
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from JSON file."""
        if Path(self.file_path).exists():
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def store(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a memory."""
        self.metadata[memory_id] = metadata.copy()
        self._save_metadata()

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a memory."""
        return self.metadata.get(memory_id)

    def update(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a memory."""
        if memory_id in self.metadata:
            self.metadata[memory_id].update(metadata)
        else:
            self.metadata[memory_id] = metadata.copy()
        self._save_metadata()

    def delete(self, memory_id: str) -> None:
        """Delete metadata for a memory."""
        if memory_id in self.metadata:
            del self.metadata[memory_id]
            self._save_metadata()

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all metadata."""
        return self.metadata.copy()


class MetadataStore:
    """
    Metadata store factory and interface.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.store_type = config.get("metadata_store.type", "sqlite")
        self._store = self._create_store()

    def _create_store(self) -> MetadataStoreInterface:
        """Create metadata store based on configuration."""
        if self.store_type == "sqlite":
            return SQLiteMetadataStore(self.config)
        elif self.store_type == "memory":
            return InMemoryMetadataStore(self.config)
        elif self.store_type == "json":
            return JSONFileMetadataStore(self.config)
        else:
            raise ValueError(f"Unsupported metadata store type: {self.store_type}")

    def store(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a memory."""
        self._store.store(memory_id, metadata)

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a memory."""
        return self._store.get(memory_id)

    def update(self, memory_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a memory."""
        self._store.update(memory_id, metadata)

    def delete(self, memory_id: str) -> None:
        """Delete metadata for a memory."""
        self._store.delete(memory_id)

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all metadata."""
        return self._store.list_all()
