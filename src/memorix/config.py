"""
YAML-based configuration loader for Memorix SDK.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """
    Configuration manager for Memorix SDK.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file or use defaults.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from YAML file or return defaults.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "vector_store": {
                "type": "faiss",
                "index_path": "./memorix_index",
                "dimension": 1536,
            },
            "embedder": {
                "type": "openai",
                "model": "text-embedding-ada-002",
                "api_key": None,
            },
            "metadata_store": {
                "type": "sqlite",
                "database_path": "./memorix_metadata.db",
            },
            "settings": {"max_memories": 10000, "similarity_threshold": 0.7},  # noqa: E501
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, config_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save configuration
        """
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        required_keys = ["vector_store.type", "embedder.type", "metadata_store.type"]

        for key in required_keys:
            if self.get(key) is None:
                return False

        return True
