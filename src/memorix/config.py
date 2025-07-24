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
YAML-based configuration loader for Memorix SDK with hierarchical memory support.
"""

from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
from pydantic import BaseModel, Field

from .memory_hierarchy import DecayStrategy, ScoringStrategy, MemoryTier


class MemoryConfig(BaseModel):
    """Configuration for hierarchical memory system."""
    
    # Tier capacities
    stm_capacity: int = Field(default=100, description="Short-term memory capacity")
    mtm_capacity: int = Field(default=1000, description="Medium-term memory capacity")
    ltm_capacity: int = Field(default=100000, description="Long-term memory capacity")
    
    # Default decay settings
    default_decay_strategy: DecayStrategy = Field(
        default=DecayStrategy.EXPONENTIAL,
        description="Default decay strategy for new memories"
    )
    default_decay_rate: float = Field(
        default=0.1,
        description="Default decay rate (0.0 to 1.0)"
    )
    
    # Default scoring settings
    default_scoring_strategy: ScoringStrategy = Field(
        default=ScoringStrategy.HYBRID,
        description="Default scoring strategy for new memories"
    )
    
    # Hybrid scoring weights
    hybrid_weights: Dict[str, float] = Field(
        default={"frequency": 0.3, "recency": 0.4, "importance": 0.3},
        description="Weights for hybrid scoring strategy"
    )
    
    # Cleanup thresholds
    stm_cleanup_threshold: float = Field(
        default=0.1,
        description="Threshold for STM cleanup (0.0 to 1.0)"
    )
    mtm_cleanup_threshold: float = Field(
        default=0.3,
        description="Threshold for MTM cleanup (0.0 to 1.0)"
    )
    ltm_cleanup_threshold: float = Field(
        default=0.5,
        description="Threshold for LTM cleanup (0.0 to 1.0)"
    )


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    
    type: str = Field(default="faiss", description="Vector store type")
    index_path: str = Field(default="./memorix_index", description="Index storage path")
    dimension: int = Field(default=1536, description="Embedding dimension")
    
    # Additional vector store options
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold for retrieval"
    )
    max_results: int = Field(
        default=100,
        description="Maximum number of results to return"
    )


class EmbedderConfig(BaseModel):
    """Configuration for embedding models."""
    
    type: str = Field(default="openai", description="Embedder type")
    model: str = Field(default="text-embedding-ada-002", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    
    # Additional embedder options
    batch_size: int = Field(default=32, description="Batch size for embeddings")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class MetadataStoreConfig(BaseModel):
    """Configuration for metadata store."""
    
    type: str = Field(default="sqlite", description="Metadata store type")
    database_path: str = Field(default="./memorix_metadata.db", description="Database path")
    
    # Additional metadata options
    auto_backup: bool = Field(default=True, description="Enable automatic backups")
    backup_interval: int = Field(default=3600, description="Backup interval in seconds")


class LoggingConfig(BaseModel):
    """Configuration for logging and tracing."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    # Tracing options
    enable_tracing: bool = Field(default=True, description="Enable operation tracing")
    trace_memory_operations: bool = Field(default=True, description="Trace memory operations")
    trace_tier_migrations: bool = Field(default=True, description="Trace tier migrations")
    trace_decay_calculations: bool = Field(default=False, description="Trace decay calculations")
    
    # Timeline hooks
    timeline_hooks: List[str] = Field(
        default=["store", "retrieve", "update", "delete", "migrate"],
        description="Operations to hook into timeline"
    )


class Config(BaseModel):
    """Main configuration class for Memorix SDK."""
    
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    metadata_store: MetadataStoreConfig = Field(default_factory=MetadataStoreConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    max_memories: int = Field(default=10000, description="Maximum total memories")
    enable_hierarchical_memory: bool = Field(default=True, description="Enable hierarchical memory")
    auto_cleanup: bool = Field(default=True, description="Enable automatic cleanup")
    cleanup_interval: int = Field(default=300, description="Cleanup interval in seconds")


class ConfigManager:
    """
    Enhanced configuration manager for Memorix SDK with hierarchical memory support.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file or use defaults.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Config:
        """
        Load configuration from YAML file or return defaults.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration object
        """
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data is None:
                        return Config()
                    return Config(**yaml_data)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                return Config()
        else:
            return Config()

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
        value = self.config.dict()

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
        config_dict = self.config.dict()
        
        # Navigate to the parent of the target key
        current = config_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
        
        # Recreate the config object
        self.config = Config(**config_dict)

    def save(self, config_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save configuration
        """
        with open(config_path, "w") as f:
            yaml.dump(self.config.dict(), f, default_flow_style=False, indent=2)

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Validate using Pydantic
            self.config.validate()
            return True
        except Exception:
            return False

    def get_memory_config(self) -> Dict[str, Any]:
        """
        Get memory configuration as dictionary.

        Returns:
            Memory configuration dictionary
        """
        return self.config.memory.dict()

    def get_vector_store_config(self) -> Dict[str, Any]:
        """
        Get vector store configuration as dictionary.

        Returns:
            Vector store configuration dictionary
        """
        return self.config.vector_store.dict()

    def get_embedder_config(self) -> Dict[str, Any]:
        """
        Get embedder configuration as dictionary.

        Returns:
            Embedder configuration dictionary
        """
        return self.config.embedder.dict()

    def get_metadata_store_config(self) -> Dict[str, Any]:
        """
        Get metadata store configuration as dictionary.

        Returns:
            Metadata store configuration dictionary
        """
        return self.config.metadata_store.dict()

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration as dictionary.

        Returns:
            Logging configuration dictionary
        """
        return self.config.logging.dict()

    def create_default_config_file(self, config_path: str) -> None:
        """
        Create a default configuration file.

        Args:
            config_path: Path to create the configuration file
        """
        default_config = Config()
        self.save(config_path)

    def merge_config(self, other_config: Dict[str, Any]) -> None:
        """
        Merge another configuration into the current one.

        Args:
            other_config: Configuration dictionary to merge
        """
        current_dict = self.config.dict()
        
        def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge two dictionaries."""
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(current_dict, other_config)
        self.config = Config(**merged_dict)


# Backward compatibility
class Config(ConfigManager):
    """Backward compatibility wrapper for the old Config class."""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for backward compatibility."""
        return self.config.dict()
