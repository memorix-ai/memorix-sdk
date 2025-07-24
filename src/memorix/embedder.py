"""
Embedding interface with support for Gemini, OpenAI, etc.
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .config import Config

from abc import ABC, abstractmethod


class EmbedderInterface(ABC):
    """
    Abstract interface for embedding models.
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbedder(EmbedderInterface):
    """
    OpenAI embedding model implementation.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.model = config.get("embedder.model", "text-embedding-ada-002")
        self.api_key = config.get("embedder.api_key")

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Placeholder for OpenAI client
        self.client = None  # Would be initialized with openai.OpenAI(api_key=self.api_key)  # noqa: E501

    def embed(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        # Placeholder implementation
        # In real implementation, would call OpenAI API
        return self._dummy_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def _dummy_embedding(self, text: str) -> List[float]:
        """Generate dummy embedding for demo purposes."""
        # Simple hash-based embedding for demo
        import hashlib

        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()

        # Convert hex to list of floats
        embedding: List[float] = []
        for i in range(0, len(hash_hex), 2):
            if len(embedding) >= 1536:  # OpenAI embedding dimension
                break
            hex_pair = hash_hex[i : i + 2]  # noqa: E203
            embedding.append(float(int(hex_pair, 16)) / 255.0)

        # Pad or truncate to 1536 dimensions
        while len(embedding) < 1536:
            embedding.append(0.0)

        return embedding[:1536]


class GeminiEmbedder(EmbedderInterface):
    """
    Google Gemini embedding model implementation.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.model = config.get("embedder.model", "models/embedding-001")
        self.api_key = config.get("embedder.api_key")

        if not self.api_key:
            raise ValueError("Gemini API key is required")

        # Placeholder for Gemini client
        self.client = None  # Would be initialized with google.generativeai

    def embed(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        # Placeholder implementation
        # In real implementation, would call Gemini API
        return self._dummy_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def _dummy_embedding(self, text: str) -> List[float]:
        """Generate dummy embedding for demo purposes."""
        # Simple hash-based embedding for demo
        import hashlib

        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()

        # Convert hex to list of floats
        embedding: List[float] = []
        for i in range(0, len(hash_hex), 2):
            if len(embedding) >= 768:  # Gemini embedding dimension
                break
            hex_pair = hash_hex[i : i + 2]  # noqa: E203
            embedding.append(float(int(hex_pair, 16)) / 255.0)

        # Pad or truncate to 768 dimensions
        while len(embedding) < 768:
            embedding.append(0.0)

        return embedding[:768]


class SentenceTransformersEmbedder(EmbedderInterface):
    """
    Sentence Transformers embedding model implementation.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.model_name = config.get("embedder.model", "all-MiniLM-L6-v2")

        # Placeholder for sentence transformers
        self.model = None  # Would be initialized with SentenceTransformer(self.model_name)  # noqa: E501

    def embed(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        # Placeholder implementation
        # In real implementation, would use sentence transformers
        return self._dummy_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def _dummy_embedding(self, text: str) -> List[float]:
        """Generate dummy embedding for demo purposes."""
        # Simple hash-based embedding for demo
        import hashlib

        hash_obj = hashlib.sha1(text.encode())
        hash_hex = hash_obj.hexdigest()

        # Convert hex to list of floats
        embedding: List[float] = []
        for i in range(0, len(hash_hex), 2):
            if len(embedding) >= 384:  # Common sentence transformer dimension
                break
            hex_pair = hash_hex[i : i + 2]  # noqa: E203
            embedding.append(float(int(hex_pair, 16)) / 255.0)

        # Pad or truncate to 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)

        return embedding[:384]


class Embedder:
    """
    Embedding factory and interface.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.embedder_type = config.get("embedder.type", "openai")
        self._embedder = self._create_embedder()

    def _create_embedder(self) -> EmbedderInterface:
        """Create embedder based on configuration."""
        if self.embedder_type == "openai":
            return OpenAIEmbedder(self.config)
        elif self.embedder_type == "gemini":
            return GeminiEmbedder(self.config)
        elif self.embedder_type == "sentence_transformers":
            return SentenceTransformersEmbedder(self.config)
        else:
            raise ValueError(
                f"Unsupported embedder type: {self.embedder_type}"
            )  # noqa: E501

    def embed(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        return self._embedder.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self._embedder.embed_batch(texts)
