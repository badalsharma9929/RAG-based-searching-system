"""
Embedding module using sentence-transformers with BGE-M3 model.

BGE-M3 is a state-of-the-art embedding model that supports:
- Dense retrieval
- Sparse retrieval (BM25-like)
- ColBERT-style late interaction

This implementation uses the dense embedding mode.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from . import config


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for generating embeddings.

    Uses BAAI/bge-m3 model which achieves top MTEB scores among
    open-source embedding models.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        device: str = config.EMBEDDING_DEVICE,
        normalize: bool = True
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ("cpu" or "cuda")
            normalize: Whether to normalize embeddings (recommended for cosine sim)
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first access."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model loaded successfully!")
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return config.VECTOR_DIMENSION

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show encoding progress

        Returns:
            numpy array of embeddings with shape (n_texts, dimension)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Encode with sentence-transformers
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )

        return embeddings

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode search queries.

        Args:
            queries: List of query strings

        Returns:
            numpy array of query embeddings
        """
        return self.encode(queries, show_progress=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode document chunks.

        Args:
            documents: List of document strings

        Returns:
            numpy array of document embeddings
        """
        return self.encode(documents, show_progress=True)

    def compute_similarity(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between queries and documents.

        Since embeddings are normalized, dot product equals cosine similarity.

        Args:
            query_embeddings: Query vectors (n_queries, dim)
            doc_embeddings: Document vectors (n_docs, dim)

        Returns:
            Similarity matrix (n_queries, n_docs)
        """
        return np.dot(query_embeddings, doc_embeddings.T)

    def get_model_info(self) -> dict:
        """Return information about the model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "normalize": self.normalize
        }


class GCPEmbeddingMock:
    """
    Mock implementation of Google Cloud Vertex AI TextEmbeddingModel.

    This simulates the behavior of GCP's textembedding-gecko model
    for local development and testing without requiring GCP credentials.

    The mock generates deterministic embeddings based on hash of input text,
    ensuring reproducibility while simulating the API interface.
    """

    def __init__(self, model_name: str = config.GCP_EMBEDDING_MODEL_NAME):
        """
        Initialize the GCP mock.

        Args:
            model_name: Name of the mocked GCP model
        """
        self.model_name = model_name
        self.dimension = config.GCP_EMBEDDING_MODEL_NAME

    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic mock embedding based on text hash.

        This creates reproducible embeddings that are semantically
        consistent (same text -> same embedding) but not actual
        semantic embeddings.

        Args:
            text: Input text string

        Returns:
            Mock embedding vector
        """
        # Use hash for deterministic but varied embeddings
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype(np.float32)

        # Normalize to unit vector (like real embedding models)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Mock implementation of GCP's get_embeddings method.

        Args:
            texts: Single text or list of texts

        Returns:
            List of embedding arrays
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            emb = self._generate_mock_embedding(text)
            embeddings.append(emb.tolist())

        return embeddings

    def embed_text(self, text: str) -> List[float]:
        """
        Mock implementation of GCP's embed_text method.

        Args:
            text: Input text string

        Returns:
            Embedding vector as list
        """
        return self._generate_mock_embedding(text).tolist()

    def get_model_info(self) -> dict:
        """Return mock model information."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "type": "mock",
            "description": "Mock of GCP textembedding-gecko"
        }