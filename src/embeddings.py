"""
Embedding module using transformers with all-MiniLM-L6-v2 model.

This implementation uses the HuggingFace transformers library directly
to avoid dependency issues with sentence-transformers.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModel

from . import config


class EmbeddingModel:
    """
    Wrapper around transformers for generating embeddings.

    Uses all-MiniLM-L6-v2 model - a fast, lightweight sentence embedding model.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        device: str = "cpu",
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
        self._tokenizer = None

    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            print(f"Model loaded successfully!")
        return self._model

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return config.VECTOR_DIMENSION

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
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
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)
                embeddings = self._mean_pooling(output, encoded['attention_mask'])

            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

            if show_progress:
                print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")

        return np.vstack(all_embeddings)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode search queries."""
        return self.encode(queries, show_progress=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode document chunks."""
        return self.encode(documents, show_progress=True)

    def compute_similarity(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between queries and documents."""
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
        self.dimension = config.GCP_EMBEDDING_DIMENSION

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