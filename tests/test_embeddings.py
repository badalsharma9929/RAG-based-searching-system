"""
Tests for embedding module.
"""

import pytest
import numpy as np
from src.embeddings import EmbeddingModel, GCPEmbeddingMock


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model for testing."""
        # Use a small, fast model for testing
        return EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

    def test_model_initialization(self, embedding_model):
        """Test model initializes correctly."""
        assert embedding_model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedding_model.dimension == 384

    def test_encode_single_text(self, embedding_model):
        """Test encoding a single text."""
        text = "This is a test sentence."
        embedding = embedding_model.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, embedding_model.dimension)

    def test_encode_multiple_texts(self, embedding_model):
        """Test encoding multiple texts."""
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        embeddings = embedding_model.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, embedding_model.dimension)

    def test_encode_returns_normalized(self, embedding_model):
        """Test that embeddings are normalized."""
        text = "Test sentence for normalization"
        embedding = embedding_model.encode(text)

        # Check L2 norm is close to 1 (normalized)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_compute_similarity(self, embedding_model):
        """Test similarity computation."""
        query = "What is machine learning?"
        doc1 = "Machine learning is a subset of AI."
        doc2 = "The weather is sunny today."

        query_emb = embedding_model.encode(query)
        doc1_emb = embedding_model.encode(doc1)
        doc2_emb = embedding_model.encode(doc2)

        # Similarity with related doc should be higher
        sim1 = embedding_model.compute_similarity(query_emb, doc1_emb.reshape(1, -1))
        sim2 = embedding_model.compute_similarity(query_emb, doc2_emb.reshape(1, -1))

        assert sim1[0][0] > sim2[0][0]

    def test_get_model_info(self, embedding_model):
        """Test model info returns correct data."""
        info = embedding_model.get_model_info()

        assert "model_name" in info
        assert "dimension" in info
        assert info["dimension"] == 384


class TestGCPEmbeddingMock:
    """Tests for GCPEmbeddingMock class."""

    @pytest.fixture
    def gcp_mock(self):
        """Create GCP mock for testing."""
        return GCPEmbeddingMock()

    def test_mock_initialization(self, gcp_mock):
        """Test mock initializes correctly."""
        assert gcp_mock.model_name == "textembedding-gecko"
        assert gcp_mock.dimension == 768

    def test_get_embeddings_single(self, gcp_mock):
        """Test getting embeddings for single text."""
        text = "Test sentence"
        embeddings = gcp_mock.get_embeddings(text)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768

    def test_get_embeddings_multiple(self, gcp_mock):
        """Test getting embeddings for multiple texts."""
        texts = ["First", "Second", "Third"]
        embeddings = gcp_mock.get_embeddings(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 768

    def test_mock_deterministic(self, gcp_mock):
        """Test that mock embeddings are deterministic."""
        text = "Same text"
        emb1 = gcp_mock.get_embeddings(text)
        emb2 = gcp_mock.get_embeddings(text)

        assert emb1 == emb2

    def test_embed_text(self, gcp_mock):
        """Test embed_text method."""
        text = "Test"
        embedding = gcp_mock.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_get_model_info(self, gcp_mock):
        """Test model info."""
        info = gcp_mock.get_model_info()

        assert info["type"] == "mock"
        assert info["dimension"] == 768