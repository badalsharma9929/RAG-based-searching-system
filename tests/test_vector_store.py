"""
Tests for vector store module.
"""

import pytest
import numpy as np
from src.vector_store import VectorStore


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def vector_store(self):
        """Create vector store for testing."""
        return VectorStore(dimension=10, metric="ip")

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        documents = [
            "Document 1 about machine learning",
            "Document 2 about deep learning",
            "Document 3 about neural networks",
            "Document 4 about the weather",
            "Document 5 about cooking food"
        ]
        embeddings = np.random.randn(5, 10).astype(np.float32)
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        return documents, embeddings, doc_ids

    def test_initialization(self):
        """Test vector store initializes correctly."""
        store = VectorStore(dimension=128)
        assert store.dimension == 128
        assert store.n_vectors == 0

    def test_add_documents(self, vector_store, sample_data):
        """Test adding documents to store."""
        documents, embeddings, doc_ids = sample_data

        vector_store.add_documents(documents, embeddings, doc_ids)

        assert vector_store.n_vectors == 5

    def test_search_returns_correct_count(self, vector_store, sample_data):
        """Test search returns correct number of results."""
        documents, embeddings, doc_ids = sample_data
        vector_store.add_documents(documents, embeddings, doc_ids)

        query = embeddings[0].reshape(1, -1)
        distances, indices = vector_store.search(query, top_k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_search_with_text(self, vector_store, sample_data):
        """Test search_with_text returns full results."""
        documents, embeddings, doc_ids = sample_data
        vector_store.add_documents(documents, embeddings, doc_ids)

        query = embeddings[0].reshape(1, -1)
        results = vector_store.search_with_text(query, top_k=3)

        assert len(results) == 3
        assert "doc_id" in results[0]
        assert "text" in results[0]
        assert "distance" in results[0]
        assert "rank" in results[0]

    def test_search_top_k_limited(self, vector_store, sample_data):
        """Test that top_k limits results correctly."""
        documents, embeddings, doc_ids = sample_data
        vector_store.add_documents(documents, embeddings, doc_ids)

        query = embeddings[0].reshape(1, -1)
        results = vector_store.search_with_text(query, top_k=2)

        assert len(results) == 2

    def test_get_document(self, vector_store, sample_data):
        """Test retrieving document by ID."""
        documents, embeddings, doc_ids = sample_data
        vector_store.add_documents(documents, embeddings, doc_ids)

        doc = vector_store.get_document("doc_1")
        assert doc == "Document 1 about machine learning"

    def test_get_document_not_found(self, vector_store):
        """Test getting non-existent document."""
        doc = vector_store.get_document("nonexistent")
        assert doc is None

    def test_get_stats(self, vector_store, sample_data):
        """Test stats return correct information."""
        documents, embeddings, doc_ids = sample_data
        vector_store.add_documents(documents, embeddings, doc_ids)

        stats = vector_store.get_stats()

        assert stats["n_documents"] == 5
        assert stats["dimension"] == 10
        assert stats["index_type"] == "flat"

    def test_reset(self, vector_store, sample_data):
        """Test resetting the store."""
        documents, embeddings, doc_ids = sample_data
        vector_store.add_documents(documents, embeddings, doc_ids)

        assert vector_store.n_vectors == 5

        vector_store.reset()

        assert vector_store.n_vectors == 0

    def test_mismatched_dimensions_raises_error(self):
        """Test that mismatched dimensions raise error."""
        store = VectorStore(dimension=10)
        embeddings = np.random.randn(3, 20).astype(np.float32)  # Wrong dimension
        documents = ["doc1", "doc2", "doc3"]

        with pytest.raises(ValueError, match="dimension"):
            store.add_documents(documents, embeddings)

    def test_mismatched_doc_count_raises_error(self):
        """Test that mismatched doc/embedding counts raise error."""
        store = VectorStore(dimension=10)
        embeddings = np.random.randn(3, 10).astype(np.float32)
        documents = ["doc1", "doc2"]  # Only 2 docs

        with pytest.raises(ValueError, match="must match"):
            store.add_documents(documents, embeddings)


class TestVectorStoreL2:
    """Tests for VectorStore with L2 metric."""

    def test_l2_metric_search(self):
        """Test search with L2 distance metric."""
        store = VectorStore(dimension=5, metric="l2")

        # Add two close vectors
        v1 = np.array([1, 0, 0, 0, 0], dtype=np.float32)
        v2 = np.array([1.1, 0, 0, 0, 0], dtype=np.float32)  # Close to v1

        store.add_documents(["doc1", "doc2"], np.array([v1, v2]), ["d1", "d2"])

        # Search with v1 - d1 should be most similar (distance 0)
        results = store.search_with_text(v1.reshape(1, -1), top_k=2)

        assert results[0]["doc_id"] == "d1"  # Should be most similar
        assert results[0]["distance"] < results[1]["distance"]  # Smaller distance = more similar