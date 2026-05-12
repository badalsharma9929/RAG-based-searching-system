"""
Tests for retrieval module.
"""

import pytest
import numpy as np
from src.embeddings import EmbeddingModel, GCPEmbeddingMock
from src.vector_store import VectorStore
from src.retrieval import Retriever, GCPRetriever, create_retriever, create_gcp_retriever


class TestRetriever:
    """Tests for Retriever class."""

    @pytest.fixture
    def setup_retriever(self):
        """Setup retriever with sample data."""
        # Create embedding model
        embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

        # Create vector store
        vector_store = VectorStore(dimension=embedding_model.dimension)

        # Add sample documents
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a programming language.",
            "Neural networks are inspired by biological neurons.",
            "The weather is sunny today."
        ]
        embeddings = embedding_model.encode(documents)
        doc_ids = ["ml", "dl", "py", "nn", "weather"]
        vector_store.add_documents(documents, embeddings, doc_ids)

        # Create retriever
        retriever = Retriever(embedding_model, vector_store, top_k=3)

        return retriever, vector_store, embedding_model

    def test_retrieve_single_query(self, setup_retriever):
        """Test retrieving for a single query."""
        retriever, _, _ = setup_retriever

        query = "What is machine learning?"
        results = retriever.retrieve(query)

        assert len(results) <= 3
        assert all("doc_id" in r for r in results)
        assert all("text" in r for r in results)

    def test_retrieve_batch(self, setup_retriever):
        """Test batch retrieval."""
        retriever, _, _ = setup_retriever

        queries = [
            "What is machine learning?",
            "Tell me about neural networks"
        ]
        results = retriever.retrieve_batch(queries)

        assert len(results) == 2  # 2 queries
        assert all(len(r) <= 3 for r in results)  # max 3 results each

    def test_retrieve_with_different_top_k(self, setup_retriever):
        """Test retrieval with custom top_k."""
        retriever, _, _ = setup_retriever

        results = retriever.retrieve("test query", top_k=2)
        assert len(results) == 2

    def test_retrieve_returns_relevant_results(self, setup_retriever):
        """Test that retrieval returns semantically similar docs."""
        retriever, _, _ = setup_retriever

        # Query about ML should return ML doc first
        results = retriever.retrieve("artificial intelligence algorithms")

        # Check if ML-related doc is in results
        doc_ids = [r["doc_id"] for r in results]
        assert "ml" in doc_ids or "dl" in doc_ids or "nn" in doc_ids


class TestGCPRetriever:
    """Tests for GCPRetriever class."""

    @pytest.fixture
    def setup_gcp_retriever(self):
        """Setup GCP retriever with sample data."""
        # Create GCP mock
        gcp_mock = GCPEmbeddingMock()

        # Create vector store (use BGE-M3 dimension for matching)
        vector_store = VectorStore(dimension=768)

        # Add sample documents (embed with mock)
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings_list = gcp_mock.get_embeddings(documents)
        embeddings = np.array(embeddings_list, dtype=np.float32)
        doc_ids = ["d1", "d2", "d3"]

        vector_store.add_documents(documents, embeddings, doc_ids)

        # Create GCP retriever
        gcp_retriever = GCPRetriever(gcp_mock, vector_store, top_k=2)

        return gcp_retriever, gcp_mock, vector_store

    def test_gcp_retrieve(self, setup_gcp_retriever):
        """Test GCP mock retrieval."""
        retriever, _, _ = setup_gcp_retriever

        results = retriever.retrieve("test query")
        assert len(results) <= 2

    def test_gcp_retrieve_batch(self, setup_gcp_retriever):
        """Test GCP batch retrieval."""
        retriever, _, _ = setup_gcp_retriever

        queries = ["q1", "q2"]
        results = retriever.retrieve_batch(queries)

        assert len(results) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_retriever(self):
        """Test create_retriever factory."""
        retriever, store, model = create_retriever(top_k=5)

        assert isinstance(retriever, Retriever)
        assert isinstance(store, VectorStore)
        assert isinstance(model, EmbeddingModel)
        assert retriever.top_k == 5

    def test_create_gcp_retriever(self):
        """Test create_gcp_retriever factory."""
        store = VectorStore(dimension=768)
        gcp_retriever, gcp_mock = create_gcp_retriever(store, top_k=3)

        assert isinstance(gcp_retriever, GCPRetriever)
        assert isinstance(gcp_mock, GCPEmbeddingMock)
        assert gcp_retriever.top_k == 3