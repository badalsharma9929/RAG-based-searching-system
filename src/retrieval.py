"""
Retrieval module for RAG system.

Handles query processing, embedding, and retrieval from vector store.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .embeddings import EmbeddingModel, GCPEmbeddingMock
from .vector_store import VectorStore
from . import config


class Retriever:
    """
    Main retrieval class that combines embedding model and vector store.

    Supports both local (sentence-transformers) and GCP-mocked retrieval.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        top_k: int = config.TOP_K
    ):
        """
        Initialize the retriever.

        Args:
            embedding_model: Embedding model for encoding queries
            vector_store: Vector store for similarity search
            top_k: Number of documents to retrieve
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Override default top_k

        Returns:
            List of retrieved documents with metadata
        """
        k = top_k or self.top_k

        # Encode query
        query_embedding = self.embedding_model.encode(query)

        # Search vector store
        results = self.vector_store.search_with_text(query_embedding, top_k=k)

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of retrieval results for each query
        """
        k = top_k or self.top_k

        # Encode all queries at once (more efficient)
        query_embeddings = self.embedding_model.encode(queries)

        results = []
        for query_embedding in query_embeddings:
            query_results = self.vector_store.search_with_text(
                query_embedding, top_k=k
            )
            results.append(query_results)

        return results

    def compute_relevance_scores(
        self,
        queries: List[str],
        documents: List[str]
    ) -> np.ndarray:
        """
        Compute relevance scores between queries and documents.

        Args:
            queries: List of query strings
            documents: List of document strings

        Returns:
            Similarity matrix (n_queries, n_documents)
        """
        # Encode queries and documents
        query_embeddings = self.embedding_model.encode_queries(queries)
        doc_embeddings = self.embedding_model.encode_documents(documents)

        # Compute similarity
        similarities = self.embedding_model.compute_similarity(
            query_embeddings, doc_embeddings
        )

        return similarities


class GCPRetriever:
    """
    GCP-mocked retriever for comparison with local retrieval.

    This simulates how retrieval would work with GCP's Vertex AI
    textembedding-gecko model.
    """

    def __init__(
        self,
        gcp_mock: GCPEmbeddingMock,
        vector_store: VectorStore,
        top_k: int = config.TOP_K
    ):
        """
        Initialize the GCP retriever.

        Args:
            gcp_mock: Mocked GCP embedding model
            vector_store: Vector store for similarity search
            top_k: Number of documents to retrieve
        """
        self.gcp_mock = gcp_mock
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using GCP mock embeddings.

        Args:
            query: Query string
            top_k: Override default top_k

        Returns:
            List of retrieved documents with metadata
        """
        k = top_k or self.top_k

        # Get mock embedding for query
        query_embedding = np.array(self.gcp_mock.get_embeddings(query))

        # Search vector store
        results = self.vector_store.search_with_text(query_embedding, top_k=k)

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve for multiple queries using GCP mock.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of retrieval results for each query
        """
        k = top_k or self.top_k

        results = []
        for query in queries:
            query_results = self.retrieve(query, top_k=k)
            results.append(query_results)

        return results


def create_retriever(
    use_gcp_mock: bool = False,
    top_k: int = config.TOP_K
) -> Tuple[Retriever, VectorStore, EmbeddingModel]:
    """
    Factory function to create a retriever with all components.

    Args:
        use_gcp_mock: Whether to use GCP mock instead of real embeddings
        top_k: Number of documents to retrieve

    Returns:
        Tuple of (retriever, vector_store, embedding_model)
    """
    # Create embedding model
    embedding_model = EmbeddingModel()

    # Create vector store
    vector_store = VectorStore(dimension=embedding_model.dimension)

    # Create retriever
    retriever = Retriever(embedding_model, vector_store, top_k)

    return retriever, vector_store, embedding_model


def create_gcp_retriever(
    vector_store: VectorStore,
    top_k: int = config.TOP_K
) -> Tuple[GCPRetriever, GCPEmbeddingMock]:
    """
    Factory function to create a GCP-mocked retriever.

    Args:
        vector_store: Pre-populated vector store
        top_k: Number of documents to retrieve

    Returns:
        Tuple of (gcp_retriever, gcp_mock)
    """
    # Create GCP mock
    gcp_mock = GCPEmbeddingMock()

    # Create GCP retriever
    gcp_retriever = GCPRetriever(gcp_mock, vector_store, top_k)

    return gcp_retriever, gcp_mock