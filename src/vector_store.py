"""
FAISS-based vector store for efficient similarity search.

FAISS (Facebook AI Similarity Search) provides fast algorithms for
similarity search and clustering of dense vectors.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any


class VectorStore:
    """
    Vector store using FAISS for similarity search.

    Supports:
    - Flat index (exact search)
    - IndexFlatIP (inner product / cosine similarity)
    - IndexFlatL2 (Euclidean distance)
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "ip"
    ):
        """
        Initialize the vector store.

        Args:
            dimension: Embedding vector dimension
            index_type: Type of FAISS index ("flat")
            metric: Distance metric ("ip" for inner product, "l2" for Euclidean)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self._index = None
        self._documents = []
        self._doc_ids = []

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.metric == "ip":
            # Inner product - use with normalized vectors for cosine similarity
            index = faiss.IndexFlatIP(self.dimension)
        else:
            # L2 distance (Euclidean)
            index = faiss.IndexFlatL2(self.dimension)

        return index

    @property
    def index(self) -> faiss.Index:
        """Get or create the FAISS index."""
        if self._index is None:
            self._index = self._create_index()
        return self._index

    @property
    def is_trained(self) -> bool:
        """Check if index is ready (always true for flat index)."""
        return True

    @property
    def n_vectors(self) -> int:
        """Return number of vectors in the index."""
        return self.index.ntotal

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: Document embeddings (n_docs, dimension)
            doc_ids: Optional custom document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        # Validate embedding dimensions
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"expected dimension {self.dimension}"
            )

        # Convert to float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store documents and IDs
        self._documents.extend(documents)

        # Generate IDs if not provided
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        self._doc_ids.extend(doc_ids)

        print(f"Added {len(documents)} documents to vector store.")
        print(f"Total documents in index: {self.n_vectors}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for most similar documents.

        Args:
            query_embedding: Query vector (dimension,) or (1, dimension)
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices)
            - distances: Array of similarity scores
            - indices: Array of document indices
        """
        # Handle single query vector
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure correct dimension
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_embedding.shape[1]} does not match "
                f"expected dimension {self.dimension}"
            )

        # Convert to float32
        query_embedding = query_embedding.astype(np.float32)

        # Search
        top_k = min(top_k, self.n_vectors)
        distances, indices = self.index.search(query_embedding, top_k)

        return distances, indices

    def search_with_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search and return documents with text content.

        Args:
            query_embedding: Query vector
            top_k: Number of results

        Returns:
            List of dicts with 'id', 'text', 'distance', 'rank'
        """
        distances, indices = self.search(query_embedding, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # Valid index
                results.append({
                    "rank": rank + 1,
                    "doc_id": self._doc_ids[idx],
                    "text": self._documents[idx],
                    "distance": float(dist),
                    "index": int(idx)
                })

        return results

    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document text by ID."""
        try:
            idx = self._doc_ids.index(doc_id)
            return self._documents[idx]
        except (ValueError, IndexError):
            return None

    def get_document_by_index(self, index: int) -> Optional[str]:
        """Get document text by index."""
        if 0 <= index < len(self._documents):
            return self._documents[index]
        return None

    def save_index(self, filepath: str) -> None:
        """Save FAISS index to file."""
        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str, documents: List[str], doc_ids: List[str]) -> None:
        """Load FAISS index from file."""
        self._index = faiss.read_index(filepath)
        self._documents = documents
        self._doc_ids = doc_ids
        print(f"Index loaded from {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the vector store."""
        return {
            "n_documents": self.n_vectors,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric
        }

    def reset(self) -> None:
        """Clear all documents and reset the index."""
        self._index = None
        self._documents = []
        self._doc_ids = []
        print("Vector store reset.")