"""
RAG Vector Search Assessment Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    VECTOR_DIMENSION,
    TOP_K,
    DATA_DIR,
    OUTPUT_DIR,
)

__all__ = [
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "VECTOR_DIMENSION",
    "TOP_K",
    "DATA_DIR",
    "OUTPUT_DIR",
]