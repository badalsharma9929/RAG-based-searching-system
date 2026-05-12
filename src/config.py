"""
Configuration settings for the RAG Vector Search project.
"""

# Embedding Model Settings
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"  # Use "cuda" if GPU available

# Vector Store Settings
VECTOR_DIMENSION = 1024  # BGE-M3 outputs 1024-dim embeddings
TOP_K = 5  # Number of results to retrieve
INDEX_TYPE = "flat"  # FAISS index type (flat, ivf, hnsw)

# Data Settings
DATA_DIR = "data"
DOCUMENTS_FILE = "documents.json"
EVAL_PAIRS_FILE = "eval_pairs.json"

# Output Settings
OUTPUT_DIR = "output"
RESULTS_FILE = "results.json"
BENCHMARK_FILE = "retrieval_benchmark.md"

# GCP Mock Settings
GCP_EMBEDDING_MODEL_NAME = "textembedding-gecko"
GCP_EMBEDDING_DIMENSION = 768  # Gecko outputs 768-dim

# Benchmark Settings
BENCHMARK_TOP_K = [1, 3, 5, 10]

# Seed for reproducibility
RANDOM_SEED = 42