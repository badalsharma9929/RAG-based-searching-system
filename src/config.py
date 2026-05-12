"""
Configuration settings for the RAG Vector Search project.
"""

# Embedding Model Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # Use "cuda" if GPU available

# Vector Store Settings
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 outputs 384-dim embeddings
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
GCP_EMBEDDING_DIMENSION = 384  # Match local model dimension

# Benchmark Settings
BENCHMARK_TOP_K = [1, 3, 5, 10]

# Seed for reproducibility
RANDOM_SEED = 42