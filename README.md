# RAG Vector Search Assessment

A local Retrieval-Augmented Generation (RAG) system that simulates Google Cloud Vertex AI behavior using open-source components. This project demonstrates semantic search with vector embeddings using FAISS and sentence-transformers.

## Features

- **Local Embeddings**: Uses BGE-M3 (BAAI/bge-m3) - state-of-the-art sentence embeddings
- **Vector Search**: FAISS-powered fast similarity search
- **GCP Simulation**: Mocks Vertex AI's `textembedding-gecko` for comparison
- **Benchmark Metrics**: Precision@K, Recall@K, MRR, NDCG@K
- **Fully Local**: No API keys, no external services needed

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-assessment

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

This will:
1. Load synthetic documents
2. Build FAISS index
3. Run retrieval benchmarks
4. Generate `output/results.json` and `output/retrieval_benchmark.md`

## Project Structure

```
rag-assessment/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── main.py                  # Entry point
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── embeddings.py        # Sentence embedding module
│   ├── vector_store.py     # FAISS vector store
│   ├── retrieval.py        # Retrieval logic
│   ├── gcp_mock.py         # GCP Vertex AI mock
│   ├── benchmark.py        # Evaluation metrics
│   ├── data_loader.py      # Data loading utilities
│   └── generate_data.py    # Synthetic data generator
├── tests/
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   └── test_benchmark.py
├── data/
│   ├── documents.json      # Pre-generated documents
│   └── eval_pairs.json     # Evaluation query-document pairs
└── output/                  # Generated output files
    ├── results.json
    └── retrieval_benchmark.md
```

## How It Works

### 1. Embedding Generation
```python
from src.embeddings import EmbeddingModel

model = EmbeddingModel()
embeddings = model.encode(["Your text here"])
```

### 2. Vector Storage
```python
from src.vector_store import VectorStore

store = VectorStore(dimension=1024)
store.add_documents(texts, embeddings)
results = store.search(query_embedding, top_k=5)
```

### 3. Benchmarking
```python
from src.benchmark import Benchmark

bench = Benchmark()
metrics = bench.evaluate(retrieved, ground_truth)
```

## Output Example

The benchmark generates:

### JSON Output (`output/results.json`)
```json
{
  "local_model": "BAAI/bge-m3",
  "gcp_model": "textembedding-gecko (mocked)",
  "metrics": {
    "precision_at_5": {"local": 0.85, "gcp": 0.82},
    "recall_at_5": {"local": 0.78, "gcp": 0.75},
    "mrr": {"local": 0.88, "gcp": 0.85},
    "ndcg_at_5": {"local": 0.86, "gcp": 0.83}
  }
}
```

### Markdown Table (`output/retrieval_benchmark.md`)

| Metric | Local (BGE-M3) | GCP (Mocked) |
|--------|----------------|---------------|
| Precision@5 | 0.85 | 0.82 |
| Recall@5 | 0.78 | 0.75 |
| MRR | 0.88 | 0.85 |
| NDCG@5 | 0.86 | 0.83 |

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_embeddings.py

# Run with coverage
pytest --cov=src
```

## Configuration

Edit `src/config.py` to customize:

```python
# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"

# Vector store settings
VECTOR_DIMENSION = 1024
TOP_K = 5

# Output settings
OUTPUT_DIR = "output"
```

## Requirements

- Python 3.9+
- 4GB+ RAM (for embedding model)
- ~500MB disk space (for model downloads)

## License

MIT License - feel free to use for learning and development.

## Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for BGE-M3
- [FAISS](https://github.com/facebookresearch/faiss) by Meta
- [Sentence-Transformers](https://sbert.net/) by UKP-TUDA