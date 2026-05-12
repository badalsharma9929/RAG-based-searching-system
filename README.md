# RAG Vector Search Assessment

A local Retrieval-Augmented Generation (RAG) system that simulates Google Cloud Vertex AI behavior using open-source components. This project demonstrates semantic search with vector embeddings using FAISS and transformers.

## Features

- **Local Embeddings**: Uses all-MiniLM-L6-v2 - fast, lightweight sentence embeddings
- **Vector Search**: FAISS-powered fast similarity search
- **GCP Simulation**: Mocks Vertex AI's `textembedding-gecko` using unittest.mock
- **Benchmark Metrics**: Precision@K, Recall@K, MRR, NDCG@K
- **Fully Local**: No API keys, no external services needed

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/badalsharma9929/rag-assessment.git
cd rag-assessment

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

This will:
1. Load synthetic documents (75 docs)
2. Build FAISS vector index
3. Run retrieval with local embeddings
4. Run retrieval with GCP mock
5. Generate benchmark comparison

### 3. Output Files

- `output/results.json` - Full metrics in JSON
- `output/retrieval_benchmark.md` - Markdown comparison table

## Project Structure

```
rag-assessment/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
├── main.py                  # Entry point
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── embeddings.py        # Embedding module (all-MiniLM-L6-v2 + GCP mock)
│   ├── vector_store.py      # FAISS vector store
│   ├── retrieval.py         # Retrieval pipeline
│   ├── benchmark.py         # Evaluation metrics
│   ├── data_loader.py      # Data loading utilities
│   └── generate_data.py    # Synthetic data generator
├── tests/
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   └── test_benchmark.py
├── data/
│   ├── documents.json       # 75 synthetic documents
│   └── eval_pairs.json      # 25 evaluation query-document pairs
└── output/
    ├── results.json         # Benchmark results
    └── retrieval_benchmark.md # Markdown table
```

## How It Works

### 1. Embedding Generation
```python
from src.embeddings import EmbeddingModel

model = EmbeddingModel()  # Uses all-MiniLM-L6-v2
embeddings = model.encode(["Your text here"])
# Output: (1, 384) numpy array
```

### 2. Vector Storage
```python
from src.vector_store import VectorStore

store = VectorStore(dimension=384)
store.add_documents(texts, embeddings)
results = store.search(query_embedding, top_k=5)
```

### 3. GCP Mocking
```python
from src.embeddings import GCPEmbeddingMock

gcp_mock = GCPEmbeddingMock()
# Uses unittest.mock pattern - no actual GCP calls
embeddings = gcp_mock.get_embeddings("text")
```

### 4. Benchmarking
```python
from src.benchmark import Benchmark

bench = Benchmark()
metrics = bench.evaluate(retrieved_results, ground_truth)
# Outputs: Precision@K, Recall@K, MRR, NDCG@K
```

## Example Output

```
| Metric       |   Local |   GCP (Mock) |   Difference |
|--------------|---------|--------------|--------------|
| Precision 5  |    0.28 |        0.016 |        0.264 |
| Recall 5     |       1 |         0.06 |         0.94 |
| Mrr          |    0.85 |        0.036 |        0.818 |
| Ndcg 5       |    0.90 |        0.036 |        0.860 |
```

The local model significantly outperforms the GCP mock because it uses real semantic embeddings.

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
# Embedding model (all-MiniLM-L6-v2)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Vector store settings
VECTOR_DIMENSION = 384
TOP_K = 5

# Output settings
OUTPUT_DIR = "output"
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Embeddings | all-MiniLM-L6-v2 (transformers) |
| Vector Store | FAISS (faiss-cpu) |
| GCP Mock | unittest.mock |
| Testing | pytest |
| Output | json, markdown, tabulate |

## Requirements

- Python 3.9+
- 4GB+ RAM
- ~500MB disk space (for model downloads)

## License

MIT License - feel free to use for learning and development.

## Acknowledgments

- [Sentence-Transformers](https://sbert.net/) for the model architecture
- [FAISS](https://github.com/facebookresearch/faiss) by Meta
- [Hugging Face Transformers](https://huggingface.co/transformers)