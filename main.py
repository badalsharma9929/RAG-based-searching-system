#!/usr/bin/env python3
"""
Main entry point for RAG Vector Search Assessment.

This script runs the complete RAG pipeline:
1. Loads documents and evaluation data
2. Builds vector index
3. Runs retrieval benchmarks
4. Generates output reports
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_DIR,
    OUTPUT_DIR,
    TOP_K,
    BENCHMARK_TOP_K
)
from src.data_loader import DataLoader
from src.embeddings import EmbeddingModel, GCPEmbeddingMock
from src.vector_store import VectorStore
from src.retrieval import Retriever, GCPRetriever
from src.benchmark import Benchmark
from src.generate_data import DataGenerator


def setup_directories():
    """Ensure required directories exist."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {OUTPUT_DIR}")


def load_or_generate_data() -> tuple:
    """
    Load existing data or generate new synthetic data.

    Returns:
        Tuple of (documents, eval_pairs)
    """
    loader = DataLoader(DATA_DIR)

    try:
        # Try to load existing data
        documents = loader.load_documents()
        eval_pairs = loader.load_eval_pairs()
        print("Loaded existing data files.")
    except FileNotFoundError:
        # Generate new data
        print("Data files not found. Generating synthetic data...")
        generator = DataGenerator()
        documents, eval_pairs = generator.generate_all(n_docs=75, n_eval=25)

        # Save generated data
        loader.save_documents(documents)
        loader.save_eval_pairs(eval_pairs)
        print("Synthetic data generated and saved.")

    return documents, eval_pairs


def build_vector_index(documents: list) -> VectorStore:
    """
    Build FAISS vector index from documents.

    Args:
        documents: List of document dictionaries

    Returns:
        Populated VectorStore
    """
    print("\n" + "="*50)
    print("Building Vector Index")
    print("="*50)

    # Initialize embedding model and vector store
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(dimension=embedding_model.dimension)

    # Extract texts and IDs
    loader = DataLoader()
    texts = loader.extract_texts(documents)
    doc_ids = loader.extract_doc_ids(documents)

    # Generate embeddings
    print(f"\nEncoding {len(texts)} documents...")
    embeddings = embedding_model.encode_documents(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Add to vector store
    print("\nAdding documents to vector store...")
    vector_store.add_documents(texts, embeddings, doc_ids)

    # Print stats
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  - Documents: {stats['n_documents']}")
    print(f"  - Dimension: {stats['dimension']}")
    print(f"  - Index Type: {stats['index_type']}")

    return vector_store


def run_local_retrieval(
    retriever: Retriever,
    eval_pairs: list
) -> list:
    """
    Run retrieval using local embeddings.

    Args:
        retriever: Local retriever
        eval_pairs: Evaluation pairs

    Returns:
        List of retrieval results
    """
    print("\n" + "="*50)
    print("Running Local Retrieval (BGE-M3)")
    print("="*50)

    loader = DataLoader()
    queries = loader.extract_queries(eval_pairs)

    print(f"Retrieving for {len(queries)} queries...")
    results = retriever.retrieve_batch(queries, top_k=10)

    # Show sample results
    print("\nSample Results (first query):")
    for i, doc in enumerate(results[0][:3], 1):
        print(f"  {i}. {doc['doc_id']} (score: {doc['distance']:.4f})")

    return results


def run_gcp_retrieval(
    gcp_retriever: GCPRetriever,
    eval_pairs: list
) -> list:
    """
    Run retrieval using GCP mock.

    Args:
        gcp_retriever: GCP-mocked retriever
        eval_pairs: Evaluation pairs

    Returns:
        List of retrieval results
    """
    print("\n" + "="*50)
    print("Running GCP Mock Retrieval (textembedding-gecko)")
    print("="*50)

    loader = DataLoader()
    queries = loader.extract_queries(eval_pairs)

    print(f"Retrieving for {len(queries)} queries...")
    results = gcp_retriever.retrieve_batch(queries, top_k=10)

    # Show sample results
    print("\nSample Results (first query):")
    for i, doc in enumerate(results[0][:3], 1):
        print(f"  {i}. {doc['doc_id']} (score: {doc['distance']:.4f})")

    return results


def run_benchmarks(local_results: list, gcp_results: list, eval_pairs: list):
    """
    Run evaluation benchmarks and generate reports.

    Args:
        local_results: Local retrieval results
        gcp_results: GCP mock results
        eval_pairs: Ground truth evaluation pairs
    """
    print("\n" + "="*50)
    print("Running Benchmark Evaluation")
    print("="*50)

    # Initialize benchmark
    benchmark = Benchmark()

    # Evaluate local model
    print("\nEvaluating Local Model...")
    local_metrics = benchmark.evaluate(local_results, eval_pairs)
    print(f"  Precision@5: {local_metrics.get('precision_5', 0):.4f}")
    print(f"  Recall@5: {local_metrics.get('recall_5', 0):.4f}")
    print(f"  MRR: {local_metrics.get('mrr', 0):.4f}")
    print(f"  NDCG@5: {local_metrics.get('ndcg_5', 0):.4f}")

    # Evaluate GCP mock
    print("\nEvaluating GCP Mock...")
    gcp_metrics = benchmark.evaluate(gcp_results, eval_pairs)
    print(f"  Precision@5: {gcp_metrics.get('precision_5', 0):.4f}")
    print(f"  Recall@5: {gcp_metrics.get('recall_5', 0):.4f}")
    print(f"  MRR: {gcp_metrics.get('mrr', 0):.4f}")
    print(f"  NDCG@5: {gcp_metrics.get('ndcg_5', 0):.4f}")

    # Compare systems
    print("\n" + "="*50)
    print("Comparison Results")
    print("="*50)

    comparison = benchmark.compare_systems(local_metrics, gcp_metrics)

    # Generate markdown table
    table = benchmark.generate_markdown_table(comparison)
    print("\n" + table)

    # Save results
    output_path = Path(OUTPUT_DIR)

    # Save JSON
    results_data = {
        "local_model": comparison["local_model"],
        "gcp_model": comparison["gcp_model"],
        "metrics": comparison["metrics"],
        "differences": comparison["differences"]
    }
    benchmark.save_results(results_data, output_path / "results.json")

    # Save markdown
    benchmark.save_markdown(comparison, output_path / "retrieval_benchmark.md")

    print("\n" + "="*50)
    print("Output Files Generated")
    print("="*50)
    print(f"  - {output_path / 'results.json'}")
    print(f"  - {output_path / 'retrieval_benchmark.md'}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print(" RAG Vector Search Assessment")
    print("="*60)

    # Setup
    setup_directories()

    # Load or generate data
    documents, eval_pairs = load_or_generate_data()

    # Build vector index
    vector_store = build_vector_index(documents)

    # Create retrievers
    embedding_model = EmbeddingModel()
    local_retriever = Retriever(embedding_model, vector_store, TOP_K)

    gcp_mock = GCPEmbeddingMock()
    gcp_retriever = GCPRetriever(gcp_mock, vector_store, TOP_K)

    # Run retrievals
    local_results = run_local_retrieval(local_retriever, eval_pairs)
    gcp_results = run_gcp_retrieval(gcp_retriever, eval_pairs)

    # Run benchmarks
    run_benchmarks(local_results, gcp_results, eval_pairs)

    print("\n" + "="*60)
    print(" Assessment Complete!")
    print("="*60)


if __name__ == "__main__":
    main()