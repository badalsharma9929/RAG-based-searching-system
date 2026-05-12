"""
Benchmark module for evaluating retrieval performance.

Computes standard retrieval metrics:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
from tabulate import tabulate

from . import config


class Benchmark:
    """
    Evaluation benchmark for RAG retrieval.

    Computes and compares retrieval metrics for local vs GCP-mocked systems.
    """

    def __init__(self):
        """Initialize the benchmark."""
        self.results = {}

    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Precision = (# of relevant retrieved) / K

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff position

        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)

        return relevant_retrieved / k

    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Recall = (# of relevant retrieved) / (# of total relevant)

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff position

        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)

        return relevant_retrieved / len(relevant)

    def mean_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = (1/N) * sum(1/rank_i) where rank_i is position of first relevant doc

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs

        Returns:
            MRR score
        """
        if len(retrieved) == 0 or len(relevant) == 0:
            return 0.0

        # Find position of first relevant document (1-indexed)
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i

        return 0.0

    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)@K.

        NDCG = DCG / IDCG

        DCG = sum(rels[i] / log2(i+2)) for i in 0 to k-1
        IDCG = ideal DCG (sorted by relevance)

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff position

        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                # Relevance is 1 if in relevant set
                dcg += 1.0 / np.log2(i + 2)

        # Calculate IDCG (ideal DCG)
        n_relevant = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_single_query(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_docs: List[str],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate a single query.

        Args:
            retrieved_docs: List of retrieved documents with 'doc_id'
            relevant_docs: List of relevant document IDs
            k_values: List of K values to compute

        Returns:
            Dictionary of metrics
        """
        retrieved_ids = [doc["doc_id"] for doc in retrieved_docs]

        metrics = {}

        for k in k_values:
            metrics[f"precision_{k}"] = self.precision_at_k(
                retrieved_ids, relevant_docs, k
            )
            metrics[f"recall_{k}"] = self.recall_at_k(
                retrieved_ids, relevant_docs, k
            )
            metrics[f"ndcg_{k}"] = self.ndcg_at_k(
                retrieved_ids, relevant_docs, k
            )

        metrics["mrr"] = self.mean_reciprocal_rank(
            retrieved_ids, relevant_docs
        )

        return metrics

    def evaluate(
        self,
        retrieval_results: List[List[Dict[str, Any]]],
        eval_pairs: List[Dict[str, Any]],
        k_values: List[int] = config.BENCHMARK_TOP_K
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance across all queries.

        Args:
            retrieval_results: List of retrieval results per query
            eval_pairs: Evaluation pairs with query, relevant docs
            k_values: List of K values

        Returns:
            Dictionary of averaged metrics
        """
        all_metrics = []

        for results, eval_pair in zip(retrieval_results, eval_pairs):
            relevant = eval_pair.get("relevant_docs", [])
            metrics = self.evaluate_single_query(results, relevant, k_values)
            all_metrics.append(metrics)

        # Average metrics across all queries
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def compare_systems(
        self,
        local_results: Dict[str, float],
        gcp_results: Dict[str, float],
        model_names: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Compare local and GCP-mocked systems.

        Args:
            local_results: Local model metrics
            gcp_results: GCP mock metrics
            model_names: Optional custom model names

        Returns:
            Comparison results
        """
        if model_names is None:
            model_names = {
                "local": "BGE-M3 (Local)",
                "gcp": "textembedding-gecko (Mock)"
            }

        comparison = {
            "local_model": model_names.get("local", "Local"),
            "gcp_model": model_names.get("gcp", "GCP Mock"),
            "metrics": {},
            "differences": {}
        }

        for metric in local_results:
            local_val = local_results.get(metric, 0)
            gcp_val = gcp_results.get(metric, 0)

            comparison["metrics"][metric] = {
                "local": round(local_val, 4),
                "gcp": round(gcp_val, 4)
            }

            # Calculate difference
            diff = local_val - gcp_val
            comparison["differences"][metric] = {
                "absolute": round(diff, 4),
                "relative": round((diff / gcp_val * 100) if gcp_val != 0 else 0, 2)
            }

        return comparison

    def generate_markdown_table(
        self,
        comparison: Dict[str, Any]
    ) -> str:
        """
        Generate markdown table from comparison results.

        Args:
            comparison: Comparison results from compare_systems

        Returns:
            Markdown formatted table
        """
        table_data = []
        headers = ["Metric", "Local", "GCP (Mock)", "Difference"]

        metrics = comparison.get("metrics", {})

        # Reorder metrics for better presentation
        metric_order = [
            "precision_1", "precision_3", "precision_5", "precision_10",
            "recall_1", "recall_3", "recall_5", "recall_10",
            "ndcg_1", "ndcg_3", "ndcg_5", "ndcg_10",
            "mrr"
        ]

        for metric in metric_order:
            if metric in metrics:
                local_val = metrics[metric]["local"]
                gcp_val = metrics[metric]["gcp"]
                diff = comparison["differences"][metric]["absolute"]

                # Format metric name
                name = metric.replace("_", " ").title()

                # Add +/- sign to difference
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"

                table_data.append([name, f"{local_val:.4f}", f"{gcp_val:.4f}", diff_str])

        table = tabulate(
            table_data,
            headers=headers,
            tablefmt="github",
            colalign=("left", "right", "right", "right")
        )

        return table

    def save_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ) -> None:
        """Save results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")

    def save_markdown(
        self,
        comparison: Dict[str, Any],
        filepath: str
    ) -> None:
        """Save markdown report."""
        table = self.generate_markdown_table(comparison)

        markdown = f"""# Retrieval Benchmark Results

## Models Compared
- **Local**: {comparison.get('local_model', 'BGE-M3')}
- **GCP (Mock)**: {comparison.get('gcp_model', 'textembedding-gecko')}

## Results

{table}

## Analysis

The local model uses BGE-M3 embeddings generated with sentence-transformers.
The GCP mock simulates Vertex AI's textembedding-gecko behavior for comparison.

Higher values indicate better performance for all metrics.
"""

        with open(filepath, "w") as f:
            f.write(markdown)

        print(f"Markdown report saved to {filepath}")