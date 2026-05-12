"""
Tests for benchmark module.
"""

import pytest
import numpy as np
from src.benchmark import Benchmark


class TestBenchmarkMetrics:
    """Tests for benchmark metric calculations."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return Benchmark()

    # Precision Tests
    def test_precision_at_k_perfect(self, benchmark):
        """Test precision with perfect retrieval."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        k = 3

        precision = benchmark.precision_at_k(retrieved, relevant, k)
        assert precision == 1.0

    def test_precision_at_k_partial(self, benchmark):
        """Test precision with partial retrieval."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc4"]
        k = 3

        precision = benchmark.precision_at_k(retrieved, relevant, k)
        assert precision == 2/3  # 2 relevant out of 3

    def test_precision_at_k_zero(self, benchmark):
        """Test precision with no relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5"]
        k = 3

        precision = benchmark.precision_at_k(retrieved, relevant, k)
        assert precision == 0.0

    def test_precision_at_k_zero_top_k(self, benchmark):
        """Test precision with k=0."""
        precision = benchmark.precision_at_k(["doc1"], ["doc1"], 0)
        assert precision == 0.0

    # Recall Tests
    def test_recall_at_k_all_retrieved(self, benchmark):
        """Test recall when all relevant are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3", "doc4"]
        k = 3

        recall = benchmark.recall_at_k(retrieved, relevant, k)
        assert recall == 3/4  # 3 out of 4

    def test_recall_at_k_none_retrieved(self, benchmark):
        """Test recall when none retrieved."""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc3", "doc4", "doc5"]
        k = 2

        recall = benchmark.recall_at_k(retrieved, relevant, k)
        assert recall == 0.0

    def test_recall_at_k_empty_relevant(self, benchmark):
        """Test recall with empty relevant set."""
        recall = benchmark.recall_at_k(["doc1"], [], 1)
        assert recall == 0.0

    # MRR Tests
    def test_mrr_first_relevant(self, benchmark):
        """Test MRR when first result is relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]

        mrr = benchmark.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 1.0  # 1/1

    def test_mrr_second_relevant(self, benchmark):
        """Test MRR when second result is relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]

        mrr = benchmark.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.5  # 1/2

    def test_mrr_none_relevant(self, benchmark):
        """Test MRR when no result is relevant."""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc3"]

        mrr = benchmark.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.0

    # NDCG Tests
    def test_ndcg_at_k_perfect(self, benchmark):
        """Test NDCG with perfect ranking."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        k = 3

        ndcg = benchmark.ndcg_at_k(retrieved, relevant, k)
        assert ndcg == 1.0

    def test_ndcg_at_k_partial(self, benchmark):
        """Test NDCG with partial retrieval."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc4"]
        k = 3

        ndcg = benchmark.ndcg_at_k(retrieved, relevant, k)
        assert 0 < ndcg < 1

    def test_ndcg_at_k_zero(self, benchmark):
        """Test NDCG with no relevant docs."""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc3", "doc4"]
        k = 2

        ndcg = benchmark.ndcg_at_k(retrieved, relevant, k)
        assert ndcg == 0.0


class TestBenchmarkEvaluation:
    """Tests for full benchmark evaluation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        eval_pairs = [
            {
                "query": "What is ML?",
                "relevant_docs": ["doc1", "doc2"]
            },
            {
                "query": "What is DL?",
                "relevant_docs": ["doc3"]
            }
        ]

        # Simulated retrieval results
        local_results = [
            [
                {"doc_id": "doc1", "distance": 0.9},
                {"doc_id": "doc2", "distance": 0.8},
                {"doc_id": "doc3", "distance": 0.5}
            ],
            [
                {"doc_id": "doc2", "distance": 0.9},
                {"doc_id": "doc3", "distance": 0.8},
                {"doc_id": "doc1", "distance": 0.3}
            ]
        ]

        gcp_results = [
            [
                {"doc_id": "doc1", "distance": 0.8},
                {"doc_id": "doc4", "distance": 0.7},
                {"doc_id": "doc2", "distance": 0.6}
            ],
            [
                {"doc_id": "doc3", "distance": 0.9},
                {"doc_id": "doc5", "distance": 0.8},
                {"doc_id": "doc2", "distance": 0.7}
            ]
        ]

        return local_results, gcp_results, eval_pairs

    def test_evaluate_local(self, benchmark, sample_data):
        """Test evaluating local model."""
        local_results, _, eval_pairs = sample_data

        metrics = benchmark.evaluate(local_results, eval_pairs)

        assert "precision_5" in metrics
        assert "recall_5" in metrics
        assert "mrr" in metrics
        assert "ndcg_5" in metrics

    def test_compare_systems(self, benchmark, sample_data):
        """Test comparing two systems."""
        local_results, gcp_results, eval_pairs = sample_data

        local_metrics = benchmark.evaluate(local_results, eval_pairs)
        gcp_metrics = benchmark.evaluate(gcp_results, eval_pairs)

        comparison = benchmark.compare_systems(local_metrics, gcp_metrics)

        assert "local_model" in comparison
        assert "gcp_model" in comparison
        assert "metrics" in comparison
        assert "differences" in comparison

    def test_generate_markdown_table(self, benchmark, sample_data):
        """Test markdown table generation."""
        local_results, gcp_results, eval_pairs = sample_data

        local_metrics = benchmark.evaluate(local_results, eval_pairs)
        gcp_metrics = benchmark.evaluate(gcp_results, eval_pairs)

        comparison = benchmark.compare_systems(local_metrics, gcp_metrics)
        table = benchmark.generate_markdown_table(comparison)

        assert "Metric" in table
        assert "Local" in table
        assert "GCP" in table


class TestBenchmarkSave:
    """Tests for saving results."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return Benchmark()

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        return tmp_path

    def test_save_results_json(self, benchmark, temp_output_dir):
        """Test saving results to JSON."""
        results = {
            "local_model": "BGE-M3",
            "gcp_model": "mock",
            "metrics": {"precision_5": 0.8}
        }

        filepath = temp_output_dir / "results.json"
        benchmark.save_results(results, str(filepath))

        assert filepath.exists()

    def test_save_markdown(self, benchmark, temp_output_dir):
        """Test saving markdown report."""
        comparison = {
            "local_model": "Local",
            "gcp_model": "GCP",
            "metrics": {"precision_5": {"local": 0.8, "gcp": 0.7}},
            "differences": {"precision_5": {"absolute": 0.1}}
        }

        filepath = temp_output_dir / "report.md"
        benchmark.save_markdown(comparison, str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "Retrieval Benchmark Results" in content