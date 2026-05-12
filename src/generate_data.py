"""
Synthetic data generator for documents and evaluation pairs.

Creates realistic tech/AI documentation and corresponding evaluation data.
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime

from . import config


class DataGenerator:
    """
    Generate synthetic documents and evaluation pairs for RAG testing.

    Creates tech/AI documentation with realistic topics and
    corresponding query-document pairs for evaluation.
    """

    # Document templates for different topics
    DOCUMENT_TOPICS = [
        # AI & Machine Learning
        {
            "topic": "machine_learning",
            "title_template": [
                "Introduction to {concept}",
                "{concept} Explained: A Comprehensive Guide",
                "Understanding {concept} in Modern AI",
                "Deep Dive into {concept}",
                "{concept}: From Basics to Advanced"
            ],
            "concepts": [
                "Neural Networks", "Deep Learning", "Transformers",
                "Convolutional Neural Networks", "Recurrent Neural Networks",
                "Reinforcement Learning", "Transfer Learning",
                "Fine-tuning Large Language Models", "Attention Mechanisms",
                "Gradient Descent", "Backpropagation", "Batch Normalization"
            ]
        },
        # RAG & Vector Search
        {
            "topic": "rag_vector",
            "title_template": [
                "Building RAG Systems with {concept}",
                "{concept} for Vector Search",
                "Implementing {concept} in Production",
                "Optimizing {concept} for Scale",
                "Best Practices for {concept}"
            ],
            "concepts": [
                "Retrieval-Augmented Generation", "Vector Databases",
                "FAISS", "Semantic Search", "Embedding Models",
                "Chunking Strategies", "Hybrid Search",
                "Reranking", "Cross-Encoder", "Bi-Encoder",
                "HNSW Indexing", "Quantization"
            ]
        },
        # NLP & Text Processing
        {
            "topic": "nlp",
            "title_template": [
                "NLP Fundamentals: {concept}",
                "Working with {concept} in Python",
                "{concept} for Text Analysis",
                "Advanced {concept} Techniques",
                "{concept}: Theory and Practice"
            ],
            "concepts": [
                "Tokenization", "Word Embeddings", "Text Classification",
                "Named Entity Recognition", "Sentiment Analysis",
                "Text Summarization", "Machine Translation",
                "Question Answering", "Text Generation", "Prompt Engineering"
            ]
        },
        # Cloud & MLOps
        {
            "topic": "cloud_mlops",
            "title_template": [
                "Deploying ML with {concept}",
                "{concept} for Machine Learning",
                "{concept} in MLOps Pipelines",
                "Scaling {concept} for Production",
                "{concept}: Best Practices"
            ],
            "concepts": [
                "Kubernetes", "Docker", "TensorFlow Serving",
                "MLflow", "Kubeflow", "Vertex AI", "SageMaker",
                "Model Monitoring", "A/B Testing", "Feature Store",
                "Data Pipeline", "Model Registry"
            ]
        }
    ]

    # Query templates
    QUERY_TEMPLATES = [
        # Conceptual questions
        "What is {concept} and how does it work?",
        "Explain {concept} with examples",
        "How do I implement {concept}?",
        "What are the best practices for {concept}?",
        "When should I use {concept}?",
        "What are the advantages of {concept}?",
        "How does {concept} differ from traditional approaches?",
        "Can you explain the architecture of {concept}?",
        "What are the key components of {concept}?",
        "How is {concept} used in production systems?",
        # Specific implementation questions
        "How to optimize {concept} for better performance?",
        "What tools are available for {concept}?",
        "Show me an example of {concept} implementation",
        "What are common pitfalls when using {concept}?",
        "How does {concept} integrate with other systems?"
    ]

    def __init__(self, seed: int = config.RANDOM_SEED):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.seed = seed

    def _generate_document_content(self, concept: str, topic: str) -> str:
        """Generate realistic document content for a concept."""
        # Create multiple paragraphs
        paragraphs = []

        # Introduction
        paragraphs.append(
            f"{concept} is a fundamental concept in {topic} that plays a crucial "
            f"role in modern artificial intelligence and machine learning applications. "
            f"In recent years, the importance of {concept} has grown significantly as "
            f"organizations seek to build more sophisticated AI systems."
        )

        # Technical explanation
        paragraphs.append(
            f"When implementing {concept}, developers need to consider several key "
            f"factors. First, the choice of algorithm and model architecture depends on "
            f"the specific use case and performance requirements. Second, data quality "
            f"and preprocessing steps are critical for achieving optimal results. "
            f"Third, computational resources and scalability constraints must be evaluated "
            f"to ensure the solution can handle production workloads."
        )

        # Practical considerations
        paragraphs.append(
            f"In practice, working with {concept} requires a deep understanding of both "
            f"theoretical foundations and practical implementation details. Many teams start "
            f"with baseline implementations and iterate based on performance metrics. "
            f"Common approaches include hyperparameter tuning, model compression, and "
            f"distributed training strategies for handling large-scale deployments."
        )

        # Best practices
        paragraphs.append(
            f"Best practices for {concept} include proper documentation of model decisions, "
            f"version control for experiments, and systematic evaluation protocols. "
            f"It's also important to implement monitoring systems that track key metrics "
            f"and detect performance degradation over time. This ensures that AI systems "
            f"remain reliable and effective in production environments."
        )

        # Conclusion
        paragraphs.append(
            f"Looking ahead, {concept} will continue to evolve as research advances and "
            f"new use cases emerge. Organizations that invest in understanding and "
            f"mastering {concept} will be better positioned to leverage the full potential "
            f"of AI technologies. The key is to stay updated with the latest developments "
            f"while maintaining a focus on practical, production-ready implementations."
        )

        return "\n\n".join(paragraphs)

    def generate_documents(self, n_docs: int = 75) -> List[Dict[str, Any]]:
        """
        Generate synthetic documents.

        Args:
            n_docs: Number of documents to generate

        Returns:
            List of document dictionaries
        """
        documents = []

        for i in range(n_docs):
            # Select random topic
            topic_data = random.choice(self.DOCUMENT_TOPICS)
            topic = topic_data["topic"]
            concepts = topic_data["concepts"]

            # Select random concept
            concept = random.choice(concepts)

            # Select random title template
            title = random.choice(topic_data["title_template"]).format(
                concept=concept
            )

            # Generate content
            text = self._generate_document_content(concept, topic)

            # Create document
            doc = {
                "id": f"doc_{i+1}",
                "title": title,
                "text": text,
                "concept": concept,
                "topic": topic,
                "word_count": len(text.split())
            }

            documents.append(doc)

        return documents

    def generate_eval_pairs(
        self,
        documents: List[Dict[str, Any]],
        n_pairs: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Generate evaluation query-document pairs.

        Args:
            documents: List of documents
            n_pairs: Number of evaluation pairs to generate

        Returns:
            List of evaluation pairs
        """
        eval_pairs = []

        # Group documents by concept for better relevance
        concept_docs = {}
        for doc in documents:
            concept = doc.get("concept", "unknown")
            if concept not in concept_docs:
                concept_docs[concept] = []
            concept_docs[concept].append(doc)

        for i in range(n_pairs):
            # Select a concept
            concept = random.choice(list(concept_docs.keys()))
            docs_for_concept = concept_docs[concept]

            # Select a query template
            query_template = random.choice(self.QUERY_TEMPLATES)
            query = query_template.format(concept=concept)

            # Select relevant documents (1-3 relevant docs)
            n_relevant = random.randint(1, min(3, len(docs_for_concept)))
            relevant_docs = random.sample(docs_for_concept, n_relevant)
            relevant_doc_ids = [doc["id"] for doc in relevant_docs]

            # Create eval pair
            eval_pair = {
                "id": f"eval_{i+1}",
                "query": query,
                "relevant_docs": relevant_doc_ids,
                "concept": concept,
                "n_relevant": n_relevant
            }

            eval_pairs.append(eval_pair)

        return eval_pairs

    def generate_all(self, n_docs: int = 75, n_eval: int = 25):
        """
        Generate both documents and evaluation pairs.

        Args:
            n_docs: Number of documents
            n_eval: Number of evaluation pairs

        Returns:
            Tuple of (documents, eval_pairs)
        """
        documents = self.generate_documents(n_docs)
        eval_pairs = self.generate_eval_pairs(documents, n_eval)

        return documents, eval_pairs


def main():
    """Run data generation and save to files."""
    from .data_loader import DataLoader

    print("Generating synthetic data...")

    generator = DataGenerator()
    documents, eval_pairs = generator.generate_all(n_docs=75, n_eval=25)

    # Save using DataLoader
    loader = DataLoader()
    loader.save_documents(documents)
    loader.save_eval_pairs(eval_pairs)

    print(f"\nData generation complete!")
    print(f"- Documents: {len(documents)}")
    print(f"- Evaluation pairs: {len(eval_pairs)}")


if __name__ == "__main__":
    main()