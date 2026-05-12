"""
Data loader for documents and evaluation pairs.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from . import config


class DataLoader:
    """
    Utility class for loading documents and evaluation data.
    """

    def __init__(self, data_dir: str = config.DATA_DIR):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)

    def load_documents(self, filename: str = config.DOCUMENTS_FILE) -> List[Dict[str, Any]]:
        """
        Load documents from JSON file.

        Args:
            filename: Name of documents file

        Returns:
            List of document dictionaries
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Documents file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            documents = json.load(f)

        print(f"Loaded {len(documents)} documents from {filename}")
        return documents

    def load_eval_pairs(self, filename: str = config.EVAL_PAIRS_FILE) -> List[Dict[str, Any]]:
        """
        Load evaluation pairs from JSON file.

        Args:
            filename: Name of eval pairs file

        Returns:
            List of evaluation pairs
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Eval pairs file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            eval_pairs = json.load(f)

        print(f"Loaded {len(eval_pairs)} evaluation pairs from {filename}")
        return eval_pairs

    def extract_texts(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Extract text content from documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of document texts
        """
        return [doc.get("text", "") for doc in documents]

    def extract_doc_ids(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Extract document IDs.

        Args:
            documents: List of document dictionaries

        Returns:
            List of document IDs
        """
        return [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]

    def extract_queries(self, eval_pairs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract queries from evaluation pairs.

        Args:
            eval_pairs: List of evaluation pairs

        Returns:
            List of query strings
        """
        return [pair.get("query", "") for pair in eval_pairs]

    def extract_relevant_docs(self, eval_pairs: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Extract relevant document IDs from evaluation pairs.

        Args:
            eval_pairs: List of evaluation pairs

        Returns:
            List of relevant doc ID lists
        """
        return [pair.get("relevant_docs", []) for pair in eval_pairs]

    def save_documents(
        self,
        documents: List[Dict[str, Any]],
        filename: str = config.DOCUMENTS_FILE
    ) -> None:
        """
        Save documents to JSON file.

        Args:
            documents: List of document dictionaries
            filename: Name of output file
        """
        filepath = self.data_dir / filename

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(documents)} documents to {filename}")

    def save_eval_pairs(
        self,
        eval_pairs: List[Dict[str, Any]],
        filename: str = config.EVAL_PAIRS_FILE
    ) -> None:
        """
        Save evaluation pairs to JSON file.

        Args:
            eval_pairs: List of evaluation pairs
            filename: Name of output file
        """
        filepath = self.data_dir / filename

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(eval_pairs, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(eval_pairs)} evaluation pairs to {filename}")