"""
Archivist - ingestion pipeline for local documents.

Modules:
- scanner: async file scanning and text extraction
- processor: cognitive processing via LLM (OllamaAgent)
- vector_store: embedding + vector search backend
"""

from .scanner import AsyncFileScanner
from .processor import DocumentProcessor
from .vector_store import VectorStore

__all__ = [
    "AsyncFileScanner",
    "DocumentProcessor",
    "VectorStore",
]
