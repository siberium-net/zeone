import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class VectorStore:
    """Простой векторный поиск на базе ChromaDB."""

    def __init__(self, persist_path: str = "vector_store"):
        self.persist_path = persist_path
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            "knowledge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    def embed_and_store(self, text_chunks: List[str], metadata: Dict[str, Any]) -> List[str]:
        ids = []
        for chunk in text_chunks:
            doc_id = f"doc_{len(ids)}_{len(chunk)}"
            self.collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[metadata],
            )
            ids.append(doc_id)
        logger.info(f"[VECTOR] Stored {len(ids)} chunks")
        return ids

    def query(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_texts=[query_text],
            n_results=limit,
        )
        hits = []
        if result and result.get("documents"):
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
                hits.append(
                    {
                        "text": doc,
                        "metadata": meta,
                        "score": dist,
                    }
                )
        return hits
