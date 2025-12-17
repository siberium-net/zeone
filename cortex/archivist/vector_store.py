import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb  # type: ignore
    from chromadb.utils import embedding_functions  # type: ignore
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    embedding_functions = None  # type: ignore[assignment]

from core.p2p_loader import P2PLoader

logger = logging.getLogger(__name__)


class VectorStore:
    """Простой векторный поиск на базе ChromaDB."""

    def __init__(self, persist_path: str = "vector_store"):
        self.persist_path = persist_path
        self.safe_mode = False
        self.embedding_fn = None

        if chromadb is None:
            self.safe_mode = True
            self.client = None
            self.collection = None
            logger.warning("[VECTOR] chromadb not installed; vector store disabled")
            return

        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_path)

        try:
            # Ensure model via P2P
            loader = P2PLoader(base_dir=Path("data/models"))
            try:
                asyncio.get_running_loop()
                asyncio.create_task(loader.ensure_model("sentence-transformers/all-MiniLM-L6-v2"))
            except RuntimeError:
                # No running loop (e.g. sync init); model ensure is best-effort.
                pass

            if embedding_functions is not None:
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        except Exception as e:
            self.safe_mode = True
            logger.warning(f"[VECTOR] Embeddings unavailable: {e}")

        self.collection = self.client.get_or_create_collection(
            "knowledge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn if self.embedding_fn else None,
        )

    def embed_and_store(self, text_chunks: List[str], metadata: Dict[str, Any]) -> List[str]:
        if self.safe_mode or not self.embedding_fn or not self.collection:
            logger.warning("[VECTOR] Safe mode active; skipping embedding store")
            return []
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
        if self.safe_mode or not self.embedding_fn or not self.collection:
            logger.warning("[VECTOR] Safe mode active; vector queries disabled")
            return []
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
