import logging
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

try:
    from insightface.app import FaceAnalysis
    _INSIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _INSIGHT_AVAILABLE = False


class FaceIndexer:
    """Face embedding extractor and clustering."""

    def __init__(self, providers=None):
        self.app = None
        if not _INSIGHT_AVAILABLE:
            logger.warning("[FACES] insightface not available")
            return
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.app = FaceAnalysis(providers=providers)
            self.app.prepare(ctx_id=0)
            logger.info("[FACES] FaceAnalysis ready")
        except Exception as e:
            logger.warning(f"[FACES] Failed to init FaceAnalysis: {e}")
            self.app = None

        self.embeddings: List[np.ndarray] = []

    def get_face_embeddings(self, image_path: str) -> List[np.ndarray]:
        """Return list of 512-d embeddings for faces in image."""
        if not self.app:
            return []
        faces = self.app.get(image_path)
        embs = []
        for f in faces:
            if f.embedding is not None:
                embs.append(np.array(f.embedding, dtype=np.float32))
                self.embeddings.append(np.array(f.embedding, dtype=np.float32))
        return embs

    def cluster_faces(self, eps: float = 0.6, min_samples: int = 2) -> Dict[str, Any]:
        """Cluster accumulated embeddings and return cluster centers."""
        if not self.embeddings:
            return {}
        X = np.stack(self.embeddings)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
        labels = clustering.labels_
        clusters: Dict[str, List[np.ndarray]] = {}
        for label, emb in zip(labels, X):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(emb)
        centers = {}
        for label, vecs in clusters.items():
            centers[f"person_cluster_{label}"] = np.stack(vecs).mean(axis=0)
        logger.info(f"[FACES] Clustered into {len(centers)} identities")
        return centers
