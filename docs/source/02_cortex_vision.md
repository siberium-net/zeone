# Cortex Vision & Multimodal Pipeline

## Components
- **VisionEngine (Florence-2-large):** detailed captions, object detection, OCR with regions, brand extraction. Runs on GPU (float16) when available.
- **FaceIndexer (InsightFace):** 512-d embeddings per face, DBSCAN clustering → `person_cluster_*` centroids.
- **MediaFingerprint:** pHash for images; video signatures via sampled pHash frames; overlap detection for fragments.
- **VideoProcessor:** smart frame sampling (OpenCV), frame analysis via VisionEngine, video summary via OllamaAgent.
- **VectorStore:** ChromaDB + SentenceTransformer embeddings for RAG across text/image/video descriptions.

## Deduplication
- Image pHash; duplicates if Hamming distance < 5.
- Video signature = sequence of frame hashes; substring/overlap detection for fragments.
- Duplicate short-circuit: if pHash matches known set, heavy vision pass is skipped.

## Workflow
1. Ingest scans files → dedup check (pHash).
2. New assets: Florence caption/OD/OCR + brands; InsightFace embeddings; embeddings pushed to Chroma.
3. Knowledge entries stored in ledger (knowledge_base) with phash/faces/brands/metadata.
4. Compliance (PII + AI judge) sets SAFE/WARNING/BLOCKED before DHT/storage.
