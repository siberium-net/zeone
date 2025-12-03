import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any

CHUNK_SIZE = 1024 * 1024  # 1 MB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(model_id: str, base_dir: Path) -> Dict[str, Any]:
    """
    Scan model directory and build manifest with hashes and sizes.
    """
    root = base_dir / model_id
    files: List[Dict[str, Any]] = []
    if not root.exists():
        return {"model_id": model_id, "files": []}
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            files.append(
                {
                    "name": rel,
                    "size": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
    return {"model_id": model_id, "files": files}


def read_chunk(path: Path, chunk_index: int) -> bytes:
    with path.open("rb") as f:
        f.seek(chunk_index * CHUNK_SIZE)
        return f.read(CHUNK_SIZE)


def write_chunk(path: Path, chunk_index: int, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r+b" if path.exists() else "wb") as f:
        f.seek(chunk_index * CHUNK_SIZE)
        f.write(data)
