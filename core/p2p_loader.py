import asyncio
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from core.distribution import build_manifest, read_chunk, write_chunk, CHUNK_SIZE
from core.events import event_bus

logger = logging.getLogger(__name__)


class P2PLoader:
    def __init__(self, kademlia=None, node=None, base_dir: Path = None):
        self.kademlia = kademlia
        self.node = node
        self.base_dir = base_dir or Path("data/models")

    async def ensure_model(
        self,
        model_id: str,
        fallback_http: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """
        Try to fetch model via P2P; fallback to HTTP if disabled/unavailable.
        """
        start_time = time.time()
        # Discover peers from DHT (simplified)
        peers = []
        if self.kademlia:
            key = model_id
            stored = await self.kademlia.storage.get(key.encode()) if hasattr(self.kademlia, "storage") else None
            if stored:
                peers.append(stored.value.decode())

        target_dir = self.base_dir / model_id
        target_dir.mkdir(parents=True, exist_ok=True)

        if not peers:
            logger.info("[P2P] No peers found for model %s", model_id)
            if fallback_http:
                await self._notify(progress_callback, model_id, 0, 0.0, 0, source="HTTP")
                return False
            return False

        # request manifest
        manifest = None
        for peer_id in peers:
            agent_mgr = getattr(self.node, "agent_manager", None)
            if not agent_mgr:
                continue
            resp = await agent_mgr.call_service(peer_id, "model_repo", {"action": "get_manifest", "model_id": model_id})
            if resp and "manifest" in resp:
                manifest = resp["manifest"]
                break
        if not manifest:
            logger.info("[P2P] Manifest unavailable from peers, fallback=%s", fallback_http)
            return False

        await self._download_from_manifest(manifest, peers, progress_callback, start_time)
        return True

    async def _download_from_manifest(
        self,
        manifest: Dict[str, Any],
        peers: List[str],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        start_time: float,
    ) -> None:
        files = manifest.get("files", [])
        total_size = sum(int(f["size"]) for f in files)
        downloaded = 0

        for f in files:
            downloaded += await self._download_file(f, peers, manifest["model_id"], progress_callback, start_time, downloaded, total_size)

    async def _download_file(
        self,
        file_info: Dict[str, Any],
        peers: List[str],
        model_id: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        start_time: float,
        downloaded_so_far: int,
        total_size: int,
    ) -> int:
        rel = file_info["name"]
        size = int(file_info["size"])
        sha = file_info["sha256"]
        target = self.base_dir / model_id / rel
        total_chunks = (size + CHUNK_SIZE - 1) // CHUNK_SIZE

        existing = target.exists()
        start_chunk = 0
        if existing:
            current_size = target.stat().st_size
            start_chunk = current_size // CHUNK_SIZE

        async def fetch_chunk(peer_id: str, idx: int) -> bytes:
            agent_mgr = getattr(self.node, "agent_manager", None)
            resp = await agent_mgr.call_service(
                peer_id,
                "model_repo",
                {"action": "get_chunk", "model_id": model_id, "file_name": rel, "chunk_index": idx},
            )
            return resp.get("data", b"")

        for idx in range(start_chunk, total_chunks):
            peer_id = peers[idx % len(peers)]
            data = await fetch_chunk(peer_id, idx)
            write_chunk(target, idx, data)
            bytes_done = downloaded_so_far + idx * CHUNK_SIZE
            percent = min(100.0, 100.0 * (bytes_done / total_size)) if total_size else 0
            elapsed = max(0.001, time.time() - start_time)
            speed = bytes_done / elapsed
            await self._notify(progress_callback, model_id, percent, speed, len(peers), source="P2P")

        if sha != self._sha256_file(target):
            logger.warning(f"[P2P] Hash mismatch for {target}")
        return size

    async def _notify(self, cb, model_id: str, percent: float, speed: float, peers: int, source: str):
        payload = {
            "model": model_id,
            "percent": round(percent, 2),
            "speed": speed,
            "peers": peers,
            "source": source,
        }
        if cb:
            try:
                res = cb(payload)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass
        await event_bus.broadcast("download_progress", payload)

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
