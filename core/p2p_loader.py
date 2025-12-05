"""
P2P Model Loader with Merkle Verification
==========================================

[SECURITY] This module downloads models from P2P peers with
chunk-level verification using Merkle proofs. Corrupted chunks
are rejected immediately without writing to disk.

Key Security Features:
- Each chunk verified against Merkle root before writing
- Malicious peers detected on first bad chunk
- No bandwidth wasted on corrupted files
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from core.distribution import write_chunk, CHUNK_SIZE
from core.security.merkle import MerkleTree, SecurityError
from core.events import event_bus

logger = logging.getLogger(__name__)


class P2PLoader:
    """
    Peer-to-peer model downloader with Merkle verification.
    
    [SECURITY] Each chunk is verified against the merkle_root from
    the manifest before being written to disk. A corrupted chunk
    is rejected immediately, and the peer may be banned.
    """
    
    def __init__(self, kademlia=None, node=None, base_dir: Path = None):
        self.kademlia = kademlia
        self.node = node
        self.base_dir = base_dir or Path("data/models")
        
        # Track bad peers for potential banning
        self._bad_peer_counts: Dict[str, int] = {}
        self._ban_threshold = 3  # Ban after 3 bad chunks

    async def ensure_model(
        self,
        model_id: str,
        fallback_http: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """
        Try to fetch model via P2P; fallback to HTTP if disabled/unavailable.
        
        Args:
            model_id: Model identifier to download
            fallback_http: Whether to fallback to HTTP if P2P fails
            progress_callback: Optional callback for progress updates
        
        Returns:
            True if model was successfully downloaded via P2P
        """
        start_time = time.time()
        
        # Discover peers from DHT
        peers = await self._discover_peers(model_id)
        
        target_dir = self.base_dir / model_id
        target_dir.mkdir(parents=True, exist_ok=True)

        if not peers:
            logger.info("[P2P] No peers found for model %s", model_id)
            if fallback_http:
                await self._notify(progress_callback, model_id, 0, 0.0, 0, source="HTTP")
            return False

        # Request manifest
        manifest = await self._fetch_manifest(model_id, peers)
        if not manifest:
            logger.info("[P2P] Manifest unavailable from peers, fallback=%s", fallback_http)
            return False

        try:
            await self._download_from_manifest(manifest, peers, progress_callback, start_time)
            return True
        except SecurityError as e:
            logger.error(f"[P2P] Security violation downloading {model_id}: {e}")
            return False

    async def _discover_peers(self, model_id: str) -> List[str]:
        """Discover peers serving this model from DHT."""
        peers = []
        if self.kademlia:
            key = model_id
            stored = await self.kademlia.storage.get(key.encode()) if hasattr(self.kademlia, "storage") else None
            if stored:
                peers.append(stored.value.decode())
        return peers

    async def _fetch_manifest(self, model_id: str, peers: List[str]) -> Optional[Dict[str, Any]]:
        """Fetch manifest from available peers."""
        for peer_id in peers:
            if self._is_peer_banned(peer_id):
                continue
            
            agent_mgr = getattr(self.node, "agent_manager", None)
            if not agent_mgr:
                continue
            
            try:
                resp = await agent_mgr.call_service(
                    peer_id, 
                    "model_repo", 
                    {"action": "get_manifest", "model_id": model_id}
                )
                if resp and "manifest" in resp:
                    return resp["manifest"]
            except Exception as e:
                logger.warning(f"[P2P] Failed to get manifest from {peer_id[:8]}...: {e}")
        
        return None

    async def _download_from_manifest(
        self,
        manifest: Dict[str, Any],
        peers: List[str],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        start_time: float,
    ) -> None:
        """Download all files in manifest with Merkle verification."""
        files = manifest.get("files", [])
        total_size = sum(int(f["size"]) for f in files)
        downloaded = 0

        for f in files:
            downloaded += await self._download_file(
                f, peers, manifest["model_id"], 
                progress_callback, start_time, downloaded, total_size
            )

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
        """
        Download a single file with chunk-level Merkle verification.
        
        [SECURITY] Each chunk is verified against merkle_root before
        writing. Bad chunks are rejected and peers may be banned.
        """
        rel = file_info["name"]
        size = int(file_info["size"])
        merkle_root = file_info["merkle_root"]  # Hex string
        total_chunks = int(file_info.get("total_chunks", (size + CHUNK_SIZE - 1) // CHUNK_SIZE))
        
        target = self.base_dir / model_id / rel
        
        # Resume support: start from last complete chunk
        start_chunk = 0
        if target.exists():
            current_size = target.stat().st_size
            start_chunk = current_size // CHUNK_SIZE

        logger.info(
            f"[P2P] Downloading {rel}: {total_chunks} chunks, "
            f"starting from {start_chunk}, merkle_root={merkle_root[:16]}..."
        )

        for idx in range(start_chunk, total_chunks):
            # Round-robin peer selection (skip banned peers)
            peer_id = self._select_peer(peers, idx)
            if not peer_id:
                raise SecurityError(
                    f"No available peers for chunk {idx}",
                    chunk_index=idx
                )
            
            try:
                chunk_data = await self._fetch_and_verify_chunk(
                    peer_id, model_id, rel, idx, merkle_root, total_chunks
                )
                
                # Verification passed - safe to write
                write_chunk(target, idx, chunk_data)
                
            except SecurityError as e:
                logger.error(f"[P2P] Chunk {idx} verification failed from {peer_id[:8]}...: {e}")
                self._record_bad_peer(peer_id)
                
                # Try another peer
                alternate_peer = self._select_peer(peers, idx, exclude=peer_id)
                if alternate_peer:
                    logger.info(f"[P2P] Retrying chunk {idx} from {alternate_peer[:8]}...")
                    chunk_data = await self._fetch_and_verify_chunk(
                        alternate_peer, model_id, rel, idx, merkle_root, total_chunks
                    )
                    write_chunk(target, idx, chunk_data)
                else:
                    raise
            
            # Progress update
            bytes_done = downloaded_so_far + (idx + 1) * CHUNK_SIZE
            percent = min(100.0, 100.0 * (bytes_done / total_size)) if total_size else 0
            elapsed = max(0.001, time.time() - start_time)
            speed = bytes_done / elapsed
            await self._notify(progress_callback, model_id, percent, speed, len(peers), source="P2P")

        logger.info(f"[P2P] Completed {rel}: all {total_chunks} chunks verified")
        return size

    async def _fetch_and_verify_chunk(
        self,
        peer_id: str,
        model_id: str,
        file_name: str,
        chunk_index: int,
        merkle_root_hex: str,
        total_chunks: int,
    ) -> bytes:
        """
        Fetch a chunk and verify it against Merkle root.
        
        Args:
            peer_id: Peer to fetch from
            model_id: Model identifier
            file_name: Relative file path
            chunk_index: Chunk to fetch
            merkle_root_hex: Expected Merkle root (hex)
            total_chunks: Total chunks in file
        
        Returns:
            Verified chunk data
        
        Raises:
            SecurityError: If chunk fails verification
        
        [SECURITY] This is the core security check. A corrupted or
        malicious chunk will fail verification and be rejected.
        """
        agent_mgr = getattr(self.node, "agent_manager", None)
        if not agent_mgr:
            raise SecurityError("No agent manager available", peer_id=peer_id)
        
        # Request chunk with proof
        resp = await agent_mgr.call_service(
            peer_id,
            "model_repo",
            {
                "action": "get_chunk",
                "model_id": model_id,
                "file_name": file_name,
                "chunk_index": chunk_index,
            },
        )
        
        if "error" in resp:
            raise SecurityError(
                f"Peer returned error: {resp['error']}",
                peer_id=peer_id,
                chunk_index=chunk_index
            )
        
        data = resp.get("data", b"")
        proof_hex = resp.get("proof", [])
        resp_total = resp.get("total_chunks", total_chunks)
        
        if not data:
            raise SecurityError(
                "Empty chunk data received",
                peer_id=peer_id,
                chunk_index=chunk_index
            )
        
        if not proof_hex:
            raise SecurityError(
                "No Merkle proof provided",
                peer_id=peer_id,
                chunk_index=chunk_index
            )
        
        # [SECURITY] Verify chunk against Merkle root
        is_valid = MerkleTree.verify_chunk_hex(
            chunk_data=data,
            proof_hex=proof_hex,
            root_hash_hex=merkle_root_hex,
            chunk_index=chunk_index,
            total_chunks=resp_total,
        )
        
        if not is_valid:
            raise SecurityError(
                f"Merkle verification failed for chunk {chunk_index}",
                peer_id=peer_id,
                chunk_index=chunk_index
            )
        
        return data

    def _select_peer(
        self, 
        peers: List[str], 
        chunk_index: int, 
        exclude: Optional[str] = None
    ) -> Optional[str]:
        """Select a non-banned peer using round-robin."""
        available = [p for p in peers if not self._is_peer_banned(p) and p != exclude]
        if not available:
            return None
        return available[chunk_index % len(available)]

    def _is_peer_banned(self, peer_id: str) -> bool:
        """Check if peer is banned due to bad chunks."""
        return self._bad_peer_counts.get(peer_id, 0) >= self._ban_threshold

    def _record_bad_peer(self, peer_id: str) -> None:
        """Record a bad chunk from peer."""
        count = self._bad_peer_counts.get(peer_id, 0) + 1
        self._bad_peer_counts[peer_id] = count
        
        if count >= self._ban_threshold:
            logger.warning(
                f"[P2P] Banning peer {peer_id[:8]}... after {count} bad chunks"
            )

    async def _notify(
        self, 
        cb: Optional[Callable], 
        model_id: str, 
        percent: float, 
        speed: float, 
        peers: int, 
        source: str
    ):
        """Send progress notification."""
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
