"""
Repository Agent - Serves model files with Merkle verification
===============================================================

[SECURITY] This agent serves model chunks along with Merkle proofs,
enabling clients to verify each chunk independently.

Key Features:
- Lazy Merkle tree building and caching
- Returns chunk data + Merkle proof
- Publishes available models to DHT
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from agents.manager import BaseAgent
from core.distribution import (
    build_manifest,
    read_chunk,
    build_file_merkle_tree,
    CHUNK_SIZE,
)
from core.security.merkle import MerkleTree
from core.dht.node import KademliaNode  # type: ignore
from core.dht.routing import key_to_id  # type: ignore

logger = logging.getLogger(__name__)


class RepoAgent(BaseAgent):
    """
    Model Repository Agent.
    
    [SECURITY] Serves chunks with Merkle proofs for verification.
    The client can verify each chunk against the manifest's merkle_root
    without trusting the server.
    
    Cached trees prevent rebuilding on every chunk request.
    """
    
    service_name = "model_repo"

    def __init__(self, base_dir: Path, kademlia: KademliaNode = None, node_id: str = ""):
        super().__init__()
        self.base_dir = base_dir
        self.kademlia = kademlia
        self.node_id = node_id
        
        # Cache: (model_id, file_name) -> MerkleTree
        self._tree_cache: Dict[Tuple[str, str], MerkleTree] = {}
        
        asyncio.create_task(self._publish())

    async def _publish(self):
        """Publish available models to DHT for discovery."""
        if not self.kademlia:
            return
        # For simplicity publish only top-level dirs
        for model_dir in self.base_dir.glob("*"):
            if model_dir.is_dir():
                key = key_to_id(model_dir.name)
                await self.kademlia.storage.store(key, self.node_id.encode(), self.kademlia.local_id)
                logger.debug(f"[REPO] Published model: {model_dir.name}")

    def _get_or_build_tree(self, model_id: str, file_name: str) -> Optional[MerkleTree]:
        """
        Get cached Merkle tree or build a new one.
        
        Args:
            model_id: Model identifier
            file_name: Relative file path within model directory
        
        Returns:
            MerkleTree or None if file doesn't exist
        
        [PERFORMANCE] Trees are cached to avoid rebuilding on every
        chunk request. Cache key is (model_id, file_name).
        """
        cache_key = (model_id, file_name)
        
        if cache_key in self._tree_cache:
            return self._tree_cache[cache_key]
        
        path = self.base_dir / model_id / file_name
        if not path.exists():
            return None
        
        try:
            tree = build_file_merkle_tree(path)
            self._tree_cache[cache_key] = tree
            logger.debug(f"[REPO] Built Merkle tree for {model_id}/{file_name}: {tree.leaf_count} chunks")
            return tree
        except Exception as e:
            logger.warning(f"[REPO] Failed to build tree for {model_id}/{file_name}: {e}")
            return None

    def _clear_cache(self, model_id: Optional[str] = None):
        """
        Clear Merkle tree cache.
        
        Args:
            model_id: If specified, only clear trees for this model.
                      If None, clear entire cache.
        """
        if model_id is None:
            self._tree_cache.clear()
        else:
            keys_to_remove = [k for k in self._tree_cache if k[0] == model_id]
            for k in keys_to_remove:
                del self._tree_cache[k]

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming requests.
        
        Actions:
        - get_manifest: Return manifest with merkle_roots
        - get_chunk: Return chunk data + Merkle proof
        - clear_cache: Clear tree cache (admin)
        """
        action = request.get("action")
        model_id = request.get("model_id")
        
        if action == "get_manifest":
            return self._handle_get_manifest(model_id)
        
        if action == "get_chunk":
            file_name = request.get("file_name")
            chunk_index = int(request.get("chunk_index", 0))
            return self._handle_get_chunk(model_id, file_name, chunk_index)
        
        if action == "clear_cache":
            self._clear_cache(model_id)
            return {"status": "ok"}
        
        return {"error": "unknown_action"}

    def _handle_get_manifest(self, model_id: str) -> Dict[str, Any]:
        """
        Return manifest for a model.
        
        [SECURITY] Manifest contains merkle_root per file, allowing
        clients to verify chunks without the full hash list.
        """
        try:
            manifest = build_manifest(model_id, self.base_dir)
            return {"manifest": manifest}
        except Exception as e:
            logger.warning(f"[REPO] Manifest build failed for {model_id}: {e}")
            return {"error": str(e)}

    def _handle_get_chunk(
        self,
        model_id: str,
        file_name: str,
        chunk_index: int,
    ) -> Dict[str, Any]:
        """
        Return chunk data with Merkle proof.
        
        Response format:
        {
            "data": bytes,           # Raw chunk bytes
            "proof": [(hex, bool)],  # Merkle proof
            "total_chunks": int,     # Total chunks in file
            "chunk_index": int       # Echoed back
        }
        
        [SECURITY] The proof allows the client to verify this chunk
        against the merkle_root from the manifest, without trusting
        this server.
        """
        path = self.base_dir / model_id / file_name
        
        if not path.exists():
            return {"error": f"file_not_found: {model_id}/{file_name}"}
        
        # Get or build Merkle tree
        tree = self._get_or_build_tree(model_id, file_name)
        if tree is None:
            return {"error": "failed_to_build_merkle_tree"}
        
        # Validate chunk index
        if chunk_index < 0 or chunk_index >= tree.leaf_count:
            return {
                "error": f"chunk_index_out_of_range: {chunk_index} not in [0, {tree.leaf_count})"
            }
        
        try:
            # Read chunk data
            data = read_chunk(path, chunk_index)
            
            # Get Merkle proof (hex format for serialization)
            proof = tree.get_proof_hex(chunk_index)
            
            return {
                "data": data,
                "proof": proof,
                "total_chunks": tree.leaf_count,
                "chunk_index": chunk_index,
            }
        except Exception as e:
            logger.warning(f"[REPO] Chunk read failed for {model_id}/{file_name}[{chunk_index}]: {e}")
            return {"error": str(e)}
