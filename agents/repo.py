import asyncio
import logging
from typing import Dict, Any
from pathlib import Path

from agents.manager import BaseAgent
from core.distribution import build_manifest, read_chunk
from core.dht.node import KademliaNode  # type: ignore
from core.dht.routing import key_to_id  # type: ignore

logger = logging.getLogger(__name__)


class RepoAgent(BaseAgent):
    service_name = "model_repo"

    def __init__(self, base_dir: Path, kademlia: KademliaNode = None, node_id: str = ""):
        super().__init__()
        self.base_dir = base_dir
        self.kademlia = kademlia
        self.node_id = node_id
        asyncio.create_task(self._publish())

    async def _publish(self):
        if not self.kademlia:
            return
        # For simplicity publish only top-level dirs
        for model_dir in self.base_dir.glob("*"):
            if model_dir.is_dir():
                key = key_to_id(model_dir.name)
                await self.kademlia.storage.store(key, self.node_id.encode(), self.kademlia.local_id)

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        model_id = request.get("model_id")
        if action == "get_manifest":
            manifest = build_manifest(model_id, self.base_dir)
            return {"manifest": manifest}
        if action == "get_chunk":
            file_name = request.get("file_name")
            chunk_index = int(request.get("chunk_index", 0))
            path = self.base_dir / model_id / file_name
            try:
                data = read_chunk(path, chunk_index)
                return {"data": data}
            except Exception as e:
                logger.warning(f"[REPO] chunk read failed: {e}")
                return {"error": str(e)}
        return {"error": "unknown_action"}
