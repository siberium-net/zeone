"""
Traffic Amplifier - DHT-backed chunk cache and lookup.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from cortex.amplifier.cache import AmplifierCache

logger = logging.getLogger(__name__)


class Amplifier:
    """
    Correlates traffic chunks, stores them locally, and advertises availability.
    """

    def __init__(
        self,
        node=None,
        kademlia=None,
        cache: Optional[AmplifierCache] = None,
        max_advertised: int = 32,
    ):
        self.node = node
        self.kademlia = kademlia
        self.cache = cache or AmplifierCache()
        self.max_advertised = max_advertised
        self._advertised: set[str] = set()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def record_chunk(self, chunk_hash: str, data: bytes) -> None:
        """Store chunk and advertise availability."""
        if not chunk_hash or not data:
            return
        await self.cache.put(chunk_hash, data)
        await self._advertise(chunk_hash)

    async def get_chunk(self, chunk_hash: str) -> Optional[bytes]:
        return await self.cache.get(chunk_hash)

    async def fetch_or_get(self, chunk_hash: str) -> Optional[bytes]:
        """Try local cache, then peers via DHT (placeholder fetch)."""
        local = await self.cache.get(chunk_hash)
        if local:
            return local
        providers = await self.find_providers(chunk_hash)
        for provider in providers:
            data = await self._fetch_from_provider(chunk_hash, provider)
            if data:
                await self.record_chunk(chunk_hash, data)
                return data
        return None

    # ------------------------------------------------------------------
    # DHT helpers
    # ------------------------------------------------------------------
    async def _advertise(self, chunk_hash: str) -> None:
        if not self.kademlia or chunk_hash in self._advertised:
            return
        if not self.node:
            return
        entry = {
            "node_id": getattr(self.node, "node_id", ""),
            "host": getattr(self.node, "host", ""),
            "port": getattr(self.node, "port", 0),
            "ts": time.time(),
        }
        try:
            key = f"cache:{chunk_hash}"
            current_raw = await self.kademlia.dht_get(key)
            current: List[Dict[str, Any]] = []
            if current_raw:
                current = json.loads(current_raw.decode())
                if isinstance(current, dict):
                    current = [current]
            current = [c for c in current if c.get("node_id") != entry["node_id"]]
            current.append(entry)
            current = sorted(current, key=lambda c: c.get("ts", 0), reverse=True)[: self.max_advertised]
            await self.kademlia.dht_put(key, json.dumps(current).encode())
            self._advertised.add(chunk_hash)
        except Exception as e:
            logger.debug(f"[AMPLIFIER] Advertise failed: {e}")

    async def find_providers(self, chunk_hash: str) -> List[Dict[str, Any]]:
        if not self.kademlia:
            return []
        try:
            raw = await self.kademlia.dht_get(f"cache:{chunk_hash}")
            if not raw:
                return []
            decoded = json.loads(raw.decode())
            if isinstance(decoded, dict):
                return [decoded]
            if isinstance(decoded, list):
                return decoded
        except Exception as e:
            logger.debug(f"[AMPLIFIER] find_providers failed: {e}")
        return []

    # ------------------------------------------------------------------
    # Networking placeholder
    # ------------------------------------------------------------------
    async def _fetch_from_provider(self, chunk_hash: str, provider: Dict[str, Any]) -> Optional[bytes]:
        """
        Placeholder for peer-to-peer fetch.
        TODO: implement hole punching / direct transfer protocol.
        """
        return None
