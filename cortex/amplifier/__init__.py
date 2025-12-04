"""
Traffic Amplifier - DHT-backed chunk cache and lookup.
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .cache import AmplifierCache
from core.transport import Message, MessageType

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
        ledger=None,
        max_advertised: int = 32,
    ):
        self.node = node
        self.kademlia = kademlia
        self.cache = cache or AmplifierCache()
        self.ledger = ledger
        self.max_advertised = max_advertised
        self._advertised: set[str] = set()
        self._lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future] = {}

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
        """Try local cache, then peers via DHT."""
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
    # Networking
    # ------------------------------------------------------------------
    async def _fetch_from_provider(self, chunk_hash: str, provider: Dict[str, Any]) -> Optional[bytes]:
        """
        Запросить chunk у конкретного пира через CACHE_REQUEST/CACHE_RESPONSE.
        """
        if not self.node:
            return None

        peer_id = provider.get("node_id") if isinstance(provider, dict) else provider
        if not peer_id:
            return None

        # Ensure connection
        peer = self.node.peer_manager.get_peer(peer_id)
        if not peer:
            host = provider.get("host")
            port = provider.get("port")
            if host and port:
                try:
                    peer = await self.node.connect_to_peer(host, port)
                except Exception:
                    peer = None
            if not peer:
                return None

        # Prepare future
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending[chunk_hash] = fut

        request = Message(
            type=MessageType.CACHE_REQUEST,
            payload={"hash": chunk_hash},
            sender_id=self.node.node_id,
        )
        sent = await self.node.send_to(peer_id, request, with_accounting=False)
        if not sent:
            self._pending.pop(chunk_hash, None)
            return None

        try:
            response: Dict[str, Any] = await asyncio.wait_for(fut, timeout=5.0)
        except asyncio.TimeoutError:
            self._pending.pop(chunk_hash, None)
            return None

        self._pending.pop(chunk_hash, None)

        if not response.get("found"):
            return None
        data_b64 = response.get("data")
        if not data_b64:
            return None
        try:
            data = base64.b64decode(data_b64)
        except Exception:
            return None

        # Verify hash
        if hashlib.sha256(data).hexdigest() != chunk_hash:
            try:
                if self.ledger:
                    await self.ledger.update_trust_score(peer_id, "invalid_message")
            except Exception:
                pass
            return None

        return data

    async def handle_cache_response(self, payload: Dict[str, Any]) -> None:
        """Handle incoming CACHE_RESPONSE from protocol handler."""
        chunk_hash = payload.get("hash")
        if not chunk_hash:
            return
        fut = self._pending.get(chunk_hash)
        if fut and not fut.done():
            fut.set_result(payload)


__all__ = ["Amplifier", "AmplifierCache"]
