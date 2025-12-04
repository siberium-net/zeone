"""
VPN pathfinder and smart routing via DHT.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from config import config
from core.protocol import PingPongHandler
from core.transport import SimpleTransport
from core.routing import VpnRouteStore, VpnRoute

logger = logging.getLogger(__name__)

VPN_SERVICE_KEY = "service:vpn_exit"


class VpnPathfinder:
    """
    DHT-based discovery and selection of VPN exit nodes with persistence.
    """

    def __init__(
        self,
        node,
        kademlia,
        ledger=None,
        route_store: Optional[VpnRouteStore] = None,
        blacklist_ttl: float = 300.0,
    ):
        self.node = node
        self.kademlia = kademlia
        self.ledger = ledger
        self.route_store = route_store or VpnRouteStore()
        self.blacklist_ttl = blacklist_ttl
        self._blacklist: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await self.route_store.initialize()

    # ------------------------------------------------------------------
    # DHT publish/discovery
    # ------------------------------------------------------------------
    async def publish_exit(
        self,
        ip: Optional[str] = None,
        geo: str = "US",
        price: float = 0.1,
        score: float = 0.9,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish this node as VPN exit in DHT."""
        if not self.kademlia:
            logger.warning("[VPN] Kademlia not available, cannot publish exit")
            return

        entry = {
            "node_id": getattr(self.node, "node_id", ""),
            "ip": ip or getattr(self.node, "host", ""),
            "port": getattr(self.node, "port", config.network.default_port),
            "geo": geo,
            "price": price,
            "score": score,
            "ts": time.time(),
        }
        if extra:
            entry.update(extra)

        current = await self._fetch_candidates_raw()
        # Deduplicate by node_id
        filtered = [c for c in current if c.get("node_id") != entry["node_id"]]
        filtered.append(entry)

        try:
            await self.kademlia.dht_put(VPN_SERVICE_KEY, json.dumps(filtered).encode())
            logger.info("[VPN] Published exit info to DHT")
        except Exception as e:
            logger.warning(f"[VPN] Failed to publish exit info: {e}")

    async def discover_exits(self, target_region: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch VPN exit candidates from DHT with optional geo filter."""
        candidates = await self._fetch_candidates_raw()
        if not candidates:
            return []

        region = (target_region or "any").lower()
        if region and region != "any":
            candidates = [c for c in candidates if str(c.get("geo", "")).lower() == region]

        return candidates

    async def _fetch_candidates_raw(self) -> List[Dict[str, Any]]:
        if not self.kademlia:
            return []
        try:
            raw = await self.kademlia.dht_get(VPN_SERVICE_KEY)
        except Exception as e:
            logger.debug(f"[VPN] DHT get failed: {e}")
            return []

        if not raw:
            return []

        try:
            decoded = json.loads(raw.decode())
            if isinstance(decoded, dict):
                return [decoded]
            if isinstance(decoded, list):
                return decoded
        except Exception as e:
            logger.warning(f"[VPN] Failed to decode DHT value: {e}")
        return []

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    async def pick_exit(self, target_country: str = "any", strategy: str = "fastest") -> Optional[Dict[str, Any]]:
        """Choose the best exit node using cache + DHT discovery."""
        await self.initialize()
        target_key = (target_country or "any").lower()

        candidates = await self.discover_exits(target_key)
        if not candidates:
            logger.info("[VPN] No VPN exits found in DHT")
            return None

        cached = await self.route_store.get_recent_route(target_key)
        if cached and not self._is_blacklisted(cached.exit_node_id):
            preferred = next((c for c in candidates if c.get("node_id") == cached.exit_node_id), None)
            if preferred:
                probed = await self._probe_candidate(preferred)
                if probed:
                    await self.route_store.upsert_route(target_key, preferred["node_id"], probed["latency"], success=True)
                    return {**preferred, **probed}
                await self.mark_failure(target_key, preferred.get("node_id", ""))

        return await self.select_best_exit(candidates, strategy=strategy, target_country=target_key)

    async def select_best_exit(
        self,
        candidates: List[Dict[str, Any]],
        strategy: str = "fastest",
        target_country: str = "any",
    ) -> Optional[Dict[str, Any]]:
        """Probe candidates and pick the best according to strategy."""
        filtered = [c for c in candidates if not self._is_blacklisted(c.get("node_id", ""))]
        if not filtered:
            return None

        probe_tasks = [self._probe_candidate(c) for c in filtered]
        results = await asyncio.gather(*probe_tasks, return_exceptions=True)

        scored = []
        for cand, res in zip(filtered, results):
            if isinstance(res, Exception) or res is None:
                await self.mark_failure(target_country, cand.get("node_id", ""))
                continue

            trust = await self._get_trust_score(cand.get("node_id", ""), cand.get("score"))
            price = float(cand.get("price", 1.0) or 1.0)
            scored.append({
                **cand,
                **res,
                "trust": trust,
                "price": price,
            })

        if not scored:
            return None

        strategy = strategy or "fastest"
        if strategy == "cheapest":
            best = min(scored, key=lambda c: (c["price"], c.get("latency", 9999)))
        elif strategy == "reliable":
            best = max(scored, key=lambda c: (c.get("trust", 0.0), -c.get("latency", 0)))
        else:
            best = min(scored, key=lambda c: (c.get("latency", 9999), c["price"]))

        await self.route_store.upsert_route(
            target_country,
            best.get("node_id", ""),
            best.get("latency", 0.0),
            success=True,
        )
        return best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_blacklisted(self, node_id: str) -> bool:
        if not node_id:
            return True
        expires = self._blacklist.get(node_id)
        if expires and expires > time.time():
            return True
        if expires:
            self._blacklist.pop(node_id, None)
        return False

    async def mark_failure(self, target_country: str, node_id: str) -> None:
        """Penalty + temporary blacklist for failed nodes."""
        if not node_id:
            return
        self._blacklist[node_id] = time.time() + self.blacklist_ttl
        await self.route_store.record_failure(target_country, node_id)

    async def _get_trust_score(self, node_id: str, fallback: Optional[float] = None) -> float:
        if self.ledger and node_id:
            try:
                return await self.ledger.get_trust_score(node_id)
            except Exception:
                pass
        return fallback if fallback is not None else 0.5

    async def _probe_candidate(self, candidate: Dict[str, Any], timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Measure latency via lightweight PING over raw connection."""
        host = candidate.get("ip") or candidate.get("host")
        port = int(candidate.get("port", config.network.default_port))
        node_id = candidate.get("node_id", "")
        if not host:
            return None

        start = time.time()
        reader = writer = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout,
            )
            ping = PingPongHandler.create_ping(self.node.crypto)
            writer.write(SimpleTransport.pack(ping))
            await writer.drain()

            header = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
            length = SimpleTransport.unpack_length(header)
            payload = await asyncio.wait_for(reader.readexactly(length), timeout=timeout)
            pong = SimpleTransport.unpack(payload)

            if not PingPongHandler.verify_pong(ping, pong, self.node.crypto):
                raise RuntimeError("Invalid PONG")

            latency = time.time() - start
            return {"latency": latency}
        except Exception as e:
            logger.debug(f"[VPN] Probe failed for {host}:{port} ({node_id[:8]}...): {e}")
            await self.mark_failure(candidate.get("geo", "any"), node_id)
            return None
        finally:
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
