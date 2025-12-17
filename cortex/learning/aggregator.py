"""
Federated Aggregator
====================
Aggregates anonymized stats from multiple nodes via weighted averaging.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NodeContribution:
    node_id: str
    timestamp: float
    stats: Dict[str, Any]
    weight: float = 1.0


class FederatedAggregator:
    MIN_CONTRIBUTIONS = 3
    AGGREGATION_INTERVAL = 3600.0

    def __init__(self):
        self._contributions: List[NodeContribution] = []
        self._last_aggregation = time.time()
        self._global_stats: Dict[str, Any] = {}

    def submit_contribution(self, node_id: str, stats: Dict[str, Any], sample_count: int = 1) -> None:
        node_hash = hashlib.sha256(node_id.encode()).hexdigest()[:16]
        self._contributions.append(
            NodeContribution(
                node_id=node_hash,
                timestamp=time.time(),
                stats=stats,
                weight=float(sample_count),
            )
        )
        logger.debug("[FED] Received contribution from %s", node_hash)

    def aggregate(self) -> Optional[Dict[str, Any]]:
        if len(self._contributions) < self.MIN_CONTRIBUTIONS:
            logger.debug("[FED] Not enough contributions: %s", len(self._contributions))
            return None

        total_weight = sum(c.weight for c in self._contributions)
        if total_weight <= 0:
            return None

        aggregated: Dict[str, float] = {}
        for contrib in self._contributions:
            for key, value in contrib.stats.items():
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + float(value) * contrib.weight / total_weight

        self._global_stats = {
            "aggregated_at": time.time(),
            "contributors": len(self._contributions),
            "total_samples": int(total_weight),
            **aggregated,
        }

        self._contributions.clear()
        self._last_aggregation = time.time()
        logger.info("[FED] Aggregated from %s nodes", self._global_stats["contributors"])
        return self._global_stats

    def get_global_stats(self) -> Dict[str, Any]:
        return self._global_stats

    def should_aggregate(self) -> bool:
        if len(self._contributions) < self.MIN_CONTRIBUTIONS:
            return False
        return time.time() - self._last_aggregation > self.AGGREGATION_INTERVAL

