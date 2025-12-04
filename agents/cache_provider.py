"""
Cache Provider Agent
====================

Обслуживает запросы на кэшированные чанки по хэшу.
"""

import asyncio
import base64
from typing import Any, Tuple, Optional

from agents.manager import BaseAgent


class CacheProviderAgent(BaseAgent):
    """
    Бесплатный провайдер кэшированных данных.
    """

    def __init__(self, amplifier=None):
        self.amplifier = amplifier

    @property
    def service_name(self) -> str:
        return "cache_provider"

    @property
    def price_per_unit(self) -> float:
        return 0.0

    @property
    def description(self) -> str:
        return "Provides cached chunks by hash (barter/free)"

    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Execute is unused directly; use get_chunk instead.
        """
        return {"ok": True}, 0.0

    def attach_amplifier(self, amplifier) -> None:
        self.amplifier = amplifier

    async def get_chunk(self, chunk_hash: str) -> Optional[bytes]:
        if not self.amplifier or not chunk_hash:
            return None
        return await self.amplifier.get_chunk(chunk_hash)
