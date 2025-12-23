import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from economy.ledger import Ledger

logger = logging.getLogger(__name__)


@dataclass
class RightsInfo:
    status: str = "UNKNOWN"
    beneficiary_address: Optional[str] = None
    revenue_share: bool = False
    similarity: float = 0.0

    @property
    def monetized(self) -> bool:
        return self.revenue_share or self.status in {"LICENSED", "COPYRIGHTED", "MONETIZED"}


@dataclass
class SeedingEconomyState:
    last_update: float = 0.0
    pending_bytes: float = 0.0


class SmartTorrentClient:
    """Smart wrapper for torrent handles with AI scrubbing and rights-aware seeding."""

    def __init__(
        self,
        ledger: Optional["Ledger"] = None,
        price_per_mb: float = 0.001,
        min_payout: float = 0.0001,
        revenue_share_ratio: float = 1.0,
    ):
        self.ledger = ledger
        self.price_per_mb = price_per_mb
        self.min_payout = min_payout
        self.revenue_share_ratio = revenue_share_ratio
        self._economy_state: Dict[str, SeedingEconomyState] = {}

    def prioritize_pieces(
        self,
        handle: Any,
        piece_indices: Iterable[int],
        priority: int = 7,
    ) -> None:
        for index in piece_indices:
            try:
                handle.piece_priority(int(index), int(priority))
            except Exception as e:
                logger.warning(f"[TORRENT] Failed to set priority for piece {index}: {e}")

    def set_play_now(self, handle: Any, enabled: bool = True) -> None:
        try:
            handle.set_sequential_download(bool(enabled))
        except Exception as e:
            logger.warning(f"[TORRENT] Failed to set sequential mode: {e}")

    def get_upload_rate(self, handle: Any) -> float:
        try:
            status = handle.status()
            rate = getattr(status, "upload_rate", 0)
            return float(rate or 0)
        except Exception:
            return 0.0

    async def tick(
        self,
        handle: Any,
        torrent_id: str,
        rights: RightsInfo,
    ) -> Optional[float]:
        upload_rate = self.get_upload_rate(handle)
        return await self.update_seeding_economy(torrent_id, upload_rate, rights)

    async def update_seeding_economy(
        self,
        torrent_id: str,
        upload_rate: float,
        rights: RightsInfo,
    ) -> Optional[float]:
        if not self.ledger or not rights.monetized or not rights.beneficiary_address:
            return None

        now = time.monotonic()
        state = self._economy_state.setdefault(torrent_id, SeedingEconomyState())
        if state.last_update > 0:
            delta = now - state.last_update
            if delta > 0:
                state.pending_bytes += max(upload_rate, 0.0) * delta
        state.last_update = now

        tokens = (
            state.pending_bytes / (1024 * 1024)
        ) * self.price_per_mb * self.revenue_share_ratio
        if tokens < self.min_payout:
            return None

        state.pending_bytes = 0.0
        tx_hash = uuid.uuid4().hex
        try:
            await self.ledger.record_payment(rights.beneficiary_address, tokens, tx_hash)
            logger.info(
                f"[TORRENT] Revenue share {tokens:.6f} to {rights.beneficiary_address} ({torrent_id})"
            )
            return tokens
        except Exception as e:
            logger.warning(f"[TORRENT] Revenue share failed: {e}")
            return None
