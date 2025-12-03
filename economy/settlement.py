import asyncio
import logging
from decimal import Decimal
from typing import Optional, Dict, Any

from economy.chain import ChainManager
from economy.ledger import Ledger
from nacl.signing import VerifyKey
from nacl.encoding import Base64Encoder

logger = logging.getLogger(__name__)


class SettlementManager:
    """Handles on-chain settlement of IOU debts."""

    def __init__(self, ledger: Ledger, chain: ChainManager, credit_rate: Decimal = Decimal("0.001")):
        """
        Args:
            ledger: Local ledger
            chain: ChainManager
            credit_rate: tokens per credit unit
        """
        self.ledger = ledger
        self.chain = chain
        self.credit_rate = credit_rate

    @staticmethod
    def verify_signed_address(address: str, signature_b64: str, peer_node_id: str) -> bool:
        """Verify wallet address signed by peer Ed25519 identity."""
        try:
            verify_key = VerifyKey(peer_node_id.encode("ascii"), encoder=Base64Encoder)
            verify_key.verify(address.encode("utf-8"), Base64Encoder.decode(signature_b64))
            return True
        except Exception:
            return False

    async def settle_debt(self, peer_id: str, peer_wallet: str, amount_credits: float) -> Optional[str]:
        """
        Convert credits to tokens and send settlement on-chain.
        Returns tx_hash or None on failure.
        """
        tokens = Decimal(str(amount_credits)) * self.credit_rate
        try:
            tx_hash = self.chain.transfer_token(peer_wallet, tokens)
            await self.ledger.record_settlement(peer_id, float(tokens), tx_hash)
            return tx_hash
        except Exception as e:
            logger.error(f"[SETTLEMENT] Failed to settle with {peer_id[:8]}: {e}")
            return None

    async def confirm_payment(self, tx_hash: str, peer_id: str, amount_tokens: float) -> bool:
        """
        Wait for on-chain confirmation then settle ledger.
        """
        ok = self.chain.wait_for_receipt(tx_hash)
        if ok:
            await self.ledger.record_payment(peer_id, amount_tokens, tx_hash)
        return ok
