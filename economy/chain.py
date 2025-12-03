import os
import json
import logging
from decimal import Decimal
from typing import Optional

from eth_account import Account
from web3 import Web3
from web3.middleware import geth_poa_middleware

logger = logging.getLogger(__name__)


ERC20_ABI = json.loads(
    """
    [
      {"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},
      {"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"},
      {"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},
      {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
      {"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"}
    ]
    """
)


class ChainManager:
    """Simple EVM bridge for ERC-20 settlements."""

    def __init__(
        self,
        provider_uri: Optional[str] = None,
        private_key: Optional[str] = None,
        token_address: Optional[str] = None,
    ):
        self.provider_uri = provider_uri or os.getenv("WEB3_PROVIDER_URI", "")
        self.private_key = private_key or os.getenv("WALLET_PRIVATE_KEY", "")
        self.token_address = token_address or os.getenv("TOKEN_CONTRACT_ADDRESS", "")

        if not self.provider_uri:
            raise ValueError("WEB3_PROVIDER_URI not set")
        if not self.private_key:
            raise ValueError("WALLET_PRIVATE_KEY not set")
        if not self.token_address:
            raise ValueError("TOKEN_CONTRACT_ADDRESS not set")

        self.web3 = Web3(Web3.HTTPProvider(self.provider_uri))
        # Support PoA chains
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)

        self.account = Account.from_key(self.private_key)
        self.address = self.account.address

        self.token = self.web3.eth.contract(
            address=Web3.to_checksum_address(self.token_address), abi=ERC20_ABI
        )
        self.decimals = self.token.functions.decimals().call()
        self.symbol = self.token.functions.symbol().call()

    def _to_wei(self, amount_tokens: Decimal) -> int:
        return int(amount_tokens * (10 ** self.decimals))

    def get_eth_balance(self) -> Decimal:
        wei = self.web3.eth.get_balance(self.address)
        return Decimal(wei) / Decimal(10**18)

    def get_token_balance(self) -> Decimal:
        bal = self.token.functions.balanceOf(self.address).call()
        return Decimal(bal) / Decimal(10**self.decimals)

    def estimate_transfer(self, to_addr: str, amount_tokens: Decimal) -> int:
        to = Web3.to_checksum_address(to_addr)
        amount_wei = self._to_wei(amount_tokens)
        tx = self.token.functions.transfer(to, amount_wei).build_transaction(
            {"from": self.address, "nonce": self.web3.eth.get_transaction_count(self.address)}
        )
        return self.web3.eth.estimate_gas(tx)

    def transfer_token(self, to_addr: str, amount_tokens: Decimal) -> str:
        to = Web3.to_checksum_address(to_addr)
        amount_wei = self._to_wei(amount_tokens)
        nonce = self.web3.eth.get_transaction_count(self.address)
        tx = self.token.functions.transfer(to, amount_wei).build_transaction(
            {
                "from": self.address,
                "nonce": nonce,
                "gasPrice": self.web3.eth.gas_price,
            }
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        logger.info(f"[CHAIN] Sent transfer {amount_tokens} {self.symbol} to {to}: {tx_hash.hex()}")
        return tx_hash.hex()

    def wait_for_receipt(self, tx_hash: str, confirmations: int = 12, timeout: int = 300) -> bool:
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            start_block = receipt.blockNumber
            while self.web3.eth.block_number - start_block < confirmations:
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
                self.web3.provider.make_request("eth_chainId", [])
                self.web3.eth.wait_for_block(start_block + confirmations)
            return receipt.status == 1
        except Exception as e:
            logger.warning(f"[CHAIN] Receipt wait failed: {e}")
            return False
