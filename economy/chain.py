"""
Siberium Chain Manager
======================

[SIBERIUM] Native SIBR token operations for ZEONE network.

Networks:
- Mainnet: ChainID 111111, RPC https://rpc.siberium.net
- Testnet: ChainID 111000, RPC https://rpc.test.siberium.net

[KEY DIFFERENCE] SIBR is the native gas token (like ETH on Ethereum).
No ERC-20 contract - uses native transfers via web3.eth.send_transaction.

[USAGE]
    manager = SiberiumManager(
        rpc_url="https://rpc.test.siberium.net",
        private_key="0x...",
    )
    
    # Check balance
    balance = manager.get_balance()
    
    # Transfer native SIBR
    tx_hash = await manager.transfer(to_addr, amount)
    
    # Interact with Settlement contract
    await manager.deposit_stake(amount)
"""

import os
import time
import logging
from decimal import Decimal
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from eth_account import Account
from eth_account.messages import encode_defunct

logger = logging.getLogger(__name__)

# Lazy import
Web3 = None


def _ensure_web3():
    global Web3
    if Web3 is None:
        from web3 import Web3 as _Web3
        Web3 = _Web3
    return Web3


# ============================================================================
# Siberium Network Configuration
# ============================================================================

@dataclass
class SiberiumNetwork:
    """Siberium network configuration."""
    name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    symbol: str = "SIBR"
    decimals: int = 18


# Network configurations
SIBERIUM_MAINNET = SiberiumNetwork(
    name="Siberium Mainnet",
    chain_id=111111,
    rpc_url="https://rpc.siberium.net",
    explorer_url="https://explorer.siberium.net",
)

SIBERIUM_TESTNET = SiberiumNetwork(
    name="Siberium Testnet",
    chain_id=111000,
    rpc_url="https://rpc.test.siberium.net",
    explorer_url="https://explorer.test.siberium.net",
)

# Local development network
SIBERIUM_LOCAL = SiberiumNetwork(
    name="Siberium Local",
    chain_id=111000,  # Same as testnet
    rpc_url="http://localhost:8545",
    explorer_url="",
)

NETWORKS = {
    111111: SIBERIUM_MAINNET,
    111000: SIBERIUM_TESTNET,
}


# ============================================================================
# Settlement Contract ABI (Native SIBR version)
# ============================================================================

SETTLEMENT_ABI = [
    # Deposit (payable)
    {"inputs": [], "name": "deposit", "outputs": [], "stateMutability": "payable", "type": "function"},
    
    # Unstake flow
    {"inputs": [{"name": "amount", "type": "uint256"}], "name": "requestUnstake", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "withdraw", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "cancelUnstake", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "amount", "type": "uint256"}], "name": "withdrawBalance", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    
    # Settlement
    {"inputs": [{"name": "payer", "type": "address"}, {"name": "amount", "type": "uint256"}, {"name": "nonce", "type": "uint256"}, {"name": "deadline", "type": "uint256"}, {"name": "signature", "type": "bytes"}], "name": "claim", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    
    # View functions
    {"inputs": [{"name": "account", "type": "address"}], "name": "stakedBalance", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalStaked", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "getAccountInfo", "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getStats", "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "getNextNonce", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "useNonce", "outputs": [{"type": "uint256"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "payer", "type": "address"}, {"name": "payee", "type": "address"}, {"name": "amount", "type": "uint256"}, {"name": "nonce", "type": "uint256"}, {"name": "deadline", "type": "uint256"}], "name": "getSettlementDigest", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "DOMAIN_SEPARATOR", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "feeRate", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "unstakeDelay", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getNetworkInfo", "outputs": [{"type": "uint256"}, {"type": "bool"}, {"type": "bool"}], "stateMutability": "view", "type": "function"},
    
    # Events
    {"anonymous": False, "inputs": [{"indexed": True, "name": "user", "type": "address"}, {"indexed": False, "name": "amount", "type": "uint256"}, {"indexed": False, "name": "totalStaked", "type": "uint256"}], "name": "Deposited", "type": "event"},
    {"anonymous": False, "inputs": [{"indexed": True, "name": "user", "type": "address"}, {"indexed": False, "name": "amount", "type": "uint256"}], "name": "Withdrawn", "type": "event"},
    {"anonymous": False, "inputs": [{"indexed": True, "name": "payer", "type": "address"}, {"indexed": True, "name": "payee", "type": "address"}, {"indexed": False, "name": "amount", "type": "uint256"}, {"indexed": False, "name": "fee", "type": "uint256"}, {"indexed": False, "name": "nonce", "type": "uint256"}], "name": "SettlementClaimed", "type": "event"},
]


# ============================================================================
# Siberium Chain Manager
# ============================================================================

class SiberiumManager:
    """
    Manager for Siberium blockchain operations.
    
    [NATIVE TOKEN] SIBR is the native gas token:
    - Balance: web3.eth.get_balance()
    - Transfer: web3.eth.send_transaction()
    - No ERC-20 contract needed
    
    [SETTLEMENT] Interacts with ZEOSettlement contract:
    - deposit() - Stake native SIBR
    - claim() - Claim settlement payments
    """
    
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        settlement_address: Optional[str] = None,
        network: Optional[SiberiumNetwork] = None,
    ):
        """
        Args:
            rpc_url: Siberium RPC URL (or from SIBERIUM_RPC_URL env)
            private_key: Wallet private key (or from WALLET_PRIVATE_KEY env)
            settlement_address: Settlement contract address (or from SETTLEMENT_ADDRESS env)
            network: Network configuration (auto-detected if None)
        """
        Web3 = _ensure_web3()
        
        # Load from environment if not provided
        self.rpc_url = rpc_url or os.getenv("SIBERIUM_RPC_URL", SIBERIUM_TESTNET.rpc_url)
        self.private_key = private_key or os.getenv("WALLET_PRIVATE_KEY", "")
        self.settlement_address = settlement_address or os.getenv("SETTLEMENT_ADDRESS", "")
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Add PoA middleware for Siberium
        from web3.middleware import geth_poa_middleware
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Detect network
        chain_id = self.w3.eth.chain_id
        self.network = network or NETWORKS.get(chain_id, SIBERIUM_TESTNET)
        
        logger.info(f"[SIBERIUM] Connected to {self.network.name} (ChainID: {chain_id})")
        
        # Initialize account if private key provided
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        # Initialize settlement contract if address provided
        self.settlement = None
        if self.settlement_address:
            self._connect_settlement()
    
    def _connect_settlement(self) -> None:
        """Connect to Settlement contract."""
        Web3 = _ensure_web3()
        self.settlement = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.settlement_address),
            abi=SETTLEMENT_ABI,
        )
        logger.info(f"[SIBERIUM] Connected to Settlement contract: {self.settlement_address}")
    
    # ========================================================================
    # Native Balance Operations
    # ========================================================================
    
    def get_balance(self, address: Optional[str] = None) -> Decimal:
        """
        Get native SIBR balance.
        
        Args:
            address: Address to check (default: own address)
        
        Returns:
            Balance in SIBR (not wei)
        """
        Web3 = _ensure_web3()
        address = address or self.address
        if not address:
            raise ValueError("No address specified")
        
        address = Web3.to_checksum_address(address)
        wei = self.w3.eth.get_balance(address)
        return Decimal(wei) / Decimal(10**18)
    
    def get_balance_wei(self, address: Optional[str] = None) -> int:
        """Get balance in wei."""
        Web3 = _ensure_web3()
        address = address or self.address
        if not address:
            raise ValueError("No address specified")
        
        address = Web3.to_checksum_address(address)
        return self.w3.eth.get_balance(address)
    
    async def transfer(
        self,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[int] = None,
    ) -> str:
        """
        Transfer native SIBR tokens.
        
        Args:
            to_address: Recipient address
            amount: Amount in SIBR (not wei)
            gas_price: Custom gas price (default: network gas price)
        
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key required for transfers")
        
        Web3 = _ensure_web3()
        to_address = Web3.to_checksum_address(to_address)
        amount_wei = int(amount * Decimal(10**18))
        
        # Build transaction
        tx = {
            "from": self.address,
            "to": to_address,
            "value": amount_wei,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": 21000,  # Standard transfer gas
            "gasPrice": gas_price or self.w3.eth.gas_price,
            "chainId": self.network.chain_id,
        }
        
        # Sign and send
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"[SIBERIUM] Transfer {amount} SIBR to {to_address}: {tx_hash.hex()}")
        return tx_hash.hex()
    
    # ========================================================================
    # Settlement Contract Operations
    # ========================================================================
    
    async def deposit_stake(self, amount: Decimal) -> str:
        """
        Deposit SIBR as stake in Settlement contract.
        
        Args:
            amount: Amount in SIBR
        
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key required")
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        amount_wei = int(amount * Decimal(10**18))
        
        # Build deposit transaction (payable)
        tx = self.settlement.functions.deposit().build_transaction({
            "from": self.address,
            "value": amount_wei,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": 100000,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.network.chain_id,
        })
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"[SIBERIUM] Deposited {amount} SIBR as stake: {tx_hash.hex()}")
        return tx_hash.hex()
    
    async def get_staked_balance(self, address: Optional[str] = None) -> Decimal:
        """Get staked balance in Settlement contract."""
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        Web3 = _ensure_web3()
        address = address or self.address
        address = Web3.to_checksum_address(address)
        
        balance_wei = self.settlement.functions.stakedBalance(address).call()
        return Decimal(balance_wei) / Decimal(10**18)
    
    async def get_account_info(self, address: Optional[str] = None) -> Dict[str, Any]:
        """
        Get full account info from Settlement contract.
        
        Returns:
            Dict with staked, pending_unstake, unlock_time
        """
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        Web3 = _ensure_web3()
        address = address or self.address
        address = Web3.to_checksum_address(address)
        
        staked, pending, unlock_time = self.settlement.functions.getAccountInfo(address).call()
        
        return {
            "staked": Decimal(staked) / Decimal(10**18),
            "staked_wei": staked,
            "pending_unstake": Decimal(pending) / Decimal(10**18),
            "pending_unstake_wei": pending,
            "unlock_time": unlock_time,
            "can_withdraw": unlock_time > 0 and time.time() >= unlock_time,
        }
    
    async def request_unstake(self, amount: Decimal) -> str:
        """Request unstake (starts 7-day timelock)."""
        if not self.account:
            raise ValueError("Private key required")
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        amount_wei = int(amount * Decimal(10**18))
        
        tx = self.settlement.functions.requestUnstake(amount_wei).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": 100000,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.network.chain_id,
        })
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"[SIBERIUM] Requested unstake {amount} SIBR: {tx_hash.hex()}")
        return tx_hash.hex()
    
    async def withdraw(self) -> str:
        """Complete unstake after timelock."""
        if not self.account:
            raise ValueError("Private key required")
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        tx = self.settlement.functions.withdraw().build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": 100000,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.network.chain_id,
        })
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"[SIBERIUM] Withdrawn stake: {tx_hash.hex()}")
        return tx_hash.hex()
    
    async def withdraw_balance(self, amount: Decimal) -> str:
        """Withdraw available balance (from settlements)."""
        if not self.account:
            raise ValueError("Private key required")
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        amount_wei = int(amount * Decimal(10**18))
        
        tx = self.settlement.functions.withdrawBalance(amount_wei).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": 100000,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.network.chain_id,
        })
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"[SIBERIUM] Withdrawn balance {amount} SIBR: {tx_hash.hex()}")
        return tx_hash.hex()
    
    # ========================================================================
    # Settlement Signatures
    # ========================================================================
    
    async def get_next_nonce(self, address: Optional[str] = None) -> int:
        """Get next available nonce for settlements."""
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        Web3 = _ensure_web3()
        address = address or self.address
        address = Web3.to_checksum_address(address)
        
        return self.settlement.functions.getNextNonce(address).call()
    
    def sign_settlement(
        self,
        payee: str,
        amount: Decimal,
        nonce: int,
        deadline: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        """
        Sign a settlement IOU (as payer).
        
        Args:
            payee: Address to pay
            amount: Amount in SIBR
            nonce: Unique nonce
            deadline: Signature expiration (default: 1 hour)
        
        Returns:
            (signature, deadline)
        """
        if not self.account:
            raise ValueError("Private key required")
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        Web3 = _ensure_web3()
        payee = Web3.to_checksum_address(payee)
        amount_wei = int(amount * Decimal(10**18))
        
        if deadline is None:
            deadline = int(time.time()) + 3600  # 1 hour
        
        # Get digest from contract
        digest = self.settlement.functions.getSettlementDigest(
            self.address,
            payee,
            amount_wei,
            nonce,
            deadline,
        ).call()
        
        # Sign the digest
        signed = self.account.sign_message(encode_defunct(digest))
        
        return signed.signature, deadline
    
    async def claim_settlement(
        self,
        payer: str,
        amount: Decimal,
        nonce: int,
        deadline: int,
        signature: bytes,
    ) -> str:
        """
        Claim a settlement payment (as payee).
        
        Args:
            payer: Address of the debtor
            amount: Amount in SIBR
            nonce: Nonce from signature
            deadline: Signature deadline
            signature: EIP-712 signature from payer
        
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key required")
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        Web3 = _ensure_web3()
        payer = Web3.to_checksum_address(payer)
        amount_wei = int(amount * Decimal(10**18))
        
        tx = self.settlement.functions.claim(
            payer,
            amount_wei,
            nonce,
            deadline,
            signature,
        ).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": 150000,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.network.chain_id,
        })
        
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        logger.info(f"[SIBERIUM] Claimed settlement from {payer}: {tx_hash.hex()}")
        return tx_hash.hex()
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def wait_for_receipt(
        self,
        tx_hash: str,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """
        Wait for transaction receipt.
        
        Args:
            tx_hash: Transaction hash
            timeout: Timeout in seconds
        
        Returns:
            Transaction receipt
        """
        if isinstance(tx_hash, str):
            tx_hash = bytes.fromhex(tx_hash.replace("0x", ""))
        
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        return dict(receipt)
    
    async def get_settlement_stats(self) -> Dict[str, Any]:
        """Get Settlement contract statistics."""
        if not self.settlement:
            raise ValueError("Settlement contract not connected")
        
        total_staked, accumulated_fees, fee_rate, unstake_delay, contract_balance = \
            self.settlement.functions.getStats().call()
        
        return {
            "total_staked": Decimal(total_staked) / Decimal(10**18),
            "accumulated_fees": Decimal(accumulated_fees) / Decimal(10**18),
            "fee_rate_bps": fee_rate,
            "fee_rate_percent": fee_rate / 100,
            "unstake_delay_seconds": unstake_delay,
            "unstake_delay_days": unstake_delay / 86400,
            "contract_balance": Decimal(contract_balance) / Decimal(10**18),
        }
    
    def get_explorer_url(self, tx_hash: str) -> str:
        """Get block explorer URL for transaction."""
        if not self.network.explorer_url:
            return ""
        return f"{self.network.explorer_url}/tx/{tx_hash}"
    
    @staticmethod
    def to_wei(amount: Decimal) -> int:
        """Convert SIBR to wei."""
        return int(amount * Decimal(10**18))
    
    @staticmethod
    def from_wei(amount_wei: int) -> Decimal:
        """Convert wei to SIBR."""
        return Decimal(amount_wei) / Decimal(10**18)


# ============================================================================
# Legacy Alias (backwards compatibility)
# ============================================================================

ChainManager = SiberiumManager


# ============================================================================
# Convenience Functions
# ============================================================================

def create_siberium_manager(
    testnet: bool = True,
    private_key: Optional[str] = None,
    settlement_address: Optional[str] = None,
) -> SiberiumManager:
    """
    Create a SiberiumManager with default configuration.
    
    Args:
        testnet: Use testnet (True) or mainnet (False)
        private_key: Wallet private key
        settlement_address: Settlement contract address
    
    Returns:
        Configured SiberiumManager
    """
    network = SIBERIUM_TESTNET if testnet else SIBERIUM_MAINNET
    
    return SiberiumManager(
        rpc_url=network.rpc_url,
        private_key=private_key,
        settlement_address=settlement_address,
        network=network,
    )
