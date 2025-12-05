"""
Smart Contract Manager
======================

[BLOCKCHAIN] Python interface for ZEONE smart contracts.

Components:
- ContractCompiler: Compiles Solidity contracts using py-solc-x
- ContractManager: Deploys and interacts with contracts
- SignatureHelper: Creates EIP-712 signatures for settlements

[USAGE]
    manager = ContractManager(web3, private_key)
    await manager.deploy_contracts()
    
    # Stake tokens
    await manager.stake(amount)
    
    # Create settlement signature
    sig = manager.sign_settlement(payee, amount, nonce)
    
    # Claim settlement (as payee)
    await manager.claim(payer, amount, nonce, deadline, signature)
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from eth_account import Account
from eth_account.messages import encode_defunct, encode_structured_data

logger = logging.getLogger(__name__)

# Lazy imports
Web3 = None


def _ensure_web3():
    global Web3
    if Web3 is None:
        from web3 import Web3 as _Web3
        Web3 = _Web3
    return Web3


# ============================================================================
# Constants
# ============================================================================

# Default contract paths
CONTRACT_DIR = Path(__file__).parent.parent / "contracts"
TOKEN_CONTRACT = "ZEOToken.sol"
SETTLEMENT_CONTRACT = "ZEOSettlement.sol"

# Solidity version
SOLC_VERSION = "0.8.20"

# Gas limits
GAS_LIMIT_DEPLOY = 5_000_000
GAS_LIMIT_TX = 300_000


# ============================================================================
# Contract Compiler
# ============================================================================

class ContractCompiler:
    """
    Compiles Solidity contracts using py-solc-x.
    
    [DEPENDENCY] Requires: pip install py-solc-x
    """
    
    def __init__(self, contract_dir: Path = CONTRACT_DIR):
        self.contract_dir = contract_dir
        self._compiled_cache: Dict[str, Dict] = {}
    
    def compile(self, contract_name: str, optimize: bool = True) -> Dict[str, Any]:
        """
        Compile a Solidity contract.
        
        Args:
            contract_name: Contract filename (e.g., "ZEOToken.sol")
            optimize: Enable optimizer
        
        Returns:
            Dict with 'abi' and 'bytecode'
        """
        if contract_name in self._compiled_cache:
            return self._compiled_cache[contract_name]
        
        try:
            import solcx
            
            # Install solc if needed
            if SOLC_VERSION not in solcx.get_installed_solc_versions():
                logger.info(f"Installing solc {SOLC_VERSION}...")
                solcx.install_solc(SOLC_VERSION)
            
            solcx.set_solc_version(SOLC_VERSION)
            
            # Read source
            source_path = self.contract_dir / contract_name
            if not source_path.exists():
                raise FileNotFoundError(f"Contract not found: {source_path}")
            
            source = source_path.read_text()
            
            # Handle imports
            sources = {contract_name: {"content": source}}
            
            # Check for local imports
            for line in source.split("\n"):
                if line.strip().startswith("import"):
                    # Extract import path
                    import_path = line.split('"')[1] if '"' in line else line.split("'")[1]
                    if import_path.startswith("./"):
                        import_file = import_path[2:]
                        import_source = (self.contract_dir / import_file).read_text()
                        sources[import_file] = {"content": import_source}
            
            # Compile
            compiled = solcx.compile_standard(
                {
                    "language": "Solidity",
                    "sources": sources,
                    "settings": {
                        "optimizer": {"enabled": optimize, "runs": 200},
                        "outputSelection": {
                            "*": {
                                "*": ["abi", "evm.bytecode.object"]
                            }
                        },
                    },
                },
                allow_paths=[str(self.contract_dir)],
            )
            
            # Extract contract name (without .sol)
            contract_key = contract_name.replace(".sol", "")
            
            contract_data = compiled["contracts"][contract_name][contract_key]
            
            result = {
                "abi": contract_data["abi"],
                "bytecode": "0x" + contract_data["evm"]["bytecode"]["object"],
            }
            
            self._compiled_cache[contract_name] = result
            logger.info(f"Compiled {contract_name}: {len(result['bytecode'])} bytes")
            
            return result
            
        except ImportError:
            logger.error("py-solc-x not installed. Run: pip install py-solc-x")
            raise
    
    def compile_all(self) -> Dict[str, Dict[str, Any]]:
        """Compile all contracts."""
        return {
            "ZEOToken": self.compile(TOKEN_CONTRACT),
            "ZEOSettlement": self.compile(SETTLEMENT_CONTRACT),
        }


# ============================================================================
# Contract ABIs (fallback if compilation unavailable)
# ============================================================================

# Minimal ABIs for interaction without compilation
MINIMAL_TOKEN_ABI = [
    {"inputs": [], "name": "name", "outputs": [{"type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "symbol", "outputs": [{"type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "decimals", "outputs": [{"type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "transfer", "outputs": [{"type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "approve", "outputs": [{"type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "from", "type": "address"}, {"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "transferFrom", "outputs": [{"type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "amount", "type": "uint256"}], "name": "burn", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "mint", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "newMinter", "type": "address"}], "name": "setMinter", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "minter", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "MAX_SUPPLY", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalEmitted", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getPendingEmission", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "blockNum", "type": "uint256"}], "name": "getBlockReward", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getEmissionStats", "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}], "stateMutability": "view", "type": "function"},
]

MINIMAL_SETTLEMENT_ABI = [
    {"inputs": [{"name": "amount", "type": "uint256"}], "name": "stake", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "amount", "type": "uint256"}], "name": "requestUnstake", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "unstake", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "cancelUnstake", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "stakedBalance", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalStaked", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "payer", "type": "address"}, {"name": "amount", "type": "uint256"}, {"name": "nonce", "type": "uint256"}, {"name": "deadline", "type": "uint256"}, {"name": "signature", "type": "bytes"}], "name": "claim", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "getNextNonce", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "useNonce", "outputs": [{"type": "uint256"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "payer", "type": "address"}, {"name": "payee", "type": "address"}, {"name": "amount", "type": "uint256"}, {"name": "nonce", "type": "uint256"}, {"name": "deadline", "type": "uint256"}], "name": "getSettlementDigest", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "getAccountInfo", "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getStats", "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "token", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "DOMAIN_SEPARATOR", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "feeRate", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "unstakeDelay", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
]


# ============================================================================
# Contract Manager
# ============================================================================

@dataclass
class ContractAddresses:
    """Deployed contract addresses."""
    token: str
    settlement: str
    chain_id: int
    deployed_at: int  # block number


class ContractManager:
    """
    Manages ZEONE smart contracts.
    
    [USAGE]
        # Initialize
        manager = ContractManager(
            rpc_url="http://localhost:8545",
            private_key="0x...",
        )
        
        # Deploy (first time)
        addresses = await manager.deploy_contracts()
        
        # Or connect to existing
        manager.connect(token_address, settlement_address)
        
        # Interact
        balance = await manager.get_balance(address)
        await manager.stake(amount)
    """
    
    def __init__(
        self,
        rpc_url: str,
        private_key: Optional[str] = None,
        token_address: Optional[str] = None,
        settlement_address: Optional[str] = None,
    ):
        """
        Args:
            rpc_url: Ethereum RPC URL
            private_key: Private key for signing (hex string)
            token_address: Existing token contract address
            settlement_address: Existing settlement contract address
        """
        Web3 = _ensure_web3()
        
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.private_key = private_key
        
        if private_key:
            self.account = Account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        self.token_contract = None
        self.settlement_contract = None
        self.addresses: Optional[ContractAddresses] = None
        
        # Connect to existing contracts if provided
        if token_address and settlement_address:
            self.connect(token_address, settlement_address)
    
    def connect(self, token_address: str, settlement_address: str) -> None:
        """
        Connect to existing deployed contracts.
        
        Args:
            token_address: ZEOToken contract address
            settlement_address: ZEOSettlement contract address
        """
        Web3 = _ensure_web3()
        
        self.token_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=MINIMAL_TOKEN_ABI,
        )
        
        self.settlement_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(settlement_address),
            abi=MINIMAL_SETTLEMENT_ABI,
        )
        
        self.addresses = ContractAddresses(
            token=token_address,
            settlement=settlement_address,
            chain_id=self.w3.eth.chain_id,
            deployed_at=0,
        )
        
        logger.info(f"Connected to contracts: token={token_address}, settlement={settlement_address}")
    
    async def deploy_contracts(self) -> ContractAddresses:
        """
        Deploy ZEOToken and ZEOSettlement contracts.
        
        Returns:
            ContractAddresses with deployed addresses
        
        [FLOW]
        1. Compile contracts
        2. Deploy ZEOToken
        3. Deploy ZEOSettlement (with token address)
        4. Set Settlement as minter for Token
        """
        if not self.account:
            raise ValueError("Private key required for deployment")
        
        compiler = ContractCompiler()
        
        # Compile contracts
        token_compiled = compiler.compile(TOKEN_CONTRACT)
        settlement_compiled = compiler.compile(SETTLEMENT_CONTRACT)
        
        # Deploy Token
        logger.info("Deploying ZEOToken...")
        token_address = await self._deploy_contract(
            token_compiled["abi"],
            token_compiled["bytecode"],
        )
        
        # Deploy Settlement
        logger.info("Deploying ZEOSettlement...")
        settlement_address = await self._deploy_contract(
            settlement_compiled["abi"],
            settlement_compiled["bytecode"],
            token_address,  # Constructor argument
        )
        
        # Connect to deployed contracts
        self.connect(token_address, settlement_address)
        
        # Set Settlement as minter
        logger.info("Setting Settlement as token minter...")
        await self._send_transaction(
            self.token_contract.functions.setMinter(settlement_address)
        )
        
        self.addresses = ContractAddresses(
            token=token_address,
            settlement=settlement_address,
            chain_id=self.w3.eth.chain_id,
            deployed_at=self.w3.eth.block_number,
        )
        
        logger.info(f"Deployment complete: {self.addresses}")
        return self.addresses
    
    async def _deploy_contract(
        self,
        abi: List[Dict],
        bytecode: str,
        *constructor_args,
    ) -> str:
        """Deploy a contract and return its address."""
        contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        
        # Build constructor transaction
        construct_txn = contract.constructor(*constructor_args).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": GAS_LIMIT_DEPLOY,
            "gasPrice": self.w3.eth.gas_price,
        })
        
        # Sign and send
        signed_txn = self.w3.eth.account.sign_transaction(construct_txn, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status != 1:
            raise RuntimeError(f"Deployment failed: {receipt}")
        
        return receipt.contractAddress
    
    async def _send_transaction(self, function) -> Dict[str, Any]:
        """Send a transaction and wait for receipt."""
        txn = function.build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": GAS_LIMIT_TX,
            "gasPrice": self.w3.eth.gas_price,
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(txn, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status != 1:
            raise RuntimeError(f"Transaction failed: {receipt}")
        
        return {"tx_hash": tx_hash.hex(), "receipt": receipt}
    
    # ========================================================================
    # Balance Functions
    # ========================================================================
    
    async def get_balance(self, address: str) -> Dict[str, int]:
        """
        Get total balance for an address.
        
        Returns:
            Dict with wallet_balance, staked_balance, pending_unstake, total
        """
        Web3 = _ensure_web3()
        address = Web3.to_checksum_address(address)
        
        # Wallet balance
        wallet_balance = self.token_contract.functions.balanceOf(address).call()
        
        # Staked balance
        staked, pending, unlock_time = self.settlement_contract.functions.getAccountInfo(address).call()
        
        return {
            "wallet_balance": wallet_balance,
            "staked_balance": staked,
            "pending_unstake": pending,
            "unlock_time": unlock_time,
            "total": wallet_balance + staked + pending,
        }
    
    async def get_staked_balance(self, address: str) -> int:
        """Get staked balance for an address."""
        Web3 = _ensure_web3()
        address = Web3.to_checksum_address(address)
        return self.settlement_contract.functions.stakedBalance(address).call()
    
    # ========================================================================
    # Staking Functions
    # ========================================================================
    
    async def stake(self, amount: int) -> Dict[str, Any]:
        """
        Stake ZEO tokens.
        
        Args:
            amount: Amount to stake (in wei)
        
        Returns:
            Transaction receipt
        """
        if not self.account:
            raise ValueError("Private key required")
        
        # Approve settlement contract
        await self._send_transaction(
            self.token_contract.functions.approve(
                self.settlement_contract.address,
                amount,
            )
        )
        
        # Stake
        return await self._send_transaction(
            self.settlement_contract.functions.stake(amount)
        )
    
    async def request_unstake(self, amount: int) -> Dict[str, Any]:
        """Request unstake (starts timelock)."""
        if not self.account:
            raise ValueError("Private key required")
        
        return await self._send_transaction(
            self.settlement_contract.functions.requestUnstake(amount)
        )
    
    async def unstake(self) -> Dict[str, Any]:
        """Complete unstake after timelock."""
        if not self.account:
            raise ValueError("Private key required")
        
        return await self._send_transaction(
            self.settlement_contract.functions.unstake()
        )
    
    # ========================================================================
    # Settlement Functions
    # ========================================================================
    
    def generate_settlement_signature(
        self,
        payee: str,
        amount: int,
        nonce: int,
        deadline: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        """
        Generate EIP-712 signature for settlement.
        
        Args:
            payee: Address to pay
            amount: Amount to pay (in wei)
            nonce: Unique nonce
            deadline: Signature expiration (default: 1 hour from now)
        
        Returns:
            (signature_bytes, deadline)
        
        [USAGE] As payer:
            sig, deadline = manager.generate_settlement_signature(payee, amount, nonce)
            # Send sig to payee off-chain
        """
        if not self.account:
            raise ValueError("Private key required")
        
        Web3 = _ensure_web3()
        payee = Web3.to_checksum_address(payee)
        
        if deadline is None:
            deadline = int(time.time()) + 3600  # 1 hour
        
        # Get digest from contract
        digest = self.settlement_contract.functions.getSettlementDigest(
            self.address,
            payee,
            amount,
            nonce,
            deadline,
        ).call()
        
        # Sign digest
        signed = self.account.sign_message(encode_defunct(digest))
        signature = signed.signature
        
        return signature, deadline
    
    async def claim(
        self,
        payer: str,
        amount: int,
        nonce: int,
        deadline: int,
        signature: bytes,
    ) -> Dict[str, Any]:
        """
        Claim settlement payment.
        
        Args:
            payer: Address of the debtor
            amount: Amount to claim
            nonce: Nonce used in signature
            deadline: Signature deadline
            signature: EIP-712 signature from payer
        
        Returns:
            Transaction receipt
        """
        if not self.account:
            raise ValueError("Private key required")
        
        Web3 = _ensure_web3()
        payer = Web3.to_checksum_address(payer)
        
        return await self._send_transaction(
            self.settlement_contract.functions.claim(
                payer,
                amount,
                nonce,
                deadline,
                signature,
            )
        )
    
    async def get_next_nonce(self, address: Optional[str] = None) -> int:
        """Get next available nonce for an address."""
        Web3 = _ensure_web3()
        address = address or self.address
        address = Web3.to_checksum_address(address)
        return self.settlement_contract.functions.getNextNonce(address).call()
    
    # ========================================================================
    # Token Functions
    # ========================================================================
    
    async def transfer(self, to: str, amount: int) -> Dict[str, Any]:
        """Transfer ZEO tokens."""
        if not self.account:
            raise ValueError("Private key required")
        
        Web3 = _ensure_web3()
        to = Web3.to_checksum_address(to)
        
        return await self._send_transaction(
            self.token_contract.functions.transfer(to, amount)
        )
    
    async def get_emission_stats(self) -> Dict[str, Any]:
        """Get token emission statistics."""
        stats = self.token_contract.functions.getEmissionStats().call()
        return {
            "current_reward": stats[0],
            "pending_emission": stats[1],
            "total_emitted": stats[2],
            "remaining_supply": stats[3],
            "current_epoch": stats[4],
            "blocks_to_halving": stats[5],
        }
    
    async def get_settlement_stats(self) -> Dict[str, Any]:
        """Get settlement contract statistics."""
        stats = self.settlement_contract.functions.getStats().call()
        return {
            "total_staked": stats[0],
            "accumulated_fees": stats[1],
            "fee_rate": stats[2],
            "unstake_delay": stats[3],
        }
    
    # ========================================================================
    # Utility Functions
    # ========================================================================
    
    @staticmethod
    def to_wei(amount: float, decimals: int = 18) -> int:
        """Convert human-readable amount to wei."""
        return int(amount * (10 ** decimals))
    
    @staticmethod
    def from_wei(amount: int, decimals: int = 18) -> float:
        """Convert wei to human-readable amount."""
        return amount / (10 ** decimals)
    
    def save_addresses(self, path: str) -> None:
        """Save contract addresses to JSON file."""
        if not self.addresses:
            raise ValueError("No addresses to save")
        
        data = {
            "token": self.addresses.token,
            "settlement": self.addresses.settlement,
            "chain_id": self.addresses.chain_id,
            "deployed_at": self.addresses.deployed_at,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_addresses(cls, path: str) -> ContractAddresses:
        """Load contract addresses from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        return ContractAddresses(
            token=data["token"],
            settlement=data["settlement"],
            chain_id=data["chain_id"],
            deployed_at=data["deployed_at"],
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_manager(
    rpc_url: str,
    private_key: str,
    addresses_file: Optional[str] = None,
) -> ContractManager:
    """
    Create a ContractManager with optional address loading.
    
    Args:
        rpc_url: Ethereum RPC URL
        private_key: Private key for signing
        addresses_file: Path to addresses JSON (optional)
    
    Returns:
        Configured ContractManager
    """
    manager = ContractManager(rpc_url, private_key)
    
    if addresses_file and Path(addresses_file).exists():
        addresses = ContractManager.load_addresses(addresses_file)
        manager.connect(addresses.token, addresses.settlement)
    
    return manager


def verify_settlement_signature(
    settlement_contract,
    payer: str,
    payee: str,
    amount: int,
    nonce: int,
    deadline: int,
    signature: bytes,
) -> bool:
    """
    Verify a settlement signature without submitting transaction.
    
    Returns:
        True if signature is valid
    """
    try:
        # Get expected digest
        digest = settlement_contract.functions.getSettlementDigest(
            payer, payee, amount, nonce, deadline
        ).call()
        
        # Recover signer
        recovered = Account.recover_message(
            encode_defunct(digest),
            signature=signature,
        )
        
        Web3 = _ensure_web3()
        return recovered.lower() == Web3.to_checksum_address(payer).lower()
        
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False

