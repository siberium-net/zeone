#!/usr/bin/env python3
"""
Siberium Settlement Contract Deployment Script
==============================================

[DEVOPS] Professional deployment script for ZEONE economy on Siberium.

Usage:
    python scripts/deploy_siberium.py [--force]

Environment Variables (via .env):
    PRIVATE_KEY         - Administrator wallet private key
    SIBERIUM_RPC_URL    - RPC endpoint (testnet/mainnet)

Output:
    data/siberium_deployment.json - Contract address and ABI

[SECURITY] Never commit .env or deployment keys to git!
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

CONTRACTS_DIR = PROJECT_ROOT / "contracts"
DATA_DIR = PROJECT_ROOT / "data"
DEPLOYMENT_FILE = DATA_DIR / "siberium_deployment.json"

# Solidity configuration
# Note: 0.8.20+ uses PUSH0 opcode not supported on all EVMs
SOLC_VERSION = "0.8.19"
CONTRACT_FILE = "ZEOSettlement.sol"

# Safety thresholds
MIN_BALANCE_SIBR = Decimal("10")  # Minimum balance for deployment
GAS_PRICE_MULTIPLIER = 1.2  # 20% buffer for gas price
DEPLOY_GAS_LIMIT = 3_000_000

# Network names
NETWORK_NAMES = {
    111111: "Siberium Mainnet",
    111000: "Siberium Testnet",
}


# ============================================================================
# Console Output Helpers
# ============================================================================

class Console:
    """Formatted console output."""
    
    @staticmethod
    def header(text: str):
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def info(text: str):
        print(f"[INFO] {text}")
    
    @staticmethod
    def success(text: str):
        print(f"[OK] {text}")
    
    @staticmethod
    def warning(text: str):
        print(f"[WARN] {text}")
    
    @staticmethod
    def error(text: str):
        print(f"[ERROR] {text}", file=sys.stderr)
    
    @staticmethod
    def step(num: int, total: int, text: str):
        print(f"\n[{num}/{total}] {text}")
        print("-" * 40)
    
    @staticmethod
    def kv(key: str, value: str, indent: int = 2):
        print(f"{' '*indent}{key}: {value}")
    
    @staticmethod
    def confirm(prompt: str) -> bool:
        while True:
            response = input(f"{prompt} [y/N]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            if response in ("n", "no", ""):
                return False


# ============================================================================
# Main Deployment Logic
# ============================================================================

def load_environment():
    """Load environment variables from .env file."""
    from dotenv import load_dotenv
    
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        Console.info(f"Loaded environment from {env_file}")
    else:
        Console.warning(f".env file not found at {env_file}")
    
    # Validate required variables
    private_key = os.getenv("PRIVATE_KEY", "")
    rpc_url = os.getenv("SIBERIUM_RPC_URL", "")
    
    if not private_key:
        Console.error("PRIVATE_KEY not set in environment")
        sys.exit(1)
    
    if not rpc_url:
        Console.error("SIBERIUM_RPC_URL not set in environment")
        sys.exit(1)
    
    # Normalize private key (add 0x if missing)
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    
    return private_key, rpc_url


def connect_to_chain(rpc_url: str):
    """Connect to Siberium blockchain."""
    from web3 import Web3
    
    Console.info(f"Connecting to {rpc_url}...")
    
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    # Add PoA middleware (required for Siberium)
    # web3 v7 uses ExtraDataToPOAMiddleware
    try:
        from web3.middleware import ExtraDataToPOAMiddleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except ImportError:
        # Fallback for older web3 versions
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    if not w3.is_connected():
        Console.error("Failed to connect to RPC endpoint")
        sys.exit(1)
    
    # Auto-detect chain ID
    chain_id = w3.eth.chain_id
    network_name = NETWORK_NAMES.get(chain_id, f"Unknown ({chain_id})")
    block_number = w3.eth.block_number
    
    Console.success("Connected to blockchain")
    Console.kv("Network", network_name)
    Console.kv("Chain ID", str(chain_id))
    Console.kv("Current Block", str(block_number))
    Console.kv("Gas Price", f"{w3.from_wei(w3.eth.gas_price, 'gwei')} Gwei")
    
    return w3, chain_id


def preflight_check(w3, private_key: str):
    """Validate deployer account."""
    from eth_account import Account
    
    account = Account.from_key(private_key)
    address = account.address
    
    Console.info(f"Deployer address: {address}")
    
    # Get balance
    balance_wei = w3.eth.get_balance(address)
    balance_sibr = Decimal(balance_wei) / Decimal(10**18)
    
    Console.kv("Balance", f"{balance_sibr:.4f} SIBR")
    
    # Check minimum balance
    if balance_sibr < MIN_BALANCE_SIBR:
        Console.error(f"Insufficient balance. Minimum required: {MIN_BALANCE_SIBR} SIBR")
        Console.info("Get test SIBR from faucet: https://faucet.test.siberium.net")
        sys.exit(1)
    
    Console.success("Pre-flight check passed")
    
    return account, address, balance_sibr


def check_existing_deployment(force: bool = False) -> bool:
    """Check if contract is already deployed."""
    if not DEPLOYMENT_FILE.exists():
        return True  # No existing deployment, proceed
    
    try:
        with open(DEPLOYMENT_FILE) as f:
            deployment = json.load(f)
        
        Console.warning("Existing deployment found:")
        Console.kv("Address", deployment.get("address", "N/A"))
        Console.kv("Deployed At", deployment.get("deployed_at", "N/A"))
        Console.kv("Chain ID", str(deployment.get("chain_id", "N/A")))
        
        if force:
            Console.info("--force flag set, proceeding with redeployment")
            return True
        
        return Console.confirm("Redeploy contract?")
        
    except Exception as e:
        Console.warning(f"Could not read existing deployment: {e}")
        return True


def compile_contract():
    """Compile Settlement contract using solcx."""
    import solcx
    
    Console.info(f"Compiling {CONTRACT_FILE}...")
    
    # Install solc if needed
    installed_versions = solcx.get_installed_solc_versions()
    if SOLC_VERSION not in [str(v) for v in installed_versions]:
        Console.info(f"Installing solc {SOLC_VERSION}...")
        solcx.install_solc(SOLC_VERSION)
    
    solcx.set_solc_version(SOLC_VERSION)
    
    # Read contract source
    contract_path = CONTRACTS_DIR / CONTRACT_FILE
    if not contract_path.exists():
        Console.error(f"Contract file not found: {contract_path}")
        sys.exit(1)
    
    source = contract_path.read_text()
    
    # Compile
    compiled = solcx.compile_standard(
        {
            "language": "Solidity",
            "sources": {
                CONTRACT_FILE: {"content": source}
            },
            "settings": {
                "optimizer": {"enabled": True, "runs": 200},
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode.object"]
                    }
                },
            },
        },
        allow_paths=[str(CONTRACTS_DIR)],
    )
    
    # Extract contract data
    contract_name = CONTRACT_FILE.replace(".sol", "")
    contract_data = compiled["contracts"][CONTRACT_FILE][contract_name]
    
    abi = contract_data["abi"]
    bytecode = "0x" + contract_data["evm"]["bytecode"]["object"]
    
    Console.success(f"Compiled successfully")
    Console.kv("ABI Functions", str(len([x for x in abi if x.get("type") == "function"])))
    Console.kv("Bytecode Size", f"{len(bytecode) // 2} bytes")
    
    return abi, bytecode


def deploy_contract(w3, account, abi: list, bytecode: str, chain_id: int):
    """Deploy contract to blockchain."""
    from eth_account import Account
    
    Console.info("Preparing deployment transaction...")
    
    # Create contract instance
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Get gas price with buffer
    gas_price = int(w3.eth.gas_price * GAS_PRICE_MULTIPLIER)
    
    # Build deployment transaction
    nonce = w3.eth.get_transaction_count(account.address)
    
    tx = contract.constructor().build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": DEPLOY_GAS_LIMIT,
        "gasPrice": gas_price,
        "chainId": chain_id,
    })
    
    # Estimate actual gas
    try:
        estimated_gas = w3.eth.estimate_gas(tx)
        Console.kv("Estimated Gas", str(estimated_gas))
        tx["gas"] = int(estimated_gas * 1.1)  # 10% buffer
    except Exception as e:
        Console.warning(f"Gas estimation failed: {e}, using default")
    
    Console.kv("Gas Limit", str(tx["gas"]))
    Console.kv("Gas Price", f"{w3.from_wei(gas_price, 'gwei')} Gwei")
    
    estimated_cost = Decimal(tx["gas"] * gas_price) / Decimal(10**18)
    Console.kv("Estimated Cost", f"{estimated_cost:.6f} SIBR")
    
    # Sign transaction
    Console.info("Signing transaction...")
    signed_tx = account.sign_transaction(tx)
    
    # Send transaction
    Console.info("Sending transaction...")
    # web3 v7 uses raw_transaction (snake_case)
    raw_tx = getattr(signed_tx, 'raw_transaction', None) or getattr(signed_tx, 'rawTransaction', None)
    tx_hash = w3.eth.send_raw_transaction(raw_tx)
    tx_hash_hex = tx_hash.hex()
    
    Console.info(f"Transaction sent: {tx_hash_hex}")
    Console.info("Waiting for confirmation...")
    
    # Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    
    if receipt.status != 1:
        Console.error("Transaction failed!")
        Console.error(f"Receipt: {dict(receipt)}")
        sys.exit(1)
    
    contract_address = receipt.contractAddress
    gas_used = receipt.gasUsed
    actual_cost = Decimal(gas_used * gas_price) / Decimal(10**18)
    
    Console.success("Contract deployed successfully!")
    Console.kv("Contract Address", contract_address)
    Console.kv("Block Number", str(receipt.blockNumber))
    Console.kv("Gas Used", str(gas_used))
    Console.kv("Actual Cost", f"{actual_cost:.6f} SIBR")
    Console.kv("Transaction Hash", tx_hash_hex)
    
    return contract_address, tx_hash_hex, receipt


def save_deployment(
    address: str,
    abi: list,
    chain_id: int,
    tx_hash: str,
    deployer: str,
):
    """Save deployment info to JSON file."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    deployment = {
        "address": address,
        "abi": abi,
        "chain_id": chain_id,
        "network": NETWORK_NAMES.get(chain_id, "Unknown"),
        "tx_hash": tx_hash,
        "deployer": deployer,
        "deployed_at": datetime.utcnow().isoformat() + "Z",
        "solc_version": SOLC_VERSION,
        "contract_name": CONTRACT_FILE.replace(".sol", ""),
    }
    
    with open(DEPLOYMENT_FILE, "w") as f:
        json.dump(deployment, f, indent=2)
    
    Console.success(f"Deployment saved to {DEPLOYMENT_FILE}")


def print_explorer_link(address: str, chain_id: int):
    """Print block explorer link."""
    if chain_id == 111000:
        explorer = f"https://explorer.test.siberium.net/address/{address}"
    elif chain_id == 111111:
        explorer = f"https://explorer.siberium.net/address/{address}"
    else:
        return
    
    Console.info(f"Explorer: {explorer}")


def main():
    """Main deployment procedure."""
    parser = argparse.ArgumentParser(description="Deploy ZEONE Settlement to Siberium")
    parser.add_argument("--force", "-f", action="store_true", help="Force redeployment")
    args = parser.parse_args()
    
    Console.header("ZEONE Settlement Contract Deployment")
    
    total_steps = 6
    
    # Step 1: Load environment
    Console.step(1, total_steps, "Loading Environment")
    private_key, rpc_url = load_environment()
    
    # Step 2: Connect to chain
    Console.step(2, total_steps, "Connecting to Blockchain")
    w3, chain_id = connect_to_chain(rpc_url)
    
    # Step 3: Pre-flight check
    Console.step(3, total_steps, "Pre-flight Check")
    account, address, balance = preflight_check(w3, private_key)
    
    # Step 4: Check existing deployment
    Console.step(4, total_steps, "Checking Existing Deployment")
    if not check_existing_deployment(args.force):
        Console.info("Deployment cancelled by user")
        sys.exit(0)
    
    # Step 5: Compile contract
    Console.step(5, total_steps, "Compiling Contract")
    abi, bytecode = compile_contract()
    
    # Step 6: Deploy
    Console.step(6, total_steps, "Deploying Contract")
    contract_address, tx_hash, receipt = deploy_contract(
        w3, account, abi, bytecode, chain_id
    )
    
    # Save deployment info
    save_deployment(contract_address, abi, chain_id, tx_hash, address)
    
    # Final summary
    Console.header("Deployment Complete")
    Console.kv("Contract", contract_address)
    Console.kv("Network", NETWORK_NAMES.get(chain_id, str(chain_id)))
    Console.kv("Deployer", address)
    Console.kv("Remaining Balance", f"{balance - Decimal(receipt.gasUsed * w3.eth.gas_price) / Decimal(10**18):.4f} SIBR")
    print_explorer_link(contract_address, chain_id)
    
    print("\n" + "="*60)
    print("  Next steps:")
    print("    1. Verify contract on explorer (optional)")
    print("    2. Update .env with SETTLEMENT_ADDRESS")
    print("    3. Run: python -m webui.main")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORT] Deployment cancelled by user")
        sys.exit(130)
    except Exception as e:
        Console.error(f"Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

