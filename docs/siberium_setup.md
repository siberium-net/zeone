# Siberium Network Setup Guide

## Overview

ZEONE operates on **Siberium** — a custom EVM-compatible PoA (Proof of Authority) blockchain.

| Network | ChainID | RPC URL | Explorer |
|---------|---------|---------|----------|
| Mainnet | 111111 | https://rpc.siberium.net | https://explorer.siberium.net |
| Testnet | 111000 | https://rpc.test.siberium.net | https://explorer.test.siberium.net |

**Key Feature:** SIBR is the native gas token (like ETH on Ethereum), not an ERC-20 contract.

---

## Quick Start

### 1. Configure Environment

Create `.env` file in project root:

```bash
# Siberium RPC
SIBERIUM_RPC_URL=https://rpc.test.siberium.net

# Your wallet private key (without 0x prefix)
WALLET_PRIVATE_KEY=your_private_key_here

# Settlement contract address (deployed)
SETTLEMENT_ADDRESS=0x...
```

### 2. Python Usage

```python
from economy.chain import SiberiumManager, create_siberium_manager

# Quick setup (testnet)
manager = create_siberium_manager(
    testnet=True,
    private_key="0x...",
    settlement_address="0x...",
)

# Check balance
balance = manager.get_balance()
print(f"Balance: {balance} SIBR")

# Deposit stake
tx_hash = await manager.deposit_stake(amount=Decimal("100"))

# Sign settlement IOU (as payer)
signature, deadline = manager.sign_settlement(
    payee="0x...",
    amount=Decimal("10"),
    nonce=await manager.get_next_nonce(),
)

# Claim settlement (as payee)
tx_hash = await manager.claim_settlement(
    payer="0x...",
    amount=Decimal("10"),
    nonce=nonce,
    deadline=deadline,
    signature=signature,
)
```

---

## Network Configuration

### MetaMask Setup

Add Siberium network to MetaMask:

**Testnet:**
- Network Name: `Siberium Testnet`
- RPC URL: `https://rpc.test.siberium.net`
- Chain ID: `111000`
- Currency Symbol: `SIBR`
- Block Explorer: `https://explorer.test.siberium.net`

**Mainnet:**
- Network Name: `Siberium Mainnet`
- RPC URL: `https://rpc.siberium.net`
- Chain ID: `111111`
- Currency Symbol: `SIBR`
- Block Explorer: `https://explorer.siberium.net`

### config.py Integration

```python
# config.py

import os
from economy.chain import SIBERIUM_TESTNET, SIBERIUM_MAINNET

# Select network based on environment
SIBERIUM_NETWORK = os.getenv("SIBERIUM_NETWORK", "testnet")

if SIBERIUM_NETWORK == "mainnet":
    CHAIN_CONFIG = SIBERIUM_MAINNET
else:
    CHAIN_CONFIG = SIBERIUM_TESTNET

# RPC URL (can override)
RPC_URL = os.getenv("SIBERIUM_RPC_URL", CHAIN_CONFIG.rpc_url)

# Contract addresses
SETTLEMENT_ADDRESS = os.getenv("SETTLEMENT_ADDRESS", "")
```

---

## Local Development

### Option 1: Geth (Recommended)

Run a local Geth node with Siberium-compatible configuration:

```bash
# Create genesis.json
cat > genesis.json << 'EOF'
{
  "config": {
    "chainId": 111000,
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "clique": {
      "period": 3,
      "epoch": 30000
    }
  },
  "difficulty": "1",
  "gasLimit": "30000000",
  "extradata": "0x0000000000000000000000000000000000000000000000000000000000000000YOUR_SIGNER_ADDRESS0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
  "alloc": {
    "YOUR_ADDRESS": { "balance": "1000000000000000000000000" }
  }
}
EOF

# Initialize geth
geth init genesis.json --datadir ./siberium-data

# Run node
geth --datadir ./siberium-data \
     --networkid 111000 \
     --http \
     --http.addr "0.0.0.0" \
     --http.port 8545 \
     --http.api "eth,net,web3,personal,miner" \
     --http.corsdomain "*" \
     --allow-insecure-unlock \
     --mine \
     --miner.etherbase YOUR_SIGNER_ADDRESS
```

### Option 2: Hardhat (Quick Testing)

```javascript
// hardhat.config.js
module.exports = {
  networks: {
    siberium_local: {
      url: "http://127.0.0.1:8545",
      chainId: 111000,
      accounts: [process.env.PRIVATE_KEY],
    },
    siberium_testnet: {
      url: "https://rpc.test.siberium.net",
      chainId: 111000,
      accounts: [process.env.PRIVATE_KEY],
    },
    siberium_mainnet: {
      url: "https://rpc.siberium.net",
      chainId: 111111,
      accounts: [process.env.PRIVATE_KEY],
    },
  },
  solidity: "0.8.20",
};
```

### Option 3: Anvil (Foundry)

```bash
# Run local chain with Siberium ChainID
anvil --chain-id 111000 --host 0.0.0.0 --port 8545
```

---

## Faucet (Getting Test SIBR)

### Testnet Faucet

1. **Web Faucet:** https://faucet.test.siberium.net
   - Enter your address
   - Receive 10 SIBR (once per 24 hours)

2. **Discord Bot:**
   - Join Siberium Discord
   - Use `!faucet <your_address>` in #faucet channel

3. **API Faucet:**
   ```bash
   curl -X POST https://faucet.test.siberium.net/api/claim \
        -H "Content-Type: application/json" \
        -d '{"address": "0xYOUR_ADDRESS"}'
   ```

### Local Development

For local Geth/Anvil nodes, fund accounts in genesis.json or use:

```bash
# Anvil (auto-funded accounts)
anvil --chain-id 111000

# Geth (use pre-funded account from genesis)
geth attach --exec "eth.sendTransaction({from: eth.accounts[0], to: 'YOUR_ADDRESS', value: web3.toWei(1000, 'ether')})"
```

---

## Deploy Settlement Contract

### Using Python

```python
from economy.contracts import ContractCompiler
from economy.chain import SiberiumManager

# Compile contract
compiler = ContractCompiler()
settlement = compiler.compile("ZEOSettlement.sol")

# Initialize manager
manager = SiberiumManager(
    rpc_url="https://rpc.test.siberium.net",
    private_key="0x...",
)

# Deploy (requires sufficient SIBR for gas)
tx = manager.w3.eth.contract(
    abi=settlement["abi"],
    bytecode=settlement["bytecode"],
).constructor().build_transaction({
    "from": manager.address,
    "nonce": manager.w3.eth.get_transaction_count(manager.address),
    "gas": 3000000,
    "gasPrice": manager.w3.eth.gas_price,
    "chainId": 111000,
})

signed = manager.account.sign_transaction(tx)
tx_hash = manager.w3.eth.send_raw_transaction(signed.rawTransaction)
receipt = manager.w3.eth.wait_for_transaction_receipt(tx_hash)

print(f"Settlement deployed at: {receipt.contractAddress}")
```

### Using Hardhat

```bash
# Deploy
npx hardhat run scripts/deploy.js --network siberium_testnet

# Verify (if explorer supports it)
npx hardhat verify --network siberium_testnet CONTRACT_ADDRESS
```

---

## Troubleshooting

### Connection Issues

```python
# Test RPC connection
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("https://rpc.test.siberium.net"))
print(f"Connected: {w3.is_connected()}")
print(f"ChainID: {w3.eth.chain_id}")
print(f"Block: {w3.eth.block_number}")
```

### PoA Middleware

Siberium uses PoA consensus. Always add the middleware:

```python
from web3.middleware import geth_poa_middleware
w3.middleware_onion.inject(geth_poa_middleware, layer=0)
```

### Insufficient Gas

PoA networks have low gas costs, but ensure you have SIBR:

```python
balance = manager.get_balance()
if balance < Decimal("0.01"):
    print("Warning: Low SIBR balance for gas")
```

### Invalid ChainID

Settlement contract validates ChainID in constructor:

```solidity
require(
    block.chainid == SIBERIUM_MAINNET || block.chainid == SIBERIUM_TESTNET,
    "Settlement: not Siberium network"
);
```

If deploying locally, modify the contract or use matching ChainID.

---

## Security Considerations

1. **Private Key Storage:**
   - Never commit private keys to git
   - Use environment variables or secure vaults
   - Consider hardware wallets for mainnet

2. **RPC Security:**
   - Use HTTPS for remote RPCs
   - For local nodes, bind to localhost only

3. **Contract Verification:**
   - Verify source code on explorer after deployment
   - Check Settlement contract address before staking

---

## Reference

### SiberiumManager Methods

| Method | Description |
|--------|-------------|
| `get_balance()` | Get native SIBR balance |
| `transfer(to, amount)` | Transfer native SIBR |
| `deposit_stake(amount)` | Deposit SIBR as stake |
| `request_unstake(amount)` | Start unstake (7-day lock) |
| `withdraw()` | Complete unstake |
| `sign_settlement(...)` | Sign IOU as payer |
| `claim_settlement(...)` | Claim payment as payee |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SIBERIUM_RPC_URL` | RPC endpoint | testnet |
| `WALLET_PRIVATE_KEY` | Signing key | - |
| `SETTLEMENT_ADDRESS` | Contract address | - |
| `SIBERIUM_NETWORK` | mainnet/testnet | testnet |

