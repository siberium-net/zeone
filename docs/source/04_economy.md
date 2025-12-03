# Hybrid Economy: Probabilistic Settlement

## Problem
Public blockchains have high gas cost and limited TPS; micro-transactions for inference/storage/relay are uneconomical if settled on-chain in real time.

## ZEONE Solution
- **Optimistic Execution:** Services are rendered instantly under IOU (credits) recorded in the local Ledger.
- **Trust Score as Collateral:** Nodes accumulate Trust Score through good behavior. Failure to settle IOUs degrades Trust Score, leading to throttling and eventual isolation.
- **Batch Settlement:** Debts are aggregated and cleared in batches (e.g., every 100 credits) via ERC-20 transfers through `economy/chain.py` and `settlement.py`, amortizing gas.

## Fast Path (Off-chain)
- Ledger records `record_claim`/`record_debt` per message/byte.
- BlockingTransport enforces debt limits; leechers are blocked.
- BALANCE_CLAIM/BALANCE_ACK during handshake synchronize views to prevent drift.

## Settlement Layer (On-chain)
- Credits → Tokens via configurable rate (default 1 credit = 0.001 token).
- ChainManager (Web3.py) signs ERC-20 transfers locally; SettlementManager waits for confirmations.
- Trust Score can optionally factor on-chain settlement success to improve Sybil resistance.

## Security
- Wallet addresses must be signed by the node’s Ed25519 identity to avoid MITM substitution.
- Private keys never leave the node; transactions are signed client-side and broadcast via RPC (Polygon/Arbitrum/Base or any EVM).
