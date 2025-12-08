# Hybrid Economy: Probabilistic Settlement

## Problem
Public blockchains have high gas cost and limited TPS; micro-transactions for inference/storage/relay are uneconomical if settled on-chain in real time.

## ZEONE Solution
- **Optimistic Execution:** Services are rendered instantly under IOU (credits) recorded in the local Ledger.
- **Trust Score as Collateral:** Nodes accumulate Trust Score through good behavior. Failure to settle IOUs degrades Trust Score, leading to throttling and eventual isolation. **[IMPLEMENTED]** Weighted Trust Score with stake weighting.
- **Batch Settlement:** Debts are aggregated and cleared in batches (e.g., every 100 credits) via **Native SIBR transfers** through `economy/chain.py` (SiberiumManager) and `settlement.py`, amortizing gas. **[UPDATED]** Native token, not ERC-20.

## Fast Path (Off-chain)
- Ledger records `record_claim`/`record_debt` per message/byte.
- BlockingTransport enforces debt limits; leechers are blocked.
- BALANCE_CLAIM/BALANCE_ACK during handshake synchronize views to prevent drift.

## Settlement Layer (On-chain)
- Credits → Tokens via configurable rate (default 1 credit = 0.001 token).
- **SiberiumManager** (Web3.py) signs **Native SIBR** transfers locally; SettlementManager waits for confirmations. **[UPDATED]**
- **ZEOSettlement.sol:** Native staking contract with deposit(), claim(), unstake flow (7-day timelock). **[IMPLEMENTED]**
- **Networks:** Siberium Mainnet (ChainID 111111), Testnet (ChainID 111000).
- Trust Score factors on-chain settlement success (Weighted Trust Score with stake weight). **[IMPLEMENTED]**

## Security
- Wallet addresses must be signed by the node's Ed25519 identity to avoid MITM substitution.
- Private keys never leave the node; transactions are signed client-side and broadcast via RPC (Siberium RPC).
- **Weighted Trust Score:** Economic Sybil resistance through stake weighting formula `T_effective = T_behavior * log10(1 + Stake / BaseStake)`. **[IMPLEMENTED]**
- **Slashing:** Invalid Merkle proofs, double-spend attempts, signature forgery → instant Trust Score = 0 + blacklist. **[IMPLEMENTED]**
