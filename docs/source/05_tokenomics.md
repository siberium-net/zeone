# Tokenomics (ZEO)

## Utility
- **Medium of Exchange:** Pay for AI inference, storage, relay bandwidth. IOU credits convert to ZEO or paired ERC-20 (e.g., USDT) at settlement.
- **Staking (Sybil Resistance):** Nodes stake ZEO to become High Trust Providers. Higher stake → higher routing priority, larger job allocations, and staking yield. Slashing on fraud/failed settlements.
- **Governance:** Staked nodes vote on network parameters: fee schedule, supported AI models, compliance thresholds, settlement batch sizes.

## Burn Mechanism
- A protocol fee is taken on each settlement; a portion is burned to create deflationary pressure.
- Burn rate is configurable via governance; transparent on-chain accounting.

## Flow of Value
```
Requestor --(IOU credits)--> Provider
   |                           |
   |----(Batch Settlement: ZEO/USDT ERC-20)---->|

Staking Pool <--- ZEO stake ---- Providers
    |                             |
    +--- Governance / Slashing ---+

Burn Sink <--- Protocol Fee (portion burned)
```

## Liquidity and Velocity
- Off-chain IOUs maximize velocity (near-zero latency).
- On-chain settlement batches provide finality and liquidity bridging to external markets.
- Staking locks supply, reducing float and aligning incentives with service quality.
