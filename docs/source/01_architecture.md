# ZEONE Architecture

## Network Stack
- **Transport:** Binary Wire Protocol v1 (98-byte header, Magic 0x5A45) with NaCl (Curve25519 + XSalsa20-Poly1305) for E2E encryption; Traffic masking for DPI evasion. **[IMPLEMENTED]** Hard Fork - no JSON backward compatibility.
- **Discovery & Storage:** Kademlia DHT (RoutingTable buckets, iterative lookups, STORE/FIND\_NODE/FIND\_VALUE).
- **Protocol Router:** Ping-Pong handshake, discovery gossip, economy balance exchange.
- **Neural Transport:** NeuroLink agent for distributed tensor transmission (Pipeline Parallelism).

## Crypto & Identity
- Node identity = Ed25519 verify key (Base64). Signatures wrap all message fields (nonce, timestamp) to resist replay.
- BlockingTransport consults Ledger to block leechers exceeding debt limit.
- **Merkle Tree:** Chunk-level verification for P2P model distribution - instant corrupted data detection. **[IMPLEMENTED]**

## Persistence & Economy
- Ledger (SQLite/async) tracks debts/claims, balances, IOU, trust scores.
- Media/assets schema for knowledge_base and media_assets (phash, faces, brands, tech_meta).

## Web3 Integration
- `economy/chain.py`: Web3.py bridge (HTTP provider), ERC-20 interface (transfer, balanceOf), account from local private key (never sent over network).
- `economy/settlement.py`: Converts credits → tokens, signs transactions locally, waits confirmations. Private key is read from env or secure input; signing happens client-side only.

## Security Notes
- Private keys remain on the node; tx are signed locally and broadcast via RPC.
- Address binding: wallet addresses should be signed by the node’s Ed25519 identity to prevent MITM substitution.

## UI Layer
- NiceGUI frontend with Three.js NeuralVis iframe. WebSocket endpoints for live graph/state.
- Background ingestion (IdleWorker) and Cortex tabs for investigations, library, compliance badges.
