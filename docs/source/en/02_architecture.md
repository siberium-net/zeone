# Core Architecture (EN)

## Protocol
- **Handshake:** TCP/UDP sockets exchange signed PING/PONG (Ed25519) to prove key ownership.
- **Encryption:** Payloads via Curve25519 + XSalsa20-Poly1305 (NaCl Box); optional HTTP masking (TrafficMasker).
- **Streaming:** `STREAM` / `VPN_DATA` messages use `StreamingMessage` with `seq` to reassemble ordered streams.
- **Cache Protocol:** `CACHE_REQUEST` / `CACHE_RESPONSE` moves cached chunks between peers with hash verification.

## DHT (Kademlia)
- XOR metric, 160-bit IDs, k-buckets; iterative FIND/STORE with TTL-based replication.
- Used to advertise services (`service:vpn_exit`) and cached chunks (`cache:<hash>`), as well as peer discovery.

## P2P Loader
- Torrent-like chunk distribution for models/data with trust-aware piece selection.
- Ledger integration: BlockingTransport enforces debt limits and records `record_claim` / `record_debt` per transfer.

## Security & Accounting
- **Ledger:** Async SQLite, IOU, Trust Score, leecher blocking.
- **Compliance hooks:** Optional filters at ingest and Cortex.
- **Rate limiting / DoS:** Available through core.security modules when enabled.
