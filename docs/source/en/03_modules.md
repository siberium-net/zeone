# Modules (EN)

## Cortex
- **Pipeline:** Ingest → Vision (Florence-2) → Vector Store → RAG responses.
- **Multimodal:** OCR, face embeddings (InsightFace), pHash deduplication.
- **Autonomous:** CortexService can run background ingestion and council-style reasoning.

## VPN & Pathfinder
- **VpnExitAgent:** Opens real TCP connections to targets, counts bytes, and bills via Ledger. Publishes metadata (country, price, bandwidth) to DHT.
- **Pathfinder:** Discovers exits (`service:vpn_exit`), probes latency, selects strategy (fastest/cheapest/reliable), caches routes in `vpn_routes`.
- **SocksServer:** Local SOCKS5 (default 1080) tunnels via chosen exit and can follow Amplifier redirects for cached data.

## Amplifier (Traffic Deduplication)
- **Correlation:** `CACHE_REQUEST` / `CACHE_RESPONSE` protocol to fetch chunks by SHA-256 hash between peers.
- **Cache:** LRU file cache; availability advertised to DHT (`cache:<hash>`).
- **Flow:** Exit reports segment hash → client caches → neighbor requests same hash from peers → served without re-downloading origin.

## Economy
- **Ledger:** IOU, Trust Score, leecher blocking.
- **Hybrid settlement:** Off-chain credits + on-chain ERC-20 settlement.
- **Billing hooks:** VpnExitAgent and BlockingTransport record bytes/cost automatically.
