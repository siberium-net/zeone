# Modules Overview

## Cortex
- **Pipeline:** Ingest → Vision (Florence-2) → Vector Store → RAG.
- **Multimodal:** OCR, face embeddings (InsightFace), pHash deduplication.
- **Autonomous Loop:** CortexService orchestrates background ingestion and council-based reasoning.

## VPN & Pathfinder
- **VpnExitAgent:** Поднимает реальные TCP соединения к целевым хостам, считает байты и списывает стоимость через Ledger. Метаданные (страна, цена, пропускная способность) публикуются в DHT.
- **Pathfinder:** Ищет выходы по ключу `service:vpn_exit`, пингует кандидатов, выбирает стратегию (fastest/cheapest/reliable), кеширует маршруты в `vpn_routes`.
- **SocksServer:** Локальный SOCKS5 (1080 по умолчанию) туннелирует трафик через выбранный exit и умеет получать редиректы от Amplifier.

## Amplifier (Traffic Deduplication)
- **Correlation:** Протокол `CACHE_REQUEST` / `CACHE_RESPONSE` для запроса чанков по SHA-256 хэшу между пирами.
- **Cache:** LRU файловый кэш, публикация наличия чанков в DHT (`cache:<hash>`).
- **Flow:** Exit узел сообщает хэш сегмента → клиент кэширует → соседний клиент запрашивает хэш у пиров → получает данные без повторной загрузки.

## Economy
- **Ledger:** IOU, Trust Score, blocking leechers.
- **Hybrid settlement:** Off-chain кредиты + ERC-20 периодическая оплата.
- **Billing hooks:** VpnExitAgent и BlockingTransport автоматически записывают байты и стоимость.
