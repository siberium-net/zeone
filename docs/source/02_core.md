# Core Architecture

## Protocol
- **Handshake:** TCP/UDP сокеты обмениваются PING/PONG с подписями NaCl Ed25519, проверяя владение ключом.
- **Encryption:** Payload шифруется через Curve25519 + XSalsa20-Poly1305 (NaCl Box). Маскировка под HTTP доступна через TrafficMasker.
- **Streaming:** `STREAM`/`VPN_DATA` сообщения используют `StreamingMessage` с `seq` для сборки потока в правильном порядке.

## DHT (Kademlia)
- XOR-метрика, 160-битные ID, k-buckets.
- Операции PUT/GET/STORE реализованы в `core.dht.*`; значения реплицируются и обновляются по TTL.
- DHT используется для рекламы услуг (например, `service:vpn_exit`) и кэшированных чанков (`cache:<hash>`).

## P2P Loader
- Torrent-like распространение моделей и данных: распределенное скачивание по чанкам с учетом доверия и учета трафика в Ledger.
- Интеграция с Ledger: BlockingTransport проверяет баланс пиров перед отправкой, предотвращая leechers.

## Security & Accounting
- **Ledger:** Async SQLite, IOU, Trust Score, блокировка пиров с большим долгом.
- **BlockingTransport:** записывает `record_claim`/`record_debt` при передаче, синхронизирован с MessageTypes.
- **Compliance hooks:** (опционально) позволяют встраивать фильтры в Cortex/ingest.
