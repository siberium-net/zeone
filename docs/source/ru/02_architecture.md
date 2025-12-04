# Архитектура (RU)

## Протокол
- **Handshake:** TCP/UDP обмениваются подписанными PING/PONG (Ed25519), подтверждая владение ключом.
- **Шифрование:** Curve25519 + XSalsa20-Poly1305 (NaCl Box); опциональная HTTP-маскировка (TrafficMasker).
- **Стриминг:** `STREAM` / `VPN_DATA` с `StreamingMessage` и `seq` для восстановления порядка.
- **Протокол кэша:** `CACHE_REQUEST` / `CACHE_RESPONSE` переносит кэшированные чанки между пирам с проверкой хэша.

## DHT (Kademlia)
- XOR-метрика, 160-битные ID, k-бакеты; итеративные FIND/STORE с TTL.
- Используется для рекламы услуг (`service:vpn_exit`) и кэшированных чанков (`cache:<hash>`), а также discovery.

## P2P Loader
- Torrent-подобная раздача моделей/данных по чанкам с учетом доверия.
- Интеграция с Ledger: BlockingTransport проверяет баланс и пишет `record_claim` / `record_debt` за каждую передачу.

## Безопасность и учет
- **Ledger:** async SQLite, IOU, Trust Score, блокировка leechers.
- **Compliance hooks:** опциональные фильтры на ingest и в Cortex.
- **Rate limit / DoS:** модули core.security при включении.
