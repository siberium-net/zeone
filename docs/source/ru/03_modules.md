# Модули (RU)

## Cortex
- **Пайплайн:** Ingest → Vision (Florence-2) → Vector Store → RAG ответы.
- **Мультимодальность:** OCR, face embeddings (InsightFace), pHash дедупликация.
- **Автономность:** CortexService может работать в фоне, собирая и обрабатывая данные.

## VPN и Pathfinder
- **VpnExitAgent:** Открывает реальные TCP соединения, считает байты и биллит через Ledger. Публикует метаданные (страна, цена, пропускная способность) в DHT.
- **Pathfinder:** Находит выходы (`service:vpn_exit`), меряет задержку, выбирает стратегию (fastest/cheapest/reliable), кэширует маршруты в `vpn_routes`.
- **SocksServer:** Локальный SOCKS5 (по умолчанию 1080) туннелирует через выбранный exit и может следовать редиректам Amplifier для кэша.

## Amplifier (дедупликация трафика)
- **Корреляция:** Протокол `CACHE_REQUEST` / `CACHE_RESPONSE` запрашивает чанки по SHA-256 между пирам.
- **Кэш:** LRU файловый кэш; наличие публикуется в DHT (`cache:<hash>`).
- **Поток:** Exit сообщает хэш сегмента → клиент кеширует → сосед запрашивает хэш у пиров → получает без повторной загрузки с источника.

## Экономика
- **Ledger:** IOU, Trust Score, блокировка leechers.
- **Гибридные расчеты:** Off-chain кредиты + on-chain ERC-20 settlement.
- **Billing hooks:** VpnExitAgent и BlockingTransport автоматически фиксируют байты/стоимость.
