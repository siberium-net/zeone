[🇺🇸 Read in English](README.md)

![ZEONE](https://img.shields.io/badge/version-v1.2.4-blue) ![Python](https://img.shields.io/badge/python-3.12%2B-blueviolet) ![License](https://img.shields.io/badge/license-MIT-black) ![Network](https://img.shields.io/badge/network-Sovereign-green)

# ZEONE — децентрализованная операционная система для когнитивного интернета

ZEONE объединяет вычисления, трафик и экономику в одноранговый стек — от NaCl-транспорта до локальных LLM и P2P-кэша видео.

## Возможности
- 🧠 **Cortex:** Локальный LLM + RAG + Vision (Florence-2) пайплайн.
- 🛡️ **VPN Tunnel:** Децентрализованный SOCKS5 с `VpnExitAgent` и умным Pathfinder (скорость/цена/надежность).
- 🚀 **Amplifier:** P2P CDN с `CACHE_REQUEST` / `CACHE_RESPONSE` обменом чанками и кэшированием видео/файлов.
- 💎 **Tokenomics:** Ledger IOU + ERC-20 settlement, Trust Score, биллинг за трафик и услуги.

## Быстрый старт (Docker)
```bash
docker-compose up
```

WebUI по умолчанию: `http://localhost:8080`.

## Запуск вручную (Python 3.12+)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --port 8468 --webui --webui-port 8080
```

Bootstrap по умолчанию: `boot.ze1.org:80` (переопределяется через `--bootstrap`).

Подключить второй узел:
```bash
python main.py --port 8469 --bootstrap 127.0.0.1:8468 --webui --webui-port 8081
```

## VPN / SOCKS5 (CLI)

Exit-узел (реклама публичного IP):
```bash
python main.py --exit-node --public-ip 1.2.3.4
```

Клиент (локальный SOCKS5 на 127.0.0.1:9999):
```bash
python main.py --vpn-client --socks-port 9999 --vpn-region US
curl --socks5-hostname 127.0.0.1:9999 https://ifconfig.me
```

## MCP (SSE)

```bash
python main.py --mcp --mcp-port 8090
```

Эндпоинты:
- `http://localhost:8090/mcp/sse`
- `http://localhost:8090/mcp/messages`

## Документация
- Сборка: `python build_docs.py`
- HTML вход: `docs/build/html/index.html`

## Лицензия
MIT
