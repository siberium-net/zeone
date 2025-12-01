# P2P Network Skeleton

Каркас децентрализованной P2P-сети нового поколения на Python (asyncio).

## [ARCH] Архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Node (узел)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Identity   │  │  Transport   │  │   Protocol   │  │    Discovery     │ │
│  │  (Ed25519)   │  │  (NaCl Box)  │  │  (Ping-Pong) │  │  (Bloom+Gossip)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Kademlia    │  │     NAT      │  │  Persistence │  │     Security     │ │
│  │    DHT       │  │  Traversal   │  │   Manager    │  │  (DoS Protect)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Ledger     │  │    Agents    │  │ PeerManager  │  │    Monitoring    │ │
│  │  (SQLite)    │  │  (Sandbox)   │  │              │  │  (Health+Metrics)│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
           ┌─────────────────────────────────────────────────┐
           │              P2P Network (Mesh)                 │
           │                                                 │
           │   [Node A] ◄──► [Node B] ◄──► [Node C]          │
           │       ▲              ▲              ▲           │
           │       └──────────────┴──────────────┘           │
           └─────────────────────────────────────────────────┘
                                      │
                                      ▼
           ┌─────────────────────────────────────────────────┐
           │          Distributed AI Inference               │
           │                                                 │
           │   [Shard 0-16] → [Shard 17-32] → [Shard 33-48] │
           │    (Node A)       (Node B)        (Node C)      │
           └─────────────────────────────────────────────────┘
```

## [INSTALL] Установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или: venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## [RUN] Запуск

### Первый узел (Bootstrap)

```bash
python main.py --port 8468
```

### Подключение к сети

```bash
# Второй узел
python main.py --port 8469 --bootstrap 127.0.0.1:8468

# Третий узел
python main.py --port 8470 --bootstrap 127.0.0.1:8468,127.0.0.1:8469
```

### Параметры командной строки

```
--port, -p      Порт для прослушивания (по умолчанию: 8468)
--host, -H      Адрес для привязки (по умолчанию: 0.0.0.0)
--bootstrap, -b Bootstrap узлы (через запятую: host:port,host:port)
--identity, -i  Путь к файлу идентичности (по умолчанию: identity.key)
--db, -d        Путь к базе данных (по умолчанию: ledger.db)
--masking, -m   Включить HTTP-маскировку трафика
--no-shell      Режим демона без интерактивной оболочки
--verbose, -v   Подробное логирование
```

### Интерактивные команды

После запуска доступны команды:

```
peers           - Список подключенных пиров
known           - Список известных (не подключенных) пиров
ping <node_id>  - Отправить PING пиру
broadcast <msg> - Отправить сообщение всем пирам
stats           - Статистика узла
trust <node_id> - Показать Trust Score пира
ledger          - Статистика реестра
id              - Показать ID этого узла
help            - Справка
quit            - Выход
```

## [STRUCT] Структура проекта

```
├── main.py                  # Точка входа
├── config.py                # Конфигурация
├── requirements.txt         # Зависимости
├── Dockerfile               # Docker образ
├── docker-compose.yml       # Multi-node setup
│
├── core/
│   ├── __init__.py
│   ├── node.py              # Класс Node, TCP сервер
│   ├── transport.py         # Шифрование (PyNaCl), HTTP-маскировка
│   ├── protocol.py          # Ping-Pong, типы сообщений
│   ├── discovery.py         # Bloom filter, Gossip с TTL
│   │
│   ├── dht/                 # [NEW] Kademlia DHT
│   │   ├── routing.py       # K-buckets, XOR-метрика
│   │   ├── storage.py       # Key-Value хранилище
│   │   └── protocol.py      # FIND_NODE, FIND_VALUE, STORE
│   │
│   ├── nat/                 # [NEW] NAT Traversal
│   │   ├── stun.py          # STUN клиент
│   │   ├── hole_punch.py    # UDP/TCP hole punching
│   │   ├── relay.py         # P2P Relay сервер
│   │   ├── ice.py           # ICE Agent
│   │   └── candidates.py    # Candidate gathering
│   │
│   ├── persistence/         # [NEW] Сохранение состояния
│   │   ├── state_manager.py # Сериализация состояния
│   │   └── peer_store.py    # Хранение пиров
│   │
│   ├── security/            # [NEW] Защита от атак
│   │   ├── rate_limiter.py  # Token bucket
│   │   └── dos_protector.py # DoS/DDoS защита
│   │
│   └── monitoring/          # [NEW] Мониторинг
│       ├── health.py        # Health checks (K8s ready)
│       └── metrics.py       # Prometheus метрики
│
├── economy/
│   ├── __init__.py
│   └── ledger.py            # SQLite, Trust Score, IOU
│
├── agents/
│   ├── __init__.py
│   ├── manager.py           # RestrictedPython sandbox
│   ├── ai_assistant.py      # [NEW] OpenAI-compatible API
│   ├── local_llm.py         # [NEW] Ollama агент
│   ├── web_reader.py        # [NEW] Web scraper
│   ├── distributed_agent.py # [NEW] Distributed LLM
│   │
│   └── distributed/         # [NEW] Pipeline Parallelism
│       ├── registry.py      # DHT реестр моделей
│       ├── shard.py         # Model shard
│       ├── worker.py        # GPU worker
│       ├── pipeline.py      # Pipeline координатор
│       └── client.py        # Inference клиент
│
└── webui/                   # [NEW] Web Interface
    └── app.py               # NiceGUI приложение
```

## [DHT] Kademlia DHT

Распределённая хеш-таблица для хранения данных в сети.

### Принципы

- **XOR-метрика**: Расстояние между узлами = XOR их ID
- **K-buckets**: 160 корзин (по битам расстояния), k=20 узлов в каждой
- **Итеративный lookup**: Alpha=3 параллельных запроса
- **Republish**: Данные периодически переопубликуются

### Операции

```python
from core.dht import RoutingTable, DHTProtocol, DHTStorage

# Сохранить значение
await kademlia.put("my_key", b"my_value")

# Получить значение
value = await kademlia.get("my_key")

# Найти ближайшие узлы
nodes = await kademlia.find_node(target_id)
```

### Структура

```
┌─────────────────────────────────────────┐
│            Routing Table                │
├─────────────────────────────────────────┤
│  Bucket 0: [nodes at distance 2^0]      │
│  Bucket 1: [nodes at distance 2^1]      │
│  ...                                    │
│  Bucket 159: [nodes at distance 2^159]  │
└─────────────────────────────────────────┘
```

## [NAT] NAT Traversal

Подключение узлов за NAT без центрального сервера.

### Типы NAT

| Тип | Описание | Hole Punch |
|-----|----------|------------|
| Full Cone | Любой внешний хост может отправить пакет | [OK] Легко |
| Restricted Cone | Только хосты, которым мы отправляли | [OK] Возможно |
| Port Restricted | Только хосты+порты, которым мы отправляли | [OK] Сложнее |
| Symmetric | Разный mapping для каждого destination | [FAIL] Нужен Relay |

### Приоритет подключения

1. **Direct** - Оба узла имеют публичный IP
2. **Hole Punch** - UDP hole punching, затем TCP
3. **P2P Relay** - Через узел с публичным IP (не центральный сервер!)

### Использование

```python
from core.nat import STUNClient, ICEAgent, HolePuncher

# Определить тип NAT
stun = STUNClient()
nat_type, public_addr = await stun.discover()

# ICE для установки соединения
ice = ICEAgent(node)
connection = await ice.connect(peer_id)
```

## [PERSISTENCE] Сохранение состояния

Восстановление узла после перезапуска.

### Что сохраняется

- Список известных пиров (IP, порт, node_id, last_seen)
- DHT routing table
- Балансы и Trust Score
- Активные сессии

### Использование

```python
from core.persistence import StateManager, PeerStore

# Сохранение
state_manager = StateManager("./data")
await state_manager.save_state(node)

# Восстановление
await state_manager.load_state(node)
```

## [SECURITY] Безопасность

### Криптография

- **Идентичность**: Ed25519 ключевая пара (через PyNaCl)
- **Подписи**: Ed25519 для верификации сообщений
- **Шифрование**: NaCl Box (Curve25519 + XSalsa20 + Poly1305)

### Ping-Pong протокол

```
[Node A]                    [Node B]
   │                           │
   │  PING (nonce + signature) │
   │ ─────────────────────────►│
   │                           │ Verify signature
   │                           │ Check timestamp
   │  PONG (original_nonce)    │
   │ ◄─────────────────────────│
   │                           │
   │ Verify PONG signature     │
   │ Verify nonce matches      │
```

### DoS Protection

```python
from core.security import RateLimiter, DoSProtector

# Rate limiter
limiter = RateLimiter(rules=[
    RateLimitRule(requests=100, period=60),   # 100 req/min
    RateLimitRule(requests=1000, period=3600), # 1000 req/hour
])

# DoS protector
protector = DoSProtector()
threat = protector.analyze(peer_id, message)
if threat.level == ThreatLevel.HIGH:
    await protector.ban(peer_id, duration=3600)
```

### Песочница (Sandbox)

Контракты выполняются в RestrictedPython с ограничениями:
- Белый список встроенных функций
- Нет доступа к файловой системе
- Нет сетевых операций
- Лимит времени выполнения (5 сек)
- Лимит размера кода (64 КБ)

## [MONITORING] Мониторинг

### Health Checks

```python
from core.monitoring import HealthChecker, HealthStatus

health = HealthChecker(node, ledger, kademlia)
status = await health.check_all()

# Для Kubernetes liveness/readiness probes
# GET /health -> {"status": "healthy", "components": {...}}
```

### Метрики

```python
from core.monitoring import MetricsCollector, Counter, Gauge

metrics = MetricsCollector()

# Счётчики
messages_sent = metrics.counter("messages_sent_total")
messages_sent.inc()

# Gauges
peers_connected = metrics.gauge("peers_connected")
peers_connected.set(len(node.peers))

# Prometheus формат
# GET /metrics -> peers_connected 5\nmessages_sent_total 1234
```

## [DISCOVERY] Улучшенный Discovery

### Оптимизации для больших сетей

| Механизм | Описание |
|----------|----------|
| **Bloom Filter** | Вероятностная проверка известных узлов (1% false positive) |
| **Exponential Backoff** | Уменьшение частоты запросов при неудачах |
| **TTL Gossip** | Ограничение глубины распространения (TTL=3) |
| **Trust Priority** | Приоритизация узлов по Trust Score |

```python
from core.discovery import OptimizedDiscovery, BloomFilter

discovery = OptimizedDiscovery(node, ledger)
await discovery.start()

# Gossip автоматически:
# - Обменивается списками пиров каждые 30 сек
# - Фильтрует известные узлы через Bloom filter
# - Приоритизирует узлы с высоким Trust Score
```

## [DISTRIBUTED] Распределённый AI Инференс

Запуск больших моделей (70B+) через P2P сеть.

### Архитектура

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Client   │───►│  Shard 0   │───►│  Shard 1   │───►│  Shard 2   │
│ (tokenize) │    │ Layers 0-16│    │Layers 17-32│    │Layers 33-48│
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                       GPU A             GPU B             GPU C
```

### Компоненты

| Компонент | Описание |
|-----------|----------|
| **ModelRegistry** | DHT реестр доступных shards |
| **ModelShard** | Часть модели (диапазон слоёв) |
| **InferenceWorker** | GPU воркер для инференса |
| **PipelineCoordinator** | Координация pipeline |
| **DistributedClient** | Клиент для запросов |

### Использование

```python
from agents.distributed import (
    DistributedInferenceClient,
    ModelRegistry,
    InferenceWorker,
)

# На GPU узле: запуск воркера
worker = InferenceWorker(
    model_name="qwen2.5-32b",
    layer_start=0,
    layer_end=16,
)
await worker.start()

# На клиенте: инференс
client = DistributedInferenceClient(node, registry)
result = await client.generate(
    model="qwen2.5-32b",
    prompt="Explain quantum computing",
    max_tokens=100,
)
```

### Fault Tolerance

- Несколько узлов могут обслуживать один shard
- Автоматическое переключение при отказе
- Health checks через DHT

## [AGENTS] AI Агенты

### Доступные агенты

| Агент | Сервис | Цена | Описание |
|-------|--------|------|----------|
| `LlmAgent` | `llm_prompt` | 50 | OpenAI-compatible API (GPT-4o-mini) |
| `OllamaAgent` | `llm_local` | 30 | Локальный Ollama (qwen3:32b) |
| `DistributedLlmAgent` | `llm_distributed` | 20 | Распределённый через P2P |
| `ReaderAgent` | `web_read` | 10 | Чтение веб-страниц |
| `EchoAgent` | `echo` | 1 | Эхо (для тестов) |

### Пример использования

```python
from agents.ai_assistant import LlmAgent
from agents.local_llm import OllamaAgent

# Cloud LLM
cloud = LlmAgent(api_key="sk-...")
result, cost = await cloud.execute({
    "prompt": "What is the meaning of life?",
})

# Local Ollama
ollama = OllamaAgent(model_name="qwen3:32b")
result, cost = await ollama.execute("Explain P2P networks")
```

## [WEBUI] Web Interface

NiceGUI интерфейс для управления узлом.

### Запуск

```python
from webui.app import P2PWebUI

webui = P2PWebUI(node, ledger, agent_manager, kademlia)
webui.run_sync(host="0.0.0.0", port=8080)
```

### Страницы

| Страница | Описание |
|----------|----------|
| `/` | Dashboard - статус узла, пиры, услуги |
| `/peers` | Управление подключениями |
| `/services` | Список и тестирование услуг |
| `/ai` | Чат с AI (Cloud/Local/Distributed) |
| `/dht` | DHT операции (put/get/delete) |
| `/economy` | Балансы и транзакции |
| `/storage` | Хранение файлов |
| `/compute` | Вычисления (eval, hash) |
| `/settings` | Настройки узла |
| `/logs` | Логи в реальном времени |

## [DOCKER] Контейнеризация

### Запуск одного узла

```bash
docker build -t p2p-node .
docker run -p 8468:8468 p2p-node
```

### Multi-node с Docker Compose

```bash
docker-compose up -d

# Node A: порт 8000 (bootstrap)
# Node B: порт 8001 (подключается к Node A)
```

### docker-compose.yml

```yaml
services:
  node-a:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - NODE_NAME=Alpha

  node-b:
    build: .
    ports:
      - "8001:8001"
    command: python main.py --port 8001 --bootstrap node-a:8000
    depends_on:
      - node-a
```

## [DECENTRALIZATION] Принципы децентрализации

### 1. Нет центрального сервера

Каждый узел равноправен. Bootstrap-узлы нужны только для первого подключения.

### 2. Криптографическая идентичность

Узел = публичный ключ. Нет центра регистрации.

```python
# Генерация новой идентичности
from core.transport import Crypto
crypto = Crypto()
print(f"Node ID: {crypto.node_id}")
```

### 3. Верификация всех сообщений

Каждое сообщение подписывается отправителем и проверяется получателем.

### 4. Распределенный реестр

Каждый узел хранит свою копию данных. Согласованность через подписи.

### 5. Trust Score

Репутация строится на поведении, а не на регистрации.

```python
# События, влияющие на Trust Score
"successful_transfer": +0.01
"failed_transfer": -0.05
"valid_message": +0.001
"invalid_message": -0.02
"iou_redeemed": +0.02
"iou_defaulted": -0.1
```

## [ECONOMY] Экономика сети

### IOU (долговые расписки)

```python
from economy.ledger import Ledger, IOU

ledger = Ledger()
await ledger.initialize()

# Узел B должен узлу A 10 кредитов
iou = await ledger.create_iou(
    debtor_id=node_b_id,
    creditor_id=node_a_id,
    amount=10.0,
    signature=signature_from_node_b,
)

# Погашение IOU
success = await ledger.redeem_iou(iou.id)
```

## [API] Программный интерфейс

### Создание узла

```python
import asyncio
from core.node import Node
from core.transport import Crypto

async def main():
    crypto = Crypto()
    node = Node(crypto=crypto, port=8468)
    
    async def on_peer(peer):
        print(f"Peer connected: {peer.node_id[:16]}...")
    
    node.on_peer_connected(on_peer)
    await node.start()
    
    # Подключаемся к другому узлу
    peer = await node.connect_to_peer("127.0.0.1", 8469)
    
    # Broadcast сообщение
    from core.transport import Message, MessageType
    msg = Message(
        type=MessageType.DATA,
        payload={"text": "Hello network!"},
        sender_id=crypto.node_id,
    )
    count = await node.broadcast(msg)
    print(f"Sent to {count} peers")
    
    await node.stop()

asyncio.run(main())
```

## [LIMITS] Текущие ограничения

| Компонент | Статус | Комментарий |
|-----------|--------|-------------|
| Discovery | [OK] | Оптимизирован с Bloom filter и TTL |
| NAT Traversal | [OK] | STUN, Hole Punch, P2P Relay, ICE |
| Kademlia DHT | [OK] | Полная реализация |
| Persistence | [OK] | Состояние сохраняется между перезапусками |
| Rate Limiting | [OK] | Token bucket + DoS protection |
| WebRTC | [PENDING] | Планируется как альтернативный транспорт |
| Tor/I2P | [PENDING] | Для анонимности |
| Mobile | [PENDING] | Мобильные клиенты |

## [TODO] Возможные улучшения

- [ ] WebRTC транспорт для браузерных клиентов
- [ ] Tor/I2P интеграция для анонимности
- [ ] Мобильные клиенты (React Native / Flutter)
- [ ] Консенсус-механизм для синхронизации данных
- [ ] IPFS-совместимое хранилище
- [ ] Multi-GPU sharding на одном узле
- [ ] Streaming inference для LLM
- [ ] Автоматический load balancing shards

## [LICENSE] Лицензия

MIT License
