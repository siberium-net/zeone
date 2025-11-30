# P2P Network Skeleton

Каркас децентрализованной P2P-сети нового поколения на Python (asyncio).

## [ARCH] Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                         Node (узел)                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Identity   │  │  Transport   │  │     Protocol         │   │
│  │  (Ed25519)   │  │  (NaCl Box)  │  │   (Ping-Pong)        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Ledger     │  │    Agents    │  │    PeerManager       │   │
│  │  (SQLite)    │  │  (Sandbox)   │  │   (Discovery)        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │           P2P Network (Mesh)            │
        │                                         │
        │   [Node A] ◄──► [Node B] ◄──► [Node C]  │
        │       ▲              ▲              ▲    │
        │       └──────────────┴──────────────┘    │
        └─────────────────────────────────────────┘
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
├── main.py              # Точка входа
├── config.py            # Конфигурация
├── requirements.txt     # Зависимости
├── core/
│   ├── __init__.py
│   ├── node.py          # Класс Node, TCP сервер, Discovery
│   ├── transport.py     # Шифрование (PyNaCl), HTTP-маскировка
│   └── protocol.py      # Ping-Pong, типы сообщений
├── economy/
│   ├── __init__.py
│   └── ledger.py        # SQLite, Trust Score, IOU
└── agents/
    ├── __init__.py
    └── manager.py       # RestrictedPython sandbox
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

PONG возвращается **только если подпись PING валидна**.

### Песочница (Sandbox)

Контракты выполняются в RestrictedPython с ограничениями:
- Белый список встроенных функций
- Нет доступа к файловой системе
- Нет сетевых операций
- Лимит времени выполнения (5 сек)
- Лимит размера кода (64 КБ)

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

```python
# Создание подписанного сообщения
message = Message(type=MessageType.DATA, payload={"text": "Hello"}, sender_id=crypto.node_id)
signed = crypto.sign_message(message)

# Проверка подписи
is_valid = crypto.verify_signature(message)
```

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
# Создание IOU
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

### Trust Score

```python
# Обновление репутации
new_score = await ledger.update_trust_score(
    node_id=peer_id,
    event="successful_transfer",
    magnitude=1.0,
)
```

## [AGENTS] Система контрактов

### Пример контракта

```python
# calculator.py
# Входные данные: a, b, operation

if operation == "add":
    _result_ = a + b
elif operation == "sub":
    _result_ = a - b
elif operation == "mul":
    _result_ = a * b
elif operation == "div":
    _result_ = a / b if b != 0 else "Division by zero"
```

### Выполнение контракта

```python
from agents.manager import AgentManager, Contract

manager = AgentManager()

contract = Contract(
    code='_result_ = a + b',
    author_id=author_node_id,
    signature=author_signature,
    name="simple_add",
)

result = await manager.execute(
    contract,
    inputs={"a": 5, "b": 3},
)

print(result.output)  # 8
```

## [API] Программный интерфейс

### Создание узла

```python
import asyncio
from core.node import Node
from core.transport import Crypto

async def main():
    # Создаем идентичность
    crypto = Crypto()
    
    # Создаем узел
    node = Node(crypto=crypto, port=8468)
    
    # Callback на подключение пира
    async def on_peer(peer):
        print(f"Peer connected: {peer.node_id[:16]}...")
    
    node.on_peer_connected(on_peer)
    
    # Запускаем
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
    
    # Останавливаем
    await node.stop()

asyncio.run(main())
```

## [LIMITS] Ограничения текущей реализации

1. **Discovery** - базовый gossip-протокол, не оптимизирован для больших сетей
2. **NAT Traversal** - не реализован (требуется STUN/TURN для работы за NAT)
3. **DHT** - нет распределенной хеш-таблицы для поиска данных
4. **Persistence** - состояние пиров не сохраняется между перезапусками
5. **Rate Limiting** - базовая защита от DoS

## [TODO] Возможные улучшения

- [ ] Kademlia DHT для распределенного хранения
- [ ] NAT traversal (STUN/TURN/ICE)
- [ ] WebRTC транспорт
- [ ] Консенсус-механизм для синхронизации данных
- [ ] Tor/I2P интеграция для анонимности
- [ ] Мобильные клиенты

## [LICENSE] Лицензия

MIT License
