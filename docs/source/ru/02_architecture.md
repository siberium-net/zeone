# Архитектура ядра ZEONE

## Обзор

Ядро ZEONE состоит из трёх основных слоёв:
1. **Transport Layer** — шифрование, подписи, wire protocol
2. **DHT Layer** — Kademlia для routing и storage
3. **Service Layer** — обработчики сообщений и RPC

---

## 1. Wire Protocol

### 1.1 Текущая реализация (Legacy)

Текущий `SimpleTransport` использует простой формат:

```
┌─────────────────┬──────────────────────────────────┐
│ Length (4 bytes)│ JSON Payload (variable)          │
│ big-endian      │ UTF-8 encoded Message            │
└─────────────────┴──────────────────────────────────┘
```

**Структура Message (JSON):**
```json
{
  "type": "PING",
  "payload": {},
  "sender_id": "base64(Ed25519_pubkey)",
  "timestamp": 1701234567.89,
  "signature": "base64(Ed25519_sig)",
  "nonce": "base64(random_16_bytes)"
}
```

**Недостатки:**
- JSON overhead: ~40% размера на служебные данные
- Нет версионирования протокола
- Отсутствует magic number для идентификации потока

### 1.2 [PROPOSED] Binary Wire Protocol v1

Предлагается бинарный формат для production:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┤
│         Magic (0x5A45)        │    Version    │     Type      │
├───────────────────────────────┴───────────────┴───────────────┤
│                        Length (32-bit)                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                        Nonce (24 bytes)                       │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                      Signature (64 bytes)                     │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                    Encrypted Payload (...)                    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Поля:**

| Поле | Размер | Описание |
|------|--------|----------|
| Magic | 2 bytes | `0x5A45` ("ZE" в ASCII) — идентификатор протокола |
| Version | 1 byte | Версия протокола (текущая: 0x01) |
| Type | 1 byte | Тип сообщения (enum MessageType) |
| Length | 4 bytes | Размер payload в байтах (big-endian) |
| Nonce | 24 bytes | XSalsa20 nonce для шифрования |
| Signature | 64 bytes | Ed25519 подпись (type + length + nonce + payload) |
| Payload | variable | Зашифрованный XSalsa20-Poly1305 payload |

**Message Types (1 byte):**

| Value | Type | Direction |
|-------|------|-----------|
| 0x01 | PING | Request |
| 0x02 | PONG | Response |
| 0x03 | DISCOVER | Request |
| 0x04 | PEER_LIST | Response |
| 0x10 | DHT_FIND_NODE | Request |
| 0x11 | DHT_FIND_NODE_RESP | Response |
| 0x12 | DHT_FIND_VALUE | Request |
| 0x13 | DHT_FIND_VALUE_RESP | Response |
| 0x14 | DHT_STORE | Request |
| 0x15 | DHT_STORE_RESP | Response |
| 0x20 | SERVICE_REQUEST | Request |
| 0x21 | SERVICE_RESPONSE | Response |
| 0x30 | VPN_CONNECT | Request |
| 0x31 | VPN_DATA | Bidirectional |
| 0x32 | VPN_CLOSE | Notification |
| 0x40 | CACHE_REQUEST | Request |
| 0x41 | CACHE_RESPONSE | Response |
| 0x50 | IOU | Bidirectional |
| 0x51 | BALANCE_CLAIM | Request |
| 0x52 | BALANCE_ACK | Response |

### 1.3 Handshake Protocol

Установление защищённого соединения между узлами:

```mermaid
sequenceDiagram
    participant A as Node A (Initiator)
    participant B as Node B (Responder)
    
    Note over A,B: Phase 1: Key Exchange
    A->>B: HELLO {pubkey_A, nonce_A}
    B->>A: HELLO_ACK {pubkey_B, nonce_B, sig_B(nonce_A)}
    A->>B: VERIFY {sig_A(nonce_B)}
    
    Note over A,B: Phase 2: Session Establishment
    Note over A: shared_secret = ECDH(privkey_A, pubkey_B)
    Note over B: shared_secret = ECDH(privkey_B, pubkey_A)
    Note over A,B: session_key = HKDF(shared_secret, nonce_A || nonce_B)
    
    Note over A,B: Phase 3: Encrypted Communication
    A->>B: Encrypted(PING)
    B->>A: Encrypted(PONG)
```

**Криптографические примитивы:**

| Операция | Алгоритм | Библиотека |
|----------|----------|------------|
| Подписи | Ed25519 | PyNaCl SigningKey |
| Обмен ключами | X25519 (Curve25519 ECDH) | PyNaCl Box |
| Шифрование | XSalsa20-Poly1305 | PyNaCl SecretBox |
| KDF | HKDF-SHA256 | — |

**Код генерации session key:**
```python
from nacl.public import PrivateKey, PublicKey, Box
from nacl.hash import blake2b

# ECDH
box = Box(my_private_key, peer_public_key)
shared_secret = box.shared_key()

# KDF
session_key = blake2b(
    shared_secret + nonce_a + nonce_b,
    digest_size=32,
    key=b"ZEONE_SESSION_KEY_V1"
)
```

---

## 2. Kademlia DHT

### 2.1 Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| K | 20 | Размер k-bucket (узлов на bucket) |
| α (ALPHA) | 3 | Параллельность lookup запросов |
| ID_BITS | 160 | Битность идентификаторов (SHA-1) |
| BUCKET_REFRESH_INTERVAL | 3600s | Интервал обновления bucket |
| DEFAULT_TTL | 86400s | Время жизни записи (24 часа) |
| REPUBLISH_INTERVAL | 3600s | Интервал републикации (1 час) |
| RPC_TIMEOUT | 5s | Таймаут RPC запроса |

### 2.2 XOR Distance Metric

Расстояние между двумя 160-битными ID:

$$
d(a, b) = a \oplus b
$$

интерпретируемое как беззнаковое целое число.

**Реализация:**
```python
def xor_distance(id1: bytes, id2: bytes) -> int:
    """
    XOR distance between two 20-byte node IDs.
    
    Returns:
        Integer distance (0 to 2^160 - 1)
    """
    xor_bytes = bytes(a ^ b for a, b in zip(id1, id2))
    return int.from_bytes(xor_bytes, byteorder='big')
```

**Свойства XOR-метрики:**
1. \( d(a, a) = 0 \) — рефлексивность
2. \( d(a, b) = d(b, a) \) — симметрия
3. \( d(a, b) + d(b, c) \geq d(a, c) \) — неравенство треугольника
4. Унарность: для любых \( a, b \) существует единственный \( c \): \( d(a, c) = b \)

### 2.3 Routing Table Structure

160 k-buckets, где bucket \( i \) содержит узлы с расстоянием \( 2^i \leq d < 2^{i+1} \):

```
Bucket[0]:   distance ∈ [1, 2)        — 1 бит отличается
Bucket[1]:   distance ∈ [2, 4)        — 2 бита
Bucket[2]:   distance ∈ [4, 8)        — 3 бита
...
Bucket[159]: distance ∈ [2^159, 2^160) — все биты разные
```

**Индекс bucket по расстоянию:**
```python
def distance_to_bucket_index(distance: int) -> int:
    if distance == 0:
        return 0
    return distance.bit_length() - 1
```

**LRU-политика k-bucket:**
1. Если узел уже в bucket → переместить в конец (most recently seen)
2. Если bucket не полон → добавить в конец
3. Если bucket полон:
   - PING самый старый узел (head)
   - Если ответил → отбросить новый узел
   - Если не ответил → заменить на новый

### 2.4 Iterative FIND_NODE

Поиск K ближайших узлов к target ID:

```mermaid
sequenceDiagram
    participant I as Initiator
    participant RT as Routing Table
    participant N1 as Node 1
    participant N2 as Node 2
    participant N3 as Node 3
    
    Note over I: shortlist = RT.find_closest(target, K)
    
    loop Until no closer nodes found
        par α=3 parallel queries
            I->>N1: FIND_NODE(target)
            I->>N2: FIND_NODE(target)
            I->>N3: FIND_NODE(target)
        end
        
        N1-->>I: [NodeInfo, NodeInfo, ...]
        N2-->>I: [NodeInfo, NodeInfo, ...]
        N3-->>I: [NodeInfo, NodeInfo, ...]
        
        Note over I: Merge results into shortlist
        Note over I: Sort by XOR distance
        Note over I: Keep K closest
    end
    
    Note over I: Return shortlist (K closest nodes)
```

**Псевдокод:**
```python
async def iterative_find_node(target_id: bytes) -> List[NodeInfo]:
    shortlist = routing_table.find_closest(target_id, K)
    queried = set()
    
    while True:
        # Select α unqueried nodes
        to_query = [n for n in shortlist if n.id not in queried][:ALPHA]
        if not to_query:
            break
        
        # Parallel queries
        for node in to_query:
            queried.add(node.id)
        
        responses = await asyncio.gather(*[
            send_find_node(node, target_id) for node in to_query
        ], return_exceptions=True)
        
        # Merge results
        found_closer = False
        for response in responses:
            if isinstance(response, Exception):
                continue
            for new_node in response.nodes:
                if new_node.id not in [n.id for n in shortlist]:
                    shortlist.append(new_node)
                    found_closer = True
        
        # Sort and truncate
        shortlist.sort(key=lambda n: xor_distance(target_id, n.id))
        shortlist = shortlist[:K]
        
        if not found_closer:
            break
    
    return shortlist
```

### 2.5 Iterative STORE

Сохранение пары key-value в DHT:

```mermaid
sequenceDiagram
    participant I as Initiator
    participant N1 as Closest Node 1
    participant N2 as Closest Node 2
    participant NK as Closest Node K
    
    Note over I: Step 1: Find K closest nodes to key
    I->>I: closest = iterative_find_node(key)
    
    Note over I: Step 2: Store on all K nodes
    par Parallel STORE
        I->>N1: STORE(key, value, ttl)
        I->>N2: STORE(key, value, ttl)
        I->>NK: STORE(key, value, ttl)
    end
    
    N1-->>I: STORE_ACK(success)
    N2-->>I: STORE_ACK(success)
    NK-->>I: STORE_ACK(success)
    
    Note over I: Return count of successful stores
```

### 2.6 Republication

Механизм поддержания данных в сети:

1. **Original Publisher Republication:**
   - Каждые `REPUBLISH_INTERVAL` (1 час) владелец данных выполняет STORE
   - Обновляет `timestamp` и `last_republish`

2. **Replica Republication:**
   - Узлы, хранящие реплику, также републикуют
   - Предотвращает потерю при уходе original publisher

3. **Expiration:**
   - Записи с `timestamp + ttl < now` удаляются при `cleanup()`
   - `CLEANUP_INTERVAL = 300s` (5 минут)

---

## 3. Storage Layer

### 3.1 DHTStorage Schema

SQLite таблица `dht_store`:

```sql
CREATE TABLE dht_store (
    key BLOB PRIMARY KEY,          -- 20 bytes (SHA-1)
    value BLOB NOT NULL,           -- up to 64KB
    publisher_id BLOB NOT NULL,    -- 20 bytes
    timestamp REAL NOT NULL,       -- UNIX timestamp
    ttl INTEGER NOT NULL,          -- seconds
    last_republish REAL            -- UNIX timestamp
);

CREATE INDEX idx_dht_expires ON dht_store(timestamp, ttl);
CREATE INDEX idx_dht_republish ON dht_store(last_republish);
```

### 3.2 Ограничения

| Параметр | Значение |
|----------|----------|
| MAX_VALUE_SIZE | 65536 bytes (64 KB) |
| Key length | 20 bytes (SHA-1) |
| DEFAULT_TTL | 86400 seconds (24 hours) |

### 3.3 Key Derivation

```python
def string_to_key(s: str) -> bytes:
    """Convert string to 20-byte DHT key."""
    return hashlib.sha1(s.encode('utf-8')).digest()

def bytes_to_key(data: bytes) -> bytes:
    """Convert bytes to 20-byte DHT key."""
    return hashlib.sha1(data).digest()
```

**Примеры ключей:**
- Сервис: `service:vpn_exit` → `sha1("service:vpn_exit")`
- Кэш: `cache:<sha256_hash>` → `sha1("cache:" + hash)`
- Модель: `model:<model_id>` → `sha1("model:" + id)`

---

## 4. Message Handlers

### 4.1 Protocol Router

Маршрутизация входящих сообщений:

```python
class ProtocolRouter:
    handlers: Dict[MessageType, MessageHandler]
    
    async def route(self, message: Message, context: Dict) -> Optional[Message]:
        handler = self.handlers.get(message.type)
        if not handler:
            return None
        return await handler.handle(message, crypto, context)
```

### 4.2 Ping-Pong Handler

**Защита от replay-атак:**
```python
MAX_PING_AGE = 60.0  # секунд

async def handle_ping(message: Message) -> Optional[Message]:
    # 1. Verify signature
    if not crypto.verify_signature(message):
        return None  # Silent drop
    
    # 2. Check timestamp
    age = time.time() - message.timestamp
    if age > MAX_PING_AGE:
        return None  # Possible replay
    
    # 3. Create signed PONG
    pong = Message(
        type=MessageType.PONG,
        payload={
            "original_nonce": message.nonce,
            "ping_timestamp": message.timestamp,
        },
        sender_id=crypto.node_id,
    )
    return crypto.sign_message(pong)
```

### 4.3 Service Request Handler

Обработка запросов услуг с экономикой:

```python
async def handle_service_request(message: Message) -> Message:
    # 1. Verify signature
    if not crypto.verify_signature(message):
        return error_response("Invalid signature")
    
    # 2. Extract request
    service_name = message.payload["service_name"]
    budget = message.payload["budget"]
    
    # 3. Execute via AgentManager
    response = await agent_manager.handle_request(
        ServiceRequest(
            service_name=service_name,
            payload=message.payload["payload"],
            requester_id=message.sender_id,
            budget=budget,
        )
    )
    
    # 4. Cost recorded in Ledger automatically
    return Message(
        type=MessageType.SERVICE_RESPONSE,
        payload=response.to_dict(),
        sender_id=crypto.node_id,
    )
```

---

## 5. Traffic Masking

### 5.1 HTTP Masking

Для обхода DPI трафик маскируется под HTTP:

```python
HTTP_REQUEST_TEMPLATE = """
POST /api/v1/sync HTTP/1.1
Host: {host}
Content-Type: application/json
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
Content-Length: {length}
Connection: keep-alive

{payload}
"""

def mask_as_http(message: Message) -> bytes:
    payload = message.to_json()
    encoded = base64.b64encode(payload.encode()).decode()
    body = json.dumps({"data": encoded, "v": "1.0"})
    return HTTP_REQUEST_TEMPLATE.format(
        host="api.example.com",
        length=len(body),
        payload=body,
    ).encode()
```

### 5.2 Детектирование

```python
def is_http_masked(data: bytes) -> bool:
    try:
        text = data.decode('utf-8')
        return (
            text.startswith("POST /api/") or 
            text.startswith("HTTP/1.")
        )
    except UnicodeDecodeError:
        return False
```

---

## Диаграмма: Полный Stack

```mermaid
flowchart TB
    subgraph Application["Application Layer"]
        VPN[VPN Agent]
        CDN[CDN/Amplifier]
        AI[Cortex AI]
    end
    
    subgraph Service["Service Layer"]
        AM[Agent Manager]
        PR[Protocol Router]
    end
    
    subgraph DHT["DHT Layer"]
        RT[Routing Table]
        ST[DHT Storage]
        KP[Kademlia Protocol]
    end
    
    subgraph Transport["Transport Layer"]
        CR[Crypto]
        TM[Traffic Masker]
        SP[SimpleTransport]
    end
    
    subgraph Network["Network"]
        TCP[TCP Socket]
        UDP[UDP Broadcast]
    end
    
    VPN --> AM
    CDN --> AM
    AI --> AM
    AM --> PR
    PR --> KP
    KP --> RT
    KP --> ST
    PR --> CR
    CR --> TM
    TM --> SP
    SP --> TCP
    KP --> UDP
```

---

## Ссылки

1. Maymounkov, P., Mazières, D. — Kademlia: A Peer-to-peer Information System Based on the XOR Metric (IPTPS 2002)
2. Bernstein, D.J. — The Salsa20 family of stream ciphers
3. Bernstein, D.J. et al. — Ed25519: high-speed high-security signatures
4. RFC 7748 — Elliptic Curves for Security (X25519)
