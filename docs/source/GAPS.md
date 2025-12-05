# ZEONE: Gaps & Future Work

Этот документ содержит известные недоработки, технический долг и предложения по улучшению системы ZEONE. Каждый пункт требует детальной проработки перед внедрением.

---

## 1. Wire Protocol

### 1.1 [HIGH] JSON Overhead

**Текущее состояние:**
- `SimpleTransport` использует 4-byte length prefix + JSON payload
- JSON encoding добавляет ~40% overhead на служебные данные
- Base64 encoding бинарных данных удваивает размер

**Проблема:**
- Неэффективно для высоконагруженных сценариев (VPN, CDN)
- Увеличивает latency при парсинге

**Предложение:**
Внедрить бинарный Wire Protocol v1 (описан в `02_architecture.md`):
```
[Magic 2B][Version 1B][Type 1B][Length 4B][Nonce 24B][Signature 64B][Payload]
```

**Сложность:** Medium  
**Приоритет:** High  
**Зависимости:** Backwards compatibility layer для миграции

---

### 1.2 [MEDIUM] Version Negotiation

**Текущее состояние:**
- Версия протокола не передаётся при handshake
- Нет механизма graceful upgrade

**Проблема:**
- Невозможно обновить протокол без разрыва совместимости
- Нет feature negotiation между узлами разных версий

**Предложение:**
```
HELLO message:
{
  "supported_versions": [1, 2],
  "preferred_version": 2,
  "capabilities": ["DHT", "VPN", "CORTEX"]
}
```

**Сложность:** Low  
**Приоритет:** Medium

---

### 1.3 [LOW] Magic Bytes Detection

**Текущее состояние:**
- Нет magic bytes для идентификации ZEONE трафика

**Проблема:**
- Сложно отличить ZEONE поток от случайных данных
- Нет раннего отклонения non-ZEONE соединений

**Предложение:**
Magic bytes: `0x5A45` ("ZE" в ASCII)

**Сложность:** Low  
**Приоритет:** Low

---

## 2. Security

### 2.1 [CRITICAL] Merkle Tree for Chunk Verification

**Текущее состояние:**
- P2P Model Distribution использует per-file SHA-256 hash
- Верификация возможна только после полной загрузки файла

**Проблема:**
- Malicious peer может отдавать corrupted chunks
- Обнаружение возможно только после загрузки всего файла (GBs)
- Нельзя определить, какой именно chunk corrupted

**Предложение:**
Merkle Tree для chunk-level verification:
```
                 Root Hash
                /         \
           H(0-1)          H(2-3)
           /    \          /    \
        H(c0)  H(c1)    H(c2)  H(c3)
          |      |        |      |
       Chunk0 Chunk1  Chunk2 Chunk3
```

Manifest structure:
```json
{
  "merkle_root": "sha256:...",
  "chunk_hashes": ["sha256:c0", "sha256:c1", ...],
  "chunk_size": 1048576
}
```

Verification flow:
1. Получить Merkle proof для chunk
2. Верифицировать chunk hash
3. Верифицировать path к root
4. Отклонить chunk если не проходит

**Сложность:** High  
**Приоритет:** Critical  
**Зависимости:** Изменение manifest format, P2PLoader refactoring

---

### 2.2 [HIGH] Trust Score Stake Weight

**Текущее состояние:**
- Trust Score не учитывает economic stake
- Sybil-атака дешёвая (только время на накопление reputation)

**Проблема:**
- Атакующий может создать много узлов с низким stake
- Trust Score накапливается без экономического залога

**Предложение:**
Weighted Trust Score:
```
T_effective = T_behavior × sqrt(stake / base_stake)
```

Где:
- `T_behavior` — текущий Trust Score из поведения
- `stake` — ZEO tokens застейкано узлом
- `base_stake` — минимальный stake (например, 100 ZEO)

**Сложность:** Medium  
**Приоритет:** High

---

### 2.3 [MEDIUM] Formal Cryptography Verification

**Текущее состояние:**
- Криптография реализована через PyNaCl (wrapper над libsodium)
- Нет формальной верификации протокола

**Проблема:**
- Protocol-level vulnerabilities могут существовать
- Composition of primitives не доказана безопасной

**Предложение:**
1. Формальная модель протокола в ProVerif/Tamarin
2. Security proof для handshake protocol
3. Third-party security audit

**Сложность:** Very High  
**Приоритет:** Medium (для production deployment)

---

### 2.4 [MEDIUM] Replay Attack Window

**Текущее состояние:**
- `MAX_PING_AGE = 60 seconds`
- Nonce stored in memory (lost on restart)

**Проблема:**
- После restart узел забывает seen nonces
- Window of 60s для replay attack

**Предложение:**
1. Persist nonce bloom filter to disk
2. Reduce MAX_PING_AGE to 30s
3. Add sequence numbers per-session

**Сложность:** Low  
**Приоритет:** Medium

---

## 3. Economics

### 3.1 [HIGH] Token Emission Curve

**Текущее состояние:**
- Эмиссия не определена в коде
- `PROPOSED` logarithmic curve в whitepaper

**Проблема:**
- Нет работающего механизма эмиссии
- Токен ZEO существует только концептуально

**Предложение:**
1. Развернуть ERC-20 контракт с fixed supply
2. Или: Inflationary model с governance control
3. Определить vesting schedule для initial distribution

**Сложность:** Medium (contract deployment)  
**Приоритет:** High (для запуска экономики)

---

### 3.2 [HIGH] Settlement Smart Contract

**Текущее состояние:**
- SettlementManager использует простой ERC-20 transfer
- Нет on-chain dispute resolution

**Проблема:**
- При спорах нет арбитража
- Peer может отрицать получение settlement

**Предложение:**
Escrow-based settlement contract:
```solidity
contract ZEONESettlement {
    struct Settlement {
        address payer;
        address payee;
        uint256 amount;
        bytes32 proofHash;
        uint256 deadline;
        bool claimed;
    }
    
    mapping(bytes32 => Settlement) public settlements;
    
    function initiateSettlement(
        address payee,
        uint256 amount,
        bytes32 proofHash
    ) external;
    
    function claimSettlement(
        bytes32 settlementId,
        bytes calldata proof
    ) external;
    
    function disputeSettlement(
        bytes32 settlementId,
        bytes calldata evidence
    ) external;
}
```

**Сложность:** High  
**Приоритет:** High

---

### 3.3 [MEDIUM] DAO Governance

**Текущее состояние:**
- Governance описан в whitepaper как `PROPOSED`
- Нет реализации on-chain voting

**Проблема:**
- Параметры сети (fees, limits) не могут быть изменены decentralized
- Нет механизма upgrade protocol

**Предложение:**
1. Governor contract (OpenZeppelin Governor)
2. Timelock для execution delay
3. Voting weight = sqrt(stake) × (1 + trust)

**Сложность:** High  
**Приоритет:** Medium

---

### 3.4 [LOW] Credit Rate Dynamics

**Текущее состояние:**
- `credit_rate = 0.001` tokens per byte (hardcoded)
- Не адаптируется к market conditions

**Проблема:**
- При изменении цены токена rates становятся нерелевантными
- Нет price discovery механизма

**Предложение:**
Oracle-based dynamic pricing или auction mechanism

**Сложность:** High  
**Приоритет:** Low

---

## 4. Cortex

### 4.1 [HIGH] Distributed Inference

**Текущее состояние:**
- Все AI inference выполняется на single node
- Нет model parallelism или pipeline parallelism

**Проблема:**
- Большие модели (7B+ params) не помещаются на consumer GPU
- Нет horizontal scaling

**Предложение:**
1. Model sharding: split layers across nodes
2. Pipeline parallelism: каждый node обрабатывает subset of layers
3. Consensus для aggregation of results

**Сложность:** Very High  
**Приоритет:** High (для enterprise use cases)

---

### 4.2 [MEDIUM] Model Verification

**Текущее состояние:**
- Модели верифицируются только по SHA-256 hash
- Нет проверки что модель делает то, что заявлено

**Проблема:**
- Malicious model может быть распространена
- Backdoor в weights не детектируется

**Предложение:**
1. Model card с benchmark results
2. Community voting на model trustworthiness
3. Sandboxed inference с output validation

**Сложность:** High  
**Приоритет:** Medium

---

### 4.3 [MEDIUM] Embedding Model Upgrades

**Текущее состояние:**
- Hardcoded `all-MiniLM-L6-v2` для embeddings
- Vector store не версионирован

**Проблема:**
- При смене embedding model нужно reindex все данные
- Нет migration path

**Предложение:**
1. Store embedding model version in metadata
2. Lazy reindexing при query time
3. Support multiple embedding spaces

**Сложность:** Medium  
**Приоритет:** Medium

---

### 4.4 [LOW] Streaming Inference

**Текущее состояние:**
- LLM inference возвращает полный ответ
- Нет token streaming

**Проблема:**
- Плохой UX для long generations
- High latency to first token

**Предложение:**
WebSocket-based streaming с token-by-token delivery

**Сложность:** Low  
**Приоритет:** Low

---

## 5. DHT

### 5.1 [MEDIUM] NAT Traversal

**Текущее состояние:**
- Basic hole punching через STUN
- Нет full ICE implementation

**Проблема:**
- ~30% узлов за symmetric NAT недостижимы
- Нужны relay nodes

**Предложение:**
1. Full ICE implementation (RFC 8445)
2. TURN relay fallback
3. Incentivized relay nodes (paid service)

**Сложность:** High  
**Приоритет:** Medium

---

### 5.2 [LOW] S/Kademlia Extensions

**Текущее состояние:**
- Standard Kademlia без S/Kademlia security extensions

**Проблема:**
- Vulnerable к некоторым Sybil variants
- Routing table pollution attacks

**Предложение:**
Implement S/Kademlia:
1. Parallel disjoint lookups
2. Sibling broadcast for replication
3. Crypto puzzles для node ID generation

**Сложность:** High  
**Приоритет:** Low

---

## 6. Operations

### 6.1 [MEDIUM] Monitoring & Observability

**Текущее состояние:**
- Basic logging
- No metrics export
- No distributed tracing

**Проблема:**
- Сложно debug distributed issues
- Нет visibility в network health

**Предложение:**
1. Prometheus metrics endpoint
2. OpenTelemetry tracing
3. Grafana dashboards

**Сложность:** Medium  
**Приоритет:** Medium

---

### 6.2 [LOW] Configuration Management

**Текущее состояние:**
- Python dataclasses в `config.py`
- Env vars для secrets

**Проблема:**
- Нет hot-reload конфигурации
- Нет distributed config (Consul/etcd)

**Предложение:**
1. YAML/TOML config files
2. Watch for file changes
3. DHT-based config propagation

**Сложность:** Low  
**Приоритет:** Low

---

## Priority Matrix

| Priority | Count | Items |
|----------|-------|-------|
| Critical | 1 | Merkle Tree verification |
| High | 6 | JSON overhead, Stake weight, Emission, Settlement contract, Distributed inference, NAT |
| Medium | 8 | Version negotiation, Formal verification, Replay window, DAO, Model verification, Embeddings, Monitoring |
| Low | 5 | Magic bytes, Credit dynamics, Streaming, S/Kademlia, Config management |

---

## Contributing

Для работы над любым из пунктов:
1. Создать issue в tracker
2. Обсудить approach в Discord
3. Написать RFC document
4. Implement в feature branch
5. Security review для Critical/High items

