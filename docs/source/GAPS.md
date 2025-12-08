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

**[UPDATED]** После Epoch 1 & 3 многие пункты реализованы и перемещены в архив (см. раздел "Implemented Features Archive" ниже).

| Priority | Count | Items | Status |
|----------|-------|-------|--------|
| ~~Critical~~ | ~~1~~ | ~~Merkle Tree verification~~ | [OK] IMPLEMENTED |
| High | 2 | Emission, NAT | Pending |
| ~~High~~ | ~~4~~ | ~~JSON overhead, Stake weight, Settlement contract, Distributed inference~~ | [OK] IMPLEMENTED |
| Medium | 8 | Version negotiation, Formal verification, Replay window, DAO, Model verification, Embeddings, Monitoring | Pending |
| Low | 4 | Credit dynamics, Streaming, S/Kademlia, Config management | Pending |
| ~~Low~~ | ~~1~~ | ~~Magic bytes~~ | [OK] IMPLEMENTED |

**Итого:**
- [OK] Implemented: 6 items (Critical: 1, High: 4, Low: 1)
- Pending: 14 items (High: 2, Medium: 8, Low: 4)

---

## Implemented Features Archive

Этот раздел содержит функции, которые ранее были в GAPS, но теперь реализованы и перемещены в основную документацию.

### [OK] 1.1 Binary Wire Protocol (JSON Overhead)

**Статус:** IMPLEMENTED  
**Код:** [`core/wire.py`](../../core/wire.py)  
**Документация:** [`docs/source/ru/02_architecture.md`](ru/02_architecture.md) раздел 1.2

Binary Wire Protocol v1 полностью реализован:
- Magic bytes `b'ZE'` (0x5A45)
- Fixed 98-byte header
- XSalsa20-Poly1305 encryption
- Ed25519 signatures
- Hard Fork: нет обратной совместимости с JSON

**Улучшения:**
- ~40% reduction в размере сообщений
- Version negotiation через header
- Immediate invalid protocol detection

---

### [OK] 1.3 Magic Bytes Detection

**Статус:** IMPLEMENTED  
**Код:** [`core/wire.py:53`](../../core/wire.py) - `MAGIC = b'ZE'`

Magic bytes 0x5A45 реализованы в Binary Wire Protocol. Любые данные без Magic → немедленное закрытие сокета.

---

### [OK] 2.1 Merkle Tree for Chunk Verification

**Статус:** IMPLEMENTED  
**Код:** [`core/security/merkle.py`](../../core/security/merkle.py)  
**Документация:** [`docs/source/ru/03_cortex.md`](ru/03_cortex.md) раздел 4.4.1

Полная реализация Merkle Tree для chunk-level verification:
- Build tree from chunk hashes
- Generate inclusion proofs
- Verify individual chunks без полной загрузки файла
- O(log N) proof size
- Integration с Trust Score (slashing на invalid proof)

**Защита:**
- Instant corrupted chunk detection
- Malicious peer identification
- MITM attack prevention на chunk level

---

### [OK] 2.2 Weighted Trust Score (Stake Weight)

**Статус:** IMPLEMENTED  
**Код:** [`economy/trust.py:176-204`](../../economy/trust.py)  
**Документация:** [`docs/source/ru/01_whitepaper.md`](ru/01_whitepaper.md) раздел 4.1

Weighted Trust Score реализован:

```python
T_effective = T_behavior * log10(1 + Stake / BaseStake)
```

**Компоненты:**
- `WeightedTrustScore` класс
- EMA для behavior score
- Stake weighting через log10
- Dust limit protection (< 10 ZEO)
- Slashing механизм для критических нарушений

**Защита от Sybil:**
- Economic barrier через stake requirement
- Квадратный корень для предотвращения плутократии
- Blacklist для slashed peers

---

### [OK] 3.2 Settlement Smart Contract (Native SIBR)

**Статус:** IMPLEMENTED на Siberium  
**Код:** [`economy/chain.py`](../../economy/chain.py) - `SiberiumManager`  
**Документация:** [`docs/source/ru/04_economy.md`](ru/04_economy.md) раздел 6

**ВАЖНО:** Реализовано как Native SIBR token (не ERC-20!):

- `ZEOSettlement.sol` контракт
- Native staking через payable deposit()
- Claim mechanism с EIP-712 signatures
- 7-day unstaking timelock
- Fee accumulation and distribution

**Сети:**
- Mainnet: ChainID 111111, https://rpc.siberium.net
- Testnet: ChainID 111000, https://rpc.test.siberium.net

---

### [OK] 4.1 Distributed Inference (Pipeline Parallelism)

**Статус:** IMPLEMENTED  
**Код:**
- [`cortex/distributed/pipeline.py`](../../cortex/distributed/pipeline.py) - PipelineWorker, Coordinator
- [`agents/neuro_link.py`](../../agents/neuro_link.py) - NeuroLinkAgent

**Документация:** [`docs/source/ru/03_cortex.md`](ru/03_cortex.md) раздел 7

Naive Pipeline Parallelism реализован:

```
HEAD Node → MIDDLE Node(s) → TAIL Node
```

**Компоненты:**
- Model sharding по layers
- Tensor transport через NeuroLink
- Binary Wire Protocol для TENSOR_DATA messages
- Chunked transfer для больших tensors
- Async pipeline с micro-batching

**Преимущества:**
- Horizontal scaling для LLM inference
- Consumer GPU → Enterprise-scale
- 7B model: 14GB → split 3 nodes → ~5GB each

---

## Contributing

Для работы над любым из пунктов:
1. Создать issue в tracker
2. Обсудить approach в Discord
3. Написать RFC document
4. Implement в feature branch
5. Security review для Critical/High items

