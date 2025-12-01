"""
Core P2P Network Module (Production-Ready)
==========================================
Содержит основные компоненты сетевого слоя:
- Node: главный класс узла с учетом трафика и услугами
- Transport: шифрование и маскировка трафика
- Protocol: типы сообщений и обработчики
- DHT: Kademlia распределённая хеш-таблица
- Discovery: оптимизированный механизм обнаружения узлов
- Persistence: сохранение и восстановление состояния
- Security: rate limiting и DoS protection
- Monitoring: health checks и metrics
"""

from .node import Node, Peer, PeerManager
from .transport import Message, MessageType, Crypto, SimpleTransport, BlockingTransport
from .protocol import (
    PingPongHandler,
    MessageHandler,
    ServiceRequestHandler,
    ServiceResponseHandler,
    ProtocolRouter,
)
from .discovery import OptimizedDiscovery, BloomFilter, PeerRecord, GossipMessage

# DHT imports (may fail if not all dependencies available)
try:
    from .dht import (
        RoutingTable,
        KBucket,
        NodeInfo,
        DHTStorage,
        DHTProtocol,
    )
    from .dht.node import KademliaNode
    _DHT_AVAILABLE = True
except ImportError:
    _DHT_AVAILABLE = False
    RoutingTable = None  # type: ignore
    KBucket = None  # type: ignore
    NodeInfo = None  # type: ignore
    DHTStorage = None  # type: ignore
    DHTProtocol = None  # type: ignore
    KademliaNode = None  # type: ignore

# Persistence imports
try:
    from .persistence import (
        StateManager,
        NodeState,
        PeerRecord as PersistentPeerRecord,
        PeerStore,
        PeerInfo,
    )
    _PERSISTENCE_AVAILABLE = True
except ImportError:
    _PERSISTENCE_AVAILABLE = False
    StateManager = None  # type: ignore
    NodeState = None  # type: ignore
    PersistentPeerRecord = None  # type: ignore
    PeerStore = None  # type: ignore
    PeerInfo = None  # type: ignore

# Security imports
try:
    from .security import (
        RateLimiter,
        RateLimitRule,
        RateLimitResult,
        DoSProtector,
        ThreatLevel,
        AttackType,
    )
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False
    RateLimiter = None  # type: ignore
    RateLimitRule = None  # type: ignore
    RateLimitResult = None  # type: ignore
    DoSProtector = None  # type: ignore
    ThreatLevel = None  # type: ignore
    AttackType = None  # type: ignore

# Monitoring imports
try:
    from .monitoring import (
        HealthChecker,
        HealthStatus,
        ComponentHealth,
        MetricsCollector,
        Counter,
        Gauge,
        Histogram,
    )
    from .monitoring.metrics import get_metrics
    _MONITORING_AVAILABLE = True
except ImportError:
    _MONITORING_AVAILABLE = False
    HealthChecker = None  # type: ignore
    HealthStatus = None  # type: ignore
    ComponentHealth = None  # type: ignore
    MetricsCollector = None  # type: ignore
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    Histogram = None  # type: ignore
    get_metrics = None  # type: ignore

__all__ = [
    # Node
    "Node",
    "Peer",
    "PeerManager",
    # Transport
    "Message",
    "MessageType", 
    "Crypto",
    "SimpleTransport",
    "BlockingTransport",
    # Protocol
    "PingPongHandler",
    "MessageHandler",
    "ServiceRequestHandler",
    "ServiceResponseHandler",
    "ProtocolRouter",
    # Discovery
    "OptimizedDiscovery",
    "BloomFilter",
    "PeerRecord",
    "GossipMessage",
    # DHT
    "RoutingTable",
    "KBucket",
    "NodeInfo",
    "DHTStorage",
    "DHTProtocol",
    "KademliaNode",
    # Persistence
    "StateManager",
    "NodeState",
    "PersistentPeerRecord",
    "PeerStore",
    "PeerInfo",
    # Security
    "RateLimiter",
    "RateLimitRule",
    "RateLimitResult",
    "DoSProtector",
    "ThreatLevel",
    "AttackType",
    # Monitoring
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "get_metrics",
]

