"""
Core P2P Network Module (Layer 3: Market + DHT)
===============================================
Содержит основные компоненты сетевого слоя:
- Node: главный класс узла с учетом трафика и услугами
- Transport: шифрование и маскировка трафика
- Protocol: типы сообщений и обработчики
- DHT: Kademlia распределённая хеш-таблица
- Discovery: оптимизированный механизм обнаружения узлов
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
]

