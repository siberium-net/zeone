"""
Core P2P Network Module (Layer 3: Market)
=========================================
Содержит основные компоненты сетевого слоя:
- Node: главный класс узла с учетом трафика и услугами
- Transport: шифрование и маскировка трафика
- Protocol: типы сообщений и обработчики
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

__all__ = [
    "Node",
    "Peer",
    "PeerManager",
    "Message",
    "MessageType", 
    "Crypto",
    "SimpleTransport",
    "BlockingTransport",
    "PingPongHandler",
    "MessageHandler",
    "ServiceRequestHandler",
    "ServiceResponseHandler",
    "ProtocolRouter",
]

