"""
Kademlia DHT Module
===================

Реализация распределённой хеш-таблицы на основе Kademlia:
- RoutingTable: K-bucket таблица маршрутизации
- DHTProtocol: FIND_NODE, FIND_VALUE, STORE операции
- DHTStorage: Локальное хранилище пар key-value
- KademliaNode: Обёртка над базовым Node с DHT функциональностью

[KADEMLIA] Ключевые принципы:
- XOR-метрика для измерения расстояния между узлами
- 160 k-buckets (по битам расстояния)
- Итеративный lookup с alpha параллельными запросами
- Republish для поддержания данных в сети
"""

from .routing import (
    RoutingTable,
    KBucket,
    NodeInfo,
    xor_distance,
    distance_to_bucket_index,
)

from .storage import DHTStorage

from .protocol import (
    DHTProtocol,
    FindNodeHandler,
    FindValueHandler,
    StoreHandler,
)

__all__ = [
    # Routing
    "RoutingTable",
    "KBucket",
    "NodeInfo",
    "xor_distance",
    "distance_to_bucket_index",
    # Storage
    "DHTStorage",
    # Protocol
    "DHTProtocol",
    "FindNodeHandler",
    "FindValueHandler",
    "StoreHandler",
]

