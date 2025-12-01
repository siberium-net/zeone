"""
Kademlia Protocol - RPC операции DHT
====================================

[KADEMLIA] Основные RPC операции:
- FIND_NODE: Найти k ближайших узлов к target_id
- FIND_VALUE: Найти значение по ключу (или k узлов если нет)
- STORE: Сохранить пару key-value на узле

[LOOKUP] Итеративный поиск:
- alpha = 3 параллельных запроса
- Продолжаем пока находим более близкие узлы
- Возвращаем k ближайших найденных узлов
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Set, TYPE_CHECKING
from enum import Enum, auto

from .routing import (
    RoutingTable, NodeInfo, K, ALPHA,
    xor_distance, key_to_id,
)
from .storage import DHTStorage, StoredValue, DEFAULT_TTL

if TYPE_CHECKING:
    from core.transport import Message, Crypto

logger = logging.getLogger(__name__)


# Новые типы сообщений для DHT
class DHTMessageType(Enum):
    """Типы DHT сообщений."""
    FIND_NODE = auto()
    FIND_NODE_RESPONSE = auto()
    FIND_VALUE = auto()
    FIND_VALUE_RESPONSE = auto()
    STORE = auto()
    STORE_RESPONSE = auto()


@dataclass
class FindNodeRequest:
    """
    Запрос FIND_NODE.
    
    [KADEMLIA] Найти k ближайших узлов к target_id.
    """
    target_id: bytes  # 20 байт - ID для поиска
    sender_id: bytes  # 20 байт - ID отправителя
    sender_host: str
    sender_port: int
    
    def to_dict(self) -> Dict:
        return {
            "type": "FIND_NODE",
            "target_id": self.target_id.hex(),
            "sender_id": self.sender_id.hex(),
            "sender_host": self.sender_host,
            "sender_port": self.sender_port,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FindNodeRequest":
        return cls(
            target_id=bytes.fromhex(data["target_id"]),
            sender_id=bytes.fromhex(data["sender_id"]),
            sender_host=data["sender_host"],
            sender_port=data["sender_port"],
        )


@dataclass
class FindNodeResponse:
    """
    Ответ на FIND_NODE.
    
    [KADEMLIA] Содержит k ближайших узлов к target_id.
    """
    nodes: List[NodeInfo]
    
    def to_dict(self) -> Dict:
        return {
            "type": "FIND_NODE_RESPONSE",
            "nodes": [n.to_dict() for n in self.nodes],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FindNodeResponse":
        nodes = [NodeInfo.from_dict(n) for n in data.get("nodes", [])]
        return cls(nodes=nodes)


@dataclass
class FindValueRequest:
    """
    Запрос FIND_VALUE.
    
    [KADEMLIA] Найти значение по ключу или k ближайших узлов.
    """
    key: bytes  # 20 байт - ключ для поиска
    sender_id: bytes
    sender_host: str
    sender_port: int
    
    def to_dict(self) -> Dict:
        return {
            "type": "FIND_VALUE",
            "key": self.key.hex(),
            "sender_id": self.sender_id.hex(),
            "sender_host": self.sender_host,
            "sender_port": self.sender_port,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FindValueRequest":
        return cls(
            key=bytes.fromhex(data["key"]),
            sender_id=bytes.fromhex(data["sender_id"]),
            sender_host=data["sender_host"],
            sender_port=data["sender_port"],
        )


@dataclass
class FindValueResponse:
    """
    Ответ на FIND_VALUE.
    
    [KADEMLIA] Содержит либо значение, либо k ближайших узлов.
    """
    value: Optional[bytes] = None  # Найденное значение
    nodes: List[NodeInfo] = field(default_factory=list)  # Или ближайшие узлы
    
    @property
    def found(self) -> bool:
        return self.value is not None
    
    def to_dict(self) -> Dict:
        result = {"type": "FIND_VALUE_RESPONSE"}
        if self.value is not None:
            result["value"] = self.value.hex()
        else:
            result["nodes"] = [n.to_dict() for n in self.nodes]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FindValueResponse":
        value = None
        if "value" in data:
            value = bytes.fromhex(data["value"])
        
        nodes = []
        if "nodes" in data:
            nodes = [NodeInfo.from_dict(n) for n in data["nodes"]]
        
        return cls(value=value, nodes=nodes)


@dataclass
class StoreRequest:
    """
    Запрос STORE.
    
    [KADEMLIA] Сохранить пару key-value на узле.
    """
    key: bytes  # 20 байт
    value: bytes
    ttl: int = DEFAULT_TTL
    sender_id: bytes = b""
    sender_host: str = ""
    sender_port: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "type": "STORE",
            "key": self.key.hex(),
            "value": self.value.hex(),
            "ttl": self.ttl,
            "sender_id": self.sender_id.hex(),
            "sender_host": self.sender_host,
            "sender_port": self.sender_port,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoreRequest":
        return cls(
            key=bytes.fromhex(data["key"]),
            value=bytes.fromhex(data["value"]),
            ttl=data.get("ttl", DEFAULT_TTL),
            sender_id=bytes.fromhex(data.get("sender_id", "0" * 40)),
            sender_host=data.get("sender_host", ""),
            sender_port=data.get("sender_port", 0),
        )


@dataclass
class StoreResponse:
    """
    Ответ на STORE.
    
    [KADEMLIA] Подтверждение сохранения.
    """
    success: bool
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "type": "STORE_RESPONSE",
            "success": self.success,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoreResponse":
        return cls(
            success=data.get("success", False),
            error=data.get("error", ""),
        )


class DHTProtocol:
    """
    Kademlia DHT Protocol - обработка RPC запросов.
    
    [KADEMLIA] Обрабатывает:
    - FIND_NODE: возвращает k ближайших узлов
    - FIND_VALUE: возвращает значение или узлы
    - STORE: сохраняет пару key-value
    
    [LOOKUP] Итеративный поиск узлов/значений.
    """
    
    def __init__(
        self,
        routing_table: RoutingTable,
        storage: DHTStorage,
        local_id: bytes,
        local_host: str,
        local_port: int,
    ):
        """
        Args:
            routing_table: Таблица маршрутизации
            storage: Локальное хранилище DHT
            local_id: Наш node_id (20 байт)
            local_host: Наш хост
            local_port: Наш порт
        """
        self.routing_table = routing_table
        self.storage = storage
        self.local_id = local_id
        self.local_host = local_host
        self.local_port = local_port
        
        # Pending requests для итеративного lookup
        self._pending_lookups: Dict[bytes, asyncio.Future] = {}
    
    # =========================================================================
    # Request Handlers
    # =========================================================================
    
    async def handle_find_node(self, request: FindNodeRequest) -> FindNodeResponse:
        """
        Обработать FIND_NODE запрос.
        
        [KADEMLIA] Возвращает k ближайших узлов к target_id.
        """
        # Добавляем отправителя в routing table
        sender_node = NodeInfo(
            node_id=request.sender_id,
            host=request.sender_host,
            port=request.sender_port,
        )
        self.routing_table.add_node(sender_node)
        
        # Находим ближайшие узлы
        closest = self.routing_table.find_closest(
            request.target_id,
            count=K,
            exclude=request.sender_id,
        )
        
        logger.debug(
            f"[DHT] FIND_NODE: target={request.target_id.hex()[:16]}..., "
            f"returning {len(closest)} nodes"
        )
        
        return FindNodeResponse(nodes=closest)
    
    async def handle_find_value(self, request: FindValueRequest) -> FindValueResponse:
        """
        Обработать FIND_VALUE запрос.
        
        [KADEMLIA] Если есть значение - возвращаем его.
        Иначе возвращаем k ближайших узлов к ключу.
        """
        # Добавляем отправителя в routing table
        sender_node = NodeInfo(
            node_id=request.sender_id,
            host=request.sender_host,
            port=request.sender_port,
        )
        self.routing_table.add_node(sender_node)
        
        # Ищем значение локально
        stored = await self.storage.get(request.key)
        
        if stored:
            logger.debug(f"[DHT] FIND_VALUE: key={request.key.hex()[:16]}... FOUND")
            return FindValueResponse(value=stored.value)
        
        # Значения нет - возвращаем ближайшие узлы
        closest = self.routing_table.find_closest(
            request.key,
            count=K,
            exclude=request.sender_id,
        )
        
        logger.debug(
            f"[DHT] FIND_VALUE: key={request.key.hex()[:16]}... "
            f"NOT FOUND, returning {len(closest)} nodes"
        )
        
        return FindValueResponse(nodes=closest)
    
    async def handle_store(self, request: StoreRequest) -> StoreResponse:
        """
        Обработать STORE запрос.
        
        [KADEMLIA] Сохраняет пару key-value локально.
        """
        # Добавляем отправителя в routing table
        if request.sender_id and len(request.sender_id) == 20:
            sender_node = NodeInfo(
                node_id=request.sender_id,
                host=request.sender_host,
                port=request.sender_port,
            )
            self.routing_table.add_node(sender_node)
        
        # Сохраняем значение
        success = await self.storage.store(
            key=request.key,
            value=request.value,
            publisher_id=request.sender_id or self.local_id,
            ttl=request.ttl,
        )
        
        if success:
            logger.debug(
                f"[DHT] STORE: key={request.key.hex()[:16]}... "
                f"value={len(request.value)} bytes OK"
            )
            return StoreResponse(success=True)
        else:
            return StoreResponse(success=False, error="Storage failed")
    
    def handle_message(self, payload: Dict) -> Optional[Dict]:
        """
        Обработать входящее DHT сообщение.
        
        Args:
            payload: Словарь с данными сообщения
        
        Returns:
            Словарь с ответом или None
        """
        msg_type = payload.get("type")
        
        if msg_type == "FIND_NODE":
            request = FindNodeRequest.from_dict(payload)
            response = asyncio.create_task(self.handle_find_node(request))
            # Note: В реальном коде нужен await, но здесь синхронный интерфейс
            return None
        
        elif msg_type == "FIND_VALUE":
            request = FindValueRequest.from_dict(payload)
            # Аналогично
            return None
        
        elif msg_type == "STORE":
            request = StoreRequest.from_dict(payload)
            return None
        
        return None
    
    # =========================================================================
    # Iterative Lookup
    # =========================================================================
    
    async def iterative_find_node(
        self,
        target_id: bytes,
        send_func,
    ) -> List[NodeInfo]:
        """
        Итеративный поиск k ближайших узлов к target_id.
        
        [KADEMLIA] Алгоритм:
        1. Начинаем с alpha ближайших известных узлов
        2. Параллельно отправляем FIND_NODE
        3. Добавляем полученные узлы в кандидаты
        4. Повторяем пока не найдём k стабильных узлов
        
        Args:
            target_id: Целевой ID
            send_func: Функция для отправки сообщений: async (node, request) -> response
        
        Returns:
            Список k ближайших узлов
        """
        # Начальные кандидаты из routing table
        shortlist = self.routing_table.find_closest(target_id, count=K)
        
        if not shortlist:
            logger.warning("[DHT] iterative_find_node: no nodes in routing table")
            return []
        
        # Множество уже опрошенных узлов
        queried: Set[bytes] = set()
        
        # Лучший известный узел (ближайший к target)
        closest_node = shortlist[0] if shortlist else None
        closest_distance = xor_distance(target_id, closest_node.node_id) if closest_node else float('inf')
        
        while True:
            # Выбираем alpha ещё не опрошенных узлов
            to_query = []
            for node in shortlist:
                if node.node_id not in queried:
                    to_query.append(node)
                    if len(to_query) >= ALPHA:
                        break
            
            if not to_query:
                break  # Все узлы опрошены
            
            # Параллельно отправляем запросы
            tasks = []
            for node in to_query:
                queried.add(node.node_id)
                request = FindNodeRequest(
                    target_id=target_id,
                    sender_id=self.local_id,
                    sender_host=self.local_host,
                    sender_port=self.local_port,
                )
                tasks.append(self._send_find_node(node, request, send_func))
            
            # Ждём ответов с таймаутом
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем ответы
            found_closer = False
            for response in responses:
                if isinstance(response, Exception):
                    continue
                
                if response is None:
                    continue
                
                # Добавляем новые узлы в shortlist
                for new_node in response.nodes:
                    if new_node.node_id == self.local_id:
                        continue
                    
                    # Добавляем в routing table
                    self.routing_table.add_node(new_node)
                    
                    # Проверяем, ближе ли новый узел
                    distance = xor_distance(target_id, new_node.node_id)
                    if distance < closest_distance:
                        found_closer = True
                        closest_distance = distance
                    
                    # Добавляем в shortlist если ещё нет
                    if not any(n.node_id == new_node.node_id for n in shortlist):
                        shortlist.append(new_node)
            
            # Сортируем shortlist по расстоянию
            shortlist.sort(key=lambda n: xor_distance(target_id, n.node_id))
            shortlist = shortlist[:K]  # Оставляем только k лучших
            
            # Если не нашли более близких - завершаем
            if not found_closer:
                break
        
        logger.debug(
            f"[DHT] iterative_find_node: target={target_id.hex()[:16]}..., "
            f"found {len(shortlist)} nodes"
        )
        
        return shortlist
    
    async def iterative_find_value(
        self,
        key: bytes,
        send_func,
    ) -> Tuple[Optional[bytes], List[NodeInfo]]:
        """
        Итеративный поиск значения по ключу.
        
        [KADEMLIA] Алгоритм:
        1. Начинаем с alpha ближайших узлов к ключу
        2. Отправляем FIND_VALUE
        3. Если узел вернул значение - возвращаем его
        4. Иначе продолжаем как FIND_NODE
        
        Args:
            key: Ключ для поиска (20 байт)
            send_func: Функция для отправки сообщений
        
        Returns:
            (value, nodes): Значение или None, список ближайших узлов
        """
        # Сначала проверяем локально
        local = await self.storage.get(key)
        if local:
            return local.value, []
        
        # Начальные кандидаты
        shortlist = self.routing_table.find_closest(key, count=K)
        
        if not shortlist:
            return None, []
        
        queried: Set[bytes] = set()
        
        while True:
            to_query = [n for n in shortlist if n.node_id not in queried][:ALPHA]
            
            if not to_query:
                break
            
            tasks = []
            for node in to_query:
                queried.add(node.node_id)
                request = FindValueRequest(
                    key=key,
                    sender_id=self.local_id,
                    sender_host=self.local_host,
                    sender_port=self.local_port,
                )
                tasks.append(self._send_find_value(node, request, send_func))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            found_closer = False
            for response in responses:
                if isinstance(response, Exception):
                    continue
                
                if response is None:
                    continue
                
                # Если нашли значение - возвращаем
                if response.found:
                    logger.debug(f"[DHT] iterative_find_value: key={key.hex()[:16]}... FOUND")
                    return response.value, shortlist
                
                # Добавляем новые узлы
                for new_node in response.nodes:
                    if new_node.node_id == self.local_id:
                        continue
                    
                    self.routing_table.add_node(new_node)
                    
                    distance = xor_distance(key, new_node.node_id)
                    if not any(n.node_id == new_node.node_id for n in shortlist):
                        shortlist.append(new_node)
                        found_closer = True
            
            shortlist.sort(key=lambda n: xor_distance(key, n.node_id))
            shortlist = shortlist[:K]
            
            if not found_closer:
                break
        
        logger.debug(f"[DHT] iterative_find_value: key={key.hex()[:16]}... NOT FOUND")
        return None, shortlist
    
    async def iterative_store(
        self,
        key: bytes,
        value: bytes,
        send_func,
        ttl: int = DEFAULT_TTL,
    ) -> int:
        """
        Сохранить значение в DHT.
        
        [KADEMLIA] Алгоритм:
        1. Находим k ближайших узлов к ключу
        2. Отправляем STORE каждому из них
        
        Args:
            key: Ключ (20 байт)
            value: Значение
            send_func: Функция для отправки
            ttl: Время жизни
        
        Returns:
            Количество узлов, успешно сохранивших значение
        """
        # Сохраняем локально
        await self.storage.store(key, value, self.local_id, ttl)
        
        # Находим k ближайших узлов
        closest = await self.iterative_find_node(key, send_func)
        
        if not closest:
            return 1  # Только локально
        
        # Отправляем STORE всем ближайшим
        tasks = []
        for node in closest:
            request = StoreRequest(
                key=key,
                value=value,
                ttl=ttl,
                sender_id=self.local_id,
                sender_host=self.local_host,
                sender_port=self.local_port,
            )
            tasks.append(self._send_store(node, request, send_func))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 1  # Локальное сохранение
        for response in responses:
            if isinstance(response, StoreResponse) and response.success:
                success_count += 1
        
        logger.debug(
            f"[DHT] iterative_store: key={key.hex()[:16]}..., "
            f"stored on {success_count} nodes"
        )
        
        return success_count
    
    # =========================================================================
    # Send Helpers
    # =========================================================================
    
    async def _send_find_node(
        self,
        node: NodeInfo,
        request: FindNodeRequest,
        send_func,
    ) -> Optional[FindNodeResponse]:
        """Отправить FIND_NODE запрос."""
        try:
            response_data = await send_func(node, request.to_dict())
            if response_data:
                return FindNodeResponse.from_dict(response_data)
        except Exception as e:
            logger.debug(f"[DHT] FIND_NODE to {node.node_id.hex()[:16]}... failed: {e}")
            node.mark_failed()
        return None
    
    async def _send_find_value(
        self,
        node: NodeInfo,
        request: FindValueRequest,
        send_func,
    ) -> Optional[FindValueResponse]:
        """Отправить FIND_VALUE запрос."""
        try:
            response_data = await send_func(node, request.to_dict())
            if response_data:
                return FindValueResponse.from_dict(response_data)
        except Exception as e:
            logger.debug(f"[DHT] FIND_VALUE to {node.node_id.hex()[:16]}... failed: {e}")
            node.mark_failed()
        return None
    
    async def _send_store(
        self,
        node: NodeInfo,
        request: StoreRequest,
        send_func,
    ) -> Optional[StoreResponse]:
        """Отправить STORE запрос."""
        try:
            response_data = await send_func(node, request.to_dict())
            if response_data:
                return StoreResponse.from_dict(response_data)
        except Exception as e:
            logger.debug(f"[DHT] STORE to {node.node_id.hex()[:16]}... failed: {e}")
            node.mark_failed()
        return None


# =============================================================================
# Message Handlers for integration with core/protocol.py
# =============================================================================

class FindNodeHandler:
    """Обработчик FIND_NODE для интеграции с ProtocolRouter."""
    
    def __init__(self, dht_protocol: DHTProtocol):
        self.dht_protocol = dht_protocol
    
    async def handle(self, payload: Dict) -> Dict:
        """Обработать FIND_NODE запрос."""
        request = FindNodeRequest.from_dict(payload)
        response = await self.dht_protocol.handle_find_node(request)
        return response.to_dict()


class FindValueHandler:
    """Обработчик FIND_VALUE для интеграции с ProtocolRouter."""
    
    def __init__(self, dht_protocol: DHTProtocol):
        self.dht_protocol = dht_protocol
    
    async def handle(self, payload: Dict) -> Dict:
        """Обработать FIND_VALUE запрос."""
        request = FindValueRequest.from_dict(payload)
        response = await self.dht_protocol.handle_find_value(request)
        return response.to_dict()


class StoreHandler:
    """Обработчик STORE для интеграции с ProtocolRouter."""
    
    def __init__(self, dht_protocol: DHTProtocol):
        self.dht_protocol = dht_protocol
    
    async def handle(self, payload: Dict) -> Dict:
        """Обработать STORE запрос."""
        request = StoreRequest.from_dict(payload)
        response = await self.dht_protocol.handle_store(request)
        return response.to_dict()

