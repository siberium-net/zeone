"""
Kademlia Node - Обёртка над базовым Node с DHT функциональностью
================================================================

[KADEMLIA] KademliaNode расширяет базовый Node:
- Интегрирует RoutingTable для маршрутизации
- Добавляет DHTStorage для хранения
- Реализует dht_put/dht_get API
- Фоновые задачи: refresh buckets, republish, cleanup

[INTEGRATION] Взаимодействие с базовым Node:
- Использует существующий транспорт для отправки сообщений
- Добавляет обработчики DHT сообщений в ProtocolRouter
- Обновляет routing table при подключении пиров
"""

import asyncio
import time
import hashlib
import logging
from typing import Optional, Dict, List, Any, Tuple, TYPE_CHECKING
from pathlib import Path

from .routing import (
    RoutingTable, NodeInfo, K, ALPHA,
    xor_distance, key_to_id, hash_to_node_id,
    BUCKET_REFRESH_INTERVAL,
)
from .storage import DHTStorage, StoredValue, DEFAULT_TTL, string_to_key
from .protocol import (
    DHTProtocol,
    FindNodeRequest, FindNodeResponse,
    FindValueRequest, FindValueResponse,
    StoreRequest, StoreResponse,
)

if TYPE_CHECKING:
    from core.node import Node, Peer
    from core.transport import Message, MessageType

logger = logging.getLogger(__name__)


# Интервалы фоновых задач
REFRESH_INTERVAL = 3600  # 1 час - обновление buckets
REPUBLISH_INTERVAL = 3600  # 1 час - republish данных
CLEANUP_INTERVAL = 300  # 5 минут - очистка истёкших данных


class KademliaNode:
    """
    Kademlia DHT Node - расширение базового P2P узла.
    
    [KADEMLIA] Функциональность:
    - dht_put(key, value): Сохранить данные в DHT
    - dht_get(key): Получить данные из DHT
    - Автоматическое обновление routing table
    - Фоновые задачи для поддержания DHT
    
    [USAGE]
    ```python
    from core.dht import KademliaNode
    
    # Создаём Kademlia узел поверх базового
    kademlia = KademliaNode(base_node)
    await kademlia.start()
    
    # Сохраняем данные
    await kademlia.dht_put("my_key", b"my_value")
    
    # Получаем данные
    value = await kademlia.dht_get("my_key")
    ```
    """
    
    def __init__(
        self,
        base_node: 'Node',
        storage_path: str = "dht_storage.db",
    ):
        """
        Args:
            base_node: Базовый P2P узел
            storage_path: Путь к файлу хранилища DHT
        """
        self.base_node = base_node
        self.storage_path = storage_path
        
        # Конвертируем node_id из hex строки в bytes (20 байт)
        self.local_id = self._node_id_to_bytes(base_node.node_id)
        
        # Инициализируем компоненты Kademlia
        self.routing_table = RoutingTable(self.local_id, k=K)
        self.storage = DHTStorage(storage_path)
        
        # DHT Protocol
        self.protocol: Optional[DHTProtocol] = None
        
        # Фоновые задачи
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info(f"[KADEMLIA] Node created: {self.local_id.hex()[:16]}...")
    
    def _node_id_to_bytes(self, node_id: str) -> bytes:
        """
        Конвертировать node_id в 20 байт.
        
        Node ID в базовом узле - это hex-строка публичного ключа (64 байта).
        Для Kademlia нужен 20-байтный ID (SHA-1).
        """
        # Берём SHA-1 от полного node_id
        return hashlib.sha1(bytes.fromhex(node_id)).digest()
    
    async def start(self) -> None:
        """Запустить DHT функциональность."""
        if self._running:
            return
        
        # Инициализируем хранилище
        await self.storage.initialize()
        
        # Создаём протокол
        self.protocol = DHTProtocol(
            routing_table=self.routing_table,
            storage=self.storage,
            local_id=self.local_id,
            local_host=self.base_node.host,
            local_port=self.base_node.port,
        )
        
        # Регистрируем обработчики DHT сообщений
        self._register_handlers()
        
        # Запускаем фоновые задачи
        self._running = True
        self._tasks.append(asyncio.create_task(self._refresh_loop()))
        self._tasks.append(asyncio.create_task(self._republish_loop()))
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))
        
        # Добавляем callback на подключение пиров
        self.base_node.on_peer_connected(self._on_peer_connected)
        
        # Добавляем существующих пиров в routing table
        for peer in self.base_node.peer_manager.get_active_peers():
            await self._add_peer_to_routing(peer)
        
        logger.info("[KADEMLIA] DHT started")
    
    async def stop(self) -> None:
        """Остановить DHT функциональность."""
        self._running = False
        
        # Отменяем фоновые задачи
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        
        # Закрываем хранилище
        await self.storage.close()
        
        logger.info("[KADEMLIA] DHT stopped")
    
    def _register_handlers(self) -> None:
        """Зарегистрировать обработчики DHT сообщений."""
        # В реальной реализации здесь нужно добавить обработчики
        # в ProtocolRouter базового узла. Пока используем упрощённый подход.
        pass
    
    async def _on_peer_connected(self, peer: 'Peer') -> None:
        """Callback при подключении нового пира."""
        await self._add_peer_to_routing(peer)
    
    async def _add_peer_to_routing(self, peer: 'Peer') -> None:
        """Добавить пира в routing table."""
        peer_id_bytes = self._node_id_to_bytes(peer.node_id)
        
        node_info = NodeInfo(
            node_id=peer_id_bytes,
            host=peer.host,
            port=peer.port,
        )
        
        added, eviction = self.routing_table.add_node(node_info)
        
        if added:
            logger.debug(f"[KADEMLIA] Added peer to routing: {peer.node_id[:16]}...")
        elif eviction:
            # Bucket полон, нужно проверить head
            # TODO: Ping eviction candidate и заменить если не отвечает
            pass
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    async def dht_put(
        self,
        key: str,
        value: bytes,
        ttl: int = DEFAULT_TTL,
    ) -> int:
        """
        Сохранить данные в DHT.
        
        [KADEMLIA] Алгоритм:
        1. Хэшируем ключ в 20-байтный ID
        2. Находим k ближайших узлов к этому ID
        3. Отправляем STORE каждому из них
        4. Сохраняем локально
        
        Args:
            key: Строковый ключ
            value: Данные для сохранения
            ttl: Время жизни в секундах
        
        Returns:
            Количество узлов, сохранивших данные
        """
        key_id = string_to_key(key)
        
        # Сохраняем локально
        await self.storage.store(key_id, value, self.local_id, ttl)
        
        if not self.protocol:
            return 1
        
        # Используем iterative_store
        count = await self.protocol.iterative_store(
            key=key_id,
            value=value,
            send_func=self._send_dht_message,
            ttl=ttl,
        )
        
        logger.info(f"[KADEMLIA] PUT '{key}' -> stored on {count} nodes")
        return count
    
    async def dht_get(self, key: str) -> Optional[bytes]:
        """
        Получить данные из DHT.
        
        [KADEMLIA] Алгоритм:
        1. Хэшируем ключ в 20-байтный ID
        2. Проверяем локальное хранилище
        3. Если нет - выполняем iterative_find_value
        
        Args:
            key: Строковый ключ
        
        Returns:
            Данные или None если не найдено
        """
        key_id = string_to_key(key)
        
        # Проверяем локально
        local = await self.storage.get(key_id)
        if local:
            logger.debug(f"[KADEMLIA] GET '{key}' -> found locally")
            return local.value
        
        if not self.protocol:
            return None
        
        # Ищем в сети
        value, _ = await self.protocol.iterative_find_value(
            key=key_id,
            send_func=self._send_dht_message,
        )
        
        if value:
            # Кэшируем локально
            await self.storage.store(key_id, value, self.local_id)
            logger.info(f"[KADEMLIA] GET '{key}' -> found in network")
        else:
            logger.debug(f"[KADEMLIA] GET '{key}' -> not found")
        
        return value
    
    async def dht_delete(self, key: str) -> bool:
        """
        Удалить данные из локального хранилища.
        
        Note: В Kademlia нет операции DELETE в сети,
        данные просто истекают по TTL.
        
        Args:
            key: Строковый ключ
        
        Returns:
            True если удалено локально
        """
        key_id = string_to_key(key)
        return await self.storage.delete(key_id)
    
    async def find_node(self, target_id: bytes) -> List[NodeInfo]:
        """
        Найти k ближайших узлов к target_id.
        
        Args:
            target_id: Целевой ID (20 байт)
        
        Returns:
            Список ближайших узлов
        """
        if not self.protocol:
            return self.routing_table.find_closest(target_id, count=K)
        
        return await self.protocol.iterative_find_node(
            target_id=target_id,
            send_func=self._send_dht_message,
        )
    
    def get_stats(self) -> Dict:
        """Получить статистику DHT."""
        routing_stats = self.routing_table.get_stats()
        
        return {
            "local_id": self.local_id.hex(),
            "routing_table": routing_stats,
            "storage": asyncio.create_task(self.storage.get_stats()),
            "running": self._running,
        }
    
    async def get_full_stats(self) -> Dict:
        """Получить полную статистику DHT (async)."""
        routing_stats = self.routing_table.get_stats()
        storage_stats = await self.storage.get_stats()
        
        return {
            "local_id": self.local_id.hex(),
            "routing_table": routing_stats,
            "storage": storage_stats,
            "running": self._running,
        }
    
    # =========================================================================
    # Message Sending
    # =========================================================================
    
    async def _send_dht_message(
        self,
        node: NodeInfo,
        request: Dict,
    ) -> Optional[Dict]:
        """
        Отправить DHT сообщение узлу.
        
        [INTEGRATION] Использует базовый узел для отправки.
        
        Args:
            node: Целевой узел
            request: Словарь с запросом
        
        Returns:
            Словарь с ответом или None
        """
        # Находим peer по адресу
        peer = None
        for p in self.base_node.peer_manager.get_active_peers():
            if p.host == node.host and p.port == node.port:
                peer = p
                break
        
        if not peer:
            # Пытаемся подключиться
            try:
                peer = await self.base_node.connect_to_peer(node.host, node.port)
            except Exception as e:
                logger.debug(f"[KADEMLIA] Failed to connect to {node.host}:{node.port}: {e}")
                return None
        
        if not peer:
            return None
        
        # Создаём DHT сообщение
        from core.transport import Message, MessageType
        
        message = Message(
            type=MessageType.DATA,  # Используем DATA с DHT payload
            payload={
                "dht": True,
                "dht_request": request,
            },
            sender_id=self.base_node.node_id,
        )
        
        # Отправляем и ждём ответ
        # TODO: Реализовать request-response паттерн
        # Пока используем упрощённый подход
        try:
            success = await peer.send(message)
            if success:
                node.touch()
                # В реальной реализации нужно ждать ответ
                # Пока возвращаем None (ответ придёт асинхронно)
                return None
        except Exception as e:
            logger.debug(f"[KADEMLIA] Send failed: {e}")
            node.mark_failed()
        
        return None
    
    # =========================================================================
    # Background Tasks
    # =========================================================================
    
    async def _refresh_loop(self) -> None:
        """
        Фоновая задача: обновление routing table.
        
        [KADEMLIA] Bucket refresh:
        - Для каждого bucket, который не обновлялся час
        - Выполняем FIND_NODE для случайного ID в диапазоне bucket
        """
        while self._running:
            try:
                await asyncio.sleep(REFRESH_INTERVAL)
                
                if not self._running:
                    break
                
                refresh_ids = self.routing_table.get_refresh_ids()
                
                for target_id in refresh_ids:
                    if not self._running:
                        break
                    
                    await self.find_node(target_id)
                    await asyncio.sleep(1)  # Не перегружаем сеть
                
                if refresh_ids:
                    logger.debug(f"[KADEMLIA] Refreshed {len(refresh_ids)} buckets")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[KADEMLIA] Refresh error: {e}")
    
    async def _republish_loop(self) -> None:
        """
        Фоновая задача: republish данных.
        
        [KADEMLIA] Republish:
        - Каждый час повторно публикуем наши данные
        - Это поддерживает данные в сети при уходе узлов
        """
        while self._running:
            try:
                await asyncio.sleep(REPUBLISH_INTERVAL)
                
                if not self._running:
                    break
                
                # Получаем данные для republish
                values = await self.storage.get_republish_keys()
                
                for stored in values:
                    if not self._running:
                        break
                    
                    # Находим ближайшие узлы и отправляем STORE
                    if self.protocol:
                        await self.protocol.iterative_store(
                            key=stored.key,
                            value=stored.value,
                            send_func=self._send_dht_message,
                            ttl=stored.ttl,
                        )
                    
                    # Обновляем время republish
                    await self.storage.mark_republished(stored.key)
                    await asyncio.sleep(0.5)
                
                if values:
                    logger.debug(f"[KADEMLIA] Republished {len(values)} values")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[KADEMLIA] Republish error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """
        Фоновая задача: очистка истёкших данных.
        """
        while self._running:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)
                
                if not self._running:
                    break
                
                deleted = await self.storage.cleanup()
                
                if deleted > 0:
                    logger.debug(f"[KADEMLIA] Cleaned up {deleted} expired entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[KADEMLIA] Cleanup error: {e}")
    
    # =========================================================================
    # Bootstrap
    # =========================================================================
    
    async def bootstrap(self, nodes: List[Tuple[str, int]]) -> int:
        """
        Bootstrap DHT с известными узлами.
        
        [KADEMLIA] Bootstrap:
        1. Подключаемся к bootstrap узлам
        2. Выполняем FIND_NODE для своего ID
        3. Это заполняет routing table ближайшими узлами
        
        Args:
            nodes: Список (host, port) bootstrap узлов
        
        Returns:
            Количество добавленных узлов в routing table
        """
        initial_count = len(self.routing_table)
        
        for host, port in nodes:
            try:
                # Подключаемся через базовый узел
                peer = await self.base_node.connect_to_peer(host, port)
                if peer:
                    await self._add_peer_to_routing(peer)
            except Exception as e:
                logger.debug(f"[KADEMLIA] Bootstrap {host}:{port} failed: {e}")
        
        # Выполняем lookup для своего ID
        await self.find_node(self.local_id)
        
        added = len(self.routing_table) - initial_count
        logger.info(f"[KADEMLIA] Bootstrap complete: added {added} nodes")
        
        return added

