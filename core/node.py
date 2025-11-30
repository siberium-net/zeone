"""
Node - Главный класс P2P узла
=============================

[DECENTRALIZATION] Этот модуль реализует полностью децентрализованный узел:
- Каждый узел равноправен (нет "главного" сервера)
- Узел одновременно является клиентом и сервером
- Идентичность узла = его криптографический публичный ключ
- Discovery работает через gossip-протокол

[SECURITY] Все соединения верифицируются криптографически.
Узел общается только с проверенными пирами.
"""

import asyncio
import logging
import time
import socket
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Callable, Awaitable, Any, Tuple
from contextlib import suppress

from config import config, NetworkConfig
from .transport import Message, MessageType, Crypto, SimpleTransport, TrafficMasker
from .protocol import (
    ProtocolRouter,
    PingPongHandler,
    DiscoverHandler,
    PeerInfo,
)

# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class Peer:
    """
    Представление подключенного пира.
    
    [DECENTRALIZATION] Каждый пир - это равноправный участник сети.
    Информация о пирах хранится локально, без центрального реестра.
    """
    
    node_id: str
    host: str
    port: int
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    trust_score: float = 0.5
    last_seen: float = field(default_factory=time.time)
    last_ping: float = 0.0
    pending_pings: Dict[str, float] = field(default_factory=dict)  # nonce -> timestamp
    is_outbound: bool = False  # True если мы инициировали соединение
    
    @property
    def is_connected(self) -> bool:
        """Проверить, активно ли соединение."""
        return self.writer is not None and not self.writer.is_closing()
    
    async def send(self, message: Message, use_masking: bool = False) -> bool:
        """
        Отправить сообщение пиру.
        
        [SECURITY] Сообщение должно быть подписано перед отправкой.
        """
        if not self.is_connected:
            return False
        
        try:
            if use_masking:
                data = TrafficMasker.mask_as_http_request(message)
            else:
                data = SimpleTransport.pack(message)
            
            self.writer.write(data)
            await self.writer.drain()
            return True
        except (ConnectionError, OSError) as e:
            logger.warning(f"[PEER] Failed to send to {self.node_id[:8]}...: {e}")
            return False
    
    async def close(self) -> None:
        """Закрыть соединение с пиром."""
        if self.writer:
            self.writer.close()
            with suppress(Exception):
                await self.writer.wait_closed()
            self.writer = None
            self.reader = None


class PeerManager:
    """
    Менеджер пиров.
    
    [DECENTRALIZATION] Управляет списком известных и подключенных пиров.
    Нет центрального списка - каждый узел поддерживает свой.
    
    Функции:
    - Хранение информации о пирах
    - Выбор пиров для подключения
    - Heartbeat (проверка живости)
    - Балансировка соединений
    """
    
    def __init__(self, max_peers: int = 50):
        self.max_peers = max_peers
        self._peers: Dict[str, Peer] = {}  # node_id -> Peer
        self._known_peers: Dict[str, PeerInfo] = {}  # node_id -> PeerInfo (не подключены)
        self._lock = asyncio.Lock()
    
    async def add_peer(self, peer: Peer) -> bool:
        """
        Добавить подключенного пира.
        
        Returns:
            True если пир добавлен, False если превышен лимит
        """
        async with self._lock:
            if len(self._peers) >= self.max_peers:
                return False
            
            self._peers[peer.node_id] = peer
            
            # Удаляем из списка известных (теперь он подключен)
            self._known_peers.pop(peer.node_id, None)
            
            logger.info(f"[PEER] Added peer {peer.node_id[:8]}... ({peer.host}:{peer.port})")
            return True
    
    async def remove_peer(self, node_id: str) -> Optional[Peer]:
        """Удалить пира."""
        async with self._lock:
            peer = self._peers.pop(node_id, None)
            if peer:
                await peer.close()
                logger.info(f"[PEER] Removed peer {node_id[:8]}...")
            return peer
    
    def get_peer(self, node_id: str) -> Optional[Peer]:
        """Получить пира по ID."""
        return self._peers.get(node_id)
    
    def get_active_peers(self) -> List[Peer]:
        """Получить список активных пиров."""
        return [p for p in self._peers.values() if p.is_connected]
    
    def add_known_peer(self, info: PeerInfo) -> None:
        """Добавить информацию о известном (но не подключенном) пире."""
        if info.node_id not in self._peers:
            self._known_peers[info.node_id] = info
    
    def get_peers_to_connect(self, count: int = 5) -> List[PeerInfo]:
        """
        Получить список пиров для подключения.
        
        [DECENTRALIZATION] Выбираем пиров с учетом:
        - Trust score (предпочитаем надежных)
        - Давность (предпочитаем недавно виденных)
        """
        # Сортируем по trust_score * recency
        now = time.time()
        scored = []
        for info in self._known_peers.values():
            if info.node_id in self._peers:
                continue  # Уже подключен
            age = now - info.last_seen
            score = info.trust_score / (1 + age / 3600)  # Decay по времени
            scored.append((score, info))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [info for _, info in scored[:count]]
    
    @property
    def peer_count(self) -> int:
        """Количество подключенных пиров."""
        return len(self._peers)
    
    @property
    def known_peer_count(self) -> int:
        """Количество известных (но не подключенных) пиров."""
        return len(self._known_peers)


class Node:
    """
    Главный класс P2P узла.
    
    [DECENTRALIZATION] Каждый Node:
    - Имеет уникальную криптографическую идентичность
    - Может принимать входящие соединения (сервер)
    - Может инициировать исходящие соединения (клиент)
    - Участвует в discovery для поиска других узлов
    - Хранит локальную копию данных
    
    [SECURITY] Все сообщения подписываются и верифицируются.
    Нет доверия к неподписанным данным.
    """
    
    def __init__(
        self,
        crypto: Crypto,
        host: str = "0.0.0.0",
        port: int = 8468,
        use_masking: bool = False,
    ):
        """
        Инициализация узла.
        
        Args:
            crypto: Криптографический модуль с ключами
            host: Адрес для прослушивания
            port: Порт для прослушивания
            use_masking: Использовать HTTP-маскировку трафика
        """
        self.crypto = crypto
        self.host = host
        self.port = port
        self.use_masking = use_masking
        
        # Менеджер пиров
        self.peer_manager = PeerManager(max_peers=config.network.max_peers)
        
        # Маршрутизатор протокола
        self.router = ProtocolRouter()
        
        # Сервер
        self._server: Optional[asyncio.Server] = None
        self._running = False
        
        # Фоновые задачи
        self._tasks: Set[asyncio.Task] = set()
        
        # Callbacks
        self._on_peer_connected: List[Callable[[Peer], Awaitable[None]]] = []
        self._on_peer_disconnected: List[Callable[[Peer], Awaitable[None]]] = []
        self._on_message: List[Callable[[Message, Peer], Awaitable[None]]] = []
        
        logger.info(f"[NODE] Initialized with ID: {self.node_id[:16]}...")
    
    @property
    def node_id(self) -> str:
        """ID узла (публичный ключ в Base64)."""
        return self.crypto.node_id
    
    async def start(self) -> None:
        """
        Запустить узел.
        
        [DECENTRALIZATION] После запуска узел:
        1. Начинает принимать входящие соединения
        2. Подключается к bootstrap-узлам
        3. Запускает discovery для поиска других пиров
        4. Запускает heartbeat для проверки живости соединений
        """
        if self._running:
            return
        
        self._running = True
        
        # Запускаем TCP сервер
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
            reuse_address=True,
        )
        
        addr = self._server.sockets[0].getsockname()
        logger.info(f"[NODE] Server listening on {addr[0]}:{addr[1]}")
        
        # Запускаем фоновые задачи
        self._tasks.add(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.add(asyncio.create_task(self._discovery_loop()))
        
        # Подключаемся к bootstrap-узлам
        await self._connect_to_bootstrap()
    
    async def stop(self) -> None:
        """
        Остановить узел.
        
        [DECENTRALIZATION] Graceful shutdown:
        - Отключаемся от всех пиров
        - Останавливаем сервер
        - Отменяем фоновые задачи
        
        Другие узлы обнаружат отключение через heartbeat timeout.
        """
        if not self._running:
            return
        
        self._running = False
        logger.info("[NODE] Stopping...")
        
        # Отменяем фоновые задачи
        for task in self._tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        
        # Отключаем всех пиров
        for peer in list(self.peer_manager._peers.values()):
            await self.peer_manager.remove_peer(peer.node_id)
        
        # Останавливаем сервер
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        
        logger.info("[NODE] Stopped")
    
    async def _connect_to_bootstrap(self) -> None:
        """
        Подключиться к bootstrap-узлам.
        
        [DECENTRALIZATION] Bootstrap-узлы - это единственная
        "централизованная" часть системы. Однако:
        - Их может быть много
        - После первого подключения они не нужны
        - Любой узел может быть bootstrap-узлом
        """
        for host, port in config.network.bootstrap_nodes:
            if host == self.host and port == self.port:
                continue  # Не подключаемся к себе
            
            try:
                await self.connect_to_peer(host, port)
            except Exception as e:
                logger.warning(f"[NODE] Failed to connect to bootstrap {host}:{port}: {e}")
    
    async def connect_to_peer(self, host: str, port: int) -> Optional[Peer]:
        """
        Подключиться к пиру.
        
        [SECURITY] После подключения выполняется handshake:
        1. Отправляем PING с нашей подписью
        2. Ожидаем PONG с подписью пира
        3. Верифицируем подпись - получаем node_id пира
        
        Это гарантирует, что пир владеет заявленным ключом.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=config.network.connection_timeout,
            )
        except (OSError, asyncio.TimeoutError) as e:
            logger.debug(f"[NODE] Connection to {host}:{port} failed: {e}")
            return None
        
        # Выполняем handshake - отправляем PING
        ping = PingPongHandler.create_ping(self.crypto)
        
        try:
            data = SimpleTransport.pack(ping)
            writer.write(data)
            await writer.drain()
            
            # Ожидаем PONG
            header = await asyncio.wait_for(
                reader.readexactly(4),
                timeout=config.network.connection_timeout,
            )
            length = SimpleTransport.unpack_length(header)
            payload = await asyncio.wait_for(
                reader.readexactly(length),
                timeout=config.network.connection_timeout,
            )
            pong = SimpleTransport.unpack(payload)
            
            # Верифицируем PONG
            if not PingPongHandler.verify_pong(ping, pong, self.crypto):
                logger.warning(f"[NODE] Invalid PONG from {host}:{port}")
                writer.close()
                await writer.wait_closed()
                return None
            
            # Handshake успешен - создаем Peer
            peer = Peer(
                node_id=pong.sender_id,
                host=host,
                port=port,
                reader=reader,
                writer=writer,
                is_outbound=True,
            )
            
            if not await self.peer_manager.add_peer(peer):
                await peer.close()
                return None
            
            # Запускаем обработку входящих сообщений
            task = asyncio.create_task(self._read_loop(peer))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            
            # Уведомляем о подключении
            for callback in self._on_peer_connected:
                await callback(peer)
            
            logger.info(f"[NODE] Connected to peer {peer.node_id[:8]}... at {host}:{port}")
            return peer
            
        except Exception as e:
            logger.warning(f"[NODE] Handshake with {host}:{port} failed: {e}")
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            return None
    
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Обработать входящее соединение.
        
        [SECURITY] Входящие соединения также проходят handshake:
        1. Получаем PING от клиента
        2. Верифицируем подпись
        3. Отправляем PONG с нашей подписью
        """
        peername = writer.get_extra_info("peername")
        logger.debug(f"[NODE] Incoming connection from {peername}")
        
        try:
            # Ожидаем PING
            header = await asyncio.wait_for(
                reader.readexactly(4),
                timeout=config.network.connection_timeout,
            )
            length = SimpleTransport.unpack_length(header)
            payload = await asyncio.wait_for(
                reader.readexactly(length),
                timeout=config.network.connection_timeout,
            )
            ping = SimpleTransport.unpack(payload)
            
            # Обрабатываем PING через протокол
            context: Dict[str, Any] = {"peer_manager": self.peer_manager}
            pong = await self.router.route(ping, self.crypto, context)
            
            if pong is None:
                logger.warning(f"[NODE] Invalid PING from {peername}")
                writer.close()
                return
            
            # Отправляем PONG
            data = SimpleTransport.pack(pong)
            writer.write(data)
            await writer.drain()
            
            # Создаем Peer
            peer = Peer(
                node_id=ping.sender_id,
                host=peername[0] if peername else "unknown",
                port=peername[1] if peername else 0,
                reader=reader,
                writer=writer,
                is_outbound=False,
            )
            
            if not await self.peer_manager.add_peer(peer):
                await peer.close()
                return
            
            # Уведомляем о подключении
            for callback in self._on_peer_connected:
                await callback(peer)
            
            # Запускаем обработку
            await self._read_loop(peer)
            
        except Exception as e:
            logger.debug(f"[NODE] Connection handling error: {e}")
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
    
    async def _read_loop(self, peer: Peer) -> None:
        """
        Цикл чтения сообщений от пира.
        
        [DECENTRALIZATION] Каждое сообщение обрабатывается независимо.
        Узел может игнорировать сообщения от ненадежных пиров.
        """
        try:
            while self._running and peer.is_connected:
                # Читаем заголовок (4 байта длины)
                header = await peer.reader.readexactly(4)
                length = SimpleTransport.unpack_length(header)
                
                # Читаем payload
                payload = await peer.reader.readexactly(length)
                message = SimpleTransport.unpack(payload)
                
                # Обновляем last_seen
                peer.last_seen = time.time()
                
                # Обрабатываем сообщение
                context: Dict[str, Any] = {
                    "peer": peer,
                    "peer_manager": self.peer_manager,
                }
                
                response = await self.router.route(message, self.crypto, context)
                
                # Отправляем ответ если есть
                if response:
                    await peer.send(response, self.use_masking)
                
                # Обрабатываем discovered peers
                if "discovered_peers" in context:
                    for info in context["discovered_peers"]:
                        if info.node_id != self.node_id:
                            self.peer_manager.add_known_peer(info)
                
                # Уведомляем о сообщении
                for callback in self._on_message:
                    await callback(message, peer)
                    
        except asyncio.IncompleteReadError:
            logger.debug(f"[NODE] Connection closed by peer {peer.node_id[:8]}...")
        except Exception as e:
            logger.warning(f"[NODE] Read error from {peer.node_id[:8]}...: {e}")
        finally:
            await self.peer_manager.remove_peer(peer.node_id)
            
            # Уведомляем об отключении
            for callback in self._on_peer_disconnected:
                await callback(peer)
    
    async def _heartbeat_loop(self) -> None:
        """
        Фоновая задача heartbeat.
        
        [DECENTRALIZATION] Heartbeat позволяет обнаруживать
        отключившиеся узлы без центрального мониторинга.
        Каждый узел независимо проверяет своих пиров.
        """
        while self._running:
            await asyncio.sleep(config.network.heartbeat_interval)
            
            now = time.time()
            for peer in self.peer_manager.get_active_peers():
                # Проверяем, нужен ли PING
                if now - peer.last_seen > config.network.heartbeat_interval:
                    ping = PingPongHandler.create_ping(self.crypto)
                    peer.pending_pings[ping.nonce] = now
                    await peer.send(ping, self.use_masking)
                
                # Проверяем таймауты
                if now - peer.last_seen > config.network.heartbeat_interval * 3:
                    logger.info(f"[NODE] Peer {peer.node_id[:8]}... timed out")
                    await self.peer_manager.remove_peer(peer.node_id)
    
    async def _discovery_loop(self) -> None:
        """
        Фоновая задача discovery.
        
        [DECENTRALIZATION] Discovery расширяет сеть контактов:
        1. Запрашиваем списки пиров у подключенных узлов
        2. Подключаемся к новым пирам
        3. Повторяем для создания mesh-топологии
        """
        while self._running:
            await asyncio.sleep(60)  # Discovery каждую минуту
            
            # Запрашиваем списки пиров у активных соединений
            discover = DiscoverHandler.create_discover(self.crypto)
            for peer in self.peer_manager.get_active_peers()[:5]:
                await peer.send(discover, self.use_masking)
            
            # Подключаемся к новым пирам если есть место
            if self.peer_manager.peer_count < config.network.max_peers // 2:
                candidates = self.peer_manager.get_peers_to_connect(count=3)
                for info in candidates:
                    await self.connect_to_peer(info.host, info.port)
    
    async def broadcast(self, message: Message) -> int:
        """
        Отправить сообщение всем подключенным пирам.
        
        [DECENTRALIZATION] Broadcast используется для распространения
        информации по сети. Каждый узел пересылает сообщение своим пирам.
        
        Returns:
            Количество успешных отправок
        """
        # Подписываем сообщение
        signed = self.crypto.sign_message(message)
        
        count = 0
        for peer in self.peer_manager.get_active_peers():
            if await peer.send(signed, self.use_masking):
                count += 1
        
        return count
    
    async def send_to(self, node_id: str, message: Message) -> bool:
        """
        Отправить сообщение конкретному пиру.
        
        Returns:
            True если сообщение отправлено
        """
        peer = self.peer_manager.get_peer(node_id)
        if not peer:
            return False
        
        signed = self.crypto.sign_message(message)
        return await peer.send(signed, self.use_masking)
    
    def on_peer_connected(self, callback: Callable[[Peer], Awaitable[None]]) -> None:
        """Зарегистрировать callback на подключение пира."""
        self._on_peer_connected.append(callback)
    
    def on_peer_disconnected(self, callback: Callable[[Peer], Awaitable[None]]) -> None:
        """Зарегистрировать callback на отключение пира."""
        self._on_peer_disconnected.append(callback)
    
    def on_message(self, callback: Callable[[Message, Peer], Awaitable[None]]) -> None:
        """Зарегистрировать callback на входящее сообщение."""
        self._on_message.append(callback)


class UDPDiscovery:
    """
    UDP Discovery для локальной сети.
    
    [DECENTRALIZATION] LAN Discovery позволяет узлам находить
    друг друга в локальной сети без bootstrap-серверов:
    
    1. Узел отправляет broadcast на специальный порт
    2. Другие узлы в LAN отвечают своим адресом
    3. Узлы подключаются друг к другу
    
    Это полностью автономный способ создания сети.
    """
    
    def __init__(
        self,
        node: Node,
        broadcast_port: int = 8469,
    ):
        self.node = node
        self.broadcast_port = broadcast_port
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._running = False
    
    async def start(self) -> None:
        """Запустить UDP discovery."""
        if self._running:
            return
        
        self._running = True
        
        # Создаем UDP сокет для broadcast
        loop = asyncio.get_event_loop()
        
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: UDPDiscoveryProtocol(self),
            local_addr=("0.0.0.0", self.broadcast_port),
            allow_broadcast=True,
        )
        
        # Запускаем периодический broadcast
        asyncio.create_task(self._broadcast_loop())
        
        logger.info(f"[UDP] Discovery listening on port {self.broadcast_port}")
    
    async def stop(self) -> None:
        """Остановить UDP discovery."""
        self._running = False
        if self._transport:
            self._transport.close()
            self._transport = None
    
    async def _broadcast_loop(self) -> None:
        """Периодически отправляем broadcast."""
        while self._running:
            await asyncio.sleep(30)
            
            if self._transport:
                announce = {
                    "type": "announce",
                    "node_id": self.node.node_id,
                    "port": self.node.port,
                }
                data = str(announce).encode("utf-8")
                self._transport.sendto(data, ("<broadcast>", self.broadcast_port))


class UDPDiscoveryProtocol(asyncio.DatagramProtocol):
    """Протокол для UDP discovery."""
    
    def __init__(self, discovery: UDPDiscovery):
        self.discovery = discovery
    
    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Обработать полученный датаграм."""
        try:
            # Простой парсинг announce
            text = data.decode("utf-8")
            if "announce" in text and "node_id" in text:
                # Извлекаем информацию о пире
                # В реальности здесь нужен более надежный парсинг
                host = addr[0]
                # Создаем задачу для подключения
                asyncio.create_task(
                    self.discovery.node.connect_to_peer(host, self.discovery.node.port)
                )
        except Exception:
            pass

