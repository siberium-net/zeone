"""
Node - Главный класс P2P узла (Layer 2: Economy)
================================================

[DECENTRALIZATION] Этот модуль реализует полностью децентрализованный узел:
- Каждый узел равноправен (нет "главного" сервера)
- Узел одновременно является клиентом и сервером
- Идентичность узла = его криптографический публичный ключ
- Discovery работает через gossip-протокол

[SECURITY] Все соединения верифицируются криптографически.
Узел общается только с проверенными пирами.

[ECONOMY] Layer 2 - Экономика и репутация:
- Автоматический учет трафика (record_debt / record_claim)
- Блокировка leechers при превышении лимита долга
- Обмен балансом при handshake
"""

import asyncio
import logging
import time
import socket
import ipaddress
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Callable, Awaitable, Any, Tuple, TYPE_CHECKING
from contextlib import suppress

from config import config, NetworkConfig
from .transport import Message, MessageType, Crypto, SimpleTransport, TrafficMasker, BlockingTransport
from .protocol import (
    ProtocolRouter,
    PingPongHandler,
    DiscoverHandler,
    PeerInfo,
)

if TYPE_CHECKING:
    from economy.ledger import Ledger
    from agents.manager import AgentManager, ServiceResponse
    from cortex.amplifier import Amplifier

# Настройка логирования
logger = logging.getLogger(__name__)


def is_public_ip(value: str) -> bool:
    """Return True if value is a public IP address."""
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_unspecified
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
    )


def is_private_ip(value: str) -> bool:
    """Return True if value is a private/loopback/unspecified IP."""
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_unspecified or ip.is_link_local


async def detect_public_ip(timeout_s: float = 2.0) -> Optional[str]:
    """
    Best-effort public IP detection via STUN or HTTPS.
    Returns None on failure.
    """
    # Try STUN first (no external HTTP dependency).
    try:
        from core.nat import STUNClient

        stun = STUNClient()
        mapped = await asyncio.wait_for(stun.get_mapped_address(0), timeout=timeout_s)
        if mapped and getattr(mapped, "ip", ""):
            if is_public_ip(mapped.ip):
                return mapped.ip
    except Exception:
        pass

    # Fallback to simple HTTPS services.
    urls = (
        "https://ident.me",
        "https://ifconfig.me/ip",
        "https://api.ipify.org",
    )
    try:
        import aiohttp  # type: ignore
    except Exception:
        return None

    for url in urls:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout_s) as resp:
                    text = (await resp.text()).strip()
            if is_public_ip(text):
                return text
        except Exception:
            continue

    return None


def resolve_host_ips(host: str) -> set[str]:
    """Resolve host to IPv4 addresses (best-effort)."""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            ip = info[4][0]
            if ip:
                ips.add(ip)
    except Exception:
        pass
    return ips


def get_local_ipv4() -> Optional[str]:
    """Best-effort local IPv4 detection."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
    except Exception:
        return None


@dataclass
class Peer:
    """
    Представление подключенного пира.
    
    [DECENTRALIZATION] Каждый пир - это равноправный участник сети.
    Информация о пирах хранится локально, без центрального реестра.
    
    [ECONOMY] Каждый peer имеет:
    - balance: текущий баланс (+ = должен нам, - = мы должны)
    - bytes_sent: всего отправлено байт
    - bytes_received: всего получено байт
    - blocked: заблокирован ли из-за долга
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
    
    # [ECONOMY] Статистика трафика
    bytes_sent: int = 0
    bytes_received: int = 0
    blocked: bool = False  # Заблокирован из-за превышения долга
    remote_balance_claim: Optional[float] = None  # Баланс, заявленный пиром при handshake
    
    @property
    def is_connected(self) -> bool:
        """Проверить, активно ли соединение."""
        return self.writer is not None and not self.writer.is_closing()
    
    async def send(self, message: Message, use_masking: bool = False) -> bool:
        """
        Отправить сообщение пиру.
        
        [SECURITY] Сообщение должно быть подписано перед отправкой.
        
        Note: Эта версия НЕ учитывает трафик в ledger.
        Для учета используйте send_with_accounting().
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
            self.bytes_sent += len(data)
            return True
        except (ConnectionError, OSError) as e:
            logger.warning(f"[PEER] Failed to send to {self.node_id[:8]}...: {e}")
            return False
    
    async def send_with_accounting(
        self,
        message: Message,
        ledger: "Ledger",
        use_masking: bool = False,
    ) -> Tuple[bool, int, str]:
        """
        Отправить сообщение с учетом в ledger.
        
        [ECONOMY] Автоматически:
        - Проверяет блокировку (превышение долга)
        - Записывает claim в ledger
        - Обновляет статистику
        
        Returns:
            (success, bytes_sent, reason)
        """
        if not self.is_connected:
            return (False, 0, "Not connected")
        
        if self.blocked:
            return (False, 0, "Peer blocked due to excessive debt")
        
        # Проверяем блокировку в ledger
        can_send, reason = await ledger.check_can_send(self.node_id)
        if not can_send:
            self.blocked = True
            logger.warning(f"[PEER] Blocking {self.node_id[:8]}...: {reason}")
            return (False, 0, reason)
        
        try:
            if use_masking:
                data = TrafficMasker.mask_as_http_request(message)
            else:
                data = SimpleTransport.pack(message)
            
            size = len(data)
            
            self.writer.write(data)
            await self.writer.drain()
            
            # Учитываем в ledger - peer теперь должен нам
            await ledger.record_claim(self.node_id, size)
            
            self.bytes_sent += size
            return (True, size, "OK")
            
        except (ConnectionError, OSError) as e:
            logger.warning(f"[PEER] Failed to send to {self.node_id[:8]}...: {e}")
            return (False, 0, str(e))
    
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
    
    [ECONOMY] Layer 2:
    - Ведет учет трафика через Ledger
    - Обменивается балансом при handshake
    - Блокирует leechers
    """
    
    def __init__(
        self,
        crypto: Crypto,
        host: str = "0.0.0.0",
        port: int = 8468,
        use_masking: bool = False,
        ledger: Optional["Ledger"] = None,
        agent_manager: Optional["AgentManager"] = None,
        amplifier: Optional[Any] = None,
    ):
        """
        Инициализация узла.
        
        Args:
            crypto: Криптографический модуль с ключами
            host: Адрес для прослушивания
            port: Порт для прослушивания
            use_masking: Использовать HTTP-маскировку трафика
            ledger: Экземпляр Ledger для учета трафика (Layer 2)
            agent_manager: Менеджер услуг (Layer 3: Market)
        """
        self.crypto = crypto
        self.host = host
        self.port = port
        self.use_masking = use_masking
        
        # [ECONOMY] Ledger для учета долгов
        self.ledger = ledger
        
        # [MARKET] AgentManager для обработки запросов услуг
        self.agent_manager = agent_manager
        self.amplifier = amplifier
        
        # Менеджер пиров
        self.peer_manager = PeerManager(max_peers=config.network.max_peers)
        
        # Маршрутизатор протокола
        self.router = ProtocolRouter()
        
        # Сервер
        self._server: Optional[asyncio.Server] = None
        self._running = False
        
        # Фоновые задачи
        self._tasks: Set[asyncio.Task] = set()
        
        # DHT integration
        self._pending_dht_requests: Dict[str, asyncio.Future] = {}
        self._dht_request_handler: Optional[Callable[[Dict[str, Any], "Peer"], Awaitable[Optional[Dict[str, Any]]]]] = None
        
        # Callbacks
        self._on_peer_connected: List[Callable[[Peer], Awaitable[None]]] = []
        self._on_peer_disconnected: List[Callable[[Peer], Awaitable[None]]] = []
        self._on_message: List[Callable[[Message, Peer], Awaitable[None]]] = []
        
        # [ECONOMY] Callback для обработки баланса
        self._on_balance_received: List[Callable[[str, float], Awaitable[None]]] = []
        
        # [MARKET] Callback для ответов на запросы услуг
        self._on_service_response: List[Callable[["ServiceResponse", str], Awaitable[None]]] = []

        # Attach node to agents after callbacks are initialized
        if self.agent_manager and hasattr(self.agent_manager, "set_node"):
            try:
                self.agent_manager.set_node(self)
            except Exception as e:
                logger.warning(f"[NODE] Failed to attach node to agent manager: {e}")
        
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
        
        if self._server.sockets:
            addr = self._server.sockets[0].getsockname()
            logger.info(f"[NODE] Server listening on {addr[0]}:{addr[1]}")
        else:
            logger.warning("[NODE] Server started without sockets (restricted environment)")
        
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
        
        # Отменяем фоновые задачи (копируем set чтобы избежать изменения во время итерации)
        tasks_to_cancel = list(self._tasks)
        for task in tasks_to_cancel:
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
            if self._is_self_bootstrap(host, port):
                continue  # Не подключаемся к себе
            
            try:
                await self.connect_to_peer(host, port)
            except Exception as e:
                logger.warning(f"[NODE] Failed to connect to bootstrap {host}:{port}: {e}")

    def _is_self_bootstrap(self, host: str, port: int) -> bool:
        if port != self.port:
            return False

        if host in {"localhost", "127.0.0.1"}:
            return True

        self_ips: set[str] = set()
        if self.host and self.host != "0.0.0.0":
            if is_public_ip(self.host) or is_private_ip(self.host):
                self_ips.add(self.host)

        public_ip = getattr(config, "public_ip", "").strip()
        if public_ip and is_public_ip(public_ip):
            self_ips.add(public_ip)

        local_ip = get_local_ipv4()
        if local_ip:
            self_ips.add(local_ip)

        if host in self_ips:
            return True

        resolved = resolve_host_ips(host)
        if "127.0.0.1" in resolved:
            return True

        return bool(resolved & self_ips)
    
    async def connect_to_peer(self, host: str, port: int) -> Optional[Peer]:
        """
        Подключиться к пиру.
        
        [SECURITY] После подключения выполняется handshake:
        1. Отправляем PING с нашей подписью
        2. Ожидаем PONG с подписью пира
        3. Верифицируем подпись - получаем node_id пира
        4. [ECONOMY] Обмениваемся балансом
        
        Это гарантирует, что пир владеет заявленным ключом.
        """
        return await self._connect_with_streams(host, port)

    async def _connect_with_streams(
        self,
        host: str,
        port: int,
        reader: Optional[asyncio.StreamReader] = None,
        writer: Optional[asyncio.StreamWriter] = None,
    ) -> Optional[Peer]:
        """Внутренний метод подключения с уже созданными потоками (для тестов/ин-мемори)."""
        try:
            if reader is None or writer is None:
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
            
            # [ECONOMY] Обмен балансом после успешного handshake
            if self.ledger:
                await self._exchange_balance(peer)
            
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
    
    async def _exchange_balance(self, peer: Peer) -> None:
        """
        Обменяться информацией о балансе с пиром.
        
        [ECONOMY] При соединении узлы сообщают друг другу:
        "Ты помнишь, что ты мне должен X байт?"
        
        Это позволяет:
        - Восстановить баланс после реконнекта
        - Обнаружить расхождения в учете
        - Заблокировать должника если нужно
        """
        if not self.ledger:
            return
        
        try:
            # Получаем наше видение баланса
            balance_claim = await self.ledger.get_balance_claim(peer.node_id)
            
            # Отправляем BALANCE_CLAIM
            msg = Message(
                type=MessageType.BALANCE_CLAIM,
                payload=balance_claim,
                sender_id=self.node_id,
            )
            signed = self.crypto.sign_message(msg)
            await peer.send(signed, self.use_masking)
            
            logger.debug(
                f"[NODE] Sent balance claim to {peer.node_id[:8]}...: "
                f"{balance_claim['claimed_balance']:.0f} bytes"
            )
            
        except Exception as e:
            logger.warning(f"[NODE] Balance exchange failed: {e}")
    
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
            
            # [ECONOMY] Обмен балансом после успешного handshake
            if self.ledger:
                await self._exchange_balance(peer)
            
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
        
        [ECONOMY] При получении данных автоматически:
        - Записывается debt в ledger (мы должны peer)
        - Обновляется статистика bytes_received
        """
        try:
            while self._running and peer.is_connected:
                # Читаем заголовок (4 байта длины)
                header = await peer.reader.readexactly(4)
                length = SimpleTransport.unpack_length(header)
                
                # Читаем payload
                payload = await peer.reader.readexactly(length)
                
                # [ECONOMY] Учитываем полученные данные
                received_bytes = len(header) + len(payload)
                peer.bytes_received += received_bytes

                message = SimpleTransport.unpack(payload)
                
                # Обновляем last_seen
                peer.last_seen = time.time()

                # [STREAMING] VPN/streaming трафик тарифицируется отдельно
                skip_accounting = message.type in {
                    MessageType.VPN_DATA,
                    MessageType.VPN_CONNECT,
                    MessageType.VPN_CONNECT_RESULT,
                    MessageType.VPN_CLOSE,
                    MessageType.STREAM,
                }

                if self.ledger and not skip_accounting:
                    # Записываем debt - мы получили данные, должны peer
                    await self.ledger.record_debt(peer.node_id, received_bytes)
                
                # DHT request/response handling
                if await self._handle_dht_payload(peer, message):
                    continue
                
                # [ECONOMY] Обработка BALANCE_CLAIM
                if message.type == MessageType.BALANCE_CLAIM:
                    await self._handle_balance_claim(peer, message)
                    continue
                
                if message.type == MessageType.BALANCE_ACK:
                    await self._handle_balance_ack(peer, message)
                    continue
                
                # Обрабатываем сообщение
                context: Dict[str, Any] = {
                    "peer": peer,
                    "peer_manager": self.peer_manager,
                    "ledger": self.ledger,
                    "agent_manager": self.agent_manager,
                    "on_service_response": self._handle_service_response,
                    "amplifier": self.amplifier,
                }
                
                response = await self.router.route(message, self.crypto, context)
                
                # Отправляем ответ если есть
                if response:
                    if self.ledger:
                        # Используем send_with_accounting для учета отправки
                        await peer.send_with_accounting(response, self.ledger, self.use_masking)
                    else:
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
    
    async def _handle_balance_claim(self, peer: Peer, message: Message) -> None:
        """
        Обработать BALANCE_CLAIM от пира.
        
        [ECONOMY] Пир сообщает нам свое видение баланса.
        Сравниваем с нашим и решаем что делать.
        """
        if not self.crypto.verify_signature(message):
            logger.warning(f"[NODE] Invalid signature on BALANCE_CLAIM from {peer.node_id[:8]}...")
            return
        
        peer_claimed = message.payload.get("claimed_balance", 0)
        peer.remote_balance_claim = peer_claimed
        
        logger.debug(
            f"[NODE] Received balance claim from {peer.node_id[:8]}...: "
            f"{peer_claimed:.0f} bytes"
        )
        
        if self.ledger:
            # Согласовываем баланс
            result = await self.ledger.reconcile_balance(peer.node_id, peer_claimed)
            
            if result["status"] == "disputed":
                logger.warning(
                    f"[NODE] Balance dispute with {peer.node_id[:8]}...: "
                    f"ours={result['our_balance']:.0f}, theirs={peer_claimed:.0f}"
                )
            
            # Отправляем подтверждение с нашим балансом
            our_balance = await self.ledger.get_balance(peer.node_id)
            ack = Message(
                type=MessageType.BALANCE_ACK,
                payload={
                    "acknowledged": True,
                    "our_balance": our_balance,
                    "peer_claimed": peer_claimed,
                    "status": result["status"],
                },
                sender_id=self.node_id,
            )
            signed = self.crypto.sign_message(ack)
            await peer.send(signed, self.use_masking)
    
    async def _handle_balance_ack(self, peer: Peer, message: Message) -> None:
        """
        Обработать BALANCE_ACK от пира.
        
        [ECONOMY] Пир подтверждает получение нашего BALANCE_CLAIM
        и сообщает свое видение баланса.
        """
        if not self.crypto.verify_signature(message):
            return
        
        their_balance = message.payload.get("our_balance", 0)
        status = message.payload.get("status", "unknown")
        
        logger.debug(
            f"[NODE] Balance ACK from {peer.node_id[:8]}...: "
            f"their_view={their_balance:.0f}, status={status}"
        )
        
        # Уведомляем callbacks
        for callback in self._on_balance_received:
            await callback(peer.node_id, their_balance)
    
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
    
    async def broadcast(self, message: Message, with_accounting: bool = True) -> int:
        """
        Отправить сообщение всем подключенным пирам.
        
        [DECENTRALIZATION] Broadcast используется для распространения
        информации по сети. Каждый узел пересылает сообщение своим пирам.
        
        [ECONOMY] Если with_accounting=True и есть ledger:
        - Учитывает отправку в ledger
        - Пропускает заблокированных пиров
        
        Returns:
            Количество успешных отправок
        """
        # Подписываем сообщение
        signed = self.crypto.sign_message(message)
        
        count = 0
        blocked_count = 0
        
        for peer in self.peer_manager.get_active_peers():
            if with_accounting and self.ledger:
                success, _, reason = await peer.send_with_accounting(
                    signed, self.ledger, self.use_masking
                )
                if success:
                    count += 1
                elif "blocked" in reason.lower():
                    blocked_count += 1
            else:
                if await peer.send(signed, self.use_masking):
                    count += 1
        
        if blocked_count > 0:
            logger.info(f"[NODE] Broadcast skipped {blocked_count} blocked peers")
        
        return count
    
    async def send_to(self, node_id: str, message: Message, with_accounting: bool = True) -> bool:
        """
        Отправить сообщение конкретному пиру.
        
        [ECONOMY] Если with_accounting=True и есть ledger:
        - Проверяет блокировку
        - Учитывает отправку
        
        Returns:
            True если сообщение отправлено
        """
        peer = self.peer_manager.get_peer(node_id)
        if not peer:
            return False
        
        signed = self.crypto.sign_message(message)
        
        if with_accounting and self.ledger:
            success, _, _ = await peer.send_with_accounting(
                signed, self.ledger, self.use_masking
            )
            return success
        else:
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
    
    def on_balance_received(self, callback: Callable[[str, float], Awaitable[None]]) -> None:
        """
        Зарегистрировать callback на получение баланса от пира.
        
        [ECONOMY] Callback вызывается когда пир сообщает свое видение баланса.
        Args передаваемые в callback: (peer_node_id, their_balance_view)
        """
        self._on_balance_received.append(callback)
    
    def on_service_response(self, callback: Callable[["ServiceResponse", str], Awaitable[None]]) -> None:
        """
        Зарегистрировать callback на ответ по услуге.
        
        [MARKET] Callback вызывается когда приходит ответ на SERVICE_REQUEST.
        Args: (response, provider_node_id)
        """
        self._on_service_response.append(callback)
    
    # ---------------------------------------------------------------------
    # DHT integration hooks
    # ---------------------------------------------------------------------

    def set_dht_handler(self, handler: Callable[[Dict[str, Any], Peer], Awaitable[Optional[Dict[str, Any]]]]) -> None:
        """Установить обработчик DHT payload (используется KademliaNode)."""
        self._dht_request_handler = handler
    
    async def _handle_dht_payload(self, peer: Peer, message: Message) -> bool:
        """
        Обработать DHT запрос/ответ инкапсулированный в DATA.
        
        Returns:
            True если сообщение обработано как DHT.
        """
        if message.type != MessageType.DATA:
            return False
        
        if not isinstance(message.payload, dict):
            return False
        
        if not message.payload.get("dht"):
            return False
        
        if not self.crypto.verify_signature(message):
            logger.warning(f"[NODE] Dropping unsigned DHT payload from {peer.node_id[:8]}...")
            return True
        
        nonce = message.payload.get("nonce")
        
        # Ответ на наш запрос
        if "dht_response" in message.payload and nonce:
            future = self._pending_dht_requests.pop(nonce, None)
            if future and not future.done():
                future.set_result(message.payload["dht_response"])
            return True
        
        # Входящий запрос
        if "dht_request" in message.payload and self._dht_request_handler:
            try:
                response_payload = await self._dht_request_handler(
                    message.payload["dht_request"],
                    peer,
                )
            except Exception as e:
                logger.warning(f"[NODE] DHT handler error from {peer.node_id[:8]}...: {e}")
                response_payload = None
            
            if response_payload is not None:
                response = Message(
                    type=MessageType.DATA,
                    payload={
                        "dht": True,
                        "dht_response": response_payload,
                        "nonce": nonce,
                    },
                    sender_id=self.node_id,
                )
                signed = self.crypto.sign_message(response)
                if self.ledger:
                    await peer.send_with_accounting(signed, self.ledger, self.use_masking)
                else:
                    await peer.send(signed, self.use_masking)
            return True
        
        return False
    
    def set_ledger(self, ledger: "Ledger") -> None:
        """
        Установить Ledger для учета трафика.
        
        [ECONOMY] Можно установить ledger после создания Node.
        """
        self.ledger = ledger
    
    def set_agent_manager(self, agent_manager: "AgentManager") -> None:
        """
        Установить AgentManager для обработки запросов услуг.
        
        [MARKET] Можно установить после создания Node.
        """
        self.agent_manager = agent_manager
    
    async def _handle_service_response(self, response: "ServiceResponse", sender_id: str) -> None:
        """
        Обработать ответ на запрос услуги.
        
        [MARKET] Вызывается когда приходит SERVICE_RESPONSE.
        """
        logger.info(
            f"[NODE] Service response from {sender_id[:8]}...: "
            f"success={response.success}, cost={response.cost:.2f}"
        )
        
        # Вызываем зарегистрированные callbacks
        for callback in self._on_service_response:
            await callback(response, sender_id)
    
    async def request_service(
        self,
        peer_id: str,
        service_name: str,
        payload: Any,
        budget: float,
    ) -> bool:
        """
        Отправить запрос на услугу пиру.
        
        [MARKET] Создает SERVICE_REQUEST и отправляет указанному пиру.
        
        Args:
            peer_id: ID пира-провайдера услуги
            service_name: Название услуги ("echo", "storage", etc.)
            payload: Данные для обработки
            budget: Максимальный бюджет
        
        Returns:
            True если запрос отправлен
        """
        from .protocol import ServiceRequestHandler
        
        peer = self.peer_manager.get_peer(peer_id)
        if not peer:
            logger.warning(f"[NODE] Peer not found for service request: {peer_id[:8]}...")
            return False
        
        # Создаем запрос
        request = ServiceRequestHandler.create_service_request(
            self.crypto,
            service_name,
            payload,
            budget,
        )
        
        # Отправляем
        if self.ledger:
            success, _, reason = await peer.send_with_accounting(
                request, self.ledger, self.use_masking
            )
        else:
            success = await peer.send(request, self.use_masking)
            reason = "OK" if success else "Send failed"
        
        if success:
            logger.info(
                f"[NODE] Service request sent to {peer_id[:8]}...: "
                f"service={service_name}, budget={budget}"
            )
        else:
            logger.warning(f"[NODE] Failed to send service request: {reason}")
        
        return success
    
    async def broadcast_service_request(
        self,
        service_name: str,
        payload: Any,
        budget: float,
    ) -> int:
        """
        Отправить запрос на услугу всем пирам.
        
        [MARKET] Полезно для поиска провайдера услуги.
        Первый ответ выигрывает.
        
        Returns:
            Количество отправленных запросов
        """
        count = 0
        for peer in self.peer_manager.get_active_peers():
            if await self.request_service(peer.node_id, service_name, payload, budget):
                count += 1
        return count


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
