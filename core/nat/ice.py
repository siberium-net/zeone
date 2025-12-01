"""
ICE Agent - Interactive Connectivity Establishment
==================================================

[ICE] RFC 8445 - Координация NAT traversal:
1. Gather candidates (host, srflx, relay)
2. Exchange candidates с пиром (через DHT или signaling)
3. Connectivity checks (проверяем каждую пару)
4. Выбираем лучшее соединение

[CONNECTION PRIORITY]
1. Direct (host-to-host если оба публичные)
2. Hole Punch (srflx-to-srflx)
3. Relay (через P2P relay)

[STATES]
- NEW: Начальное состояние
- GATHERING: Сбор candidates
- CONNECTING: Проверка connectivity
- CONNECTED: Соединение установлено
- FAILED: Не удалось установить соединение
- CLOSED: Соединение закрыто
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any, Callable

from .stun import STUNClient, MappedAddress, NATType
from .candidates import (
    Candidate, CandidateType, CandidateGatherer,
    CandidatePair, prioritize_candidate_pairs, TransportType,
)
from .hole_punch import HolePuncher, PunchResult, HolePunchResult
from .relay import RelayClient, RelayConnection

logger = logging.getLogger(__name__)


class ICEState(Enum):
    """Состояние ICE агента."""
    NEW = auto()
    GATHERING = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    FAILED = auto()
    CLOSED = auto()


@dataclass
class ICEConnection:
    """
    Установленное ICE соединение.
    
    [CONNECTION] Содержит:
    - socket: Сокет для отправки/получения (UDP или TCP)
    - relay: RelayConnection если через relay
    - local/remote candidate: Выбранная пара
    """
    
    local_candidate: Candidate
    remote_candidate: Candidate
    connection_type: str  # "direct", "hole_punch", "relay"
    socket: Optional[Any] = None
    relay: Optional[RelayConnection] = None
    established_at: float = field(default_factory=time.time)
    latency_ms: float = 0
    
    @property
    def is_direct(self) -> bool:
        return self.connection_type in ("direct", "hole_punch")
    
    @property
    def is_relayed(self) -> bool:
        return self.connection_type == "relay"
    
    async def send(self, data: bytes) -> bool:
        """Отправить данные."""
        try:
            if self.relay:
                return await self.relay.send(data)
            elif self.socket:
                if hasattr(self.socket, 'sendto'):
                    # UDP
                    self.socket.sendto(data, self.remote_candidate.address)
                else:
                    # TCP
                    self.socket.send(data)
                return True
        except Exception as e:
            logger.error(f"[ICE] Send error: {e}")
        return False
    
    async def close(self) -> None:
        """Закрыть соединение."""
        try:
            if self.relay:
                await self.relay.close()
            elif self.socket:
                self.socket.close()
        except Exception:
            pass
    
    def to_dict(self) -> Dict:
        return {
            "local": self.local_candidate.to_dict(),
            "remote": self.remote_candidate.to_dict(),
            "type": self.connection_type,
            "latency_ms": self.latency_ms,
            "established_at": self.established_at,
        }


class ICEAgent:
    """
    ICE Agent - координация NAT traversal.
    
    [USAGE]
    ```python
    agent = ICEAgent(
        node_id="my_node_id",
        local_port=8468,
    )
    
    # Собираем свои candidates
    local_candidates = await agent.gather_candidates()
    
    # Публикуем в DHT
    await dht.put(f"ice:{node_id}", [c.to_dict() for c in local_candidates])
    
    # Получаем candidates пира из DHT
    remote_data = await dht.get(f"ice:{peer_id}")
    remote_candidates = [Candidate.from_dict(c) for c in remote_data]
    
    # Устанавливаем соединение
    connection = await agent.connect(remote_candidates)
    
    if connection:
        await connection.send(b"Hello!")
    ```
    """
    
    def __init__(
        self,
        node_id: str,
        local_port: int,
        stun_servers: Optional[List[Tuple[str, int]]] = None,
        relay_servers: Optional[List[Tuple[str, int]]] = None,
    ):
        """
        Args:
            node_id: ID этого узла
            local_port: Локальный порт
            stun_servers: STUN серверы для srflx candidates
            relay_servers: P2P relay серверы
        """
        self.node_id = node_id
        self.local_port = local_port
        self.stun_servers = stun_servers
        self.relay_servers = relay_servers or []
        
        # Компоненты
        self.stun_client = STUNClient(stun_servers) if stun_servers else STUNClient()
        self.candidate_gatherer = CandidateGatherer(
            local_port=local_port,
            stun_client=self.stun_client,
            relay_servers=relay_servers,
        )
        self.hole_puncher = HolePuncher(node_id)
        
        # Состояние
        self._state = ICEState.NEW
        self._local_candidates: List[Candidate] = []
        self._connections: Dict[str, ICEConnection] = {}  # peer_id -> connection
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_connection: Optional[Callable] = None
    
    @property
    def state(self) -> ICEState:
        return self._state
    
    @property
    def local_candidates(self) -> List[Candidate]:
        return self._local_candidates
    
    def _set_state(self, state: ICEState) -> None:
        """Изменить состояние."""
        old_state = self._state
        self._state = state
        logger.debug(f"[ICE] State: {old_state.name} -> {state.name}")
        
        if self._on_state_change:
            try:
                self._on_state_change(old_state, state)
            except Exception:
                pass
    
    def on_state_change(self, callback: Callable) -> None:
        """Установить callback для изменения состояния."""
        self._on_state_change = callback
    
    def on_connection(self, callback: Callable) -> None:
        """Установить callback для нового соединения."""
        self._on_connection = callback
    
    async def gather_candidates(self) -> List[Candidate]:
        """
        Собрать все candidates для этого узла.
        
        [ICE GATHERING]
        1. Host candidates (локальные IP)
        2. Server Reflexive (через STUN)
        3. Relay candidates (через relay серверы)
        
        Returns:
            Список candidates
        """
        self._set_state(ICEState.GATHERING)
        
        try:
            self._local_candidates = await self.candidate_gatherer.gather()
            
            logger.info(f"[ICE] Gathered {len(self._local_candidates)} candidates:")
            for c in self._local_candidates:
                logger.info(f"[ICE]   {c.type.value}: {c.ip}:{c.port} ({c.transport.value})")
            
            return self._local_candidates
            
        except Exception as e:
            logger.error(f"[ICE] Gathering failed: {e}")
            self._set_state(ICEState.FAILED)
            return []
    
    async def connect(
        self,
        remote_candidates: List[Candidate],
        peer_id: str = "",
        timeout: float = 30.0,
    ) -> Optional[ICEConnection]:
        """
        Установить соединение с пиром.
        
        [ICE CONNECTIVITY]
        1. Создаём пары (local, remote) candidates
        2. Сортируем по приоритету
        3. Проверяем каждую пару
        4. Возвращаем первое успешное соединение
        
        Args:
            remote_candidates: Candidates пира
            peer_id: ID пира
            timeout: Общий таймаут
        
        Returns:
            ICEConnection или None
        """
        if not self._local_candidates:
            await self.gather_candidates()
        
        if not self._local_candidates or not remote_candidates:
            logger.error("[ICE] No candidates available")
            self._set_state(ICEState.FAILED)
            return None
        
        self._set_state(ICEState.CONNECTING)
        
        # Создаём пары candidates
        pairs = prioritize_candidate_pairs(self._local_candidates, remote_candidates)
        logger.info(f"[ICE] Checking {len(pairs)} candidate pairs")
        
        start_time = time.time()
        
        # Группируем по типу для оптимизации
        direct_pairs = []
        punch_pairs = []
        relay_pairs = []
        
        for pair in pairs:
            if pair.local.type == CandidateType.HOST and pair.remote.type == CandidateType.HOST:
                if pair.remote.is_public:
                    direct_pairs.append(pair)
            elif pair.local.type in (CandidateType.HOST, CandidateType.SRFLX):
                if pair.remote.type in (CandidateType.HOST, CandidateType.SRFLX):
                    punch_pairs.append(pair)
            elif pair.local.type == CandidateType.RELAY or pair.remote.type == CandidateType.RELAY:
                relay_pairs.append(pair)
        
        # 1. Пробуем direct connection
        for pair in direct_pairs[:3]:
            if time.time() - start_time > timeout:
                break
            
            connection = await self._try_direct(pair)
            if connection:
                self._connections[peer_id] = connection
                self._set_state(ICEState.CONNECTED)
                return connection
        
        # 2. Пробуем hole punch
        for pair in punch_pairs[:5]:
            if time.time() - start_time > timeout:
                break
            
            connection = await self._try_hole_punch(pair)
            if connection:
                self._connections[peer_id] = connection
                self._set_state(ICEState.CONNECTED)
                return connection
        
        # 3. Пробуем relay
        for pair in relay_pairs[:3]:
            if time.time() - start_time > timeout:
                break
            
            connection = await self._try_relay(pair, peer_id)
            if connection:
                self._connections[peer_id] = connection
                self._set_state(ICEState.CONNECTED)
                return connection
        
        # Не удалось
        logger.warning("[ICE] All connectivity checks failed")
        self._set_state(ICEState.FAILED)
        return None
    
    async def _try_direct(self, pair: CandidatePair) -> Optional[ICEConnection]:
        """Попытка прямого подключения."""
        logger.debug(f"[ICE] Trying direct: {pair.local} -> {pair.remote}")
        
        try:
            if pair.local.transport == TransportType.TCP:
                # TCP connect
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(pair.remote.ip, pair.remote.port),
                    timeout=5.0,
                )
                
                # Используем writer.get_extra_info('socket') для raw socket
                sock = writer.get_extra_info('socket')
                
                return ICEConnection(
                    local_candidate=pair.local,
                    remote_candidate=pair.remote,
                    connection_type="direct",
                    socket=sock,
                    latency_ms=0,
                )
            else:
                # UDP - создаём сокет и проверяем доступность
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setblocking(False)
                sock.bind(("0.0.0.0", pair.local.port))
                
                # Отправляем тестовый пакет
                sock.sendto(b"ICE_CHECK", pair.remote.address)
                
                # Пробуем получить ответ
                loop = asyncio.get_event_loop()
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(sock, 1024),
                        timeout=2.0,
                    )
                    
                    return ICEConnection(
                        local_candidate=pair.local,
                        remote_candidate=pair.remote,
                        connection_type="direct",
                        socket=sock,
                    )
                except asyncio.TimeoutError:
                    sock.close()
                    return None
                    
        except Exception as e:
            logger.debug(f"[ICE] Direct failed: {e}")
            return None
    
    async def _try_hole_punch(self, pair: CandidatePair) -> Optional[ICEConnection]:
        """Попытка hole punch."""
        logger.debug(f"[ICE] Trying hole punch: {pair.local} -> {pair.remote}")
        
        try:
            result = await self.hole_puncher.punch(
                local_port=pair.local.port,
                remote_ip=pair.remote.ip,
                remote_port=pair.remote.port,
                udp_timeout=5.0,
                tcp_timeout=5.0,
            )
            
            if result.success:
                return ICEConnection(
                    local_candidate=pair.local,
                    remote_candidate=pair.remote,
                    connection_type="hole_punch",
                    socket=result.socket,
                    latency_ms=result.latency_ms,
                )
                
        except Exception as e:
            logger.debug(f"[ICE] Hole punch failed: {e}")
        
        return None
    
    async def _try_relay(
        self,
        pair: CandidatePair,
        peer_id: str,
    ) -> Optional[ICEConnection]:
        """Попытка подключения через relay."""
        # Определяем relay сервер
        relay_candidate = None
        if pair.local.type == CandidateType.RELAY:
            relay_candidate = pair.local
        elif pair.remote.type == CandidateType.RELAY:
            relay_candidate = pair.remote
        
        if not relay_candidate or not relay_candidate.relay_server:
            return None
        
        logger.debug(f"[ICE] Trying relay: {relay_candidate.relay_server}")
        
        try:
            # Парсим адрес relay
            parts = relay_candidate.relay_server.split(":")
            relay_host = parts[0]
            relay_port = int(parts[1]) if len(parts) > 1 else 8469
            
            # Подключаемся к relay
            client = RelayClient(self.node_id)
            connected = await client.connect(relay_host, relay_port)
            
            if not connected:
                return None
            
            # Запрашиваем соединение с пиром
            await client.connect_to_peer(peer_id)
            
            # Ждём немного для установления
            await asyncio.sleep(1.0)
            
            if client.is_connected:
                return ICEConnection(
                    local_candidate=pair.local,
                    remote_candidate=pair.remote,
                    connection_type="relay",
                    relay=RelayConnection(client=client, peer_id=peer_id),
                )
            
            await client.disconnect()
            
        except Exception as e:
            logger.debug(f"[ICE] Relay failed: {e}")
        
        return None
    
    def get_connection(self, peer_id: str) -> Optional[ICEConnection]:
        """Получить соединение с пиром."""
        return self._connections.get(peer_id)
    
    async def close_connection(self, peer_id: str) -> None:
        """Закрыть соединение с пиром."""
        connection = self._connections.pop(peer_id, None)
        if connection:
            await connection.close()
    
    async def close_all(self) -> None:
        """Закрыть все соединения."""
        for peer_id in list(self._connections.keys()):
            await self.close_connection(peer_id)
        
        self._set_state(ICEState.CLOSED)
    
    def get_stats(self) -> Dict:
        """Статистика ICE агента."""
        return {
            "state": self._state.name,
            "local_candidates": len(self._local_candidates),
            "connections": len(self._connections),
            "connection_types": {
                peer_id: conn.connection_type
                for peer_id, conn in self._connections.items()
            },
        }


async def create_ice_connection(
    node_id: str,
    peer_id: str,
    local_port: int,
    remote_candidates: List[Dict],
) -> Optional[ICEConnection]:
    """
    Удобная функция для создания ICE соединения.
    
    Args:
        node_id: Наш node_id
        peer_id: ID пира
        local_port: Локальный порт
        remote_candidates: Candidates пира (как словари)
    
    Returns:
        ICEConnection или None
    """
    agent = ICEAgent(node_id=node_id, local_port=local_port)
    
    # Собираем свои candidates
    await agent.gather_candidates()
    
    # Конвертируем remote candidates
    candidates = [Candidate.from_dict(c) for c in remote_candidates]
    
    # Устанавливаем соединение
    return await agent.connect(candidates, peer_id)

