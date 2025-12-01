"""
P2P Relay - Ретрансляция трафика через узлы с публичным IP
==========================================================

[P2P RELAY] В отличие от TURN:
- Любой узел с публичным IP может быть relay
- Не нужен выделенный сервер
- Децентрализованно: relay узлы регистрируются в DHT

[PROTOCOL]
1. Client (за NAT) подключается к Relay
2. Client регистрирует peer_id для relay
3. Другой Client подключается к тому же Relay
4. Relay пересылает пакеты между ними

[DHT INTEGRATION]
- Relay регистрируется: dht.put("relay:{node_id}", {ip, port, capacity})
- Клиент ищет relay: dht.get("relay:*") или через discovery

[MESSAGE FORMAT]
+-------+--------+----------+---------+
| Magic | Type   | Peer ID  | Payload |
| 4     | 1      | 32       | ...     |
+-------+--------+----------+---------+
"""

import asyncio
import struct
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, List, Tuple, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# Protocol constants
RELAY_MAGIC = b"P2PR"  # 4 bytes
RELAY_VERSION = 1

# Message types
class RelayMessageType(IntEnum):
    REGISTER = 1       # Клиент регистрируется на relay
    REGISTER_ACK = 2   # Подтверждение регистрации
    CONNECT = 3        # Запрос на соединение с пиром
    CONNECT_ACK = 4    # Подтверждение соединения
    DATA = 5           # Данные для пересылки
    PING = 6           # Keep-alive
    PONG = 7           # Ответ на ping
    DISCONNECT = 8     # Отключение
    ERROR = 9          # Ошибка


# Limits
MAX_RELAYED_PEERS = 100  # Максимум пиров через один relay
MAX_PACKET_SIZE = 65536  # 64KB
RELAY_TIMEOUT = 60.0  # Секунды до отключения неактивного клиента
PING_INTERVAL = 20.0  # Интервал ping


@dataclass
class RelayedPeer:
    """Информация о пире, подключенном через relay."""
    peer_id: str
    writer: asyncio.StreamWriter
    reader: asyncio.StreamReader
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    bytes_relayed: int = 0
    target_peer_id: Optional[str] = None  # К кому подключен
    
    def touch(self):
        self.last_activity = time.time()
    
    @property
    def is_stale(self) -> bool:
        return time.time() - self.last_activity > RELAY_TIMEOUT


class RelayServer:
    """
    P2P Relay Server - пересылка трафика между узлами за NAT.
    
    [DECENTRALIZATION] Любой узел с публичным IP может запустить relay:
    - Не требует специальной инфраструктуры
    - Узлы автоматически становятся relay если имеют публичный IP
    - Регистрация в DHT для обнаружения
    
    [USAGE]
    ```python
    # На узле с публичным IP
    relay = RelayServer(host="0.0.0.0", port=8469)
    await relay.start()
    
    # Регистрация в DHT
    await dht.put(f"relay:{node_id}", {
        "ip": public_ip,
        "port": 8469,
        "capacity": relay.available_slots,
    })
    ```
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8469,
        max_peers: int = MAX_RELAYED_PEERS,
        node_id: str = "",
    ):
        """
        Args:
            host: Адрес для прослушивания
            port: Порт для прослушивания
            max_peers: Максимум подключенных пиров
            node_id: ID этого узла
        """
        self.host = host
        self.port = port
        self.max_peers = max_peers
        self.node_id = node_id
        
        # Подключенные пиры
        self._peers: Dict[str, RelayedPeer] = {}
        
        # Сервер
        self._server: Optional[asyncio.Server] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Статистика
        self.total_bytes_relayed = 0
        self.total_connections = 0
    
    async def start(self) -> None:
        """Запустить relay сервер."""
        if self._running:
            return
        
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        
        self._running = True
        
        # Фоновая задача для очистки
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))
        
        logger.info(f"[RELAY] Server started on {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Остановить relay сервер."""
        self._running = False
        
        # Отменяем задачи
        for task in self._tasks:
            task.cancel()
        
        # Закрываем соединения
        for peer in list(self._peers.values()):
            try:
                peer.writer.close()
                await peer.writer.wait_closed()
            except Exception:
                pass
        
        self._peers.clear()
        
        # Останавливаем сервер
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        logger.info("[RELAY] Server stopped")
    
    @property
    def available_slots(self) -> int:
        """Доступные слоты для новых пиров."""
        return max(0, self.max_peers - len(self._peers))
    
    @property
    def is_available(self) -> bool:
        """Есть ли свободные слоты."""
        return self.available_slots > 0
    
    def get_stats(self) -> Dict:
        """Статистика relay."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "connected_peers": len(self._peers),
            "max_peers": self.max_peers,
            "available_slots": self.available_slots,
            "total_connections": self.total_connections,
            "total_bytes_relayed": self.total_bytes_relayed,
        }
    
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Обработка подключения клиента."""
        peer_addr = writer.get_extra_info("peername")
        logger.debug(f"[RELAY] New connection from {peer_addr}")
        
        peer: Optional[RelayedPeer] = None
        
        try:
            while self._running:
                # Читаем сообщение
                msg = await self._read_message(reader)
                if not msg:
                    break
                
                msg_type, peer_id, payload = msg
                
                if msg_type == RelayMessageType.REGISTER:
                    # Регистрация нового пира
                    if not self.is_available:
                        await self._send_error(writer, "Relay is full")
                        break
                    
                    peer = RelayedPeer(
                        peer_id=peer_id,
                        writer=writer,
                        reader=reader,
                    )
                    self._peers[peer_id] = peer
                    self.total_connections += 1
                    
                    # Отправляем подтверждение
                    await self._send_message(
                        writer,
                        RelayMessageType.REGISTER_ACK,
                        self.node_id,
                        b"OK",
                    )
                    
                    logger.info(f"[RELAY] Peer registered: {peer_id[:16]}...")
                
                elif msg_type == RelayMessageType.CONNECT:
                    # Запрос на соединение с другим пиром
                    if not peer:
                        await self._send_error(writer, "Not registered")
                        continue
                    
                    target_id = payload.decode("utf-8")
                    
                    if target_id in self._peers:
                        peer.target_peer_id = target_id
                        self._peers[target_id].target_peer_id = peer_id
                        
                        await self._send_message(
                            writer,
                            RelayMessageType.CONNECT_ACK,
                            self.node_id,
                            b"CONNECTED",
                        )
                        
                        # Уведомляем целевого пира
                        await self._send_message(
                            self._peers[target_id].writer,
                            RelayMessageType.CONNECT_ACK,
                            peer_id,
                            b"INCOMING",
                        )
                        
                        logger.debug(f"[RELAY] Connected: {peer_id[:8]}... <-> {target_id[:8]}...")
                    else:
                        await self._send_error(writer, f"Peer not found: {target_id[:16]}")
                
                elif msg_type == RelayMessageType.DATA:
                    # Пересылка данных
                    if not peer or not peer.target_peer_id:
                        continue
                    
                    target = self._peers.get(peer.target_peer_id)
                    if target:
                        await self._send_message(
                            target.writer,
                            RelayMessageType.DATA,
                            peer_id,
                            payload,
                        )
                        
                        peer.touch()
                        peer.bytes_relayed += len(payload)
                        self.total_bytes_relayed += len(payload)
                
                elif msg_type == RelayMessageType.PING:
                    await self._send_message(
                        writer,
                        RelayMessageType.PONG,
                        self.node_id,
                        b"",
                    )
                    if peer:
                        peer.touch()
                
                elif msg_type == RelayMessageType.DISCONNECT:
                    break
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"[RELAY] Client error: {e}")
        finally:
            # Очистка
            if peer:
                # Уведомляем target если есть
                if peer.target_peer_id and peer.target_peer_id in self._peers:
                    try:
                        await self._send_message(
                            self._peers[peer.target_peer_id].writer,
                            RelayMessageType.DISCONNECT,
                            peer.peer_id,
                            b"",
                        )
                        self._peers[peer.target_peer_id].target_peer_id = None
                    except Exception:
                        pass
                
                self._peers.pop(peer.peer_id, None)
                logger.debug(f"[RELAY] Peer disconnected: {peer.peer_id[:16]}...")
            
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
    
    async def _read_message(
        self,
        reader: asyncio.StreamReader,
    ) -> Optional[Tuple[RelayMessageType, str, bytes]]:
        """Прочитать сообщение от клиента."""
        try:
            # Header: magic (4) + type (1) + peer_id_len (1) + payload_len (2)
            header = await asyncio.wait_for(
                reader.readexactly(8),
                timeout=RELAY_TIMEOUT,
            )
            
            magic = header[:4]
            if magic != RELAY_MAGIC:
                return None
            
            msg_type = header[4]
            peer_id_len = header[5]
            payload_len = struct.unpack(">H", header[6:8])[0]
            
            # Peer ID
            peer_id = ""
            if peer_id_len > 0:
                peer_id_bytes = await reader.readexactly(peer_id_len)
                peer_id = peer_id_bytes.decode("utf-8")
            
            # Payload
            payload = b""
            if payload_len > 0:
                payload = await reader.readexactly(payload_len)
            
            return RelayMessageType(msg_type), peer_id, payload
            
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None
    
    async def _send_message(
        self,
        writer: asyncio.StreamWriter,
        msg_type: RelayMessageType,
        peer_id: str,
        payload: bytes,
    ) -> bool:
        """Отправить сообщение клиенту."""
        try:
            peer_id_bytes = peer_id.encode("utf-8")[:255]
            
            header = (
                RELAY_MAGIC +
                bytes([msg_type.value]) +
                bytes([len(peer_id_bytes)]) +
                struct.pack(">H", len(payload))
            )
            
            writer.write(header + peer_id_bytes + payload)
            await writer.drain()
            return True
            
        except Exception as e:
            logger.debug(f"[RELAY] Send error: {e}")
            return False
    
    async def _send_error(
        self,
        writer: asyncio.StreamWriter,
        error: str,
    ) -> None:
        """Отправить сообщение об ошибке."""
        await self._send_message(
            writer,
            RelayMessageType.ERROR,
            self.node_id,
            error.encode("utf-8"),
        )
    
    async def _cleanup_loop(self) -> None:
        """Фоновая задача для очистки неактивных соединений."""
        while self._running:
            try:
                await asyncio.sleep(30)
                
                stale = [
                    peer_id for peer_id, peer in self._peers.items()
                    if peer.is_stale
                ]
                
                for peer_id in stale:
                    peer = self._peers.pop(peer_id, None)
                    if peer:
                        try:
                            peer.writer.close()
                        except Exception:
                            pass
                        logger.debug(f"[RELAY] Removed stale peer: {peer_id[:16]}...")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[RELAY] Cleanup error: {e}")


class RelayClient:
    """
    Клиент для подключения к P2P Relay.
    
    [USAGE]
    ```python
    client = RelayClient(peer_id="my_node_id")
    
    # Подключаемся к relay
    await client.connect("relay.example.com", 8469)
    
    # Запрашиваем соединение с другим пиром
    await client.connect_to_peer("target_peer_id")
    
    # Отправляем данные
    await client.send(b"Hello!")
    
    # Получаем данные
    data = await client.recv()
    ```
    """
    
    def __init__(self, peer_id: str):
        """
        Args:
            peer_id: ID этого узла
        """
        self.peer_id = peer_id
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._target_peer_id: Optional[str] = None
        self._relay_node_id: Optional[str] = None
        
        # Callbacks
        self._on_data: Optional[callable] = None
        self._on_disconnect: Optional[callable] = None
        
        # Фоновые задачи
        self._tasks: List[asyncio.Task] = []
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self, host: str, port: int) -> bool:
        """
        Подключиться к relay серверу.
        
        Args:
            host: Адрес relay
            port: Порт relay
        
        Returns:
            True если успешно
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=10.0,
            )
            
            # Регистрируемся
            await self._send_message(
                RelayMessageType.REGISTER,
                self.peer_id,
                b"",
            )
            
            # Ждём подтверждения
            msg = await self._read_message()
            if msg and msg[0] == RelayMessageType.REGISTER_ACK:
                self._relay_node_id = msg[1]
                self._connected = True
                
                # Запускаем прослушивание
                self._tasks.append(asyncio.create_task(self._recv_loop()))
                self._tasks.append(asyncio.create_task(self._ping_loop()))
                
                logger.info(f"[RELAY_CLIENT] Connected to relay at {host}:{port}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[RELAY_CLIENT] Connect failed: {e}")
            return False
    
    async def connect_to_peer(self, target_peer_id: str) -> bool:
        """
        Запросить соединение с другим пиром через relay.
        
        Args:
            target_peer_id: ID целевого пира
        
        Returns:
            True если успешно
        """
        if not self._connected:
            return False
        
        await self._send_message(
            RelayMessageType.CONNECT,
            self.peer_id,
            target_peer_id.encode("utf-8"),
        )
        
        self._target_peer_id = target_peer_id
        return True
    
    async def send(self, data: bytes) -> bool:
        """
        Отправить данные через relay.
        
        Args:
            data: Данные для отправки
        
        Returns:
            True если успешно
        """
        if not self._connected:
            return False
        
        return await self._send_message(
            RelayMessageType.DATA,
            self.peer_id,
            data,
        )
    
    async def disconnect(self) -> None:
        """Отключиться от relay."""
        self._connected = False
        
        # Отменяем задачи
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        
        # Отправляем DISCONNECT
        if self._writer:
            try:
                await self._send_message(
                    RelayMessageType.DISCONNECT,
                    self.peer_id,
                    b"",
                )
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        
        self._reader = None
        self._writer = None
    
    def on_data(self, callback: callable) -> None:
        """Установить callback для входящих данных."""
        self._on_data = callback
    
    def on_disconnect(self, callback: callable) -> None:
        """Установить callback для отключения."""
        self._on_disconnect = callback
    
    async def _send_message(
        self,
        msg_type: RelayMessageType,
        peer_id: str,
        payload: bytes,
    ) -> bool:
        """Отправить сообщение на relay."""
        if not self._writer:
            return False
        
        try:
            peer_id_bytes = peer_id.encode("utf-8")[:255]
            
            header = (
                RELAY_MAGIC +
                bytes([msg_type.value]) +
                bytes([len(peer_id_bytes)]) +
                struct.pack(">H", len(payload))
            )
            
            self._writer.write(header + peer_id_bytes + payload)
            await self._writer.drain()
            return True
            
        except Exception as e:
            logger.debug(f"[RELAY_CLIENT] Send error: {e}")
            return False
    
    async def _read_message(
        self,
    ) -> Optional[Tuple[RelayMessageType, str, bytes]]:
        """Прочитать сообщение от relay."""
        if not self._reader:
            return None
        
        try:
            header = await asyncio.wait_for(
                self._reader.readexactly(8),
                timeout=RELAY_TIMEOUT,
            )
            
            magic = header[:4]
            if magic != RELAY_MAGIC:
                return None
            
            msg_type = header[4]
            peer_id_len = header[5]
            payload_len = struct.unpack(">H", header[6:8])[0]
            
            peer_id = ""
            if peer_id_len > 0:
                peer_id_bytes = await self._reader.readexactly(peer_id_len)
                peer_id = peer_id_bytes.decode("utf-8")
            
            payload = b""
            if payload_len > 0:
                payload = await self._reader.readexactly(payload_len)
            
            return RelayMessageType(msg_type), peer_id, payload
            
        except Exception:
            return None
    
    async def _recv_loop(self) -> None:
        """Фоновое получение сообщений."""
        while self._connected:
            try:
                msg = await self._read_message()
                if not msg:
                    break
                
                msg_type, peer_id, payload = msg
                
                if msg_type == RelayMessageType.DATA:
                    if self._on_data:
                        try:
                            await self._on_data(peer_id, payload)
                        except Exception as e:
                            logger.error(f"[RELAY_CLIENT] Callback error: {e}")
                
                elif msg_type == RelayMessageType.CONNECT_ACK:
                    logger.info(f"[RELAY_CLIENT] Connected to peer: {peer_id[:16]}...")
                
                elif msg_type == RelayMessageType.DISCONNECT:
                    logger.info(f"[RELAY_CLIENT] Peer disconnected: {peer_id[:16]}...")
                    if self._on_disconnect:
                        await self._on_disconnect(peer_id)
                
                elif msg_type == RelayMessageType.ERROR:
                    logger.error(f"[RELAY_CLIENT] Error: {payload.decode()}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[RELAY_CLIENT] Recv loop error: {e}")
                break
        
        self._connected = False
    
    async def _ping_loop(self) -> None:
        """Фоновый ping для keep-alive."""
        while self._connected:
            try:
                await asyncio.sleep(PING_INTERVAL)
                await self._send_message(
                    RelayMessageType.PING,
                    self.peer_id,
                    b"",
                )
            except asyncio.CancelledError:
                break
            except Exception:
                pass


@dataclass
class RelayConnection:
    """
    Обёртка над relay соединением для унифицированного API.
    
    Позволяет использовать relay так же как обычное соединение.
    """
    client: RelayClient
    peer_id: str
    
    async def send(self, data: bytes) -> bool:
        """Отправить данные."""
        return await self.client.send(data)
    
    async def close(self) -> None:
        """Закрыть соединение."""
        await self.client.disconnect()
    
    @property
    def is_open(self) -> bool:
        return self.client.is_connected

