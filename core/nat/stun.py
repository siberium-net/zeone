"""
STUN Client - Session Traversal Utilities for NAT
=================================================

[STUN] RFC 5389 - Обнаружение публичного адреса:
- Отправляем Binding Request на STUN сервер
- Получаем MAPPED-ADDRESS (наш публичный IP:port)
- Определяем тип NAT

[NAT TYPES]
- Full Cone: Лучший случай, hole punch почти всегда работает
- Restricted Cone: Нужно сначала отправить пакет
- Port Restricted: Нужно отправить на конкретный порт
- Symmetric: Самый сложный, часто нужен relay

[PUBLIC STUN SERVERS]
- stun.l.google.com:19302
- stun.stunprotocol.org:3478
- stun.cloudflare.com:3478
"""

import asyncio
import struct
import os
import socket
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


# STUN Message Types
STUN_BINDING_REQUEST = 0x0001
STUN_BINDING_RESPONSE = 0x0101
STUN_BINDING_ERROR = 0x0111

# STUN Attributes
ATTR_MAPPED_ADDRESS = 0x0001
ATTR_XOR_MAPPED_ADDRESS = 0x0020
ATTR_ERROR_CODE = 0x0009
ATTR_SOFTWARE = 0x8022

# STUN Magic Cookie (RFC 5389)
STUN_MAGIC_COOKIE = 0x2112A442

# Default STUN servers
DEFAULT_STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
    ("stun.stunprotocol.org", 3478),
]

# Timeouts
STUN_TIMEOUT = 3.0  # seconds
STUN_RETRIES = 2


class NATType(Enum):
    """Тип NAT."""
    UNKNOWN = auto()
    OPEN = auto()           # Публичный IP, нет NAT
    FULL_CONE = auto()      # Любой может подключиться
    RESTRICTED_CONE = auto()  # Только после нашего пакета
    PORT_RESTRICTED = auto()  # Только после пакета на конкретный порт
    SYMMETRIC = auto()      # Разный mapping для каждого destination
    BLOCKED = auto()        # UDP заблокирован


@dataclass
class MappedAddress:
    """
    Публичный адрес, полученный через STUN.
    
    [STUN] XOR-MAPPED-ADDRESS содержит:
    - family: IPv4 (0x01) или IPv6 (0x02)
    - port: XOR с magic cookie
    - address: XOR с magic cookie (и transaction id для IPv6)
    """
    ip: str
    port: int
    nat_type: NATType = NATType.UNKNOWN
    local_ip: str = ""
    local_port: int = 0
    
    @property
    def is_public(self) -> bool:
        """Проверить, является ли IP публичным."""
        if not self.ip:
            return False
        
        # Приватные диапазоны
        parts = self.ip.split(".")
        if len(parts) != 4:
            return False
        
        first = int(parts[0])
        second = int(parts[1])
        
        # 10.x.x.x
        if first == 10:
            return False
        # 172.16.x.x - 172.31.x.x
        if first == 172 and 16 <= second <= 31:
            return False
        # 192.168.x.x
        if first == 192 and second == 168:
            return False
        # 127.x.x.x (loopback)
        if first == 127:
            return False
        
        return True
    
    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "port": self.port,
            "nat_type": self.nat_type.name,
            "local_ip": self.local_ip,
            "local_port": self.local_port,
            "is_public": self.is_public,
        }


class STUNClient:
    """
    STUN Client для обнаружения публичного адреса.
    
    [USAGE]
    ```python
    client = STUNClient()
    mapped = await client.get_mapped_address()
    print(f"Public: {mapped.ip}:{mapped.port}")
    print(f"NAT Type: {mapped.nat_type.name}")
    ```
    """
    
    def __init__(
        self,
        stun_servers: Optional[List[Tuple[str, int]]] = None,
        local_port: int = 0,
    ):
        """
        Args:
            stun_servers: Список STUN серверов [(host, port), ...]
            local_port: Локальный порт для привязки (0 = случайный)
        """
        self.stun_servers = stun_servers or DEFAULT_STUN_SERVERS
        self.local_port = local_port
        self._socket: Optional[socket.socket] = None
    
    async def get_mapped_address(
        self,
        local_port: Optional[int] = None,
    ) -> Optional[MappedAddress]:
        """
        Получить публичный адрес через STUN.
        
        Args:
            local_port: Порт для привязки (или self.local_port)
        
        Returns:
            MappedAddress или None если не удалось
        """
        port = local_port or self.local_port
        
        # Пробуем STUN серверы по очереди
        for stun_host, stun_port in self.stun_servers:
            try:
                result = await self._query_stun(stun_host, stun_port, port)
                if result:
                    logger.info(
                        f"[STUN] Mapped address: {result.ip}:{result.port} "
                        f"(via {stun_host})"
                    )
                    return result
            except Exception as e:
                logger.debug(f"[STUN] {stun_host}:{stun_port} failed: {e}")
                continue
        
        logger.warning("[STUN] All servers failed")
        return None
    
    async def detect_nat_type(self, local_port: int = 0) -> NATType:
        """
        Определить тип NAT.
        
        [ALGORITHM]
        1. Запрос к серверу A -> получаем mapped address
        2. Запрос к серверу B -> если mapped address отличается = Symmetric
        3. Проверка на Full/Restricted Cone требует сервера с двумя IP
        
        Упрощённая версия: определяем Open/Symmetric/Unknown
        """
        # Первый запрос
        result1 = await self._query_stun(
            self.stun_servers[0][0],
            self.stun_servers[0][1],
            local_port,
        )
        
        if not result1:
            return NATType.BLOCKED
        
        # Проверяем, публичный ли наш IP
        if not result1.is_public:
            # Возможно локальная сеть без NAT
            pass
        
        # Запрос к другому серверу с того же порта
        if len(self.stun_servers) > 1:
            result2 = await self._query_stun(
                self.stun_servers[1][0],
                self.stun_servers[1][1],
                local_port,
            )
            
            if result2:
                if result1.ip != result2.ip or result1.port != result2.port:
                    # Разный mapping = Symmetric NAT
                    return NATType.SYMMETRIC
        
        # Проверяем, совпадает ли mapped port с local port
        if result1.local_port == result1.port:
            # Возможно Open или Full Cone
            return NATType.FULL_CONE
        
        # Не можем точно определить без специального STUN сервера
        return NATType.PORT_RESTRICTED
    
    async def _query_stun(
        self,
        stun_host: str,
        stun_port: int,
        local_port: int = 0,
    ) -> Optional[MappedAddress]:
        """
        Отправить STUN Binding Request и получить ответ.
        """
        loop = asyncio.get_event_loop()
        
        # Создаём UDP сокет
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        
        try:
            # Привязываем к локальному порту
            sock.bind(("0.0.0.0", local_port))
            actual_local_port = sock.getsockname()[1]
            local_ip = self._get_local_ip()
            
            # Формируем STUN Binding Request
            transaction_id = os.urandom(12)
            request = self._build_binding_request(transaction_id)
            
            # Резолвим адрес STUN сервера
            stun_addr = await loop.run_in_executor(
                None,
                lambda: socket.getaddrinfo(stun_host, stun_port, socket.AF_INET)[0][4]
            )
            
            # Отправляем с ретраями
            for attempt in range(STUN_RETRIES + 1):
                await loop.sock_sendto(sock, request, stun_addr)
                
                try:
                    # Ждём ответ с таймаутом
                    data = await asyncio.wait_for(
                        loop.sock_recv(sock, 1024),
                        timeout=STUN_TIMEOUT,
                    )
                    
                    # Парсим ответ
                    mapped = self._parse_binding_response(data, transaction_id)
                    if mapped:
                        mapped.local_ip = local_ip
                        mapped.local_port = actual_local_port
                        return mapped
                        
                except asyncio.TimeoutError:
                    if attempt < STUN_RETRIES:
                        logger.debug(f"[STUN] Retry {attempt + 1}/{STUN_RETRIES}")
                    continue
            
            return None
            
        finally:
            sock.close()
    
    def _build_binding_request(self, transaction_id: bytes) -> bytes:
        """
        Построить STUN Binding Request.
        
        [FORMAT]
        0                   1                   2                   3
        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |0 0|     STUN Message Type     |         Message Length        |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                         Magic Cookie                          |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                                                               |
        |                     Transaction ID (96 bits)                  |
        |                                                               |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        """
        # Header: type (2) + length (2) + magic cookie (4) + transaction id (12)
        header = struct.pack(
            ">HHI",
            STUN_BINDING_REQUEST,
            0,  # No attributes
            STUN_MAGIC_COOKIE,
        )
        
        return header + transaction_id
    
    def _parse_binding_response(
        self,
        data: bytes,
        expected_transaction_id: bytes,
    ) -> Optional[MappedAddress]:
        """
        Парсить STUN Binding Response.
        """
        if len(data) < 20:
            return None
        
        # Parse header
        msg_type, msg_length, magic_cookie = struct.unpack(">HHI", data[:8])
        transaction_id = data[8:20]
        
        # Verify
        if msg_type != STUN_BINDING_RESPONSE:
            logger.debug(f"[STUN] Unexpected message type: 0x{msg_type:04x}")
            return None
        
        if magic_cookie != STUN_MAGIC_COOKIE:
            logger.debug(f"[STUN] Invalid magic cookie: 0x{magic_cookie:08x}")
            return None
        
        if transaction_id != expected_transaction_id:
            logger.debug("[STUN] Transaction ID mismatch")
            return None
        
        # Parse attributes
        offset = 20
        mapped_ip = None
        mapped_port = None
        
        while offset < len(data):
            if offset + 4 > len(data):
                break
            
            attr_type, attr_length = struct.unpack(">HH", data[offset:offset + 4])
            offset += 4
            
            if offset + attr_length > len(data):
                break
            
            attr_value = data[offset:offset + attr_length]
            
            if attr_type == ATTR_XOR_MAPPED_ADDRESS:
                # XOR-MAPPED-ADDRESS (preferred)
                mapped_ip, mapped_port = self._parse_xor_mapped_address(
                    attr_value, transaction_id
                )
            elif attr_type == ATTR_MAPPED_ADDRESS and not mapped_ip:
                # MAPPED-ADDRESS (fallback)
                mapped_ip, mapped_port = self._parse_mapped_address(attr_value)
            
            # Align to 4 bytes
            offset += attr_length
            if attr_length % 4:
                offset += 4 - (attr_length % 4)
        
        if mapped_ip and mapped_port:
            return MappedAddress(ip=mapped_ip, port=mapped_port)
        
        return None
    
    def _parse_xor_mapped_address(
        self,
        data: bytes,
        transaction_id: bytes,
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Парсить XOR-MAPPED-ADDRESS атрибут.
        
        [FORMAT]
        0                   1                   2                   3
        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |x x x x x x x x|    Family     |         X-Port                |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                X-Address (Variable)                           |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        """
        if len(data) < 8:
            return None, None
        
        family = data[1]
        xor_port = struct.unpack(">H", data[2:4])[0]
        
        # XOR with magic cookie
        port = xor_port ^ (STUN_MAGIC_COOKIE >> 16)
        
        if family == 0x01:  # IPv4
            xor_addr = struct.unpack(">I", data[4:8])[0]
            addr = xor_addr ^ STUN_MAGIC_COOKIE
            ip = socket.inet_ntoa(struct.pack(">I", addr))
            return ip, port
        
        # IPv6 not implemented
        return None, None
    
    def _parse_mapped_address(self, data: bytes) -> Tuple[Optional[str], Optional[int]]:
        """
        Парсить MAPPED-ADDRESS атрибут (legacy).
        """
        if len(data) < 8:
            return None, None
        
        family = data[1]
        port = struct.unpack(">H", data[2:4])[0]
        
        if family == 0x01:  # IPv4
            ip = socket.inet_ntoa(data[4:8])
            return ip, port
        
        return None, None
    
    def _get_local_ip(self) -> str:
        """Получить локальный IP."""
        try:
            # Создаём сокет и подключаемся к внешнему адресу
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


async def get_public_address(local_port: int = 0) -> Optional[MappedAddress]:
    """
    Удобная функция для получения публичного адреса.
    
    Args:
        local_port: Порт для привязки
    
    Returns:
        MappedAddress или None
    """
    client = STUNClient()
    return await client.get_mapped_address(local_port)

