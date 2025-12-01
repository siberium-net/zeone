"""
Hole Punching - UDP и TCP hole punching
=======================================

[HOLE PUNCH] Принцип работы:
1. Оба узла одновременно отправляют пакеты друг другу
2. NAT создаёт mapping для исходящего пакета
3. Входящий пакет от пира проходит через созданный mapping
4. Соединение установлено!

[UDP HOLE PUNCH]
- Проще реализовать
- Работает с большинством NAT
- Connectionless - нужно отправлять keep-alive

[TCP HOLE PUNCH] (Simultaneous Open)
- Сложнее: оба должны одновременно сделать connect()
- Работает с меньшим количеством NAT
- Connection-oriented - надёжнее после установки

[LIMITATIONS]
- Symmetric NAT: часто не работает (разный mapping для каждого destination)
- Carrier-grade NAT (CGNAT): может не работать
- Firewalls: могут блокировать
"""

import asyncio
import socket
import struct
import os
import time
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


# Constants
PUNCH_TIMEOUT = 5.0  # seconds
PUNCH_INTERVAL = 0.1  # seconds between attempts
PUNCH_RETRIES = 50  # total attempts = PUNCH_TIMEOUT / PUNCH_INTERVAL
KEEP_ALIVE_INTERVAL = 15.0  # seconds


class HolePunchResult(Enum):
    """Результат hole punching."""
    SUCCESS = auto()
    TIMEOUT = auto()
    REFUSED = auto()
    ERROR = auto()


@dataclass
class PunchResult:
    """Результат попытки hole punch."""
    success: bool
    result: HolePunchResult
    socket: Optional[Any] = None  # Успешный сокет
    local_addr: Tuple[str, int] = ("", 0)
    remote_addr: Tuple[str, int] = ("", 0)
    latency_ms: float = 0
    error: str = ""


# Hole punch message magic
PUNCH_MAGIC = b"P2P_PUNCH"
PUNCH_SYN = b"P2P_PUNCH_SYN"
PUNCH_ACK = b"P2P_PUNCH_ACK"


class UDPHolePuncher:
    """
    UDP Hole Punching.
    
    [ALGORITHM]
    1. Оба узла знают публичные адреса друг друга (через STUN)
    2. Одновременно начинают отправлять UDP пакеты
    3. NAT создаёт mapping для исходящих пакетов
    4. Когда пакет от пира приходит, NAT пропускает его
    
    [USAGE]
    ```python
    puncher = UDPHolePuncher()
    result = await puncher.punch(
        local_port=8468,
        remote_ip="203.0.113.1",
        remote_port=54321,
    )
    if result.success:
        # Используем result.socket для отправки/получения
        pass
    ```
    """
    
    def __init__(self, node_id: str = ""):
        """
        Args:
            node_id: ID узла для идентификации в сообщениях
        """
        self.node_id = node_id
    
    async def punch(
        self,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        timeout: float = PUNCH_TIMEOUT,
    ) -> PunchResult:
        """
        Выполнить UDP hole punch.
        
        Args:
            local_port: Локальный порт
            remote_ip: Публичный IP пира
            remote_port: Публичный порт пира
            timeout: Таймаут в секундах
        
        Returns:
            PunchResult с результатом
        """
        loop = asyncio.get_event_loop()
        
        # Создаём UDP сокет
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(("0.0.0.0", local_port))
            local_addr = sock.getsockname()
            
            logger.info(
                f"[PUNCH] UDP starting: {local_addr} -> {remote_ip}:{remote_port}"
            )
            
            remote_addr = (remote_ip, remote_port)
            start_time = time.time()
            received_syn = False
            sent_ack = False
            
            # Формируем SYN сообщение
            syn_msg = PUNCH_SYN + self.node_id.encode()[:32]
            ack_msg = PUNCH_ACK + self.node_id.encode()[:32]
            
            while time.time() - start_time < timeout:
                # Отправляем SYN
                try:
                    await loop.sock_sendto(sock, syn_msg, remote_addr)
                except Exception as e:
                    logger.debug(f"[PUNCH] Send error: {e}")
                
                # Пробуем получить ответ
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(sock, 1024),
                        timeout=PUNCH_INTERVAL,
                    )
                    
                    if data.startswith(PUNCH_SYN):
                        logger.debug(f"[PUNCH] Received SYN from {addr}")
                        received_syn = True
                        # Отправляем ACK
                        await loop.sock_sendto(sock, ack_msg, addr)
                        
                    elif data.startswith(PUNCH_ACK):
                        logger.debug(f"[PUNCH] Received ACK from {addr}")
                        if received_syn:
                            # Успех!
                            latency = (time.time() - start_time) * 1000
                            logger.info(
                                f"[PUNCH] UDP success: {local_addr} <-> {addr} "
                                f"({latency:.1f}ms)"
                            )
                            return PunchResult(
                                success=True,
                                result=HolePunchResult.SUCCESS,
                                socket=sock,
                                local_addr=local_addr,
                                remote_addr=addr,
                                latency_ms=latency,
                            )
                        else:
                            # Получили ACK без SYN, отправляем свой SYN
                            received_syn = True
                            
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.debug(f"[PUNCH] Recv error: {e}")
            
            # Таймаут
            sock.close()
            logger.warning(f"[PUNCH] UDP timeout: {remote_ip}:{remote_port}")
            return PunchResult(
                success=False,
                result=HolePunchResult.TIMEOUT,
                error="Timeout waiting for response",
            )
            
        except Exception as e:
            sock.close()
            logger.error(f"[PUNCH] UDP error: {e}")
            return PunchResult(
                success=False,
                result=HolePunchResult.ERROR,
                error=str(e),
            )


class TCPHolePuncher:
    """
    TCP Hole Punching (Simultaneous Open).
    
    [ALGORITHM]
    1. Оба узла одновременно вызывают connect() друг к другу
    2. TCP SYN пакеты создают mappings в обоих NAT
    3. Когда SYN от пира приходит, происходит "simultaneous open"
    4. TCP соединение установлено!
    
    [CHALLENGES]
    - Timing критичен: оба должны сделать connect() почти одновременно
    - Многие NAT не поддерживают TCP simultaneous open
    - Fallback: если не работает, используем relay
    
    [USAGE]
    ```python
    puncher = TCPHolePuncher()
    result = await puncher.punch(
        local_port=8468,
        remote_ip="203.0.113.1",
        remote_port=54321,
    )
    ```
    """
    
    def __init__(self, node_id: str = ""):
        """
        Args:
            node_id: ID узла
        """
        self.node_id = node_id
    
    async def punch(
        self,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        timeout: float = PUNCH_TIMEOUT,
    ) -> PunchResult:
        """
        Выполнить TCP hole punch (simultaneous open).
        
        Args:
            local_port: Локальный порт
            remote_ip: Публичный IP пира
            remote_port: Публичный порт пира
            timeout: Таймаут в секундах
        
        Returns:
            PunchResult с результатом
        """
        logger.info(
            f"[PUNCH] TCP starting: port {local_port} -> {remote_ip}:{remote_port}"
        )
        
        start_time = time.time()
        
        # Пробуем несколько раз с разными сокетами
        for attempt in range(int(timeout / 0.5)):
            if time.time() - start_time >= timeout:
                break
            
            result = await self._try_connect(
                local_port, remote_ip, remote_port, 0.5
            )
            
            if result.success:
                return result
            
            # Небольшая пауза между попытками
            await asyncio.sleep(0.1)
        
        logger.warning(f"[PUNCH] TCP failed: {remote_ip}:{remote_port}")
        return PunchResult(
            success=False,
            result=HolePunchResult.TIMEOUT,
            error="TCP hole punch timeout",
        )
    
    async def _try_connect(
        self,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        timeout: float,
    ) -> PunchResult:
        """Одна попытка TCP connect."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # SO_REUSEPORT если доступен (Linux)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        
        try:
            sock.bind(("0.0.0.0", local_port))
            local_addr = sock.getsockname()
            
            loop = asyncio.get_event_loop()
            
            try:
                # Асинхронный connect с таймаутом
                await asyncio.wait_for(
                    loop.sock_connect(sock, (remote_ip, remote_port)),
                    timeout=timeout,
                )
                
                # Успех!
                latency = (time.time() - time.time()) * 1000
                logger.info(
                    f"[PUNCH] TCP success: {local_addr} <-> {remote_ip}:{remote_port}"
                )
                
                return PunchResult(
                    success=True,
                    result=HolePunchResult.SUCCESS,
                    socket=sock,
                    local_addr=local_addr,
                    remote_addr=(remote_ip, remote_port),
                    latency_ms=latency,
                )
                
            except asyncio.TimeoutError:
                sock.close()
                return PunchResult(
                    success=False,
                    result=HolePunchResult.TIMEOUT,
                )
            except ConnectionRefusedError:
                sock.close()
                return PunchResult(
                    success=False,
                    result=HolePunchResult.REFUSED,
                )
            except OSError as e:
                sock.close()
                # EINPROGRESS - это нормально для неблокирующего connect
                if e.errno == 115:  # EINPROGRESS
                    return PunchResult(
                        success=False,
                        result=HolePunchResult.TIMEOUT,
                    )
                return PunchResult(
                    success=False,
                    result=HolePunchResult.ERROR,
                    error=str(e),
                )
                
        except Exception as e:
            sock.close()
            return PunchResult(
                success=False,
                result=HolePunchResult.ERROR,
                error=str(e),
            )


class HolePuncher:
    """
    Комбинированный hole puncher (UDP + TCP fallback).
    
    [STRATEGY]
    1. Сначала пробуем UDP (быстрее, проще)
    2. Если UDP не работает, пробуем TCP
    3. Если оба не работают, используем relay
    
    [USAGE]
    ```python
    puncher = HolePuncher(node_id="abc123")
    result = await puncher.punch(
        local_port=8468,
        remote_ip="203.0.113.1",
        remote_port=54321,
    )
    
    if result.success:
        print(f"Connected via {result.transport}")
        # result.socket готов к использованию
    else:
        print(f"Failed: {result.error}")
        # Используем relay
    ```
    """
    
    def __init__(self, node_id: str = "", try_tcp: bool = True):
        """
        Args:
            node_id: ID узла
            try_tcp: Пробовать TCP если UDP не работает
        """
        self.node_id = node_id
        self.try_tcp = try_tcp
        self.udp_puncher = UDPHolePuncher(node_id)
        self.tcp_puncher = TCPHolePuncher(node_id)
    
    async def punch(
        self,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        udp_timeout: float = PUNCH_TIMEOUT,
        tcp_timeout: float = PUNCH_TIMEOUT,
    ) -> PunchResult:
        """
        Выполнить hole punch (UDP, затем TCP).
        
        Args:
            local_port: Локальный порт
            remote_ip: Публичный IP пира
            remote_port: Публичный порт пира
            udp_timeout: Таймаут для UDP
            tcp_timeout: Таймаут для TCP
        
        Returns:
            PunchResult с результатом
        """
        # Сначала UDP
        logger.info(f"[PUNCH] Trying UDP to {remote_ip}:{remote_port}")
        result = await self.udp_puncher.punch(
            local_port, remote_ip, remote_port, udp_timeout
        )
        
        if result.success:
            return result
        
        # Если UDP не работает и разрешён TCP
        if self.try_tcp:
            logger.info(f"[PUNCH] UDP failed, trying TCP")
            result = await self.tcp_puncher.punch(
                local_port, remote_ip, remote_port, tcp_timeout
            )
            
            if result.success:
                return result
        
        # Оба метода не сработали
        logger.warning(f"[PUNCH] Both UDP and TCP failed for {remote_ip}:{remote_port}")
        return PunchResult(
            success=False,
            result=HolePunchResult.TIMEOUT,
            error="Both UDP and TCP hole punch failed",
        )
    
    async def punch_with_coordination(
        self,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        start_time: float,
        timeout: float = PUNCH_TIMEOUT,
    ) -> PunchResult:
        """
        Hole punch с координацией времени начала.
        
        [SYNCHRONIZATION]
        Оба узла должны начать punch примерно одновременно.
        start_time = согласованное время начала (Unix timestamp)
        
        Args:
            local_port: Локальный порт
            remote_ip: Публичный IP пира
            remote_port: Публичный порт пира
            start_time: Согласованное время начала
            timeout: Таймаут
        
        Returns:
            PunchResult
        """
        # Ждём до start_time
        now = time.time()
        if start_time > now:
            wait_time = start_time - now
            logger.debug(f"[PUNCH] Waiting {wait_time:.2f}s until start_time")
            await asyncio.sleep(wait_time)
        
        return await self.punch(local_port, remote_ip, remote_port, timeout)


async def coordinated_punch(
    local_port: int,
    remote_ip: str,
    remote_port: int,
    node_id: str = "",
) -> PunchResult:
    """
    Удобная функция для hole punch.
    
    Args:
        local_port: Локальный порт
        remote_ip: Публичный IP пира
        remote_port: Публичный порт пира
        node_id: ID узла
    
    Returns:
        PunchResult
    """
    puncher = HolePuncher(node_id)
    return await puncher.punch(local_port, remote_ip, remote_port)

