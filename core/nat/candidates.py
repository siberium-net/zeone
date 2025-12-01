"""
ICE Candidates - Кандидаты для подключения
==========================================

[ICE] Interactive Connectivity Establishment (RFC 8445):
- Собираем все возможные способы подключения
- Обмениваемся candidates с пиром
- Проверяем каждую пару
- Выбираем лучшее соединение

[CANDIDATE TYPES]
1. Host: Локальный IP (192.168.x.x, 10.x.x.x)
2. Server Reflexive (srflx): Публичный IP через STUN
3. Peer Reflexive (prflx): Обнаружен во время проверки
4. Relay: Через relay сервер (гарантированный путь)

[PRIORITY]
priority = (2^24) * type_pref + (2^8) * local_pref + (2^0) * component_id
- Host: type_pref = 126
- Srflx: type_pref = 100
- Prflx: type_pref = 110
- Relay: type_pref = 0
"""

import asyncio
import socket
import logging
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any

from .stun import STUNClient, MappedAddress, NATType

logger = logging.getLogger(__name__)


class CandidateType(Enum):
    """Тип кандидата."""
    HOST = "host"           # Локальный адрес
    SRFLX = "srflx"         # Server Reflexive (через STUN)
    PRFLX = "prflx"         # Peer Reflexive (обнаружен при проверке)
    RELAY = "relay"         # Через relay сервер


class TransportType(Enum):
    """Тип транспорта."""
    UDP = "udp"
    TCP = "tcp"


# Type preferences for priority calculation
TYPE_PREFERENCES = {
    CandidateType.HOST: 126,
    CandidateType.PRFLX: 110,
    CandidateType.SRFLX: 100,
    CandidateType.RELAY: 0,
}


@dataclass
class Candidate:
    """
    ICE Candidate - один способ подключения к узлу.
    
    [ICE] Каждый candidate содержит:
    - foundation: Идентификатор для группировки
    - component: 1 для RTP (у нас всегда 1)
    - transport: UDP или TCP
    - priority: Приоритет (чем выше, тем лучше)
    - ip/port: Адрес
    - type: host/srflx/prflx/relay
    - related_ip/port: Базовый адрес (для srflx/relay)
    """
    
    ip: str
    port: int
    type: CandidateType
    transport: TransportType = TransportType.UDP
    priority: int = 0
    foundation: str = ""
    component: int = 1
    related_ip: str = ""
    related_port: int = 0
    relay_server: str = ""  # Для relay candidates
    
    def __post_init__(self):
        if not self.foundation:
            self.foundation = self._generate_foundation()
        if self.priority == 0:
            self.priority = self._calculate_priority()
    
    def _generate_foundation(self) -> str:
        """Генерировать foundation для группировки candidates."""
        data = f"{self.type.value}:{self.transport.value}:{self.related_ip or self.ip}"
        return hashlib.md5(data.encode()).hexdigest()[:8]
    
    def _calculate_priority(self) -> int:
        """
        Вычислить приоритет кандидата.
        
        [ICE] priority = (2^24) * type_pref + (2^8) * local_pref + (2^0) * (256 - component)
        """
        type_pref = TYPE_PREFERENCES.get(self.type, 0)
        local_pref = 65535  # Максимум для одного интерфейса
        
        # Для UDP приоритет выше чем для TCP
        if self.transport == TransportType.TCP:
            local_pref -= 1000
        
        return (2**24) * type_pref + (2**8) * local_pref + (256 - self.component)
    
    @property
    def address(self) -> Tuple[str, int]:
        """Адрес как кортеж."""
        return (self.ip, self.port)
    
    @property
    def is_public(self) -> bool:
        """Проверить, публичный ли IP."""
        if not self.ip:
            return False
        
        parts = self.ip.split(".")
        if len(parts) != 4:
            return False
        
        first = int(parts[0])
        second = int(parts[1])
        
        # Приватные диапазоны
        if first == 10:
            return False
        if first == 172 and 16 <= second <= 31:
            return False
        if first == 192 and second == 168:
            return False
        if first == 127:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "ip": self.ip,
            "port": self.port,
            "type": self.type.value,
            "transport": self.transport.value,
            "priority": self.priority,
            "foundation": self.foundation,
            "component": self.component,
            "related_ip": self.related_ip,
            "related_port": self.related_port,
            "relay_server": self.relay_server,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        """Десериализация из словаря."""
        return cls(
            ip=data["ip"],
            port=data["port"],
            type=CandidateType(data["type"]),
            transport=TransportType(data.get("transport", "udp")),
            priority=data.get("priority", 0),
            foundation=data.get("foundation", ""),
            component=data.get("component", 1),
            related_ip=data.get("related_ip", ""),
            related_port=data.get("related_port", 0),
            relay_server=data.get("relay_server", ""),
        )
    
    def __str__(self) -> str:
        return f"{self.type.value}:{self.ip}:{self.port}/{self.transport.value}"
    
    def __hash__(self) -> int:
        return hash((self.ip, self.port, self.transport.value))
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Candidate):
            return (self.ip, self.port, self.transport) == (other.ip, other.port, other.transport)
        return False


@dataclass
class CandidatePair:
    """
    Пара кандидатов (локальный + удалённый).
    
    [ICE] Connectivity Check проверяет пары:
    - Отправляем STUN Binding Request
    - Получаем ответ = пара работает
    """
    
    local: Candidate
    remote: Candidate
    priority: int = 0
    state: str = "waiting"  # waiting, in_progress, succeeded, failed, frozen
    nominated: bool = False
    
    def __post_init__(self):
        if self.priority == 0:
            self.priority = self._calculate_pair_priority()
    
    def _calculate_pair_priority(self) -> int:
        """
        Вычислить приоритет пары.
        
        [ICE] pair_priority = 2^32 * MIN(G,D) + 2 * MAX(G,D) + (G>D ? 1 : 0)
        G = controlling agent priority, D = controlled agent priority
        """
        g = self.local.priority
        d = self.remote.priority
        return 2**32 * min(g, d) + 2 * max(g, d) + (1 if g > d else 0)
    
    @property
    def id(self) -> str:
        """Уникальный ID пары."""
        return f"{self.local.foundation}:{self.remote.foundation}"
    
    def to_dict(self) -> Dict:
        return {
            "local": self.local.to_dict(),
            "remote": self.remote.to_dict(),
            "priority": self.priority,
            "state": self.state,
            "nominated": self.nominated,
        }


class CandidateGatherer:
    """
    Собирает все возможные candidates для подключения.
    
    [ICE] Gathering process:
    1. Host candidates: все локальные IP
    2. Server reflexive: через STUN
    3. Relay: через relay серверы
    
    [USAGE]
    ```python
    gatherer = CandidateGatherer(local_port=8468)
    candidates = await gatherer.gather()
    for c in candidates:
        print(f"{c.type.value}: {c.ip}:{c.port}")
    ```
    """
    
    def __init__(
        self,
        local_port: int,
        stun_client: Optional[STUNClient] = None,
        relay_servers: Optional[List[Tuple[str, int]]] = None,
        gather_tcp: bool = True,
    ):
        """
        Args:
            local_port: Порт для привязки
            stun_client: STUN клиент (создаётся автоматически если None)
            relay_servers: Список relay серверов [(host, port), ...]
            gather_tcp: Собирать TCP candidates
        """
        self.local_port = local_port
        self.stun_client = stun_client or STUNClient()
        self.relay_servers = relay_servers or []
        self.gather_tcp = gather_tcp
        
        self._candidates: List[Candidate] = []
        self._gathering = False
    
    async def gather(self) -> List[Candidate]:
        """
        Собрать все candidates.
        
        Returns:
            Список candidates, отсортированный по приоритету
        """
        if self._gathering:
            return self._candidates
        
        self._gathering = True
        self._candidates = []
        
        # Собираем параллельно
        tasks = [
            self._gather_host_candidates(),
            self._gather_srflx_candidates(),
        ]
        
        if self.relay_servers:
            tasks.append(self._gather_relay_candidates())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Сортируем по приоритету (убывание)
        self._candidates.sort(key=lambda c: c.priority, reverse=True)
        
        self._gathering = False
        
        logger.info(f"[ICE] Gathered {len(self._candidates)} candidates")
        for c in self._candidates:
            logger.debug(f"[ICE]   {c}")
        
        return self._candidates
    
    async def _gather_host_candidates(self) -> None:
        """Собрать host candidates (локальные IP)."""
        local_ips = self._get_local_ips()
        
        for ip in local_ips:
            # UDP candidate
            self._candidates.append(Candidate(
                ip=ip,
                port=self.local_port,
                type=CandidateType.HOST,
                transport=TransportType.UDP,
            ))
            
            # TCP candidate
            if self.gather_tcp:
                self._candidates.append(Candidate(
                    ip=ip,
                    port=self.local_port,
                    type=CandidateType.HOST,
                    transport=TransportType.TCP,
                ))
    
    async def _gather_srflx_candidates(self) -> None:
        """Собрать server reflexive candidates через STUN."""
        try:
            mapped = await self.stun_client.get_mapped_address(self.local_port)
            
            if mapped and mapped.is_public:
                # UDP srflx
                self._candidates.append(Candidate(
                    ip=mapped.ip,
                    port=mapped.port,
                    type=CandidateType.SRFLX,
                    transport=TransportType.UDP,
                    related_ip=mapped.local_ip,
                    related_port=mapped.local_port,
                ))
                
                # TCP srflx (если порт тот же)
                if self.gather_tcp:
                    self._candidates.append(Candidate(
                        ip=mapped.ip,
                        port=mapped.port,
                        type=CandidateType.SRFLX,
                        transport=TransportType.TCP,
                        related_ip=mapped.local_ip,
                        related_port=mapped.local_port,
                    ))
                    
        except Exception as e:
            logger.warning(f"[ICE] Failed to gather srflx: {e}")
    
    async def _gather_relay_candidates(self) -> None:
        """Собрать relay candidates."""
        for relay_host, relay_port in self.relay_servers:
            try:
                # Здесь будет запрос к relay серверу
                # Пока добавляем placeholder
                self._candidates.append(Candidate(
                    ip=relay_host,
                    port=relay_port,
                    type=CandidateType.RELAY,
                    transport=TransportType.UDP,
                    related_ip="",
                    related_port=0,
                    relay_server=f"{relay_host}:{relay_port}",
                ))
                
            except Exception as e:
                logger.warning(f"[ICE] Failed to gather relay from {relay_host}: {e}")
    
    def _get_local_ips(self) -> List[str]:
        """Получить все локальные IP адреса."""
        ips = []
        
        try:
            # Получаем все интерфейсы
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if ip not in ips and not ip.startswith("127."):
                    ips.append(ip)
        except Exception:
            pass
        
        # Fallback: получаем основной IP
        if not ips:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                if ip not in ips:
                    ips.append(ip)
            except Exception:
                pass
        
        # Последний fallback
        if not ips:
            ips.append("127.0.0.1")
        
        return ips
    
    @property
    def candidates(self) -> List[Candidate]:
        """Получить собранные candidates."""
        return self._candidates


def prioritize_candidate_pairs(
    local_candidates: List[Candidate],
    remote_candidates: List[Candidate],
) -> List[CandidatePair]:
    """
    Создать и отсортировать пары кандидатов.
    
    [ICE] Pairing rules:
    - Только совместимые транспорты (UDP-UDP, TCP-TCP)
    - Приоритет: direct > hole_punch > relay
    
    Args:
        local_candidates: Наши candidates
        remote_candidates: Candidates пира
    
    Returns:
        Список CandidatePair, отсортированный по приоритету
    """
    pairs = []
    
    for local in local_candidates:
        for remote in remote_candidates:
            # Только совместимые транспорты
            if local.transport != remote.transport:
                continue
            
            pair = CandidatePair(local=local, remote=remote)
            pairs.append(pair)
    
    # Сортируем по приоритету (убывание)
    pairs.sort(key=lambda p: p.priority, reverse=True)
    
    return pairs

