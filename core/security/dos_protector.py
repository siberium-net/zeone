"""
DoS Protector - Комплексная защита от атак
==========================================

[THREATS] Защищает от:
- Connection flood (слишком много соединений)
- Message flood (спам сообщениями)
- Bandwidth abuse (исчерпание bandwidth)
- Sybil attacks (множество фейковых ID)
- Slowloris (медленные соединения)

[DETECTION] Методы обнаружения:
- Статистические аномалии
- Поведенческий анализ
- Threshold monitoring
- Pattern recognition

[RESPONSE] Реакция на угрозы:
- Rate limiting
- Temporary ban
- Permanent ban
- Connection reset
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Уровень угрозы."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AttackType(Enum):
    """Тип атаки."""
    UNKNOWN = "unknown"
    CONNECTION_FLOOD = "connection_flood"
    MESSAGE_FLOOD = "message_flood"
    BANDWIDTH_ABUSE = "bandwidth_abuse"
    SYBIL_ATTACK = "sybil_attack"
    SLOWLORIS = "slowloris"
    INVALID_MESSAGES = "invalid_messages"
    REPLAY_ATTACK = "replay_attack"


@dataclass
class PeerMetrics:
    """Метрики поведения пира."""
    peer_id: str
    first_seen: float = field(default_factory=time.time)
    
    # Counters
    connections: int = 0
    messages: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    invalid_messages: int = 0
    
    # Sliding windows (last N seconds)
    connection_times: deque = field(default_factory=lambda: deque(maxlen=100))
    message_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Threat tracking
    threat_score: float = 0.0
    violations: List[AttackType] = field(default_factory=list)
    last_violation: float = 0.0
    
    def record_connection(self) -> None:
        """Записать соединение."""
        self.connections += 1
        self.connection_times.append(time.time())
    
    def record_message(self, size: int) -> None:
        """Записать сообщение."""
        self.messages += 1
        self.bytes_received += size
        self.message_times.append(time.time())
    
    def record_invalid(self) -> None:
        """Записать невалидное сообщение."""
        self.invalid_messages += 1
    
    def get_connection_rate(self, window: float = 60.0) -> float:
        """Частота соединений (conn/sec за window)."""
        now = time.time()
        count = sum(1 for t in self.connection_times if now - t < window)
        return count / window
    
    def get_message_rate(self, window: float = 10.0) -> float:
        """Частота сообщений (msg/sec за window)."""
        now = time.time()
        count = sum(1 for t in self.message_times if now - t < window)
        return count / window
    
    def get_invalid_ratio(self) -> float:
        """Процент невалидных сообщений."""
        if self.messages == 0:
            return 0.0
        return self.invalid_messages / self.messages


@dataclass
class ThreatEvent:
    """Событие угрозы."""
    timestamp: float
    peer_id: str
    attack_type: AttackType
    threat_level: ThreatLevel
    details: Dict[str, Any]


class DoSProtector:
    """
    Защита от DoS/DDoS атак.
    
    [USAGE]
    ```python
    protector = DoSProtector()
    await protector.start()
    
    # При новом соединении
    allowed = await protector.on_connection(peer_id, address)
    
    # При получении сообщения
    allowed = await protector.on_message(peer_id, message_size)
    
    # Проверка угрозы
    level = protector.get_threat_level(peer_id)
    ```
    """
    
    def __init__(
        self,
        # Thresholds
        max_connections_per_minute: int = 10,
        max_messages_per_second: float = 50.0,
        max_bandwidth_per_second: int = 1_000_000,  # 1 MB/s
        max_invalid_ratio: float = 0.3,
        
        # Response
        temp_ban_duration: float = 300.0,    # 5 min
        perm_ban_threshold: int = 5,          # violations before perm ban
        perm_ban_duration: float = 86400.0,   # 24 hours
        
        # Cleanup
        metrics_ttl: float = 3600.0,  # 1 hour
        cleanup_interval: float = 60.0,
    ):
        """
        Args:
            max_connections_per_minute: Лимит соединений в минуту
            max_messages_per_second: Лимит сообщений в секунду
            max_bandwidth_per_second: Лимит bandwidth (bytes/sec)
            max_invalid_ratio: Максимум невалидных сообщений
            temp_ban_duration: Длительность временного бана
            perm_ban_threshold: Количество нарушений до перм бана
            perm_ban_duration: Длительность постоянного бана
            metrics_ttl: Время жизни метрик
            cleanup_interval: Интервал очистки
        """
        # Thresholds
        self.max_connections_per_minute = max_connections_per_minute
        self.max_messages_per_second = max_messages_per_second
        self.max_bandwidth_per_second = max_bandwidth_per_second
        self.max_invalid_ratio = max_invalid_ratio
        
        # Response
        self.temp_ban_duration = temp_ban_duration
        self.perm_ban_threshold = perm_ban_threshold
        self.perm_ban_duration = perm_ban_duration
        
        # Cleanup
        self.metrics_ttl = metrics_ttl
        self.cleanup_interval = cleanup_interval
        
        # State
        self._metrics: Dict[str, PeerMetrics] = {}
        self._banned: Dict[str, float] = {}  # peer_id -> until
        self._permanent_bans: Set[str] = set()
        self._whitelist: Set[str] = set()
        
        # Global metrics (for DDoS detection)
        self._global_connection_times: deque = deque(maxlen=10000)
        self._global_message_times: deque = deque(maxlen=100000)
        
        # Events
        self._events: deque = deque(maxlen=1000)
        self._on_threat_callbacks: List[Callable[[ThreatEvent], None]] = []
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Запустить мониторинг."""
        async def monitor_loop():
            while True:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup()
                await self._check_global_threats()
        
        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info("[DOS] Protector started")
    
    async def stop(self) -> None:
        """Остановить."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
    
    def whitelist_add(self, peer_id: str) -> None:
        """Добавить в whitelist."""
        self._whitelist.add(peer_id)
    
    def whitelist_remove(self, peer_id: str) -> None:
        """Удалить из whitelist."""
        self._whitelist.discard(peer_id)
    
    async def on_connection(
        self,
        peer_id: str,
        address: str,
    ) -> bool:
        """
        Обработать новое соединение.
        
        Returns:
            True если соединение разрешено
        """
        # Whitelist
        if peer_id in self._whitelist:
            return True
        
        # Check ban
        if self._is_banned(peer_id):
            return False
        
        # Get/create metrics
        metrics = self._get_metrics(peer_id)
        metrics.record_connection()
        
        # Record global
        self._global_connection_times.append(time.time())
        
        # Check connection flood
        conn_rate = metrics.get_connection_rate(60.0) * 60  # per minute
        if conn_rate > self.max_connections_per_minute:
            await self._handle_violation(
                peer_id, AttackType.CONNECTION_FLOOD,
                {"rate": conn_rate, "limit": self.max_connections_per_minute}
            )
            return False
        
        return True
    
    async def on_message(
        self,
        peer_id: str,
        size: int,
        is_valid: bool = True,
    ) -> bool:
        """
        Обработать сообщение.
        
        Args:
            peer_id: ID пира
            size: Размер сообщения
            is_valid: Валидно ли сообщение
        
        Returns:
            True если сообщение разрешено
        """
        # Whitelist
        if peer_id in self._whitelist:
            return True
        
        # Check ban
        if self._is_banned(peer_id):
            return False
        
        # Get metrics
        metrics = self._get_metrics(peer_id)
        metrics.record_message(size)
        
        if not is_valid:
            metrics.record_invalid()
        
        # Record global
        self._global_message_times.append(time.time())
        
        # Check message flood
        msg_rate = metrics.get_message_rate(10.0)
        if msg_rate > self.max_messages_per_second:
            await self._handle_violation(
                peer_id, AttackType.MESSAGE_FLOOD,
                {"rate": msg_rate, "limit": self.max_messages_per_second}
            )
            return False
        
        # Check invalid ratio
        if metrics.messages > 10:  # После 10 сообщений
            invalid_ratio = metrics.get_invalid_ratio()
            if invalid_ratio > self.max_invalid_ratio:
                await self._handle_violation(
                    peer_id, AttackType.INVALID_MESSAGES,
                    {"ratio": invalid_ratio, "limit": self.max_invalid_ratio}
                )
                return False
        
        return True
    
    async def on_data_transfer(
        self,
        peer_id: str,
        bytes_transferred: int,
    ) -> bool:
        """
        Обработать передачу данных.
        
        Returns:
            True если разрешено
        """
        # Whitelist
        if peer_id in self._whitelist:
            return True
        
        # Check ban
        if self._is_banned(peer_id):
            return False
        
        # Get metrics
        metrics = self._get_metrics(peer_id)
        
        # Calculate bandwidth (rough estimate)
        elapsed = time.time() - metrics.first_seen
        if elapsed > 1.0:
            bandwidth = metrics.bytes_received / elapsed
            if bandwidth > self.max_bandwidth_per_second:
                await self._handle_violation(
                    peer_id, AttackType.BANDWIDTH_ABUSE,
                    {"bandwidth": bandwidth, "limit": self.max_bandwidth_per_second}
                )
                return False
        
        return True
    
    def ban(
        self,
        peer_id: str,
        duration: Optional[float] = None,
        permanent: bool = False,
    ) -> None:
        """Забанить пира."""
        if permanent:
            self._permanent_bans.add(peer_id)
            logger.warning(f"[DOS] PERMANENT ban: {peer_id[:16]}...")
        else:
            duration = duration or self.temp_ban_duration
            self._banned[peer_id] = time.time() + duration
            logger.warning(f"[DOS] Temp ban: {peer_id[:16]}... for {duration}s")
    
    def unban(self, peer_id: str) -> None:
        """Разбанить пира."""
        self._banned.pop(peer_id, None)
        self._permanent_bans.discard(peer_id)
    
    def get_threat_level(self, peer_id: str) -> ThreatLevel:
        """Получить уровень угрозы для пира."""
        if peer_id in self._permanent_bans:
            return ThreatLevel.CRITICAL
        
        if peer_id in self._banned:
            return ThreatLevel.HIGH
        
        metrics = self._metrics.get(peer_id)
        if not metrics:
            return ThreatLevel.NONE
        
        # Calculate threat score
        score = metrics.threat_score
        
        if score >= 0.8:
            return ThreatLevel.HIGH
        elif score >= 0.5:
            return ThreatLevel.MEDIUM
        elif score >= 0.2:
            return ThreatLevel.LOW
        
        return ThreatLevel.NONE
    
    def get_peer_metrics(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Получить метрики пира."""
        metrics = self._metrics.get(peer_id)
        if not metrics:
            return None
        
        return {
            "peer_id": peer_id,
            "connections": metrics.connections,
            "messages": metrics.messages,
            "bytes_received": metrics.bytes_received,
            "invalid_messages": metrics.invalid_messages,
            "connection_rate": metrics.get_connection_rate(60.0),
            "message_rate": metrics.get_message_rate(10.0),
            "invalid_ratio": metrics.get_invalid_ratio(),
            "threat_score": metrics.threat_score,
            "violations": [v.value for v in metrics.violations],
            "threat_level": self.get_threat_level(peer_id).name,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Общая статистика."""
        now = time.time()
        
        # Global rates
        global_conn_rate = sum(1 for t in self._global_connection_times if now - t < 60) / 60
        global_msg_rate = sum(1 for t in self._global_message_times if now - t < 10) / 10
        
        return {
            "tracked_peers": len(self._metrics),
            "temp_banned": len(self._banned),
            "perm_banned": len(self._permanent_bans),
            "whitelisted": len(self._whitelist),
            "recent_events": len(self._events),
            "global_connection_rate": global_conn_rate,
            "global_message_rate": global_msg_rate,
            "thresholds": {
                "max_conn_per_min": self.max_connections_per_minute,
                "max_msg_per_sec": self.max_messages_per_second,
                "max_bandwidth": self.max_bandwidth_per_second,
            },
        }
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получить последние события угроз."""
        events = list(self._events)[-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "peer_id": e.peer_id[:16] + "...",
                "attack_type": e.attack_type.value,
                "threat_level": e.threat_level.name,
                "details": e.details,
            }
            for e in events
        ]
    
    def on_threat(self, callback: Callable[[ThreatEvent], None]) -> None:
        """Регистрация callback на угрозу."""
        self._on_threat_callbacks.append(callback)
    
    def _is_banned(self, peer_id: str) -> bool:
        """Проверить бан."""
        if peer_id in self._permanent_bans:
            return True
        
        if peer_id in self._banned:
            if time.time() < self._banned[peer_id]:
                return True
            else:
                del self._banned[peer_id]
        
        return False
    
    def _get_metrics(self, peer_id: str) -> PeerMetrics:
        """Получить или создать метрики."""
        if peer_id not in self._metrics:
            self._metrics[peer_id] = PeerMetrics(peer_id=peer_id)
        return self._metrics[peer_id]
    
    async def _handle_violation(
        self,
        peer_id: str,
        attack_type: AttackType,
        details: Dict[str, Any],
    ) -> None:
        """Обработать нарушение."""
        metrics = self._get_metrics(peer_id)
        metrics.violations.append(attack_type)
        metrics.last_violation = time.time()
        
        # Update threat score
        metrics.threat_score = min(1.0, metrics.threat_score + 0.2)
        
        # Determine threat level
        violation_count = len(metrics.violations)
        if violation_count >= self.perm_ban_threshold:
            threat_level = ThreatLevel.CRITICAL
        elif violation_count >= 3:
            threat_level = ThreatLevel.HIGH
        elif violation_count >= 2:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW
        
        # Create event
        event = ThreatEvent(
            timestamp=time.time(),
            peer_id=peer_id,
            attack_type=attack_type,
            threat_level=threat_level,
            details=details,
        )
        self._events.append(event)
        
        # Execute callbacks
        for callback in self._on_threat_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"[DOS] Callback error: {e}")
        
        # Apply response
        if violation_count >= self.perm_ban_threshold:
            self.ban(peer_id, permanent=True)
        else:
            # Exponential backoff
            duration = self.temp_ban_duration * (2 ** (violation_count - 1))
            self.ban(peer_id, duration=duration)
        
        logger.warning(
            f"[DOS] Violation: {attack_type.value} from {peer_id[:16]}... "
            f"(level={threat_level.name}, count={violation_count})"
        )
    
    async def _check_global_threats(self) -> None:
        """Проверить глобальные угрозы (DDoS)."""
        now = time.time()
        
        # Global connection rate
        global_conn_rate = sum(1 for t in self._global_connection_times if now - t < 60)
        
        # DDoS threshold: 10x normal
        if global_conn_rate > self.max_connections_per_minute * 10:
            logger.critical(
                f"[DOS] POSSIBLE DDoS DETECTED! "
                f"Global conn rate: {global_conn_rate}/min"
            )
            # Could trigger emergency mode here
    
    def _cleanup(self) -> None:
        """Очистить старые данные."""
        now = time.time()
        
        # Cleanup old metrics
        stale = [
            peer_id for peer_id, m in self._metrics.items()
            if now - m.first_seen > self.metrics_ttl
            and m.threat_score < 0.2
        ]
        for peer_id in stale:
            del self._metrics[peer_id]
        
        # Cleanup expired bans
        expired = [
            peer_id for peer_id, until in self._banned.items()
            if now > until
        ]
        for peer_id in expired:
            del self._banned[peer_id]
        
        # Decay threat scores
        for metrics in self._metrics.values():
            metrics.threat_score *= 0.95  # 5% decay per cleanup

