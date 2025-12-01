"""
Peer Store - Хранение информации о пирах
========================================

[PERSISTENCE] Специализированное хранилище для:
- Информация о пирах
- История соединений
- Статистика доступности
- Кэширование для быстрого доступа

[FEATURES]
- LRU кэш в памяти
- Периодическая синхронизация с диском
- Умное управление capacity
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Callable, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Детальная информация о пире."""
    node_id: str
    host: str
    port: int
    
    # Временные метки
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Статус
    is_connected: bool = False
    is_reachable: bool = True
    is_banned: bool = False
    ban_until: float = 0.0
    ban_reason: str = ""
    
    # Статистика
    ping_count: int = 0
    pong_count: int = 0
    avg_latency_ms: float = 0.0
    
    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    protocol_version: str = "1.0"
    user_agent: str = ""
    
    def update_latency(self, latency_ms: float) -> None:
        """Обновить среднюю задержку (EWMA)."""
        alpha = 0.3
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
    
    @property
    def is_available(self) -> bool:
        """Пир доступен для соединения."""
        if self.is_banned:
            if time.time() > self.ban_until:
                self.is_banned = False
            else:
                return False
        return self.is_reachable
    
    @property
    def reliability_score(self) -> float:
        """Оценка надёжности (0-1)."""
        if self.ping_count == 0:
            return 0.5
        return min(1.0, self.pong_count / self.ping_count)


class PeerStore:
    """
    Кэшированное хранилище информации о пирах.
    
    [USAGE]
    ```python
    store = PeerStore(max_peers=1000)
    
    # Добавить/обновить пира
    store.upsert(PeerInfo(node_id="abc", host="1.2.3.4", port=8000))
    
    # Получить пира
    peer = store.get("abc")
    
    # Получить лучших пиров
    best = store.get_best_peers(limit=10)
    
    # Забанить пира
    store.ban_peer("abc", duration=3600, reason="DoS attack")
    ```
    """
    
    def __init__(
        self,
        max_peers: int = 1000,
        cleanup_interval: float = 300.0,
        stale_threshold: float = 3600.0,
    ):
        """
        Args:
            max_peers: Максимум пиров в кэше
            cleanup_interval: Интервал очистки устаревших
            stale_threshold: Порог устаревания (секунды)
        """
        self.max_peers = max_peers
        self.cleanup_interval = cleanup_interval
        self.stale_threshold = stale_threshold
        
        # LRU-ordered dict
        self._peers: OrderedDict[str, PeerInfo] = OrderedDict()
        self._connected: Set[str] = set()
        self._banned: Set[str] = set()
        
        # Callbacks
        self._on_peer_added: List[Callable[[PeerInfo], None]] = []
        self._on_peer_removed: List[Callable[[str], None]] = []
        self._on_peer_banned: List[Callable[[str, str], None]] = []
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def upsert(self, peer: PeerInfo) -> None:
        """Добавить или обновить пира."""
        existing = self._peers.get(peer.node_id)
        
        if existing:
            # Обновляем существующего
            existing.host = peer.host
            existing.port = peer.port
            existing.last_seen = time.time()
            existing.capabilities.update(peer.capabilities)
            if peer.protocol_version:
                existing.protocol_version = peer.protocol_version
            if peer.user_agent:
                existing.user_agent = peer.user_agent
            # Move to end (most recent)
            self._peers.move_to_end(peer.node_id)
        else:
            # Добавляем нового
            self._peers[peer.node_id] = peer
            for callback in self._on_peer_added:
                callback(peer)
        
        # Проверяем capacity
        self._enforce_capacity()
    
    def get(self, node_id: str) -> Optional[PeerInfo]:
        """Получить пира по ID."""
        peer = self._peers.get(node_id)
        if peer:
            self._peers.move_to_end(node_id)  # LRU update
        return peer
    
    def remove(self, node_id: str) -> bool:
        """Удалить пира."""
        if node_id in self._peers:
            del self._peers[node_id]
            self._connected.discard(node_id)
            for callback in self._on_peer_removed:
                callback(node_id)
            return True
        return False
    
    def mark_connected(self, node_id: str) -> None:
        """Отметить пира как подключённого."""
        peer = self.get(node_id)
        if peer:
            peer.is_connected = True
            peer.last_activity = time.time()
            self._connected.add(node_id)
    
    def mark_disconnected(self, node_id: str) -> None:
        """Отметить пира как отключённого."""
        peer = self.get(node_id)
        if peer:
            peer.is_connected = False
            self._connected.discard(node_id)
    
    def mark_unreachable(self, node_id: str) -> None:
        """Отметить пира как недоступного."""
        peer = self.get(node_id)
        if peer:
            peer.is_reachable = False
    
    def ban_peer(
        self,
        node_id: str,
        duration: float = 3600.0,
        reason: str = "Unspecified",
    ) -> None:
        """
        Забанить пира.
        
        Args:
            node_id: ID пира
            duration: Длительность бана в секундах
            reason: Причина бана
        """
        peer = self.get(node_id)
        if peer:
            peer.is_banned = True
            peer.ban_until = time.time() + duration
            peer.ban_reason = reason
            self._banned.add(node_id)
            
            for callback in self._on_peer_banned:
                callback(node_id, reason)
            
            logger.warning(f"[PEERS] Banned {node_id[:16]}... for {duration}s: {reason}")
    
    def unban_peer(self, node_id: str) -> None:
        """Разбанить пира."""
        peer = self.get(node_id)
        if peer:
            peer.is_banned = False
            peer.ban_until = 0.0
            peer.ban_reason = ""
            self._banned.discard(node_id)
            logger.info(f"[PEERS] Unbanned {node_id[:16]}...")
    
    def is_banned(self, node_id: str) -> bool:
        """Проверить забанен ли пир."""
        peer = self.get(node_id)
        if not peer:
            return False
        return peer.is_banned and time.time() < peer.ban_until
    
    def get_connected(self) -> List[PeerInfo]:
        """Получить подключённых пиров."""
        return [
            self._peers[node_id]
            for node_id in self._connected
            if node_id in self._peers
        ]
    
    def get_best_peers(
        self,
        limit: int = 10,
        exclude: Optional[Set[str]] = None,
        require_capabilities: Optional[Set[str]] = None,
    ) -> List[PeerInfo]:
        """
        Получить лучших пиров по надёжности.
        
        Args:
            limit: Максимум пиров
            exclude: Исключить этих пиров
            require_capabilities: Требуемые capabilities
        
        Returns:
            Список отсортированный по reliability
        """
        exclude = exclude or set()
        require_capabilities = require_capabilities or set()
        
        candidates = []
        for peer in self._peers.values():
            if peer.node_id in exclude:
                continue
            if not peer.is_available:
                continue
            if require_capabilities and not require_capabilities.issubset(peer.capabilities):
                continue
            candidates.append(peer)
        
        # Сортировка: connected > reliable > recent
        candidates.sort(key=lambda p: (
            p.is_connected,
            p.reliability_score,
            -p.avg_latency_ms if p.avg_latency_ms > 0 else 0,
            p.last_seen,
        ), reverse=True)
        
        return candidates[:limit]
    
    def get_random_peers(
        self,
        count: int = 5,
        exclude: Optional[Set[str]] = None,
    ) -> List[PeerInfo]:
        """Получить случайных пиров (для gossip)."""
        import random
        
        exclude = exclude or set()
        candidates = [
            p for p in self._peers.values()
            if p.node_id not in exclude and p.is_available
        ]
        
        return random.sample(candidates, min(count, len(candidates)))
    
    def record_ping(self, node_id: str) -> None:
        """Записать отправку PING."""
        peer = self.get(node_id)
        if peer:
            peer.ping_count += 1
    
    def record_pong(self, node_id: str, latency_ms: float) -> None:
        """Записать получение PONG."""
        peer = self.get(node_id)
        if peer:
            peer.pong_count += 1
            peer.update_latency(latency_ms)
            peer.last_activity = time.time()
            peer.is_reachable = True
    
    def on_peer_added(self, callback: Callable[[PeerInfo], None]) -> None:
        """Регистрация callback на добавление пира."""
        self._on_peer_added.append(callback)
    
    def on_peer_removed(self, callback: Callable[[str], None]) -> None:
        """Регистрация callback на удаление пира."""
        self._on_peer_removed.append(callback)
    
    def on_peer_banned(self, callback: Callable[[str, str], None]) -> None:
        """Регистрация callback на бан пира."""
        self._on_peer_banned.append(callback)
    
    async def start_cleanup(self) -> None:
        """Запустить периодическую очистку."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup_stale()
                self._cleanup_expired_bans()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_cleanup(self) -> None:
        """Остановить очистку."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    def _enforce_capacity(self) -> None:
        """Удалить старых пиров если превышен лимит."""
        while len(self._peers) > self.max_peers:
            # Удаляем самого старого (начало OrderedDict)
            oldest_id, oldest_peer = next(iter(self._peers.items()))
            
            # Не удаляем подключённых
            if oldest_peer.is_connected:
                self._peers.move_to_end(oldest_id)
                continue
            
            self.remove(oldest_id)
    
    def _cleanup_stale(self) -> None:
        """Удалить устаревших пиров."""
        now = time.time()
        stale_ids = []
        
        for node_id, peer in self._peers.items():
            if peer.is_connected:
                continue
            if now - peer.last_seen > self.stale_threshold:
                stale_ids.append(node_id)
        
        for node_id in stale_ids:
            self.remove(node_id)
        
        if stale_ids:
            logger.debug(f"[PEERS] Cleaned up {len(stale_ids)} stale peers")
    
    def _cleanup_expired_bans(self) -> None:
        """Снять истекшие баны."""
        now = time.time()
        unbanned = []
        
        for node_id in list(self._banned):
            peer = self._peers.get(node_id)
            if peer and peer.is_banned and now > peer.ban_until:
                self.unban_peer(node_id)
                unbanned.append(node_id)
        
        if unbanned:
            logger.debug(f"[PEERS] Unbanned {len(unbanned)} peers (expired)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика хранилища."""
        return {
            "total": len(self._peers),
            "connected": len(self._connected),
            "banned": len(self._banned),
            "max_peers": self.max_peers,
            "available": sum(1 for p in self._peers.values() if p.is_available),
        }
    
    def __len__(self) -> int:
        return len(self._peers)
    
    def __contains__(self, node_id: str) -> bool:
        return node_id in self._peers
    
    def __iter__(self):
        return iter(self._peers.values())

