"""
State Manager - Централизованное управление состоянием узла
===========================================================

[PERSISTENCE] Сохраняет и восстанавливает:
- Список известных пиров (с историей подключений)
- Состояние DHT (routing table, stored values)
- Балансы с пирами
- Настройки узла

[RECOVERY] Стратегии восстановления:
- Приоритет по последнему успешному соединению
- Фильтрация недоступных узлов
- Постепенное восстановление DHT

[STORAGE]
- SQLite для надёжности
- JSON backup для читаемости
- Atomic writes для консистентности
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Set
from pathlib import Path
from contextlib import suppress

logger = logging.getLogger(__name__)

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False


@dataclass
class PeerRecord:
    """Запись о пире для persistence."""
    node_id: str
    host: str
    port: int
    
    # Статистика
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_connected: float = 0.0
    connection_count: int = 0
    failed_connections: int = 0
    
    # Характеристики
    trust_score: float = 0.5
    is_bootstrap: bool = False
    supports_dht: bool = False
    supports_relay: bool = False
    
    # NAT информация
    nat_type: str = "unknown"
    public_ip: str = ""
    public_port: int = 0
    
    @property
    def success_rate(self) -> float:
        """Процент успешных подключений."""
        total = self.connection_count + self.failed_connections
        if total == 0:
            return 0.5
        return self.connection_count / total
    
    @property
    def priority_score(self) -> float:
        """Приоритет для reconnect (больше = лучше)."""
        recency = 1.0 / (1.0 + (time.time() - self.last_connected) / 3600)  # Decay за час
        return self.trust_score * self.success_rate * recency
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class NodeState:
    """Полное состояние узла для сохранения."""
    node_id: str
    host: str
    port: int
    
    # Время
    created_at: float = field(default_factory=time.time)
    last_saved: float = 0.0
    uptime_total: float = 0.0
    
    # Пиры
    peers: Dict[str, PeerRecord] = field(default_factory=dict)
    bootstrap_nodes: List[str] = field(default_factory=list)
    
    # DHT
    dht_node_id: str = ""
    dht_stored_keys: int = 0
    
    # Статистика
    total_connections: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['peers'] = {k: v.to_dict() for k, v in self.peers.items()}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeState":
        peers_data = data.pop('peers', {})
        state = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        state.peers = {k: PeerRecord.from_dict(v) for k, v in peers_data.items()}
        return state


class StateManager:
    """
    Менеджер состояния узла.
    
    [USAGE]
    ```python
    state_manager = StateManager("node_state.db")
    await state_manager.initialize()
    
    # Сохранить состояние
    await state_manager.save_state(node_state)
    
    # Восстановить состояние
    state = await state_manager.load_state()
    
    # Получить лучших пиров для reconnect
    peers = await state_manager.get_best_peers_for_reconnect(limit=10)
    ```
    """
    
    def __init__(
        self,
        db_path: str = "node_state.db",
        backup_path: Optional[str] = None,
        auto_save_interval: float = 60.0,
    ):
        """
        Args:
            db_path: Путь к SQLite базе
            backup_path: Путь для JSON backup (опционально)
            auto_save_interval: Интервал автосохранения в секундах
        """
        self.db_path = db_path
        self.backup_path = backup_path or db_path.replace('.db', '_backup.json')
        self.auto_save_interval = auto_save_interval
        
        self._initialized = False
        self._current_state: Optional[NodeState] = None
        self._auto_save_task: Optional[asyncio.Task] = None
        self._dirty = False  # Есть несохранённые изменения
    
    async def initialize(self) -> None:
        """Инициализация хранилища."""
        if not AIOSQLITE_AVAILABLE:
            logger.error("[STATE] aiosqlite not available")
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            # Таблица состояния узла
            await db.execute("""
                CREATE TABLE IF NOT EXISTS node_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    node_id TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    created_at REAL,
                    last_saved REAL,
                    uptime_total REAL DEFAULT 0,
                    dht_node_id TEXT,
                    dht_stored_keys INTEGER DEFAULT 0,
                    total_connections INTEGER DEFAULT 0,
                    total_messages_sent INTEGER DEFAULT 0,
                    total_messages_received INTEGER DEFAULT 0,
                    total_bytes_sent INTEGER DEFAULT 0,
                    total_bytes_received INTEGER DEFAULT 0,
                    extra_data TEXT
                )
            """)
            
            # Таблица пиров
            await db.execute("""
                CREATE TABLE IF NOT EXISTS peers (
                    node_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    first_seen REAL,
                    last_seen REAL,
                    last_connected REAL,
                    connection_count INTEGER DEFAULT 0,
                    failed_connections INTEGER DEFAULT 0,
                    trust_score REAL DEFAULT 0.5,
                    is_bootstrap INTEGER DEFAULT 0,
                    supports_dht INTEGER DEFAULT 0,
                    supports_relay INTEGER DEFAULT 0,
                    nat_type TEXT DEFAULT 'unknown',
                    public_ip TEXT,
                    public_port INTEGER DEFAULT 0
                )
            """)
            
            # Индексы для быстрого поиска
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_peers_last_connected 
                ON peers(last_connected DESC)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_peers_trust 
                ON peers(trust_score DESC)
            """)
            
            # Таблица bootstrap узлов
            await db.execute("""
                CREATE TABLE IF NOT EXISTS bootstrap_nodes (
                    address TEXT PRIMARY KEY,
                    added_at REAL,
                    last_success REAL,
                    success_count INTEGER DEFAULT 0
                )
            """)
            
            await db.commit()
        
        self._initialized = True
        logger.info(f"[STATE] Initialized: {self.db_path}")
    
    async def save_state(self, state: NodeState) -> bool:
        """
        Сохранить состояние узла.
        
        Args:
            state: Состояние для сохранения
        
        Returns:
            True если успешно
        """
        if not self._initialized:
            return False
        
        state.last_saved = time.time()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Сохраняем основное состояние
                await db.execute("""
                    INSERT OR REPLACE INTO node_state 
                    (id, node_id, host, port, created_at, last_saved, uptime_total,
                     dht_node_id, dht_stored_keys, total_connections,
                     total_messages_sent, total_messages_received,
                     total_bytes_sent, total_bytes_received)
                    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.node_id, state.host, state.port,
                    state.created_at, state.last_saved, state.uptime_total,
                    state.dht_node_id, state.dht_stored_keys, state.total_connections,
                    state.total_messages_sent, state.total_messages_received,
                    state.total_bytes_sent, state.total_bytes_received,
                ))
                
                # Сохраняем пиров
                for peer in state.peers.values():
                    await db.execute("""
                        INSERT OR REPLACE INTO peers
                        (node_id, host, port, first_seen, last_seen, last_connected,
                         connection_count, failed_connections, trust_score,
                         is_bootstrap, supports_dht, supports_relay,
                         nat_type, public_ip, public_port)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        peer.node_id, peer.host, peer.port,
                        peer.first_seen, peer.last_seen, peer.last_connected,
                        peer.connection_count, peer.failed_connections, peer.trust_score,
                        int(peer.is_bootstrap), int(peer.supports_dht), int(peer.supports_relay),
                        peer.nat_type, peer.public_ip, peer.public_port,
                    ))
                
                await db.commit()
            
            # JSON backup
            await self._save_json_backup(state)
            
            self._current_state = state
            self._dirty = False
            
            logger.debug(f"[STATE] Saved: {len(state.peers)} peers")
            return True
            
        except Exception as e:
            logger.error(f"[STATE] Save failed: {e}")
            return False
    
    async def load_state(self) -> Optional[NodeState]:
        """
        Загрузить состояние узла.
        
        Returns:
            NodeState или None
        """
        if not self._initialized:
            return None
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Загружаем основное состояние
                async with db.execute("SELECT * FROM node_state WHERE id = 1") as cursor:
                    row = await cursor.fetchone()
                
                if not row:
                    logger.info("[STATE] No saved state found")
                    return None
                
                state = NodeState(
                    node_id=row['node_id'],
                    host=row['host'],
                    port=row['port'],
                    created_at=row['created_at'] or time.time(),
                    last_saved=row['last_saved'] or 0,
                    uptime_total=row['uptime_total'] or 0,
                    dht_node_id=row['dht_node_id'] or "",
                    dht_stored_keys=row['dht_stored_keys'] or 0,
                    total_connections=row['total_connections'] or 0,
                    total_messages_sent=row['total_messages_sent'] or 0,
                    total_messages_received=row['total_messages_received'] or 0,
                    total_bytes_sent=row['total_bytes_sent'] or 0,
                    total_bytes_received=row['total_bytes_received'] or 0,
                )
                
                # Загружаем пиров
                async with db.execute("SELECT * FROM peers") as cursor:
                    async for row in cursor:
                        peer = PeerRecord(
                            node_id=row['node_id'],
                            host=row['host'],
                            port=row['port'],
                            first_seen=row['first_seen'] or 0,
                            last_seen=row['last_seen'] or 0,
                            last_connected=row['last_connected'] or 0,
                            connection_count=row['connection_count'] or 0,
                            failed_connections=row['failed_connections'] or 0,
                            trust_score=row['trust_score'] or 0.5,
                            is_bootstrap=bool(row['is_bootstrap']),
                            supports_dht=bool(row['supports_dht']),
                            supports_relay=bool(row['supports_relay']),
                            nat_type=row['nat_type'] or "unknown",
                            public_ip=row['public_ip'] or "",
                            public_port=row['public_port'] or 0,
                        )
                        state.peers[peer.node_id] = peer
                
                # Загружаем bootstrap узлы
                async with db.execute("SELECT address FROM bootstrap_nodes ORDER BY last_success DESC") as cursor:
                    async for row in cursor:
                        state.bootstrap_nodes.append(row['address'])
            
            self._current_state = state
            logger.info(f"[STATE] Loaded: {len(state.peers)} peers, uptime: {state.uptime_total:.0f}s")
            return state
            
        except Exception as e:
            logger.error(f"[STATE] Load failed: {e}")
            # Попробуем восстановить из JSON backup
            return await self._load_json_backup()
    
    async def get_best_peers_for_reconnect(
        self,
        limit: int = 20,
        min_trust: float = 0.3,
        max_age_hours: float = 24.0,
    ) -> List[PeerRecord]:
        """
        Получить лучших пиров для восстановления соединения.
        
        Args:
            limit: Максимум пиров
            min_trust: Минимальный trust score
            max_age_hours: Максимальный возраст последнего соединения
        
        Returns:
            Список PeerRecord отсортированный по приоритету
        """
        if not self._initialized:
            return []
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Запрос с фильтрацией и сортировкой
                async with db.execute("""
                    SELECT * FROM peers
                    WHERE trust_score >= ?
                      AND last_connected >= ?
                      AND (connection_count + failed_connections) > 0
                    ORDER BY 
                        trust_score * (1.0 * connection_count / (connection_count + failed_connections + 1)) DESC,
                        last_connected DESC
                    LIMIT ?
                """, (min_trust, cutoff_time, limit)) as cursor:
                    
                    peers = []
                    async for row in cursor:
                        peer = PeerRecord(
                            node_id=row['node_id'],
                            host=row['host'],
                            port=row['port'],
                            first_seen=row['first_seen'] or 0,
                            last_seen=row['last_seen'] or 0,
                            last_connected=row['last_connected'] or 0,
                            connection_count=row['connection_count'] or 0,
                            failed_connections=row['failed_connections'] or 0,
                            trust_score=row['trust_score'] or 0.5,
                            is_bootstrap=bool(row['is_bootstrap']),
                            supports_dht=bool(row['supports_dht']),
                            supports_relay=bool(row['supports_relay']),
                        )
                        peers.append(peer)
                    
                    return peers
                    
        except Exception as e:
            logger.error(f"[STATE] Get peers failed: {e}")
            return []
    
    async def record_connection_success(self, node_id: str, host: str, port: int) -> None:
        """Записать успешное соединение."""
        if not self._initialized:
            return
        
        now = time.time()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO peers (node_id, host, port, first_seen, last_seen, last_connected, connection_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(node_id) DO UPDATE SET
                        host = excluded.host,
                        port = excluded.port,
                        last_seen = excluded.last_seen,
                        last_connected = excluded.last_connected,
                        connection_count = connection_count + 1
                """, (node_id, host, port, now, now, now))
                await db.commit()
            
            self._dirty = True
            
        except Exception as e:
            logger.debug(f"[STATE] Record success failed: {e}")
    
    async def record_connection_failure(self, node_id: str) -> None:
        """Записать неудачное соединение."""
        if not self._initialized:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE peers SET failed_connections = failed_connections + 1
                    WHERE node_id = ?
                """, (node_id,))
                await db.commit()
            
            self._dirty = True
            
        except Exception as e:
            logger.debug(f"[STATE] Record failure failed: {e}")
    
    async def update_peer_trust(self, node_id: str, trust_score: float) -> None:
        """Обновить trust score пира."""
        if not self._initialized:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE peers SET trust_score = ? WHERE node_id = ?
                """, (trust_score, node_id))
                await db.commit()
            
            self._dirty = True
            
        except Exception as e:
            logger.debug(f"[STATE] Update trust failed: {e}")
    
    async def add_bootstrap_node(self, address: str) -> None:
        """Добавить bootstrap узел."""
        if not self._initialized:
            return
        
        now = time.time()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO bootstrap_nodes (address, added_at, last_success, success_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(address) DO UPDATE SET
                        last_success = excluded.last_success,
                        success_count = success_count + 1
                """, (address, now, now))
                await db.commit()
                
        except Exception as e:
            logger.debug(f"[STATE] Add bootstrap failed: {e}")
    
    async def get_bootstrap_nodes(self, limit: int = 10) -> List[str]:
        """Получить список bootstrap узлов."""
        if not self._initialized:
            return []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT address FROM bootstrap_nodes
                    ORDER BY last_success DESC
                    LIMIT ?
                """, (limit,)) as cursor:
                    return [row[0] async for row in cursor]
                    
        except Exception as e:
            logger.debug(f"[STATE] Get bootstrap failed: {e}")
            return []
    
    async def start_auto_save(self, state_provider) -> None:
        """
        Запустить автосохранение.
        
        Args:
            state_provider: Callable, возвращающий текущее NodeState
        """
        async def auto_save_loop():
            while True:
                await asyncio.sleep(self.auto_save_interval)
                try:
                    state = state_provider()
                    if state:
                        await self.save_state(state)
                except Exception as e:
                    logger.error(f"[STATE] Auto-save failed: {e}")
        
        self._auto_save_task = asyncio.create_task(auto_save_loop())
        logger.info(f"[STATE] Auto-save started (interval: {self.auto_save_interval}s)")
    
    async def stop_auto_save(self) -> None:
        """Остановить автосохранение."""
        if self._auto_save_task:
            self._auto_save_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._auto_save_task
            self._auto_save_task = None
    
    async def _save_json_backup(self, state: NodeState) -> None:
        """Сохранить JSON backup."""
        try:
            backup_data = state.to_dict()
            backup_data['_backup_time'] = time.time()
            
            # Atomic write
            temp_path = self.backup_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            Path(temp_path).rename(self.backup_path)
            
        except Exception as e:
            logger.debug(f"[STATE] JSON backup failed: {e}")
    
    async def _load_json_backup(self) -> Optional[NodeState]:
        """Загрузить из JSON backup."""
        try:
            if not Path(self.backup_path).exists():
                return None
            
            with open(self.backup_path, 'r') as f:
                data = json.load(f)
            
            data.pop('_backup_time', None)
            return NodeState.from_dict(data)
            
        except Exception as e:
            logger.error(f"[STATE] JSON backup load failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            "initialized": self._initialized,
            "db_path": self.db_path,
            "auto_save_interval": self.auto_save_interval,
            "dirty": self._dirty,
            "current_state": bool(self._current_state),
        }

