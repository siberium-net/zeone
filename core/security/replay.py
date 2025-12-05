"""
Persistent Replay Attack Protection
====================================

[SECURITY] Защита от повторного воспроизведения пакетов (Replay Attack).

Проблема текущей реализации:
- Nonces хранятся в RAM
- При перезагрузке узла злоумышленник может повторить старую транзакцию

Решение:
- Disk-backed storage через aiosqlite
- Атомарная проверка is_nonce_fresh
- Фоновая ротация старых записей (MAX_PACKET_AGE = 60 сек)

[PERSISTENCE] Таблица:
    seen_nonces (nonce BLOB PRIMARY KEY, timestamp REAL)

[PERFORMANCE] Оптимизации:
- In-memory кэш для hot path
- Batch cleanup каждые CLEANUP_INTERVAL секунд
- Индекс по timestamp для быстрого удаления
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Set
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Настройки защиты от Replay
MAX_PACKET_AGE: float = 60.0          # Максимальный возраст пакета (секунды)
CLEANUP_INTERVAL: float = 30.0         # Интервал очистки (секунды)
MEMORY_CACHE_SIZE: int = 10000         # Размер in-memory кэша


class ReplayProtector:
    """
    Защита от Replay-атак с персистентным хранением.
    
    [SECURITY] Гарантии:
    - Каждый nonce может быть использован только один раз
    - Защита сохраняется после перезагрузки узла
    - Атомарная проверка через SQLite transaction
    
    [USAGE]
        protector = ReplayProtector("data/replay.db")
        await protector.initialize()
        
        if await protector.is_nonce_fresh(nonce):
            # Обрабатываем сообщение
            ...
        else:
            # Replay attack detected!
            ...
    """
    
    def __init__(
        self,
        db_path: str = "data/replay.db",
        max_age: float = MAX_PACKET_AGE,
        cleanup_interval: float = CLEANUP_INTERVAL,
    ):
        """
        Args:
            db_path: Путь к файлу базы данных
            max_age: Максимальный возраст nonce в секундах
            cleanup_interval: Интервал очистки в секундах
        """
        self.db_path = db_path
        self.max_age = max_age
        self.cleanup_interval = cleanup_interval
        
        self._db: Optional["aiosqlite.Connection"] = None
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # In-memory кэш для hot path
        self._memory_cache: Set[bytes] = set()
        self._cache_timestamps: dict = {}  # nonce -> timestamp
    
    async def initialize(self) -> None:
        """
        Инициализировать базу данных и запустить cleanup task.
        
        [PERSISTENCE] Создаёт таблицу seen_nonces если не существует.
        """
        import aiosqlite
        
        # Создаём директорию если нужно
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        self._db = await aiosqlite.connect(self.db_path)
        
        # WAL mode для лучшей производительности
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        
        # Создаём таблицу
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS seen_nonces (
                nonce BLOB PRIMARY KEY,
                timestamp REAL NOT NULL
            )
        """)
        
        # Индекс для быстрого cleanup
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_nonce_timestamp 
            ON seen_nonces(timestamp)
        """)
        
        await self._db.commit()
        
        # Загружаем недавние nonces в кэш
        await self._load_recent_to_cache()
        
        # Запускаем фоновую очистку
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"[REPLAY] Initialized with max_age={self.max_age}s, cleanup_interval={self.cleanup_interval}s")
    
    async def _load_recent_to_cache(self) -> None:
        """Загрузить недавние nonces в memory cache."""
        if not self._db:
            return
        
        cutoff = time.time() - self.max_age
        cursor = await self._db.execute(
            "SELECT nonce, timestamp FROM seen_nonces WHERE timestamp > ? LIMIT ?",
            (cutoff, MEMORY_CACHE_SIZE)
        )
        rows = await cursor.fetchall()
        
        for row in rows:
            nonce = row[0]
            ts = row[1]
            self._memory_cache.add(nonce)
            self._cache_timestamps[nonce] = ts
        
        logger.debug(f"[REPLAY] Loaded {len(rows)} nonces to cache")
    
    async def close(self) -> None:
        """Закрыть соединение и остановить cleanup task."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._db:
            await self._db.close()
            self._db = None
        
        self._memory_cache.clear()
        self._cache_timestamps.clear()
        
        logger.info("[REPLAY] Closed")
    
    async def is_nonce_fresh(self, nonce: bytes) -> bool:
        """
        Проверить, является ли nonce свежим (не использовался ранее).
        
        [SECURITY] Атомарная операция:
        1. Проверяем существование в БД
        2. Если не существует - добавляем
        3. Возвращаем результат
        
        [ATOMIC] Использует SQLite transaction для атомарности.
        
        Args:
            nonce: Уникальный nonce сообщения (bytes)
        
        Returns:
            True если nonce свежий (можно использовать)
            False если nonce уже был использован (replay attack)
        """
        # Быстрая проверка в memory cache
        if nonce in self._memory_cache:
            return False
        
        async with self._lock:
            if not self._db:
                # Fallback на memory-only если БД недоступна
                if nonce in self._memory_cache:
                    return False
                self._add_to_cache(nonce)
                return True
            
            try:
                # Атомарная проверка и вставка
                cursor = await self._db.execute(
                    "SELECT 1 FROM seen_nonces WHERE nonce = ?",
                    (nonce,)
                )
                row = await cursor.fetchone()
                
                if row:
                    # Nonce уже существует - replay attack!
                    return False
                
                # Добавляем новый nonce
                now = time.time()
                await self._db.execute(
                    "INSERT INTO seen_nonces (nonce, timestamp) VALUES (?, ?)",
                    (nonce, now)
                )
                await self._db.commit()
                
                # Добавляем в кэш
                self._add_to_cache(nonce)
                
                return True
                
            except Exception as e:
                logger.error(f"[REPLAY] Database error: {e}")
                # Fallback: проверяем только кэш
                if nonce in self._memory_cache:
                    return False
                self._add_to_cache(nonce)
                return True
    
    def _add_to_cache(self, nonce: bytes) -> None:
        """Добавить nonce в memory cache с ротацией."""
        now = time.time()
        
        # Ротация кэша если переполнен
        if len(self._memory_cache) >= MEMORY_CACHE_SIZE:
            self._evict_old_from_cache()
        
        self._memory_cache.add(nonce)
        self._cache_timestamps[nonce] = now
    
    def _evict_old_from_cache(self) -> None:
        """Удалить старые записи из кэша."""
        cutoff = time.time() - self.max_age
        to_remove = []
        
        for nonce, ts in list(self._cache_timestamps.items()):
            if ts < cutoff:
                to_remove.append(nonce)
        
        for nonce in to_remove:
            self._memory_cache.discard(nonce)
            self._cache_timestamps.pop(nonce, None)
        
        # Если всё ещё переполнен - удаляем самые старые
        if len(self._memory_cache) >= MEMORY_CACHE_SIZE:
            sorted_nonces = sorted(
                self._cache_timestamps.items(),
                key=lambda x: x[1]
            )
            # Удаляем 20% самых старых
            to_remove_count = max(1, len(sorted_nonces) // 5)
            for nonce, _ in sorted_nonces[:to_remove_count]:
                self._memory_cache.discard(nonce)
                self._cache_timestamps.pop(nonce, None)
    
    async def _cleanup_loop(self) -> None:
        """
        Фоновая задача очистки старых nonces.
        
        [PERFORMANCE] Удаляет записи старше MAX_PACKET_AGE.
        """
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_nonces()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[REPLAY] Cleanup error: {e}")
    
    async def _cleanup_old_nonces(self) -> None:
        """Удалить nonces старше max_age."""
        if not self._db:
            return
        
        cutoff = time.time() - self.max_age
        
        async with self._lock:
            try:
                cursor = await self._db.execute(
                    "DELETE FROM seen_nonces WHERE timestamp < ?",
                    (cutoff,)
                )
                deleted = cursor.rowcount
                await self._db.commit()
                
                if deleted > 0:
                    logger.debug(f"[REPLAY] Cleaned up {deleted} old nonces")
                
                # Очищаем memory cache
                self._evict_old_from_cache()
                
            except Exception as e:
                logger.error(f"[REPLAY] Cleanup DB error: {e}")
    
    async def get_stats(self) -> dict:
        """Получить статистику защиты."""
        stats = {
            "memory_cache_size": len(self._memory_cache),
            "max_age_seconds": self.max_age,
            "cleanup_interval": self.cleanup_interval,
        }
        
        if self._db:
            try:
                cursor = await self._db.execute(
                    "SELECT COUNT(*) FROM seen_nonces"
                )
                row = await cursor.fetchone()
                stats["db_nonce_count"] = row[0] if row else 0
            except Exception:
                stats["db_nonce_count"] = -1
        
        return stats
    
    def check_nonce_sync(self, nonce: bytes) -> bool:
        """
        Синхронная проверка nonce (только memory cache).
        
        [PERFORMANCE] Для hot path где async недопустим.
        Не гарантирует полную защиту - только быстрая проверка кэша.
        
        Returns:
            False если nonce точно был использован
            True если nonce возможно свежий (требует async проверки)
        """
        return nonce not in self._memory_cache


# Singleton instance
_replay_protector: Optional[ReplayProtector] = None


async def get_replay_protector(
    db_path: str = "data/replay.db",
) -> ReplayProtector:
    """
    Получить глобальный экземпляр ReplayProtector.
    
    [USAGE]
        protector = await get_replay_protector()
        if await protector.is_nonce_fresh(nonce):
            # Process message
            ...
    """
    global _replay_protector
    if _replay_protector is None:
        _replay_protector = ReplayProtector(db_path)
        await _replay_protector.initialize()
    return _replay_protector


async def cleanup_replay_protector() -> None:
    """Закрыть глобальный ReplayProtector."""
    global _replay_protector
    if _replay_protector is not None:
        await _replay_protector.close()
        _replay_protector = None

