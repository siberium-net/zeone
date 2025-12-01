"""
DHT Storage - Локальное хранилище для Kademlia DHT
===================================================

[KADEMLIA] Хранилище пар key-value:
- SQLite для персистентности
- TTL для автоматического удаления
- Republish для поддержания данных в сети

[STORAGE] Правила:
- Ключ = 160-битный SHA-1 хэш
- Значение = произвольные байты (до 64KB)
- TTL = время жизни (по умолчанию 24 часа)
- Republish = повторная публикация каждые 60 минут
"""

import asyncio
import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


# Константы
DEFAULT_TTL = 86400  # 24 часа в секундах
REPUBLISH_INTERVAL = 3600  # 60 минут
MAX_VALUE_SIZE = 65536  # 64 KB
CLEANUP_INTERVAL = 300  # 5 минут


@dataclass
class StoredValue:
    """
    Значение, хранящееся в DHT.
    
    [KADEMLIA] Каждое значение содержит:
    - key: 160-битный идентификатор (SHA-1 хэш ключа)
    - value: данные
    - publisher_id: ID узла, опубликовавшего данные
    - timestamp: время публикации
    - ttl: время жизни
    """
    
    key: bytes  # 20 байт
    value: bytes
    publisher_id: bytes  # 20 байт - кто опубликовал
    timestamp: float = field(default_factory=time.time)
    ttl: int = DEFAULT_TTL
    
    @property
    def key_hex(self) -> str:
        return self.key.hex()
    
    @property
    def expires_at(self) -> float:
        """Время истечения TTL."""
        return self.timestamp + self.ttl
    
    @property
    def is_expired(self) -> bool:
        """Проверить, истёк ли TTL."""
        return time.time() > self.expires_at
    
    @property
    def needs_republish(self, interval: float = REPUBLISH_INTERVAL) -> bool:
        """Проверить, нужна ли повторная публикация."""
        return time.time() - self.timestamp > interval
    
    def to_dict(self) -> Dict:
        """Сериализация."""
        return {
            "key": self.key.hex(),
            "value": self.value.hex(),
            "publisher_id": self.publisher_id.hex(),
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoredValue":
        """Десериализация."""
        return cls(
            key=bytes.fromhex(data["key"]),
            value=bytes.fromhex(data["value"]),
            publisher_id=bytes.fromhex(data["publisher_id"]),
            timestamp=data.get("timestamp", time.time()),
            ttl=data.get("ttl", DEFAULT_TTL),
        )


class DHTStorage:
    """
    Локальное хранилище DHT с персистентностью в SQLite.
    
    [KADEMLIA] Функции:
    - store(key, value): Сохранить пару key-value
    - get(key): Получить значение по ключу
    - delete(key): Удалить значение
    - cleanup(): Удалить истёкшие записи
    - get_republish_keys(): Получить ключи для republish
    
    [PERSISTENCE] SQLite таблица dht_store:
    - key: BLOB PRIMARY KEY (20 байт)
    - value: BLOB (до 64KB)
    - publisher_id: BLOB (20 байт)
    - timestamp: REAL
    - ttl: INTEGER
    """
    
    def __init__(self, db_path: str = "dht_storage.db"):
        """
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Инициализировать хранилище и создать таблицы."""
        if self._initialized:
            return
        
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        
        # Создаём таблицу для DHT
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS dht_store (
                key BLOB PRIMARY KEY,
                value BLOB NOT NULL,
                publisher_id BLOB NOT NULL,
                timestamp REAL NOT NULL,
                ttl INTEGER NOT NULL,
                last_republish REAL
            )
        """)
        
        # Индекс для поиска истёкших записей
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_dht_expires 
            ON dht_store(timestamp, ttl)
        """)
        
        # Индекс для republish
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_dht_republish 
            ON dht_store(last_republish)
        """)
        
        await self._db.commit()
        self._initialized = True
        
        logger.info(f"[DHT_STORAGE] Initialized: {self.db_path}")
    
    async def close(self) -> None:
        """Закрыть соединение с базой данных."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False
    
    async def store(
        self,
        key: bytes,
        value: bytes,
        publisher_id: bytes,
        ttl: int = DEFAULT_TTL,
    ) -> bool:
        """
        Сохранить значение в DHT.
        
        [KADEMLIA] STORE RPC:
        - Сохраняем пару key-value
        - Перезаписываем если ключ уже существует
        - Обновляем timestamp
        
        Args:
            key: 20-байтный ключ (SHA-1)
            value: Данные для хранения
            publisher_id: ID узла-публикатора
            ttl: Время жизни в секундах
        
        Returns:
            True если успешно сохранено
        """
        if len(key) != 20:
            logger.warning(f"[DHT_STORAGE] Invalid key length: {len(key)}")
            return False
        
        if len(value) > MAX_VALUE_SIZE:
            logger.warning(f"[DHT_STORAGE] Value too large: {len(value)} > {MAX_VALUE_SIZE}")
            return False
        
        if len(publisher_id) != 20:
            logger.warning(f"[DHT_STORAGE] Invalid publisher_id length: {len(publisher_id)}")
            return False
        
        async with self._lock:
            try:
                await self._db.execute(
                    """
                    INSERT OR REPLACE INTO dht_store 
                    (key, value, publisher_id, timestamp, ttl, last_republish)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, value, publisher_id, time.time(), ttl, time.time())
                )
                await self._db.commit()
                
                logger.debug(f"[DHT_STORAGE] Stored: {key.hex()[:16]}... ({len(value)} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"[DHT_STORAGE] Store error: {e}")
                return False
    
    async def get(self, key: bytes) -> Optional[StoredValue]:
        """
        Получить значение из DHT.
        
        Args:
            key: 20-байтный ключ
        
        Returns:
            StoredValue или None если не найдено/истекло
        """
        if len(key) != 20:
            return None
        
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT * FROM dht_store WHERE key = ?",
                (key,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            # Проверяем TTL
            stored = StoredValue(
                key=row["key"],
                value=row["value"],
                publisher_id=row["publisher_id"],
                timestamp=row["timestamp"],
                ttl=row["ttl"],
            )
            
            if stored.is_expired:
                # Удаляем истёкшую запись
                await self._db.execute("DELETE FROM dht_store WHERE key = ?", (key,))
                await self._db.commit()
                return None
            
            return stored
    
    async def delete(self, key: bytes) -> bool:
        """
        Удалить значение из DHT.
        
        Args:
            key: 20-байтный ключ
        
        Returns:
            True если удалено
        """
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM dht_store WHERE key = ?",
                (key,)
            )
            await self._db.commit()
            return cursor.rowcount > 0
    
    async def has_key(self, key: bytes) -> bool:
        """Проверить наличие ключа."""
        value = await self.get(key)
        return value is not None
    
    async def get_all_keys(self) -> List[bytes]:
        """Получить все ключи (не истёкшие)."""
        async with self._lock:
            now = time.time()
            cursor = await self._db.execute(
                "SELECT key FROM dht_store WHERE timestamp + ttl > ?",
                (now,)
            )
            rows = await cursor.fetchall()
            return [row["key"] for row in rows]
    
    async def get_republish_keys(self, interval: float = REPUBLISH_INTERVAL) -> List[StoredValue]:
        """
        Получить значения для republish.
        
        [KADEMLIA] Republish:
        - Каждый час повторно публикуем значения
        - Это поддерживает данные в сети
        
        Args:
            interval: Интервал republish в секундах
        
        Returns:
            Список StoredValue для republish
        """
        async with self._lock:
            now = time.time()
            threshold = now - interval
            
            cursor = await self._db.execute(
                """
                SELECT * FROM dht_store 
                WHERE last_republish < ? AND timestamp + ttl > ?
                """,
                (threshold, now)
            )
            rows = await cursor.fetchall()
            
            values = []
            for row in rows:
                values.append(StoredValue(
                    key=row["key"],
                    value=row["value"],
                    publisher_id=row["publisher_id"],
                    timestamp=row["timestamp"],
                    ttl=row["ttl"],
                ))
            
            return values
    
    async def mark_republished(self, key: bytes) -> None:
        """Отметить ключ как republished."""
        async with self._lock:
            await self._db.execute(
                "UPDATE dht_store SET last_republish = ? WHERE key = ?",
                (time.time(), key)
            )
            await self._db.commit()
    
    async def cleanup(self) -> int:
        """
        Удалить истёкшие записи.
        
        Returns:
            Количество удалённых записей
        """
        async with self._lock:
            now = time.time()
            cursor = await self._db.execute(
                "DELETE FROM dht_store WHERE timestamp + ttl <= ?",
                (now,)
            )
            await self._db.commit()
            
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"[DHT_STORAGE] Cleanup: removed {deleted} expired entries")
            
            return deleted
    
    async def get_stats(self) -> Dict:
        """Получить статистику хранилища."""
        async with self._lock:
            now = time.time()
            
            # Общее количество записей
            cursor = await self._db.execute("SELECT COUNT(*) as count FROM dht_store")
            row = await cursor.fetchone()
            total = row["count"]
            
            # Не истёкшие записи
            cursor = await self._db.execute(
                "SELECT COUNT(*) as count FROM dht_store WHERE timestamp + ttl > ?",
                (now,)
            )
            row = await cursor.fetchone()
            active = row["count"]
            
            # Общий размер данных
            cursor = await self._db.execute(
                "SELECT SUM(LENGTH(value)) as size FROM dht_store"
            )
            row = await cursor.fetchone()
            total_size = row["size"] or 0
            
            return {
                "total_entries": total,
                "active_entries": active,
                "expired_entries": total - active,
                "total_size_bytes": total_size,
                "db_path": self.db_path,
            }
    
    async def get_local_values_for_node(self, node_id: bytes, count: int = 10) -> List[StoredValue]:
        """
        Получить значения, которые должны быть у узла.
        
        [KADEMLIA] При подключении нового узла:
        - Передаём ему значения, к которым он ближе всех
        
        Args:
            node_id: ID нового узла
            count: Максимальное количество значений
        
        Returns:
            Список StoredValue для передачи
        """
        # Получаем все активные записи
        keys = await self.get_all_keys()
        
        # Сортируем по XOR-расстоянию до node_id
        from .routing import xor_distance
        
        distances = [(xor_distance(k, node_id), k) for k in keys]
        distances.sort(key=lambda x: x[0])
        
        # Получаем значения для ближайших ключей
        values = []
        for _, key in distances[:count]:
            value = await self.get(key)
            if value:
                values.append(value)
        
        return values


def string_to_key(s: str) -> bytes:
    """
    Преобразовать строку в 20-байтный ключ DHT.
    
    Args:
        s: Строка для хэширования
    
    Returns:
        20-байтный SHA-1 хэш
    """
    return hashlib.sha1(s.encode('utf-8')).digest()


def bytes_to_key(data: bytes) -> bytes:
    """
    Преобразовать байты в 20-байтный ключ DHT.
    
    Args:
        data: Данные для хэширования
    
    Returns:
        20-байтный SHA-1 хэш
    """
    return hashlib.sha1(data).digest()

