"""
Lightweight async wrapper around sqlite3.

[RATIONALE]
Some environments can hang or deadlock when SQLite operations are executed
from worker threads. This wrapper provides a minimal async interface
(`execute`, `fetchone`, `fetchall`, `commit`, `close`) while performing
SQLite calls synchronously under an asyncio lock.
"""

import asyncio
import sqlite3
from typing import Any, Iterable, Optional


class AsyncCursor:
    def __init__(self, cursor: sqlite3.Cursor, lock: asyncio.Lock):
        self._cursor = cursor
        self._lock = lock

    async def fetchone(self) -> Optional[sqlite3.Row]:
        async with self._lock:
            return self._cursor.fetchone()

    async def fetchall(self) -> Iterable[sqlite3.Row]:
        async with self._lock:
            return self._cursor.fetchall()


class AsyncConnection:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = asyncio.Lock()

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, factory):
        self._conn.row_factory = factory

    async def execute(self, sql: str, params: tuple = ()):
        async with self._lock:
            cursor = self._conn.execute(sql, params)
        return AsyncCursor(cursor, self._lock)

    async def executemany(self, sql: str, seq_of_params):
        async with self._lock:
            cursor = self._conn.executemany(sql, seq_of_params)
        return AsyncCursor(cursor, self._lock)

    async def commit(self) -> None:
        async with self._lock:
            self._conn.commit()

    async def close(self) -> None:
        async with self._lock:
            self._conn.close()


async def connect(path: str, row_factory=None) -> AsyncConnection:
    conn = sqlite3.connect(
        path,
        check_same_thread=False,
        isolation_level=None,  # autocommit; explicit commits still used
    )
    if row_factory:
        conn.row_factory = row_factory
    return AsyncConnection(conn)
