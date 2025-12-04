"""
Routing helpers for VPN exit selection and persistence.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, List

from core.utils.async_db import connect as async_connect
from config import config


@dataclass
class VpnRoute:
    """Cached VPN route information."""

    target_country: str
    exit_node_id: str
    last_latency: float
    success_rate: float
    last_used: float


class VpnRouteStore:
    """
    Lightweight SQLite-backed cache for VPN exit choices.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.ledger.database_path
        self._db = None  # type: ignore
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._db = await async_connect(self.db_path)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS vpn_routes (
                target_country TEXT NOT NULL,
                exit_node_id TEXT NOT NULL,
                last_latency REAL DEFAULT 0,
                success_rate REAL DEFAULT 0.5,
                last_used REAL DEFAULT 0,
                PRIMARY KEY (target_country, exit_node_id)
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_vpn_routes_last_used ON vpn_routes(target_country, last_used DESC)"
        )
        await self._db.commit()
        self._initialized = True

    async def get_recent_route(
        self,
        target_country: str,
        max_age: float = 86400.0,
    ) -> Optional[VpnRoute]:
        """Return most recently used route for a country if still fresh."""
        await self.initialize()
        now = time.time()
        cursor = await self._db.execute(
            """
            SELECT target_country, exit_node_id, last_latency, success_rate, last_used
            FROM vpn_routes
            WHERE target_country = ?
            ORDER BY last_used DESC
            LIMIT 1
            """,
            (target_country,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        if row[4] and now - row[4] > max_age:
            return None
        return VpnRoute(
            target_country=row[0],
            exit_node_id=row[1],
            last_latency=row[2],
            success_rate=row[3],
            last_used=row[4],
        )

    async def upsert_route(
        self,
        target_country: str,
        exit_node_id: str,
        latency: float,
        success: bool = True,
    ) -> None:
        """Insert or update a cached route with exponential smoothing on success."""
        await self.initialize()
        now = time.time()
        cursor = await self._db.execute(
            """
            SELECT success_rate FROM vpn_routes
            WHERE target_country = ? AND exit_node_id = ?
            """,
            (target_country, exit_node_id),
        )
        row = await cursor.fetchone()
        prev_success = row[0] if row else 0.5
        new_success = 0.7 * prev_success + 0.3 * (1.0 if success else 0.0)
        await self._db.execute(
            """
            INSERT INTO vpn_routes (target_country, exit_node_id, last_latency, success_rate, last_used)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(target_country, exit_node_id)
            DO UPDATE SET
                last_latency=excluded.last_latency,
                success_rate=excluded.success_rate,
                last_used=excluded.last_used
            """,
            (target_country, exit_node_id, latency, new_success, now),
        )
        await self._db.commit()

    async def record_failure(self, target_country: str, exit_node_id: str) -> None:
        """Drop success score when a route fails."""
        await self.upsert_route(target_country, exit_node_id, latency=0.0, success=False)

    async def list_routes(self, target_country: Optional[str] = None) -> List[VpnRoute]:
        """List cached routes optionally filtered by target."""
        await self.initialize()
        if target_country:
            cursor = await self._db.execute(
                """
                SELECT target_country, exit_node_id, last_latency, success_rate, last_used
                FROM vpn_routes
                WHERE target_country = ?
                ORDER BY success_rate DESC, last_used DESC
                """,
                (target_country,),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT target_country, exit_node_id, last_latency, success_rate, last_used
                FROM vpn_routes
                ORDER BY success_rate DESC, last_used DESC
                """
            )
        rows = await cursor.fetchall()
        return [
            VpnRoute(
                target_country=r[0],
                exit_node_id=r[1],
                last_latency=r[2],
                success_rate=r[3],
                last_used=r[4],
            )
            for r in rows
        ]
