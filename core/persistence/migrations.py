import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

MIGRATIONS: List[Tuple[int, str]] = [
    (1, """
        CREATE TABLE IF NOT EXISTS peers (
            node_id TEXT PRIMARY KEY,
            public_key TEXT,
            trust_score REAL DEFAULT 0.5,
            total_sent REAL DEFAULT 0,
            total_received REAL DEFAULT 0,
            first_seen REAL,
            last_seen REAL,
            metadata TEXT
        );
    """),
    (2, "ALTER TABLE peers ADD COLUMN zkp_proof TEXT;"),
    (3, """
        CREATE TABLE IF NOT EXISTS media_assets (
            id TEXT PRIMARY KEY,
            phash TEXT,
            parent_id TEXT,
            faces TEXT,
            brands TEXT,
            tech_meta TEXT,
            created_at REAL,
            path TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_media_phash ON media_assets(phash);
    """),
]


def _ensure_schema_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_versions(
            version INTEGER PRIMARY KEY
        )
    """)
    conn.commit()


def _get_current_version(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT MAX(version) FROM schema_versions")
    row = cur.fetchone()
    return row[0] or 0


def run_migrations(project_root: Path) -> None:
    db_path = project_root / "ledger.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema_table(conn)
        current = _get_current_version(conn)
        for version, sql in MIGRATIONS:
            if version > current:
                try:
                    conn.executescript(sql)
                    conn.execute("INSERT INTO schema_versions(version) VALUES (?)", (version,))
                    conn.commit()
                    logger.info(f"[MIGRATION] Applied v{version}")
                except Exception as e:
                    logger.error(f"[MIGRATION] v{version} failed: {e}")
                    conn.rollback()
                    break
    finally:
        conn.close()
