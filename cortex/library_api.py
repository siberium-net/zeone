import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

class LibraryAPI:
    """Lightweight access to persisted knowledge items."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        if not self.db_path.exists():
            return
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cid TEXT,
                    path TEXT,
                    summary TEXT,
                    tags TEXT,
                    created_at REAL,
                    size INTEGER,
                    metadata TEXT,
                    compliance_status TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def get_recent_items(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT id, path, summary, tags, created_at, metadata, compliance_status FROM knowledge_base ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [self._row_to_item(r) for r in rows]
        finally:
            conn.close()

    def search_items(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                """
                SELECT id, path, summary, tags, created_at, metadata, compliance_status
                FROM knowledge_base
                WHERE summary LIKE ? OR path LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%", limit),
            )
            rows = cur.fetchall()
            return [self._row_to_item(r) for r in rows]
        finally:
            conn.close()

    def get_item_details(self, item_id: int) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM knowledge_base WHERE id = ?",
                (item_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_item(row, include_raw=True)
        finally:
            conn.close()

    def _row_to_item(self, row: sqlite3.Row, include_raw: bool = False) -> Dict[str, Any]:
        meta = {}
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except Exception:
                meta = {}
        return {
            "id": row["id"],
            "path": row["path"],
            "summary": row["summary"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "created_at": row["created_at"],
            "metadata": meta,
            "compliance_status": row["compliance_status"] or meta.get("compliance_status") or "UNKNOWN",
            **({"raw": dict(row)} if include_raw else {}),
        }
