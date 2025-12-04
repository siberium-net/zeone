"""
Lightweight LRU cache for amplified content.
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


DEFAULT_CACHE_DIR = Path("/tmp/zeone_cache")
DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # 512 MB
DEFAULT_MAX_FILES = 1000


@dataclass
class CacheEntry:
    key: str
    path: Path
    size: int
    last_access: float = field(default_factory=time.time)


class AmplifierCache:
    """
    Simple file-backed LRU cache used by the Amplifier.
    """

    def __init__(
        self,
        root: Path = DEFAULT_CACHE_DIR,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
    ):
        self.root = root
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes) -> None:
        """Store data keyed by hash."""
        async with self._lock:
            path = self.root / key
            size = len(data)
            await asyncio.to_thread(path.write_bytes, data)
            entry = CacheEntry(key=key, path=path, size=size, last_access=time.time())
            self._entries[key] = entry
            await self._evict_if_needed()

    async def get(self, key: str) -> Optional[bytes]:
        """Return cached data if present."""
        async with self._lock:
            entry = self._entries.get(key)
            if not entry or not entry.path.exists():
                return None
            entry.last_access = time.time()
            return await asyncio.to_thread(entry.path.read_bytes)

    def has(self, key: str) -> bool:
        entry = self._entries.get(key)
        return bool(entry and entry.path.exists())

    async def _evict_if_needed(self) -> None:
        total_bytes = sum(e.size for e in self._entries.values())
        while len(self._entries) > self.max_files or total_bytes > self.max_bytes:
            oldest_key = min(self._entries.values(), key=lambda e: e.last_access).key
            entry = self._entries.pop(oldest_key, None)
            if entry:
                try:
                    entry.path.unlink(missing_ok=True)
                except Exception:
                    pass
                total_bytes -= entry.size

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Utility to compute sha256 for ad-hoc chunk hashing."""
        return hashlib.sha256(data).hexdigest()
