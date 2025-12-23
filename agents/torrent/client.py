import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.events import event_bus

try:
    import libtorrent as lt  # type: ignore
    _LT_AVAILABLE = True
except Exception:  # pragma: no cover
    lt = None  # type: ignore
    _LT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TorrentItem:
    info_hash: str
    magnet_uri: str
    handle: Any
    save_path: Path
    name: str = ""
    num_pieces: int = 0
    piece_length: int = 0
    total_size: int = 0
    preview_applied: bool = False
    added_at: float = field(default_factory=time.time)
    torrent_info: Optional[Any] = None


class TorrentManager:
    """Manage libtorrent session and emit events on completion."""

    def __init__(
        self,
        listen_port: int = 6881,
        save_path: str = "storage/torrents",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.listen_port = listen_port
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._loop = loop
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._items: Dict[str, TorrentItem] = {}

        self._session = None
        if _LT_AVAILABLE:
            self._session = lt.session()
            self._configure_session()
        else:
            logger.warning("[TORRENT] libtorrent not available")

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> None:
        if self._running or not self._session:
            return
        self._running = True
        self._thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._thread.start()
        logger.info("[TORRENT] TorrentManager started")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("[TORRENT] TorrentManager stopped")

    def add_magnet(self, link: str) -> Optional[str]:
        if not link:
            return None
        info_hash = self._extract_info_hash(link)
        if not info_hash:
            info_hash = f"unknown-{int(time.time() * 1000)}"
        if not self._session:
            logger.warning("[TORRENT] Session unavailable; cannot add magnet")
            return info_hash

        params = lt.add_torrent_params()
        params.save_path = str(self.save_path)
        handle = lt.add_magnet_uri(self._session, link, params)

        item = TorrentItem(
            info_hash=info_hash,
            magnet_uri=link,
            handle=handle,
            save_path=self.save_path,
        )
        with self._lock:
            self._items[info_hash] = item

        self.start()
        return info_hash

    def pause(self, info_hash: Optional[str] = None) -> None:
        if not self._session:
            return
        if info_hash:
            item = self._items.get(info_hash)
            if item:
                item.handle.pause()
            return
        self._session.pause()

    def resume(self, info_hash: Optional[str] = None) -> None:
        if not self._session:
            return
        if info_hash:
            item = self._items.get(info_hash)
            if item:
                item.handle.resume()
            return
        self._session.resume()

    def get_stats(self) -> Dict[str, Any]:
        items = []
        with self._lock:
            values = list(self._items.values())
        for item in values:
            try:
                status = item.handle.status()
                items.append(
                    {
                        "info_hash": item.info_hash,
                        "name": item.name or getattr(status, "name", ""),
                        "progress": float(getattr(status, "progress", 0.0)),
                        "download_rate": float(getattr(status, "download_rate", 0.0)),
                        "upload_rate": float(getattr(status, "upload_rate", 0.0)),
                        "num_peers": int(getattr(status, "num_peers", 0)),
                        "state": str(getattr(status, "state", "")),
                    }
                )
            except Exception:
                continue
        return {
            "running": self._running,
            "torrent_count": len(items),
            "torrents": items,
        }

    def get_files(self, info_hash: str) -> List[Dict[str, Any]]:
        item = self._items.get(info_hash)
        info = self._get_torrent_info(item)
        if not item or not info:
            return []
        files = []
        try:
            storage = info.files()
            save_path = Path(item.handle.save_path())
            for idx in range(storage.num_files()):
                rel_path = storage.file_path(idx)
                files.append(
                    {
                        "index": idx,
                        "path": str(save_path / rel_path),
                        "size": storage.file_size(idx),
                        "offset": storage.file_offset(idx),
                    }
                )
        except Exception:
            return []
        return files

    def get_file_path(self, info_hash: str, file_index: int) -> Optional[Tuple[str, int]]:
        files = self.get_files(info_hash)
        for item in files:
            if item["index"] == file_index:
                return item["path"], int(item.get("size", 0))
        return None

    def get_piece_index(self, info_hash: str, file_index: int, offset: int) -> Optional[int]:
        item = self._items.get(info_hash)
        info = self._get_torrent_info(item)
        if not item or not info:
            return None
        try:
            storage = info.files()
            file_offset = storage.file_offset(file_index)
            piece_length = info.piece_length()
            piece = int((file_offset + max(0, offset)) // max(piece_length, 1))
            return min(piece, max(info.num_pieces() - 1, 0))
        except Exception:
            return None

    def has_piece(self, info_hash: str, piece_index: int) -> bool:
        item = self._items.get(info_hash)
        if not item:
            return False
        try:
            return bool(item.handle.have_piece(int(piece_index)))
        except Exception:
            return False

    def prioritize_byte_range(
        self,
        info_hash: str,
        file_index: int,
        start_byte: int,
        length: int,
    ) -> List[int]:
        item = self._items.get(info_hash)
        info = self._get_torrent_info(item)
        if not item or not info:
            return []
        try:
            storage = info.files()
            file_size = storage.file_size(file_index)
            start = max(0, int(start_byte))
            length = max(0, int(length))
            if length == 0:
                length = file_size - start
            end = min(file_size - 1, start + length - 1)
            if end < start:
                return []

            file_offset = storage.file_offset(file_index)
            piece_length = info.piece_length()
            start_piece = int((file_offset + start) // max(piece_length, 1))
            end_piece = int((file_offset + end) // max(piece_length, 1))
            end_piece = min(end_piece, max(info.num_pieces() - 1, 0))

            pieces = list(range(start_piece, end_piece + 1))
            for idx in pieces:
                self._set_piece_priority(item.handle, idx)
            return pieces
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _configure_session(self) -> None:
        try:
            self._session.listen_on(self.listen_port, self.listen_port + 10)
        except Exception:
            try:
                self._session.apply_settings(
                    {"listen_interfaces": f"0.0.0.0:{self.listen_port}"}
                )
            except Exception as e:
                logger.warning(f"[TORRENT] Failed to bind port: {e}")
        try:
            self._session.set_alert_mask(lt.alert.category_t.all_categories)
        except Exception:
            pass

    def _alert_loop(self) -> None:
        while self._running and self._session:
            try:
                alert = self._session.wait_for_alert(0.5)
                if alert:
                    for item in self._session.pop_alerts():
                        self._handle_alert(item)
            except Exception as e:
                logger.warning(f"[TORRENT] Alert loop error: {e}")
                time.sleep(0.5)

    def _handle_alert(self, alert: Any) -> None:
        name = alert.__class__.__name__
        if name == "metadata_received_alert":
            self._handle_metadata(alert)
        elif name == "torrent_finished_alert":
            self._handle_finished(alert)

    def _handle_metadata(self, alert: Any) -> None:
        handle = getattr(alert, "handle", None)
        if not handle:
            return
        info_hash = self._get_info_hash(handle)
        if not info_hash:
            info_hash = self._find_info_hash(handle)
        item = self._items.get(info_hash)
        if not item:
            return
        try:
            torrent_info = handle.torrent_file()
            item.torrent_info = torrent_info
            item.name = torrent_info.name()
            item.num_pieces = torrent_info.num_pieces()
            item.piece_length = torrent_info.piece_length()
            item.total_size = torrent_info.total_size()
            if not item.preview_applied:
                self._apply_preview_priority(handle, item.num_pieces)
                item.preview_applied = True
        except Exception:
            pass

    def _handle_finished(self, alert: Any) -> None:
        handle = getattr(alert, "handle", None)
        if not handle:
            return
        info_hash = self._get_info_hash(handle)
        if not info_hash:
            info_hash = self._find_info_hash(handle)
        item = self._items.get(info_hash)
        if not item:
            return
        payload = {
            "info_hash": info_hash,
            "save_path": str(item.save_path),
            "files": self._list_files(handle),
            "magnet_uri": item.magnet_uri,
        }
        self._broadcast("TORRENT_FINISHED", payload)

    def _apply_preview_priority(self, handle: Any, num_pieces: int) -> None:
        if num_pieces <= 0:
            return
        edge_count = max(1, int(num_pieces * 0.01))
        indices = list(range(edge_count)) + list(range(max(num_pieces - edge_count, 0), num_pieces))
        for idx in indices:
            self._set_piece_priority(handle, idx)

    def _list_files(self, handle: Any) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        try:
            info = handle.torrent_file()
            storage = info.files()
            for idx in range(storage.num_files()):
                files.append(
                    {
                        "path": storage.file_path(idx),
                        "size": storage.file_size(idx),
                    }
                )
        except Exception:
            pass
        return files

    def _get_torrent_info(self, item: Optional[TorrentItem]) -> Optional[Any]:
        if not item:
            return None
        if item.torrent_info is not None:
            return item.torrent_info
        try:
            item.torrent_info = item.handle.torrent_file()
        except Exception:
            return None
        return item.torrent_info

    @staticmethod
    def _set_piece_priority(handle: Any, idx: int) -> None:
        try:
            handle.set_piece_deadline(int(idx), 0)
        except Exception:
            try:
                handle.piece_deadline(int(idx), 0)
            except Exception:
                pass
        try:
            handle.piece_priority(int(idx), 7)
        except Exception:
            pass

    def _broadcast(self, name: str, payload: Dict[str, Any]) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(event_bus.broadcast(name, payload), self._loop)
            return
        try:
            asyncio.run(event_bus.broadcast(name, payload))
        except RuntimeError:
            pass

    def _extract_info_hash(self, magnet_uri: str) -> str:
        if _LT_AVAILABLE:
            try:
                params = lt.parse_magnet_uri(magnet_uri)
                info_hash = getattr(params, "info_hash", None)
                if info_hash is None and hasattr(params, "info_hashes"):
                    info_hash = params.info_hashes().v1
                if info_hash:
                    return str(info_hash)
            except Exception:
                pass
        match = re.search(r"xt=urn:btih:([A-Za-z0-9]+)", magnet_uri)
        return match.group(1) if match else ""

    def _get_info_hash(self, handle: Any) -> str:
        try:
            return str(handle.info_hash())
        except Exception:
            try:
                return str(handle.info_hashes().v1)
            except Exception:
                return ""

    def _find_info_hash(self, handle: Any) -> str:
        with self._lock:
            for key, item in self._items.items():
                if item.handle == handle:
                    return key
        return ""
