import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.events import event_bus
from agents.torrent.smart_client import SmartTorrentClient, RightsInfo
from cortex.media.scrubber import SmartScrubber
from cortex.compliance.rights import RightsManager, IndexPublisher, MagnetMetadata

try:
    import libtorrent as lt  # type: ignore
    _LT_AVAILABLE = True
except Exception:  # pragma: no cover
    lt = None  # type: ignore
    _LT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TorrentTask:
    info_hash: str
    magnet_uri: str
    handle: Any
    scrubber: SmartScrubber
    metadata: MagnetMetadata
    torrent_info: Optional[Any] = None
    priority_pieces: set = field(default_factory=set)
    processed_pieces: set = field(default_factory=set)
    pending_reads: set = field(default_factory=set)
    frames_dir: Path = field(default_factory=Path)
    rights_info: RightsInfo = field(default_factory=RightsInfo)
    expected_pieces: int = 0
    published: bool = False
    last_progress: float = 0.0


class MediaService:
    """Orchestrates torrent scrubbing, rights checks, and DHT publishing."""

    _instance: Optional["MediaService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        storage_dir: str = "storage/media",
        loop: Optional[asyncio.AbstractEventLoop] = None,
        kademlia: Any = None,
        dht_storage: Any = None,
        ledger: Any = None,
        frame_reward: float = 0.1,
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._loop = loop
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._session = None
        if _LT_AVAILABLE:
            self._session = lt.session()
            self._session.set_alert_mask(lt.alert.category_t.all_categories)

        self.scrubber_factory = SmartScrubber
        self.smart_client = SmartTorrentClient(ledger=ledger)
        self.rights_manager = RightsManager(
            persist_path=str(self.storage_dir / "rights"),
            kademlia=kademlia,
            dht_storage=dht_storage,
        )
        self.index_publisher = IndexPublisher(kademlia=kademlia, dht_storage=dht_storage)

        self._tasks: Dict[str, TorrentTask] = {}
        self._index: Dict[str, MagnetMetadata] = {}
        self._frame_reward = frame_reward
        self._similarity_threshold = 0.9
        self._economy_interval = 2.0
        self._last_economy_tick = 0.0

    @classmethod
    def get_instance(cls) -> "MediaService":
        return cls._instance or cls()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._session:
            self._thread = threading.Thread(target=self._loop_thread, daemon=True)
            self._thread.start()
        logger.info("[MEDIA] MediaService started")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("[MEDIA] MediaService stopped")

    def add_magnet(self, link: str) -> Optional[str]:
        if not link:
            return None
        info_hash = self._extract_info_hash(link)
        if not info_hash:
            info_hash = f"unknown-{int(time.time() * 1000)}"
        if not self._session:
            logger.warning("[MEDIA] libtorrent unavailable; cannot add magnet")
            return info_hash

        params = lt.add_torrent_params()
        params.save_path = str(self.storage_dir / "downloads")
        handle = lt.add_magnet_uri(self._session, link, params)

        task = self._create_task(handle, link, info_hash)
        with self._lock:
            self._tasks[task.info_hash] = task

        self.start()
        self._broadcast("media_added", {"info_hash": task.info_hash, "magnet": link})
        return task.info_hash

    def stream(self, info_hash: str) -> bool:
        task = self._tasks.get(info_hash)
        if not task or not task.handle:
            return False
        self.smart_client.set_play_now(task.handle, True)
        self._broadcast("media_stream", {"info_hash": info_hash})
        return True

    def search(self, query: str) -> List[Dict[str, Any]]:
        q = (query or "").strip().lower()
        results = []
        with self._lock:
            items = list(self._index.values())
        for metadata in items:
            if not q or self._matches_query(metadata, q):
                results.append(metadata.to_dict())
        return results

    def get_metadata(self, info_hash: str) -> Optional[MagnetMetadata]:
        return self._index.get(info_hash)

    # ------------------------------------------------------------------
    # Internal loop and alert handling
    # ------------------------------------------------------------------

    def _loop_thread(self) -> None:
        while self._running:
            try:
                alert = self._session.wait_for_alert(0.5) if self._session else None
                if alert:
                    alerts = self._session.pop_alerts()
                    for item in alerts:
                        self._handle_alert(item)
                self._tick_economy()
            except Exception as e:
                logger.warning(f"[MEDIA] Alert loop error: {e}")
                time.sleep(0.5)

    def _handle_alert(self, alert: Any) -> None:
        name = alert.__class__.__name__
        if name == "metadata_received_alert":
            self._handle_metadata(alert)
        elif name == "piece_finished_alert":
            self._handle_piece_finished(alert)
        elif name == "read_piece_alert":
            self._handle_read_piece(alert)
        elif name == "torrent_finished_alert":
            self._handle_torrent_finished(alert)

    def _handle_metadata(self, alert: Any) -> None:
        handle = getattr(alert, "handle", None)
        if not handle:
            return
        info_hash = self._get_info_hash(handle)
        task = self._tasks.get(info_hash)
        if not task:
            task = self._create_task(handle, "", info_hash)
            self._tasks[info_hash] = task

        torrent_info = handle.torrent_file() if hasattr(handle, "torrent_file") else None
        task.torrent_info = torrent_info
        if torrent_info:
            task.metadata.name = torrent_info.name()
            task.metadata.size_bytes = torrent_info.total_size()
            task.metadata.piece_length = torrent_info.piece_length()
            task.metadata.num_pieces = torrent_info.num_pieces()

        priority = task.scrubber.select_priority_pieces(torrent_info)
        task.priority_pieces = set(priority)
        task.expected_pieces = len(task.priority_pieces)
        task.frames_dir = self.storage_dir / "frames" / info_hash
        task.frames_dir.mkdir(parents=True, exist_ok=True)
        self.smart_client.prioritize_pieces(handle, priority)

        with self._lock:
            self._index[info_hash] = task.metadata
        self._broadcast(
            "media_metadata",
            {"info_hash": info_hash, "metadata": task.metadata.to_dict()},
        )

    def _handle_piece_finished(self, alert: Any) -> None:
        handle = getattr(alert, "handle", None)
        if not handle:
            return
        info_hash = self._get_info_hash(handle)
        task = self._tasks.get(info_hash)
        if not task:
            return
        piece_index = self._read_attr(alert, ["piece_index", "piece", "index"], -1)
        if piece_index < 0:
            return
        if piece_index not in task.priority_pieces:
            return
        if piece_index in task.processed_pieces or piece_index in task.pending_reads:
            return
        try:
            handle.read_piece(piece_index)
            task.pending_reads.add(piece_index)
        except Exception as e:
            logger.warning(f"[MEDIA] read_piece failed for {piece_index}: {e}")

    def _handle_read_piece(self, alert: Any) -> None:
        handle = getattr(alert, "handle", None)
        if not handle:
            return
        info_hash = self._get_info_hash(handle)
        task = self._tasks.get(info_hash)
        if not task:
            return
        piece_index = self._read_attr(alert, ["piece", "piece_index", "index"], -1)
        if piece_index not in task.pending_reads:
            return
        buffer = getattr(alert, "buffer", b"")
        piece_data = bytes(buffer) if buffer is not None else b""
        if not piece_data:
            task.pending_reads.discard(piece_index)
            return
        entry = None
        try:
            entry = task.scrubber.process_piece(
                piece_data,
                piece_index,
                torrent_info=task.torrent_info,
                output_dir=str(task.frames_dir),
            )
        except Exception as e:
            logger.warning(f"[MEDIA] Scrub failed for piece {piece_index}: {e}")
        finally:
            task.pending_reads.discard(piece_index)

        if not entry:
            return

        time_sec = task.scrubber._resolve_time_for_piece(
            piece_index, task.torrent_info, None
        )
        frame_path = task.frames_dir / f"frame_{int(time_sec * 1000):010d}.jpg"
        entry["piece_index"] = piece_index
        entry["time_sec"] = time_sec
        if frame_path.exists():
            entry["frame_path"] = str(frame_path)

        with self._lock:
            task.metadata.storyboard.append(entry)
            task.processed_pieces.add(piece_index)
        self._broadcast_progress(task)

        if not task.published and task.expected_pieces > 0:
            if len(task.processed_pieces) >= task.expected_pieces:
                self._finalize_index(task)

    def _handle_torrent_finished(self, alert: Any) -> None:
        handle = getattr(alert, "handle", None)
        if not handle:
            return
        info_hash = self._get_info_hash(handle)
        task = self._tasks.get(info_hash)
        if task and not task.published:
            self._finalize_index(task)

    def _tick_economy(self) -> None:
        now = time.monotonic()
        if now - self._last_economy_tick < self._economy_interval:
            return
        self._last_economy_tick = now
        for task in list(self._tasks.values()):
            if not task.rights_info.monetized:
                continue
            self._run_async(
                self.smart_client.tick(task.handle, task.info_hash, task.rights_info),
                wait=False,
            )

    # ------------------------------------------------------------------
    # Rights and publishing
    # ------------------------------------------------------------------

    def _finalize_index(self, task: TorrentTask) -> None:
        best_similarity = 0.0
        best_content_id = None
        best_beneficiary = None
        for entry in task.metadata.storyboard:
            phash = entry.get("img_hash", "")
            match = self._match_rights_phash(phash)
            if match and match[0] > best_similarity:
                best_similarity, best_content_id, best_beneficiary = match

        if best_similarity >= self._similarity_threshold:
            task.metadata.rights_status = "LICENSED"
            task.rights_info.status = "LICENSED"
            task.rights_info.revenue_share = True
        else:
            task.metadata.rights_status = "UNLICENSED"
            task.rights_info.status = "UNLICENSED"
        task.rights_info.similarity = best_similarity

        if best_content_id:
            beneficiary = self._run_async(
                self.rights_manager.resolve_beneficiary(best_content_id, best_beneficiary),
                wait=True,
            )
            task.metadata.beneficiary_address = beneficiary
            task.rights_info.beneficiary_address = beneficiary
            task.rights_info.similarity = best_similarity

        self._run_async(self.index_publisher.publish(task.metadata), wait=False)
        task.published = True
        self._broadcast(
            "media_index_complete",
            {"info_hash": task.info_hash, "metadata": task.metadata.to_dict()},
        )

    def _match_rights_phash(self, phash: str) -> Optional[tuple]:
        if not phash:
            return None
        try:
            embedding = self.rights_manager._phash_to_vector(phash)
            match = self.rights_manager._query_best_match(embedding)
            if not match:
                return None
            match_id, meta = match
            match_phash = meta.get("phash", "")
            similarity = self.rights_manager._phash_similarity(phash, match_phash)
            return similarity, meta.get("content_id"), meta.get("beneficiary_address")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _broadcast_progress(self, task: TorrentTask) -> None:
        processed = len(task.processed_pieces)
        total = max(task.expected_pieces, 1)
        earned = processed * self._frame_reward
        payload = {
            "info_hash": task.info_hash,
            "processed": processed,
            "total": total,
            "earned": earned,
            "metadata": task.metadata.to_dict(),
        }
        self._broadcast("media_index_progress", payload)
        self._broadcast("media_storyboard", payload)

    def _broadcast(self, name: str, payload: Dict[str, Any]) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(event_bus.broadcast(name, payload), self._loop)
            return
        try:
            asyncio.run(event_bus.broadcast(name, payload))
        except RuntimeError:
            pass

    def _run_async(self, coro: Any, wait: bool = False) -> Optional[Any]:
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            if wait:
                try:
                    return future.result(timeout=5)
                except Exception:
                    return None
            return None
        try:
            return asyncio.run(coro)
        except RuntimeError:
            return None

    def _matches_query(self, metadata: MagnetMetadata, query: str) -> bool:
        if query in (metadata.info_hash or "").lower():
            return True
        if query in (metadata.name or "").lower():
            return True
        for entry in metadata.storyboard:
            desc = entry.get("desc", "")
            tags = entry.get("tags", [])
            if isinstance(desc, str) and query in desc.lower():
                return True
            if any(query in str(t).lower() for t in tags):
                return True
        return False

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

    def _create_task(self, handle: Any, magnet_uri: str, info_hash: str) -> TorrentTask:
        scrubber = self.scrubber_factory()
        metadata = MagnetMetadata(
            info_hash=info_hash,
            name="",
            size_bytes=0,
            piece_length=0,
            num_pieces=0,
            magnet_uri=magnet_uri,
            storyboard=[],
        )
        return TorrentTask(
            info_hash=info_hash,
            magnet_uri=magnet_uri,
            handle=handle,
            scrubber=scrubber,
            metadata=metadata,
            frames_dir=self.storage_dir / "frames" / info_hash,
        )

    def _get_info_hash(self, handle: Any) -> str:
        try:
            return str(handle.info_hash())
        except Exception:
            try:
                return str(handle.info_hashes().v1)
            except Exception:
                return ""

    @staticmethod
    def _read_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                return value() if callable(value) else value
        return default
