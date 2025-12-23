import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import requests

from cortex.archivist.vector_store import VectorStore
from cortex.vision import VisionEngine
from agents.local_llm import OllamaAgent

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    import libtorrent as lt  # type: ignore
    _LT_AVAILABLE = True
except Exception:  # pragma: no cover
    lt = None  # type: ignore
    _LT_AVAILABLE = False

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_EXTS = {".mp3", ".flac", ".aac", ".wav", ".m4a"}
EBOOK_EXTS = {".pdf", ".epub", ".mobi"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z"}


@dataclass
class MagnetRecord:
    info_hash: str
    magnet_uri: str
    name: str
    files: List[str]
    summary: str
    tags: List[str]
    trust_score: float
    preview: Dict[str, Any]
    source: str
    created_at: float = field(default_factory=time.time)


class MagnetIndexer:
    """Crawls magnets, fetches metadata, runs AI analysis, and stores vectors."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[OllamaAgent] = None,
        vision: Optional[VisionEngine] = None,
        ffmpeg_path: str = "ffmpeg",
        temp_dir: Optional[str] = None,
        metadata_timeout: float = 20.0,
    ):
        self.vector_store = vector_store or VectorStore()
        self.llm = llm
        self._vision = vision
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "magnet_indexer"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_timeout = metadata_timeout

        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session = lt.session() if _LT_AVAILABLE else None
        if self._session:
            try:
                self._session.set_alert_mask(lt.alert.category_t.all_categories)
            except Exception:
                pass

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop_worker())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def enqueue(self, magnet_uri: str, *, trust_score: float = 0.0, source: str = "manual", preview_path: Optional[str] = None, torrent_url: Optional[str] = None) -> None:
        payload = {
            "magnet_uri": magnet_uri,
            "trust_score": trust_score,
            "source": source,
            "preview_path": preview_path,
            "torrent_url": torrent_url,
        }
        self._queue.put_nowait(payload)

    async def _loop_worker(self) -> None:
        while self._running:
            try:
                payload = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self.index_magnet(**payload)
            except Exception as e:
                logger.warning(f"[MAGNET] Index failed: {e}")

    # ------------------------------------------------------------------
    # Crawlers
    # ------------------------------------------------------------------

    def crawl_rss(self, urls: Sequence[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                results.extend(self._parse_rss(response.text, url))
            except Exception as e:
                logger.warning(f"[MAGNET] RSS crawl failed {url}: {e}")
        return results

    def crawl_dht(self) -> List[Dict[str, Any]]:
        logger.info("[MAGNET] DHT crawling not implemented; plug in mainline DHT here")
        return []

    # ------------------------------------------------------------------
    # Indexing pipeline
    # ------------------------------------------------------------------

    async def index_magnet(
        self,
        magnet_uri: str,
        trust_score: float = 0.0,
        source: str = "manual",
        preview_path: Optional[str] = None,
        torrent_url: Optional[str] = None,
    ) -> Optional[MagnetRecord]:
        if not magnet_uri and not torrent_url:
            return None

        metadata = await self.fetch_metadata(magnet_uri, torrent_url=torrent_url)
        name = metadata.get("name") or ""
        files = metadata.get("files") or []
        info_hash = metadata.get("info_hash") or ""

        summary, tags = self._infer_from_names(name, files)
        summary, tags = await self._llm_enrich(name, files, summary, tags)
        preview = self._analyze_preview(preview_path)
        if preview.get("description"):
            summary = f"{summary} | {preview['description']}"
        tags.extend(preview.get("tags", []))

        record = MagnetRecord(
            info_hash=info_hash,
            magnet_uri=magnet_uri,
            name=name,
            files=files,
            summary=summary.strip() or name,
            tags=self._dedupe(tags),
            trust_score=trust_score,
            preview=preview,
            source=source,
        )

        payload = {
            "info_hash": record.info_hash,
            "magnet_uri": record.magnet_uri,
            "name": record.name,
            "files": record.files,
            "tags": record.tags,
            "trust_score": record.trust_score,
            "source": record.source,
        }
        self.vector_store.embed_and_store([record.summary], metadata=payload)
        return record

    async def fetch_metadata(self, magnet_uri: str, torrent_url: Optional[str] = None) -> Dict[str, Any]:
        if torrent_url:
            data = self._load_torrent_data(torrent_url)
            info = self._parse_torrent_data(data) if data else None
            if info:
                return info
        if magnet_uri and magnet_uri.startswith("magnet:"):
            return await self._fetch_magnet_metadata(magnet_uri)
        data = self._load_torrent_data(magnet_uri)
        info = self._parse_torrent_data(data) if data else None
        return info or self._parse_magnet(magnet_uri)

    async def _fetch_magnet_metadata(self, magnet_uri: str) -> Dict[str, Any]:
        parsed = self._parse_magnet(magnet_uri)
        if not self._session or not _LT_AVAILABLE:
            return parsed

        params = lt.add_torrent_params()
        params.save_path = str(self.temp_dir)
        handle = lt.add_magnet_uri(self._session, magnet_uri, params)
        start = time.time()

        while time.time() - start < self.metadata_timeout:
            if handle.has_metadata():
                try:
                    info = handle.torrent_file()
                    return self._metadata_from_torrent_info(info, parsed)
                except Exception:
                    break
            alert = self._session.wait_for_alert(0.5)
            if alert:
                _ = self._session.pop_alerts()
            await asyncio.sleep(0.1)
        return parsed

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _infer_from_names(self, name: str, files: Sequence[str]) -> Tuple[str, List[str]]:
        haystack = " ".join([name] + list(files)).lower()
        tags: List[str] = []

        content_type = "unknown"
        if any(Path(f).suffix.lower() in VIDEO_EXTS for f in files):
            content_type = "video"
        elif any(Path(f).suffix.lower() in AUDIO_EXTS for f in files):
            content_type = "audio"
        elif any(Path(f).suffix.lower() in EBOOK_EXTS for f in files):
            content_type = "ebook"
        elif any(Path(f).suffix.lower() in ARCHIVE_EXTS for f in files):
            content_type = "archive"

        tags.append(content_type)

        for token in ["1080p", "720p", "2160p", "4k", "hdr", "x264", "x265", "h264", "h265", "aac", "dts"]:
            if token in haystack:
                tags.append(token)

        match = re.search(r"\b(19|20)\d{2}\b", haystack)
        if match:
            tags.append(match.group(0))

        if re.search(r"s\d{2}e\d{2}", haystack):
            tags.append("series")
        elif content_type == "video":
            tags.append("movie")

        summary = f"{content_type} content: {name}".strip()
        return summary, tags

    async def _llm_enrich(self, name: str, files: Sequence[str], summary: str, tags: List[str]) -> Tuple[str, List[str]]:
        if not self.llm:
            return summary, tags
        prompt = (
            "Classify this torrent from filenames and title. "
            "Return a short description and 5-10 comma-separated tags.\n\n"
            f"Title: {name}\n"
            f"Files: {', '.join(files[:20])}"
        )
        try:
            result, _ = await self.llm.execute({"prompt": prompt})
            text = result.get("response") if isinstance(result, dict) else str(result)
            if not text:
                return summary, tags
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                summary = lines[0]
            if len(lines) > 1:
                maybe_tags = [t.strip() for t in re.split(r"[,;]", lines[1]) if t.strip()]
                tags.extend(maybe_tags)
        except Exception as e:
            logger.warning(f"[MAGNET] LLM enrich failed: {e}")
        return summary, tags

    def _analyze_preview(self, preview_path: Optional[str]) -> Dict[str, Any]:
        if not preview_path:
            return {}
        path = Path(preview_path)
        if not path.exists():
            return {}

        vision = self._ensure_vision()
        if not vision:
            return {}

        if path.suffix.lower() in VIDEO_EXTS:
            frame = self._extract_preview_frame(path)
            if frame:
                try:
                    return vision.analyze_image(str(frame))
                finally:
                    frame.unlink(missing_ok=True)
            return {}

        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return vision.analyze_image(str(path))
        return {}

    def _extract_preview_frame(self, video_path: Path) -> Optional[Path]:
        frame_path = self.temp_dir / f"preview_{int(time.time() * 1000)}.jpg"
        if shutil.which(self.ffmpeg_path):
            cmd = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "5",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ]
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode == 0 and frame_path.exists():
                return frame_path

        if _CV2_AVAILABLE:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(frame_path), frame)
                return frame_path
        return None

    def _ensure_vision(self) -> Optional[VisionEngine]:
        if self._vision:
            return self._vision
        try:
            self._vision = VisionEngine()
        except Exception as e:
            logger.warning(f"[MAGNET] VisionEngine unavailable: {e}")
            self._vision = None
        return self._vision

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_magnet(self, magnet_uri: str) -> Dict[str, Any]:
        info_hash = ""
        name = ""
        trackers: List[str] = []
        if magnet_uri.startswith("magnet:"):
            parsed = urlparse(magnet_uri)
            params = parse_qs(parsed.query)
            xt = params.get("xt", [""])[0]
            if xt.startswith("urn:btih:"):
                info_hash = xt.split(":")[-1]
            name = params.get("dn", [""])[0]
            trackers = params.get("tr", [])
        return {
            "info_hash": info_hash,
            "name": name,
            "trackers": trackers,
            "files": [],
        }

    def _load_torrent_data(self, link: str) -> Optional[bytes]:
        if not link:
            return None
        if link.startswith("http://") or link.startswith("https://"):
            try:
                response = requests.get(link, timeout=10)
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.warning(f"[MAGNET] Failed to download torrent: {e}")
                return None
        path = Path(link)
        if path.exists() and path.is_file():
            try:
                return path.read_bytes()
            except Exception:
                return None
        return None

    def _parse_torrent_data(self, data: Optional[bytes]) -> Optional[Dict[str, Any]]:
        if not data or not _LT_AVAILABLE:
            return None
        info = None
        try:
            info = lt.torrent_info(data)
        except Exception:
            try:
                info = lt.torrent_info(lt.bdecode(data))
            except Exception:
                info = None
        if not info:
            return None
        return self._metadata_from_torrent_info(info, {})

    def _metadata_from_torrent_info(self, info: Any, base: Dict[str, Any]) -> Dict[str, Any]:
        files = []
        try:
            storage = info.files()
            for idx in range(storage.num_files()):
                files.append(storage.file_path(idx))
        except Exception:
            pass
        info_hash = ""
        try:
            info_hash = str(info.info_hash())
        except Exception:
            try:
                info_hash = str(info.info_hashes().v1)
            except Exception:
                info_hash = ""
        metadata = {
            "info_hash": info_hash,
            "name": info.name(),
            "files": files,
            "total_size": info.total_size(),
        }
        metadata.update(base)
        return metadata

    def _parse_rss(self, xml_text: str, source: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        root = ET.fromstring(xml_text)
        for item in root.iter():
            if self._strip_tag(item.tag) != "item":
                continue
            magnet = None
            torrent_url = None
            seeders = None
            leechers = None
            title = ""
            for child in item:
                tag = self._strip_tag(child.tag)
                text = (child.text or "").strip()
                if tag == "title":
                    title = text
                if tag in {"link", "guid"} and text.startswith("magnet:"):
                    magnet = text
                if tag == "enclosure":
                    url = child.attrib.get("url", "")
                    if url.startswith("magnet:"):
                        magnet = url
                    elif url.endswith(".torrent"):
                        torrent_url = url
                if tag in {"seeders", "seeds"}:
                    seeders = self._safe_int(text)
                if tag in {"leechers", "peers"}:
                    leechers = self._safe_int(text)
            if not magnet and torrent_url:
                magnet = ""
            if magnet is None and not torrent_url:
                continue
            trust = self._compute_trust_score(seeders, leechers)
            results.append(
                {
                    "magnet_uri": magnet,
                    "torrent_url": torrent_url,
                    "trust_score": trust,
                    "source": source,
                    "title": title,
                }
            )
        return results

    @staticmethod
    def _strip_tag(tag: str) -> str:
        if "}" in tag:
            tag = tag.split("}")[-1]
        return tag.lower()

    @staticmethod
    def _safe_int(value: str) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _compute_trust_score(seeders: Optional[int], leechers: Optional[int]) -> float:
        if seeders is None and leechers is None:
            return 0.0
        seeders = seeders or 0
        leechers = leechers or 0
        return seeders / max(seeders + leechers, 1)

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for v in values:
            if not v:
                continue
            key = str(v).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(str(v))
        return out
