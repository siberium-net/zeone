import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from cortex.vision import VisionEngine

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScrubPoint:
    piece_index: int
    ratio: float
    time_sec: float


def _read_attr(obj: Any, names: Sequence[str], default: Optional[Any] = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) else value
    return default


def _format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class SmartScrubber:
    """Selects torrent pieces for preview scrubbing and builds storyboards."""

    def __init__(
        self,
        vision: Optional[VisionEngine] = None,
        avg_bitrate_mbps: float = 5.0,
        ffmpeg_path: str = "ffmpeg",
        tmp_dir: Optional[str] = None,
    ):
        self.vision = vision
        self.avg_bitrate_mbps = avg_bitrate_mbps
        self.ffmpeg_path = ffmpeg_path
        self.tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir()) / "scrubber"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.scrub_points: List[ScrubPoint] = []

    def select_priority_pieces(
        self,
        torrent_info: Any,
        video_duration: Optional[float] = None,
    ) -> List[int]:
        total_size, num_pieces, _piece_length = self._extract_torrent_stats(torrent_info)
        if num_pieces <= 0:
            logger.warning("[SCRUBBER] Unable to determine num_pieces")
            return []
        duration = self._resolve_duration(video_duration, total_size)
        points = self._build_scrub_points(num_pieces, duration)
        self.scrub_points = points
        priority: List[int] = []
        seen = set()
        for point in points:
            if point.piece_index in seen:
                continue
            seen.add(point.piece_index)
            priority.append(point.piece_index)
        return priority

    def process_piece(
        self,
        piece_payload: Union[str, bytes, Path],
        piece_index: int,
        torrent_info: Optional[Any] = None,
        video_duration: Optional[float] = None,
        output_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        piece_path, cleanup = self._materialize_piece(piece_payload, piece_index)
        try:
            time_sec = self._resolve_time_for_piece(
                piece_index=piece_index,
                torrent_info=torrent_info,
                video_duration=video_duration,
            )
            frame_path = self._extract_keyframe(piece_path, time_sec, output_dir)
            if not frame_path:
                return None
            vision = self._ensure_vision()
            analysis = vision.analyze_image(str(frame_path)) if vision else {}
            entry = {
                "time": _format_time(time_sec),
                "desc": analysis.get("description", ""),
                "tags": analysis.get("tags", []) if analysis else [],
                "img_hash": analysis.get("phash", "") if analysis else "",
            }
            return entry
        finally:
            if cleanup:
                try:
                    piece_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def build_storyboard(self, entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [entry for entry in entries if entry]

    def _ensure_vision(self) -> Optional[VisionEngine]:
        if self.vision:
            return self.vision
        try:
            self.vision = VisionEngine(device="cuda")
        except Exception as e:
            logger.warning(f"[SCRUBBER] VisionEngine unavailable: {e}")
            self.vision = None
        return self.vision

    def _extract_torrent_stats(self, torrent_info: Any) -> Tuple[int, int, int]:
        total_size = _read_attr(
            torrent_info,
            ["total_size", "size", "total_length", "length", "total_size_bytes"],
            0,
        )
        piece_length = _read_attr(
            torrent_info,
            ["piece_length", "piece_size", "piece_len"],
            0,
        )
        num_pieces = _read_attr(
            torrent_info,
            ["num_pieces", "piece_count", "pieces"],
            0,
        )
        if isinstance(num_pieces, (list, tuple)):
            num_pieces = len(num_pieces)
        if not num_pieces and total_size and piece_length:
            num_pieces = int(math.ceil(total_size / float(piece_length)))
        return int(total_size or 0), int(num_pieces or 0), int(piece_length or 0)

    def _resolve_duration(self, video_duration: Optional[float], total_size: int) -> float:
        if video_duration and video_duration > 0:
            return float(video_duration)
        if total_size <= 0:
            return 0.0
        bytes_per_sec = (self.avg_bitrate_mbps * 1_000_000) / 8.0
        return float(total_size) / max(bytes_per_sec, 1.0)

    def _build_scrub_points(self, num_pieces: int, duration: float) -> List[ScrubPoint]:
        ratios: List[float] = [0.0, 0.01]
        ratios.extend([i / 100.0 for i in range(10, 100, 10)])
        ratios.extend([0.99, 1.0])
        points = []
        for ratio in ratios:
            piece_index = int(round(ratio * max(num_pieces - 1, 0)))
            piece_index = max(0, min(num_pieces - 1, piece_index)) if num_pieces > 0 else 0
            time_sec = duration * ratio if duration > 0 else 0.0
            points.append(ScrubPoint(piece_index=piece_index, ratio=ratio, time_sec=time_sec))
        return points

    def _resolve_time_for_piece(
        self,
        piece_index: int,
        torrent_info: Optional[Any] = None,
        video_duration: Optional[float] = None,
    ) -> float:
        for point in self.scrub_points:
            if point.piece_index == piece_index:
                return point.time_sec
        if torrent_info is None:
            return 0.0
        total_size, num_pieces, _piece_len = self._extract_torrent_stats(torrent_info)
        duration = self._resolve_duration(video_duration, total_size)
        if num_pieces <= 1 or duration <= 0:
            return 0.0
        ratio = piece_index / float(num_pieces - 1)
        return max(ratio * duration, 0.0)

    def _materialize_piece(
        self,
        piece_payload: Union[str, bytes, Path],
        piece_index: int,
    ) -> Tuple[Path, bool]:
        if isinstance(piece_payload, Path):
            return piece_payload, False
        if isinstance(piece_payload, str):
            return Path(piece_payload), False
        tmp_path = self.tmp_dir / f"piece_{piece_index}.bin"
        tmp_path.write_bytes(piece_payload)
        return tmp_path, True

    def _extract_keyframe(
        self,
        video_path: Path,
        time_sec: float,
        output_dir: Optional[str] = None,
    ) -> Optional[Path]:
        frames_dir = Path(output_dir) if output_dir else self.tmp_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / f"frame_{int(time_sec * 1000):010d}.jpg"
        if frame_path.exists():
            return frame_path

        if self._extract_keyframe_ffmpeg(video_path, frame_path, time_sec):
            return frame_path
        if self._extract_keyframe_cv2(video_path, frame_path, time_sec):
            return frame_path
        return None

    def _extract_keyframe_ffmpeg(self, video_path: Path, frame_path: Path, time_sec: float) -> bool:
        if not shutil.which(self.ffmpeg_path):
            return False
        timestamp = max(time_sec, 0.0)
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-skip_frame",
            "nokey",
            "-ss",
            str(timestamp),
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
            return True
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        return result.returncode == 0 and frame_path.exists()

    def _extract_keyframe_cv2(self, video_path: Path, frame_path: Path, time_sec: float) -> bool:
        if not _CV2_AVAILABLE:
            return False
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_MSEC, max(time_sec, 0.0) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return False
        return bool(cv2.imwrite(str(frame_path), frame))
