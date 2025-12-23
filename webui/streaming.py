import asyncio
import mimetypes
import time
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from agents.torrent.client import TorrentManager

_torrent_manager: Optional[TorrentManager] = None
_routes_registered = False


def set_torrent_manager(manager: TorrentManager) -> None:
    global _torrent_manager
    _torrent_manager = manager


def register_streaming_routes() -> None:
    global _routes_registered
    if _routes_registered:
        return
    try:
        from nicegui import app
    except Exception:
        return

    @app.get("/stream/{info_hash}")
    async def _stream(info_hash: str, request: Request):
        return await video_stream_endpoint(request)

    _routes_registered = True


async def video_stream_endpoint(request: Request):
    info_hash = request.path_params.get("info_hash", "")
    if not info_hash:
        raise HTTPException(status_code=400, detail="Missing info_hash")

    manager = _torrent_manager
    if manager is None:
        raise HTTPException(status_code=503, detail="Torrent manager unavailable")

    files = manager.get_files(info_hash)
    if not files:
        raise HTTPException(status_code=404, detail="Torrent not found")

    target = max(files, key=lambda item: item.get("size", 0))
    file_index = target["index"]
    file_path = Path(target["path"])
    file_size = int(target.get("size", 0))
    if file_size <= 0:
        raise HTTPException(status_code=404, detail="No files available")

    range_header = request.headers.get("range") or request.headers.get("Range")
    start, end = _parse_range(range_header, file_size)
    length = end - start + 1

    manager.prioritize_byte_range(info_hash, file_index, start, length)

    media_type, _ = mimetypes.guess_type(str(file_path))
    media_type = media_type or "application/octet-stream"

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(length),
    }

    async def _generator():
        timeout = 10.0
        wait_start = None

        if not file_path.exists():
            wait_start = time.monotonic()
            while not file_path.exists():
                if time.monotonic() - wait_start > timeout:
                    return
                await asyncio.sleep(0.1)

        with open(file_path, "rb") as handle:
            handle.seek(start)
            remaining = length
            current_offset = start

            while remaining > 0:
                piece_index = manager.get_piece_index(info_hash, file_index, current_offset)
                if piece_index is not None and not manager.has_piece(info_hash, piece_index):
                    wait_start = wait_start or time.monotonic()
                    if time.monotonic() - wait_start > timeout:
                        return
                    await asyncio.sleep(0.1)
                    continue

                wait_start = None
                chunk_size = min(64 * 1024, remaining)
                data = handle.read(chunk_size)
                if not data:
                    wait_start = wait_start or time.monotonic()
                    if time.monotonic() - wait_start > timeout:
                        return
                    await asyncio.sleep(0.1)
                    continue

                remaining -= len(data)
                current_offset += len(data)
                yield data

    return StreamingResponse(
        _generator(),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


def _parse_range(range_header: Optional[str], size: int) -> Tuple[int, int]:
    if not range_header:
        return 0, max(size - 1, 0)

    if not range_header.startswith("bytes="):
        return 0, max(size - 1, 0)

    ranges = range_header.replace("bytes=", "").split("-")
    if len(ranges) != 2:
        return 0, max(size - 1, 0)

    start_str, end_str = ranges
    if start_str == "":
        # suffix range
        length = int(end_str) if end_str.isdigit() else 0
        if length <= 0:
            return 0, max(size - 1, 0)
        start = max(size - length, 0)
        end = size - 1
        return start, end

    start = int(start_str) if start_str.isdigit() else 0
    end = int(end_str) if end_str.isdigit() else size - 1
    if start >= size:
        raise HTTPException(status_code=416, detail="Range not satisfiable")
    end = min(end, size - 1)
    if end < start:
        end = start
    return start, end
