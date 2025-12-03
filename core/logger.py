import logging
import re
from collections import deque
from typing import Deque, Dict, Any

from core.events import event_bus


COLOR_MAP = {
    "AI": "purple",
    "NET": "cyan",
    "LEGAL": "red",
    "ECO": "gold",
}


def colorize(msg: str) -> str:
    for key, color in COLOR_MAP.items():
        if f"[{key}]" in msg:
            return f"<span style='color:{color}'>{msg}</span>"
    return msg


class UIStreamHandler(logging.Handler):
    """Logging handler that pushes events to UI via event bus."""

    def __init__(self, buffer: Deque[str], maxlen: int = 1000):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            html = colorize(msg)
            self.buffer.append(html)
            # broadcast
            payload = {"message": html, "level": record.levelname}
            # fire-and-forget
            try:
                import asyncio
                asyncio.create_task(event_bus.broadcast("activity_log", payload))
            except Exception:
                pass
        except Exception:
            self.handleError(record)
