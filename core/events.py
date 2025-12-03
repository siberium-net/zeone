import asyncio
from typing import Callable, Dict, Any, List


class EventBus:
    """Minimal async event bus for in-process notifications."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        async with self._lock:
            self._subscribers.setdefault(event_name, []).append(callback)

    async def unsubscribe(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        async with self._lock:
            if event_name in self._subscribers:
                self._subscribers[event_name] = [cb for cb in self._subscribers[event_name] if cb != callback]

    async def broadcast(self, event_name: str, payload: Dict[str, Any]) -> None:
        callbacks = []
        async with self._lock:
            callbacks = list(self._subscribers.get(event_name, []))
        for cb in callbacks:
            try:
                result = cb(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # swallow to avoid breaking other listeners
                continue


# Global singleton
event_bus = EventBus()
