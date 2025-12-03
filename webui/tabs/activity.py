from collections import deque
from typing import List

from nicegui import ui

from core.events import event_bus


class ActivityTab:
    def __init__(self, buffer: deque):
        self.buffer = buffer
        self._log = None
        self._filters = {"NET": True, "AI": True, "ERROR": True, "LEGAL": True, "ECO": True}

    def create_page(self, parent):
        @ui.page('/activity')
        async def activity():
            await parent._create_header()
            await parent._create_sidebar()

            with ui.column().classes('w-full p-4'):
                ui.label('Activity Stream').classes('text-2xl font-bold mb-4')
                with ui.row().classes('gap-2 mb-2'):
                    for key in ["NET", "AI", "LEGAL", "ECO"]:
                        ui.checkbox(f"Show {key}", value=True, on_change=self._toggle(key))
                    ui.checkbox("Show Errors", value=True, on_change=self._toggle("ERROR"))

                self._log = ui.log(max_lines=1000).classes('w-full h-[600px]')
                # preload buffer
                for msg in list(self.buffer)[-200:]:
                    self._log.push(msg, keep_html=True)
                # subscribe to events
                await event_bus.subscribe("activity_log", self._on_event)

    def _toggle(self, key: str):
        def handler(e):
            self._filters[key] = e.value
        return handler

    async def _on_event(self, payload):
        msg = payload.get("message", "")
        level = payload.get("level", "")
        if not self._filter_msg(msg, level):
            return
        if self._log:
            self._log.push(msg, keep_html=True)
            self._log.scroll_to_end()

    def _filter_msg(self, msg: str, level: str) -> bool:
        if "NET" in msg and not self._filters.get("NET"):
            return False
        if "AI" in msg and not self._filters.get("AI"):
            return False
        if "LEGAL" in msg and not self._filters.get("LEGAL"):
            return False
        if "ECO" in msg and not self._filters.get("ECO"):
            return False
        if level == "ERROR" and not self._filters.get("ERROR"):
            return False
        return True
