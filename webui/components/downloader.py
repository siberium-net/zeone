from nicegui import ui
from typing import Dict, Any


class DownloadManager:
    """Floating panel to show model download progress."""

    def __init__(self):
        self._container = None
        self._label = None
        self._progress = None
        self._visible = False
        self._source_label = None
        self._peers_label = None
        self._mounted = False

    def mount(self):
        if self._mounted:
            return
        with ui.footer(value=True).classes('fixed bottom-0 left-0 right-0 p-2 bg-gray-800 text-white'):
            with ui.row().classes('items-center gap-4 w-full'):
                self._label = ui.label('Model: -')
                self._progress = ui.linear_progress(value=0).props('rounded stripe color="blue"').classes('w-1/2')
                self._source_label = ui.label('Source: -')
                self._peers_label = ui.label('Peers: -')
        self.hide()
        self._mounted = True

    def update(self, payload: Dict[str, Any]) -> None:
        self.show()
        self._label.text = f"Model: {payload.get('model')}"
        percent = payload.get('percent', 0) / 100
        self._progress.value = percent
        speed = payload.get('speed', 0)
        mbps = speed / (1024 * 1024)
        self._source_label.text = f"{payload.get('percent', 0)}% | {mbps:.2f} MB/s | Source: {payload.get('source', '-')}"
        self._peers_label.text = f"Peers: {payload.get('peers', 0)}"

    def show(self):
        if not self._mounted:
            return
        self._visible = True
        self._progress.visible = True
        self._label.visible = True
        self._source_label.visible = True
        self._peers_label.visible = True

    def hide(self):
        if not self._mounted:
            return
        self._visible = False
        self._progress.visible = False
        self._label.visible = False
        self._source_label.visible = False
        self._peers_label.visible = False
