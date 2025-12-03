from nicegui import ui
from typing import List, Dict, Any


class Gallery:
    """Simple image gallery with captions."""

    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self._grid = None

    def mount(self):
        with ui.grid(columns=4).classes('gap-2') as grid:
            self._grid = grid
        self.refresh()

    def add_item(self, item: Dict[str, Any]) -> None:
        self.items.append(item)
        self.refresh()

    def refresh(self):
        if not self._grid:
            return
        self._grid.clear()
        with self._grid:
            for item in self.items:
                path = item.get("path", "")
                caption = item.get("caption", "")
                tags = ", ".join(item.get("tags", []))
                with ui.card().classes('w-64'):
                    if path:
                        ui.image(path).classes('w-full h-40 object-cover')
                    ui.label(caption[:120]).classes('text-sm')
                    if tags:
                        ui.label(f"Objects: {tags}").classes('text-xs text-gray-500')
