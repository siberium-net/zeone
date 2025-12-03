from nicegui import ui
from typing import List, Dict, Any
from datetime import datetime

from cortex.library_api import LibraryAPI
from core.events import event_bus


class LibraryTab:
    def __init__(self, db_path, parent_app):
        self.api = LibraryAPI(db_path)
        self.parent = parent_app
        self._table = None
        self._dialog = None
        self._detail_title = None
        self._detail_body = None
        self._register_listener()

    def _register_listener(self):
        async def on_new(payload):
            if self._table:
                await self._refresh()
        try:
            import asyncio
            asyncio.create_task(event_bus.subscribe("knowledge_added", on_new))
        except Exception:
            pass

    def create_page(self):
        @ui.page('/library')
        async def library():
            await self.parent._create_header()
            await self.parent._create_sidebar()

            with ui.column().classes('w-full p-4'):
                ui.label('Knowledge Library').classes('text-2xl font-bold mb-4')

                with ui.row().classes('gap-2 mb-2'):
                    search_box = ui.input('Search').classes('w-64')
                    ui.button('Search', icon='search', on_click=lambda: self._search(search_box.value))
                    ui.button('Refresh', icon='refresh', on_click=self._refresh)

                columns = [
                    {'name': 'type', 'label': 'Type', 'field': 'type', 'align': 'left'},
                    {'name': 'title', 'label': 'Title', 'field': 'title'},
                    {'name': 'tags', 'label': 'Tags', 'field': 'tags'},
                    {'name': 'date', 'label': 'Date', 'field': 'date'},
                    {'name': 'legal', 'label': 'Legal', 'field': 'legal'},
                ]
                self._table = ui.table(columns=columns, rows=[], row_key='id', on_row_click=self._show_details).classes('w-full')
                await self._refresh()

                self._dialog = ui.dialog()
                with self._dialog, ui.card().classes('w-[700px]'):
                    self._detail_title = ui.label('').classes('text-xl font-bold')
                    self._detail_body = ui.column().classes('w-full')
                    ui.button('Close', on_click=self._dialog.close).classes('mt-2')

    async def _refresh(self):
        items = self.api.get_recent_items()
        rows = [self._to_row(i) for i in items]
        if self._table:
            self._table.rows = rows
            self._table.update()

    async def _search(self, query: str):
        if not query:
            await self._refresh()
            return
        items = self.api.search_items(query)
        rows = [self._to_row(i) for i in items]
        if self._table:
            self._table.rows = rows
            self._table.update()

    def _to_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        date_str = datetime.fromtimestamp(item["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if item.get("created_at") else "-"
        status = item.get("compliance_status", "UNKNOWN")
        legal = status
        tags = ", ".join(item.get("tags", []))
        return {
            "id": item["id"],
            "type": "doc",
            "title": item.get("path") or item.get("summary", "")[:50],
            "tags": tags,
            "date": date_str,
            "legal": legal,
            "raw": item,
        }

    def _show_details(self, e):
        row = e.args.get('row') or {}
        raw = row.get("raw", {})
        if self._detail_body and self._detail_title:
            self._detail_title.text = raw.get("path") or raw.get("summary", "")
            self._detail_body.clear()
            with self._detail_body:
                ui.label(f"Summary: {raw.get('summary','')}")
                ui.label(f"Tags: {', '.join(raw.get('tags', []))}")
                ui.label(f"Compliance: {raw.get('compliance_status','UNKNOWN')}")
                ui.label(f"Metadata: {raw.get('metadata',{})}")
            self._dialog.open()
