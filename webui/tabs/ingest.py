import asyncio
import os
from pathlib import Path
from typing import Optional

from nicegui import ui

from cortex.archivist import AsyncFileScanner, DocumentProcessor
from core.events import event_bus
from webui.components.gallery import Gallery


class IngestTab:
    def __init__(self, gallery: Gallery):
        self._progress = None
        self._path_input = None
        self._text_flag = True
        self._vision_flag = False
        self._video_flag = False
        self._dream_flag = False
        self.gallery = gallery

    def create_page(self, parent):
        @ui.page('/ingest')
        async def ingest():
            await parent._create_header()
            await parent._create_sidebar()

            with ui.column().classes('w-full p-4'):
                ui.label('Ingestion Control').classes('text-2xl font-bold mb-4')
                self._path_input = ui.input('Directory path', value=str(Path.cwd())).classes('w-full')
                self._text_flag = ui.checkbox('Process Text/PDF', value=True)
                self._vision_flag = ui.checkbox('Enable Vision (GPU Heavy)', value=False)
                self._video_flag = ui.checkbox('Enable Video Analysis (Slow)', value=False)
                self._dream_flag = ui.checkbox('Dream Mode (Only when idle)', value=False)

                ui.button('Start Ingestion', icon='play_arrow', on_click=self._start_ingest)
                self._progress = ui.label('Idle').classes('text-sm text-gray-500')

                ui.separator()
                ui.label('Vision Gallery').classes('text-xl font-bold')
                self.gallery.mount()

    async def _start_ingest(self):
        path = self._path_input.value if self._path_input else ""
        if not path or not os.path.exists(path):
            ui.notify('Invalid path', type='warning')
            return
        ui.notify('Ingestion started...', type='info')
        asyncio.create_task(self._run_ingest(Path(path)))

    async def _run_ingest(self, folder: Path):
        scanner = AsyncFileScanner()
        processor = DocumentProcessor()
        total = 0
        processed = 0
        # Count files first
        for _, _, files in os.walk(folder):
            total += len(files)
        async for doc in scanner.scan_directory(str(folder)):
            processed += 1
            if self._progress:
                self._progress.text = f"Processing {processed}/{total}..."
            # Process text
            if self._text_flag:
                _ = await processor.process_document(doc.text, doc.metadata)
                await event_bus.broadcast("knowledge_added", {"path": doc.metadata.get("path"), "summary": doc.text[:120]})
            # Vision placeholder: show in gallery
            if self._vision_flag and doc.metadata.get("ext") in {".png", ".jpg", ".jpeg"}:
                self.gallery.add_item({"path": doc.metadata.get("path"), "caption": doc.metadata.get("path"), "tags": []})
        if self._progress:
            self._progress.text = "Completed"
        ui.notify('Ingestion completed', type='positive')
