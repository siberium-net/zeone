import asyncio
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

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
        self._current_title = None
        self._current_image = None
        self._current_tags_container = None
        self._current_summary = None
        self._progress_bar = None
        self._stats_label = None
        self._last_update = 0.0
        self.gallery = gallery

    def create_page(self, parent):
        @ui.page('/ingest')
        async def ingest():
            await parent._create_header()
            await parent._create_sidebar()

            with ui.row().classes('w-full p-4 gap-6'):
                with ui.column().classes('w-1/3'):
                    ui.label('Ingestion Control').classes('text-2xl font-bold mb-4')
                    self._path_input = ui.input('Directory path', value=str(Path.cwd())).classes('w-full')
                    self._text_flag = ui.checkbox('Process Text/PDF', value=True)
                    self._vision_flag = ui.checkbox('Enable Vision (GPU Heavy)', value=False)
                    self._video_flag = ui.checkbox('Enable Video Analysis (Slow)', value=False)
                    self._dream_flag = ui.checkbox('Dream Mode (Only when idle)', value=False)

                    ui.button('Start Ingestion', icon='play_arrow', on_click=self._start_ingest)
                    self._progress = ui.label('Idle').classes('text-sm text-gray-500')
                    self._progress_bar = ui.linear_progress(value=0).classes('w-full')
                    self._stats_label = ui.label('').classes('text-xs text-gray-500')
                with ui.column().classes('w-2/3'):
                    ui.label('Live Retina').classes('text-xl font-bold mb-2')
                    self._current_title = ui.label('Waiting...').classes('text-sm text-gray-400')
                    self._current_image = ui.image('').classes('w-full h-64 object-contain bg-black')
                    self._current_tags_container = ui.row().classes('gap-1 flex-wrap')
                    self._current_summary = ui.label('').classes('text-sm text-green-300')

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
        total_bytes = 0
        start_time = time.time()
        # Count files first
        for _, _, files in os.walk(folder):
            total += len(files)
            for f in files:
                try:
                    total_bytes += (folder / f).stat().st_size
                except Exception:
                    pass
        if self._progress_bar:
            self._progress_bar.value = 0
        async for doc in scanner.scan_directory(str(folder)):
            processed += 1
            if self._progress:
                self._progress.text = f"Processing {processed}/{total}..."
            if self._progress_bar:
                self._progress_bar.value = processed / max(1, total)
            elapsed = max(0.001, time.time() - start_time)
            if self._stats_label:
                speed = processed / elapsed
                mb = total_bytes / (1024 * 1024) if total_bytes else 0
                self._stats_label.text = f"Files/s: {speed:.2f} | Total size: {mb:.1f} MB"
            # Process text
            if self._text_flag:
                _ = await processor.process_document(doc.text, doc.metadata, progress_callback=self._ingest_event)
                await event_bus.broadcast("knowledge_added", {"path": doc.metadata.get("path"), "summary": doc.text[:120]})
            # Vision placeholder: show in gallery
            if self._vision_flag and doc.metadata.get("ext") in {".png", ".jpg", ".jpeg"}:
                self.gallery.add_item({"path": doc.metadata.get("path"), "caption": doc.metadata.get("path"), "tags": []})
        if self._progress:
            self._progress.text = "Completed"
        ui.notify('Ingestion completed', type='positive')

    def _ingest_event(self, payload: Dict[str, Any]):
        # Throttle UI updates
        now = time.time()
        if now - self._last_update < 0.1:
            return
        self._last_update = now
        event = payload.get("event")
        data = payload.get("data", {})
        if event == "start_file":
            if self._current_title:
                self._current_title.text = f"Processing: {data.get('path','')}"
        if event == "vision_preview":
            b64 = data.get("base64_image")
            if b64 and self._current_image:
                self._current_image.set_source(f"data:image/jpeg;base64,{b64}")
        if event == "metadata_extracted":
            tags_text = data.get("tags") or ""
            if self._current_tags_container:
                self._current_tags_container.clear()
                tags = tags_text if isinstance(tags_text, list) else [tags_text]
                with self._current_tags_container:
                    for t in tags:
                        if t:
                            ui.chip(t)
            if self._current_summary:
                self._current_summary.text = data.get("summary", "")
