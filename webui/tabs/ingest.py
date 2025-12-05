import asyncio
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from nicegui import ui, context

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
        self._running = False
        self._task: Optional[asyncio.Task] = None
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

                    with ui.row().classes('gap-2'):
                        ui.button('Start Ingestion', icon='play_arrow', on_click=self._start_ingest)
                        ui.button('Stop', icon='stop', on_click=self._stop_ingest).props('color=negative')
                    
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
        if self._running:
            ui.notify('Ingestion already running', type='warning')
            return
            
        path = self._path_input.value if self._path_input else ""
        if not path or not os.path.exists(path):
            ui.notify('Invalid path', type='warning')
            return
        
        self._running = True
        ui.notify('Ingestion started...', type='info')
        self._task = asyncio.create_task(self._run_ingest(Path(path)))

    async def _stop_ingest(self):
        if not self._running:
            ui.notify('Not running', type='info')
            return
        
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        ui.notify('Ingestion stopped', type='warning')
        self._safe_update_ui(self._progress, 'Stopped')

    def _safe_update_ui(self, element, value):
        """Safely update UI element, ignoring errors if client disconnected."""
        if element is None:
            return
        try:
            if hasattr(element, 'text'):
                element.text = value
            elif hasattr(element, 'value'):
                element.value = value
        except RuntimeError:
            # Client disconnected, ignore
            pass
        except Exception:
            pass

    async def _run_ingest(self, folder: Path):
        scanner = AsyncFileScanner()
        processor = DocumentProcessor()
        total = 0
        processed = 0
        total_bytes = 0
        start_time = time.time()
        
        try:
            # Count files first
            for root, _, files in os.walk(folder):
                if not self._running:
                    return
                total += len(files)
                for f in files:
                    try:
                        total_bytes += Path(root, f).stat().st_size
                    except Exception:
                        pass
            
            self._safe_update_ui(self._progress_bar, 0)
            
            async for doc in scanner.scan_directory(str(folder)):
                if not self._running:
                    break
                    
                processed += 1
                self._safe_update_ui(self._progress, f"Processing {processed}/{total}...")
                self._safe_update_ui(self._progress_bar, processed / max(1, total))
                
                elapsed = max(0.001, time.time() - start_time)
                speed = processed / elapsed
                mb = total_bytes / (1024 * 1024) if total_bytes else 0
                self._safe_update_ui(self._stats_label, f"Files/s: {speed:.2f} | Total size: {mb:.1f} MB")
                
                # Process text
                try:
                    text_enabled = self._text_flag.value if hasattr(self._text_flag, 'value') else self._text_flag
                except Exception:
                    text_enabled = True
                    
                if text_enabled:
                    try:
                        _ = await processor.process_document(
                            doc.text, 
                            doc.metadata, 
                            progress_callback=self._ingest_event
                        )
                        await event_bus.broadcast(
                            "knowledge_added", 
                            {"path": doc.metadata.get("path"), "summary": doc.text[:120]}
                        )
                    except Exception as e:
                        # Log but continue
                        pass
                
                # Vision placeholder
                try:
                    vision_enabled = self._vision_flag.value if hasattr(self._vision_flag, 'value') else self._vision_flag
                except Exception:
                    vision_enabled = False
                    
                if vision_enabled and doc.metadata.get("ext") in {".png", ".jpg", ".jpeg"}:
                    try:
                        self.gallery.add_item({
                            "path": doc.metadata.get("path"), 
                            "caption": doc.metadata.get("path"), 
                            "tags": []
                        })
                    except Exception:
                        pass
                
                # Yield to event loop
                await asyncio.sleep(0)
            
            self._safe_update_ui(self._progress, "Completed")
            try:
                ui.notify('Ingestion completed', type='positive')
            except Exception:
                pass
                
        except asyncio.CancelledError:
            self._safe_update_ui(self._progress, "Cancelled")
        except Exception as e:
            self._safe_update_ui(self._progress, f"Error: {str(e)[:50]}")
        finally:
            self._running = False

    def _ingest_event(self, payload: Dict[str, Any]):
        # Throttle UI updates
        now = time.time()
        if now - self._last_update < 0.1:
            return
        self._last_update = now
        
        event = payload.get("event")
        data = payload.get("data", {})
        
        try:
            if event == "start_file":
                self._safe_update_ui(self._current_title, f"Processing: {data.get('path','')}")
                
            if event == "vision_preview":
                b64 = data.get("base64_image")
                if b64 and self._current_image:
                    try:
                        self._current_image.set_source(f"data:image/jpeg;base64,{b64}")
                    except Exception:
                        pass
                        
            if event == "metadata_extracted":
                tags_text = data.get("tags") or ""
                if self._current_tags_container:
                    try:
                        self._current_tags_container.clear()
                        tags = tags_text if isinstance(tags_text, list) else [tags_text]
                        with self._current_tags_container:
                            for t in tags:
                                if t:
                                    ui.chip(t)
                    except Exception:
                        pass
                        
                self._safe_update_ui(self._current_summary, data.get("summary", ""))
        except Exception:
            # Ignore UI update errors
            pass
