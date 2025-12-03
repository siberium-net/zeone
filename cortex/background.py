import asyncio
import logging
import contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import psutil

from cortex.archivist import AsyncFileScanner, DocumentProcessor, VectorStore
from cortex.vision import VisionEngine
from cortex.video import VideoProcessor
from agents.local_llm import OllamaAgent
from economy.ledger import Ledger

logger = logging.getLogger(__name__)


class IdleWorker:
    """
    Background ingestion worker.
    
    Priority: text -> images -> video
    Pauses when CPU high.
    """

    def __init__(
        self,
        ledger: Ledger,
        vector_store: VectorStore,
        llm: Optional[OllamaAgent] = None,
    ):
        self.ledger = ledger
        self.vector_store = vector_store
        self.llm = llm or OllamaAgent()
        self.vision = VisionEngine(device="cuda" if psutil.cpu_count() else "cpu")
        self.video = VideoProcessor(self.vision, self.llm)
        self.scanner = AsyncFileScanner()
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    def enqueue(self, paths: List[str]) -> None:
        for p in paths:
            self.queue.put_nowait(p)
    
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
    
    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
    
    def _can_run(self) -> bool:
        cpu = psutil.cpu_percent(interval=0.1)
        return cpu < 20.0
    
    async def _loop(self) -> None:
        while self._running:
            try:
                path = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                await asyncio.sleep(0.5)
                continue
            
            if not self._can_run():
                await asyncio.sleep(2.0)
                self.queue.put_nowait(path)
                continue
            
            await self._process_path(path)
    
    async def _process_path(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    self.queue.put_nowait(str(child))
            return
        
        suffix = p.suffix.lower()
        try:
            if suffix in {".txt", ".md", ".json", ".py", ".pdf"}:
                async for doc in self.scanner.scan_directory(str(p.parent if p.is_dir() else p.parent)):
                    if doc.path != p:
                        continue
                    await self._handle_text(doc.text, doc.metadata)
            elif suffix in {".jpg", ".jpeg", ".png"}:
                res = self.vision.analyze_image(str(p))
                desc = res.get("description", "")
                self.vector_store.embed_and_store([desc], metadata=res | {"path": str(p)})
                await self.ledger.add_knowledge_entry(
                    cid=desc[:32],
                    path=str(p),
                    summary=desc[:512],
                    tags=",".join(res.get("tags", [])),
                    size=p.stat().st_size,
                    metadata=res,
                )
            elif suffix in {".mp4", ".mov", ".avi", ".mkv"}:
                result = await self.video.process_video(str(p))
                summary = result.get("summary", "")
                self.vector_store.embed_and_store([summary], metadata={"path": str(p), "type": "video"})
                await self.ledger.add_knowledge_entry(
                    cid=summary[:32],
                    path=str(p),
                    summary=summary[:512],
                    tags="video",
                    size=p.stat().st_size,
                    metadata=result,
                )
        except Exception as e:
            logger.warning(f"[IDLE] Failed to process {p}: {e}")
    
    async def _handle_text(self, text: str, metadata: Dict[str, Any]) -> None:
        processor = DocumentProcessor(self.llm)
        result = await processor.process_document(text, metadata)
        summary = result.get("summary", "")
        self.vector_store.embed_and_store([text], metadata=metadata)
        await self.ledger.add_knowledge_entry(
            cid=summary[:32],
            path=metadata.get("path", ""),
            summary=str(summary)[:512],
            tags="",
            size=int(metadata.get("size", 0)),
            metadata=result,
        )
