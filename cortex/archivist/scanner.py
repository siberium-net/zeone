import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Dict, Tuple

try:
    import aiofiles
except ImportError:  # pragma: no cover
    aiofiles = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


@dataclass
class ExtractedDocument:
    path: Path
    text: str
    metadata: Dict[str, str]


class AsyncFileScanner:
    """Асинхронный сканер файловой системы с текстовой экстракцией."""

    SUPPORTED_EXT = {".txt", ".md", ".pdf", ".py", ".json"}

    def __init__(self):
        if aiofiles is None:
            raise ImportError("aiofiles is required for AsyncFileScanner")

    async def scan_directory(self, root: str) -> AsyncGenerator[ExtractedDocument, None]:
        """Обойти директорию и извлечь текст поддерживаемых файлов."""
        root_path = Path(root).expanduser().resolve()
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                path = Path(dirpath) / fname
                if path.suffix.lower() not in self.SUPPORTED_EXT:
                    continue
                try:
                    text, meta = await self.extract_text(path)
                    yield ExtractedDocument(path=path, text=text, metadata=meta)
                except Exception:
                    continue

    async def extract_text(self, path: Path) -> Tuple[str, Dict[str, str]]:
        """Извлечь чистый текст и метаданные из файла."""
        stat = path.stat()
        metadata = {
            "path": str(path),
            "size": str(stat.st_size),
            "created_at": str(stat.st_ctime),
            "modified_at": str(stat.st_mtime),
            "name": path.name,
            "ext": path.suffix.lower(),
        }

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".py"}:
            async with aiofiles.open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = await f.read()
        elif suffix == ".json":
            async with aiofiles.open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = await f.read()
            try:
                text = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
            except Exception:
                text = raw
        elif suffix == ".pdf":
            if PdfReader is None:
                raise ImportError("pypdf is required for PDF support")
            text = await asyncio.to_thread(self._read_pdf, path)
        else:
            text = ""

        return text, metadata

    def _read_pdf(self, path: Path) -> str:
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
