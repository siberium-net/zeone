import logging
from typing import Dict, Any, Optional

import base64
import io
import time
from pathlib import Path
from typing import Callable

from agents.local_llm import OllamaAgent
from cortex.compliance.pii import PIIGuard
from cortex.compliance.judge import ComplianceJudge
from core.events import event_bus

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Когнитивный конвейер через LLM (Ollama)."""

    def __init__(self, llm: Optional[OllamaAgent] = None):
        self.llm = llm or OllamaAgent()
        self.pii = PIIGuard()
        self.judge = ComplianceJudge(self.llm)

    async def _ask(self, prompt: str, system: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {"prompt": prompt}
        if system:
            payload["system"] = system
        result, _ = await self.llm.execute(payload)
        if isinstance(result, dict) and "response" in result:
            return result["response"]
        if isinstance(result, str):
            return result
        return str(result)

    async def process_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Запускает многошаговый анализ текста."""
        started = time.time()
        await self._emit(progress_callback, "start_file", {"path": metadata.get("path"), "type": metadata.get("ext")})
        if self._is_image(metadata):
            thumb = self._make_thumbnail(metadata.get("path"))
            await self._emit(progress_callback, "vision_preview", {"base64_image": thumb, "bboxes": []})
        summary_prompt = (
            "Analyze this text. Return JSON with keys: "
            "summary (short), tags (list of strings), uniqueness_score (0-10), utility_score (0-10). "
            "Text:\n" + text[:5000]
        )
        entities_prompt = (
            "Extract key entities (people, organizations) and a timeline of events. "
            "Identify any copyright markers or licenses. "
            "Return JSON with keys: entities (list), timeline (list), licenses (list). "
            "Text:\n" + text[:5000]
        )
        gap_prompt = (
            "What is missing from this text? What questions does it raise but not answer? "
            "Return bullet points."
        )

        try:
            summary_raw = await self._ask(summary_prompt)
        except Exception as e:
            logger.warning(f"[ARCHIVIST] Summary failed: {e}")
            summary_raw = "{}"

        try:
            entities_raw = await self._ask(entities_prompt)
        except Exception as e:
            logger.warning(f"[ARCHIVIST] Entity extraction failed: {e}")
            entities_raw = "{}"

        try:
            gaps_raw = await self._ask(gap_prompt)
        except Exception as e:
            logger.warning(f"[ARCHIVIST] Gap analysis failed: {e}")
            gaps_raw = ""

        compliance = {"status": "SAFE", "risk": 0.0}
        try:
            pii_result = self.pii.scan_text(text)
            compliance["pii"] = pii_result
            compliance["risk"] = max(compliance["risk"], pii_result.get("risk", 0.0))
            if pii_result.get("status") == "BLOCKED_PII":
                compliance["status"] = "BLOCKED"
            else:
                judge_result = await self.judge.evaluate_legality(summary_raw or text[:2000])
                compliance["judge"] = judge_result
                if not judge_result.get("allowed", False):
                    compliance["status"] = "WARNING"
        except Exception as e:
            logger.warning(f"[COMPLIANCE] Failed: {e}")

        result = {
            "summary": summary_raw,
            "entities": entities_raw,
            "gaps": gaps_raw,
            "metadata": metadata,
            "compliance_status": compliance.get("status", "SAFE"),
            "compliance": compliance,
        }
        await self._emit(
            progress_callback,
            "metadata_extracted",
            {
                "path": metadata.get("path"),
                "tags": result.get("summary", ""),
                "summary": result.get("summary", ""),
                "duration": time.time() - started,
            },
        )
        await self._emit(progress_callback, "finish_file", {"status": "indexed"})
        return result

    async def _emit(self, cb, event: str, data: Dict[str, Any]) -> None:
        payload = {"event": event, "data": data}
        if cb:
            try:
                res = cb(payload)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass
        await event_bus.broadcast("ingest_event", payload)

    def _is_image(self, metadata: Dict[str, Any]) -> bool:
        ext = (metadata.get("ext") or "").lower()
        return ext in {".png", ".jpg", ".jpeg", ".webp"}

    def _make_thumbnail(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            from PIL import Image
            p = Path(path)
            if not p.exists():
                return None
            img = Image.open(p).convert("RGB")
            img.thumbnail((480, 480))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None
