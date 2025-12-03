import logging
from typing import Dict, Any, Optional

from agents.local_llm import OllamaAgent
from cortex.compliance.pii import PIIGuard
from cortex.compliance.judge import ComplianceJudge

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

    async def process_document(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Запускает многошаговый анализ текста."""
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
        return result
