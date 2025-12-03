import logging
from typing import Dict, Any

from agents.local_llm import OllamaAgent

logger = logging.getLogger(__name__)


class ComplianceJudge:
    """AI moderator for legality/risk."""

    def __init__(self, llm: OllamaAgent = None):
        self.llm = llm or OllamaAgent()

    async def evaluate_legality(self, content_summary: str, jurisdiction: str = "EU") -> Dict[str, Any]:
        prompt = (
            f"You are a Compliance Officer. Current Jurisdiction: {jurisdiction}. "
            "Analyze the content for: Hate Speech, Extremism, Illegal pornography, Copyright infringement. "
            "Return JSON: {\"allowed\": bool, \"risk_category\": \"...\", \"reasoning\": \"...\"}. "
            "Be conservative."
        )
        result, _ = await self.llm.execute({"prompt": content_summary, "system": prompt})
        if isinstance(result, dict):
            return result
        return {"allowed": False, "risk_category": "unknown", "reasoning": str(result)}
