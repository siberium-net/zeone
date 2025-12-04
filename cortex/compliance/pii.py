import logging
from typing import List, Dict, Any

from config import COMPLIANCE_ENABLED

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer
    from presidio_analyzer.predefined_recognizers import CreditCardRecognizer
    _PRESIDIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PRESIDIO_AVAILABLE = False


class PIIGuard:
    """PII scanner powered by Presidio."""

    def __init__(self):
        if not COMPLIANCE_ENABLED:
            self.engine = None
            self.active = False
            logger.warning("[COMPLIANCE] Compliance module disabled. PII scanning skipped.")
            return
        if not _PRESIDIO_AVAILABLE:
            logger.warning("[COMPLIANCE] Presidio not available")
            self.engine = None
            self.active = False
            return
        self.engine = AnalyzerEngine()
        self.engine.registry.add_recognizer(CreditCardRecognizer())
        self.active = True

    def scan_text(self, text: str) -> Dict[str, Any]:
        if not self.active or not self.engine:
            return {"findings": [], "risk": 0.0, "status": "DISABLED"}

        findings = self.engine.analyze(
            text=text,
            entities=[
                "CREDIT_CARD",
                "CRYPTO",
                "EMAIL",
                "IBAN",
                "PERSON",
                "PHONE_NUMBER",
                "US_SSN",
            ],
            language="en",
        )
        risk = 0.0
        status = "SAFE"
        extracted: List[Dict[str, Any]] = []
        for f in findings:
            score = f.score or 0.0
            risk = max(risk, score)
            extracted.append({"entity": f.entity_type, "score": score, "start": f.start, "end": f.end})
            if f.entity_type in ("CREDIT_CARD", "US_SSN", "IBAN") and score > 0.4:
                status = "BLOCKED_PII"
        if status != "BLOCKED_PII" and risk > 0.25:
            status = "WARNING"
        return {"findings": extracted, "risk": min(1.0, risk), "status": status}
