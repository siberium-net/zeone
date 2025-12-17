"""
Privacy-Preserving Data Collector
=================================
Collects anonymized learning signals for federated training.

[PRIVACY]
Collector is DISABLED by default and must be enabled with explicit consent.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SignalType(Enum):
    PREFERENCE_PAIR = "preference_pair"
    CORRECTION = "correction"
    TIMING = "timing"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"


@dataclass
class LearningSignal:
    signal_type: SignalType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    session_hash: str = ""

    def anonymize(self) -> "LearningSignal":
        anon_data: Dict[str, Any] = {}
        for key, value in self.data.items():
            if isinstance(value, str) and len(value) > 50:
                anon_data[key] = hashlib.sha256(value.encode()).hexdigest()[:16]
            else:
                anon_data[key] = value
        return LearningSignal(
            signal_type=self.signal_type,
            timestamp=self.timestamp,
            data=anon_data,
            session_hash=self.session_hash,
        )


class PrivacyCollector:
    """Collects local signals and returns only aggregated stats."""

    MAX_BUFFER_SIZE = 1000

    def __init__(self, session_id: Optional[str] = None):
        sid = session_id or str(time.time())
        self._session_hash = hashlib.sha256(sid.encode()).hexdigest()[:16]
        self._buffer: List[LearningSignal] = []
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True
        logger.info("[LEARNING] Data collection ENABLED (with user consent)")

    def disable(self) -> None:
        self._enabled = False
        self._buffer.clear()
        logger.info("[LEARNING] Data collection DISABLED")

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    @property
    def session_hash(self) -> str:
        return str(self._session_hash)

    def record_preference(self, chosen: str, rejected: str, context: Optional[str] = None) -> None:
        if not self._enabled:
            return
        signal = LearningSignal(
            signal_type=SignalType.PREFERENCE_PAIR,
            data={
                "chosen_hash": hashlib.sha256(chosen.encode()).hexdigest()[:16],
                "rejected_hash": hashlib.sha256(rejected.encode()).hexdigest()[:16],
                "context_length": len(context) if context else 0,
            },
            session_hash=self._session_hash,
        )
        self._add_signal(signal)

    def record_correction(self, original: str, corrected: str, time_to_correct_ms: float) -> None:
        if not self._enabled:
            return
        signal = LearningSignal(
            signal_type=SignalType.CORRECTION,
            data={
                "original_len": len(original),
                "corrected_len": len(corrected),
                "edit_distance": self._levenshtein(original, corrected),
                "time_ms": float(time_to_correct_ms),
            },
            session_hash=self._session_hash,
        )
        self._add_signal(signal)

    def record_acceptance(self, suggestion_type: str) -> None:
        if not self._enabled:
            return
        self._add_signal(
            LearningSignal(
                signal_type=SignalType.ACCEPTANCE,
                data={"type": suggestion_type},
                session_hash=self._session_hash,
            )
        )

    def record_rejection(self, suggestion_type: str) -> None:
        if not self._enabled:
            return
        self._add_signal(
            LearningSignal(
                signal_type=SignalType.REJECTION,
                data={"type": suggestion_type},
                session_hash=self._session_hash,
            )
        )

    def _add_signal(self, signal: LearningSignal) -> None:
        if len(self._buffer) >= self.MAX_BUFFER_SIZE:
            self._buffer.pop(0)
        self._buffer.append(signal.anonymize())

    def get_aggregated_stats(self) -> Dict[str, Any]:
        if not self._buffer:
            return {}

        acceptances = sum(1 for s in self._buffer if s.signal_type == SignalType.ACCEPTANCE)
        rejections = sum(1 for s in self._buffer if s.signal_type == SignalType.REJECTION)
        corrections = [s for s in self._buffer if s.signal_type == SignalType.CORRECTION]

        stats: Dict[str, Any] = {
            "total_signals": len(self._buffer),
            "acceptance_rate": 0.0,
            "avg_correction_time_ms": 0.0,
            "avg_edit_distance": 0.0,
            # Optional per-type breakdown (e.g. action names or suggestion categories).
            "acceptance_by_type": {},
            "rejection_by_type": {},
            "top_accepted_types": [],
        }

        if acceptances + rejections > 0:
            stats["acceptance_rate"] = acceptances / (acceptances + rejections)

        if corrections:
            stats["avg_correction_time_ms"] = sum(float(s.data.get("time_ms", 0.0)) for s in corrections) / len(corrections)
            stats["avg_edit_distance"] = sum(int(s.data.get("edit_distance", 0)) for s in corrections) / len(corrections)

        acceptance_by_type: Dict[str, int] = {}
        rejection_by_type: Dict[str, int] = {}
        for s in self._buffer:
            if s.signal_type not in (SignalType.ACCEPTANCE, SignalType.REJECTION):
                continue
            t = s.data.get("type")
            if not isinstance(t, str) or not t.strip():
                continue
            key = t.strip()[:64]
            if s.signal_type == SignalType.ACCEPTANCE:
                acceptance_by_type[key] = acceptance_by_type.get(key, 0) + 1
            else:
                rejection_by_type[key] = rejection_by_type.get(key, 0) + 1

        stats["acceptance_by_type"] = acceptance_by_type
        stats["rejection_by_type"] = rejection_by_type
        stats["top_accepted_types"] = [
            k for k, _v in sorted(acceptance_by_type.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ]

        return stats

    def clear(self) -> None:
        self._buffer.clear()

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return PrivacyCollector._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
