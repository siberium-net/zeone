"""
Runtime singleton for learning collector.

The collector is intentionally local-only and disabled by default.
UI or other components may enable it with explicit user consent.
"""

from __future__ import annotations

import os
from typing import Optional

from .collector import PrivacyCollector

_collector: Optional[PrivacyCollector] = None


def get_collector() -> PrivacyCollector:
    global _collector
    if _collector is None:
        session_id = os.getenv("ZEONE_LEARNING_SESSION_ID") or None
        _collector = PrivacyCollector(session_id=session_id)
    return _collector

