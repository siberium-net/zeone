"""
Grammar / AST for evolutionary agents.

The grammar is **dynamic**: terminals (sensors/actions) are configured
from a species spec via `configure_grammar`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Terminals (dynamic)
# ---------------------------------------------------------------------------


@dataclass
class SensorTerminal:
    """Leaf node that reads a sensor value from api."""

    name: str


@dataclass
class ActionTerminal:
    """Leaf node that performs an action on api."""

    name: str
    amount: Optional[float] = None


@dataclass
class AskHumanTerminal:
    """Leaf node representing a request to the node owner (human-in-the-loop)."""

    query: str
    options: List[str] = field(default_factory=lambda: ["Approve", "Reject"])


@dataclass
class CheckHumanStatusTerminal:
    """Leaf node that checks whether the owner is online/linked."""

    pass


# ---------------------------------------------------------------------------
# Expressions / Logic
# ---------------------------------------------------------------------------


@dataclass
class ConstNode:
    """Constant numeric value."""

    value: float


@dataclass
class IndicatorNode:
    """Indicator such as RSI or SMA."""

    name: str
    period: int = 14
    source: Optional[Any] = None  # SensorTerminal / IndicatorNode / ConstNode


@dataclass
class ComparisonNode:
    """Binary comparison: left <op> right."""

    left: Any
    comparator: str
    right: Any


@dataclass
class LogicNode:
    """Logical composition of conditions."""

    operator: str  # "AND" | "OR"
    operands: List[Any] = field(default_factory=list)


@dataclass
class IfNode:
    """If/Else control node."""

    condition: Any
    on_true: Optional[Any] = None
    on_false: Optional[Any] = None


# ---------------------------------------------------------------------------
# Backward compatible nodes (legacy)
# ---------------------------------------------------------------------------


@dataclass
class ActionNode:
    """Legacy action node (buy/sell/hold)."""

    action: str
    amount: float = 0.0


@dataclass
class ConditionNode:
    """Legacy condition node (if balance < X)."""

    comparator: str
    threshold: float
    on_true: Optional[Any] = None
    on_false: Optional[Any] = None


@dataclass
class Gene:
    """Sequence of rules/conditions for an agent."""

    rules: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dynamic grammar context / factory
# ---------------------------------------------------------------------------


@dataclass
class GrammarContext:
    allowed_sensors: List[str] = field(default_factory=lambda: ["get_balance"])
    allowed_actions: List[str] = field(default_factory=lambda: ["buy", "sell", "hold", "wait"])
    rule_count: Tuple[int, int] = (2, 5)  # (min, max)
    max_depth: int = 2


# Global context used as default by genetics.
GRAMMAR_CONTEXT = GrammarContext()


def configure_grammar(spec: Dict[str, Any]) -> GrammarContext:
    """
    Configure dynamic terminals from a species spec.

    Supports both:
    - flat spec: {"sensors": [...], "actions": [...]}
    - architect spec: {"grammar": {"actions": [...], "rule_count": {...}, "max_depth": n}}
    """
    global GRAMMAR_CONTEXT

    grammar_spec = spec.get("grammar") if isinstance(spec, dict) else None
    if not isinstance(grammar_spec, dict):
        grammar_spec = {}

    # Actions
    raw_actions: Any = spec.get("actions", grammar_spec.get("actions", None))
    actions = _extract_names(raw_actions)
    if actions:
        GRAMMAR_CONTEXT.allowed_actions = actions

    # Sensors
    raw_sensors: Any = spec.get("sensors", grammar_spec.get("sensors", None))
    sensors = _extract_names(raw_sensors)
    if sensors:
        GRAMMAR_CONTEXT.allowed_sensors = sensors

    # Rule count / depth (optional)
    rc = grammar_spec.get("rule_count")
    if isinstance(rc, dict):
        min_rules = _clamp_int(rc.get("min"), 1, 20, GRAMMAR_CONTEXT.rule_count[0])
        max_rules = _clamp_int(rc.get("max"), min_rules, 20, GRAMMAR_CONTEXT.rule_count[1])
        GRAMMAR_CONTEXT.rule_count = (min_rules, max_rules)

    md = grammar_spec.get("max_depth")
    if md is not None:
        GRAMMAR_CONTEXT.max_depth = _clamp_int(md, 1, 5, GRAMMAR_CONTEXT.max_depth)

    return GRAMMAR_CONTEXT


def _extract_names(raw: Any) -> List[str]:
    if raw is None:
        return []
    names: List[str] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, dict)):
        for item in raw:
            if isinstance(item, str):
                n = item.strip()
                if n:
                    names.append(n)
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                n = item["name"].strip()
                if n:
                    names.append(n)
    elif isinstance(raw, str):
        names = [raw.strip()] if raw.strip() else []
    return names


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(value)
    except Exception:
        return default
    return max(lo, min(hi, iv))
