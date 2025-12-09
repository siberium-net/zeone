"""
Draft grammar/AST for evolutionary agents.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class ActionNode:
    """Represents an action (e.g., buy/sell/hold)."""

    action: str
    amount: float = 0.0


@dataclass
class ConditionNode:
    """Conditional guard (e.g., if balance < X)."""

    comparator: str
    threshold: float
    on_true: Optional[Any] = None  # Could reference ActionNode or Gene
    on_false: Optional[Any] = None


@dataclass
class Gene:
    """Sequence of rules/conditions for an agent."""

    rules: List[Any] = field(default_factory=list)

