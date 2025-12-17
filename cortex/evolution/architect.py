"""
Species Architect
=================

Turns a human goal into a robust configuration ("species spec")
for the GGGP evolutionary engine.

This module focuses on **validation and hardening** of LLM output.
Even if the model responds with partial or sloppy JSON, we clamp and
fill defaults so the genetic engine receives a safe, minimal spec.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from cortex.ai_interface import IntellectualCore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (aligned with current minimal DSL)
# ---------------------------------------------------------------------------

DEFAULT_ACTIONS = [
    {"name": "buy", "params": {"amount": {"min": 0.1, "max": 10.0}}},
    {"name": "sell", "params": {"amount": {"min": 0.1, "max": 10.0}}},
    {"name": "hold", "params": {}},
]

DEFAULT_CONDITIONS = [
    {
        "left": "balance",
        "comparators": ["<", "<=", ">", ">=", "=="],
        "threshold_range": [0, 100],
    }
]


def default_species_spec(user_goal: str) -> Dict[str, Any]:
    goal_l = (user_goal or "").lower()
    domain = "trading" if any(k in goal_l for k in ("trade", "торг", "buy", "sell", "profit", "прибыл")) else "general"
    target_asset = "SIBR" if ("sibr" in goal_l or "сибр" in goal_l) else None

    return {
        "species": {
            "name": _slugify(user_goal) or "species",
            "goal": user_goal,
            "domain": domain,
            "target_asset": target_asset,
            "description": "",
            "constraints": {
                "risk": "medium",
                "time_horizon": "mid",
                "notes": "",
            },
        },
        "grammar": {
            "actions": list(DEFAULT_ACTIONS),
            "conditions": list(DEFAULT_CONDITIONS),
            "rule_count": {"min": 2, "max": 5},
            "max_depth": 2,
        },
        "fitness": {
            "objective": "maximize_balance",
            "signals": ["balance"],
            "penalties": [],
            "notes": "Default fitness uses api.get_balance() deltas.",
        },
    }


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def sanitize_species_spec(spec: Dict[str, Any], user_goal: str = "") -> Dict[str, Any]:
    """
    Validate and harden a raw spec dict.

    Unknown keys are preserved, but core fields are normalized.
    """
    base = default_species_spec(user_goal or _safe_get(spec, "species.goal", ""))
    merged: Dict[str, Any] = _deep_merge(base, spec)

    # species
    species = merged.get("species", {})
    if not isinstance(species, dict):
        species = {}
    species_goal = species.get("goal") or user_goal or base["species"]["goal"]
    if not isinstance(species_goal, str):
        species_goal = str(species_goal)
    species["goal"] = species_goal.strip()

    name = species.get("name") or base["species"]["name"]
    if not isinstance(name, str):
        name = str(name)
    species["name"] = _slugify(name) or base["species"]["name"]

    domain = species.get("domain") or base["species"]["domain"]
    if not isinstance(domain, str):
        domain = base["species"]["domain"]
    domain = domain.lower().strip()
    if domain not in ("trading", "general", "other"):
        domain = base["species"]["domain"]
    species["domain"] = domain

    target_asset = species.get("target_asset")
    if isinstance(target_asset, str):
        ta = target_asset.strip().upper()
        if not re.match(r"^[A-Z0-9]{2,10}$", ta):
            ta = None
        target_asset = ta
    else:
        target_asset = None
    species["target_asset"] = target_asset

    description = species.get("description", "")
    if not isinstance(description, str):
        description = str(description)
    species["description"] = description.strip()[:500]

    constraints = species.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
    risk = constraints.get("risk") or base["species"]["constraints"]["risk"]
    if not isinstance(risk, str):
        risk = base["species"]["constraints"]["risk"]
    risk = risk.lower().strip()
    if risk not in ("low", "medium", "high"):
        risk = base["species"]["constraints"]["risk"]
    constraints["risk"] = risk

    th = constraints.get("time_horizon") or base["species"]["constraints"]["time_horizon"]
    if not isinstance(th, str):
        th = base["species"]["constraints"]["time_horizon"]
    th = th.lower().strip()
    if th not in ("short", "mid", "long"):
        th = base["species"]["constraints"]["time_horizon"]
    constraints["time_horizon"] = th

    notes = constraints.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)
    constraints["notes"] = notes.strip()[:500]

    species["constraints"] = constraints
    merged["species"] = species

    # grammar
    grammar = merged.get("grammar", {})
    if not isinstance(grammar, dict):
        grammar = {}

    grammar["actions"] = _sanitize_actions(grammar.get("actions"), base["grammar"]["actions"])
    grammar["conditions"] = _sanitize_conditions(grammar.get("conditions"), base["grammar"]["conditions"])

    rule_count = grammar.get("rule_count", base["grammar"]["rule_count"])
    if not isinstance(rule_count, dict):
        rule_count = base["grammar"]["rule_count"]
    min_rules = _clamp_int(rule_count.get("min"), 1, 20, base["grammar"]["rule_count"]["min"])
    max_rules = _clamp_int(rule_count.get("max"), min_rules, 20, base["grammar"]["rule_count"]["max"])
    grammar["rule_count"] = {"min": min_rules, "max": max_rules}

    max_depth = _clamp_int(grammar.get("max_depth"), 1, 5, base["grammar"]["max_depth"])
    grammar["max_depth"] = max_depth

    merged["grammar"] = grammar

    # fitness
    fitness = merged.get("fitness", {})
    if not isinstance(fitness, dict):
        fitness = {}

    objective = fitness.get("objective") or base["fitness"]["objective"]
    if not isinstance(objective, str):
        objective = base["fitness"]["objective"]
    objective = objective.lower().strip()
    objective = _normalize_objective(objective, base["fitness"]["objective"])
    fitness["objective"] = objective

    signals = fitness.get("signals", base["fitness"]["signals"])
    if not isinstance(signals, list):
        signals = base["fitness"]["signals"]
    signals = [s for s in signals if isinstance(s, str) and s.strip()]
    if not signals:
        signals = base["fitness"]["signals"]
    fitness["signals"] = signals

    penalties = fitness.get("penalties", [])
    if not isinstance(penalties, list):
        penalties = []
    fitness["penalties"] = penalties[:20]

    notes = fitness.get("notes") or base["fitness"]["notes"]
    if not isinstance(notes, str):
        notes = str(notes)
    fitness["notes"] = notes.strip()[:1000]

    merged["fitness"] = fitness

    return merged


def _sanitize_actions(raw: Any, default: Any) -> list:
    if not isinstance(raw, list) or not raw:
        return list(default)

    actions = []
    for item in raw:
        if isinstance(item, str):
            name = item
            params = {}
        elif isinstance(item, dict):
            name = item.get("name")
            params = item.get("params", {})
        else:
            continue

        if not isinstance(name, str):
            continue
        name = name.lower().strip()

        if not isinstance(params, dict):
            params = {}

        if name in ("buy", "sell"):
            amount_cfg = params.get("amount", {})
            if not isinstance(amount_cfg, dict):
                amount_cfg = {}
            min_amt = _clamp_float(amount_cfg.get("min"), 0.1, 10.0, 0.1)
            max_amt = _clamp_float(amount_cfg.get("max"), min_amt, 10.0, 10.0)
            params = {"amount": {"min": min_amt, "max": max_amt}}
        else:
            params = {}

        actions.append({"name": name, "params": params})

    return actions or list(default)


def _sanitize_conditions(raw: Any, default: Any) -> list:
    if not isinstance(raw, list) or not raw:
        return list(default)

    allowed_comparators = {"<", "<=", ">", ">=", "=="}
    conds = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        left = item.get("left", "balance")
        if not isinstance(left, str):
            left = "balance"
        left = left.lower().strip()

        comps = item.get("comparators", default[0]["comparators"])
        if not isinstance(comps, list):
            comps = default[0]["comparators"]
        comps = [c for c in comps if isinstance(c, str) and c in allowed_comparators]
        if not comps:
            comps = default[0]["comparators"]

        tr = item.get("threshold_range", default[0]["threshold_range"])
        if not (isinstance(tr, list) and len(tr) == 2):
            tr = default[0]["threshold_range"]
        lo = _clamp_float(tr[0], 0.0, 100.0, default[0]["threshold_range"][0])
        hi = _clamp_float(tr[1], lo, 100.0, default[0]["threshold_range"][1])

        conds.append({"left": left, "comparators": comps, "threshold_range": [lo, hi]})

    return conds or list(default)


def _normalize_objective(obj: str, fallback: str) -> str:
    mappings = {
        "profit": "maximize_profit",
        "pnl": "maximize_profit",
        "maximize_pnl": "maximize_profit",
        "make_money": "maximize_profit",
        "balance": "maximize_balance",
        "maximize_funds": "maximize_balance",
        "min_risk": "minimize_risk",
        "risk": "minimize_risk",
    }
    obj = mappings.get(obj, obj)
    if obj not in ("maximize_balance", "maximize_profit", "minimize_risk", "other"):
        return fallback
    return obj


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(value)
    except Exception:
        return default
    return max(lo, min(hi, iv))


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        fv = float(value)
    except Exception:
        return default
    return max(lo, min(hi, fv))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return slug[:64]


def apply_collector_stats(spec: Dict[str, Any], stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply PrivacyCollector aggregated stats to a species spec.

    When humans frequently accept certain suggestion types (e.g. action names),
    include those actions in the base grammar so evolution can leverage them.
    """
    if not isinstance(spec, dict) or not isinstance(stats, dict) or not stats:
        return spec

    grammar = spec.get("grammar")
    if not isinstance(grammar, dict):
        return spec

    actions = grammar.get("actions")
    if not isinstance(actions, list):
        return spec

    existing_names = set(_extract_action_names(actions))
    candidates: list[str] = []

    top = stats.get("top_accepted_types")
    if isinstance(top, list):
        for t in top:
            if isinstance(t, str) and t.strip():
                candidates.append(t.strip())

    if not candidates:
        by_type = stats.get("acceptance_by_type")
        if isinstance(by_type, dict):
            for k, _v in sorted(by_type.items(), key=lambda kv: kv[1], reverse=True):
                if isinstance(k, str) and k.strip():
                    candidates.append(k.strip())

    added = 0
    for name in candidates:
        safe = _sanitize_action_name(name)
        if not safe or safe in existing_names:
            continue
        actions.append({"name": safe, "params": {}})
        existing_names.add(safe)
        added += 1
        if added >= 5:
            break

    grammar["actions"] = actions
    spec["grammar"] = grammar
    return spec


def _extract_action_names(actions: list) -> list[str]:
    names: list[str] = []
    for item in actions:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
            names.append(item["name"].strip())
    return names


def _sanitize_action_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    if not re.match(r"^[a-zA-Z0-9_:\-]{1,64}$", n):
        n = _slugify(n)
    return n[:64]


class SpeciesArchitect:
    """
    High-level facade that uses IntellectualCore to build a species spec.
    """

    def __init__(self, core: Optional[IntellectualCore] = None):
        self.core = core or IntellectualCore()

    async def design(
        self,
        user_goal: str,
        *,
        collector: Optional[Any] = None,
        collector_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw = await self.core.generate_spec(user_goal)
        spec = sanitize_species_spec(raw, user_goal=user_goal)

        stats = collector_stats
        if stats is None and collector is not None and hasattr(collector, "get_aggregated_stats"):
            try:
                stats = collector.get_aggregated_stats()
            except Exception as e:
                logger.debug("[ARCHITECT] Collector stats unavailable: %s", e)
                stats = None

        return apply_collector_stats(spec, stats)
