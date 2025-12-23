"""
Pricing challenges for GGGP evolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cortex.self import NodeNiche
from cortex.evolution.sandbox import AgentSandbox
from cortex.evolution.simulator import MarketAPI, MarketEnvironment, MarketTick


@dataclass
class PricingChallengeResult:
    total_profit: float
    overload_minutes: int
    overload_penalty: float
    fitness: float
    ticks: List[MarketTick]


class PricingChallenge:
    def __init__(
        self,
        niche: NodeNiche = NodeNiche.NEURAL_MINER,
        hours: int = 24,
        step_minutes: int = 5,
        overload_threshold: float = 95.0,
        max_overload_minutes: int = 10,
        overload_penalty: float = 50.0,
        seed: Optional[int] = 42,
    ):
        self.niche = niche
        self.hours = hours
        self.step_minutes = step_minutes
        self.overload_threshold = overload_threshold
        self.max_overload_minutes = max_overload_minutes
        self.overload_penalty = overload_penalty
        self.seed = seed

    def species_spec(self) -> Dict[str, Any]:
        return {
            "species": {
                "name": "pricing_challenge",
                "goal": "Dynamic pricing under volatile demand",
                "domain": "pricing",
                "description": "Evolve pricing formulas with volatility and spikes",
            },
            "grammar": {
                "sensors": [
                    "market_demand",
                    "market_price",
                    "time_of_day",
                    "volatility",
                    "spike",
                    "load",
                    "last_price",
                ],
                "actions": ["set_price", "adjust_price_up", "adjust_price_down"],
                "rule_count": {"min": 2, "max": 6},
                "max_depth": 4,
            },
            "fitness": {
                "objective": "maximize_profit_penalize_overload",
                "signals": ["profit", "overload_minutes"],
            },
            "fitness_logic": (
                "def evaluate(agent_output, history):\n"
                "    profit = float(agent_output.get('profit', 0.0))\n"
                "    overload = float(agent_output.get('overload_minutes', 0.0))\n"
                "    penalty = float(agent_output.get('overload_penalty', 0.0))\n"
                "    return profit - penalty * overload\n"
            ),
        }

    def simulate_agent_code(self, code: str, use_sandbox: bool = False) -> PricingChallengeResult:
        steps = max(1, int(self.hours * 60 / max(1, self.step_minutes)))
        env = MarketEnvironment(seed=self.seed)

        history: List[Dict[str, Any]] = []
        sandbox = AgentSandbox(timeout=0.2)
        total_profit = 0.0
        overload_minutes = 0
        overload_streak = 0
        last_price = env.base_price

        for _ in range(steps):
            obs = env.observe()
            api = MarketAPI(obs, history, last_price)

            if use_sandbox:
                sandbox.run(code, api)
            else:
                self._exec_inline(code, api)

            price = api.price
            _deals, profit, load_pct, _obs = env.step(price)

            if load_pct > self.overload_threshold:
                overload_streak += self.step_minutes
                if overload_streak > self.max_overload_minutes:
                    overload_minutes += self.step_minutes
            else:
                overload_streak = 0

            total_profit += profit
            history.append(
                {
                    "market_demand": obs["demand"],
                    "market_price": obs["market_price"],
                    "time_of_day": obs["time_of_day"],
                    "volatility": obs["volatility"],
                    "spike": obs["spike"],
                    "load": load_pct,
                    "last_price": price,
                    "profit": profit,
                }
            )
            last_price = price

        fitness = total_profit - (self.overload_penalty * overload_minutes)
        return PricingChallengeResult(
            total_profit=total_profit,
            overload_minutes=overload_minutes,
            overload_penalty=self.overload_penalty,
            fitness=fitness,
            ticks=env.history,
        )

    @staticmethod
    def baseline_code() -> str:
        return "\n".join(
            [
                "agent_output = {'actions': []}",
                "price = api.get_market_price()",
                "api.set_price(price)",
                "agent_output['actions'].append('set_price')",
                "agent_output['profit'] = 0.0",
            ]
        )

    @staticmethod
    def _exec_inline(code: str, api: MarketAPI) -> None:
        safe_builtins = {
            "True": True,
            "False": False,
            "None": None,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "abs": abs,
            "float": float,
            "int": int,
            "round": round,
        }
        env: Dict[str, Any] = {"__builtins__": safe_builtins, "api": api, "math": math}
        local_env: Dict[str, Any] = {}
        try:
            exec(code, env, local_env)
        except Exception:
            return


__all__ = ["PricingChallenge", "PricingChallengeResult"]
