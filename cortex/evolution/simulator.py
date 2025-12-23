"""
Market simulator for GGGP pricing challenges.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional


@dataclass
class MarketTick:
    minute: int
    time_of_day: float
    demand: float
    market_price: float
    agent_price: float
    deals: float
    profit: float
    load_pct: float
    spike: bool
    volatility: float


class MarketEnvironment:
    def __init__(
        self,
        base_demand: float = 80.0,
        demand_amplitude: float = 40.0,
        noise_sigma: float = 6.0,
        spike_chance: float = 0.03,
        spike_scale: float = 90.0,
        base_price: float = 1.0,
        price_volatility: float = 0.12,
        elasticity: float = 2.0,
        capacity: float = 120.0,
        min_price: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.base_demand = base_demand
        self.demand_amplitude = demand_amplitude
        self.noise_sigma = noise_sigma
        self.spike_chance = spike_chance
        self.spike_scale = spike_scale
        self.base_price = base_price
        self.price_volatility = price_volatility
        self.elasticity = elasticity
        self.capacity = capacity
        self.min_price = min_price
        self.minute = 0
        self._rng = random.Random(seed)
        self._last_obs: Optional[Dict[str, Any]] = None
        self.history: List[MarketTick] = []

    def observe(self) -> Dict[str, Any]:
        cycle = math.sin(2.0 * math.pi * (self.minute % 1440) / 1440.0)
        noise = self._rng.gauss(0.0, self.noise_sigma)
        spike = self._rng.random() < self.spike_chance
        spike_value = self._rng.uniform(self.spike_scale * 0.5, self.spike_scale) if spike else 0.0
        demand = max(0.0, self.base_demand + self.demand_amplitude * cycle + noise + spike_value)

        price_jitter = self._rng.uniform(-self.price_volatility, self.price_volatility)
        market_price = max(self.min_price, self.base_price * (1.0 + self.price_volatility * cycle + price_jitter))

        self._last_obs = {
            "minute": self.minute,
            "time_of_day": (self.minute % 1440) / 60.0,
            "demand": demand,
            "market_price": market_price,
            "volatility": abs(noise),
            "spike": spike,
        }
        return self._last_obs

    def step(self, agent_price: float) -> tuple[float, float, float, Dict[str, Any]]:
        if self._last_obs is None:
            self.observe()

        assert self._last_obs is not None
        demand = self._last_obs["demand"]
        market_price = self._last_obs["market_price"]

        price = max(self.min_price, float(agent_price))
        price_ratio = price / max(market_price, 1e-6)
        if price_ratio > 1.0:
            demand *= math.exp(-self.elasticity * (price_ratio - 1.0))
        else:
            demand *= 1.0 + min(0.6, (1.0 - price_ratio) * 0.5)

        deals = max(0.0, demand)
        profit = deals * price
        load_pct = min(100.0, (deals / max(self.capacity, 1e-6)) * 100.0)

        tick = MarketTick(
            minute=self.minute,
            time_of_day=self._last_obs["time_of_day"],
            demand=self._last_obs["demand"],
            market_price=market_price,
            agent_price=price,
            deals=deals,
            profit=profit,
            load_pct=load_pct,
            spike=bool(self._last_obs["spike"]),
            volatility=float(self._last_obs["volatility"]),
        )
        self.history.append(tick)
        self.minute += 1
        return deals, profit, load_pct, self._last_obs


class MarketAPI:
    def __init__(
        self,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
        last_price: float,
    ):
        self._obs = observation
        self._history = history
        self._price = float(last_price)

    def log(self, _msg: str) -> None:
        return None

    def get_balance(self) -> float:
        return 0.0

    def send_message(self, _peer: str, _msg: str) -> None:
        return None

    def set_price(self, price: float) -> None:
        try:
            self._price = float(price)
        except Exception:
            pass

    def adjust_price_up(self, delta: float = 1.0) -> None:
        try:
            self._price += float(delta)
        except Exception:
            pass

    def adjust_price_down(self, delta: float = 1.0) -> None:
        try:
            self._price -= float(delta)
        except Exception:
            pass

    def get_market_demand(self) -> float:
        return float(self._obs.get("demand", 0.0))

    def get_market_price(self) -> float:
        return float(self._obs.get("market_price", 0.0))

    def get_time_of_day(self) -> float:
        return float(self._obs.get("time_of_day", 0.0))

    def get_volatility(self) -> float:
        return float(self._obs.get("volatility", 0.0))

    def get_spike(self) -> float:
        return 1.0 if self._obs.get("spike") else 0.0

    def get_last_price(self) -> float:
        if self._history:
            return float(self._history[-1].get("last_price", self._price))
        return float(self._price)

    def get_load(self) -> float:
        if self._history:
            return float(self._history[-1].get("load", 0.0))
        return 0.0

    def get_history_value(self, sensor: str, steps_back: int = 1) -> float:
        if not self._history or steps_back <= 0:
            return 0.0
        idx = max(0, len(self._history) - steps_back)
        return float(self._history[idx].get(sensor, 0.0))

    @property
    def price(self) -> float:
        return float(self._price)


__all__ = ["MarketEnvironment", "MarketAPI", "MarketTick"]
