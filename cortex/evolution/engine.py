"""
Evolution Engine for GGGP agents.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cortex.evolution.genetics import (
    compile_genome,
    crossover,
    mutate,
    random_gene,
)
from cortex.evolution.grammar import Gene, GrammarContext, configure_grammar
from cortex.evolution.sandbox import AgentSandbox

logger = logging.getLogger(__name__)


class SimpleAPI:
    """Minimal Zeone-like API for fitness evaluation."""

    def __init__(
        self,
        balance: float = None,
        log_func=None,
        sensors: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
    ):
        self.balance = balance if balance is not None else random.uniform(0, 100)
        self._log = log_func or (lambda msg: None)
        self._sensors = set(sensors or [])
        self._actions = set(actions or [])

    def log(self, msg: str) -> None:
        self._log(msg)

    def get_balance(self) -> float:
        return self.balance

    def send_message(self, peer: str, msg: str) -> None:
        self._log(f"send_message to {peer}: {msg}")

    def get_history_value(self, sensor: str, steps_back: int = 1) -> float:
        return 0.0

    # Legacy actions used by old genomes
    def buy(self, amount: float = 1.0) -> None:
        self.balance -= float(amount)

    def sell(self, amount: float = 1.0) -> None:
        self.balance += float(amount)

    def hold(self) -> None:
        return None

    def wait(self) -> None:
        return None

    def __getattr__(self, name: str):
        """
        Dynamic sensors/actions:
        - Sensors are methods starting with 'get_'.
        - Actions are any other callable names from spec.
        """
        # Safe access via __dict__ to avoid recursion if attributes are missing.
        sensors = self.__dict__.get("_sensors", set())
        actions = self.__dict__.get("_actions", set())

        if name.startswith("get_") or name in sensors:
            def _sensor(*_args, **_kwargs):
                return random.uniform(0.0, 100.0)

            return _sensor

        if name in actions:
            def _action(*args, **_kwargs):
                amount = args[0] if args else 1.0
                try:
                    amount_f = float(amount)
                except Exception:
                    amount_f = 1.0
                if "buy" in name:
                    self.balance -= amount_f
                elif "sell" in name:
                    self.balance += amount_f
                return None

            return _action

        raise AttributeError(name)


class EvolutionEngine:
    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        elite: int = 2,
        output_dir: Path = Path("data/evolution"),
        species_spec: Optional[Dict[str, Any]] = None,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite = elite
        self.population: List[Gene] = []
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation = 0
        self.species_spec: Dict[str, Any] = species_spec or {}
        self.grammar_context: GrammarContext = configure_grammar(self.species_spec)
        self._fitness_fn: Callable[[Dict[str, Any], List[Any]], float] = self._compile_fitness(
            self.species_spec
        )

    def initialize_population(self, size: int = None) -> None:
        size = size or self.population_size
        self.population = [
            random_gene(depth=self.grammar_context.max_depth, context=self.grammar_context)
            for _ in range(size)
        ]

    async def evaluate_gene(self, gene: Gene) -> float:
        sandbox = AgentSandbox(timeout=1.0)
        api = SimpleAPI(
            log_func=logger.debug,
            sensors=self.grammar_context.allowed_sensors,
            actions=self.grammar_context.allowed_actions,
        )
        code = compile_genome(gene)
        result = sandbox.run(code, api)
        if not result.get("ok"):
            return -1.0

        agent_output = result.get("agent_output") or {}
        history = result.get("history") or []
        if not isinstance(agent_output, dict):
            agent_output = {}
        if not isinstance(history, list):
            history = []

        try:
            fit = float(self._fitness_fn(agent_output, history))
        except Exception as e:
            logger.debug(f"[EVO] Fitness fn failed: {e}")
            fit = float(agent_output.get("final_balance") or api.get_balance())
        return fit

    async def run_epoch(self) -> Tuple[float, float, Gene]:
        if not self.population:
            self.initialize_population()

        fitnesses: List[float] = []
        for gene in self.population:
            fit = await self.evaluate_gene(gene)
            fitnesses.append(fit)

        # Selection
        paired = list(zip(self.population, fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)

        best_gene, best_fit = paired[0]
        avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        # Save best
        self._save_best(best_gene, self.generation, best_fit)

        # Elitism
        new_population: List[Gene] = [paired[i][0] for i in range(min(self.elite, len(paired)))]

        # Tournament selection
        def tournament(k: int = 3) -> Gene:
            contenders = random.sample(paired, k=min(k, len(paired)))
            contenders.sort(key=lambda x: x[1], reverse=True)
            return contenders[0][0]

        while len(new_population) < self.population_size:
            parent_a = tournament()
            parent_b = tournament()
            child = crossover(parent_a, parent_b)
            child = mutate(child, rate=self.mutation_rate)
            new_population.append(child)

        self.population = new_population[: self.population_size]
        self.generation += 1

        return best_fit, avg_fit, best_gene

    def _save_best(self, gene: Gene, generation: int, fitness: float) -> None:
        code = compile_genome(gene)
        path = self.output_dir / f"best_gen_{generation}.py"
        try:
            path.write_text(code)
        except Exception as e:
            logger.warning(f"[EVO] Failed to save best genome: {e}")

    def _compile_fitness(self, spec: Dict[str, Any]) -> Callable[[Dict[str, Any], List[Any]], float]:
        """
        Compile fitness function once from spec['fitness_logic'].

        If missing or unsafe, use a conservative default.
        """
        logic = spec.get("fitness_logic")
        if not isinstance(logic, str) or not logic.strip():
            return lambda agent_output, _history: float(agent_output.get("final_balance", 0.0))

        if not self._is_safe_logic(logic):
            logger.warning("[EVO] Unsafe fitness_logic rejected, using default.")
            return lambda agent_output, _history: float(agent_output.get("final_balance", 0.0))

        if "def evaluate" in logic:
            code = logic
        else:
            code = (
                "def evaluate(agent_output, history):\n"
                f"    return float({logic.strip()})\n"
            )

        safe_builtins = {
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "abs": abs,
            "float": float,
            "int": int,
        }
        safe_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
        local_ns: Dict[str, Any] = {}
        try:
            exec(code, safe_globals, local_ns)
            fn = local_ns.get("evaluate") or safe_globals.get("evaluate")
            if callable(fn):
                return fn  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"[EVO] Failed to compile fitness_logic: {e}")

        return lambda agent_output, _history: float(agent_output.get("final_balance", 0.0))

    @staticmethod
    def _is_safe_logic(logic: str) -> bool:
        import re

        # NOTE: Use token-ish patterns to avoid false positives like "evaluate" matching "eval".
        banned_patterns = [
            r"\bimport\b",
            r"\bexec\b",
            r"\beval\b",
            r"open\s*\(",
            r"__",
            r"\bos\.",
            r"\bsys\.",
            r"subprocess",
            r"socket",
            r"shutil",
            r"pathlib",
        ]
        low = logic.lower()
        return not any(re.search(pat, low) for pat in banned_patterns)


async def run_background(
    engine: EvolutionEngine,
    interval: float = 5.0,
    log_every: int = 10,
) -> None:
    best_seen: Optional[float] = None
    while True:
        best, avg, _ = await engine.run_epoch()

        if best_seen is None or best > best_seen:
            best_seen = best
            logger.info(
                "[EVO] Gen %s: Max Fitness %.2f, Avg %.2f (new best)",
                engine.generation,
                best,
                avg,
            )
        elif log_every and engine.generation % log_every == 0:
            logger.info(
                "[EVO] Gen %s: Max Fitness %.2f, Avg %.2f",
                engine.generation,
                best,
                avg,
            )

        await asyncio.sleep(interval)
