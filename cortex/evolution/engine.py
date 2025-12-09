"""
Evolution Engine for GGGP agents.
"""

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

from cortex.evolution.genetics import (
    compile_genome,
    crossover,
    mutate,
    random_gene,
)
from cortex.evolution.grammar import Gene
from cortex.evolution.sandbox import AgentSandbox

logger = logging.getLogger(__name__)


class SimpleAPI:
    """Minimal Zeone-like API for fitness evaluation."""

    def __init__(self, balance: float = None, log_func=None):
        self.balance = balance if balance is not None else random.uniform(0, 100)
        self._log = log_func or (lambda msg: None)

    def log(self, msg: str) -> None:
        self._log(msg)

    def get_balance(self) -> float:
        return self.balance

    def send_message(self, peer: str, msg: str) -> None:
        self._log(f"send_message to {peer}: {msg}")


class EvolutionEngine:
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.1, elite: int = 2, output_dir: Path = Path("data/evolution")):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite = elite
        self.population: List[Gene] = []
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation = 0

    def initialize_population(self, size: int = None) -> None:
        size = size or self.population_size
        self.population = [random_gene() for _ in range(size)]

    async def evaluate_gene(self, gene: Gene) -> float:
        sandbox = AgentSandbox(timeout=1.0)
        api = SimpleAPI(log_func=logger.debug)
        code = compile_genome(gene)
        result = sandbox.run(code, api)
        if not result.get("ok"):
            return -1.0
        fitness = result.get("fitness")
        if fitness is None:
            fitness = api.get_balance()
        return float(fitness)

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


async def run_background(engine: EvolutionEngine, interval: float = 5.0):
    while True:
        best, avg, _ = await engine.run_epoch()
        logger.info(f"[EVO] Gen {engine.generation}: Max Fitness {best:.2f}, Avg {avg:.2f}")
        await asyncio.sleep(interval)
