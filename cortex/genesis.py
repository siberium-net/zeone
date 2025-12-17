"""
Genesis Protocol
================
Auto-diagnostics + evolutionary bootstrap on first start.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from cortex.evolution.engine import EvolutionEngine
from cortex.evolution.genetics import compile_genome
from cortex.evolution.species import get_species_spec
from cortex.self import NodeNiche, NodeProfile, SelfAwareness

logger = logging.getLogger(__name__)


class GenesisProtocol:
    """
    "First Breath" protocol: diagnose node and evolve an initial agent.

    Usage:
        genesis = GenesisProtocol()
        result = await genesis.run()
    """

    DEFAULT_EPOCHS = 100
    DEFAULT_POPULATION = 50

    def __init__(
        self,
        *,
        data_dir: Path = Path("data"),
        epochs: int = DEFAULT_EPOCHS,
        population_size: int = DEFAULT_POPULATION,
    ):
        self.data_dir = data_dir
        self.genesis_dir = data_dir / "genesis"
        self.epochs = int(epochs)
        self.population_size = int(population_size)

        self._awareness = SelfAwareness()
        self._profile: Optional[NodeProfile] = None
        self._engine: Optional[EvolutionEngine] = None
        self._running = False

    async def run(self, *, niche_override: Optional[NodeNiche] = None) -> Dict[str, Any]:
        logger.info("[GENESIS] Starting Genesis Protocol...")

        self._profile = await self._awareness.diagnose()
        niche = niche_override or self._profile.dominant_niche or NodeNiche.ARBITRAGEUR
        if niche_override is not None:
            logger.info("[GENESIS] Niche override: %s", niche_override.name)
        logger.info("[GENESIS] Selected Niche: %s", niche.name)
        logger.info("[GENESIS] Profile tags=%s", self._profile.tags)

        spec = get_species_spec(niche)
        logger.info("[GENESIS] Selected species: %s", spec.get("species", {}).get("name"))

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.genesis_dir.mkdir(parents=True, exist_ok=True)
        self._engine = EvolutionEngine(
            population_size=self.population_size,
            species_spec=spec,
            output_dir=self.genesis_dir / "evolution",
        )
        self._engine.initialize_population()

        self._running = True

        best_fitness = float("-inf")
        best_gene = None

        for epoch in range(max(0, self.epochs)):
            if not self._running:
                break

            fit, avg, gene = await self._engine.run_epoch()
            logger.info("[EVO] Gen %s: Max Fitness %.3f, Avg %.3f", self._engine.generation, fit, avg)
            if best_gene is None or fit > best_fitness:
                best_fitness = fit
                best_gene = gene

        self._running = False

        if best_gene is None:
            # Conservative fallback: compile something deterministic from current population.
            if self._engine and self._engine.population:
                best_gene = self._engine.population[0]
                best_fitness = float(best_fitness if best_fitness != float("-inf") else 0.0)
            else:
                raise RuntimeError("Genesis finished with empty population")

        alpha_path = self.data_dir / "alpha_agent.py"
        alpha_path.write_text(compile_genome(best_gene))
        logger.info("[GENESIS] Alpha agent saved to %s", alpha_path)

        return {
            "profile": self._profile,
            "niche": niche.value,
            "species": spec.get("species", {}).get("name"),
            "best_fitness": best_fitness if best_fitness != float("-inf") else 0.0,
            "alpha_path": str(alpha_path),
            "epochs_completed": int(self._engine.generation if self._engine else 0),
        }

    def stop(self) -> None:
        self._running = False

    @property
    def profile(self) -> Optional[NodeProfile]:
        return self._profile


async def run_genesis(
    *,
    epochs: int = 100,
    population: int = 50,
    data_dir: str = "data",
    niche: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for `GenesisProtocol`."""
    genesis = GenesisProtocol(
        data_dir=Path(data_dir),
        epochs=epochs,
        population_size=population,
    )
    niche_override: Optional[NodeNiche] = None
    if niche:
        try:
            niche_override = NodeNiche(str(niche).lower())
        except Exception as e:
            raise ValueError(f"Unknown niche '{niche}'. Expected one of: {[n.value for n in NodeNiche]}") from e

    return await genesis.run(niche_override=niche_override)
