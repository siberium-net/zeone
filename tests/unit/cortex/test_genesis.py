"""
Genesis Protocol unit tests.
"""

from __future__ import annotations

from pathlib import Path

from cortex.self import NodeNiche, SelfAwareness
from cortex.evolution.species import get_all_species_names, get_species_spec


async def test_self_awareness_diagnose():
    awareness = SelfAwareness()
    profile = await awareness.diagnose()

    assert profile is not None
    assert profile.hardware.cpu_cores > 0
    assert profile.hardware.ram_total_gb > 0
    assert profile.dominant_niche is not None
    assert len(profile.niches) > 0


def test_species_catalog():
    names = get_all_species_names()
    assert len(names) >= 4

    for niche in NodeNiche:
        spec = get_species_spec(niche)
        assert "species" in spec
        assert "grammar" in spec
        assert "fitness" in spec
        assert "sensors" in spec["grammar"]
        assert "actions" in spec["grammar"]


async def test_genesis_protocol(tmp_path):
    from cortex.genesis import GenesisProtocol

    genesis = GenesisProtocol(
        data_dir=Path(tmp_path),
        epochs=3,
        population_size=8,
    )

    result = await genesis.run()

    assert "niche" in result
    assert "best_fitness" in result
    assert "alpha_path" in result
    assert Path(result["alpha_path"]).exists()

