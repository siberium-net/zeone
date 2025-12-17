import pytest


@pytest.mark.asyncio
async def test_architect_applies_collector_stats_to_actions():
    from cortex.evolution.architect import SpeciesArchitect

    class DummyCore:
        async def generate_spec(self, _user_goal: str):
            # Minimal valid spec; architect sanitizes and then applies stats.
            return {
                "species": {"name": "demo", "goal": "demo", "domain": "general"},
                "grammar": {"actions": [{"name": "buy", "params": {}}, {"name": "sell", "params": {}}]},
            }

    architect = SpeciesArchitect(core=DummyCore())
    spec = await architect.design(
        "demo",
        collector_stats={
            "top_accepted_types": ["accept_job", "adjust_price_up"],
            "acceptance_by_type": {"accept_job": 10},
        },
    )

    names = []
    for a in spec.get("grammar", {}).get("actions", []):
        if isinstance(a, dict) and isinstance(a.get("name"), str):
            names.append(a["name"])

    assert "accept_job" in names
    assert "adjust_price_up" in names
