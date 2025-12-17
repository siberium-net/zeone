"""
Species Catalog
===============
Predefined species specs for common node niches.

Specs are compatible with `cortex.evolution.engine.EvolutionEngine`:
- `grammar.sensors` / `grammar.actions` configure dynamic terminals
- `fitness_logic` is compiled once for evaluation (safe subset)
"""

from __future__ import annotations

from typing import Any, Dict, List

from cortex.self import NodeNiche


def get_species_spec(niche: NodeNiche) -> Dict[str, Any]:
    """Return a species spec for the given niche."""
    specs: Dict[NodeNiche, Dict[str, Any]] = {
        NodeNiche.NEURAL_MINER: NEURAL_MINER_SPEC,
        NodeNiche.TRAFFIC_WEAVER: TRAFFIC_WEAVER_SPEC,
        NodeNiche.STORAGE_KEEPER: STORAGE_KEEPER_SPEC,
        NodeNiche.ARBITRAGEUR: ARBITRAGEUR_SPEC,
        NodeNiche.CHAIN_WEAVER: CHAIN_WEAVER_SPEC,
    }
    return specs.get(niche, ARBITRAGEUR_SPEC)


# ---------------------------------------------------------------------------
# Fitness logic helpers (must be safe for EvolutionEngine sandbox)
# ---------------------------------------------------------------------------


DEFAULT_FITNESS_LOGIC = """\
def evaluate(agent_output, history):
    balance = float(agent_output.get("final_balance", 0.0))
    actions = agent_output.get("actions", [])
    return balance + 0.01 * len(actions)
"""


# ---------------------------------------------------------------------------
# Species definitions
# ---------------------------------------------------------------------------


NEURAL_MINER_SPEC: Dict[str, Any] = {
    "species": {
        "name": "neural_miner",
        "goal": "Maximize profit from GPU rental while maintaining hardware health",
        "domain": "compute",
        "description": "Manages GPU workloads and pricing",
    },
    "grammar": {
        "sensors": [
            "gpu_load",
            "gpu_temperature",
            "queue_length",
            "current_job_price",
            "my_balance",
        ],
        "actions": [
            "accept_job",
            "reject_job",
            "adjust_price_up",
            "adjust_price_down",
            "download_model",
            "sleep",
        ],
        "rule_count": {"min": 3, "max": 8},
        "max_depth": 3,
    },
    "fitness": {
        "objective": "maximize_profit",
        "signals": ["final_balance", "actions"],
        "notes": "MVP uses final_balance proxy.",
    },
    "fitness_logic": DEFAULT_FITNESS_LOGIC,
}


TRAFFIC_WEAVER_SPEC: Dict[str, Any] = {
    "species": {
        "name": "traffic_weaver",
        "goal": "Optimize routing for throughput and profit",
        "domain": "network",
        "description": "Manages traffic routing and peer selection",
    },
    "grammar": {
        "sensors": [
            "bandwidth_usage",
            "packet_loss",
            "ping_to_peers",
            "active_connections",
            "my_balance",
        ],
        "actions": [
            "adjust_price_up",
            "adjust_price_down",
            "drop_slow_peer",
            "add_fast_peer",
            "enable_caching",
            "disable_caching",
        ],
        "rule_count": {"min": 3, "max": 7},
        "max_depth": 2,
    },
    "fitness": {
        "objective": "maximize_throughput_profit",
        "signals": ["final_balance", "actions"],
        "notes": "MVP uses final_balance proxy.",
    },
    "fitness_logic": DEFAULT_FITNESS_LOGIC,
}


STORAGE_KEEPER_SPEC: Dict[str, Any] = {
    "species": {
        "name": "storage_keeper",
        "goal": "Store popular chunks, GC old data, maintain trust",
        "domain": "storage",
        "description": "Manages DHT storage and caching",
    },
    "grammar": {
        "sensors": [
            "disk_usage_percent",
            "chunk_popularity",
            "chunk_age_days",
            "trust_score",
            "my_balance",
        ],
        "actions": [
            "store_chunk",
            "delete_chunk",
            "adjust_price_up",
            "adjust_price_down",
            "replicate_popular",
        ],
        "rule_count": {"min": 3, "max": 6},
        "max_depth": 2,
    },
    "fitness": {
        "objective": "maximize_storage_profit",
        "signals": ["final_balance", "actions"],
        "notes": "MVP uses final_balance proxy.",
    },
    "fitness_logic": DEFAULT_FITNESS_LOGIC,
}


ARBITRAGEUR_SPEC: Dict[str, Any] = {
    "species": {
        "name": "arbitrageur",
        "goal": "Find price imbalances and profit from arbitrage",
        "domain": "trading",
        "description": "Trades SIBR/resources for profit",
    },
    "grammar": {
        "sensors": [
            "price_sibr",
            "price_inference",
            "price_storage",
            "market_volatility",
            "my_balance",
        ],
        "actions": [
            "buy_sibr",
            "sell_sibr",
            "buy_inference",
            "sell_inference",
            "hold",
        ],
        "rule_count": {"min": 2, "max": 5},
        "max_depth": 2,
    },
    "fitness": {
        "objective": "maximize_balance",
        "signals": ["final_balance"],
        "notes": "Uses final_balance directly.",
    },
    "fitness_logic": """\
def evaluate(agent_output, history):
    return float(agent_output.get("final_balance", 0.0))
""",
}


CHAIN_WEAVER_SPEC: Dict[str, Any] = {
    "species": {
        "name": "chain_weaver",
        "goal": "Automate blockchain operations and smart contract deployment",
        "domain": "blockchain",
        "description": "Manages on-chain operations",
    },
    "grammar": {
        "sensors": [
            "gas_price",
            "my_balance_chain",
            "pending_settlements",
            "stake_amount",
        ],
        "actions": [
            "settle_batch",
            "deploy_contract",
            "stake_tokens",
            "unstake_tokens",
            "wait",
        ],
        "rule_count": {"min": 2, "max": 4},
        "max_depth": 2,
    },
    "fitness": {
        "objective": "minimize_gas_maximize_settlements",
        "signals": ["final_balance", "actions"],
        "notes": "MVP uses final_balance proxy.",
    },
    "fitness_logic": DEFAULT_FITNESS_LOGIC,
}


def get_all_species_names() -> List[str]:
    """Return all available species names (by niche value)."""
    return [niche.value for niche in NodeNiche]

