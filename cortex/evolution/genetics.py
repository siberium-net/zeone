"""
Genetic operators and compilation for GGGP agents.
"""

import copy
import random
from typing import Any, List

from cortex.evolution.grammar import ActionNode, ConditionNode, Gene


def random_action() -> ActionNode:
    action = random.choice(["buy", "sell", "hold"])
    amount = round(random.uniform(0.1, 10.0), 2)
    return ActionNode(action=action, amount=amount)


def random_condition(depth: int = 0) -> ConditionNode:
    comparator = random.choice(["<", "<=", ">", ">=", "=="])
    threshold = round(random.uniform(0.0, 100.0), 2)
    # Shallow random children
    on_true = random_action()
    on_false = random_action() if depth > 0 else None
    return ConditionNode(comparator=comparator, threshold=threshold, on_true=on_true, on_false=on_false)


def random_gene(rule_count: int = 3) -> Gene:
    rules: List[Any] = []
    for _ in range(rule_count):
        if random.random() < 0.5:
            rules.append(random_action())
        else:
            rules.append(random_condition())
    return Gene(rules=rules)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_genome(gene_root: Gene) -> str:
    """
    Translate Gene AST into executable Python source.
    Safe DSL: uses only local variables, no imports.
    """

    def compile_node(node: Any, indent: int = 0) -> List[str]:
        pad = " " * indent
        lines: List[str] = []
        if isinstance(node, ActionNode):
            if node.action == "buy":
                lines.append(f"{pad}fitness -= {node.amount}")
            elif node.action == "sell":
                lines.append(f"{pad}fitness += {node.amount}")
            else:
                lines.append(f"{pad}fitness += 0  # hold")
        elif isinstance(node, ConditionNode):
            lines.append(f"{pad}if balance {node.comparator} {node.threshold}:")
            if node.on_true:
                lines.extend(compile_node(node.on_true, indent + 4))
            else:
                lines.append(f"{pad}    pass")
            lines.append(f"{pad}else:")
            if node.on_false:
                lines.extend(compile_node(node.on_false, indent + 4))
            else:
                lines.append(f"{pad}    pass")
        elif isinstance(node, Gene):
            for rule in node.rules:
                lines.extend(compile_node(rule, indent))
        else:
            lines.append(f"{pad}# Unknown node")
        return lines

    body_lines = compile_node(gene_root, indent=0)
    code_lines = [
        "fitness = api.get_balance()",
        "balance = fitness",
    ]
    code_lines.extend(body_lines)
    code_lines.append("api.log(f'fitness={fitness}')")
    code_lines.append(" ")
    return "\n".join(code_lines)


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def _collect_nodes(node: Any) -> List[Any]:
    nodes: List[Any] = []
    if isinstance(node, Gene):
        nodes.append(node)
        for r in node.rules:
            nodes.extend(_collect_nodes(r))
    elif isinstance(node, ConditionNode):
        nodes.append(node)
        if node.on_true:
            nodes.extend(_collect_nodes(node.on_true))
        if node.on_false:
            nodes.extend(_collect_nodes(node.on_false))
    elif isinstance(node, ActionNode):
        nodes.append(node)
    return nodes


def crossover(parent_a: Gene, parent_b: Gene) -> Gene:
    """Swap random compatible subtrees."""
    a_copy = copy.deepcopy(parent_a)
    b_copy = copy.deepcopy(parent_b)

    a_nodes = _collect_nodes(a_copy)
    b_nodes = _collect_nodes(b_copy)
    if not a_nodes or not b_nodes:
        return a_copy

    node_a = random.choice(a_nodes)
    same_type_nodes = [n for n in b_nodes if isinstance(n, type(node_a))]
    if not same_type_nodes:
        return a_copy
    node_b = random.choice(same_type_nodes)

    def replace(root: Any, target: Any, replacement: Any) -> Any:
        if root is target:
            return replacement
        if isinstance(root, Gene):
            root.rules = [replace(r, target, replacement) for r in root.rules]
        elif isinstance(root, ConditionNode):
            if root.on_true is target:
                root.on_true = replacement
            else:
                root.on_true = replace(root.on_true, target, replacement) if root.on_true else None
            if root.on_false is target:
                root.on_false = replacement
            else:
                root.on_false = replace(root.on_false, target, replacement) if root.on_false else None
        return root

    replace(a_copy, node_a, copy.deepcopy(node_b))
    return a_copy


def mutate(gene: Gene, rate: float = 0.1) -> Gene:
    """Randomly mutate nodes with given probability."""
    mutated = copy.deepcopy(gene)

    def mutate_node(node: Any) -> Any:
        if isinstance(node, ActionNode):
            if random.random() < rate:
                return random_action()
        elif isinstance(node, ConditionNode):
            if random.random() < rate:
                return random_condition()
            if node.on_true:
                node.on_true = mutate_node(node.on_true)
            if node.on_false:
                node.on_false = mutate_node(node.on_false)
        elif isinstance(node, Gene):
            node.rules = [mutate_node(r) for r in node.rules]
        return node

    return mutate_node(mutated)

