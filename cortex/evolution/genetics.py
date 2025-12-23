"""
Genetic operators and compilation for GGGP agents.
"""

import copy
import random
from typing import Any, List, Optional

from cortex.evolution.grammar import (
    ActionNode,
    ActionTerminal,
    ComparisonNode,
    ConditionNode,
    ConstNode,
    FuncNode,
    Gene,
    GrammarContext,
    GRAMMAR_CONTEXT,
    HistoryNode,
    IfNode,
    IndicatorNode,
    LogicNode,
    MathNode,
    SensorTerminal,
)


def random_action(context: Optional[GrammarContext] = None) -> Any:
    """Random action terminal (dynamic)."""
    ctx = context or GRAMMAR_CONTEXT
    name = random.choice(ctx.allowed_actions) if ctx.allowed_actions else "hold"

    # Legacy buy/sell amount support if action is buy/sell.
    if name in ("buy", "sell"):
        amount = round(random.uniform(0.1, 10.0), 2)
        return ActionNode(action=name, amount=amount)

    if name in {"set_price", "adjust_price", "adjust_price_up", "adjust_price_down"}:
        amount_expr = random_value(2, ctx)
        return ActionTerminal(name=name, amount=amount_expr)

    return ActionTerminal(name=name)


def random_sensor(context: Optional[GrammarContext] = None) -> SensorTerminal:
    ctx = context or GRAMMAR_CONTEXT
    name = random.choice(ctx.allowed_sensors) if ctx.allowed_sensors else "get_balance"
    return SensorTerminal(name=name)


def random_value(depth: int, context: Optional[GrammarContext]) -> Any:
    """Random numeric expression."""
    if depth <= 0 or random.random() < 0.3:
        if random.random() < 0.6:
            return random_sensor(context)
        return ConstNode(value=round(random.uniform(0.0, 100.0), 3))

    if random.random() < 0.2:
        ctx = context or GRAMMAR_CONTEXT
        sensor = random.choice(ctx.allowed_sensors) if ctx.allowed_sensors else "get_balance"
        steps_back = random.randint(1, 6)
        return HistoryNode(sensor=sensor, steps_back=steps_back)

    # Occasionally emit indicator node.
    if random.random() < 0.3:
        ind_name = random.choice(["rsi", "sma"])
        period = random.choice([7, 14, 21, 50])
        return IndicatorNode(name=ind_name, period=period, source=random_sensor(context))

    if random.random() < 0.4:
        op = random.choice(["Add", "Sub", "Mul", "Div"])
        left = random_value(depth - 1, context)
        right = random_value(depth - 1, context)
        return MathNode(operator=op, left=left, right=right)

    if random.random() < 0.3:
        fn = random.choice(["Log", "Exp", "Sin", "Min", "Max"])
        if fn in {"Min", "Max"}:
            args = [random_value(depth - 1, context), random_value(depth - 1, context)]
        else:
            args = [random_value(depth - 1, context)]
        return FuncNode(name=fn, args=args)

    return random_sensor(context)


def random_comparison(depth: int, context: Optional[GrammarContext]) -> ComparisonNode:
    comparator = random.choice(["<", "<=", ">", ">=", "=="])
    left = random_value(depth - 1, context)
    right = random_value(0, context)
    return ComparisonNode(left=left, comparator=comparator, right=right)


def random_logic(depth: int, context: Optional[GrammarContext]) -> Any:
    if depth <= 0 or random.random() < 0.5:
        return random_comparison(depth, context)
    op = random.choice(["AND", "OR"])
    operands = [random_comparison(depth - 1, context), random_comparison(depth - 1, context)]
    return LogicNode(operator=op, operands=operands)


def random_condition(depth: int = 2, context: Optional[GrammarContext] = None) -> Any:
    """Random condition tree."""
    ctx = context or GRAMMAR_CONTEXT
    return random_logic(depth, ctx)


def random_gene(depth: int = 2, context: Optional[GrammarContext] = None) -> Gene:
    """
    Random genome generation using dynamic context.

    Args:
        depth: maximum depth of conditional trees.
        context: GrammarContext with allowed terminals.
    """
    ctx = context or GRAMMAR_CONTEXT
    min_rules, max_rules = ctx.rule_count
    rule_count = random.randint(min_rules, max_rules)
    rules: List[Any] = []

    for _ in range(rule_count):
        if depth > 0 and random.random() < 0.6:
            cond = random_condition(depth, ctx)
            on_true = random_action(ctx)
            on_false = random_action(ctx) if random.random() < 0.8 else None
            rules.append(IfNode(condition=cond, on_true=on_true, on_false=on_false))
        else:
            rules.append(random_action(ctx))

    return Gene(rules=rules)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_genome(gene_root: Gene) -> str:
    """
    Translate Gene AST into executable Python source.
    Safe DSL: uses only local variables, no imports.
    """

    def compile_expr(node: Any) -> str:
        if isinstance(node, (int, float)):
            return repr(float(node))

        if isinstance(node, ConstNode):
            return repr(float(node.value))

        if isinstance(node, SensorTerminal):
            n = node.name.strip()
            if n.startswith("get_"):
                return f"api.{n}()"
            return f"api.get_{n}()"

        if isinstance(node, HistoryNode):
            sensor_name = node.sensor
            if isinstance(sensor_name, SensorTerminal):
                sensor_name = sensor_name.name
            sensor_key = str(sensor_name).strip()
            if sensor_key.startswith("get_"):
                sensor_key = sensor_key[4:]
            steps = int(node.steps_back) if node.steps_back else 1
            return f"api.get_history_value('{sensor_key}', {steps})"

        if isinstance(node, IndicatorNode):
            n = node.name.strip()
            if n.startswith("get_"):
                fn = n
            else:
                fn = f"get_{n}"
            args: List[str] = []
            if node.source is not None:
                args.append(compile_expr(node.source))
            if node.period:
                args.append(str(int(node.period)))
            return f"api.{fn}({', '.join(args)})" if args else f"api.{fn}()"

        if isinstance(node, MathNode):
            op = node.operator
            left = compile_expr(node.left)
            right = compile_expr(node.right)
            if op == "Add":
                return f"({left} + {right})"
            if op == "Sub":
                return f"({left} - {right})"
            if op == "Mul":
                return f"({left} * {right})"
            return f"safe_div({left}, {right})"

        if isinstance(node, FuncNode):
            name = node.name
            args = [compile_expr(a) for a in node.args]
            if name == "Log":
                return f"safe_log({args[0] if args else 1.0})"
            if name == "Exp":
                return f"math.exp({args[0] if args else 0.0})"
            if name == "Sin":
                return f"math.sin({args[0] if args else 0.0})"
            if name == "Min":
                return f"min({', '.join(args[:2])})"
            if name == "Max":
                return f"max({', '.join(args[:2])})"
            return "0.0"

        if isinstance(node, ComparisonNode):
            return f"({compile_expr(node.left)} {node.comparator} {compile_expr(node.right)})"

        if isinstance(node, LogicNode):
            op = "and" if node.operator.upper() == "AND" else "or"
            inner = f" {op} ".join(compile_expr(o) for o in node.operands)
            return f"({inner})" if inner else "False"

        # Legacy ConditionNode
        if isinstance(node, ConditionNode):
            return f"(balance {node.comparator} {node.threshold})"

        return "False"

    def compile_node(node: Any, indent: int = 0) -> List[str]:
        pad = " " * indent
        lines: List[str] = []
        if isinstance(node, ActionTerminal):
            name = node.name.strip()
            if node.amount is not None:
                lines.append(f"{pad}api.{name}({compile_expr(node.amount)})")
            else:
                lines.append(f"{pad}api.{name}()")
            lines.append(f"{pad}agent_output['actions'].append('{name}')")
        elif isinstance(node, ActionNode):
            # Legacy actions still affect balance.
            if node.action == "buy":
                lines.append(f"{pad}api.buy({node.amount})")
                lines.append(f"{pad}agent_output['actions'].append('buy')")
            elif node.action == "sell":
                lines.append(f"{pad}api.sell({node.amount})")
                lines.append(f"{pad}agent_output['actions'].append('sell')")
            else:
                lines.append(f"{pad}api.hold()")
                lines.append(f"{pad}agent_output['actions'].append('hold')")
        elif isinstance(node, IfNode):
            lines.append(f"{pad}if {compile_expr(node.condition)}:")
            if node.on_true:
                lines.extend(compile_node(node.on_true, indent + 4))
            else:
                lines.append(f"{pad}    pass")
            lines.append(f"{pad}else:")
            if node.on_false:
                lines.extend(compile_node(node.on_false, indent + 4))
            else:
                lines.append(f"{pad}    pass")
        elif isinstance(node, ConditionNode):
            # Legacy conditional
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
        "def safe_div(a, b):",
        "    if b == 0:",
        "        return 0.0",
        "    return a / b",
        "def safe_log(x):",
        "    if x <= 0:",
        "        return 0.0",
        "    return math.log(x)",
        "history = []",
        "agent_output = {'actions': []}",
        "balance = api.get_balance()",
    ]
    code_lines.extend(body_lines)
    code_lines.append("agent_output['final_balance'] = api.get_balance()")
    code_lines.append("history.append({'balance': agent_output['final_balance']})")
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
    elif isinstance(node, IfNode):
        nodes.append(node)
        nodes.extend(_collect_nodes(node.condition))
        if node.on_true:
            nodes.extend(_collect_nodes(node.on_true))
        if node.on_false:
            nodes.extend(_collect_nodes(node.on_false))
    elif isinstance(node, LogicNode):
        nodes.append(node)
        for o in node.operands:
            nodes.extend(_collect_nodes(o))
    elif isinstance(node, ComparisonNode):
        nodes.append(node)
        nodes.extend(_collect_nodes(node.left))
        nodes.extend(_collect_nodes(node.right))
    elif isinstance(node, IndicatorNode):
        nodes.append(node)
        if node.source:
            nodes.extend(_collect_nodes(node.source))
    elif isinstance(node, MathNode):
        nodes.append(node)
        nodes.extend(_collect_nodes(node.left))
        nodes.extend(_collect_nodes(node.right))
    elif isinstance(node, FuncNode):
        nodes.append(node)
        for a in node.args:
            nodes.extend(_collect_nodes(a))
    elif isinstance(node, HistoryNode):
        nodes.append(node)
    elif isinstance(node, (SensorTerminal, ActionTerminal, ConstNode, ActionNode, ConditionNode)):
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
        elif isinstance(root, IfNode):
            if root.condition is target:
                root.condition = replacement
            else:
                root.condition = replace(root.condition, target, replacement)
            if root.on_true is target:
                root.on_true = replacement
            else:
                root.on_true = replace(root.on_true, target, replacement) if root.on_true else None
            if root.on_false is target:
                root.on_false = replacement
            else:
                root.on_false = replace(root.on_false, target, replacement) if root.on_false else None
        elif isinstance(root, LogicNode):
            root.operands = [replace(o, target, replacement) for o in root.operands]
        elif isinstance(root, ComparisonNode):
            root.left = replace(root.left, target, replacement)
            root.right = replace(root.right, target, replacement)
        elif isinstance(root, IndicatorNode):
            if root.source is target:
                root.source = replacement
            else:
                root.source = replace(root.source, target, replacement) if root.source else None
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
        if isinstance(node, (ActionTerminal, ActionNode)):
            if random.random() < rate:
                return random_action()
            if isinstance(node, ActionTerminal) and node.amount is not None:
                node.amount = mutate_node(node.amount)
        elif isinstance(node, SensorTerminal):
            if random.random() < rate:
                return random_sensor()
        elif isinstance(node, ConstNode):
            if random.random() < rate:
                return ConstNode(value=round(random.uniform(0.0, 100.0), 3))
        elif isinstance(node, IndicatorNode):
            if random.random() < rate:
                return random_value(1, None)
            if node.source:
                node.source = mutate_node(node.source)
        elif isinstance(node, MathNode):
            if random.random() < rate:
                return random_value(2, None)
            node.left = mutate_node(node.left)
            node.right = mutate_node(node.right)
        elif isinstance(node, FuncNode):
            if random.random() < rate:
                return random_value(2, None)
            node.args = [mutate_node(a) for a in node.args]
        elif isinstance(node, HistoryNode):
            if random.random() < rate:
                return HistoryNode(sensor="market_price", steps_back=random.randint(1, 6))
        elif isinstance(node, ComparisonNode):
            if random.random() < rate:
                return random_comparison(1, None)
            node.left = mutate_node(node.left)
            node.right = mutate_node(node.right)
        elif isinstance(node, LogicNode):
            if random.random() < rate:
                return random_logic(1, None)
            node.operands = [mutate_node(o) for o in node.operands]
        elif isinstance(node, IfNode):
            if random.random() < rate:
                cond = random_condition(1, None)
                return IfNode(condition=cond, on_true=random_action(), on_false=random_action())
            node.condition = mutate_node(node.condition)
            if node.on_true:
                node.on_true = mutate_node(node.on_true)
            if node.on_false:
                node.on_false = mutate_node(node.on_false)
        elif isinstance(node, ConditionNode):
            if random.random() < rate:
                return ConditionNode(
                    comparator=random.choice(["<", "<=", ">", ">=", "=="]),
                    threshold=round(random.uniform(0.0, 100.0), 2),
                    on_true=random_action(),
                    on_false=random_action(),
                )
            if node.on_true:
                node.on_true = mutate_node(node.on_true)
            if node.on_false:
                node.on_false = mutate_node(node.on_false)
        elif isinstance(node, Gene):
            node.rules = [mutate_node(r) for r in node.rules]
        return node

    return mutate_node(mutated)
