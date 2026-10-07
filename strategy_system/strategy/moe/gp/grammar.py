"""Safe expression grammar and causal time-series operators for GP."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


_ARITHMETIC = ("add", "sub", "mul", "div", "power")
_UNARY = ("neg", "abs", "log", "sqrt", "signed_log", "signed_sqrt")
_TEMPORAL = ("lag", "diff", "derivative", "time_derivative", "rolling_mean", "rolling_std", "window", "block")
_CALCULUS = ("partial",)


def get_primitive_features(frame: pd.DataFrame | None = None, columns=None) -> tuple[str, ...]:
    """Return numeric causal terminals, excluding labels, IDs, and future data."""
    forbidden = ("future_", "target", "label", "barrier", "latent", "trade_allowed", "embedding")
    if columns is None:
        if frame is None:
            return ()
        columns = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    result = tuple(str(c) for c in columns if not any(x in str(c).lower() for x in forbidden)
                   and "timestamp" not in str(c).lower() and not str(c).lower().endswith("_id"))
    return result


def get_operators() -> tuple[str, ...]:
    """List operators supported by the safe expression interpreter."""
    return _ARITHMETIC + _UNARY + _TEMPORAL + _CALCULUS


def get_operator_rules() -> dict[str, dict[str, object]]:
    """Describe arity and recursion policy; temporal/calculus operators are single-use."""
    rules = {op: {"arity": 2, "recursive": True, "family": "arithmetic"} for op in _ARITHMETIC}
    rules.update({op: {"arity": 1, "recursive": True, "family": "arithmetic"} for op in _UNARY})
    rules.update({op: {"arity": 1, "recursive": False, "family": "temporal"} for op in _TEMPORAL})
    rules.update({op: {"arity": 1, "recursive": False, "family": "calculus"} for op in _CALCULUS})
    return rules


def calculate_expression_depth(node: "Node") -> int:
    return node.height


def calculate_expression_complexity(node: "Node") -> int:
    return node.size


def validate_expression(node: "Node", primitive_features=None, max_depth: int = 10) -> bool:
    """Validate shape, terminals, depth, and restricted temporal recursion."""
    rules = get_operator_rules()
    allowed = set(primitive_features) if primitive_features is not None else None
    if node.height > max_depth:
        return False
    def visit(current: Node, restricted_parent: bool = False) -> bool:
        if current.op == "var":
            return bool(current.value is not None and (allowed is None or str(current.value) in allowed))
        if current.op == "const":
            try:
                return np.isfinite(float(current.value))
            except (TypeError, ValueError):
                return False
        rule = rules.get(current.op)
        if rule is None or len(current.children) != rule["arity"] or (restricted_parent and current.op in _TEMPORAL + _CALCULUS):
            return False
        if current.op in _TEMPORAL and (not isinstance(current.value, (int, np.integer)) or int(current.value) < 1):
            return False
        restricted_branch = restricted_parent or current.op in _TEMPORAL + _CALCULUS
        return all(visit(child, restricted_branch) for child in current.children)
    return visit(node)


@dataclass(frozen=True)
class Node:
    """One immutable node in an expression tree."""

    op: str
    children: tuple["Node", ...] = ()
    value: str | float | int | None = None

    @property
    def height(self) -> int:
        return 1 + max((child.height for child in self.children), default=0)

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    def __str__(self) -> str:
        if self.op == "var":
            return str(self.value)
        if self.op == "const":
            return f"{float(self.value):g}"
        if self.op in _TEMPORAL + _CALCULUS:
            if self.op == "partial":
                return f"partial[{self.value}]({self.children[0]})"
            return f"{self.op}[{self.value}]({self.children[0]})"
        if self.op in _UNARY:
            return f"{self.op}({self.children[0]})"
        symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/", "power": "**"}
        if self.op in symbols:
            return f"({self.children[0]} {symbols[self.op]} {self.children[1]})"
        return f"{self.op}({', '.join(map(str, self.children))})"


def node_to_dict(node: Node) -> dict:
    return {"op": node.op, "value": node.value,
            "children": [node_to_dict(child) for child in node.children]}


def node_from_dict(data: dict) -> Node:
    return Node(op=str(data["op"]), value=data.get("value"),
                children=tuple(node_from_dict(child) for child in data.get("children", ())))


_MAX_MAGNITUDE = 1e6
_EPSILON = 1e-8


def _bounded(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan).clip(-_MAX_MAGNITUDE, _MAX_MAGNITUDE)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.where(denominator.abs() >= _EPSILON)
    return numerator.div(safe)


def _rolling(node_op: str, values: pd.Series, window: int) -> pd.Series:
    rolling = values.rolling(window=window, min_periods=window)
    if node_op == "rolling_mean":
        return rolling.mean()
    if node_op == "rolling_std":
        return rolling.std(ddof=0)
    if node_op == "window":
        return rolling.sum()
    raise ValueError(f"unknown rolling operator {node_op!r}")


def evaluate(node: Node, inputs: pd.DataFrame, wrt: str | None = None) -> tuple[pd.Series, pd.Series]:
    """Evaluate a tree and its partial derivative with respect to one input.

    The derivative propagates analytically through pointwise and temporal
    operators. Temporal operators use the same causal window on values and
    tangents; rolling standard deviation uses its exact population-variance
    derivative. No source-code evaluation or future observation is used.
    """
    if wrt is None:
        return _evaluate_value(node, inputs), pd.Series(0.0, index=inputs.index, dtype=np.float64)
    if node.op == "var":
        name = str(node.value)
        if name not in inputs:
            raise KeyError(f"missing GP input column {name!r}")
        value = pd.to_numeric(inputs[name], errors="coerce").astype(np.float64)
        tangent = pd.Series(1.0 if name == wrt else 0.0, index=inputs.index, dtype=np.float64)
        return value, tangent
    if node.op == "const":
        value = pd.Series(float(node.value), index=inputs.index, dtype=np.float64)
        return value, pd.Series(0.0, index=inputs.index, dtype=np.float64)

    if node.op == "partial":
        _, partial = evaluate(node.children[0], inputs, str(node.value))
        # Second-order derivatives are not part of this grammar.
        return _bounded(partial), pd.Series(0.0, index=inputs.index, dtype=np.float64)

    values = [evaluate(child, inputs, wrt) for child in node.children]
    (x, dx) = values[0]
    if node.op in {"lag", "diff", "derivative"}:
        lag = int(node.value)
        if node.op == "lag":
            return x.shift(lag), dx.shift(lag)
        value = x - x.shift(lag)
        tangent = dx - dx.shift(lag)
        if node.op == "derivative":
            value = value / lag
            tangent = tangent / lag
        return _bounded(value), _bounded(tangent)
    if node.op == "time_derivative":
        lag = int(node.value)
        delta = inputs.index.to_series().diff().dt.total_seconds() if isinstance(inputs.index, pd.DatetimeIndex) else pd.Series(1.0, index=inputs.index)
        denominator = delta.replace(0.0, np.nan).rolling(lag, min_periods=lag).sum()
        return _bounded(x.diff(lag).div(denominator)), _bounded(dx.diff(lag).div(denominator))
    if node.op in {"rolling_mean", "window", "block"}:
        window = int(node.value)
        if node.op == "block":
            groups = np.arange(len(x)) // window
            return (_bounded(x.groupby(groups).transform(lambda s: s.expanding().mean())),
                    _bounded(dx.groupby(groups).transform(lambda s: s.expanding().mean())))
        rolling_op = "window" if node.op == "window" else "rolling_mean"
        return _rolling(rolling_op, x, window), _rolling(rolling_op, dx, window)
    if node.op == "rolling_std":
        window = int(node.value)
        rolling = x.rolling(window=window, min_periods=window)
        mean_x = rolling.mean()
        mean_dx = dx.rolling(window=window, min_periods=window).mean()
        mean_xdx = (x * dx).rolling(window=window, min_periods=window).mean()
        value = rolling.std(ddof=0)
        tangent = _safe_divide(mean_xdx - mean_x * mean_dx, value)
        return _bounded(value), _bounded(tangent)
    if node.op == "neg":
        return -x, -dx
    if node.op == "abs":
        return x.abs(), np.sign(x) * dx
    if node.op == "signed_log":
        return np.sign(x) * np.log1p(x.abs()), _safe_divide(dx, 1.0 + x.abs())
    if node.op == "log":
        return _bounded(np.log(x.abs().clip(lower=_EPSILON))), _bounded(_safe_divide(dx, x))
    if node.op == "sqrt":
        value = np.sqrt(x.abs())
        return _bounded(value), _bounded(np.sign(x) * _safe_divide(dx, 2.0 * value.clip(lower=_EPSILON)))
    if node.op == "signed_sqrt":
        value = np.sign(x) * np.sqrt(x.abs())
        tangent = _safe_divide(dx, 2.0 * np.sqrt(x.abs()).clip(lower=_EPSILON))
        return _bounded(value), _bounded(tangent)
    if node.op in {"add", "sub", "mul", "div", "power"}:
        y, dy = values[1]
        if node.op == "add":
            return _bounded(x + y), _bounded(dx + dy)
        if node.op == "sub":
            return _bounded(x - y), _bounded(dx - dy)
        if node.op == "mul":
            return _bounded(x * y), _bounded(dx * y + x * dy)
        if node.op == "power":
            exponent = y.round().clip(-4, 4)
            value = x.pow(exponent)
            # Exponents are rounded to a small integer in this protected
            # operator, so their derivative is intentionally zero.
            tangent = exponent * x.pow(exponent - 1) * dx
            return _bounded(value), _bounded(tangent)
        value = _safe_divide(x, y)
        tangent = _safe_divide(dx * y - x * dy, y.pow(2))
        return _bounded(value), _bounded(tangent)
    raise ValueError(f"unknown GP operator {node.op!r}")


def evaluate_expression(node: Node, inputs: pd.DataFrame) -> pd.Series:
    """Evaluate an expression as one causal series."""
    return evaluate(node, inputs)[0]


def _evaluate_value(node: Node, inputs: pd.DataFrame) -> pd.Series:
    """Fast value-only interpreter used by GP scoring (no tangent propagation)."""
    if node.op == "var":
        name = str(node.value)
        if name not in inputs:
            raise KeyError(f"missing GP input column {name!r}")
        return pd.to_numeric(inputs[name], errors="coerce").astype(np.float64)
    if node.op == "const":
        return pd.Series(float(node.value), index=inputs.index, dtype=np.float64)
    if node.op == "partial":
        return _bounded(evaluate(node.children[0], inputs, wrt=str(node.value))[1])
    values = [_evaluate_value(child, inputs) for child in node.children]
    x = values[0]
    if node.op == "lag":
        return x.shift(int(node.value))
    if node.op in {"diff", "derivative"}:
        lag = int(node.value)
        result = x.diff(lag)
        return _bounded(result / lag) if node.op == "derivative" else _bounded(result)
    if node.op == "time_derivative":
        lag = int(node.value)
        if isinstance(inputs.index, pd.DatetimeIndex):
            delta = inputs.index.to_series().diff().dt.total_seconds()
        else:
            delta = pd.Series(1.0, index=inputs.index)
        denom = delta.replace(0.0, np.nan).rolling(lag, min_periods=lag).sum()
        return _bounded(x.diff(lag).div(denom))
    if node.op in {"rolling_mean", "window", "rolling_std"}:
        return _rolling(node.op, x, int(node.value))
    if node.op == "block":
        window = int(node.value)
        groups = np.arange(len(x)) // window
        return _bounded(x.groupby(groups).transform(lambda s: s.expanding().mean()))
    if node.op == "neg":
        return -x
    if node.op == "abs":
        return x.abs()
    if node.op == "signed_log":
        return np.sign(x) * np.log1p(x.abs())
    if node.op == "log":
        return _bounded(np.log(x.abs().clip(lower=_EPSILON)))
    if node.op == "sqrt":
        return _bounded(np.sqrt(x.abs()))
    if node.op == "signed_sqrt":
        return _bounded(np.sign(x) * np.sqrt(x.abs()))
    y = values[1]
    if node.op == "add":
        return _bounded(x + y)
    if node.op == "sub":
        return _bounded(x - y)
    if node.op == "mul":
        return _bounded(x * y)
    if node.op == "div":
        return _bounded(_safe_divide(x, y))
    if node.op == "power":
        return _bounded(x.pow(y.round().clip(-4, 4)))
    raise ValueError(f"unknown GP operator {node.op!r}")


def expression_partial(node: Node, variable: str, inputs: pd.DataFrame) -> pd.Series:
    """Evaluate ``d(expression)/d(variable)`` for the current input path."""
    return evaluate(node, inputs, wrt=variable)[1]


def tree_depth(node: Node) -> int:
    return node.height


def _random_terminal(rng: np.random.Generator, variables: tuple[str, ...]) -> Node:
    if rng.random() < 0.82:
        return Node("var", value=str(rng.choice(variables)))
    return Node("const", value=float(rng.choice((-2.0, -1.0, -0.5, 0.5, 1.0, 2.0))))


def variables_in(node: Node) -> tuple[str, ...]:
    """Return distinct variable terminals in first-occurrence order."""
    found = []
    for child in node.children:
        found.extend(variables_in(child))
    if node.op == "var":
        found.insert(0, str(node.value))
    return tuple(dict.fromkeys(found))


def _has_nonlinear_operation(node: Node) -> bool:
    return node.op in {"mul", "div", "power", "abs", "log", "sqrt", "signed_log", "signed_sqrt", "rolling_std"} or any(
        _has_nonlinear_operation(child) for child in node.children
    )


def random_tree(
    rng: np.random.Generator,
    variables: tuple[str, ...],
    max_depth: int,
    temporal_windows: tuple[int, ...] = (2, 4, 8, 16),
    grow: bool = True,
    allow_partial: bool = True,
    allow_temporal: bool = True,
) -> Node:
    """Create a bounded-depth tree using the configured grammar."""
    if not variables:
        raise ValueError("at least one GP input variable is required")
    if max_depth <= 1 or (grow and rng.random() < 0.32):
        return _random_terminal(rng, variables)

    binary_ops = ("add", "sub", "mul", "div", "power")
    unary_ops = ("neg", "abs", "log", "sqrt", "signed_log", "signed_sqrt") + (_TEMPORAL if allow_temporal else ())
    if rng.random() < 0.42:
        op = str(rng.choice(binary_ops))
        return Node(op, (
            random_tree(rng, variables, max_depth - 1, temporal_windows, grow, allow_partial, allow_temporal),
            random_tree(rng, variables, max_depth - 1, temporal_windows, grow, allow_partial, allow_temporal),
        ))

    if allow_partial and rng.random() < 0.12:
        child = random_tree(rng, variables, max_depth - 1, temporal_windows, grow, allow_partial=False, allow_temporal=False)
        eligible = variables_in(child)
        if eligible and _has_nonlinear_operation(child):
            return Node("partial", (child,), value=str(rng.choice(eligible)))
        return child

    op = str(rng.choice(unary_ops))
    # Restricted operators wrap an arithmetic-only subtree, preventing chains of
    # temporal transforms while retaining recursive arithmetic below them.
    restricted = op in _TEMPORAL + _CALCULUS
    child = random_tree(rng, variables, max_depth - 1, temporal_windows, grow, allow_partial=False, allow_temporal=False) if restricted else random_tree(rng, variables, max_depth - 1, temporal_windows, grow, allow_partial, allow_temporal)
    parameter = int(rng.choice(temporal_windows)) if op in _TEMPORAL else None
    return Node(op, (child,), value=parameter)


def node_at_path(node: Node, path: tuple[int, ...]) -> Node:
    current = node
    for index in path:
        current = current.children[index]
    return current


def replace_at_path(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    if not path:
        return replacement
    index = path[0]
    children = list(node.children)
    children[index] = replace_at_path(children[index], path[1:], replacement)
    return Node(node.op, tuple(children), node.value)


def all_paths(node: Node, prefix: tuple[int, ...] = ()) -> tuple[tuple[int, ...], ...]:
    paths = [prefix]
    for index, child in enumerate(node.children):
        paths.extend(all_paths(child, prefix + (index,)))
    return tuple(paths)


def has_nested_partial(node: Node, inside_partial: bool = False) -> bool:
    """Return whether a partial derivative is nested inside another one."""
    if inside_partial and node.op == "partial":
        return True
    nested = inside_partial or node.op == "partial"
    return any(has_nested_partial(child, nested) for child in node.children)
