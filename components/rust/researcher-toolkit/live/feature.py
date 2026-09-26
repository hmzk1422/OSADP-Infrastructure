"""Declarative feature definition, compiled for the Rust executor.

The research side writes features here; the Rust side executes the compiled
JSON. The contract is deliberately small and total: every operation has exactly
one Rust counterpart, and ``stresstest/`` asserts the two produce identical
``float`` values.

Causality
---------
``window`` is a strictly **trailing** window: at row ``i`` it aggregates rows
``max(0, i - size + 1) ..= i``. A short window at the start of a stream
aggregates over what exists so far (equivalent to pandas ``min_periods=1``), so
a feature is defined from the first row. ``window`` and ``zscore`` never look
forward.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

AGGREGATIONS = ("mean", "sum", "min", "max", "last", "std")


@dataclass(frozen=True)
class Expr:
    """A feature expression node.

    Supports ``+ - * /`` in both directions, so ``col("a") - col("b")`` reads
    naturally and ``1 - col("x")`` also works.
    """

    op: str
    args: tuple

    def _bin(self, op, other):
        return Expr(op, (self, _as_expr(other)))

    __add__ = lambda self, x: self._bin("add", x)
    __sub__ = lambda self, x: self._bin("sub", x)
    __mul__ = lambda self, x: self._bin("mul", x)
    __truediv__ = lambda self, x: self._bin("div", x)
    __radd__ = lambda self, x: _as_expr(x)._bin("add", self)
    __rsub__ = lambda self, x: _as_expr(x)._bin("sub", self)
    __rmul__ = lambda self, x: _as_expr(x)._bin("mul", self)
    __rtruediv__ = lambda self, x: _as_expr(x)._bin("div", self)

    def to_json(self):
        return {
            "op": self.op,
            "args": [a.to_json() if isinstance(a, Expr) else a for a in self.args],
        }


def _as_expr(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return lit(value)
    raise TypeError("features accept only Expr, int, or float values")


def col(name: str) -> Expr:
    """Reference an input column, e.g. ``col("bid_price")``."""
    if not isinstance(name, str) or not name:
        raise ValueError("column name must be a non-empty string")
    return Expr("column", (name,))


def lit(value: float) -> Expr:
    """A constant."""
    return Expr("literal", (float(value),))


def window(expression: Expr, size: int, aggregation: str = "mean") -> Expr:
    """Causal trailing window over ``expression``."""
    if not isinstance(expression, Expr):
        raise TypeError("window requires an Expr")
    if not isinstance(size, int) or size < 1:
        raise ValueError("window size must be a positive integer")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unsupported window aggregation: {aggregation}")
    return Expr("window", (expression, size, aggregation))


def zscore(expression: Expr, size: int) -> Expr:
    """Causal z-score of ``expression`` over a trailing window."""
    if not isinstance(expression, Expr):
        raise TypeError("zscore requires an Expr")
    if not isinstance(size, int) or size < 1:
        raise ValueError("window size must be a positive integer")
    return Expr("zscore", (expression, size))


def reconstruct(expression: Expr, size: int, aggregation: str = "mean") -> Expr:
    """Bar reconstruction: group ``size`` consecutive rows and aggregate.

    Only completed groups are emitted, so a trailing partial bar is dropped —
    the same rule the Rust executor applies.
    """
    if not isinstance(expression, Expr):
        raise TypeError("reconstruct requires an Expr")
    if not isinstance(size, int) or size < 1:
        raise ValueError("reconstruct size must be a positive integer")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    return Expr("reconstruct", (expression, size, aggregation))


def bid_ask() -> Expr:
    """Top-of-book spread: ``ask_price - bid_price``."""
    return col("best_ask_price") - col("best_bid_price")


def imbalance() -> Expr:
    """Book imbalance in ``[-1, 1]``: bid size minus ask size, over their sum."""
    return (col("best_bid_qty") - col("best_ask_qty")) / (
        col("best_bid_qty") + col("best_ask_qty")
    )


def microprice() -> Expr:
    """Size-weighted mid: the price a taker would cross to."""
    return (col("best_ask_price") * col("best_bid_qty") + col("best_bid_price") * col("best_ask_qty")) / (
        col("best_bid_qty") + col("best_ask_qty")
    )


class FeaturePipeline:
    """Collects named features and compiles them for the Rust executor."""

    def __init__(self):
        self._features = []

    def AddFeature(self, name: str, expression: Expr) -> "FeaturePipeline":
        """Declare a feature. Chainable, so declarations read as a pipeline."""
        if not isinstance(name, str) or not name:
            raise ValueError("feature name must be non-empty")
        if not isinstance(expression, Expr):
            raise TypeError("AddFeature requires an Expr; use col(), lit(), or operators")
        self._features.append((name, expression))
        return self

    add_feature = AddFeature

    # Exposed as methods for parity with the documented research API; they
    # return Expr, so `pipeline.window(...)` composes like a free function.
    def window(self, expression, size, aggregation="mean"):
        return window(expression, size, aggregation)

    def zscore(self, expression, size):
        return zscore(expression, size)

    def reconstruct(self, expression, size, aggregation="mean"):
        return reconstruct(expression, size, aggregation)

    @property
    def feature_names(self):
        return [name for name, _ in self._features]

    def compile(self) -> str:
        """Deterministic JSON the Rust `spec.rs` parses.

        ``allow_nan=False`` guarantees the document is strict JSON; a NaN in a
        literal is a bug worth surfacing at compile time, not at runtime.
        """
        return json.dumps(
            {
                "version": 1,
                "features": [
                    {"name": name, "expr": expr.to_json()} for name, expr in self._features
                ],
            },
            separators=(",", ":"),
            allow_nan=False,
            sort_keys=False,
        )


def _aggregate(values, aggregation):
    """Reduce a list of floats. Summation order matches the Rust side exactly."""
    if not values:
        return math.nan
    if aggregation == "mean":
        total = 0.0
        for v in values:
            total += v
        return total / len(values)
    if aggregation == "sum":
        total = 0.0
        for v in values:
            total += v
        return total
    if aggregation == "min":
        return min(values)
    if aggregation == "max":
        return max(values)
    if aggregation == "last":
        return values[-1]
    if aggregation == "std":
        return _std(values)
    raise ValueError(f"unsupported aggregation: {aggregation}")


def _std(values):
    """Sample standard deviation, matching pandas `.std()` (Bessel-corrected).

    A single sample has no spread, so this is NaN rather than 0.0 — "not enough
    history" must not read as "no variance".
    """
    n = len(values)
    if n < 2:
        return math.nan
    total = 0.0
    for v in values:
        total += v
    mean = total / n
    sq = 0.0
    for v in values:
        sq += (v - mean) * (v - mean)
    return math.sqrt(sq / (n - 1))


def _eval(expr_json, row, history):
    """Evaluate one compiled node.

    ``history`` is every row up to and including the current one, oldest first.
    Stateful nodes read only a **trailing** slice of it, so nothing here can
    see the future.
    """
    op = expr_json["op"]
    args = expr_json["args"]

    if op == "column":
        name = args[0]
        if name not in row:
            raise KeyError(
                f"missing required column: {name!r}. A window or z-score takes an "
                f"expression, not a feature name: pass bid_ask(), not "
                f"col({name!r})."
            )
        return float(row[name])

    if op == "literal":
        return float(args[0])

    if op == "window":
        size = int(args[1])
        aggregation = args[2]
        # Trailing window ending at the current row (inclusive).
        trailing = history[-size:] if size <= len(history) else history
        values = [_eval(args[0], r, history) for r in trailing]
        return _aggregate(values, aggregation)

    if op == "zscore":
        size = int(args[1])
        trailing = history[-size:] if size <= len(history) else history
        values = [_eval(args[0], r, history) for r in trailing]
        current = values[-1]
        mean = _aggregate(values, "mean")
        sd = _std(values)
        if math.isnan(sd):
            return math.nan
        return (current - mean) / sd

    if op == "reconstruct":
        # Streaming form: only complete groups are produced, so the value at row
        # i is the aggregate of the group ending at i.
        return _ReconstructState(args[0], int(args[1]), args[2]).value_at(history)

    a, b = (_eval(x, row, history) for x in args)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        # Match Rust: an exact zero is a hard failure, not `inf`.
        if b == 0.0:
            raise ZeroDivisionError("division by zero")
        return a / b
    raise ValueError(f"unknown op: {op!r}")


class _ReconstructState:
    """Bar reconstruction state, derived from the full history.

    `value_at` returns the aggregate of the completed group ending at the last
    history row, or NaN when that group is still partial. Keeping it derived
    (rather than incremental) is what makes a batch call and a streaming call
    agree exactly.
    """

    def __init__(self, inner_json, size, aggregation):
        self.inner_json = inner_json
        self.size = size
        self.aggregation = aggregation

    def value_at(self, history):
        n = len(history)
        index = n - 1
        if (index + 1) % self.size != 0:
            return math.nan
        group = history[index - self.size + 1: index + 1]
        values = [_eval(self.inner_json, r, history) for r in group]
        return _aggregate(values, self.aggregation)


def evaluate(spec, rows):
    """Run a compiled spec over ``rows`` and return one feature dict per row.

    A feature that is legitimately undefined (for example a std-dev over a
    single sample, or an incomplete reconstruct group) is reported as NaN so the
    caller can distinguish it from a real zero.
    """
    document = json.loads(spec) if isinstance(spec, str) else spec
    out = []
    for index, row in enumerate(rows):
        history = rows[: index + 1]
        result = {}
        for feature in document["features"]:
            try:
                result[feature["name"]] = _eval(feature["expr"], row, history)
            except ZeroDivisionError:
                raise
        out.append(result)
    return out


def reconstruct_rows(rows, size, aggregation="mean", time_col="event_time_ms", value_col="price"):
    """Batch bar reconstruction, returning one row per completed bar.

    The last (incomplete) bar is dropped, matching the streaming behaviour of the
    compiled `reconstruct` op.
    """
    if size < 1:
        raise ValueError("reconstruct size must be a positive integer")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    out = []
    for start in range(0, len(rows) - size + 1, size):
        group = rows[start: start + size]
        values = [float(r[value_col]) for r in group]
        out.append(
            {
                time_col: group[-1][time_col],
                value_col: _aggregate(values, aggregation),
            }
        )
    return out


