"""Research-to-executor feature DSL.

Only values represented by this DSL are promised to be lossless across the
Python and Rust implementations.  User callables are intentionally rejected;
there is no safe or deterministic way to infer their semantics.
"""

from .feature import (
    AGGREGATIONS,
    Expr,
    FeaturePipeline,
    bid_ask,
    col,
    evaluate,
    imbalance,
    lit,
    microprice,
    reconstruct,
    reconstruct_rows,
    window,
    zscore,
)

__all__ = [
    "AGGREGATIONS",
    "Expr",
    "FeaturePipeline",
    "bid_ask",
    "col",
    "evaluate",
    "imbalance",
    "lit",
    "microprice",
    "reconstruct",
    "reconstruct_rows",
    "window",
    "zscore",
]
