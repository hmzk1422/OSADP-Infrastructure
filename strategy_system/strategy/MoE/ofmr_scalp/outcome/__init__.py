"""Outcome and supervised-target construction for OFMR scalp."""

from .timeout import (
    LinearTimeoutModel,
    build_forward_excursion_targets,
    build_forward_return_targets,
    fit_timeout_regression,
    select_timeout_horizon,
)
from .triple_barrier import (
    build_triple_barrier_dynamic,
    build_triple_barrier_fixed,
    build_triple_barrier_predicted,
    build_triple_barrier_sequence,
)

__all__ = [
    "LinearTimeoutModel",
    "build_forward_excursion_targets",
    "build_forward_return_targets",
    "fit_timeout_regression",
    "select_timeout_horizon",
    "build_triple_barrier_fixed",
    "build_triple_barrier_dynamic",
    "build_triple_barrier_predicted",
    "build_triple_barrier_sequence",
]
