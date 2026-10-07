"""Outcome and supervised-target construction for OFMR scalp.

Live API only — ``timeout.py`` stays in this package fully dormant and is
intentionally NOT re-exported (S3C of the refactor plan).
"""

from .alignment import (
    align_regimes_to_index,
    attach_raw_columns,
    build_gp_source,
    build_model_frame,
    chronological_train_mask,
)
from .outcomes import (
    MetalabelResult,
    OutcomeResult,
    StateLinregModel,
    StateLinregResult,
    build_forward_excursion_targets,
    build_forward_return_targets,
    build_predicted_barriers,
    build_supervision_targets,
    fit_metalabel_tree,
    fit_state_linregs,
    predict_excursions,
    run_outcomes,
)
from .triple_barrier import (
    build_triple_barrier_dynamic,
    build_triple_barrier_fixed,
    build_triple_barrier_predicted,
    build_triple_barrier_sequence,
)

__all__ = [
    "MetalabelResult",
    "OutcomeResult",
    "StateLinregModel",
    "StateLinregResult",
    "align_regimes_to_index",
    "attach_raw_columns",
    "build_forward_excursion_targets",
    "build_forward_return_targets",
    "build_gp_source",
    "build_model_frame",
    "build_predicted_barriers",
    "build_supervision_targets",
    "build_triple_barrier_dynamic",
    "build_triple_barrier_fixed",
    "build_triple_barrier_predicted",
    "build_triple_barrier_sequence",
    "chronological_train_mask",
    "fit_metalabel_tree",
    "fit_state_linregs",
    "predict_excursions",
    "run_outcomes",
]
