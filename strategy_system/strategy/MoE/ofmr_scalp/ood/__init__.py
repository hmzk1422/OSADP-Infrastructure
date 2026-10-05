"""Causal recurring regime discovery and diagnostics."""

from .base_regime_encoder import (
    BaseRegimeEncoder,
    TCNRegimeEncoder,
    TS2VecRegimeEncoder,
    TransformerRegimeEncoder,
    TemporalWindowEncoder,
    build_temporal_windows,
    encode_windows,
    prepare_regime_input,
)
from .base_regime_segmenter import (
    BaseRegimeSegmenter,
    HysteresisClusterRegimeSegmenter,
    StickyHMMRegimeSegmenter,
)
from .ood import OODRegimeDetector, OODSystem, build_primitive_observations
from .regime_validator import RegimeValidator

__all__ = [
    "BaseRegimeEncoder", "TemporalWindowEncoder", "TCNRegimeEncoder",
    "TS2VecRegimeEncoder", "TransformerRegimeEncoder", "prepare_regime_input",
    "build_temporal_windows", "encode_windows", "BaseRegimeSegmenter",
    "HysteresisClusterRegimeSegmenter", "StickyHMMRegimeSegmenter",
    "RegimeValidator", "OODSystem", "OODRegimeDetector",
    "build_primitive_observations",
]
