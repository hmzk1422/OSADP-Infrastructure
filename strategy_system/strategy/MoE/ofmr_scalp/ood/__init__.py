"""Out-of-distribution detection over causal raw-market temporal patterns."""

from .detector import OODRegimeDetector, detect_ood_regimes
from .encoder import TemporalPrimitiveEncoder, build_primitive_observations
from .regimes import OnlineRegimeRegistry

__all__ = [
    "OODRegimeDetector",
    "detect_ood_regimes",
    "TemporalPrimitiveEncoder",
    "build_primitive_observations",
    "OnlineRegimeRegistry",
]
