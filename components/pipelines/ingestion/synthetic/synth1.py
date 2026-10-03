"""
Synth1: clean synthetic market for latent-state identification.

Purpose
-------
This is intentionally EASY data.

The goal is to test:

    "Can a latent-state model recover a known hidden regime?"

It is NOT intended to be a realistic trading simulator.

Structure
---------
Hidden HMM state
        ↓
regime-dependent information intensity
        ↓
observed feature groups
        ↓
LAGGED feature signals
        ↓
next-bar return
        ↓
price

Feature roles
-------------
0 = directional
1 = volatility
2 = mean-reversion
3 = momentum
4 = noise/context

Important causal rule
---------------------
Features at time t affect return at t+1.

There is NO use of feature[t] to generate return[t].
This prevents contemporaneous information leakage while keeping
Synth1 deliberately clean.

Properties
----------
- 4 persistent HMM states
- stable transition matrix
- stable feature loadings
- Gaussian shocks
- clear role-specific information intensity
- no endogenous state transitions
- no drifting feature loadings
- no heavy tails
- no microstructure
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# REGIME DEFINITION
# ============================================================

@dataclass(frozen=True)
class Regime:
    """
    One hidden market regime.

    drift:
        Baseline directional return component.

    volatility:
        Baseline return volatility.

    mean_reversion:
        Strength of price pull toward its slow reference.

    momentum:
        Strength of persistence in the directional signal.

    role_intensity:
        Information intensity for:

            directional
            volatility
            mean-reversion
            momentum
            noise/context

    role_bias:
        Clean regime-specific mean of each latent factor.
        This makes Synth1 intentionally identifiable.
    """

    drift: float
    volatility: float
    mean_reversion: float
    momentum: float

    role_intensity: tuple[
        float,
        float,
        float,
        float,
        float,
    ]

    role_bias: tuple[
        float,
        float,
        float,
        float,
        float,
    ]


# ============================================================
# REGIMES
# ============================================================

def _build_regimes(
    n_states: int,
) -> list[Regime]:
    """
    Build economically distinct clean regimes.

    For the intended n_states=4 case:

        State 0:
            calm / balanced / mildly mean-reverting

        State 1:
            bullish directional / momentum

        State 2:
            high-volatility / mean-reverting

        State 3:
            bearish directional / volatile
    """
    if n_states != 4:
        raise ValueError(
            "Synth1 currently requires n_states=4"
        )

    return [
        # ----------------------------------------------------
        # State 0: calm / balanced
        # ----------------------------------------------------
        Regime(
            drift=0.00000,
            volatility=0.0040,
            mean_reversion=0.025,
            momentum=0.015,

            # directional, vol, MR, momentum, noise
            role_intensity=(
                0.45,
                0.30,
                0.70,
                0.25,
                0.15,
            ),

            # Small/no directional bias.
            role_bias=(
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
            ),
        ),

        # ----------------------------------------------------
        # State 1: bullish / momentum
        # ----------------------------------------------------
        Regime(
            drift=+0.00030,
            volatility=0.0050,
            mean_reversion=0.010,
            momentum=0.070,

            role_intensity=(
                1.00,
                0.45,
                0.15,
                1.00,
                0.15,
            ),

            # Strong positive directional/momentum bias.
            role_bias=(
                +0.80,
                0.00,
                0.00,
                +0.70,
                0.00,
            ),
        ),

        # ----------------------------------------------------
        # State 2: high-volatility / mean-reverting
        # ----------------------------------------------------
        Regime(
            drift=0.00000,
            volatility=0.0100,
            mean_reversion=0.120,
            momentum=0.010,

            role_intensity=(
                0.15,
                1.00,
                1.00,
                0.10,
                0.20,
            ),

            # Strong volatility + MR information.
            role_bias=(
                0.00,
                +0.80,
                +0.80,
                0.00,
                0.00,
            ),
        ),

        # ----------------------------------------------------
        # State 3: bearish / volatile
        # ----------------------------------------------------
        Regime(
            drift=-0.00030,
            volatility=0.0075,
            mean_reversion=0.015,
            momentum=0.055,

            role_intensity=(
                1.00,
                0.80,
                0.20,
                0.90,
                0.15,
            ),

            # Strong negative directional bias.
            role_bias=(
                -0.80,
                +0.40,
                0.00,
                -0.60,
                0.00,
            ),
        ),
    ]


# ============================================================
# HMM
# ============================================================

def _build_transition_matrix(
    n_states: int,
    persistence: float = 0.965,
) -> np.ndarray:
    """
    Simple persistent HMM transition matrix.

    Synth1 intentionally keeps this symmetric and clean.
    More realistic asymmetric transitions belong in Synth2/3.
    """
    if not 0.0 < persistence < 1.0:
        raise ValueError(
            "persistence must be between 0 and 1"
        )

    transition = np.full(
        (n_states, n_states),
        (1.0 - persistence)
        / (n_states - 1),
        dtype=np.float64,
    )

    np.fill_diagonal(
        transition,
        persistence,
    )

    return transition


def _sample_hmm_states(
    n_bars: int,
    transition: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a latent HMM state sequence.
    """
    n_states = transition.shape[0]

    states = np.empty(
        n_bars,
        dtype=np.int64,
    )

    # Uniform initial state.
    states[0] = rng.integers(
        0,
        n_states,
    )

    for t in range(1, n_bars):
        previous = states[t - 1]

        states[t] = rng.choice(
            n_states,
            p=transition[previous],
        )

    return states


# ============================================================
# FEATURE ROLES
# ============================================================

def _feature_roles(
    n_features: int,
) -> np.ndarray:
    """
    Assign features evenly across five structural roles.

    Role IDs:

        0 = directional
        1 = volatility
        2 = mean-reversion
        3 = momentum
        4 = noise/context
    """
    if n_features < 5:
        raise ValueError(
            "Synth1 requires at least 5 features"
        )

    roles = np.arange(
        n_features,
        dtype=np.int64,
    )

    # Repeat 0..4 approximately evenly.
    roles %= 5

    return roles


# ============================================================
# SYNTH1
# ============================================================

def simulate_market_data(
    n_bars: int = 5_000,
    n_features: int = 50,
    n_states: int = 4,
    seed: int = 84,
    feature_noise: float = 0.5,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generate Synth1 clean synthetic market data.

    Parameters
    ----------
    n_bars:
        Number of observations.

    n_features:
        Number of observed features f0 ... f(N-1).

    n_states:
        Must be 4 for Synth1.

    seed:
        Reproducibility seed.

    feature_noise:
        Gaussian observation noise on features.

    initial_price:
        Initial asset price.

    Returns
    -------
    pandas.DataFrame
        Columns:

            f0 ... f(N-1)
            state
            price
            log_price
            return
            volatility

    Important
    ---------
    Feature information is strictly lagged.

        f[t] -> return[t+1]

    Therefore a model cannot exploit same-bar leakage.
    """
    if n_bars < 2:
        raise ValueError(
            "n_bars must be >= 2"
        )

    if n_features < 5:
        raise ValueError(
            "n_features must be >= 5"
        )

    if feature_noise < 0.0:
        raise ValueError(
            "feature_noise must be >= 0"
        )

    if initial_price <= 0.0:
        raise ValueError(
            "initial_price must be > 0"
        )

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Hidden regime process
    # --------------------------------------------------------

    regimes = _build_regimes(
        n_states
    )

    transition = _build_transition_matrix(
        n_states=n_states,
        persistence=0.965,
    )

    states = _sample_hmm_states(
        n_bars=n_bars,
        transition=transition,
        rng=rng,
    )

    # --------------------------------------------------------
    # Feature structure
    # --------------------------------------------------------

    roles = _feature_roles(
        n_features
    )

    # Stable feature-specific loadings.
    #
    # Synth1 deliberately does NOT drift these through time.
    feature_loading = rng.uniform(
        0.75,
        1.25,
        size=n_features,
    )

    # Stable AR persistence per feature.
    feature_phi = rng.uniform(
        0.80,
        0.95,
        size=n_features,
    )

    # Number of features assigned to each role.
    role_counts = np.bincount(
        roles,
        minlength=5,
    )

    features = np.zeros(
        (n_bars, n_features),
        dtype=np.float64,
    )

    # Role-level latent factors.
    role_factors = np.zeros(
        (5, n_bars),
        dtype=np.float64,
    )

    # Deliberately simple stable persistence.
    role_phi = np.asarray(
        [
            0.90,  # directional
            0.80,  # volatility
            0.92,  # mean-reversion
            0.95,  # momentum
            0.15,  # noise/context
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Price process
    # --------------------------------------------------------

    log_price = np.empty(
        n_bars,
        dtype=np.float64,
    )

    log_price[0] = np.log(
        initial_price
    )

    returns = np.zeros(
        n_bars,
        dtype=np.float64,
    )

    realized_volatility = np.zeros(
        n_bars,
        dtype=np.float64,
    )

    # Slow fair-value reference.
    reference = np.empty(
        n_bars,
        dtype=np.float64,
    )

    reference[0] = log_price[0]

    reference_alpha = 0.03

    # --------------------------------------------------------
    # Generate features + returns
    # --------------------------------------------------------

    for t in range(n_bars):

        current_state = int(
            states[t]
        )

        current_regime = regimes[
            current_state
        ]

        # ====================================================
        # 1. Generate the current latent factors
        # ====================================================

        for role in range(5):

            innovation = rng.normal()

            if t == 0:
                factor = innovation
            else:
                previous = role_factors[
                    role,
                    t - 1,
                ]

                factor = (
                    role_phi[role]
                    * previous
                    + np.sqrt(
                        1.0
                        - role_phi[role] ** 2
                    )
                    * innovation
                )

            # Regime-dependent mean / information structure.
            factor += current_regime.role_bias[
                role
            ]

            role_factors[
                role,
                t
            ] = factor

        # ====================================================
        # 2. Create observed features
        #
        # State controls how visible each latent factor is.
        #
        # This is the "information intensity" mechanism.
        # ====================================================

        for j in range(n_features):

            role = int(
                roles[j]
            )

            latent_signal = (
                feature_loading[j]
                * current_regime.role_intensity[
                    role
                ]
                * role_factors[
                    role,
                    t
                ]
            )

            noise = (
                feature_noise
                * rng.normal()
            )

            features[
                t,
                j
            ] = (
                latent_signal
                + noise
            )

        # ====================================================
        # 3. Aggregate CURRENT feature information
        #
        # These are intentionally stored separately because
        # they will drive the NEXT return, not this return.
        # ====================================================

        directional_signal = 0.0
        volatility_signal = 0.0
        mean_reversion_signal = 0.0
        momentum_signal = 0.0

        for j in range(n_features):

            role = int(
                roles[j]
            )

            value = features[
                t,
                j
            ]

            loading = feature_loading[
                j
            ]

            if role == 0:
                directional_signal += (
                    loading * value
                )

            elif role == 1:
                volatility_signal += (
                    loading * value
                )

            elif role == 2:
                mean_reversion_signal += (
                    loading * value
                )

            elif role == 3:
                momentum_signal += (
                    loading * value
                )

        # Normalize by role size.
        if role_counts[0] > 0:
            directional_signal /= np.sqrt(
                role_counts[0]
            )

        if role_counts[1] > 0:
            volatility_signal /= np.sqrt(
                role_counts[1]
            )

        if role_counts[2] > 0:
            mean_reversion_signal /= np.sqrt(
                role_counts[2]
            )

        if role_counts[3] > 0:
            momentum_signal /= np.sqrt(
                role_counts[3]
            )

        # ====================================================
        # 4. At time t, use information from t-1
        #
        # THIS IS THE IMPORTANT FIX.
        #
        # feature[t-1] -> return[t]
        # ====================================================

        if t > 0:

            previous_state = int(
                states[t - 1]
            )

            previous_regime = regimes[
                previous_state
            ]

            previous_directional = (
                previous_directional_signal
            )

            previous_volatility = (
                previous_volatility_signal
            )

            previous_mean_reversion = (
                previous_mean_reversion_signal
            )

            previous_momentum = (
                previous_momentum_signal
            )

            # -----------------------------------------------
            # Volatility
            # -----------------------------------------------

            sigma_multiplier = np.exp(
                0.10
                * previous_volatility
            )

            sigma = (
                previous_regime.volatility
                * sigma_multiplier
            )

            sigma = float(
                np.clip(
                    sigma,
                    0.001,
                    0.050,
                )
            )

            realized_volatility[t] = (
                sigma
            )

            # -----------------------------------------------
            # Update slow reference
            # -----------------------------------------------

            reference[t] = (
                (1.0 - reference_alpha)
                * reference[t - 1]
                + reference_alpha
                * log_price[t - 1]
            )

            price_deviation = (
                reference[t]
                - log_price[t - 1]
            )

            # -----------------------------------------------
            # Mean reversion
            # -----------------------------------------------

            mean_reversion_strength = (
                previous_regime.mean_reversion
                * (
                    1.0
                    + 0.10
                    * previous_mean_reversion
                )
            )

            mean_reversion_strength = float(
                np.clip(
                    mean_reversion_strength,
                    0.0,
                    0.20,
                )
            )

            mean_reversion_component = (
                mean_reversion_strength
                * price_deviation
            )

            # -----------------------------------------------
            # Directional feature contribution
            # -----------------------------------------------

            directional_component = (
                0.00025
                * previous_directional
            )

            # -----------------------------------------------
            # Momentum contribution
            # -----------------------------------------------

            momentum_component = (
                0.0010
                * previous_regime.momentum
                * previous_momentum
            )

            # -----------------------------------------------
            # GBM-style shock
            # -----------------------------------------------

            shock = rng.normal()

            log_return = (
                previous_regime.drift
                + directional_component
                + momentum_component
                + mean_reversion_component
                - 0.5 * sigma**2
                + sigma * shock
            )

            # Keep the clean benchmark numerically stable.
            log_return = float(
                np.clip(
                    log_return,
                    -0.10,
                    0.10,
                )
            )

            returns[t] = (
                log_return
            )

            log_price[t] = (
                log_price[t - 1]
                + log_return
            )

        else:
            # No previous information exists yet.
            realized_volatility[t] = (
                current_regime.volatility
            )

            reference[t] = (
                log_price[t]
            )

            returns[t] = 0.0

        # Save the CURRENT information signals so they become
        # the PREVIOUS signals on the next iteration.
        previous_directional_signal = (
            directional_signal
        )

        previous_volatility_signal = (
            volatility_signal
        )

        previous_mean_reversion_signal = (
            mean_reversion_signal
        )

        previous_momentum_signal = (
            momentum_signal
        )

    # ========================================================
    # Final price
    # ========================================================

    price = np.exp(
        log_price
    )

    # ========================================================
    # Output
    # ========================================================

    data = {
        f"f{j}": features[:, j]
        for j in range(n_features)
    }

    data["state"] = states
    data["price"] = price
    data["log_price"] = log_price
    data["return"] = returns
    data["volatility"] = realized_volatility

    return pd.DataFrame(data)