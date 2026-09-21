"""
Synth2: stochastic synthetic market for harder latent-state identification.

Purpose
-------
Synth2 is deliberately harder than Synth1.

Synth1 asks:

    "Can the model recover a clean hidden regime?"

Synth2 asks:

    "Can the model recover a hidden regime when the relationship between
     regime, features, and market behavior is noisy and non-stationary?"

Synth2 is still a controlled benchmark.

The true latent state is known and remains available in:

    df["state"]

But the observed feature -> state relationship is substantially less clean.

Structure
---------

Hidden HMM state
        ↓
stochastic regime information intensity
        ↓
time-varying latent factors
        ↓
drifting feature loadings
        ↓
cross-role contamination
        ↓
observed features
        ↓
LAGGED feature information
        ↓
next-bar return
        ↓
price

Harder aspects
--------------

1. Feature loadings drift slowly through time.
2. Information intensity changes inside a regime.
3. Neighboring regimes have overlapping feature distributions.
4. Feature groups contaminate each other.
5. Some features temporarily lose information.
6. Feature persistence differs by feature.
7. Transition probabilities are asymmetric.
8. Volatility varies stochastically.
9. Returns have Student-t shocks.
10. Market state is still known exactly.

Important causal rule
---------------------

Observed features at t influence return at t+1.

There is no feature[t] -> return[t] leakage.

Not yet included
-----------------

Synth3 territory includes:

    - strong price -> feature feedback
    - stress-driven regime transitions
    - explicit liquidity/order-book mechanics
    - volume/endogeneity
    - market-impact effects
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
    Hidden market regime.

    Compared with Synth1, these regimes are intentionally less
    distinct. Their feature intensity profiles overlap.
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

def _build_synth2_regimes(
    n_states: int,
) -> list[Regime]:
    """
    Build overlapping regimes.

    The states still have different dominant structures, but they
    are intentionally less separable than Synth1.

    State 0:
        relatively calm / balanced

    State 1:
        directional / momentum

    State 2:
        high-volatility / mean-reverting

    State 3:
        bearish / volatile / directional
    """

    if n_states != 4:
        raise ValueError(
            "Synth2 currently requires n_states=4"
        )

    return [
        # ----------------------------------------------------
        # State 0
        # ----------------------------------------------------
        Regime(
            drift=+0.00000,
            volatility=0.0045,
            mean_reversion=0.035,
            momentum=0.025,
            role_intensity=(
                0.65,
                0.45,
                0.70,
                0.40,
                0.25,
            ),
            role_bias=(
                +0.05,
                0.00,
                +0.05,
                +0.02,
                0.00,
            ),
        ),

        # ----------------------------------------------------
        # State 1
        # ----------------------------------------------------
        Regime(
            drift=+0.00022,
            volatility=0.0055,
            mean_reversion=0.020,
            momentum=0.060,
            role_intensity=(
                0.85,
                0.55,
                0.30,
                0.80,
                0.25,
            ),
            role_bias=(
                +0.30,
                +0.05,
                0.00,
                +0.25,
                0.00,
            ),
        ),

        # ----------------------------------------------------
        # State 2
        # ----------------------------------------------------
        Regime(
            drift=+0.00000,
            volatility=0.0085,
            mean_reversion=0.080,
            momentum=0.020,
            role_intensity=(
                0.40,
                0.85,
                0.85,
                0.30,
                0.30,
            ),
            role_bias=(
                0.00,
                +0.25,
                +0.30,
                0.00,
                0.00,
            ),
        ),

        # ----------------------------------------------------
        # State 3
        # ----------------------------------------------------
        Regime(
            drift=-0.00022,
            volatility=0.0070,
            mean_reversion=0.030,
            momentum=0.050,
            role_intensity=(
                0.85,
                0.75,
                0.35,
                0.70,
                0.30,
            ),
            role_bias=(
                -0.30,
                +0.20,
                0.00,
                -0.22,
                0.00,
            ),
        ),
    ]


# ============================================================
# ASYMMETRIC HMM
# ============================================================

def _build_synth2_transition_matrix() -> np.ndarray:
    """
    Asymmetric transition matrix.

    Synth1 uses a simple symmetric persistent HMM.

    Synth2 introduces realistic asymmetry while remaining controlled.

    Particularly:
        - calm/directional states can persist;
        - transitions into the high-volatility state are more probable
          than transitions back out;
        - the bearish state can be reached from multiple states.
    """

    transition = np.asarray(
        [
            # S0      S1      S2      S3
            [0.955,  0.020,  0.015, 0.010],
            [0.015,  0.955, 0.015, 0.015],
            [0.030,  0.025, 0.900, 0.045],
            [0.025,  0.020, 0.040, 0.915],
        ],
        dtype=np.float64,
    )

    if not np.allclose(
        transition.sum(axis=1),
        1.0,
    ):
        raise RuntimeError(
            "invalid Synth2 transition matrix"
        )

    return transition


def _sample_synth2_states(
    n_bars: int,
    transition: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample the hidden state sequence.
    """

    n_states = transition.shape[0]

    states = np.empty(
        n_bars,
        dtype=np.int64,
    )

    states[0] = rng.integers(
        0,
        n_states,
    )

    for t in range(1, n_bars):
        previous = int(
            states[t - 1]
        )

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
    Assign approximately equal numbers of features to each role.

    Role IDs:

        0 = directional
        1 = volatility
        2 = mean-reversion
        3 = momentum
        4 = noise/context
    """

    if n_features < 5:
        raise ValueError(
            "Synth2 requires at least 5 features"
        )

    roles = np.arange(
        n_features,
        dtype=np.int64,
    )

    roles %= 5

    return roles


# ============================================================
# SYNTH2
# ============================================================

def simulate_market_stochastic(
    n_bars: int = 5_000,
    n_features: int = 50,
    n_states: int = 4,
    seed: int = 84,
    feature_noise: float = 0.80,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generate Synth2 stochastic synthetic market data.

    Parameters
    ----------
    n_bars:
        Number of observations.

    n_features:
        Number of observed features f0 ... f(N-1).

    n_states:
        Must be 4.

    seed:
        Reproducibility seed.

    feature_noise:
        Base observation noise.

    initial_price:
        Initial asset price.

    Returns
    -------
    pandas.DataFrame
        f0 ... f(N-1)
        state
        price
        log_price
        return
        volatility

    The schema intentionally matches Synth1.
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

    # ========================================================
    # Hidden regime
    # ========================================================

    regimes = _build_synth2_regimes(
        n_states
    )

    transition = (
        _build_synth2_transition_matrix()
    )

    states = _sample_synth2_states(
        n_bars=n_bars,
        transition=transition,
        rng=rng,
    )

    # ========================================================
    # Feature structure
    # ========================================================

    roles = _feature_roles(
        n_features
    )

    role_counts = np.bincount(
        roles,
        minlength=5,
    )

    # Stable baseline loading.
    base_loading = rng.uniform(
        0.60,
        1.30,
        size=n_features,
    )

    # Slowly moving actual loading.
    loading = base_loading.copy()

    # Each feature has a different temporal memory.
    feature_phi = rng.uniform(
        0.65,
        0.97,
        size=n_features,
    )

    # Some features are more likely to become temporarily useless.
    dropout_probability = rng.uniform(
        0.002,
        0.015,
        size=n_features,
    )

    feature_dropout = np.zeros(
        n_features,
        dtype=bool,
    )

    # ========================================================
    # Feature storage
    # ========================================================

    # IMPORTANT:
    # The original implementation forgot to allocate this array.
    features = np.zeros(
        (n_bars, n_features),
        dtype=np.float64,
    )

    # ========================================================
    # Role-level latent factors
    # ========================================================

    role_factors = np.zeros(
        (5, n_bars),
        dtype=np.float64,
    )

    role_phi = np.asarray(
        [
            0.86,  # directional
            0.72,  # volatility
            0.90,  # mean-reversion
            0.94,  # momentum
            0.10,  # noise/context
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Price process
    # ========================================================

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

    reference = np.empty(
        n_bars,
        dtype=np.float64,
    )

    reference[0] = log_price[0]

    reference_alpha = 0.025

    # ========================================================
    # Stored lagged signals
    #
    # These are the signals generated by feature[t-1] and used
    # to calculate return[t].
    # ========================================================

    previous_directional_signal = 0.0
    previous_volatility_signal = 0.0
    previous_mean_reversion_signal = 0.0
    previous_momentum_signal = 0.0

    # ========================================================
    # Generate market
    # ========================================================

    for t in range(n_bars):

        state = int(
            states[t]
        )

        regime = regimes[
            state
        ]

        # ====================================================
        # 1. Stochastic information intensity
        #
        # Unlike Synth1, information intensity is not constant
        # throughout a regime.
        # ====================================================

        intensity_jitter = np.exp(
            0.30
            * rng.normal(
                size=5
            )
        )

        effective_intensity = (
            np.asarray(
                regime.role_intensity,
                dtype=np.float64,
            )
            * intensity_jitter
        )

        # Prevent occasional explosive intensities.
        effective_intensity = np.clip(
            effective_intensity,
            0.05,
            2.50,
        )

        # ====================================================
        # 2. Latent role factors
        # ====================================================

        for role in range(5):

            innovation = rng.normal()

            if t == 0:
                factor = innovation

            else:
                factor = (
                    role_phi[role]
                    * role_factors[
                        role,
                        t - 1,
                    ]
                    + np.sqrt(
                        1.0
                        - role_phi[role] ** 2
                    )
                    * innovation
                )

            # Regime-specific information bias.
            factor += (
                regime.role_bias[
                    role
                ]
                * effective_intensity[role]
            )

            role_factors[
                role,
                t,
            ] = factor

        # ====================================================
        # 3. Slowly drift feature loadings
        #
        # This is one of the major differences from Synth1.
        # ====================================================

        for j in range(n_features):

            loading_noise = rng.normal()

            loading[j] = (
                0.995
                * loading[j]
                + 0.005
                * base_loading[j]
                * np.exp(
                    0.10
                    * loading_noise
                )
            )

            loading[j] = float(
                np.clip(
                    loading[j],
                    0.15,
                    2.00,
                )
            )

        # ====================================================
        # 4. Temporary feature information dropout
        #
        # A feature can temporarily stop representing its
        # normal latent factor.
        # ====================================================

        for j in range(n_features):

            if (
                feature_dropout[j]
                and rng.random() < 0.15
            ):
                # Recover.
                feature_dropout[j] = False

            elif (
                not feature_dropout[j]
                and rng.random()
                < dropout_probability[j]
            ):
                # Enter dropout.
                feature_dropout[j] = True

        # ====================================================
        # 5. Generate observed features
        #
        # Features contain:
        #
        #   own latent role
        #   cross-role contamination
        #   drifting loading
        #   stochastic information intensity
        #   observation noise
        #   occasional information dropout
        #   feature-specific persistence
        # ====================================================

        for j in range(n_features):

            role = int(
                roles[j]
            )

            own_signal = (
                loading[j]
                * effective_intensity[role]
                * role_factors[
                    role,
                    t,
                ]
            )

            # Cross-role contamination.
            cross_role_1 = role_factors[
                (role + 1) % 5,
                t,
            ]

            cross_role_2 = role_factors[
                (role + 2) % 5,
                t,
            ]

            contamination = (
                0.10 * cross_role_1
                + 0.05 * cross_role_2
            )

            # Occasionally an entire feature becomes almost
            # uninformative.
            if feature_dropout[j]:
                own_signal *= 0.15

            observation_noise = (
                feature_noise
                * rng.normal()
            )

            # Some regimes make observations inherently harder.
            state_noise_multiplier = (
                1.0
                + 0.15 * state
            )

            raw_value = (
                own_signal
                + contamination
                + state_noise_multiplier
                * observation_noise
            )

            # Feature-specific temporal persistence.
            if t == 0:

                features[
                    t,
                    j,
                ] = raw_value

            else:

                phi = feature_phi[j]

                features[
                    t,
                    j,
                ] = (
                    phi
                    * features[
                        t - 1,
                        j,
                    ]
                    + (1.0 - phi)
                    * raw_value
                )

        # ====================================================
        # 6. Current feature information
        #
        # IMPORTANT:
        # These signals are NOT used for return[t].
        # They become the lagged signal for return[t+1].
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
                j,
            ]

            effective_loading = (
                loading[j]
            )

            if role == 0:

                directional_signal += (
                    effective_loading
                    * value
                )

            elif role == 1:

                volatility_signal += (
                    effective_loading
                    * value
                )

            elif role == 2:

                mean_reversion_signal += (
                    effective_loading
                    * value
                )

            elif role == 3:

                momentum_signal += (
                    effective_loading
                    * value
                )

        # Normalize group contributions.
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
        # 7. Generate return[t] from information available
        #    at t-1
        # ====================================================

        if t > 0:

            previous_state = int(
                states[t - 1]
            )

            previous_regime = regimes[
                previous_state
            ]

            # -----------------------------------------------
            # Stochastic volatility
            # -----------------------------------------------

            sigma_multiplier = np.exp(
                0.12
                * previous_volatility_signal
            )

            # Additional state-independent volatility noise.
            sigma_multiplier *= np.exp(
                0.05
                * rng.normal()
            )

            sigma = (
                previous_regime.volatility
                * sigma_multiplier
            )

            sigma = float(
                np.clip(
                    sigma,
                    0.001,
                    0.060,
                )
            )

            realized_volatility[t] = sigma

            # -----------------------------------------------
            # Slow price reference
            # -----------------------------------------------

            reference[t] = (
                (1.0 - reference_alpha)
                * reference[t - 1]
                + reference_alpha
                * log_price[t - 1]
            )

            deviation = (
                reference[t]
                - log_price[t - 1]
            )

            # -----------------------------------------------
            # Mean reversion
            # -----------------------------------------------

            mean_reversion_strength = (
                previous_regime.mean_reversion
                * np.exp(
                    0.08
                    * previous_mean_reversion_signal
                )
            )

            # Add a little randomness to how strongly
            # mean-reversion actually manifests.
            mean_reversion_strength *= np.exp(
                0.05
                * rng.normal()
            )

            mean_reversion_strength = float(
                np.clip(
                    mean_reversion_strength,
                    0.0,
                    0.16,
                )
            )

            mean_reversion_component = (
                mean_reversion_strength
                * deviation
            )

            # -----------------------------------------------
            # Direction
            # -----------------------------------------------

            directional_component = (
                0.00020
                * previous_directional_signal
            )

            # -----------------------------------------------
            # Momentum
            # -----------------------------------------------

            momentum_component = (
                0.00080
                * previous_regime.momentum
                * previous_momentum_signal
            )

            # -----------------------------------------------
            # Student-t shock
            #
            # df=7 keeps heavier tails than Gaussian but is
            # still substantially less aggressive than extreme
            # stress-test distributions.
            # -----------------------------------------------

            shock = rng.standard_t(
                df=7.0
            )

            shock /= np.sqrt(
                7.0 / (7.0 - 2.0)
            )

            log_return = (
                previous_regime.drift
                + directional_component
                + momentum_component
                + mean_reversion_component
                - 0.5 * sigma**2
                + sigma * shock
            )

            # Avoid mathematically extreme values from the
            # stochastic shock.
            log_return = float(
                np.clip(
                    log_return,
                    -0.15,
                    0.15,
                )
            )

            returns[t] = log_return

            log_price[t] = (
                log_price[t - 1]
                + log_return
            )

        else:

            realized_volatility[t] = (
                regime.volatility
            )

            reference[t] = (
                log_price[t]
            )

            returns[t] = 0.0

        # ====================================================
        # 8. Current feature information becomes available
        #    for the NEXT return.
        # ====================================================

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