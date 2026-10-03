"""
Synth3: endogenous stochastic synthetic market for harder strategy research.

Purpose
-------
Synth3 is a harder, more market-like controlled benchmark.

Synth1 asks:

    "Can the model recover a clean hidden regime?"

Synth2 asks:

    "Can the model recover a hidden regime when the relationship between
     regime, features, and market behavior is noisy and non-stationary?"

Synth3 asks:

    "Can the model recover useful structure when the market itself is
     stochastic, endogenous, partially unstable, and subject to stress?"

Synth3 adds:

    1. Endogenous regime transitions.
    2. Persistent stochastic volatility.
    3. Fast + slow volatility dynamics.
    4. Volatility clustering.
    5. Asymmetric volatility response to returns.
    6. Explicit jump process.
    7. Latent liquidity / stress state.
    8. Price -> feature feedback.
    9. Slowly drifting feature loadings.
    10. Rare structural breaks.
    11. Temporary feature information dropout.
    12. Cross-role contamination.
    13. Feature-specific persistence.
    14. Nonlinear mean reversion.
    15. Heavy-tailed continuous shocks.

Important causal rule
---------------------

Observed features at time t influence return at t+1.

Price information used to construct features at t comes only from
information already available through t-1.

The current state therefore affects current observed features, but
does not directly leak into return[t].

The true hidden state remains available in:

    df["state"]

The output schema intentionally matches Synth1 / Synth2.
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

    Compared with Synth1 / Synth2, the regime is no longer the
    complete description of market behavior. It provides a
    baseline around which stochastic volatility, liquidity,
    jumps, and endogenous transitions operate.

    drift:
        Baseline directional return component.

    volatility:
        Baseline volatility level.

    mean_reversion:
        Baseline nonlinear mean-reversion strength.

    momentum:
        Baseline momentum strength.

    role_intensity:
        Information intensity for:

            directional
            volatility
            mean-reversion
            momentum
            noise/context

    role_bias:
        Regime-specific latent factor bias.

    jump_probability:
        Baseline probability of an explicit jump.
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

    jump_probability: float


# ============================================================
# REGIMES
# ============================================================

def _build_synth3_regimes(
    n_states: int,
) -> list[Regime]:
    """
    Build overlapping regimes.

    State 0:
        calm / balanced

    State 1:
        bullish / directional / momentum

    State 2:
        high-volatility / mean-reverting / stressed

    State 3:
        bearish / volatile / directional
    """

    if n_states != 4:
        raise ValueError(
            "Synth3 currently requires n_states=4"
        )

    return [
        # ----------------------------------------------------
        # State 0: calm / balanced
        # ----------------------------------------------------
        Regime(
            drift=0.00000,
            volatility=0.0035,
            mean_reversion=0.040,
            momentum=0.020,

            role_intensity=(
                0.55,
                0.45,
                0.65,
                0.35,
                0.30,
            ),

            role_bias=(
                +0.05,
                0.00,
                +0.05,
                +0.02,
                0.00,
            ),

            jump_probability=0.0010,
        ),

        # ----------------------------------------------------
        # State 1: bullish / directional / momentum
        # ----------------------------------------------------
        Regime(
            drift=+0.00025,
            volatility=0.0055,
            mean_reversion=0.015,
            momentum=0.070,

            role_intensity=(
                0.85,
                0.55,
                0.35,
                0.80,
                0.30,
            ),

            role_bias=(
                +0.25,
                +0.05,
                0.00,
                +0.25,
                0.00,
            ),

            jump_probability=0.0020,
        ),

        # ----------------------------------------------------
        # State 2: high-volatility / mean-reverting
        # ----------------------------------------------------
        Regime(
            drift=0.00000,
            volatility=0.0090,
            mean_reversion=0.100,
            momentum=0.015,

            role_intensity=(
                0.45,
                0.90,
                0.85,
                0.25,
                0.35,
            ),

            role_bias=(
                0.00,
                +0.35,
                +0.30,
                0.00,
                0.00,
            ),

            jump_probability=0.0080,
        ),

        # ----------------------------------------------------
        # State 3: bearish / volatile / directional
        # ----------------------------------------------------
        Regime(
            drift=-0.00025,
            volatility=0.0075,
            mean_reversion=0.020,
            momentum=0.060,

            role_intensity=(
                0.85,
                0.75,
                0.40,
                0.75,
                0.35,
            ),

            role_bias=(
                -0.25,
                +0.20,
                0.00,
                -0.20,
                0.00,
            ),

            jump_probability=0.0060,
        ),
    ]


# ============================================================
# BASE TRANSITION MATRIX
# ============================================================

def _build_synth3_transition_matrix() -> np.ndarray:
    """
    Base asymmetric transition matrix.

    Synth3 subsequently modifies these probabilities using
    observed market stress, volatility, liquidity and direction.

    Therefore the transition process is endogenous rather than
    being a completely independent HMM.
    """

    transition = np.asarray(
        [
            # S0      S1      S2      S3
            [0.955,  0.015,  0.020, 0.010],
            [0.015,  0.945,  0.020, 0.020],
            [0.025,  0.025,  0.900, 0.050],
            [0.015,  0.020,  0.050, 0.915],
        ],
        dtype=np.float64,
    )

    if not np.allclose(
        transition.sum(axis=1),
        1.0,
    ):
        raise RuntimeError(
            "invalid Synth3 transition matrix"
        )

    return transition


# ============================================================
# ENDOGENOUS TRANSITION
# ============================================================

def _build_endogenous_transition(
    previous_state: int,
    previous_return: float,
    previous_sigma: float,
    previous_liquidity: float,
    base_transition: np.ndarray,
) -> np.ndarray:
    """
    Modify transition probabilities according to market stress.

    Large moves and poor liquidity increase the probability of
    entering a stressed / high-volatility regime.

    Direction also matters:

        positive stress -> bullish / momentum state
        negative stress -> bearish / directional state

    This is one of the major differences from Synth1 / Synth2.
    """

    probabilities = (
        base_transition[previous_state].copy()
    )

    standardized_move = (
        abs(previous_return)
        / max(previous_sigma, 1e-8)
    )

    stress = (
        0.55
        * np.clip(
            standardized_move / 3.0,
            0.0,
            1.0,
        )
        + 0.45
        * previous_liquidity
    )

    stress = float(
        np.clip(
            stress,
            0.0,
            2.0,
        )
    )

    # --------------------------------------------------------
    # Stress reduces probability of remaining calm.
    # --------------------------------------------------------

    probabilities[0] *= np.exp(
        -0.90 * stress
    )

    # --------------------------------------------------------
    # Stress increases probability of entering the
    # high-volatility state.
    # --------------------------------------------------------

    probabilities[2] *= np.exp(
        +0.90 * stress
    )

    # --------------------------------------------------------
    # Directional stress biases the directional regime.
    # --------------------------------------------------------

    if previous_return > 0.0:

        probabilities[1] *= np.exp(
            +0.55 * stress
        )

        probabilities[3] *= np.exp(
            -0.25 * stress
        )

    elif previous_return < 0.0:

        probabilities[3] *= np.exp(
            +0.55 * stress
        )

        probabilities[1] *= np.exp(
            -0.25 * stress
        )

    probabilities = np.maximum(
        probabilities,
        1e-15,
    )

    probabilities /= (
        probabilities.sum()
    )

    return probabilities


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
            "Synth3 requires at least 5 features"
        )

    roles = np.arange(
        n_features,
        dtype=np.int64,
    )

    roles %= 5

    return roles


# ============================================================
# SYNTH3
# ============================================================

def simulate_market_realistic(
    n_bars: int = 5_000,
    n_features: int = 50,
    n_states: int = 4,
    seed: int = 84,
    feature_noise: float = 0.90,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generate Synth3 stochastic synthetic market data.

    Parameters
    ----------
    n_bars:
        Number of observations.

    n_features:
        Number of observed features f0 ... f(N-1).

    n_states:
        Must be 4 for Synth3.

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

    Causality
    ---------
    Features at t influence return[t+1].

    Price feedback into features uses only price information
    known before return[t] is generated.
    """

    # ========================================================
    # Validation
    # ========================================================

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

    # ========================================================
    # RNG
    # ========================================================

    rng = np.random.default_rng(
        seed
    )

    # ========================================================
    # Hidden regime configuration
    # ========================================================

    regimes = _build_synth3_regimes(
        n_states
    )

    base_transition = (
        _build_synth3_transition_matrix()
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

    # Stable baseline feature loadings.
    base_loading = rng.uniform(
        0.55,
        1.35,
        size=n_features,
    )

    # Slowly drifting actual feature loadings.
    loading = base_loading.copy()

    # Rare structural breaks.
    structural_break_multiplier = np.ones(
        n_features,
        dtype=np.float64,
    )

    # Each feature has different temporal persistence.
    feature_phi = rng.uniform(
        0.55,
        0.97,
        size=n_features,
    )

    # Some features can temporarily lose information.
    dropout_probability = rng.uniform(
        0.001,
        0.012,
        size=n_features,
    )

    feature_dropout = np.zeros(
        n_features,
        dtype=bool,
    )

    # ========================================================
    # Feature storage
    # ========================================================

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
            0.84,  # directional
            0.68,  # volatility
            0.89,  # mean-reversion
            0.93,  # momentum
            0.08,  # noise/context
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

    # Slow fair-value reference.
    reference = np.empty(
        n_bars,
        dtype=np.float64,
    )

    reference[0] = (
        log_price[0]
    )

    reference_alpha = 0.025

    # ========================================================
    # Latent liquidity / stress
    #
    # 0 = normal liquidity
    # 1 = severe stress
    # ========================================================

    liquidity = np.zeros(
        n_bars,
        dtype=np.float64,
    )

    # ========================================================
    # Multi-timescale volatility state
    # ========================================================

    initial_state = int(
        0
    )

    # The starting state is sampled below once.
    initial_state = rng.integers(
        0,
        n_states,
    )

    states = np.empty(
        n_bars,
        dtype=np.int64,
    )

    states[0] = initial_state

    initial_volatility = (
        regimes[initial_state].volatility
    )

    fast_log_sigma = np.log(
        initial_volatility
    )

    slow_log_sigma = np.log(
        initial_volatility
    )

    # ========================================================
    # Lagged feature signals
    # ========================================================

    previous_directional_signal = 0.0
    previous_volatility_signal = 0.0
    previous_mean_reversion_signal = 0.0
    previous_momentum_signal = 0.0

    # ========================================================
    # Generate market
    # ========================================================

    for t in range(n_bars):

        # ====================================================
        # 0. Endogenous state transition
        #
        # The current state depends partly on the previous
        # market realization.
        # ====================================================

        if t > 0:

            previous_state = int(
                states[t - 1]
            )

            transition_probability = (
                _build_endogenous_transition(
                    previous_state=previous_state,
                    previous_return=returns[t - 1],
                    previous_sigma=realized_volatility[t - 1],
                    previous_liquidity=liquidity[t - 1],
                    base_transition=base_transition,
                )
            )

            states[t] = rng.choice(
                n_states,
                p=transition_probability,
            )

        state = int(
            states[t]
        )

        regime = regimes[
            state
        ]

        # ====================================================
        # 1. Latent liquidity / stress
        #
        # Liquidity is persistent and becomes worse after
        # large market moves.
        # ====================================================

        if t > 0:

            previous_sigma = max(
                realized_volatility[t - 1],
                1e-8,
            )

            standardized_abs_return = (
                abs(returns[t - 1])
                / previous_sigma
            )

            liquidity_shock_probability = (
                0.0015
                + 0.0040
                * liquidity[t - 1]
            )

            stress_shock = (
                1.0
                if rng.random()
                < liquidity_shock_probability
                else 0.0
            )

            liquidity_target = (
                0.70
                * liquidity[t - 1]
                + 0.20
                * np.clip(
                    standardized_abs_return / 3.0,
                    0.0,
                    1.0,
                )
                + 0.10
                * state
                / max(n_states - 1, 1)
            )

            liquidity[t] = float(
                np.clip(
                    liquidity_target
                    + 0.35 * stress_shock
                    + 0.025 * rng.normal(),
                    0.0,
                    1.0,
                )
            )

        else:

            liquidity[t] = float(
                np.clip(
                    0.05
                    + 0.02 * rng.normal(),
                    0.0,
                    1.0,
                )
            )

        # ====================================================
        # 2. Stochastic information intensity
        #
        # Even inside the same hidden state, information
        # visibility changes through time.
        # ====================================================

        intensity_jitter = np.exp(
            0.35
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

        # Information becomes harder to observe during stress.
        effective_intensity *= (
            1.0
            - 0.20 * liquidity[t]
        )

        effective_intensity = np.clip(
            effective_intensity,
            0.03,
            2.50,
        )

        # ====================================================
        # 3. Persistent latent role factors
        # ====================================================

        for role in range(5):

            innovation = rng.normal()

            if t == 0:

                factor = innovation

            else:

                previous_factor = (
                    role_factors[
                        role,
                        t - 1,
                    ]
                )

                factor = (
                    role_phi[role]
                    * previous_factor
                    + np.sqrt(
                        1.0
                        - role_phi[role] ** 2
                    )
                    * innovation
                )

            factor += (
                regime.role_bias[role]
                * effective_intensity[role]
            )

            role_factors[
                role,
                t,
            ] = factor

        # ====================================================
        # 4. Structural loading drift
        #
        # Loadings move slowly even without an explicit break.
        # ====================================================

        for j in range(n_features):

            loading_noise = rng.normal()

            loading[j] = (
                0.995
                * loading[j]
                + 0.005
                * base_loading[j]
                * np.exp(
                    0.12
                    * loading_noise
                )
            )

            loading[j] = float(
                np.clip(
                    loading[j],
                    0.10,
                    2.20,
                )
            )

        # ====================================================
        # 5. Rare structural breaks
        #
        # A small subset of features can suddenly change
        # information content, then slowly recover.
        # ====================================================

        structural_break_probability = (
            0.0008
            + 0.0025 * liquidity[t]
        )

        if (
            rng.random()
            < structural_break_probability
        ):

            break_mask = (
                rng.random(
                    n_features
                )
                < 0.15
            )

            number_broken = (
                int(
                    break_mask.sum()
                )
            )

            if number_broken > 0:

                structural_break_multiplier[
                    break_mask
                ] *= np.exp(
                    rng.normal(
                        0.0,
                        0.30,
                        size=number_broken,
                    )
                )

        # Slow recovery toward normal structure.
        structural_break_multiplier = (
            0.9985
            * structural_break_multiplier
            + 0.0015
        )

        # ====================================================
        # 6. Temporary feature dropout
        #
        # Stress increases the probability that a feature
        # becomes temporarily uninformative.
        # ====================================================

        for j in range(n_features):

            if feature_dropout[j]:

                recovery_probability = (
                    0.10
                    + 0.08
                    * (1.0 - liquidity[t])
                )

                if (
                    rng.random()
                    < recovery_probability
                ):
                    feature_dropout[j] = False

            else:

                dropout_probability_t = (
                    dropout_probability[j]
                    * (
                        1.0
                        + 2.5 * liquidity[t]
                    )
                )

                if (
                    rng.random()
                    < dropout_probability_t
                ):
                    feature_dropout[j] = True

        # ====================================================
        # 7. Price information available before return[t]
        #
        # These values are all known before current return[t]
        # is generated.
        # ====================================================

        if t > 0:

            short_price_move = (
                returns[t - 1]
            )

            medium_start = max(
                0,
                t - 20,
            )

            medium_price_move = (
                log_price[t - 1]
                - log_price[medium_start]
            )

            price_reference_gap = (
                log_price[t - 1]
                - reference[t - 1]
            )

        else:

            short_price_move = 0.0
            medium_price_move = 0.0
            price_reference_gap = 0.0

        # ====================================================
        # 8. Generate observed features
        #
        # Features contain:
        #
        #   own latent role
        #   cross-role contamination
        #   drifting loading
        #   structural breaks
        #   stochastic information intensity
        #   price feedback
        #   liquidity distortion
        #   observation noise
        #   temporary dropout
        #   feature-specific persistence
        # ====================================================

        for j in range(n_features):

            role = int(
                roles[j]
            )

            own_signal = (
                loading[j]
                * structural_break_multiplier[j]
                * effective_intensity[role]
                * role_factors[
                    role,
                    t,
                ]
            )

            # -----------------------------------------------
            # Cross-role contamination.
            # -----------------------------------------------

            cross_role_1 = (
                role_factors[
                    (role + 1) % 5,
                    t,
                ]
            )

            cross_role_2 = (
                role_factors[
                    (role + 2) % 5,
                    t,
                ]
            )

            contamination = (
                (
                    0.12
                    + 0.08 * liquidity[t]
                )
                * cross_role_1
                + 0.06 * cross_role_2
            )

            # -----------------------------------------------
            # Price -> feature feedback.
            #
            # Only prior price information is used.
            # -----------------------------------------------

            price_feedback = (
                0.08
                * short_price_move
                + 0.04
                * medium_price_move
                + 0.04
                * np.tanh(
                    price_reference_gap
                    / 0.02
                )
            )

            # -----------------------------------------------
            # Stress distorts observed features.
            # -----------------------------------------------

            stress_distortion = (
                liquidity[t]
                * rng.normal(
                    0.0,
                    0.10
                    + 0.05
                    * abs(role - 2),
                )
            )

            # -----------------------------------------------
            # Temporary information dropout.
            # -----------------------------------------------

            if feature_dropout[j]:

                own_signal *= 0.10

            # -----------------------------------------------
            # Observation noise increases under stress.
            # -----------------------------------------------

            observation_noise = (
                feature_noise
                * (
                    1.0
                    + 0.45 * liquidity[t]
                )
                * rng.normal()
            )

            raw_value = (
                own_signal
                + contamination
                + price_feedback
                + stress_distortion
                + observation_noise
            )

            # -----------------------------------------------
            # Feature-specific persistence.
            # -----------------------------------------------

            if t == 0:

                features[
                    t,
                    j,
                ] = raw_value

            else:

                phi = (
                    feature_phi[j]
                )

                features[
                    t,
                    j,
                ] = (
                    phi
                    * features[
                        t - 1,
                        j,
                    ]
                    + (
                        1.0
                        - phi
                    )
                    * raw_value
                )

        # ====================================================
        # 9. Current feature information
        #
        # NOT used for return[t].
        #
        # These signals become the information available for
        # return[t+1].
        # ====================================================

        directional_signal = 0.0
        volatility_signal = 0.0
        mean_reversion_signal = 0.0
        momentum_signal = 0.0

        for j in range(n_features):

            role = int(
                roles[j]
            )

            value = (
                features[
                    t,
                    j,
                ]
            )

            effective_loading = (
                loading[j]
                * structural_break_multiplier[j]
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
        # 10. Generate return[t] from t-1 information
        # ====================================================

        if t > 0:

            previous_state = int(
                states[t - 1]
            )

            previous_regime = regimes[
                previous_state
            ]

            previous_sigma = max(
                realized_volatility[t - 1],
                1e-8,
            )

            # =================================================
            # Persistent stochastic volatility
            #
            # Fast component reacts quickly to shocks.
            # Slow component moves much more slowly.
            # =================================================

            target_log_sigma = np.log(
                previous_regime.volatility
            )

            standardized_abs_return = (
                abs(
                    returns[t - 1]
                )
                / previous_sigma
            )

            # -----------------------------------------------
            # Fast volatility process.
            # -----------------------------------------------

            fast_log_sigma = (
                0.92
                * fast_log_sigma
                + 0.08
                * target_log_sigma
                + 0.10
                * (
                    np.clip(
                        standardized_abs_return,
                        0.0,
                        6.0,
                    )
                    - 0.80
                )
                + 0.035
                * rng.normal()
            )

            # -----------------------------------------------
            # Slow volatility process.
            # -----------------------------------------------

            slow_log_sigma = (
                0.995
                * slow_log_sigma
                + 0.005
                * target_log_sigma
                + 0.008
                * rng.normal()
            )

            # -----------------------------------------------
            # Liquidity increases volatility.
            # -----------------------------------------------

            liquidity_sigma_multiplier = np.exp(
                0.25
                * liquidity[t]
            )

            # -----------------------------------------------
            # The lagged volatility feature also contributes.
            # -----------------------------------------------

            feature_sigma_multiplier = np.exp(
                0.08
                * np.clip(
                    previous_volatility_signal,
                    -5.0,
                    5.0,
                )
            )

            blended_log_sigma = (
                0.70
                * fast_log_sigma
                + 0.30
                * slow_log_sigma
            )

            sigma = (
                np.exp(
                    blended_log_sigma
                )
                * liquidity_sigma_multiplier
                * feature_sigma_multiplier
            )

            sigma = float(
                np.clip(
                    sigma,
                    0.001,
                    0.080,
                )
            )

            realized_volatility[t] = (
                sigma
            )

            # =================================================
            # Slow price reference
            # =================================================

            reference[t] = (
                (
                    1.0
                    - reference_alpha
                )
                * reference[t - 1]
                + reference_alpha
                * log_price[t - 1]
            )

            deviation = (
                reference[t]
                - log_price[t - 1]
            )

            # =================================================
            # Nonlinear mean reversion
            #
            # The response saturates for extreme deviations.
            # =================================================

            mean_reversion_strength = (
                previous_regime.mean_reversion
                * np.exp(
                    0.08
                    * np.clip(
                        previous_mean_reversion_signal,
                        -5.0,
                        5.0,
                    )
                )
            )

            mean_reversion_strength *= np.exp(
                0.06
                * rng.normal()
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
                * 0.02
                * np.tanh(
                    deviation
                    / 0.02
                )
            )

            # =================================================
            # Direction
            # =================================================

            directional_component = (
                0.00018
                * previous_directional_signal
            )

            # =================================================
            # Momentum
            # =================================================

            momentum_component = (
                0.00075
                * previous_regime.momentum
                * previous_momentum_signal
            )

            # =================================================
            # Heavy-tailed continuous shock
            # =================================================

            shock = rng.standard_t(
                df=6.0
            )

            shock /= np.sqrt(
                6.0
                / (6.0 - 2.0)
            )

            # =================================================
            # Explicit jump process
            #
            # Jumps become more likely during stress.
            # =================================================

            jump_probability = (
                previous_regime.jump_probability
                + 0.006
                * liquidity[t]
                + 0.002
                * min(
                    standardized_abs_return,
                    3.0,
                )
            )

            jump_probability = float(
                np.clip(
                    jump_probability,
                    0.0,
                    0.08,
                )
            )

            if (
                rng.random()
                < jump_probability
            ):

                jump_shock = (
                    rng.standard_t(
                        df=4.0
                    )
                )

                jump_shock /= np.sqrt(
                    4.0
                    / (4.0 - 2.0)
                )

                jump_scale = (
                    0.015
                    + 0.025
                    * liquidity[t]
                )

                jump_component = (
                    jump_scale
                    * jump_shock
                )

            else:

                jump_component = 0.0

            # =================================================
            # Final return
            # =================================================

            log_return = (
                previous_regime.drift
                + directional_component
                + momentum_component
                + mean_reversion_component
                - 0.5
                * sigma**2
                + sigma
                * shock
                + jump_component
            )

            # Numerical stability guard.
            log_return = float(
                np.clip(
                    log_return,
                    -0.20,
                    0.20,
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

            # No lagged information exists at t=0.
            realized_volatility[t] = (
                regime.volatility
            )

            reference[t] = (
                log_price[t]
            )

            returns[t] = 0.0

        # ====================================================
        # 11. Current signals become lagged signals for t+1.
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
    #
    # Keep the schema identical to Synth1 / Synth2.
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