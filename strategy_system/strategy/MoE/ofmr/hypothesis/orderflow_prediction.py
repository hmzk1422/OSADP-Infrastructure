from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import rankdata


HORIZONS = [
    ("100ms", 1),
    ("200ms", 2),
    ("300ms", 3),
    ("400ms", 4),
    ("500ms", 5),
    ("1s", 10),
    ("2s", 20),
    ("3s", 30),
    ("4s", 40),
    ("5s", 50),
    ("10s", 100),
    ("20s", 200),
    ("30s", 300),
    ("1m", 600),
    ("2m", 1200),
    ("3m", 1800),
    ("4m", 2400),
    ("5m", 3000),
]


# ============================================================
# Helpers
# ============================================================

def _prefix(a):
    """Prefix sum with a leading 0."""
    out = np.empty(a.size + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(
        a,
        dtype=np.float64,
        out=out[1:],
    )
    return out


def _rolling_spearman(
    x,
    y,
    m,
    s,
    e,
    min_n,
    block_size=32,
):
    """
    Exact rolling Spearman correlation.

    Same definition as the original implementation:

        for each window:
            mask = joint-valid observations
            rank x within that window
            rank y within that window
            Pearson correlation of those local ranks

    The optimization is that multiple windows are ranked together
    along axis=1 instead of running rankdata() in a Python loop
    for every individual window.

    block_size controls memory usage.
    """

    n_windows = e.size
    out = np.full(
        n_windows,
        np.nan,
        dtype=np.float64,
    )

    n_out = np.zeros(
        n_windows,
        dtype=np.int64,
    )

    if n_windows == 0:
        return out, n_out

    window = int(e[0] - s[0])

    if window <= 0:
        return out, n_out

    if n_windows > 1:
        step = int(s[1] - s[0])
    else:
        step = 1

    # Sliding-window views.
    # These are views until the selected block is transformed.
    x_windows = np.lib.stride_tricks.sliding_window_view(
        x,
        window,
    )[::step]

    y_windows = np.lib.stride_tricks.sliding_window_view(
        y,
        window,
    )[::step]

    m_windows = np.lib.stride_tricks.sliding_window_view(
        m,
        window,
    )[::step]

    for b0 in range(
        0,
        n_windows,
        block_size,
    ):
        b1 = min(
            b0 + block_size,
            n_windows,
        )

        xb = x_windows[b0:b1]
        yb = y_windows[b0:b1]
        mb = m_windows[b0:b1]

        # Number of jointly valid observations in each window.
        nb = mb.sum(axis=1).astype(
            np.int64,
            copy=False,
        )

        n_out[b0:b1] = nb

        valid_rows = nb >= max(
            2,
            min_n,
        )

        if not np.any(valid_rows):
            continue

        # ----------------------------------------------------
        # Fast path:
        # if the whole block is valid, no NaN masking needed.
        # ----------------------------------------------------
        if mb.all():

            rx = rankdata(
                xb,
                axis=1,
                method="average",
            ).astype(
                np.float64,
                copy=False,
            )

            ry = rankdata(
                yb,
                axis=1,
                method="average",
            ).astype(
                np.float64,
                copy=False,
            )

        else:

            # ------------------------------------------------
            # Exact joint-valid mask.
            #
            # This is important: ranks must be based on the
            # observations valid for BOTH x and y.
            # ------------------------------------------------
            xmasked = np.where(
                mb,
                xb,
                np.nan,
            )

            ymasked = np.where(
                mb,
                yb,
                np.nan,
            )

            rx = rankdata(
                xmasked,
                axis=1,
                method="average",
                nan_policy="omit",
            ).astype(
                np.float64,
                copy=False,
            )

            ry = rankdata(
                ymasked,
                axis=1,
                method="average",
                nan_policy="omit",
            ).astype(
                np.float64,
                copy=False,
            )

        # ----------------------------------------------------
        # Pearson correlation of LOCAL ranks.
        #
        # Invalid entries contribute zero.
        # ----------------------------------------------------
        valid_rank = mb

        rx0 = np.where(
            valid_rank,
            rx,
            0.0,
        )

        ry0 = np.where(
            valid_rank,
            ry,
            0.0,
        )

        n = nb.astype(
            np.float64,
        )

        sx = rx0.sum(axis=1)
        sy = ry0.sum(axis=1)

        sxx = np.einsum(
            "ij,ij->i",
            rx0,
            rx0,
        )

        syy = np.einsum(
            "ij,ij->i",
            ry0,
            ry0,
        )

        sxy = np.einsum(
            "ij,ij->i",
            rx0,
            ry0,
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            cov = (
                sxy / n
                - (sx * sy) / n**2
            )

            vx = (
                sxx / n
                - (sx / n) ** 2
            )

            vy = (
                syy / n
                - (sy / n) ** 2
            )

            corr = (
                cov
                / np.sqrt(vx * vy)
            )

        # Only write windows satisfying min_n.
        target = np.flatnonzero(
            valid_rows,
        )

        out[
            b0 + target
        ] = corr[target]

    return out, n_out


def _dir_stats(
    sig,
    h,
    s,
    e,
):
    """
    Hit rate + directional expectancy (bps)
    for a signed signal.
    """

    act = (
        (sig != 0)
        & (h["sgn"] != 0)
    )

    hit = (
        act
        & (sig == h["sgn"])
    )

    Pa = _prefix(
        act.astype(np.float64)
    )

    Ph = _prefix(
        hit.astype(np.float64)
    )

    Pd = _prefix(
        h["f0"] * sig
    )

    na = (
        Pa[e] - Pa[s]
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        hit_rate = (
            (Ph[e] - Ph[s])
            / na
        )

        expectancy = (
            (Pd[e] - Pd[s])
            / na
            * 1e4
        )

    return (
        hit_rate,
        expectancy,
    )


def _forward_return(
    price,
    k,
):
    """Forward k-sample return."""

    out = np.full(
        price.size,
        np.nan,
        dtype=np.float64,
    )

    if k < price.size:
        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            out[:-k] = (
                price[k:]
                / price[:-k]
                - 1.0
            )

    return out


# ============================================================
# Rolling study
# ============================================================

def rolling_orderflow_study(
    df,
    feat_col,
    price_col="mid_price",
    rolling_window=36000,
    step=600,
    horizons=HORIZONS,
    min_valid_frac=0.8,
    spearman_block_size=32,
):
    """
    Long-format rolling results.

    One row per:

        feature × horizon × rolling window

    This retains the original rolling methodology.
    """

    price = df[
        price_col
    ].to_numpy(
        dtype=np.float64,
    )

    N = price.size

    if N < rolling_window:
        return pd.DataFrame()

    e = np.arange(
        rolling_window,
        N + 1,
        step,
    )

    s = e - rolling_window

    w_end = df.index[
        e - 1
    ]

    min_n = int(
        np.ceil(
            min_valid_frac
            * rolling_window
        )
    )

    # --------------------------------------------------------
    # Future returns.
    # --------------------------------------------------------

    H = {}

    for name, k in horizons:

        fut = _forward_return(
            price,
            k,
        )

        ok = np.isfinite(
            fut
        )

        f0 = np.where(
            ok,
            fut,
            0.0,
        )

        Pa = _prefix(
            np.abs(f0)
            * 1e4
        )

        Pk = _prefix(
            ok.astype(np.float64)
        )

        H[name] = {
            "f0": f0,
            "ok": ok,
            "sgn": np.sign(f0),
            "abs_bps": (
                Pa[e] - Pa[s]
            )
            / np.maximum(
                Pk[e] - Pk[s],
                1,
            ),
        }

    # --------------------------------------------------------
    # Per feature.
    # --------------------------------------------------------

    frames = []

    for col in feat_col:

        x = df[
            col
        ].to_numpy(
            dtype=np.float64,
        )

        dx = np.empty_like(x)

        dx[0] = np.nan

        dx[1:] = np.diff(x)

        x_ok = np.isfinite(
            x
        )

        d_ok = np.isfinite(
            dx
        )

        xs = np.sign(
            np.where(
                x_ok,
                x,
                0.0,
            )
        )

        ds = np.sign(
            np.where(
                d_ok,
                dx,
                0.0,
            )
        )

        for name, _ in horizons:

            h = H[name]

            # ------------------------------------------------
            # Base feature
            # ------------------------------------------------

            base_ok = (
                x_ok
                & h["ok"]
            )

            ic, n = _rolling_spearman(
                x,
                h["f0"],
                base_ok,
                s,
                e,
                min_n,
                block_size=spearman_block_size,
            )

            hr, ex = _dir_stats(
                xs,
                h,
                s,
                e,
            )

            # ------------------------------------------------
            # Feature change
            # ------------------------------------------------

            change_ok = (
                d_ok
                & h["ok"]
            )

            icc, nc = _rolling_spearman(
                dx,
                h["f0"],
                change_ok,
                s,
                e,
                min_n,
                block_size=spearman_block_size,
            )

            chr_, cex = _dir_stats(
                ds,
                h,
                s,
                e,
            )

            # ------------------------------------------------
            # Result frame
            # ------------------------------------------------

            frames.append(
                pd.DataFrame(
                    {
                        "feature": col,
                        "horizon": name,
                        "window_end": w_end,

                        "rank_ic": np.where(
                            n >= min_n,
                            ic,
                            np.nan,
                        ),

                        "rank_ic_change": np.where(
                            nc >= min_n,
                            icc,
                            np.nan,
                        ),

                        "hit_rate": hr,

                        "change_hit_rate": chr_,

                        "directional_expectancy_bps": ex,

                        "change_directional_expectancy_bps": cex,

                        "mean_abs_return_bps": h[
                            "abs_bps"
                        ],

                        "sign_agreement":
                            2 * hr - 1,

                        "change_sign_agreement":
                            2 * chr_ - 1,

                        "n": n,
                    }
                )
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# CVaR
# ============================================================

def _cvar_series(
    x,
    alpha,
    upper=False,
):
    x = x.dropna()

    if x.empty:
        return np.nan

    q = x.quantile(
        alpha
    )

    if upper:
        tail = x[x >= q]
    else:
        tail = x[x <= q]

    if tail.empty:
        return np.nan

    return tail.mean()


# ============================================================
# Historical summary
# ============================================================

def summarize(
    res,
    top_n=5,
):
    """
    Latest window + historical p05/p95/CVaR envelope.

    Same report structure as the original.
    """

    keys = [
        "feature",
        "horizon",
    ]

    metrics = [
        "rank_ic",
        "rank_ic_change",
        "hit_rate",
        "change_hit_rate",
        "directional_expectancy_bps",
        "change_directional_expectancy_bps",
        "mean_abs_return_bps",
    ]

    # --------------------------------------------------------
    # Historical p05 / p95
    # --------------------------------------------------------

    q = (
        res
        .groupby(keys)[metrics]
        .quantile(
            [0.05, 0.95]
        )
        .unstack()
    )

    q.columns = [
        f"{m}_p{round(p * 100):02d}"
        for m, p in q.columns
    ]

    q = q.reset_index()

    # --------------------------------------------------------
    # Historical mean
    # --------------------------------------------------------

    mean_values = (
        res
        .groupby(keys)[metrics]
        .mean()
        .add_suffix("_mean")
        .reset_index()
    )

    # --------------------------------------------------------
    # Worst tail
    # --------------------------------------------------------

    cvar05 = (
        res
        .groupby(keys)[metrics]
        .agg(
            lambda x: _cvar_series(
                x,
                0.05,
                upper=False,
            )
        )
        .add_suffix("_cvar05")
        .reset_index()
    )

    # --------------------------------------------------------
    # Best tail
    # --------------------------------------------------------

    cvar95 = (
        res
        .groupby(keys)[metrics]
        .agg(
            lambda x: _cvar_series(
                x,
                0.95,
                upper=True,
            )
        )
        .add_suffix("_cvar95")
        .reset_index()
    )

    # --------------------------------------------------------
    # Latest rolling window
    # --------------------------------------------------------

    latest = (
        res
        .sort_values(
            "window_end"
        )
        .groupby(keys)
        .tail(1)
        .copy()
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    out = (
        latest
        .merge(
            mean_values,
            on=keys,
            how="left",
        )
        .merge(
            cvar05,
            on=keys,
            how="left",
        )
        .merge(
            cvar95,
            on=keys,
            how="left",
        )
        .merge(
            q,
            on=keys,
            how="left",
        )
        .rename(
            columns={
                "window_end":
                    "latest_window_end",
            }
        )
    )

    # --------------------------------------------------------
    # Rank horizons by strongest current absolute IC.
    # --------------------------------------------------------

    out["_a"] = (
        out[
            [
                "rank_ic",
                "rank_ic_change",
            ]
        ]
        .abs()
        .max(axis=1)
    )

    return (
        out
        .sort_values(
            [
                "feature",
                "_a",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby("feature")
        .head(top_n)
        .drop(
            columns="_a"
        )
        .reset_index(drop=True)
    )


# ============================================================
# Display formatting
# ============================================================

_REPORT_SPEC = [
    (
        "rank_ic",
        "IC",
        "+.3f",
        1,
        "",
    ),
    (
        "rank_ic_change",
        "dIC",
        "+.3f",
        1,
        "",
    ),
    (
        "hit_rate",
        "hit",
        ".1f",
        100,
        "%",
    ),
    (
        "change_hit_rate",
        "dhit",
        ".1f",
        100,
        "%",
    ),
    (
        "directional_expectancy_bps",
        "exp bps",
        "+.2f",
        1,
        "",
    ),
    (
        "change_directional_expectancy_bps",
        "dexp bps",
        "+.2f",
        1,
        "",
    ),
    (
        "mean_abs_return_bps",
        "|ret| bps",
        ".2f",
        1,
        "",
    ),
]


def format_report(
    summary,
    show_latest=False,
):
    """
    Compact report:

        mean [worst-tail – best-tail]

    With show_latest=True:

        latest mean [worst-tail – best-tail]
    """

    summary = pd.DataFrame(
        summary
    )

    out = summary[
        [
            "feature",
            "horizon",
            "latest_window_end",
        ]
    ].copy()

    out = out.rename(
        columns={
            "latest_window_end":
                "window_end",
        }
    )

    for (
        col,
        label,
        fmt,
        k,
        suf,
    ) in _REPORT_SPEC:

        mean_value = (
            summary[
                f"{col}_mean"
            ]
            * k
        )

        lo = (
            summary[
                f"{col}_cvar05"
            ]
            * k
        )

        hi = (
            summary[
                f"{col}_cvar95"
            ]
            * k
        )

        cur = (
            summary[col]
            * k
        )

        def cell(
            current,
            mean,
            low,
            high,
        ):

            if (
                pd.isna(mean)
                or pd.isna(low)
                or pd.isna(high)
            ):
                return "NaN"

            rng = (
                f"{mean:{fmt}}{suf} "
                f"[{low:{fmt}}{suf} – "
                f"{high:{fmt}}{suf}]"
            )

            if (
                show_latest
                and pd.notna(current)
            ):
                return (
                    f"{current:{fmt}}{suf} "
                    f"{rng}"
                )

            return rng

        out[label] = [
            cell(
                current,
                mean,
                low,
                high,
            )
            for (
                current,
                mean,
                low,
                high,
            ) in zip(
                cur,
                mean_value,
                lo,
                hi,
            )
        ]

    return out


# ============================================================
# ReportFrame
# ============================================================

class ReportFrame(pd.DataFrame):
    """
    Full numeric DataFrame whose display is the compact report.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    def to_string(
        self,
        *args,
        **kwargs,
    ):
        return format_report(
            self
        ).to_string(
            *args,
            **kwargs,
        )

    def __repr__(self):
        return self.to_string(
            index=False
        )

    def _repr_html_(self):
        return format_report(
            self
        ).to_html(
            index=False
        )


def print_report(
    summary,
    **kw,
):
    print(
        format_report(
            summary,
            **kw,
        ).to_string(
            index=False
        )
    )


# ============================================================
# Public study function
# ============================================================

def orderflow_study_1(
    df,
    feat_col,
    price_col="mid_price",
    rolling_window=36000,
    step=600,
    top_n=5,
    return_rolling=False,
    report=True,
    spearman_block_size=32,
):
    """
    Drop-in replacement for the original orderflow_study_1.

    The statistical meaning and report format are preserved.

    return_rolling=False:
        ReportFrame

    return_rolling=True:
        (ReportFrame, full rolling DataFrame)

    report=False:
        Full rolling DataFrame
    """

    res = rolling_orderflow_study(
        df,
        list(feat_col),
        price_col=price_col,
        rolling_window=rolling_window,
        step=step,
        spearman_block_size=spearman_block_size,
    )

    if res.empty:
        return (
            (res, res)
            if return_rolling
            else res
        )

    if report:

        out = ReportFrame(
            summarize(
                res,
                top_n=top_n,
            )
        )

        if return_rolling:
            return out, res

        return out

    return res