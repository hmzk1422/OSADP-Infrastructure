import pandas as pd
import numpy as np

def test_nulls(
    df_1,
    df_2,
    joined=None,
    *,
    macro_feat_col=None,
    micro_feat_col=None,
    causal_col="agg_trade_id",
    htf_prefix="htf_",
    name_1="macro",
    name_2="micro",
):
    """
    Diagnose missing values across reconstruction and joining.

    Checks:
        1. Row counts.
        2. agg_trade_id existence and NaN rate.
        3. agg_trade_id range.
        4. Joined HTF NaN rate.
        5. Joined micro-feature NaN rate.
        6. Per-column NaN counts and rates.
    """

    print("=" * 70)
    print("DATASET SIZE")
    print("=" * 70)

    print(
        f"{name_1} rows:",
        len(df_1),
    )

    print(
        f"{name_2} rows:",
        len(df_2),
    )

    if joined is not None:
        print(
            "joined rows:",
            len(joined),
        )

    # ========================================================
    # CAUSAL KEY CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print(f"CAUSAL KEY: {causal_col}")
    print("=" * 70)

    for frame, name in [
        (df_1, name_1),
        (df_2, name_2),
    ]:

        if causal_col not in frame.columns:

            print(
                f"{name}: {causal_col} NOT FOUND"
            )

            continue

        series = pd.to_numeric(
            frame[causal_col],
            errors="coerce",
        )

        print(
            f"{name}: "
            f"exists=True, "
            f"NaN rate={series.isna().mean():.4%}"
        )

        if series.notna().any():

            print(
                f"{name}: "
                f"min={series.min()}, "
                f"max={series.max()}"
            )

    # ========================================================
    # JOIN DIAGNOSTICS
    # ========================================================

    if joined is not None:

        print("\n" + "=" * 70)
        print("JOINED DATA")
        print("=" * 70)

        # ----------------------------------------------------
        # HTF columns
        # ----------------------------------------------------

        htf_cols = [
            column
            for column in joined.columns
            if column.startswith(
                htf_prefix
            )
        ]

        if htf_cols:

            htf_nan_rate = (
                joined[htf_cols]
                .isna()
                .mean()
                .mean()
            )

            print(
                f"HTF NaN rate: "
                f"{htf_nan_rate:.4%}"
            )

        else:

            print(
                "No HTF columns found."
            )

        # ----------------------------------------------------
        # Micro columns
        # ----------------------------------------------------

        if micro_feat_col:

            existing_micro = [
                column
                for column in micro_feat_col
                if column in joined.columns
            ]

            if existing_micro:

                micro_nan_rate = (
                    joined[existing_micro]
                    .isna()
                    .mean()
                    .mean()
                )

                print(
                    f"Micro NaN rate: "
                    f"{micro_nan_rate:.4%}"
                )

            else:

                print(
                    "No micro feature columns found."
                )

        # ====================================================
        # PER-COLUMN NA DIAGNOSTIC
        # ====================================================

        print("\n" + "=" * 70)
        print("COLUMNS WITH NaN")
        print("=" * 70)

        nan_stats = (
            joined
            .isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        nan_stats = nan_stats[
            nan_stats > 0
        ]

        if nan_stats.empty:

            print(
                "No NaN values found."
            )

        else:

            total_rows = len(joined)

            for column, count in nan_stats.items():

                rate = (
                    count
                    / total_rows
                )

                print(
                    f"{column:<40} "
                    f"{count:>10} "
                    f"({rate:.4%})"
                )

        # ====================================================
        # FEATURE-ONLY DIAGNOSTIC
        # ====================================================

        if (
            macro_feat_col
            or micro_feat_col
        ):

            feature_cols = []

            if macro_feat_col:
                feature_cols.extend(
                    [
                        f"{htf_prefix}{column}"
                        for column in macro_feat_col
                    ]
                )

            if micro_feat_col:
                feature_cols.extend(
                    micro_feat_col
                )

            feature_cols = [
                column
                for column in feature_cols
                if column in joined.columns
            ]

            if feature_cols:

                print("\n" + "=" * 70)
                print("FEATURE NaN DIAGNOSTIC")
                print("=" * 70)

                feature_nan = (
                    joined[feature_cols]
                    .isna()
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                feature_nan = feature_nan[
                    feature_nan > 0
                ]

                if feature_nan.empty:

                    print(
                        "No NaN values in "
                        "research features."
                    )

                else:

                    total_rows = len(joined)

                    for column, count in feature_nan.items():

                        rate = (
                            count
                            / total_rows
                        )

                        print(
                            f"{column:<40} "
                            f"{count:>10} "
                            f"({rate:.4%})"
                        )

    print("\n" + "=" * 70)
    print("END NULL DIAGNOSTIC")
    print("=" * 70)