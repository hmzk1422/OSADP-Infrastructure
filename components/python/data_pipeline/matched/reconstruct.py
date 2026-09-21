import re

import numpy as np
import pandas as pd


# ============================================================
# RULES
# ============================================================

def rule(selection=1):
    """
    Return predefined reconstruction rules.

    First matching rule wins.

    Special rules:
        bool     -> boolean columns
        id       -> columns containing 'id'
        time     -> columns containing 'time' / 'timestamp'
        numeric  -> remaining numeric columns
        default  -> everything else
    """

    rules = {
        "price": "ohlc",

        "qty": "last",
        "quantity": "last",
        "notional": "last",

        "bool": "mean",
        "id": "last",
        "time": "last",

        "numeric": "ohlc",

        "default": "last",
    }

    if selection == 1:
        return rules

    raise ValueError(
        f"Unknown rule selection: {selection}"
    )


# ============================================================
# TIMEFRAME NORMALIZATION
# ============================================================

def _normalize_timeframe(timeframe):
    """
    Normalize convenient timeframe strings.

    Examples:
        "1m"     -> "1min"
        "5m"     -> "5min"
        "1min"   -> "1min"
        "100ms"  -> "100ms"
        "1s"     -> "1s"
        "1h"     -> "1h"
        "1d"     -> "1D"
    """

    if not isinstance(timeframe, str):
        return timeframe

    timeframe = timeframe.strip().lower()

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*([a-z]+)",
        timeframe,
    )

    if not match:
        raise ValueError(
            f"Invalid timeframe: {timeframe}"
        )

    value, unit = match.groups()

    unit_map = {
        "m": "min",
        "min": "min",
        "mins": "min",
        "minute": "min",
        "minutes": "min",

        "s": "s",
        "sec": "s",
        "secs": "s",
        "second": "s",
        "seconds": "s",

        "ms": "ms",
        "millisecond": "ms",
        "milliseconds": "ms",

        "us": "us",
        "microsecond": "us",
        "microseconds": "us",

        "h": "h",
        "hr": "h",
        "hour": "h",
        "hours": "h",

        "d": "D",
        "day": "D",
        "days": "D",
    }

    if unit not in unit_map:
        raise ValueError(
            f"Unsupported timeframe unit: '{unit}'"
        )

    return f"{value}{unit_map[unit]}"


# ============================================================
# TIMESTAMP CONVERSION
# ============================================================

def _convert_timestamp(series):
    """
    Convert timestamp safely.

    Numeric timestamps are assumed to be milliseconds.
    Datetime/string timestamps are parsed normally.
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(
            series,
            unit="ms",
            errors="coerce",
        )

    return pd.to_datetime(
        series,
        errors="coerce",
    )


# ============================================================
# RECONSTRUCT
# ============================================================

def reconstruct(
    df,
    timeframe,
    rules=None,
    time_col="timestamp",
):
    """
    Reconstruct raw event/trade data into completed
    time bars or completed tick bars.

    Important causal convention
    ---------------------------
    Every output row is timestamped at the earliest moment
    when every value in that row is fully known.

    Time bars:
        [12:00, 12:01) -> timestamp 12:01

    Tick bars:
        ticks 1..10 -> timestamp of tick 10

    Incomplete final bars are removed.
    """

    # --------------------------------------------------------
    # Copy
    # --------------------------------------------------------

    df = df.copy()

    # --------------------------------------------------------
    # Validate timestamp
    # --------------------------------------------------------

    if time_col not in df.columns:
        raise ValueError(
            f"'{time_col}' not found in dataframe."
        )

    # --------------------------------------------------------
    # Rules
    # --------------------------------------------------------

    if rules is None:
        rules = rule()

    valid_aggs = {
        "ohlc",
        "sum",
        "last",
        "first",
        "mean",
        "min",
        "max",
    }

    for key, agg in rules.items():

        if agg not in valid_aggs:
            raise ValueError(
                f"Invalid aggregation '{agg}' "
                f"for rule '{key}'. "
                f"Valid aggregations: "
                f"{sorted(valid_aggs)}"
            )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df[time_col] = _convert_timestamp(
        df[time_col]
    )

    df = df.dropna(
        subset=[time_col]
    )

    # Use agg_trade_id as a tie-breaker when available.
    if (
        "agg_trade_id" in df.columns
        and pd.api.types.is_numeric_dtype(
            df["agg_trade_id"]
        )
    ):
        df = df.sort_values(
            [time_col, "agg_trade_id"],
            kind="mergesort",
        )
    else:
        df = df.sort_values(
            time_col,
            kind="mergesort",
        )

    df = df.reset_index(drop=True)

    if df.empty:
        return df.set_index(time_col)

    # --------------------------------------------------------
    # Determine aggregation for each column
    # --------------------------------------------------------

    def get_aggregation(column):

        name = column.lower()

        for key, agg in rules.items():

            # Boolean
            if key == "bool":

                if pd.api.types.is_bool_dtype(
                    df[column]
                ):
                    return agg

                continue

            # Numeric
            if key == "numeric":

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):
                    return agg

                continue

            # ID
            if key == "id":

                if "id" in name:
                    return agg

                continue

            # Time
            if key == "time":

                if (
                    "time" in name
                    or "timestamp" in name
                ):
                    return agg

                continue

            # Default
            if key == "default":
                continue

            # Normal substring rule
            if key.lower() in name:
                return agg

        return rules.get(
            "default",
            "last",
        )

    aggregations = {
        column: get_aggregation(column)
        for column in df.columns
        if column != time_col
    }

    # ========================================================
    # TICK AGGREGATION
    # ========================================================

    if isinstance(
        timeframe,
        (int, np.integer),
    ):

        if timeframe <= 0:
            raise ValueError(
                "Tick timeframe must be greater than 0."
            )

        # ----------------------------------------------------
        # Sequential tick buckets
        # ----------------------------------------------------

        df["_tick"] = (
            np.arange(len(df))
            // timeframe
        )

        # ----------------------------------------------------
        # Remove incomplete final tick bar
        # ----------------------------------------------------

        group_sizes = (
            df.groupby("_tick", sort=True)
            .size()
        )

        complete_groups = group_sizes[
            group_sizes == timeframe
        ].index

        df = df[
            df["_tick"].isin(complete_groups)
        ].copy()

        if df.empty:
            return pd.DataFrame(
                index=pd.DatetimeIndex(
                    [],
                    name=time_col,
                )
            )

        grouped = df.groupby(
            "_tick",
            sort=True,
        )

        result = {}

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        for column, agg in aggregations.items():

            if agg == "ohlc":

                if not pd.api.types.is_numeric_dtype(
                    df[column]
                ):
                    raise TypeError(
                        f"Cannot apply OHLC aggregation "
                        f"to non-numeric column "
                        f"'{column}'."
                    )

                x = grouped[column].ohlc()

                result[
                    f"{column}_open"
                ] = x["open"]

                result[
                    f"{column}_high"
                ] = x["high"]

                result[
                    f"{column}_low"
                ] = x["low"]

                result[
                    f"{column}_close"
                ] = x["close"]

            else:

                result[column] = (
                    grouped[column]
                    .agg(agg)
                )

        result = pd.DataFrame(result)

        # ----------------------------------------------------
        # Completion timestamp
        # ----------------------------------------------------

        result[time_col] = (
            grouped[time_col]
            .last()
            .to_numpy()
        )

        return (
            result
            .set_index(time_col)
            .sort_index()
        )

    # ========================================================
    # TIME AGGREGATION
    # ========================================================

    timeframe = _normalize_timeframe(
        timeframe
    )

    # Keep original bounds for partial-bar detection
    first_timestamp = df[time_col].min()
    last_timestamp = df[time_col].max()

    df = df.set_index(time_col)

    # --------------------------------------------------------
    # Critical causal convention:
    #
    # [12:00, 12:01) gets label 12:01.
    #
    # Anything happening exactly at 12:01 belongs to
    # the next bar.
    # --------------------------------------------------------

    grouped = df.resample(
        timeframe,
        label="right",
        closed="left",
    )

    result = {}

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    for column, agg in aggregations.items():

        if agg == "ohlc":

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                raise TypeError(
                    f"Cannot apply OHLC aggregation "
                    f"to non-numeric column "
                    f"'{column}'."
                )

            x = grouped[column].ohlc()

            result[
                f"{column}_open"
            ] = x["open"]

            result[
                f"{column}_high"
            ] = x["high"]

            result[
                f"{column}_low"
            ] = x["low"]

            result[
                f"{column}_close"
            ] = x["close"]

        else:

            result[column] = (
                grouped[column]
                .agg(agg)
            )

    result = pd.DataFrame(result)

    # --------------------------------------------------------
    # Remove completely empty bars
    # --------------------------------------------------------

    result = result.dropna(
        how="all"
    )

    if result.empty:
        return result

    # --------------------------------------------------------
    # Remove incomplete first/last bars
    # --------------------------------------------------------

    duration = pd.to_timedelta(
        timeframe
    )

    complete = (
        (result.index - duration >= first_timestamp)
        &
        (result.index <= last_timestamp)
    )

    result = result.loc[complete]

    return result

# ============================================================
# JOIN TIMEFRAMES
# ============================================================

def join_timeframes(
    lower_df,
    higher_df,
    time_col="timestamp",
    prefix="htf_",
    causal_col="agg_trade_id",
):
    """
    Join a completed higher timeframe onto a completed
    lower timeframe without lookahead.

    If agg_trade_id exists in both dataframes, it is used
    as the primary causal ordering key.

    Otherwise timestamp is used.
    """

    # ========================================================
    # PREPARE LOWER
    # ========================================================

    lower = lower_df.copy()

    if isinstance(
        lower.index,
        pd.DatetimeIndex,
    ):
        index_name = lower.index.name
        lower = lower.reset_index()

        if time_col not in lower.columns:
            if index_name is None:
                lower = lower.rename(
                    columns={
                        "index": time_col
                    }
                )
            else:
                lower = lower.rename(
                    columns={
                        index_name: time_col
                    }
                )

    elif time_col not in lower.columns:
        raise ValueError(
            f"'{time_col}' not found in lower dataframe."
        )

    # ========================================================
    # PREPARE HIGHER
    # ========================================================

    higher = higher_df.copy()

    if isinstance(
        higher.index,
        pd.DatetimeIndex,
    ):
        index_name = higher.index.name
        higher = higher.reset_index()

        if time_col not in higher.columns:
            if index_name is None:
                higher = higher.rename(
                    columns={
                        "index": time_col
                    }
                )
            else:
                higher = higher.rename(
                    columns={
                        index_name: time_col
                    }
                )

    elif time_col not in higher.columns:
        raise ValueError(
            f"'{time_col}' not found in higher dataframe."
        )

    # ========================================================
    # TIMESTAMP CLEANUP
    # ========================================================

    lower[time_col] = _convert_timestamp(
        lower[time_col]
    )

    higher[time_col] = _convert_timestamp(
        higher[time_col]
    )

    lower = lower.dropna(
        subset=[time_col]
    )

    higher = higher.dropna(
        subset=[time_col]
    )

    if lower.empty:
        return lower.set_index(time_col)

    if higher.empty:
        return lower.set_index(time_col)

    # ========================================================
    # NORMALIZE CAUSAL KEY
    # ========================================================

    use_causal_col = (
        causal_col is not None
        and causal_col in lower.columns
        and causal_col in higher.columns
    )

    if use_causal_col:

        # ----------------------------------------------------
        # Convert both sides to numeric
        # ----------------------------------------------------

        lower[causal_col] = pd.to_numeric(
            lower[causal_col],
            errors="coerce",
        )

        higher[causal_col] = pd.to_numeric(
            higher[causal_col],
            errors="coerce",
        )

        # Remove rows where the causal key does not exist
        lower = lower.dropna(
            subset=[causal_col]
        )

        higher = higher.dropna(
            subset=[causal_col]
        )

        # ----------------------------------------------------
        # Verify IDs are actually integer-valued
        # ----------------------------------------------------

        lower_key = lower[causal_col]
        higher_key = higher[causal_col]

        if not np.all(
            np.isfinite(
                lower_key.to_numpy()
            )
        ):
            raise ValueError(
                f"Invalid values in lower '{causal_col}'."
            )

        if not np.all(
            np.isfinite(
                higher_key.to_numpy()
            )
        ):
            raise ValueError(
                f"Invalid values in higher '{causal_col}'."
            )

        if not np.all(
            lower_key.to_numpy()
            == lower_key.to_numpy().astype(np.int64)
        ):
            raise ValueError(
                f"Lower '{causal_col}' contains "
                f"non-integer values."
            )

        if not np.all(
            higher_key.to_numpy()
            == higher_key.to_numpy().astype(np.int64)
        ):
            raise ValueError(
                f"Higher '{causal_col}' contains "
                f"non-integer values."
            )

        # ----------------------------------------------------
        # Force identical dtype
        # ----------------------------------------------------

        lower[causal_col] = (
            lower[causal_col]
            .astype(np.int64)
        )

        higher[causal_col] = (
            higher[causal_col]
            .astype(np.int64)
        )

        # ----------------------------------------------------
        # Sort by event order
        # ----------------------------------------------------

        lower = lower.sort_values(
            causal_col,
            kind="mergesort",
        )

        higher = higher.sort_values(
            causal_col,
            kind="mergesort",
        )

        # ----------------------------------------------------
        # Prefix higher timeframe columns
        # ----------------------------------------------------

        higher = higher.rename(
            columns={
                column: f"{prefix}{column}"
                for column in higher.columns
                if column != causal_col
            }
        )

        # ----------------------------------------------------
        # Causal merge
        # ----------------------------------------------------

        result = pd.merge_asof(
            lower,
            higher,
            on=causal_col,
            direction="backward",
        )

    else:

        # ====================================================
        # TIMESTAMP FALLBACK
        # ====================================================

        lower = lower.sort_values(
            time_col,
            kind="mergesort",
        )

        higher = higher.sort_values(
            time_col,
            kind="mergesort",
        )

        higher = higher.rename(
            columns={
                column: f"{prefix}{column}"
                for column in higher.columns
                if column != time_col
            }
        )

        result = pd.merge_asof(
            lower,
            higher,
            on=time_col,
            direction="backward",
        )

    # ========================================================
    # RESTORE TIMESTAMP AS INDEX
    # ========================================================

    result = (
        result
        .set_index(time_col)
        .sort_index()
    ).dropna(axis=0, how="any")

    return result