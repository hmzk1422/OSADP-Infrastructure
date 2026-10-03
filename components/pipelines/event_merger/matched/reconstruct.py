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

        "qty": "sum",
        "quantity": "sum",
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
# MERGED-SCHEMA ADAPTER
# ============================================================

# The merged orderbook namespaced its columns by source
# (``agg__price``, ``book_state__best_bid_price``, ...). Downstream
# transforms predate that layout and still ask for flat names
# (``price``, ``quantity``, ``is_buyer_maker``, ``price_close`` ...).
# This table maps the namespaced names onto the flat ones; it is applied
# automatically by :func:`normalize_merged_frame`.
_LEGACY_COLUMN_ALIASES: dict[str, str] = {
    "agg__price": "price",
    "agg__quantity": "quantity",
    "agg__is_buyer_maker": "is_buyer_maker",
    "agg__agg_trade_id": "agg_trade_id",
    "agg__first_trade_id": "first_trade_id",
    "agg__last_trade_id": "last_trade_id",
    "agg__transact_time": "transact_time",
    "depth__depth": "depth",
    "depth__notional": "notional",
    "depth__percentage": "percentage",
}

# Columns that only duplicate a namespaced counterpart and are not
# needed by the strategy layer.
_REDUNDANT_PREFIXES = (
    "source_agg__",
    "source_depth__",
    "lob__",
    "depth__",
    "depth_state__",
)


def _legacy_book_state_names(columns) -> dict[str, str]:
    """
    Flatten ``book_state__<name>`` to ``<name>`` (and ``best_bid``/``best_ask``
    stay as-is, which is what the transforms expect).
    """
    mapping: dict[str, str] = {}
    for column in columns:
        if column.startswith("book_state__"):
            mapping[column] = column[len("book_state__"):]
    return mapping


def normalize_merged_frame(
    df: pd.DataFrame,
    *,
    drop_redundant: bool = True,
) -> pd.DataFrame:
    """
    Adapt a merged-orderbook frame to the flat schema the strategy layer uses.

    - ``agg__price`` -> ``price``, ``agg__quantity`` -> ``quantity``, ...
    - ``book_state__best_bid_price`` -> ``best_bid_price``, ...
    - ``timestamp`` is converted to ``datetime64[ns]`` (unit auto-detected)
    - redundant ``source_*`` / ``depth_*`` duplicates are dropped

    Existing flat columns are never overwritten, so legacy frames pass
    through untouched.
    """
    if df is None:
        raise TypeError("normalize_merged_frame() got None (did a reader return None?)")

    frame = df.copy()

    if "Unnamed: 0" in frame.columns:
        frame = frame.drop(columns=["Unnamed: 0"])

    if "timestamp" in frame.columns:
        frame["timestamp"] = _convert_timestamp(frame["timestamp"])

    renames = dict(_LEGACY_COLUMN_ALIASES)
    renames.update(_legacy_book_state_names(frame.columns))
    renames = {
        source: target
        for source, target in renames.items()
        if source in frame.columns and target not in frame.columns
    }
    if renames:
        frame = frame.rename(columns=renames)

    if drop_redundant:
        drop = [
            column
            for column in frame.columns
            if any(column.startswith(prefix) for prefix in _REDUNDANT_PREFIXES)
        ]
        if drop:
            frame = frame.drop(columns=drop)

    return frame


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

# Magnitude thresholds used to infer the unit of a numeric epoch
# timestamp. Values are chosen from the float64 mantissa so the
# comparison never needs to touch the data:
#
#     1.7e18 -> 2025-06-30 in nanoseconds
#     1.7e15 -> 2025-06-30 in microseconds
#     1.7e12 -> 2025-06-30 in milliseconds
#     1.7e9  -> 2025-06-30 in seconds
#
# Anything above NANOSECOND_THRESHOLD cannot be milliseconds without
# overflowing the pandas datetime range (it would land in the year
# 55,000,000), so a naive "assume ms" would explode downstream.
_UNIT_THRESHOLDS = (
    (1e17, "ns"),
    (1e14, "us"),
    (1e11, "ms"),
    (None, "s"),
)


def infer_time_unit(series) -> str | None:
    """
    Infer the epoch unit of a numeric timestamp column.

    Returns ``"ns" | "us" | "ms" | "s"``, or ``None`` when the column holds
    no usable numeric value.
    """
    values = pd.to_numeric(series, errors="coerce")
    if values.empty:
        return None

    finite = values[np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))]
    if finite.empty:
        return None

    # The widest magnitude decides: one ns-scale row is enough.
    magnitude = float(np.abs(finite.to_numpy(dtype="float64", na_value=np.nan)).max())

    for threshold, unit in _UNIT_THRESHOLDS:
        if threshold is None or magnitude >= threshold:
            return unit
    return "s"


def _convert_timestamp(series, unit: str | None = None):
    """
    Convert a timestamp column to ``datetime64[ns]``.

    Numeric input is interpreted as an epoch value whose unit is inferred
    from its magnitude unless ``unit`` is given. The merged orderbook stores
    **nanoseconds**, so hard-coding milliseconds (the previous behaviour)
    produced year ~55,000,000 timestamps, which then blew up every
    resample downstream.

    Datetime/string input is parsed normally.
    """
    if pd.api.types.is_numeric_dtype(series):
        resolved = unit or infer_time_unit(series)
        if resolved is None:
            return pd.to_datetime(series, errors="coerce")
        return pd.to_datetime(series, unit=resolved, errors="coerce")

    if isinstance(series, pd.Series) and pd.api.types.is_datetime64_any_dtype(series):
        return series

    return pd.to_datetime(series, errors="coerce")


# ============================================================
# RECONSTRUCT
# ============================================================

def reconstruct(
    df,
    timeframe,
    rules=None,
    time_col="timestamp",
    normalize: bool = True,
    columns=None,
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

    Parameters
    ----------
    normalize:
        Run :func:`normalize_merged_frame` first, so a merged-orderbook
        frame (``agg__price``, ``book_state__*``, nanosecond timestamps) can
        be passed in directly. Set False for a frame that already uses the
        flat legacy schema.
    columns:
        Optional whitelist of columns to aggregate. Everything else is
        dropped before grouping. This matters for merged frames, which carry
        ~128 columns where only a handful are useful, and each numeric column
        adds an OHLC expansion (4 columns per numeric).
    """
    if df is None:
        raise TypeError(
            "reconstruct() got None for 'df'. A reader (read_parquet / "
            "load_day_snapshot / ...) returned None — check the path exists "
            "and the file is a readable parquet."
        )

    if normalize:
        df = normalize_merged_frame(df)

    # --------------------------------------------------------
    # Copy
    # --------------------------------------------------------

    df = df.copy()

    if columns is not None:
        keep = [time_col] + [c for c in columns if c != time_col and c in df.columns]
        df = df[keep]

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
    )

    return result

def check_missing_time(
    df,
    timeframe,
    causal_col="agg_trade_id",
):
    """
    Validate continuity of a reconstructed sequence.

    Timeframe
    ---------
    String:
        Time-based timeframe, e.g. "1m", "5m", "1h".

        Checks for missing timestamps at the expected frequency.

    Integer:
        Tick-based timeframe, e.g. 1, 10, 100.

        There is no fixed time interval between tick bars, so
        timestamp frequency cannot be checked directly.

        When `causal_col` exists, the function checks that the
        reconstructed tick bars advance continuously according
        to the requested tick size.

    Raises
    ------
    TypeError
        Invalid dataframe index.

    ValueError
        Duplicate timestamps, non-monotonic timestamps, or
        missing time/tick observations.
    """

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "DataFrame must have a DatetimeIndex."
        )

    if df.empty:
        return

    timestamps = df.index

    # --------------------------------------------------------
    # Timestamp ordering
    # --------------------------------------------------------

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not monotonically increasing."
        )

    if timestamps.has_duplicates:
        raise ValueError(
            "Duplicate timestamps found."
        )

    # ========================================================
    # TICK TIMEFRAME
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
        # There is no expected timestamp frequency for ticks.
        #
        # Example:
        #
        # tick 1  -> 12:00:00.001
        # tick 2  -> 12:00:00.018
        # tick 3  -> 12:00:01.203
        #
        # All are perfectly valid.
        #
        # Therefore use the causal event ID instead.
        # ----------------------------------------------------

        if causal_col not in df.columns:
            raise ValueError(
                f"Cannot validate tick continuity because "
                f"'{causal_col}' is not present."
            )

        ids = pd.to_numeric(
            df[causal_col],
            errors="coerce",
        )

        if ids.isna().any():
            raise ValueError(
                f"'{causal_col}' contains NaN values."
            )

        if not ids.is_monotonic_increasing:
            raise ValueError(
                f"'{causal_col}' is not monotonically increasing."
            )

        if ids.duplicated().any():
            raise ValueError(
                f"Duplicate '{causal_col}' values found."
            )

        # ----------------------------------------------------
        # A reconstructed N-tick bar should advance by N
        # aggregate trades if the underlying event stream is
        # complete.
        #
        # Example for timeframe=10:
        #
        #     1000
        #     1010
        #     1020
        #     1030
        #
        # A jump:
        #
        #     1000
        #     1020
        #
        # implies missing source events.
        # ----------------------------------------------------

        diff = ids.diff().dropna()

        expected_step = timeframe

        missing = diff[
            diff != expected_step
        ]

        if not missing.empty:

            examples = []

            for index in missing.index[:20]:

                position = df.index.get_loc(
                    index
                )

                previous_id = ids.iloc[
                    position - 1
                ]

                current_id = ids.iloc[
                    position
                ]

                actual_step = (
                    current_id
                    - previous_id
                )

                examples.append(
                    {
                        "timestamp": index,
                        "previous_id": previous_id,
                        "current_id": current_id,
                        "expected_step": expected_step,
                        "actual_step": actual_step,
                    }
                )

            raise ValueError(
                f"Missing or irregular tick sequence: "
                f"{len(missing)} irregular transitions.\n"
                f"Examples:\n{examples}"
            )

        return

    # ========================================================
    # TIMEFRAME
    # ========================================================

    if not isinstance(
        timeframe,
        str,
    ):
        raise TypeError(
            "timeframe must be either an integer tick size "
            "or a string timeframe such as '1m', '5m', '1h'."
        )

    # --------------------------------------------------------
    # Use the same normalization logic as reconstruct().
    # --------------------------------------------------------

    timeframe = _normalize_timeframe(
        timeframe
    )

    duration = pd.to_timedelta(
        timeframe
    )

    expected = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq=timeframe,
    )

    missing = expected.difference(
        timestamps
    )

    if len(missing) > 0:

        raise ValueError(
            f"Missing timestamps for timeframe "
            f"'{timeframe}': "
            f"{len(missing)} missing.\n"
            f"First missing timestamps:\n"
            f"{missing[:20].tolist()}"
        )