import pandas as pd
import numpy as np
import pyarrow.parquet as pq

INPUT = "00.parquet"

SUMMARY_OUT = "23_check_summary.csv"
NULLS_OUT = "23_check_nulls.csv"
EVENT_TYPES_OUT = "23_check_event_types.csv"


df = pd.read_parquet(INPUT)


# ============================================================
# TIMESTAMP
# ============================================================

if "event_time" in df.columns:
    time_col = "event_time"
elif "timestamp" in df.columns:
    time_col = "timestamp"
else:
    raise ValueError(
        "No event_time or timestamp column found."
    )

times = (
    df[time_col]
    .dropna()
    .drop_duplicates()
    .sort_values()
)

gaps = times.diff().dropna()

# Detect timestamp unit
if len(times) > 0:
    max_time = abs(times.iloc[-1])

    if max_time > 1e17:
        divisor = 1e9
        unit = "ns"
    elif max_time > 1e14:
        divisor = 1e6
        unit = "us"
    elif max_time > 1e11:
        divisor = 1e3
        unit = "ms"
    else:
        divisor = 1
        unit = "s"
else:
    divisor = 1
    unit = "unknown"


# ============================================================
# SUMMARY
# ============================================================

summary = []


def add(category, metric, value, unit_value=""):
    summary.append({
        "category": category,
        "metric": metric,
        "value": value,
        "unit": unit_value,
    })


add("file", "rows", len(df))
add("file", "columns", len(df.columns))

add(
    "time",
    "timestamp_column",
    time_col,
)

add(
    "time",
    "timestamp_unit",
    unit,
)

add(
    "time",
    "unique_timestamps",
    len(times),
)

if len(times) > 1:
    add(
        "time",
        "minimum_gap",
        gaps.min(),
        unit,
    )

    add(
        "time",
        "1_percent_gap",
        gaps.quantile(0.01),
        unit,
    )

    add(
        "time",
        "5_percent_gap",
        gaps.quantile(0.05),
        unit,
    )

    add(
        "time",
        "10_percent_gap",
        gaps.quantile(0.10),
        unit,
    )

    add(
        "time",
        "25_percent_gap",
        gaps.quantile(0.25),
        unit,
    )

    add(
        "time",
        "median_gap",
        gaps.quantile(0.50),
        unit,
    )

    add(
        "time",
        "75_percent_gap",
        gaps.quantile(0.75),
        unit,
    )

    add(
        "time",
        "90_percent_gap",
        gaps.quantile(0.90),
        unit,
    )

    add(
        "time",
        "95_percent_gap",
        gaps.quantile(0.95),
        unit,
    )

    add(
        "time",
        "99_percent_gap",
        gaps.quantile(0.99),
        unit,
    )

    add(
        "time",
        "maximum_gap",
        gaps.max(),
        unit,
    )

    add(
        "time",
        "average_gap",
        gaps.mean(),
        unit,
    )

    add(
        "time",
        "average_gap_seconds",
        gaps.mean() / divisor,
        "seconds",
    )

    add(
        "time",
        "median_gap_seconds",
        gaps.quantile(0.50) / divisor,
        "seconds",
    )

    add(
        "time",
        "1_percent_gap_seconds",
        gaps.quantile(0.01) / divisor,
        "seconds",
    )

    add(
        "time",
        "5_percent_gap_seconds",
        gaps.quantile(0.05) / divisor,
        "seconds",
    )

    add(
        "time",
        "95_percent_gap_seconds",
        gaps.quantile(0.95) / divisor,
        "seconds",
    )

    add(
        "time",
        "99_percent_gap_seconds",
        gaps.quantile(0.99) / divisor,
        "seconds",
    )

    add(
        "time",
        "maximum_gap_seconds",
        gaps.max() / divisor,
        "seconds",
    )

    add(
        "time",
        "total_time_span_seconds",
        (
            times.iloc[-1] - times.iloc[0]
        ) / divisor,
        "seconds",
    )

    add(
        "time",
        "gaps_gt_2x_median",
        int(
            (gaps > gaps.median() * 2).sum()
        ),
    )

    add(
        "time",
        "gaps_gt_5x_median",
        int(
            (gaps > gaps.median() * 5).sum()
        ),
    )

    add(
        "time",
        "gaps_gt_10x_median",
        int(
            (gaps > gaps.median() * 10).sum()
        ),
    )


# ============================================================
# NULL / NaN
# ============================================================

null_counts = df.isna().sum()

for column in df.columns:
    count = int(null_counts[column])

    percentage = (
        count / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    add(
        "nulls",
        f"{column}_null_count",
        count,
    )

    add(
        "nulls",
        f"{column}_null_percent",
        percentage,
        "percent",
    )


add(
    "nulls",
    "total_null_cells",
    int(null_counts.sum()),
)


# ============================================================
# INFINITY
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns

total_infinity = 0

for column in numeric_cols:
    values = df[column].to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    count = int(
        np.isinf(values).sum()
    )

    total_infinity += count

    if count:
        add(
            "infinity",
            f"{column}_infinity_count",
            count,
        )

add(
    "infinity",
    "total_infinity_cells",
    total_infinity,
)


# ============================================================
# DUPLICATES
# ============================================================

duplicate_rows = int(
    df.duplicated().sum()
)

add(
    "duplicates",
    "completely_duplicate_rows",
    duplicate_rows,
)


# ============================================================
# EVENT TYPES
# ============================================================

if "event_type" in df.columns:
    event_counts = (
        df["event_type"]
        .value_counts(dropna=False)
        .rename_axis("event_type")
        .reset_index(name="row_count")
    )

    event_counts["percent"] = (
        event_counts["row_count"]
        / len(df)
        * 100
    )

    event_counts.to_csv(
        EVENT_TYPES_OUT,
        index=False,
    )


# ============================================================
# WRITE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)

# Separate, easy-to-read column null report
null_report = pd.DataFrame({
    "column": df.columns,
    "null_count": [
        int(null_counts[col])
        for col in df.columns
    ],
})

null_report["null_percent"] = (
    null_report["null_count"]
    / len(df)
    * 100
)

null_report.to_csv(
    NULLS_OUT,
    index=False,
)


print("Done.")
print(f"Summary:      {SUMMARY_OUT}")
print(f"Nulls:        {NULLS_OUT}")

if "event_type" in df.columns:
    print(f"Event types:  {EVENT_TYPES_OUT}")