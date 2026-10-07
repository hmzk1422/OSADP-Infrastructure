"""Time-based mapping from raw observations to VDC reset intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _interval_duration(h_interval) -> pd.Timedelta:
    """Normalize h_interval; bare numeric values are seconds."""
    if isinstance(h_interval, (int, float, np.integer, np.floating)):
        if not np.isfinite(h_interval) or h_interval <= 0:
            raise ValueError("h_interval must be a positive duration")
        duration = pd.to_timedelta(float(h_interval), unit="s")
    else:
        try:
            duration = pd.Timedelta(h_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("h_interval must be seconds or a timedelta value") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("h_interval must be a positive duration")
    return duration


def assign_time_intervals(
    timestamps: pd.DatetimeIndex,
    h_interval=60,
    sessionize: bool = True,
) -> pd.DataFrame:
    """Map each raw timestamp to a time bin and mark endpoint observations.

    Numeric ``h_interval`` values are seconds. When ``sessionize`` is true,
    bins are anchored to UTC midnight and fund state resets at each UTC day;
    otherwise bins are anchored to the Unix epoch and state can span days.
    The interval endpoint is the last available observation in each bin, so
    missing bars do not create synthetic observations.
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError("timestamps must be a DatetimeIndex")
    if timestamps.hasnans:
        raise ValueError("timestamps cannot contain NaT")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in increasing order")
    duration = _interval_duration(h_interval)
    utc = timestamps
    if utc.tz is None:
        utc = utc.tz_localize("UTC")
    else:
        utc = utc.tz_convert("UTC")

    if sessionize:
        session_key = utc.normalize()
        elapsed = utc - session_key
        # Use Timedelta division so the interval calculation is independent of
        # the DatetimeIndex storage resolution (us, ns, etc.). ``asi8`` and
        # ``Timedelta.value`` need not use the same unit on newer pandas.
        bin_number = np.asarray(elapsed // duration, dtype=np.int64)
    else:
        session_key = pd.DatetimeIndex([pd.Timestamp("1970-01-01", tz="UTC")] * len(utc))
        elapsed = utc - pd.Timestamp("1970-01-01", tz="UTC")
        bin_number = np.asarray(elapsed // duration, dtype=np.int64)

    key = pd.MultiIndex.from_arrays([session_key.asi8, bin_number])
    interval_code, _ = pd.factorize(key, sort=False)
    session_code, _ = pd.factorize(session_key, sort=False)
    n = len(utc)
    if n == 0:
        return pd.DataFrame(index=timestamps, columns=[
            "session_id", "interval_id", "interval_start", "interval_end",
            "interval_start_observation", "interval_end", "session_start", "session_end",
        ])

    group_first = np.r_[True, interval_code[1:] != interval_code[:-1]]
    group_last = np.r_[interval_code[:-1] != interval_code[1:], True]
    session_first = np.r_[True, session_code[1:] != session_code[:-1]]
    session_last = np.r_[session_code[:-1] != session_code[1:], True]
    starts = np.empty(n, dtype=object)
    ends = np.empty(n, dtype=object)
    for i in range(n):
        if group_first[i]:
            group_start = utc[i]
        starts[i] = group_start
        if group_last[i]:
            group_end = utc[i]
            ends[i] = group_end
    # Fill each bin's known endpoint backward so all of its raw rows share
    # the same interval_end timestamp.
    for i in range(n - 2, -1, -1):
        if interval_code[i] == interval_code[i + 1]:
            ends[i] = ends[i + 1]
    return pd.DataFrame(
        {
            "session_id": session_code,
            "interval_id": interval_code,
            "interval_start": pd.to_datetime(starts, utc=True),
            "interval_end": pd.to_datetime(ends, utc=True),
            "interval_start_observation": group_first,
            "fund_interval_boundary": group_last,
            "session_start": session_first,
            "session_end": session_last,
        },
        index=timestamps,
    )
