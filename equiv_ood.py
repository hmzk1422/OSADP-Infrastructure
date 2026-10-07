"""Equivalence test: old materialized-window path vs new view-slice path."""
import numpy as np
import pandas as pd

from strategy_system.strategy.moe.ood.base_regime_encoder import (
    TemporalWindowEncoder,
    build_temporal_windows,
)

rng = np.random.default_rng(0)
n, f = 600, 6
df = pd.DataFrame(
    rng.normal(size=(n, f)),
    columns=[f"x{i}" for i in range(f)],
    index=pd.date_range("2025-01-01", periods=n, freq="1s", tz="UTC"),
)
# NaN cases: mid-sample hole, a fully-NaN row, early rolling warmup NaNs.
df.iloc[250, 0] = np.nan
df.iloc[300] = np.nan
df.iloc[301, 2] = np.inf  # inf must also invalidate its windows

enc = TemporalWindowEncoder(
    window=8, bins=4, normalization_window=5, normalization_min_periods=2,
    slow_feature_window=7, slow_min_periods=2,
)
enc.fit_encoder(df)
new_out = enc.transform(df)

# --- Old implementation, verbatim (pre-fix transform body) ------------------
values = enc.prepare_input(df)
history = values.shift(1)
mean = history.rolling(enc.normalization_window, min_periods=enc.normalization_min_periods).mean()
scale = history.rolling(enc.normalization_window, min_periods=enc.normalization_min_periods).std(ddof=0)
delta = values - mean
z = delta.div(scale.where(scale > 1e-12)).clip(-enc.clip_z, enc.clip_z)
flat = scale.le(1e-12) & mean.notna()
z = z.mask(flat & delta.abs().le(1e-12), 0.0)
z = z.mask(flat & delta.gt(1e-12), enc.clip_z).mask(flat & delta.lt(-1e-12), -enc.clip_z)
slow_levels = {}
for column in values.columns:
    series = values[column]
    slow_levels[column] = (
        series.rolling(enc.slow_feature_window, min_periods=enc.slow_min_periods).std(ddof=0)
        if column in {"mid_return", "log_return"}
        else series.rolling(enc.slow_feature_window, min_periods=enc.slow_min_periods).mean()
    )
slow = pd.DataFrame(slow_levels, index=values.index)
slow_history = slow.shift(1)
slow_mean = slow_history.expanding(min_periods=enc.slow_min_periods).mean()
slow_scale = slow_history.expanding(min_periods=enc.slow_min_periods).std(ddof=0)
slow_delta = slow - slow_mean
slow_z = slow_delta.div(slow_scale.where(slow_scale > 1e-12))
slow_flat = slow_scale.le(1e-12) & slow_mean.notna()
slow_z = slow_z.mask(slow_flat & slow_delta.abs().le(1e-12), 0.0)
slow_z = slow_z.mask(slow_flat & slow_delta.gt(1e-12), enc.clip_z)
slow_z = slow_z.mask(slow_flat & slow_delta.lt(-1e-12), -enc.clip_z)
slow_z = (enc.slow_level_weight * slow_z).clip(-enc.clip_z, enc.clip_z)
windows = build_temporal_windows(z, enc.window)
old_out = np.full((len(values), len(enc.embedding_columns_)), np.nan)
if len(values) >= enc.window:
    bin_indices = np.array_split(np.arange(enc.window), enc.bins)
    for row in range(enc.window - 1, len(values)):
        sequence = windows[row]
        if not np.isfinite(sequence).all():
            continue
        means = [sequence[idx].mean(axis=0) for idx in bin_indices]
        dispersion = sequence.std(axis=0, ddof=0)
        centered = np.linspace(-1.0, 1.0, enc.window)
        centered -= centered.mean()
        slope = centered @ sequence / max(float(centered @ centered), 1e-12)
        slow_row = slow_z.iloc[row].to_numpy(dtype=np.float64)
        if not np.isfinite(slow_row).all():
            continue
        old_out[row] = np.concatenate([*means, dispersion, slope, slow_row])
old_out = pd.DataFrame(old_out, index=df.index, columns=enc.embedding_columns_)

pd.testing.assert_frame_equal(new_out, old_out)
skipped = int(new_out.isna().all(axis=1).sum())
print(f"EQUIVALENCE OK: {n} rows x {len(enc.embedding_columns_)} cols identical; {skipped} skipped rows")
