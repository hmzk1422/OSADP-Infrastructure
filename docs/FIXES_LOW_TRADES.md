# Fixes for Low Trade Count — Evidence & Documentation

**Date:** 2026-08-03
**Scope:** Align live executor with research (`ml_state_optimized.py`) to restore trade frequency and label consistency.
**Feature mismatch (price_roc_3 time-scale) intentionally NOT addressed** per user instruction.

---

## Root Cause Summary

The config constants were already synced (research imports from `main.config`). The low trade count was caused by **structural code-path divergences**, not config values:

1. **Hysteresis detector disarmed outside the hunting zone** → the biggest trade killer.
2. **Training labels used the old sqrt-inflated barrier formula** that research deleted → model learned wrong WIN/LOSS definitions.
3. **`BARRIER_VOL_BAR_WINDOW` was not defined in config** → silent fallback drift risk.

---

## Change 1 — `main/config.py`: Add `BARRIER_VOL_BAR_WINDOW`

**Evidence (before):** `features.py` line 111 used `globals().get('BARRIER_VOL_BAR_WINDOW', 20)` and `ml_state_optimized.py` lines 53-56 had a `try/except ImportError` fallback to 20. Both warned this could silently drift.

**Change:** Added explicit constant:
```python
BARRIER_VOL_BAR_WINDOW = 20
```
with a comment explaining it closes the fallback so live engine, research pipeline, and training builder all use the same window.

**Why it matters:** This window sizes the barrier (`bar_vol_std` over trailing N bars). If it ever changed in one place but not another, research barriers and live barriers would diverge silently.

---

## Change 2 — `main/executor.py`: Gate hysteresis detector to the hunting zone

**Evidence (before):** In `_one_sec_tick()`, `_update_signal_state(p_win)` was called on **every** 1-second tick regardless of RSI:
```python
fresh_signal = self._update_signal_state(p_win)   # ran unconditionally

if rsi_2_val >= RSI_FILTER_THRESHOLD:
    action = 'RSI_FILTERED'
```

**The bug:** The detector could disarm while RSI(2) >= 30 (P(WIN) >= threshold, but the RSI filter then rejected the entry). It only re-armed once P(WIN) dropped below `threshold - hysteresis` (0.42). On a saturated model that never dips below 0.42, the detector stayed disarmed for the whole session → ~0-1 trades.

**Research comparison:** `ml_state_optimized.py` line 630 — `X = df_training_zone[features]` filters RSI(2) < 30 **before** `find_debounced_signals()` runs, so the detector only ever sees hunting-zone rows.

**Change:**
```python
fresh_signal = False
if rsi_2_val < RSI_FILTER_THRESHOLD:
    fresh_signal = self._update_signal_state(p_win)
```
Now the detector can only disarm when a trade could actually fire, mirroring research.

**Expected effect:** Large increase in trade count — the detector no longer gets "stuck" disarmed outside the hunting zone.

---

## Change 3 — `main/executor.py`: Rewrite `HistoricalDataBuilder` barrier labeling to research formula

**Evidence (before):** `fetch_training_data()` used the OLD formula research explicitly deleted:
```python
ret_std_1m = df_features['close'].pct_change().rolling(window=BARRIER_VOL_WINDOW).std()  # ddof=1, 100 bars
horizon_std = ret_std_1m * np.sqrt(TIMEFRAME_BARRIER)   # sqrt(300) ~ 17.3x inflation
profit_barrier = horizon_std * BARRIER_MULTIPLIER
stop_barrier = horizon_std * BARRIER_MULTIPLIER
lookahead_bars = max(1, LOOKAHEAD_SECONDS // 60)        # 3000//60 = 50 bars = 250 min
```

**The bug:** This produced ~17x wider training barriers than live exits use (`PositionTracker.get_barrier_pct()` primary path = `_bar_vol_std * BARRIER_MULTIPLIER`, NO sqrt). The model learned WIN/LOSS definitions that never matched what the live engine trades against. Also the horizon was 5x too long (250 min vs research's 50 min).

**Research comparison:** `ml_state_optimized.py` `add_barrier_labels()` (lines 155-198):
- `bar_vol_std` = std of bar-to-bar % returns over trailing `BARRIER_VOL_BAR_WINDOW` bars, **ddof=0** (`np.nanstd` default)
- `stop_barrier = bar_vol_std * BARRIER_MULTIPLIER`
- `profit_barrier = bar_vol_std * BARRIER_MULTIPLIER * RR_RATIO` (via `_sequential_triple_barrier`: `upper = p0 * (1 + pt * RR_RATIO)`)
- **NO sqrt(horizon) scaling**
- Bootstrap floor: `bar_vol_std.where(bar_vol_std > 0, 0.001)`

**Change:**
```python
bar_rets = df_features['close'].pct_change()
bar_vol_std = bar_rets.rolling(window=BARRIER_VOL_BAR_WINDOW, min_periods=2).std(ddof=0)
bar_vol_std = bar_vol_std.where(bar_vol_std > 0, 0.001)
stop_barrier = bar_vol_std * BARRIER_MULTIPLIER
profit_barrier = stop_barrier * RR_RATIO
lookahead_bars = max(1, LOOKAHEAD_SECONDS // TIMEFRAME_BARRIER)   # 3000//300 = 10 bars = 50 min
```

**Expected effect:** The deployed model is now trained on labels that match the live exit barriers, so P(WIN) values become meaningful and the confidence threshold behaves like research.

---

## Important Follow-up Action

**Delete the stale model** at `main/models/lgb_model.txt` (or set `TRAIN_ON_STARTUP=True` and remove the file) so the executor rebuilds on the corrected labels. Otherwise the old model trained on the broken sqrt-inflated barriers keeps serving miscalibrated P(WIN).

---

## What Was NOT Changed (per user instruction)

- **Feature mismatch #2** — `price_roc_3` in `features.py` is computed on 1-second bars (3-second ROC) while research computes it on 300-second bars (15-minute ROC). This remains as-is for now.
- **Feature mismatch #4** — research trains 16 features (7 + 9 orderflow), live serves 7. Not addressed.

---

## Verification

Run a syntax check on the modified files:
```bash
python -m py_compile main/config.py main/executor.py
```

Then run the executor in dry-run (`DRY_RUN=True` is the default) and observe the decision log — you should see `ENTER_SHORT` actions firing when RSI(2) < 30 and P(WIN) >= 0.5, instead of the detector being stuck disarmed.