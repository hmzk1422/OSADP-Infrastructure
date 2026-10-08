import pandas as pd
import numpy as np

#: BTCUSDT instrument tick. Order/barrier prices are snapped with
#: (price / tick).round() * tick -- identical formula to
#: strategy_system.exchange.loader.quantize_price and
#: uets VenueSpec::round_price. Keeps backtest fills, equality checks and
#: live orders on-tick so the venue never returns INVALID_TICK_SIZE.
TICK_SIZE = 0.01


def quantize_price(price, tick: float = TICK_SIZE):
    """Snap price(s) to the instrument tick (exact Decimal grid value).

    Series- and scalar-safe; NaN passes through.
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(str(tick))

    def _one(v):
        import math
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        if not math.isfinite(f):
            return v
        if isinstance(v, float) and math.isnan(f):
            return v
        return float(Decimal(str(f)).quantize(q, rounding=ROUND_HALF_UP))

    try:
        if hasattr(price, "apply"):
            return price.apply(_one)
        return _one(price)
    except Exception:
        return price

def add_triple_barrier_sequence(
    df: pd.DataFrame,
    signal_col: str = "rsi_d2_signal",
    price_col: str = "price_close",
    pt: float = 0.005,
    sl: float = 0.003,
    horizon: int = 30,
    high_col: str | None = None,
    low_col: str | None = None,
) -> pd.DataFrame:
    """
    Convert trading signals into triple-barrier events.

    Parameters
    ----------
    df:
        DataFrame containing price and signal columns.

    signal_col:
        Signal column.
        Expected values:
            +1 = long
            -1 = short
             0 = no event

    price_col:
        Entry/close price column.

    pt:
        Profit-taking threshold as a decimal.
        Example: 0.005 = 0.5%.

    sl:
        Stop-loss threshold as a decimal.
        Example: 0.003 = 0.3%.

    horizon:
        Maximum number of bars to hold the event.

    high_col:
        Optional high-price column.
        When supplied, upper/lower barriers are detected
        using intrabar highs/lows.

    low_col:
        Optional low-price column.

    Returns
    -------
    pd.DataFrame
        One row per signal event with:

        t0
        t1
        entry_price
        side
        profit_barrier
        stop_barrier
        exit_price
        return
        label
        barrier
        bars_held
    """

    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    if pt <= 0:
        raise ValueError("pt must be > 0")

    if sl <= 0:
        raise ValueError("sl must be > 0")

    required = [signal_col, price_col]

    if high_col is not None:
        required.append(high_col)

    if low_col is not None:
        required.append(low_col)

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    data = df.copy()

    # -------------------------------------------------------
    # Ensure deterministic ordering.
    # -------------------------------------------------------

    data = data.sort_index()

    prices = data[price_col].to_numpy(dtype=float)
    signals = data[signal_col].fillna(0).to_numpy()

    if high_col is not None:
        highs = data[high_col].to_numpy(dtype=float)
    else:
        highs = prices

    if low_col is not None:
        lows = data[low_col].to_numpy(dtype=float)
    else:
        lows = prices

    timestamps = data.index.to_numpy()

    events = []

    # -------------------------------------------------------
    # Process every non-zero signal.
    # -------------------------------------------------------

    for i, signal in enumerate(signals):

        if signal == 0:
            continue

        if not np.isfinite(prices[i]):
            continue

        # Only support long / short signals.
        if signal not in (-1, 1):
            raise ValueError(
                f"{signal_col} must contain only -1, 0, +1. "
                f"Found {signal!r} at row {i}."
            )

        side = int(signal)

        entry_price = quantize_price(prices[i], TICK_SIZE)

        # ---------------------------------------------------
        # Side-aware barriers, snapped to the instrument tick so generated
        # limit / stop / take-profit prices are venue-valid and equality
        # comparisons against tick-grid book prices behave.
        # ---------------------------------------------------

        profit_barrier = quantize_price(entry_price * (1.0 + side * pt), TICK_SIZE)
        stop_barrier = quantize_price(entry_price * (1.0 - side * sl), TICK_SIZE)

        # ---------------------------------------------------
        # Vertical barrier.
        #
        # +1 because bar i itself is the entry bar.
        # ---------------------------------------------------

        end = min(i + horizon, len(data) - 1)

        exit_idx = end
        barrier = "vertical"

        # ---------------------------------------------------
        # Search forward for first barrier touch.
        # ---------------------------------------------------

        for j in range(i + 1, end + 1):

            high = highs[j]
            low = lows[j]

            if not np.isfinite(high) or not np.isfinite(low):
                continue

            # -----------------------------------------------
            # Long:
            #   profit when high >= profit_barrier
            #   stop   when low  <= stop_barrier
            #
            # Short:
            #   profit when low  <= profit_barrier
            #   stop   when high >= stop_barrier
            # -----------------------------------------------

            profit_hit = (
                high >= profit_barrier
                if side == 1
                else low <= profit_barrier
            )

            stop_hit = (
                low <= stop_barrier
                if side == 1
                else high >= stop_barrier
            )

            if profit_hit and stop_hit:
                # With OHLC bars we don't know which barrier
                # was touched first within the bar.
                #
                # Use stop as the conservative tie-break.
                exit_idx = j
                barrier = "stop"
                break

            if profit_hit:
                exit_idx = j
                barrier = "profit"
                break

            if stop_hit:
                exit_idx = j
                barrier = "stop"
                break

        # ---------------------------------------------------
        # Exit price.
        #
        # For a barrier hit we use the barrier price.
        # For vertical expiration we use the closing price.
        # ---------------------------------------------------

        if barrier == "profit":
            exit_price = profit_barrier

        elif barrier == "stop":
            exit_price = stop_barrier

        else:
            exit_price = quantize_price(prices[exit_idx], TICK_SIZE)

        # ---------------------------------------------------
        # Side-adjusted return.
        #
        # Long:
        #   (exit / entry) - 1
        #
        # Short:
        #   1 - (exit / entry)
        # ---------------------------------------------------

        raw_return = (exit_price / entry_price) - 1.0
        strategy_return = side * raw_return

        # ---------------------------------------------------
        # Triple-barrier label.
        #
        # +1 = profit barrier
        # -1 = stop barrier
        #  0 = vertical barrier
        # ---------------------------------------------------

        if barrier == "profit":
            label = 1
        elif barrier == "stop":
            label = -1
        else:
            label = 0

        events.append(
            {
                "t0": timestamps[i],
                "t1": timestamps[exit_idx],
                "entry_price": entry_price,
                "side": side,
                "profit_barrier": profit_barrier,
                "stop_barrier": stop_barrier,
                "exit_price": exit_price,
                "return": strategy_return,
                "label": label,
                "barrier": barrier,
                "bars_held": exit_idx - i,
            }
        )

    return pd.DataFrame(events)