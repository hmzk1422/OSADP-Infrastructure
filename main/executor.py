"""
Live Strategy Executor — 1-Second Resolution
==============================================
The core loop that ties everything together, matching research/ml-state.ipynb.

Research Architecture (ml-state.ipynb):
    df_micro = df_trades.resample('1s')       → 1-second bars
    df_macro = df_ob_1m (resample '1min')     → RSI, candle, orderflow
    df_state_space = df_micro.join(df_macro, how='left').ffill()
    → model runs on EVERY 1-SECOND row ←

Live Architecture:
    WebSocket trades → LiveFeatureEngine (1s micro + 1min macro forward-filled)
    Every 1 second: compute_1s_feature_vector() → model.predict_proba()
    → RSI(2) filter → if P(WIN) >= threshold → ENTER SHORT (fade)

Usage:
    python -m main.executor
"""

import sys
import os
import asyncio
import logging
import json
import signal
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict

# Suppress sklearn feature name warnings (numpy arrays don't have column names)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Local imports ---
from main.config import *
from main.features import LiveFeatureEngine
from main.execution import QuantSimulator

# --- Utils ---
from utils.indicators import get_wilders_rsi, sma
from utils.bybit import BybitConnector

# --- ML ---
import lightgbm as lgb

# --- Helper ---
def _pad_front(arr, target_len):
    """Pad array with NaN at front to match target length."""
    out = np.full(target_len, np.nan)
    out[-len(arr):] = arr
    return out


# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('main/logs/executor.log', mode='a'),
    ],
)
logger = logging.getLogger("executor")

# Ensure logs directory exists
os.makedirs('main/logs', exist_ok=True)


# ============================================================================
# Model Manager
# ============================================================================
class ModelManager:
    """Manages the LightGBM model lifecycle."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[lgb.Booster] = None
        self.feature_columns: list = []
        self.is_trained: bool = False

    def load(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                self.model = lgb.Booster(model_file=self.model_path)
                self.is_trained = True
                logger.info(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return False

    def save(self) -> None:
        if self.model is not None:
            try:
                self.model.booster_.save_model(self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        neg_pos_ratio = sum(y == 0) / max(sum(y == 1), 1)
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            scale_pos_weight=neg_pos_ratio,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X, y)
        self.is_trained = True
        self.save()
        train_score = self.model.score(X, y)
        logger.info(f"Model trained on {len(X)} samples. Train accuracy: {train_score:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([[0.5, 0.5]])
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([0])
        return self.model.predict(X)


# ============================================================================
# Historical Data Builder (for initial training)
# ============================================================================
class HistoricalDataBuilder:
    """
    Fetches historical 1-min OHLCV + orderflow data to build
    a training dataset matching the research pipeline.
    """

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
        self.connector = BybitConnector(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)

    def fetch_training_data(self, n_bars: int = N_HISTORICAL_BARS) -> pd.DataFrame:
        logger.info(f"Fetching {n_bars} bars of historical data for training...")
        df_ohlcv = self.connector.get_ohlcv(
            symbol=self.symbol, interval='1m', total_limit=n_bars,
        )
        if df_ohlcv.empty:
            logger.error("No historical data fetched!")
            return pd.DataFrame()

        df_ohlcv = df_ohlcv.rename(columns={'timestamp': 'datetime'}).set_index('datetime')
        price = df_ohlcv['close'].values
        diffs = np.diff(price, prepend=price[0])

        # RSI features
        rsi_2 = get_wilders_rsi(diffs, RSI_2_WINDOW)
        rsi_14 = get_wilders_rsi(diffs, RSI_14_WINDOW)
        sma_14_vals = sma(rsi_14, SMA_14_WINDOW)

        # Micro features (approximate from 1-min data)
        buy_vol = df_ohlcv['volume'] * 0.52
        sell_vol = df_ohlcv['volume'] * 0.48
        micro_OFI = (buy_vol - sell_vol) / 60.0
        micro_CVD = micro_OFI.rolling(CVD_ROLLING).sum()

        # Candle features
        candle_range = (df_ohlcv['high'] - df_ohlcv['low']).replace(0, EPS)
        candle_type = abs(df_ohlcv['close'] - df_ohlcv['open']) / candle_range
        candle_size = candle_range / (candle_range.rolling(14).mean().replace(0, EPS))

        # Orderflow features
        total_vol = df_ohlcv['volume'].replace(0, EPS)
        largest_buyer = (buy_vol * 0.15) / total_vol * 100.0
        largest_seller = (sell_vol * 0.15) / total_vol * 100.0
        OFI_LOB = 100.0 * (total_vol * 0.04) / (total_vol + EPS)
        OFI_AGG = 100.0 * (buy_vol - sell_vol) / (buy_vol + sell_vol + EPS)
        CVD_raw = buy_vol - sell_vol
        cvd_delta = CVD_raw.diff()
        CVD_z = ((cvd_delta - cvd_delta.rolling(OFI_LOOKBACK).mean())
                 / (cvd_delta.rolling(OFI_LOOKBACK).std() + EPS))

        df_features = pd.DataFrame({
            'micro_OFI': micro_OFI,
            'micro_CVD': micro_CVD,
            'rsi_2': _pad_front(rsi_2, len(df_ohlcv)),
            'rsi_14': _pad_front(rsi_14, len(df_ohlcv)),
            'sma_14': _pad_front(sma_14_vals, len(df_ohlcv)),
            'largest_buyer': largest_buyer,
            'largest_seller': largest_seller,
            'OFI_LOB': OFI_LOB,
            'OFI_AGG': OFI_AGG,
            'CVD_z': CVD_z,
            'candle_type': candle_type,
            'candle_size': candle_size,
            'close': price,
        }, index=df_ohlcv.index)

        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_features) < 200:
            logger.warning(f"Only {len(df_features)} rows after feature computation.")

        # Barrier labeling
        ret_std_1m = df_features['close'].pct_change().rolling(window=BARRIER_VOL_WINDOW).std()
        horizon_std = ret_std_1m * np.sqrt(LOOKAHEAD_SECONDS)
        profit_barrier = horizon_std * BARRIER_MULTIPLIER
        stop_barrier = horizon_std * BARRIER_MULTIPLIER
        lookahead_bars = max(1, LOOKAHEAD_SECONDS // 60)

        labels = []
        for i in range(len(df_features) - lookahead_bars):
            future_prices = df_features['close'].iloc[i + 1:i + 1 + lookahead_bars]
            entry = df_features['close'].iloc[i]
            pb = profit_barrier.iloc[i]
            sb = stop_barrier.iloc[i]
            if pd.isna(pb) or pd.isna(sb) or pb == 0:
                labels.append(np.nan)
                continue
            profit_level = entry * (1.0 + pb)
            stop_level = entry * (1.0 - sb)
            hit_profit = (future_prices >= profit_level).any()
            hit_stop = (future_prices <= stop_level).any()
            if hit_profit and not hit_stop:
                labels.append(LABEL_WIN)
            elif hit_stop:
                labels.append(LABEL_LOSS)
            else:
                last_price = future_prices.iloc[-1]
                labels.append(LABEL_WIN if last_price > entry else LABEL_LOSS)

        labels = list(labels) + [np.nan] * lookahead_bars
        df_features['target'] = labels
        df_features = df_features.dropna(subset=['target'])
        df_features['target'] = df_features['target'].astype(int)

        # RSI filter
        df_hunting = df_features[df_features['rsi_2'] < RSI_FILTER_THRESHOLD].copy()
        logger.info(
            f"Training data: {len(df_features)} labeled rows, "
            f"{len(df_hunting)} in hunting zone (rsi_2 < {RSI_FILTER_THRESHOLD})"
        )
        return df_hunting

    def build_training_set(self, n_bars: int = N_HISTORICAL_BARS) -> tuple:
        df = self.fetch_training_data(n_bars)
        if df.empty:
            return None, None, None
        X = df[FEATURE_COLUMNS].values
        y = df['target'].values
        n_win = int((y == LABEL_WIN).sum())
        n_loss = int((y == LABEL_LOSS).sum())
        logger.info(f"Training set: {len(X)} samples | WIN: {n_win} | LOSS: {n_loss}")
        return X, y, FEATURE_COLUMNS


# ============================================================================
# Main Strategy Executor
# ============================================================================
class StrategyExecutor:
    """
    The main orchestrator. Runs the live strategy loop at 1-SECOND resolution.

    Flow (matches research/ml-state.ipynb exactly):
        1. WebSocket streams → on_trade() / on_orderbook()
        2. LiveFeatureEngine aggregates into 1-second micro bars
        3. Every 1 second: _one_sec_tick() fires
            a. compute_1s_feature_vector() (12 features)
            b. RSI(2) filter: must be < 30
            c. model.predict_proba()
            d. If P(WIN) >= threshold → ENTER SHORT (fade strategy)
        4. Position monitoring runs on EVERY trade tick
        5. Exit on barrier hit or timeout (barriers from execution.py)
    """

    def __init__(self, symbol: str = SYMBOL, dry_run: bool = DRY_RUN):
        self.symbol = symbol
        self.dry_run = dry_run

        # Components
        self.feature_engine = LiveFeatureEngine()
        self.simulator = QuantSimulator()
        self.model_mgr = ModelManager()

        # State
        self.running = False
        self.second_count = 0
        self.last_logged_rsi_2 = 50.0
        self.last_logged_p_win = 0.5

        # For logging per-second decisions (throttled to avoid log spam)
        self.decision_log: list = []

        # Rate limiting
        self._last_trade_time: Optional[datetime] = None

        logger.info(
            f"StrategyExecutor initialized | Symbol: {symbol} | "
            f"Dry-run: {dry_run} | Resolution: 1-second"
        )

    async def initialize(self) -> bool:
        """Initialize: load/train model + warm up feature engine."""
        # 1. Model
        loaded = self.model_mgr.load()
        if not loaded and TRAIN_ON_STARTUP:
            logger.info("No model found. Training on historical data...")
            builder = HistoricalDataBuilder(self.symbol)
            X, y, feature_cols = builder.build_training_set()
            if X is not None and len(X) > 100:
                self.model_mgr.train(X, y)
                self.model_mgr.feature_columns = feature_cols
            else:
                logger.error("Insufficient training data. Cannot start.")
                return False
        elif loaded:
            logger.info("Model loaded successfully.")
        else:
            logger.error("No model found and TRAIN_ON_STARTUP is False.")
            return False

        # 2. Warm up feature engine with recent historical data
        logger.info("Warming up feature engine with historical data...")
        connector = BybitConnector(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)
        df_warmup = connector.get_ohlcv(self.symbol, '1m', total_limit=200)

        if not df_warmup.empty:
            for _, row in df_warmup.iterrows():
                ts = int(row['timestamp'].timestamp() * 1000)
                self.feature_engine.on_trade(ts, row['close'], row['volume'] / 2, 'buy')
                self.feature_engine.on_trade(ts, row['close'], row['volume'] / 2, 'sell')
                self.feature_engine.on_orderbook(
                    ts, row['close'] * 0.999, 10.0, row['close'] * 1.001, 10.0
                )
            logger.info(f"Warmup complete: {self.bar_count if hasattr(self, 'bar_count') else 'N/A'} bars")

        return True

    async def on_trade(self, trade: Dict) -> None:
        """Process a single trade from WebSocket."""
        timestamp_ms = trade.get('timestamp', 0)
        price = float(trade.get('price', 0))
        amount = float(trade.get('amount', 0))
        side = trade.get('side', 'buy')

        self.feature_engine.on_trade(timestamp_ms, price, amount, side)

        # Update position monitoring on EVERY tick
        if self.simulator.position.active:
            exit_result = self.simulator.update_tick(price)
            if exit_result:
                self._log_trade(exit_result)

    async def on_orderbook(self, ob: Dict) -> None:
        """Process an orderbook snapshot from WebSocket."""
        timestamp_ms = ob.get('timestamp', 0)
        best_bid = ob.get('bids', [[0, 0]])[0]
        best_ask = ob.get('asks', [[0, 0]])[0]
        self.feature_engine.on_orderbook(
            timestamp_ms,
            float(best_bid[0]),
            float(best_bid[1]),
            float(best_ask[0]),
            float(best_ask[1]),
        )

    async def _one_sec_tick(self) -> None:
        """
        Called EVERY SECOND by the 1-second clock.
        This is the main strategy decision point — matching research
        where model runs on every row of the 1-second state space.

        TRANSPARENT ALGORITHM (matches research/ml-state.ipynb):
        ┌─────────────────────────────────────────────────────────────┐
        │  1. Compute 12 features from current 1-second bar          │
        │  2. FILTER: rsi_2 < 30 (hunting zone, research line 580)   │
        │  3. INFER: model.predict_proba(feature_vector)             │
        │  4. DECIDE: if P(WIN) >= 0.50 → ENTER SHORT (fade)        │
        │            else → HOLD (no trade this second)              │
        └─────────────────────────────────────────────────────────────┘
        """
        # ──[1] Compute 12-feature vector for this second ──
        features = self.feature_engine.compute_1s_feature_vector()
        if not features:
            return

        self.second_count += 1
        current_price = features.get('close', 0)
        rsi_2_val = features.get('rsi_2', 100)
        rsi_14_val = features.get('rsi_14', 100)

        # Extract all 12 features for transparency
        feature_cols = FEATURE_COLUMNS
        feature_values = {col: features.get(col, 0.0) for col in feature_cols}

        # ──[2] Model inference ──
        feature_vector = self.feature_engine.extract_feature_vector(features)
        proba = self.model_mgr.predict_proba(feature_vector)[0]
        p_win = proba[1]
        p_loss = proba[0]

        # Build decision record (only stored when action changes, logged every time)
        decision = {
            'time': datetime.now().isoformat(),
            'sec': self.second_count,
            'close': current_price,
            'rsi_2': rsi_2_val,
            'rsi_14': rsi_14_val,
            'p_win': float(p_win),
            'p_loss': float(p_loss),
            'in_position': self.simulator.position.active,
            'features': feature_values,
        }

        # ──[3] RSI Filter: rsi_2 < 30 (matches research line 580) ──
        if rsi_2_val >= RSI_FILTER_THRESHOLD:
            # Log every 60 seconds (once per minute) when in SKIP state to avoid spam
            if self.second_count % 60 == 0:
                logger.info(
                    f"[SEC {self.second_count}] SKIP | RSI(2)={rsi_2_val:.1f} >= {RSI_FILTER_THRESHOLD} | "
                    f"Close={current_price:.2f} | P(WIN)={p_win:.3f}"
                )
            decision['action'] = 'SKIP (rsi_2 too high)'
            self._store_decision(decision)
            return

        # ──[4] Check if already in position ──
        if self.simulator.position.active:
            if self.second_count % 60 == 0:
                logger.info(
                    f"[SEC {self.second_count}] IN POSITION | RSI(2)={rsi_2_val:.1f} | "
                    f"P(WIN)={p_win:.3f} | Close={current_price:.2f}"
                )
            decision['action'] = 'IN_POSITION'
            self._store_decision(decision)
            return

        # ──[5] Check rate limit ──
        if not self.simulator.can_trade():
            if self.second_count % 60 == 0:
                logger.info(
                    f"[SEC {self.second_count}] SKIP (rate limit) | Trades/hour used"
                )
            decision['action'] = 'SKIP (rate limit)'
            self._store_decision(decision)
            return

        # ──[6] TRADE DECISION ──
        # Research: P(WIN) high → price likely goes UP → we FADE → SHORT
        # This matches research INVERT_RETURNS=True
        if p_win >= CONFIDENCE_THRESHOLD:
            # ──[6a] Calculate position size ──
            qty = self.simulator.calculate_quantity(current_price)
            if qty < MIN_ORDER_SIZE:
                decision['action'] = f'SKIP (qty too small: {qty:.6f})'
                self._store_decision(decision)
                return

            # ──[6b] ENTER SHORT ──
            receipt = self.simulator.enter_short(current_price, qty)
            if receipt is not None:
                decision['action'] = 'ENTER SHORT'
                decision['qty'] = qty
                decision['fill_price'] = receipt['fill_price']
                decision['slippage_bps'] = receipt['slippage_bps']
                decision['stop_barrier'] = receipt['stop_barrier']
                decision['profit_barrier'] = receipt['profit_barrier']
                decision['barrier_pct'] = receipt['barrier_pct']

                # ALWAYS log trade entry (critical event)
                logger.info(
                    f"[TRADE ENTER] Sec {self.second_count} | "
                    f"RSI(2)={rsi_2_val:.1f} | "
                    f"P(WIN)={p_win:.3f} >= {CONFIDENCE_THRESHOLD} | "
                    f"→ ENTER SHORT @ {receipt['fill_price']:.2f} | "
                    f"qty={qty:.4f} | slippage={receipt['slippage_bps']:.2f}bps"
                )
                # Log all 12 feature values that triggered the trade
                logger.info(
                    f"[TRADE FEATURES] Sec {self.second_count} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in feature_values.items())
                )
            else:
                decision['action'] = 'FAILED (entry rejected)'
        else:
            # HOLD: model doesn't see the pattern
            # Log every 10 seconds when in hunting zone but no trade
            if self.second_count % 10 == 0:
                logger.info(
                    f"[SEC {self.second_count}] HOLD | RSI(2)={rsi_2_val:.1f} | "
                    f"P(WIN)={p_win:.3f} < {CONFIDENCE_THRESHOLD} | "
                    f"Close={current_price:.2f}"
                )

            decision['action'] = 'HOLD'
            self._store_decision(decision)

    def _store_decision(self, decision: Dict) -> None:
        """Store decision in rolling log (keep last 1000 for summary)."""
        self.decision_log.append(decision)
        if len(self.decision_log) > 1000:
            self.decision_log.pop(0)

    def _log_trade(self, trade_result: Dict) -> None:
        """Log a completed trade with win/loss outcome."""
        reason = trade_result.get('reason', 'unknown')
        net_pnl_pct = trade_result.get('net_pnl_pct', 0.0)
        net_pnl_usd = trade_result.get('net_pnl_usd', 0.0)
        gross_pnl_pct = trade_result.get('gross_pnl_pct', 0.0)
        r_multiple = trade_result.get('r_multiple', 0.0)
        entry_price = trade_result.get('entry_price', 0.0)
        exit_price = trade_result.get('exit_price', 0.0)
        duration = trade_result.get('duration_seconds', 0)

        # Determine win/loss
        outcome = "WIN" if net_pnl_usd > 0 else "LOSS"

        logger.info(
            f"[TRADE EXIT] {outcome} | {reason:12s} | "
            f"Entry: {entry_price:.2f} → Exit: {exit_price:.2f} | "
            f"Gross: {gross_pnl_pct*100:+.3f}% | "
            f"Net: {net_pnl_pct*100:+.3f}% | "
            f"${net_pnl_usd:+.2f} | "
            f"R: {r_multiple:+.4f} | "
            f"Duration: {duration:.0f}s"
        )

        # Save to trade log file
        log_file = 'main/logs/trades.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                **trade_result,
                'outcome': outcome,
                'entry_time': str(trade_result.get('entry_time', '')),
                'exit_time': str(trade_result.get('exit_time', '')),
            }) + '\n')

    async def _create_exchange(self):
        """Create a new ccxt.pro exchange instance."""
        import ccxt.pro as ccxtpro
        import aiohttp
        exchange = ccxtpro.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
            },
        })
        if TESTNET:
            exchange.set_sandbox_mode(True)
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        exchange.session = aiohttp.ClientSession(connector=connector)
        await exchange.load_markets()
        return exchange

    async def _one_sec_clock(self):
        """
        Background task: fires _one_sec_tick() EVERY SECOND on the wall clock.
        This is the heartbeat that drives inference at 1-second resolution.
        """
        while self.running:
            now = datetime.now()
            # Sleep until next exact second
            next_sec = now.replace(microsecond=0) + timedelta(seconds=1)
            sleep_sec = (next_sec - now).total_seconds()
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

            if not self.running:
                break

            await self._one_sec_tick()

    async def _data_watchdog(self):
        """Monitor data freshness every 120 seconds."""
        last_second_count = self.second_count
        stale_warnings = 0
        while self.running:
            await asyncio.sleep(120)
            if not self.running:
                break
            if self.second_count == last_second_count and self.second_count > 0:
                stale_warnings += 1
                logger.warning(
                    f"WATCHDOG: No new seconds for 120s (total: {self.second_count}). "
                    f"Stream may be stalled. Warning #{stale_warnings}"
                )
            else:
                stale_warnings = 0
            last_second_count = self.second_count

    async def _stream_loop(self, exchange):
        """
        Core streaming loop: watches trades + orderbook.
        Returns True if reconnect needed, False on normal exit.
        """
        last_data_time = datetime.now()

        while self.running:
            try:
                trades_task = asyncio.create_task(exchange.watch_trades(self.symbol))
                ob_task = asyncio.create_task(exchange.watch_order_book(self.symbol))

                done, pending = await asyncio.wait(
                    [trades_task, ob_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=30,
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                if not done:
                    elapsed = (datetime.now() - last_data_time).total_seconds()
                    if elapsed > 60:
                        logger.warning(f"No data for {elapsed:.0f}s. Reconnecting...")
                        return True
                    continue

                for task in done:
                    try:
                        result = task.result()
                    except Exception as e:
                        logger.error(f"Stream task error: {e}")
                        continue

                    if task == trades_task:
                        for trade in result:
                            await self.on_trade(trade)
                        last_data_time = datetime.now()
                    elif task == ob_task:
                        await self.on_orderbook(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                await asyncio.sleep(5)
                return True

        return False

    async def run_forever(self) -> None:
        """
        Main loop: connect to WebSocket streams and process data.
        Uses ccxt.pro for real-time data with automatic reconnection.
        """
        self.running = True

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass

        # Start 1-second clock (fires inference every second)
        one_sec_clock_task = asyncio.create_task(self._one_sec_clock())

        # Start data watchdog
        watchdog_task = asyncio.create_task(self._data_watchdog())

        # Main reconnection loop
        max_retries = 1000
        retry_count = 0
        exchange = None

        while self.running and retry_count < max_retries:
            try:
                if exchange is not None:
                    try:
                        await exchange.close()
                    except Exception:
                        pass

                exchange = await self._create_exchange()
                logger.info(f"Connected to Bybit. Starting live stream for {self.symbol}...")

                should_reconnect = await self._stream_loop(exchange)

                if should_reconnect and self.running:
                    retry_count += 1
                    wait = min(5 * retry_count, 60)
                    logger.info(f"Reconnecting in {wait}s (attempt {retry_count}/{max_retries})...")
                    await asyncio.sleep(wait)
                else:
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fatal stream error: {e}", exc_info=True)
                retry_count += 1
                wait = min(5 * retry_count, 60)
                logger.info(f"Reconnecting in {wait}s (attempt {retry_count}/{max_retries})...")
                await asyncio.sleep(wait)

        # Cleanup
        if exchange is not None:
            try:
                await exchange.close()
            except Exception:
                pass

        one_sec_clock_task.cancel()
        try:
            await one_sec_clock_task
        except (asyncio.CancelledError, Exception):
            pass

        watchdog_task.cancel()
        try:
            await watchdog_task
        except (asyncio.CancelledError, Exception):
            pass

        self.running = False

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        self.simulator.close()
        self.print_summary()

    def print_summary(self) -> None:
        """Print execution summary."""
        self.simulator.print_summary()
        print(f"Seconds Processed : {self.second_count}")
        print(f"Minutes Elapsed   : {self.second_count // 60}")

        # Print last 10 decisions
        print("\nLast 10 decisions:")
        for dec in self.decision_log[-10:]:
            action = dec.get('action', 'N/A')
            rsi = dec.get('rsi_2', 0)
            p_win = dec.get('p_win', 0.5)
            close = dec.get('close', 0)
            print(f"  {dec['time'][:19]} | Sec {dec['sec']:>6d} | "
                  f"Close: {close:.2f} | RSI(2): {rsi:.1f} | "
                  f"P(WIN): {p_win:.3f} | {action}")


# ============================================================================
# Entry Point
# ============================================================================
async def main():
    """Main entry point."""
    print("="*70)
    print(" INTRADAY MOMENTUM COMPOSITE — LIVE EXECUTOR (1-SECOND RESOLUTION)")
    print("="*70)
    print(f"Symbol    : {SYMBOL}")
    print(f"Testnet   : {TESTNET}")
    print(f"Dry Run   : {DRY_RUN}")
    print(f"Model     : {MODEL_PATH}")
    print(f"Threshold : P(WIN) >= {CONFIDENCE_THRESHOLD}")
    print(f"Filter    : RSI(2) < {RSI_FILTER_THRESHOLD}")
    print(f"Lookahead : {LOOKAHEAD_SECONDS}s")
    print(f"Barrier   : {BARRIER_MULTIPLIER}x vol")
    print("="*70)

    executor = StrategyExecutor(symbol=SYMBOL, dry_run=DRY_RUN)
    initialized = await executor.initialize()
    if not initialized:
        logger.error("Failed to initialize. Exiting.")
        return

    await executor.run_forever()


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")