"""
Live Strategy Executor — 1-Second Resolution
==============================================
The core loop that ties everything together, matching research/ml_state_optimized.py.

Research Architecture (ml_state_optimized.py):
    df_micro = df_trades.resample('1s')       → 1-second bars
    df_macro = df_ob_1m (resample '1min')     → RSI, MA slope, spread, etc.
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
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict

import ccxt
import ccxt.pro as ccxtpro
import aiohttp

warnings.filterwarnings('ignore')

from main.config import *
from main.features import LiveFeatureEngine
from main.execution import QuantSimulator

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
from utils.indicators import get_wilders_rsi, sma, calculate_zscore
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
        """
        Train model with parameters matching ml_state_optimized.py exactly.
        
        Research params (ml_state_optimized.py lines 612-624):
            n_estimators=200, learning_rate=0.01, max_depth=3,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, class_weight='balanced', random_state=42
        """
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
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
            
        # Check if the model is a raw Booster (loaded from file)
        if isinstance(self.model, lgb.Booster):
            # Raw booster .predict() returns 1D array of positive class probabilities
            preds = self.model.predict(X)
            
            # Reconstruct the 2D array [[P(Loss), P(Win)]] to match scikit-learn format
            return np.vstack((1.0 - preds, preds)).T
            
        # If it's still an LGBMClassifier (right after live training)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([0])
            
        if isinstance(self.model, lgb.Booster):
            preds = self.model.predict(X)
            return (preds > 0.5).astype(int)
            
        return self.model.predict(X)


# ============================================================================
# Historical Data Builder (for initial training)
# ============================================================================
class HistoricalDataBuilder:
    """
    Fetches historical 1-min OHLCV data to build
    a training dataset matching the ml_state_optimized.py pipeline.
    
    Computes the 7 optimized features:
        spread_bps_z, ma_slope_5, rsi_x_ma, ma_accel, price_roc_3, rsi_2, sma_14
    """

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
        self.connector = BybitConnector(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)

    def fetch_training_data(self, n_bars: int = N_HISTORICAL_BARS) -> pd.DataFrame:
        # e.g. TIMEFRAME_BARRIER=300 -> "5m". (Previously f'{TIMEFRAME_BARRIER/60}m'
        # produced malformed strings like "5.0m" that exchanges don't recognize.)
        interval_str = f"{max(1, TIMEFRAME_BARRIER // 60)}m"
        logger.info(f"Fetching {n_bars} bars of historical data ({interval_str}) for training...")
        df_ohlcv = self.connector.get_ohlcv(
            symbol=self.symbol, interval=interval_str, total_limit=n_bars,
        )
        if df_ohlcv.empty:
            logger.error("No historical data fetched!")
            return pd.DataFrame()

        df_ohlcv = df_ohlcv.rename(columns={'timestamp': 'datetime'}).set_index('datetime')
        price = df_ohlcv['close'].values
        diffs = np.diff(price, prepend=price[0])

        # ================================================================
        # RSI features (ml_state_optimized.py lines 397-400)
        # ================================================================
        rsi_2 = get_wilders_rsi(diffs, RSI_2_WINDOW)
        rsi_14 = get_wilders_rsi(diffs, RSI_14_WINDOW)
        sma_14_vals = sma(rsi_14, SMA_14_WINDOW)

        # ================================================================
        # MA Slope (velocity) — ml_state_optimized.py lines 404-406
        # ma_5 = np.convolve(close_arr, np.ones(5)/5, mode='same')
        # ma_slope_5 = np.diff(ma_5)
        # ================================================================
        close_arr = price.astype(float)
        ma_5 = np.convolve(close_arr, np.ones(MA_SLOPE_WINDOW) / MA_SLOPE_WINDOW, mode='same')
        ma_5[:MA_SLOPE_WINDOW - 1] = np.nan
        ma_slope_5 = _pad_front(np.diff(ma_5), len(df_ohlcv))

        # ================================================================
        # MA Acceleration (2nd derivative) — ml_state_optimized.py lines 409-412
        # ma_accel = np.diff(ma_slope)
        # ================================================================
        ma_slope_arr = ma_slope_5.copy()
        ma_accel = _pad_front(np.diff(ma_slope_arr), len(df_ohlcv))

        # ================================================================
        # Price rate of change — ml_state_optimized.py line 415
        # price_roc_3 = close.pct_change(3)
        # ================================================================
        price_series = df_ohlcv['close']
        price_roc_3 = price_series.pct_change(PRICE_ROC_WINDOW_3)

        # ================================================================
        # DeepLOB: Spread in bps, z-scored — ml_state_optimized.py lines 430-431
        # spread_bps_z = calculate_zscore(depth_1m['spread_bps'], window=10)
        # ================================================================
        # Approximate spread from OHLCV data (no orderbook in historical)
        # Use high-low range as a proxy for spread volatility
        spread_approx = (df_ohlcv['high'] - df_ohlcv['low']) / df_ohlcv['close'] * 10000.0
        spread_bps_z = calculate_zscore(spread_approx.fillna(0).values, window=DEEPLOB_Z_WINDOW)

        # ================================================================
        # RSI × MA slope interaction — ml_state_optimized.py lines 437-439
        # rsi_z = calculate_zscore(rsi_2, window=20)
        # rsi_x_ma = rsi_z * ma_slope_5
        # ================================================================
        rsi_z = calculate_zscore(rsi_2, window=RSI_Z_WINDOW)
        rsi_x_ma = rsi_z * ma_slope_5

        # ================================================================
        # Build feature DataFrame
        # ================================================================
        df_features = pd.DataFrame({
            'spread_bps_z': spread_bps_z,
            'ma_slope_5': ma_slope_5,
            'rsi_x_ma': rsi_x_ma,
            'ma_accel': ma_accel,
            'price_roc_3': price_roc_3,
            'rsi_2': _pad_front(rsi_2, len(df_ohlcv)),
            'sma_14': _pad_front(sma_14_vals, len(df_ohlcv)),
            'close': price,
        }, index=df_ohlcv.index)

        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_features) < 200:
            logger.warning(f"Only {len(df_features)} rows after feature computation.")

        # ================================================================
        # Barrier labeling (matching ml_state_optimized.py add_barrier_labels)
        # ================================================================
        ret_std_1m = df_features['close'].pct_change().rolling(window=BARRIER_VOL_WINDOW).std()
        horizon_std = ret_std_1m * np.sqrt(TIMEFRAME_BARRIER)
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
    def __init__(self, symbol: str = SYMBOL, dry_run: bool = DRY_RUN):
        self.symbol = symbol
        self.dry_run = dry_run

        self.feature_engine = LiveFeatureEngine()
        self.simulator = QuantSimulator()
        self.model_mgr = ModelManager()

        # --- Live dashboard (observational only — never touches order flow) ---
        from main.dashboard import Dashboard
        self.dashboard = Dashboard(symbol=self.symbol)

        self.running = False
        self.second_count = 0
        self.decision_log: list = []

    async def initialize(self) -> bool:
        # --- Dashboard: start the web server before anything else so it's
        # already reachable while historical data warms up ---
        await self.dashboard.start()

        # Let the UI label the RSI/vol panel from the real config instead of
        # hardcoding numbers that can drift out of sync with config.py.
        self.dashboard.push_config({
            "timeframe_barrier_seconds": TIMEFRAME_BARRIER,
            "rsi_filter_threshold": RSI_FILTER_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "lookahead_seconds": LOOKAHEAD_SECONDS,
            "trade_direction": TRADE_DIRECTION,
            "candle_seconds": self.dashboard.candle_seconds,
        })

        # --- NEW CODE: Seed the buffers before doing anything else ---
        logger.info("Warming up live feature engine with historical data...")
        self.feature_engine.seed_historical_data(self.symbol, limit=150)
        # -------------------------------------------------------------

        loaded = self.model_mgr.load()
        if not loaded and TRAIN_ON_STARTUP:
            logger.info("No model found. Fetching historical data...")
            try:
                builder = HistoricalDataBuilder(self.symbol)
                X, y, feature_cols = builder.build_training_set()
                if X is not None and len(X) > 100:
                    self.model_mgr.train(X, y)
                    self.model_mgr.feature_columns = feature_cols
                else:
                    logger.error("Insufficient training data. Exiting.")
                    return False
            except Exception as e:
                logger.error(f"Failed historical training initialization: {e}")
                return False
        return True

    async def on_trade(self, trade: Dict) -> None:
        try:
            timestamp_ms = trade.get('timestamp', 0)
            price = float(trade.get('price', 0))
            amount = float(trade.get('amount', 0))
            side = trade.get('side', 'buy')

            self.feature_engine.on_trade(timestamp_ms, price, amount, side)

            if price > 0:
                self.dashboard.push_tick(price, timestamp_ms)

            # NOTE: volatility sampling moved to _one_sec_tick() so the barrier
            # vol buffer stays a true 1-second series, matching the research
            # pipeline's ret_std_1s. Do not sample it here — trades arrive at
            # irregular, sub-second frequency.

            if self.simulator.position.active:
                exit_result = self.simulator.update_tick(price)
                if exit_result:
                    self._log_trade(exit_result)
                    self.dashboard.push_trade_exit(exit_result)
                    self.dashboard.push_status(self._dashboard_status(price))
        except Exception as e:
            logger.error(f"Error processing trade event: {e}")

    async def on_orderbook(self, ob: Dict) -> None:
        try:
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
            # NOTE: volatility sampling moved to _one_sec_tick() — see on_trade().
        except Exception as e:
            logger.error(f"Error processing orderbook event: {e}")

    async def _one_sec_tick(self) -> None:
        """Shielded 1-second clock tick logic."""
        try:
            features = self.feature_engine.compute_1s_feature_vector()
            if not features:
                return

            self.second_count += 1
            current_price = features.get('close', 0)
            rsi_2_val = features.get('rsi_2', 100)

            # Sample the barrier-vol buffer once per second here (not from
            # on_trade/on_orderbook) so _price_buffer is a true evenly-spaced
            # 1-second series. This is now only a bootstrap fallback used
            # before the first aggregated bar completes.
            if current_price > 0:
                self.simulator.update_volatility(current_price)

            # Primary barrier-sizing input: std of returns between aggregated
            # TIMEFRAME_BARRIER bar closes — same series RSI(2) reads from.
            bar_vol_std = features.get('bar_vol_std', 0.0)
            if bar_vol_std > 0:
                self.simulator.update_bar_volatility(bar_vol_std)

            feature_vector = self.feature_engine.extract_feature_vector(features)
            proba = self.model_mgr.predict_proba(feature_vector)[0]
            p_win = proba[1]

            # Track what happened this tick so the dashboard's decision feed
            # reflects the exact branch taken below (also feeds decision_log,
            # which print_summary() already reads from).
            action = 'WAIT'

            if rsi_2_val >= RSI_FILTER_THRESHOLD:
                action = 'RSI_FILTERED'
            elif self.simulator.position.active:
                action = 'IN_POSITION'
            elif not self.simulator.can_trade():
                action = 'RATE_LIMITED'
            elif p_win >= CONFIDENCE_THRESHOLD:
                qty = self.simulator.calculate_quantity(current_price)
                if qty < MIN_ORDER_SIZE:
                    action = 'QTY_TOO_SMALL'
                else:
                    # Direction switch: 'short' = fade (bet against P(WIN)),
                    # 'long' = snapback (bet with P(WIN))
                    if TRADE_DIRECTION == 'long':
                        receipt = self.simulator.enter_long(current_price, qty)
                        if receipt is not None:
                            action = 'ENTER_LONG'
                            # For dashboard display only — lets the UI show/verify
                            # the exact RSI(2) and P(WIN) that fired this entry.
                            receipt['rsi_2'] = rsi_2_val
                            receipt['p_win'] = float(p_win)
                            logger.info(f"[ENTER LONG] Sec {self.second_count} @ {receipt['fill_price']:.2f}")
                            self.dashboard.push_trade_enter(receipt)
                    else:
                        receipt = self.simulator.enter_short(current_price, qty)
                        if receipt is not None:
                            action = 'ENTER_SHORT'
                            receipt['rsi_2'] = rsi_2_val
                            receipt['p_win'] = float(p_win)
                            logger.info(f"[ENTER SHORT] Sec {self.second_count} @ {receipt['fill_price']:.2f}")
                            self.dashboard.push_trade_enter(receipt)
            else:
                action = 'LOW_CONFIDENCE'
                # ADDED: This will print whenever the RSI is good, but the AI confidence is too weak.
                logger.info(f"Signal ignored: RSI is good, but P(WIN) confidence ({p_win:.3f}) is below {CONFIDENCE_THRESHOLD}.")

            decision = {
                'time': datetime.now().isoformat(),
                'sec': self.second_count,
                'close': current_price,
                'rsi_2': rsi_2_val,
                'bar_vol_std': bar_vol_std,
                'p_win': float(p_win),
                'action': action,
            }
            self.decision_log.append(decision)
            self.dashboard.push_decision(decision)
            self.dashboard.push_status(self._dashboard_status(current_price))

        except Exception as e:
            logger.error(f"Unhandled error in 1-sec tick loop: {e}")

    def _dashboard_status(self, current_price: Optional[float] = None) -> Dict:
        """QuantSimulator.get_status() extended with live position + mode
        details the dashboard needs but the simulator doesn't track itself."""
        status = self.simulator.get_status()
        pos = self.simulator.position

        unrealized_usd = 0.0
        unrealized_pct = 0.0
        elapsed_sec = 0.0
        if pos.active and current_price:
            if pos.side == 'short':
                unrealized_pct = (pos.entry_price - current_price) / pos.entry_price
            else:
                unrealized_pct = (current_price - pos.entry_price) / pos.entry_price
            unrealized_usd = unrealized_pct * pos.notional_usd
            if pos.entry_time:
                elapsed_sec = (datetime.now() - pos.entry_time).total_seconds()

        status.update({
            'side': pos.side,
            'stop_barrier': pos.stop_barrier,
            'profit_barrier': pos.profit_barrier,
            'quantity': pos.quantity,
            'notional_usd': pos.notional_usd,
            'barrier_pct': pos.barrier_pct,
            'unrealized_pnl_usd': unrealized_usd,
            'unrealized_pnl_pct': unrealized_pct,
            'elapsed_seconds': elapsed_sec,
            'lookahead_seconds': LOOKAHEAD_SECONDS,
            'mode': 'DRY-RUN — simulated fills, no live orders' if self.dry_run else 'LIVE',
        })
        return status

    def _log_trade(self, trade_result: Dict) -> None:
        logger.info(f"[TRADE EXIT] PnL: ${trade_result.get('net_pnl_usd', 0):+.2f}")

    async def _create_exchange(self):
        """Build exchange connection with TCP connector and timeouts."""
        exchange = ccxtpro.bybit({
            'enableRateLimit': True,
            'timeout': 10000,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
            },
        })
        if TESTNET:
            exchange.set_sandbox_mode(True)

        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver(), keepalive_timeout=30)
        exchange.session = aiohttp.ClientSession(connector=connector)
        
        return exchange

    async def _one_sec_clock(self):
        """Isolated background clock process."""
        while self.running:
            try:
                now = datetime.now()
                next_sec = now.replace(microsecond=0) + timedelta(seconds=1)
                sleep_sec = (next_sec - now).total_seconds()
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)

                if self.running:
                    await self._one_sec_tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in clock timer: {e}")
                await asyncio.sleep(1)

    async def _stream_loop(self, exchange):
        """Stream watcher with automatic exception detection for Wi-Fi drops."""
        last_data_time = datetime.now()

        while self.running:
            try:
                trades_task = asyncio.create_task(exchange.watch_trades(self.symbol))
                ob_task = asyncio.create_task(exchange.watch_order_book(self.symbol))

                done, pending = await asyncio.wait(
                    [trades_task, ob_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=20,  # 20-second timeout to detect silent network drops
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                if not done:
                    # Timeout triggered - check if network is stall
                    elapsed = (datetime.now() - last_data_time).total_seconds()
                    if elapsed > 30:
                        logger.warning(f"No stream data received for {elapsed:.0f}s. Reconnecting network...")
                        return True
                    continue

                for task in done:
                    result = task.result()
                    if task == trades_task:
                        for trade in result:
                            await self.on_trade(trade)
                        last_data_time = datetime.now()
                    elif task == ob_task:
                        await self.on_orderbook(result)

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
                logger.warning(f"Network / Disconnection detected in stream loop: {e}")
                return True
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(2)
                return True

        return False

    async def run_forever(self) -> None:
        """Main loop featuring exponential backoff reconnects."""
        self.running = True

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass

        one_sec_clock_task = asyncio.create_task(self._one_sec_clock())

        backoff_sec = 2
        max_backoff_sec = 60

        while self.running:
            exchange = None
            try:
                logger.info(f"Connecting to exchange WebSocket stream for {self.symbol}...")
                exchange = await self._create_exchange()
                
                # Reset backoff on successful connection
                backoff_sec = 2
                logger.info("Stream connected successfully. Listening to market data...")

                should_reconnect = await self._stream_loop(exchange)

                if not should_reconnect or not self.running:
                    break

            except (ccxt.NetworkError, aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
                logger.warning(f"Network error during connection setup: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in main run loop: {e}", exc_info=True)
            finally:
                if exchange is not None:
                    try:
                        await exchange.close()
                    except Exception:
                        pass

            if self.running:
                logger.info(f"Retrying connection in {backoff_sec} seconds...")
                await asyncio.sleep(backoff_sec)
                # Exponential backoff formula
                backoff_sec = min(backoff_sec * 2, max_backoff_sec)

        one_sec_clock_task.cancel()
        try:
            await one_sec_clock_task
        except (asyncio.CancelledError, Exception):
            pass

        self.running = False

    async def shutdown(self) -> None:
        logger.info("Shutting down executor cleanly...")
        self.running = False
        self.simulator.close()
        self.simulator.print_summary()
        await self.dashboard.stop()

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
def parse_args():
    """Parse command-line arguments for multi-setup execution."""
    parser = argparse.ArgumentParser(
        description="Intraday Momentum Composite — Live Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m main.executor --symbol BTC/USDT:USDT --direction short
  python -m main.executor --symbol ETH/USDT:USDT --direction long --confidence 0.85
  python -m main.executor    (uses config.py defaults)
        """,
    )
    parser.add_argument(
        '--symbol', type=str, default=None,
        help='Trading symbol (e.g. BTC/USDT:USDT). Default: from config.py',
    )
    parser.add_argument(
        '--direction', type=str, default=None, choices=['short', 'long'],
        help='Trade direction: short=fade, long=snapback. Default: from config.py',
    )
    parser.add_argument(
        '--confidence', type=float, default=None,
        help='P(WIN) threshold to fire a trade (0.0-1.0). Default: from config.py',
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Override config defaults with CLI args if provided
    global SYMBOL, TRADE_DIRECTION, CONFIDENCE_THRESHOLD
    if args.symbol is not None:
        SYMBOL = args.symbol
    if args.direction is not None:
        TRADE_DIRECTION = args.direction
    if args.confidence is not None:
        CONFIDENCE_THRESHOLD = args.confidence

    print("="*70)
    print(" INTRADAY MOMENTUM COMPOSITE — LIVE EXECUTOR (1-SECOND RESOLUTION)")
    print("="*70)
    print(f"Symbol    : {SYMBOL}")
    print(f"Testnet   : {TESTNET}")
    print(f"Dry Run   : {DRY_RUN}")
    print(f"Model     : {MODEL_PATH}")
    print(f"Direction : {TRADE_DIRECTION.upper()}")
    print(f"Threshold : P(WIN) >= {CONFIDENCE_THRESHOLD}")
    print(f"Filter    : RSI(2) < {RSI_FILTER_THRESHOLD}")
    print(f"Vol Filter: {MIN_BARRIER_PCT:.4%} <= barrier <= {MAX_BARRIER_PCT:.4%}")
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