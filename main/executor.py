"""
Live Strategy Executor — Main Orchestrator
============================================
The core loop that ties everything together:

    WebSocket (trades + orderbook)
        → LiveFeatureEngine (1-min bars)
            → RSI(14) filter (hunting zone)
                → LightGBM inference
                    → if P(LONG) >= threshold → enter SHORT
                        → PositionTracker monitors barriers
                            → exit on profit/stop/timeout

Architecture:
    - Single async event loop
    - ccxt.pro WebSocket for real-time trades + orderbook
    - 1-second micro-bar aggregation
    - 1-minute bar close triggers feature computation + model inference
    - Position monitoring runs on every tick while in position

Usage:
    python -m main.executor
"""

import sys
import os
import asyncio
import logging
import json
import signal
from datetime import datetime, timedelta
from typing import Optional, Dict
from collections import deque

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
    """
    Manages the LightGBM model lifecycle:
        - Load from disk
        - Train on startup from historical data
        - Inference on feature vectors
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[lgb.Booster] = None
        self.feature_columns: list = []
        self.is_trained: bool = False

    def load(self) -> bool:
        """Load a trained model from disk."""
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
        """Save model to disk."""
        if self.model is not None:
            try:
                self.model.save_model(self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a new LightGBM model on historical data.
        Mirrors the training setup from corrected_state.py.
        """
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
        """Get probability predictions. Returns [P(short), P(long)]."""
        if self.model is None:
            return np.array([[0.5, 0.5]])
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
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
        """
        Fetch historical 1-min OHLCV and build feature matrix + labels.
        Returns a DataFrame with features + 'target' column.
        """
        logger.info(f"Fetching {n_bars} bars of historical data for training...")

        # Fetch OHLCV
        df_ohlcv = self.connector.get_ohlcv(
            symbol=self.symbol,
            interval='1m',
            total_limit=n_bars,
        )

        if df_ohlcv.empty:
            logger.error("No historical data fetched!")
            return pd.DataFrame()

        # Rename columns to match research format
        df_ohlcv = df_ohlcv.rename(columns={
            'timestamp': 'datetime',
        }).set_index('datetime')

        # --- Compute features ---
        price = df_ohlcv['close'].values
        diffs = np.diff(price, prepend=price[0])

        # RSI features
        rsi_2 = get_wilders_rsi(diffs, 2)
        rsi_14 = get_wilders_rsi(diffs, 14)
        sma_14_vals = sma(rsi_14, 14)

        # Candle features
        candle_range = df_ohlcv['high'] - df_ohlcv['low']
        candle_type = abs(df_ohlcv['close'] - df_ohlcv['open']) / candle_range.replace(0, 1e-9)
        candle_size = candle_range / candle_range.rolling(14).mean().replace(0, 1e-9)

        # Build feature DataFrame
        df_features = pd.DataFrame({
            'rsi_2': rsi_2,
            'rsi_14': rsi_14,
            'sma_14': sma_14_vals,
            'candle_type': candle_type,
            'candle_size': candle_size,
            'close': price,
        }, index=df_ohlcv.index)

        # Drop NaN rows
        df_features = df_features.dropna()

        # --- Barrier labeling (simplified for training) ---
        # Use volatility-based barriers matching research
        ret_std = df_features['close'].pct_change().rolling(window=100).std()
        barrier_pct = ret_std * np.sqrt(LOOKAHEAD_SECONDS) * BARRIER_MULTIPLIER

        # Simple labeling: look ahead LOOKAHEAD_SECONDS bars (1-min bars)
        # For short strategy: label=1 if price goes UP (bad for short), 0 if DOWN (good for short)
        lookahead_bars = LOOKAHEAD_SECONDS // 60  # convert seconds to 1-min bars
        future_price = df_features['close'].shift(-lookahead_bars)
        df_features['target'] = (future_price > df_features['close']).astype(int)

        # Apply RSI filter (only train on oversold regime)
        df_features = df_features[df_features['rsi_14'] < RSI_FILTER_THRESHOLD].copy()

        # Drop close price from features
        feature_cols = [c for c in df_features.columns if c not in ['close', 'target']]
        df_features = df_features.dropna(subset=feature_cols + ['target'])

        logger.info(f"Training data: {len(df_features)} rows after filtering")
        return df_features

    def build_training_set(self, n_bars: int = N_HISTORICAL_BARS) -> tuple:
        """
        Build X, y training set from historical data.
        Returns (X, y, feature_columns).
        """
        df = self.fetch_training_data(n_bars)
        if df.empty:
            return None, None, None

        feature_cols = [c for c in df.columns if c not in ['close', 'target']]
        X = df[feature_cols].values
        y = df['target'].values

        return X, y, feature_cols


# ============================================================================
# Main Strategy Executor
# ============================================================================
class StrategyExecutor:
    """
    The main orchestrator. Runs the live strategy loop:

    1. Connect to WebSocket streams (trades + orderbook)
    2. Feed data into LiveFeatureEngine
    3. On each 1-min bar close:
        a. Compute feature vector
        b. Check RSI(14) filter (must be < 30 for hunting zone)
        c. Run model inference
        d. If P(LONG) >= threshold → enter SHORT
    4. Monitor active position on every tick
    5. Exit on barrier hit or timeout
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
        self.bar_count = 0
        self.last_bar_minute: Optional[int] = None
        self.last_bar_time: Optional[datetime] = None

        # For logging bar-level data
        self.bar_log: list = []

        # Rate limiting
        self._last_trade_time: Optional[datetime] = None

        logger.info(f"StrategyExecutor initialized | Symbol: {symbol} | Dry-run: {dry_run}")

    async def initialize(self) -> bool:
        """
        Initialize the executor:
        1. Load or train the model
        2. Warm up the feature engine with historical data
        """
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
                # Simulate trades at close price for warmup
                ts = int(row['timestamp'].timestamp() * 1000)
                self.feature_engine.on_trade(ts, row['close'], row['volume'] / 2, 'buy')
                self.feature_engine.on_trade(ts, row['close'], row['volume'] / 2, 'sell')
                # Simulate orderbook
                self.feature_engine.on_orderbook(ts, row['close'] * 0.999, 10.0, row['close'] * 1.001, 10.0)

                # Simulate bar close
                features = self.feature_engine.on_bar_close()
                if features:
                    self.bar_count += 1

            logger.info(f"Warmup complete: {self.bar_count} bars processed")

        return True

    async def on_trade(self, trade: Dict) -> None:
        """Process a single trade from WebSocket."""
        timestamp_ms = trade.get('timestamp', 0)
        price = float(trade.get('price', 0))
        amount = float(trade.get('amount', 0))
        side = trade.get('side', 'buy')

        self.feature_engine.on_trade(timestamp_ms, price, amount, side)

        # Update position monitoring via simulator
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

    async def on_bar_close(self) -> None:
        """
        Called when a 1-minute bar closes.
        This is the main strategy decision point.
        """
        features = self.feature_engine.on_bar_close()
        if not features:
            return

        self.bar_count += 1
        current_price = features.get('close', 0)
        rsi_14 = features.get('rsi_14', 100)

        # Log bar
        bar_record = {
            'time': datetime.now().isoformat(),
            'close': current_price,
            'rsi_14': rsi_14,
            'rsi_2': features.get('rsi_2', 0),
            'in_position': self.simulator.position.active,
        }

        # --- RSI Filter: Only trade in hunting zone (RSI(14) < 30) ---
        if rsi_14 >= RSI_FILTER_THRESHOLD:
            bar_record['action'] = 'SKIP (RSI too high)'
            self.bar_log.append(bar_record)
            logger.debug(f"Bar {self.bar_count}: RSI(14)={rsi_14:.1f} >= {RSI_FILTER_THRESHOLD} — skipping")
            return

        # --- Check if already in position ---
        if self.simulator.position.active:
            bar_record['action'] = 'IN_POSITION'
            self.bar_log.append(bar_record)
            return

        # --- Check rate limit ---
        if not self.simulator.can_trade():
            bar_record['action'] = 'SKIP (rate limit)'
            self.bar_log.append(bar_record)
            return

        # --- Model Inference ---
        feature_vector = self.feature_engine.extract_feature_vector(features)
        proba = self.model_mgr.predict_proba(feature_vector)[0]

        # proba[0] = P(SHORT), proba[1] = P(LONG)
        p_long = proba[1]
        p_short = proba[0]

        bar_record['p_long'] = float(p_long)
        bar_record['p_short'] = float(p_short)

        # --- Decision: Enter SHORT when P(LONG) >= threshold ---
        # This is the "fade" strategy: model says price will go UP,
        # but we're in oversold RSI, so we fade it by going SHORT
        if p_long >= CONFIDENCE_THRESHOLD:
            # Calculate position size via simulator
            qty = self.simulator.calculate_quantity(current_price)
            if qty < MIN_ORDER_SIZE:
                bar_record['action'] = f'SKIP (qty too small: {qty:.6f})'
                self.bar_log.append(bar_record)
                return

            # Simulate short entry with mid-price and slippage
            receipt = self.simulator.enter_short(current_price, qty)
            if receipt is not None:
                bar_record['action'] = f'ENTER SHORT (P_long={p_long:.3f})'
                bar_record['qty'] = qty
                bar_record['fill_price'] = receipt['fill_price']
                bar_record['slippage_bps'] = receipt['slippage_bps']
                bar_record['stop'] = receipt['stop_barrier']
                bar_record['profit'] = receipt['profit_barrier']
                logger.info(
                    f"TRADE SIGNAL | Bar {self.bar_count} | "
                    f"RSI(14)={rsi_14:.1f} | P(LONG)={p_long:.3f} >= {CONFIDENCE_THRESHOLD} | "
                    f"Entering SHORT | Mid: {current_price:.2f} | "
                    f"Fill: {receipt['fill_price']:.2f} | Slippage: {receipt['slippage_bps']:.2f}bps"
                )
            else:
                bar_record['action'] = 'FAILED (entry rejected)'
        else:
            bar_record['action'] = f'HOLD (P_long={p_long:.3f} < {CONFIDENCE_THRESHOLD})'

        self.bar_log.append(bar_record)

    def _log_trade(self, trade_result: Dict) -> None:
        """Log a completed trade to the trade log."""
        logger.info(
            f"TRADE CLOSED | {trade_result['reason']} | "
            f"PnL: {trade_result['pnl_pct']*100:.3f}% | "
            f"${trade_result['pnl_usd']:.2f}"
        )

        # Save to trade log file
        log_file = 'main/logs/trades.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                **trade_result,
                'entry_time': str(trade_result['entry_time']),
                'exit_time': str(trade_result['exit_time']),
            }) + '\n')

    async def run_forever(self) -> None:
        """
        Main loop: connect to WebSocket streams and process data.
        Uses ccxt.pro for real-time data.
        """
        import ccxt.pro as ccxtpro
        import aiohttp

        self.running = True

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Initialize exchange
        exchange = ccxtpro.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
            },
        })

        if TESTNET:
            exchange.set_sandbox_mode(True)

        # Use ThreadedResolver for Windows compatibility
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        exchange.session = aiohttp.ClientSession(connector=connector)

        try:
            await exchange.load_markets()
            logger.info(f"Connected to Bybit. Starting live stream for {self.symbol}...")

            # Track bar timing
            self.last_bar_minute = None

            # Main loop: watch trades and orderbook simultaneously
            while self.running:
                try:
                    # Watch trades
                    trades = await exchange.watch_trades(self.symbol)
                    for trade in trades:
                        await self.on_trade(trade)

                    # Watch orderbook (every ~100ms)
                    orderbook = await exchange.watch_order_book(self.symbol)
                    await self.on_orderbook(orderbook)

                    # Check for 1-minute bar close
                    now = datetime.now()
                    current_minute = now.minute

                    if self.last_bar_minute is None:
                        self.last_bar_minute = current_minute
                    elif current_minute != self.last_bar_minute:
                        # Bar just closed
                        self.last_bar_minute = current_minute
                        await self.on_bar_close()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Stream error: {e}", exc_info=True)
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            await exchange.close()
            self.running = False

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False

        # Close any open position via simulator
        self.simulator.close()

        # Print summary
        self.print_summary()

    def print_summary(self) -> None:
        """Print execution summary."""
        self.simulator.print_summary()
        print(f"Bars Processed  : {self.bar_count}")

        # Print last 10 bar actions
        print("\nLast 10 bars:")
        for bar in self.bar_log[-10:]:
            action = bar.get('action', 'N/A')
            rsi = bar.get('rsi_14', 0)
            close = bar.get('close', 0)
            print(f"  {bar['time'][:19]} | Close: {close:.2f} | RSI(14): {rsi:.1f} | {action}")


# ============================================================================
# Entry Point
# ============================================================================
async def main():
    """Main entry point."""
    print("="*60)
    print(" INTRADAY MOMENTUM COMPOSITE — LIVE EXECUTOR")
    print("="*60)
    print(f"Symbol  : {SYMBOL}")
    print(f"Testnet : {TESTNET}")
    print(f"Dry Run : {DRY_RUN}")
    print(f"Model   : {MODEL_PATH}")
    print("="*60)

    executor = StrategyExecutor(symbol=SYMBOL, dry_run=DRY_RUN)

    # Initialize (load/train model, warm up features)
    initialized = await executor.initialize()
    if not initialized:
        logger.error("Failed to initialize. Exiting.")
        return

    # Run forever
    await executor.run_forever()


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")