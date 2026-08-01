"""
Professional Quant Dry-Run Execution Module
=============================================

CONTRACT: This module ALWAYS operates in dry-run mode.
    - Connects to REAL Bybit API for live market data (orderbook, ticker)
    - SIMULATES all fills using professional quant methodology
    - NEVER places a real order on the exchange
    - All PnL is mark-to-market, no look-ahead bias

Slippage Model:
    - Entry: short = sell at bid-adjusted price (mid - half-spread × multiplier)
    - Exit: cover = buy at ask-adjusted price (mid + half-spread × multiplier)
    - Slippage scales with order size relative to top-of-book depth
    - Calibrated from real-time orderbook feed (if available)

PnL Accounting (matching corrected_state.py research):
    - Gross PnL% = (entry - exit) / entry (before fees)
    - Fees = 2 × FEE_BPS (entry + exit, taker both ways)
    - Net PnL% = gross PnL% - fees
    - R-Multiple = gross PnL% / barrier_pct (clipped to [-1, 1] for non-timeout)
    - R-Multiple is the EXACT metric from the research backtest
"""

import sys
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import numpy as np

# Matplotlib setup for server/headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt
from main.config import *

logger = logging.getLogger("execution")


# ============================================================================
# Professional Slippage Model
# ============================================================================
class SlippageModel:
    """
    Realistic market-impact calibrated slippage for BTC/USDT perpetuals.

    For a market order of size Q on Bybit:
        - Entry (sell): fills against the bid side. Fill price = weighted avg of
          visible liquidity up to Q. Slippage = (mid - fill) / mid in bps.
        - Exit (buy): fills against the ask side.

    Calibration note: BTC/USDT:USDT on Bybit typically has:
        - Spread: 0.5-2 bps
        - Top-5 bid/ask depth: 10-50 contracts each
        - A $100 notional order (~0.0017 BTC) rarely moves the book
        - Conservative estimate: 0.3-1.0 bps slippage per side
    """

    def __init__(self, fallback_spread_bps: float = 1.5):
        self.fallback_half_spread = fallback_spread_bps / 2.0
        self._bids: List[Tuple[float, float]] = []
        self._asks: List[Tuple[float, float]] = []

    def update_book(self, bids: list, asks: list) -> None:
        """Update with real-time orderbook."""
        self._bids = [(float(p), float(s)) for p, s in (bids or [])[:10]]
        self._asks = [(float(p), float(s)) for p, s in (asks or [])[:10]]

    def _book_slippage(self, qty: float, side_entries: list, is_bid: bool) -> float:
        """
        Calculate slippage by walking the orderbook.

        side_entries: list of (price, size) from the relevant side
        is_bid: True if we're hitting the bid (selling), False if hitting ask (buying)

        Returns slippage in bps (positive = worse than mid).
        """
        if not side_entries:
            return self.fallback_half_spread

        cum_qty = 0.0
        weighted_price = 0.0
        mid = self._get_mid()
        if mid == 0:
            return self.fallback_half_spread

        for p, s in side_entries:
            fill = min(s, qty - cum_qty)
            weighted_price += fill * p
            cum_qty += fill
            if cum_qty >= qty:
                break

        if cum_qty == 0:
            return self.fallback_half_spread

        avg_fill = weighted_price / cum_qty
        bps = abs(avg_fill - mid) / mid * 10000.0
        return float(bps)

    def entry_slippage_bps(self, qty: float) -> float:
        """Slippage for short entry (sell = hit the bid)."""
        return self._book_slippage(qty, self._bids, is_bid=True)

    def exit_slippage_bps(self, qty: float) -> float:
        """Slippage for short exit (buy = hit the ask)."""
        return self._book_slippage(qty, self._asks, is_bid=False)

    def _get_mid(self) -> float:
        if self._bids and self._asks:
            return (self._bids[0][0] + self._asks[0][0]) / 2.0
        return 0.0

    def get_spread_bps(self) -> float:
        """Current best bid-ask spread in bps."""
        if self._bids and self._asks:
            spread = self._asks[0][0] - self._bids[0][0]
            mid = (self._bids[0][0] + self._asks[0][0]) / 2.0
            return spread / mid * 10000.0 if mid > 0 else 0.0
        return self.fallback_half_spread * 2.0


# ============================================================================
# Position Tracker — Research-Aligned PnL + R-Multiple
# ============================================================================
class PositionTracker:
    """
    Tracks a single short position with professional dry-run accounting.

    Entry simulation:
        fill_price = mid × (1 - slippage_bps / 10000)
    Exit simulation:
        fill_price = barrier_price × (1 + slippage_bps / 10000)
        (for timeout: uses current_mid instead of barrier)

    PnL (exact research methodology):
        R_SHORT = (entry - exit) / (barrier_pct × entry)
        For non-timeout exits: R is clipped to [-1.0, 1.0]
        Net PnL = R × barrier_pct × notional - fees
    """

    def __init__(self):
        # Position state
        self.active: bool = False
        self.side: Optional[str] = None
        self.entry_price: float = 0.0           # simulated fill price
        self.entry_mid: float = 0.0             # mid-price at entry
        self.entry_time: Optional[datetime] = None
        self.quantity: float = 0.0
        self.notional_usd: float = 0.0

        # Barriers
        self.barrier_pct: float = 0.0
        self.stop_barrier: float = 0.0
        self.profit_barrier: float = 0.0

        # Exit
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.reason: Optional[str] = None
        self.is_timeout: bool = False

        # PnL accounting
        self.gross_pnl_pct: float = 0.0
        self.net_pnl_pct: float = 0.0
        self.gross_pnl_usd: float = 0.0
        self.net_pnl_usd: float = 0.0
        self.fees_paid: float = 0.0
        self.r_multiple: float = 0.0
        self.entry_slippage_bps: float = 0.0
        self.exit_slippage_bps: float = 0.0

        # Slippage model (attached externally)
        self._slippage: Optional[SlippageModel] = None

        # Vol buffer
        self._price_buffer: List[float] = []

        # Intra-bar tracking (updated on every tick while in position)
        self._bar_high: float = 0.0
        self._bar_low: float = float('inf')

    def attach_slippage(self, model: SlippageModel) -> None:
        self._slippage = model

    def update_volatility(self, price: float) -> None:
        """Feed latest mid-price into rolling vol estimate."""
        self._price_buffer.append(price)
        if len(self._price_buffer) > BARRIER_VOL_WINDOW:
            self._price_buffer.pop(0)

    def get_barrier_pct(self) -> float:
        """
        Calculate barrier_pct matching corrected_state.py:
            barrier_pct = ret_std × sqrt(lookahead) × barrier_multiplier
        """
        if len(self._price_buffer) < 10:
            return 0.003
        prices = np.array(self._price_buffer[-100:])
        rets = np.diff(prices) / (prices[:-1] + 1e-9)
        ret_std = float(np.nanstd(rets))
        if ret_std <= 0:
            return 0.003
        return ret_std * np.sqrt(LOOKAHEAD_SECONDS) * BARRIER_MULTIPLIER

    def enter_short(self, mid_price: float, quantity: float,
                    timestamp: Optional[datetime] = None) -> Dict:
        """
        Simulate short entry.
        Fill price = mid × (1 - slippage_bps/10000) — we sell at the bid.
        """
        if self._slippage is not None:
            self.entry_slippage_bps = self._slippage.entry_slippage_bps(quantity)
        else:
            self.entry_slippage_bps = 0.75  # default: 0.75 bps slippage

        self.entry_price = mid_price * (1.0 - self.entry_slippage_bps / 10000.0)
        self.entry_mid = mid_price
        self.entry_time = timestamp or datetime.now()
        self.quantity = quantity
        self.notional_usd = quantity * self.entry_price
        self.side = 'short'
        self.active = True

        # Vol-scaled barriers (symmetric)
        self.barrier_pct = self.get_barrier_pct()
        self.stop_barrier = self.entry_price * (1.0 + self.barrier_pct)
        self.profit_barrier = self.entry_price * (1.0 - self.barrier_pct)

        # Reset intra-bar extremes
        self._bar_high = mid_price
        self._bar_low = mid_price

        receipt = {
            'side': 'short',
            'action': 'ENTER',
            'mid_price': mid_price,
            'fill_price': self.entry_price,
            'slippage_bps': self.entry_slippage_bps,
            'quantity': quantity,
            'notional_usd': self.notional_usd,
            'barrier_pct': self.barrier_pct,
            'stop_barrier': self.stop_barrier,
            'profit_barrier': self.profit_barrier,
            'timestamp': str(self.entry_time),
        }

        logger.info(
            f"ENTER SHORT | Mid: {mid_price:.2f} | Fill: {self.entry_price:.2f} | "
            f"Slippage: {self.entry_slippage_bps:.2f}bps | Qty: {quantity:.4f} | "
            f"Stop: {self.stop_barrier:.2f} | Profit: {self.profit_barrier:.2f}"
        )

        return receipt

    def update_tick(self, mid_price: float) -> None:
        """
        Called on every tick while in position.
        Tracks intra-bar high/low for barrier monitoring.
        """
        if not self.active:
            return
        self._bar_high = max(self._bar_high, mid_price)
        self._bar_low = min(self._bar_low, mid_price)

    def check_exit(self, current_mid: float,
                   timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """
        Check if position should be exited.
        Barrier logic matches corrected_state.py exactly.

        Returns exit result dict if triggered, None otherwise.
        """
        if not self.active:
            return None

        ts = timestamp or datetime.now()
        high = self._bar_high
        low = self._bar_low

        # Barrier checks (short: stop above, profit below)
        hit_stop = high >= self.stop_barrier
        hit_profit = low <= self.profit_barrier

        # Timeout
        timeout = False
        if self.entry_time is not None:
            elapsed = (ts - self.entry_time).total_seconds()
            timeout = elapsed >= LOOKAHEAD_SECONDS

        if not (hit_stop or hit_profit):
            if timeout:
                return self._exit('timeout', current_mid, ts)
            return None

        # First touch wins
        if hit_stop and hit_profit:
            stop_dist = self.stop_barrier - self.entry_price
            profit_dist = self.entry_price - self.profit_barrier
            if profit_dist <= stop_dist:
                return self._exit('take_profit', self.profit_barrier, ts)
            else:
                return self._exit('stop_loss', self.stop_barrier, ts)

        if hit_stop:
            # consider switching to high/low to account market order slippage.
            return self._exit('stop_loss', self.stop_barrier, ts)
        if hit_profit:
            return self._exit('take_profit', self.profit_barrier, ts)

        return None

    def _exit(self, reason: str, theoretical_exit: float, timestamp: datetime) -> Dict:
        """
        Simulate exit with slippage and compute full PnL accounting.

        Exit (covering short) = buy at ask = theoretical × (1 + slippage/10000).
        For barrier exits: theoretical_exit = barrier level.
        For timeout: theoretical_exit = current_mid.

        R-Multiple (matching corrected_state.py Step 4):
            R_SHORT = (entry - exit) / (barrier_pct × entry)
            Clipped to [-1, 1] for non-timeout (barrier-touch) exits.
        """
        self.active = False
        self.reason = reason
        self.exit_time = timestamp
        self.is_timeout = (reason == 'timeout')

        # Exit slippage
        if self._slippage is not None:
            self.exit_slippage_bps = self._slippage.exit_slippage_bps(self.quantity)
        else:
            self.exit_slippage_bps = 0.75

        # Exit fill: covering short = buy at ask = theoretical + slippage
        self.exit_price = theoretical_exit * (1.0 + self.exit_slippage_bps / 10000.0)

        # ================================================
        # Gross PnL (before fees)
        # Short profitable if entry > exit (price went down)
        # ================================================
        self.gross_pnl_pct = (self.entry_price - self.exit_price) / self.entry_price
        self.gross_pnl_usd = self.gross_pnl_pct * self.notional_usd

        # ================================================
        # Fees: taker on both entry and exit
        # ================================================
        entry_fee = self.notional_usd * FEE_BPS / 10000.0
        exit_notional = self.quantity * self.exit_price
        exit_fee = exit_notional * FEE_BPS / 10000.0
        self.fees_paid = entry_fee + exit_fee
        total_fee_pct = self.fees_paid / self.notional_usd

        # ================================================
        # Net PnL
        # ================================================
        self.net_pnl_pct = self.gross_pnl_pct - total_fee_pct
        self.net_pnl_usd = self.gross_pnl_usd - self.fees_paid

        # ================================================
        # R-Multiple (matching corrected_state.py)
        # R = gross PnL% / barrier_pct (scaled return per unit risk)
        # Clipped to [-1, 1] for barrier-touch exits as in research
        # ================================================
        bp = self.barrier_pct if self.barrier_pct > 1e-9 else 0.003
        
        # 1. Calculate Gross R and clip to [-1, 1] for barrier touches
        base_gross_r = (self.entry_price - theoretical_exit) / self.entry_price / bp
        if not self.is_timeout:
            base_gross_r = float(np.clip(base_gross_r, -1.0, 1.0))

        slippage_r_penalty = (self.exit_slippage_bps / 10000.0) / bp
        gross_r = base_gross_r - slippage_r_penalty
            
        # 2. Calculate the fee cost in terms of R-units
        fee_in_r = total_fee_pct / bp
        
        # 3. Final Net R-Multiple
        self.r_multiple = gross_r - fee_in_r

        result = {
            'side': 'short',
            'reason': reason,
            'entry_price': self.entry_price,
            'entry_mid': self.entry_mid,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'notional_usd': self.notional_usd,
            'entry_slippage_bps': self.entry_slippage_bps,
            'exit_slippage_bps': self.exit_slippage_bps,
            'gross_pnl_pct': self.gross_pnl_pct,
            'gross_pnl_usd': self.gross_pnl_usd,
            'fees_paid': self.fees_paid,
            'net_pnl_pct': self.net_pnl_pct,
            'net_pnl_usd': self.net_pnl_usd,
            'r_multiple': self.r_multiple,
            'barrier_pct': self.barrier_pct,
            'is_timeout': self.is_timeout,
            'entry_time': str(self.entry_time),
            'exit_time': str(self.exit_time),
            'duration_seconds': (timestamp - self.entry_time).total_seconds() if self.entry_time else 0,
        }

        logger.info(
            f"EXIT SHORT | {reason:12s} | "
            f"Entry: {self.entry_price:.2f} | Exit: {self.exit_price:.2f} | "
            f"Gross: {self.gross_pnl_pct*100:+.3f}% | "
            f"Fees: ${self.fees_paid:.4f} | "
            f"Net: {self.net_pnl_pct*100:+.3f}% | "
            f"R: {self.r_multiple:+.4f} | ${self.net_pnl_usd:+.2f}"
        )

        return result

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# QuantSimulator — Professional Dry-Run Engine
# ============================================================================
class QuantSimulator:
    """
    Professional paper-trading / dry-run execution engine with automated PnL charting
    and resilience against network interruptions.
    """

    def __init__(self, api_key: str = API_KEY, api_secret: str = API_SECRET, quick_init: bool = False):
        self.exchange = None
        self._quick_init = quick_init

        if not quick_init:
            exchange_config = {
                'apiKey': '',
                'secret': '',
                'enableRateLimit': True,
                'timeout': 10000,  # 10s timeout to prevent hanging on network drops
                'options': {
                    'defaultType': 'linear',
                    'adjustForTimeDifference': True,
                },
            }
            self.exchange = ccxt.bybit(exchange_config)

            try:
                self.exchange.load_markets()
                logger.info("QuantSimulator: Connected to real Bybit API for market data.")
            except Exception as e:
                logger.warning(f"QuantSimulator: Could not load markets initially ({e}). Will retry on demand.")
        else:
            logger.info("QuantSimulator: Quick init (no exchange connection).")

        from main.execution import SlippageModel, PositionTracker
        self.slippage_model = SlippageModel()
        self.position = PositionTracker()
        self.position.attach_slippage(self.slippage_model)

        self.trades: List[Dict] = []
        self.total_gross_pnl: float = 0.0
        self.total_net_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.total_r_multiple: float = 0.0
        self.winning_trades: int = 0
        self.losing_trades: int = 0

        self._recent_entries: List[datetime] = []
        self._journal_file = 'main/logs/trade_journal.jsonl'
        os.makedirs(os.path.dirname(self._journal_file), exist_ok=True)

    def get_current_mid(self) -> float:
        """Fetch orderbook, update slippage model, and return current mid-price."""
        if self.exchange is None:
            return self.position.entry_mid if self.position.entry_mid > 0 else 0.0

        try:
            # 1. Fetch the top 10 levels of the orderbook instead of just the ticker
            ob = self.exchange.fetch_order_book(SYMBOL, limit=10)
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])

            # 2. Feed this live liquidity data into your slippage model
            self.slippage_model.update_book(bids, asks)

            # 3. Calculate and return the mid-price from the top of the book
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                return (best_bid + best_ask) / 2.0
            
            # Fallback if orderbook is completely empty (extremely rare)
            return self.position.entry_mid if self.position.entry_mid > 0 else 0.0

        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            logger.warning(f"Network error fetching orderbook: {e}. Falling back to last cached price.")
            return self.position.entry_mid if self.position.entry_mid > 0 else 0.0
        except Exception as e:
            logger.error(f"Unexpected error fetching orderbook: {e}")
            return self.position.entry_mid if self.position.entry_mid > 0 else 0.0

    def calculate_quantity(self, price: float, usdt_amount: float = MAX_POSITION_USD) -> float:
        """Calculate contract quantity safely."""
        if price <= 0:
            return MIN_ORDER_SIZE
        qty = usdt_amount / price
        try:
            market = self.exchange.market(SYMBOL) if self.exchange else {}
            prec = int(market.get('precision', {}).get('amount', 4))
            qty = round(qty, prec)
        except Exception:
            qty = round(qty, 4)
        return max(qty, MIN_ORDER_SIZE)

    def can_trade(self) -> bool:
        now = datetime.now()
        self._recent_entries = [t for t in self._recent_entries if now - t < timedelta(hours=1)]
        return len(self._recent_entries) < MAX_TRADES_PER_HOUR and not self.position.active

    def enter_short(self, mid_price: float, quantity: float) -> Optional[Dict]:
        if not self.can_trade():
            logger.warning(f"Trade rejected: {len(self._recent_entries)}/{MAX_TRADES_PER_HOUR} in last hour")
            return None

        receipt = self.position.enter_short(mid_price, quantity)
        self._recent_entries.append(datetime.now())
        return receipt

    def update_tick(self, mid_price: float) -> Optional[Dict]:
        self.position.update_tick(mid_price)
        result = self.position.check_exit(current_mid=mid_price)
        
        if result:
            # Save to JSONL and update the internal trade lists
            self._journal_trade(result)
            self.trades.append(result)
            self.total_gross_pnl += result['gross_pnl_usd']
            self.total_net_pnl += result['net_pnl_usd']
            self.total_fees += result['fees_paid']
            self.total_r_multiple += result['r_multiple']
            
            if result['net_pnl_usd'] > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
                
            # Auto-update chart on every natural trade exit
            self.plot_pnl()
            
        return result

    def force_exit(self, reason: str = 'manual', exit_price: Optional[float] = None) -> Optional[Dict]:
        if not self.position.active:
            return None

        if exit_price is None:
            exit_price = self.get_current_mid()

        result = self.position.check_exit(current_mid=exit_price)
        if result is None:
            result = self.position._exit(reason, exit_price, datetime.now())

        if result:
            self._journal_trade(result)
            self.trades.append(result)
            self.total_gross_pnl += result['gross_pnl_usd']
            self.total_net_pnl += result['net_pnl_usd']
            self.total_fees += result['fees_paid']
            self.total_r_multiple += result['r_multiple']
            if result['net_pnl_usd'] > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            # Auto-update chart on every trade exit
            self.plot_pnl()

        return result

    def _journal_trade(self, result: Dict) -> None:
        try:
            with open(self._journal_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
        except Exception as e:
            logger.error(f"Failed to journal trade: {e}")

    # =========================================================================
    # PnL Charting Feature
    # =========================================================================
    def plot_pnl(self, save_path: str = "main/logs/pnl_chart.png") -> None:
        """Generates and saves an equity curve chart for Net RR and Gross RR."""
        if not self.trades:
            logger.info("No completed trades to plot yet.")
            return

        try:
            trade_numbers = list(range(1, len(self.trades) + 1))
            
            # Calculate R-Multiples on the fly from the saved percentages
            net_r = [(t['net_pnl_pct'] / t['barrier_pct']) for t in self.trades]
            gross_r = [(t['gross_pnl_pct'] / t['barrier_pct']) for t in self.trades]
            
            cum_net_r = np.cumsum(net_r)
            cum_gross_r = np.cumsum(gross_r)

            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            
            # Plot Net and Gross RR
            ax.plot(trade_numbers, cum_net_r, label="Net R-Multiple", color="#2ecc71", linewidth=2.5, marker='o', markersize=4)
            ax.plot(trade_numbers, cum_gross_r, label="Gross R-Multiple", color="#3498db", linestyle="--", alpha=0.7)
            ax.axhline(0, color="#7f8c8d", linestyle=":", alpha=0.7)

            ax.set_title(f"Cumulative R-Multiple Performance ({SYMBOL})", fontsize=12, fontweight='bold')
            ax.set_xlabel("Trade Count", fontsize=10)
            ax.set_ylabel("Cumulative RR", fontsize=10)
            ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Highlight Final Score
            final_rr = cum_net_r[-1]
            color_text = "green" if final_rr >= 0 else "red"
            ax.text(0.98, 0.05, f"Total Net RR: {final_rr:+.2f}R", transform=ax.transAxes,
                    fontsize=10, fontweight='bold', color=color_text, ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
            logger.info(f"PnL chart successfully generated in RR units: {save_path}")
        except Exception as e:
            logger.error(f"Error generating PnL chart: {e}")

    def print_summary(self) -> None:
        s = self.get_status()
        print("\n" + "=" * 65)
        print(" QUANT SIMULATOR — PERFORMANCE SUMMARY")
        print("=" * 65)
        print(f"  Total Trades      : {s['total_trades']}")
        print(f"  Win / Loss        : {s['winning_trades']} / {s['losing_trades']}")
        print(f"  Win Rate          : {s['win_rate']:.2%}")
        print(f"  Gross PnL         : ${s['total_gross_pnl_usd']:+.2f}")
        print(f"  Total Fees        : ${s['total_fees_usd']:.2f}")
        print(f"  Net PnL           : ${s['total_net_pnl_usd']:+.2f}")
        print(f"  Total R-Multiple  : {s['total_r_multiple']:+.4f}")
        print("=" * 65)
        self.plot_pnl()

    def get_status(self) -> Dict:
        return {
            'in_position': self.position.active,
            'entry_price': self.position.entry_price,
            'total_trades': len(self.trades),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_gross_pnl_usd': self.total_gross_pnl,
            'total_net_pnl_usd': self.total_net_pnl,
            'total_fees_usd': self.total_fees,
            'total_r_multiple': self.total_r_multiple,
            'win_rate': self.winning_trades / len(self.trades) if self.trades else 0.0,
        }

    def close(self) -> None:
        if self.position.active:
            mid = self.get_current_mid()
            self.force_exit('shutdown', mid)
        self.plot_pnl()
def main():
    """Entry point for dry-run execution test."""
    # Basic logging setup if executed directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logger.info("Initializing QuantSimulator dry-run...")
    
    # Create simulator instance
    simulator = QuantSimulator()
    
    # Fetch live mid-price from Bybit
    mid_price = simulator.get_current_mid()
    
    if mid_price > 0:
        logger.info(f"Live mid-price for {SYMBOL}: ${mid_price:.2f}")
        
        # Calculate sample quantity based on configured max position USD
        qty = simulator.calculate_quantity(mid_price)
        logger.info(f"Sample position size for ${MAX_POSITION_USD}: {qty} BTC")
        
        # Output current simulator status summary
        simulator.print_summary()
    else:
        logger.error("Could not retrieve current market price. Check network or API configuration.")


if __name__ == "__main__":
    main()