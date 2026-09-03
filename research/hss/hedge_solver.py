"""
hedge_solver.py

Takes your current option position(s) + a pool of hedge candidates
(underlying perp AND other options), reads their live Greeks off Deribit's
ticker feed, and solves for the cheapest combination of trades that
brings delta and gamma (the two Greeks that actually drive P&L off a move
in the underlying) back near zero. Vega is optional -- pass a vega_band if
you want IV-risk controlled too, but it's off by default since it's a
different kind of risk than "price moved and I'm exposed."

Why "near zero" via a band rather than exactly zero: with discrete
instruments (you can't buy 0.0001 of a perp contract in practice, and your
hedge options come in whatever strikes exist), exact zero is usually
either infeasible or requires wildly overtrading. A tight band (e.g.
delta_band=0.02) gets you effectively flat while the optimizer still
minimizes cost. Set the band to 0 if you want to force it to try for
exact zero anyway.

Two pieces:
    - portfolio_greeks(positions, state): aggregates your Greeks exposure
    - build_candidates(...): turns live ticker state into hedge options
    - solve_hedge(...): the LP that picks the combination
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from scipy.optimize import linprog


@dataclass
class HedgeCandidate:
    symbol: str
    delta: float
    gamma: float = 0.0
    vega: float = 0.0
    cost_per_unit: float = 0.0   # half the bid-ask spread: cost to cross and trade immediately
    max_qty: Optional[float] = None


@dataclass
class HedgeResult:
    trades: Dict[str, float]
    total_cost: float
    resulting_delta: float
    resulting_gamma: float
    feasible: bool
    message: str = ""


def portfolio_greeks(positions: dict, state: dict) -> Optional[dict]:
    """
    positions: {symbol: qty} for what you're market-making (signed: short = negative).
    state: {symbol: {"delta":..., "gamma":..., "vega":...}} from live ticker data.
    Returns None if any position's Greeks haven't arrived over the websocket yet --
    better to wait than to compute exposure off stale/missing data.
    """
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    for symbol, qty in positions.items():
        s = state.get(symbol)
        if s is None or s.get("delta") is None:
            return None
        total["delta"] += qty * (s["delta"] or 0.0)
        total["gamma"] += qty * (s["gamma"] or 0.0)
        total["vega"] += qty * (s["vega"] or 0.0)
    return total


def hedge_book_greeks(hedge_book: dict, underlying_symbol: str, state: dict) -> Optional[dict]:
    """
    Same as portfolio_greeks, but for a book that may include the underlying.
    The underlying never carries exchange-computed Greeks (it's not an
    option) -- its delta is 1.0 per unit by construction, gamma/vega are 0.
    portfolio_greeks() would incorrectly return None for it since state[underlying]
    has no "delta" field; this function handles that case explicitly instead.
    """
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    for symbol, qty in hedge_book.items():
        if symbol == underlying_symbol:
            total["delta"] += qty
            continue
        s = state.get(symbol)
        if s is None or s.get("delta") is None:
            return None
        total["delta"] += qty * (s["delta"] or 0.0)
        total["gamma"] += qty * (s["gamma"] or 0.0)
        total["vega"] += qty * (s["vega"] or 0.0)
    return total


def build_candidates(underlying_symbol: str, hedge_option_symbols: List[str], state: dict) -> List[HedgeCandidate]:
    """
    Builds the hedge pool: the underlying (delta=1, no gamma/vega) plus any
    other options you're streaming. Skips any candidate whose ticker data
    hasn't arrived yet rather than guessing its Greeks.
    """
    candidates = []

    u = state.get(underlying_symbol)
    if u and u.get("bid") is not None and u.get("ask") is not None:
        cost = (u["ask"] - u["bid"]) / 2.0
        candidates.append(HedgeCandidate(symbol=underlying_symbol, delta=1.0, gamma=0.0, vega=0.0, cost_per_unit=cost))

    for sym in hedge_option_symbols:
        s = state.get(sym)
        if s is None or s.get("delta") is None or s.get("bid") is None or s.get("ask") is None:
            continue
        cost = (s["ask"] - s["bid"]) / 2.0
        candidates.append(HedgeCandidate(
            symbol=sym, delta=s["delta"] or 0.0, gamma=s["gamma"] or 0.0, vega=s["vega"] or 0.0,
            cost_per_unit=cost,
        ))

    return candidates


def solve_hedge(
    portfolio_delta: float,
    portfolio_gamma: float,
    candidates: List[HedgeCandidate],
    delta_band: float = 0.02,
    gamma_band: Optional[float] = None,
    portfolio_vega: float = 0.0,
    vega_band: Optional[float] = None,
) -> HedgeResult:
    """
    Minimum-cost combination of trades across `candidates` (underlying +
    other options) that brings delta (and gamma, if gamma_band is set;
    vega, if vega_band is set) within its band of zero.
    """
    already_ok = abs(portfolio_delta) <= delta_band
    if gamma_band is not None:
        already_ok = already_ok and abs(portfolio_gamma) <= gamma_band
    if vega_band is not None:
        already_ok = already_ok and abs(portfolio_vega) <= vega_band

    if already_ok:
        return HedgeResult(trades={}, total_cost=0.0, resulting_delta=portfolio_delta,
                            resulting_gamma=portfolio_gamma, feasible=True,
                            message="Within band, no hedge needed.")

    n = len(candidates)
    if n == 0:
        return HedgeResult(trades={}, total_cost=0.0, resulting_delta=portfolio_delta,
                            resulting_gamma=portfolio_gamma, feasible=False,
                            message="No hedge candidates available (nothing warmed up yet?).")

    # Variables: [x_pos_1..n, x_neg_1..n], all >= 0. Signed trade_i = x_pos_i - x_neg_i.
    c = np.array([cand.cost_per_unit for cand in candidates] * 2)

    A_ub, b_ub = [], []

    def add_band_constraints(greek_values, portfolio_value, band):
        row_upper = list(greek_values) + [-g for g in greek_values]
        A_ub.append(row_upper)
        b_ub.append(band - portfolio_value)
        row_lower = [-g for g in greek_values] + list(greek_values)
        A_ub.append(row_lower)
        b_ub.append(band + portfolio_value)

    add_band_constraints([c.delta for c in candidates], portfolio_delta, delta_band)
    if gamma_band is not None:
        add_band_constraints([c.gamma for c in candidates], portfolio_gamma, gamma_band)
    if vega_band is not None:
        add_band_constraints([c.vega for c in candidates], portfolio_vega, vega_band)

    bounds = [(0, cand.max_qty) for cand in candidates]
    bounds = bounds + bounds

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return HedgeResult(trades={}, total_cost=0.0, resulting_delta=portfolio_delta,
                            resulting_gamma=portfolio_gamma, feasible=False,
                            message=f"Optimizer infeasible: {result.message}")

    x = result.x
    x_pos, x_neg = x[:n], x[n:]
    signed = x_pos - x_neg

    trades = {cand.symbol: float(qty) for cand, qty in zip(candidates, signed) if abs(qty) > 1e-9}

    resulting_delta = portfolio_delta + sum(c.delta * q for c, q in zip(candidates, signed))
    resulting_gamma = portfolio_gamma + sum(c.gamma * q for c, q in zip(candidates, signed))

    return HedgeResult(
        trades=trades, total_cost=float(result.fun),
        resulting_delta=float(resulting_delta), resulting_gamma=float(resulting_gamma),
        feasible=True, message="OK",
    )