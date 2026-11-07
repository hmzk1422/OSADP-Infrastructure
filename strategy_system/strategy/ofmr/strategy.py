"""
OFMR production trading strategy.
Loads the trader.txt LightGBM text model, computes features via features.py,
runs inference, and emits trade decisions (EnterLong / Hold / Exit).
Mirrors uets/live_strategy/ofmr/strategy.rs exactly.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

_OFMR_DIR = os.path.dirname(os.path.abspath(__file__))
_STRATEGY_DIR = os.path.abspath(os.path.join(_OFMR_DIR, ".."))
for _p in (_STRATEGY_DIR, _OFMR_DIR):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from config import (  # noqa: E402
    BARRIER_MULTIPLIER,
    CONFIDENCE_THRESHOLD,
    FEE_BPS,
    LOCKOUT_SECONDS,
    MAX_POSITION_USD,
    MAX_TRADES_PER_HOUR,
    MIN_ORDER_SIZE,
    RR_RATIO,
    STOP_LOSS_BPS,
    TAKE_PROFIT_BPS,
    TIMEFRAME_BARRIER,
    TRADE_DIRECTION,
)
from features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    compute_features,
    compute_features_from_matched,
)

try:
    import lightgbm as lgb  # noqa: E402
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

FEATURE_COLUMNS = [f"Column_{i}" for i in range(N_FEATURES)]

DEFAULT_MODEL_PATH = os.path.join(
    _OFMR_DIR, "models", "trader.txt"
)
if not os.path.exists(DEFAULT_MODEL_PATH):
    _ROOT = os.path.abspath(os.path.join(_OFMR_DIR, "..", "..", ".."))
    alt = os.path.join(_ROOT, "uets", "models", "ofmr", "trader.txt")
    if os.path.exists(alt):
        DEFAULT_MODEL_PATH = alt


class Position(Enum):
    FLAT = "flat"
    LONG = "long"


class TradeDecision(Enum):
    HOLD = "hold"
    ENTER_LONG = "enter_long"
    EXIT = "exit"


@dataclass
class Trade:
    entry_time: str
    entry_price: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    return_r: Optional[float] = None
    side: str = "long"


@dataclass
class OfmrStrategy:
    model_path: str = DEFAULT_MODEL_PATH
    threshold: float = CONFIDENCE_THRESHOLD
    trade_direction: str = TRADE_DIRECTION
    barrier_multiplier: float = BARRIER_MULTIPLIER
    rr_ratio: float = RR_RATIO
    lockout_seconds: float = LOCKOUT_SECONDS
    max_position_usd: float = MAX_POSITION_USD
    min_order_size: float = MIN_ORDER_SIZE
    stop_loss_bps: float = STOP_LOSS_BPS
    take_profit_bps: float = TAKE_PROFIT_BPS
    max_trades_per_hour: int = MAX_TRADES_PER_HOUR
    timeframe_barrier_secs: int = TIMEFRAME_BARRIER

    _booster: Optional[object] = field(default=None, init=False, repr=False)
    _position: Position = field(default=Position.FLAT, init=False)
    _entry_price: float = 0.0
    _entry_time: Optional[str] = field(default=None, init=False, repr=False)
    _last_exit_time: float = 0.0
    _trades_this_hour: int = 0
    _current_hour: Optional[int] = field(default=None, init=False, repr=False)
    _trade_log: list[Trade] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if _HAS_LGB:
            self._booster = lgb.Booster(model_file=self.model_path)
        else:
            self._booster = _parse_lightgbm_txt(self.model_path)

    def _predict_proba(self, features: np.ndarray) -> float:
        if features.shape[0] != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[0]}")
        if _HAS_LGB and self._booster is not None and hasattr(self._booster, "predict"):
            raw = self._booster.predict(
                features.reshape(1, -1),
                num_iteration=self._booster.current_iteration(),
                raw_score=True,
            )
            score = float(raw[0])
            if score >= 0.0:
                return 1.0 / (1.0 + math.exp(-score))
            exp_score = math.exp(score)
            return exp_score / (1.0 + exp_score)
        return _predict_proba_python(self._booster, features)  # type: ignore[arg-type]

    def _barrier_fractions(self, bar_vol_std: Optional[float]) -> tuple[float, float]:
        vol = bar_vol_std
        if vol is None or not math.isfinite(vol) or vol <= 0.0:
            vol = 0.001
        vol = max(vol, 0.001)
        stop = vol * self.barrier_multiplier
        target = stop * self.rr_ratio
        return stop, target

    def _in_hour(self, timestamp_ms: float) -> bool:
        if self._current_hour is None:
            return True
        return int(timestamp_ms // 3_600_000) == self._current_hour

    def update(
        self,
        features: dict[str, float] | np.ndarray,
        timestamp_ms: float,
        price: float,
        bar_vol_std: Optional[float] = None,
    ) -> tuple[TradeDecision, dict]:
        if isinstance(features, dict):
            x = np.array([features[name] for name in FEATURE_COLUMNS], dtype=float)
        else:
            x = np.asarray(features, dtype=float)

        if np.any(~np.isfinite(x)):
            return TradeDecision.HOLD, {
                "probability": 0.0,
                "position": self._position.value,
                "reason": "non_finite_features",
            }

        proba = self._predict_proba(x)

        now_hour = int(timestamp_ms // 3_600_000)
        if self._current_hour is not None and now_hour != self._current_hour:
            self._trades_this_hour = 0
            self._current_hour = now_hour
        elif self._current_hour is None:
            self._current_hour = now_hour

        info = {
            "probability": proba,
            "position": self._position.value,
            "timestamp_ms": timestamp_ms,
            "price": price,
        }

        if self._position == Position.FLAT:
            if timestamp_ms - self._last_exit_time < self.lockout_seconds * 1000.0:
                info["reason"] = "lockout"
                return TradeDecision.HOLD, info
            if self._trades_this_hour >= self.max_trades_per_hour:
                info["reason"] = "max_trades_per_hour"
                return TradeDecision.HOLD, info
            if self.trade_direction == "long" and proba >= self.threshold:
                stop_frac, target_frac = self._barrier_fractions(bar_vol_std)
                stop_frac = max(stop_frac, self.stop_loss_bps / 10000.0)
                self._position = Position.LONG
                self._entry_price = price
                self._entry_time = str(timestamp_ms)
                self._trades_this_hour += 1
                info.update({
                    "decision": TradeDecision.ENTER_LONG.value,
                    "stop_frac": stop_frac,
                    "target_frac": target_frac,
                    "stop_price": price * (1.0 - stop_frac),
                    "target_price": price * (1.0 + target_frac * self.rr_ratio),
                })
                return TradeDecision.ENTER_LONG, info
            info["reason"] = "below_threshold"
            return TradeDecision.HOLD, info

        stop_frac, target_frac = self._barrier_fractions(bar_vol_std)
        stop_price = self._entry_price * (1.0 - stop_frac)
        target_price = self._entry_price * (1.0 + target_frac * self.rr_ratio)
        fixed_stop = self._entry_price * (1.0 - self.stop_loss_bps / 10000.0)
        fixed_target = self._entry_price * (1.0 + self.take_profit_bps / 10000.0)
        stop_price = max(stop_price, fixed_stop)
        target_price = min(target_price, fixed_target)
        hit_stop = price <= stop_price
        hit_target = price >= target_price
        if hit_stop or hit_target:
            ret_r = (price - self._entry_price) / self._entry_price / stop_frac if stop_frac > 0 else 0.0
            trade = Trade(
                entry_time=self._entry_time,
                entry_price=self._entry_price,
                exit_time=str(timestamp_ms),
                exit_price=price,
                return_r=ret_r,
                side="long",
            )
            self._trade_log.append(trade)
            self._position = Position.FLAT
            self._last_exit_time = timestamp_ms
            self._entry_price = 0.0
            self._entry_time = None
            info.update({
                "decision": TradeDecision.EXIT.value,
                "return_r": ret_r,
                "hit_stop": hit_stop,
                "hit_target": hit_target,
                "stop_price": stop_price,
                "target_price": target_price,
            })
            return TradeDecision.EXIT, info

        if self._entry_time is not None:
            entry_ms = float(self._entry_time)
            if timestamp_ms - entry_ms > self.timeframe_barrier_secs * 1000.0:
                ret_r = (price - self._entry_price) / self._entry_price / stop_frac if stop_frac > 0 else 0.0
                trade = Trade(
                    entry_time=self._entry_time,
                    entry_price=self._entry_price,
                    exit_time=str(timestamp_ms),
                    exit_price=price,
                    return_r=ret_r,
                    side="long",
                )
                self._trade_log.append(trade)
                self._position = Position.FLAT
                self._last_exit_time = timestamp_ms
                self._entry_price = 0.0
                self._entry_time = None
                info.update({
                    "decision": TradeDecision.EXIT.value,
                    "return_r": ret_r,
                    "reason": "timeout",
                })
                return TradeDecision.EXIT, info

        info["reason"] = "holding"
        return TradeDecision.HOLD, info

    def reset(self) -> None:
        self._position = Position.FLAT
        self._entry_price = 0.0
        self._entry_time = None
        self._last_exit_time = 0.0
        self._trades_this_hour = 0
        self._current_hour = None
        self._trade_log.clear()

    @property
    def position(self) -> Position:
        return self._position

    @property
    def trade_log(self) -> list[Trade]:
        return list(self._trade_log)


def _parse_lightgbm_txt(path: str) -> dict:
    trees: list[dict] = []
    sigmoid_scale = 1.0
    objective = ""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("objective="):
            objective = line.split("=", 1)[1].strip()
            for token in objective.split()[1:]:
                if token.startswith("sigmoid:"):
                    sigmoid_scale = float(token.split(":", 1)[1])
        elif line.startswith("Tree="):
            tree = _parse_tree_block(lines, i)
            trees.append(tree)
            i = tree["_next_i"] + 1
            continue
        i += 1
    return {"trees": trees, "sigmoid_scale": sigmoid_scale, "objective": objective}


def _parse_tree_block(lines: list[str], start: int) -> dict:
    tree: dict = {"nodes": [], "leaf_values": [], "_next_i": start}
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Tree=") and i > start:
            break
        if line.startswith("split_feature="):
            tree["split_feature"] = [int(x) for x in line.split("=", 1)[1].split()]
        elif line.startswith("threshold="):
            tree["threshold"] = [float(x) for x in line.split("=", 1)[1].split()]
        elif line.startswith("left_child="):
            tree["left_child"] = [int(x) for x in line.split("=", 1)[1].split()]
        elif line.startswith("right_child="):
            tree["right_child"] = [int(x) for x in line.split("=", 1)[1].split()]
        elif line.startswith("leaf_value="):
            tree["leaf_values"] = [float(x) for x in line.split("=", 1)[1].split()]
        elif line.startswith("shrinkage="):
            tree["shrinkage"] = float(line.split("=", 1)[1].strip())
        i += 1
    tree["_next_i"] = i - 1
    return tree


def _predict_proba_python(model: dict, features: np.ndarray) -> float:
    trees = model.get("trees", [])
    if not trees:
        return 0.5
    raw_score = 0.0
    for tree in trees:
        split_features = tree.get("split_feature", [])
        thresholds = tree.get("threshold", [])
        left_children = tree.get("left_child", [])
        right_children = tree.get("right_child", [])
        leaf_values = tree.get("leaf_values", [])
        if not split_features:
            raw_score += leaf_values[0] if leaf_values else 0.0
            continue
        node_idx = 0
        while node_idx >= 0:
            sf = split_features[node_idx]
            thr = thresholds[node_idx]
            if features[sf] <= thr:
                node_idx = left_children[node_idx]
            else:
                node_idx = right_children[node_idx]
        leaf_idx = -node_idx - 1
        if 0 <= leaf_idx < len(leaf_values):
            raw_score += leaf_values[leaf_idx]
    raw_score *= float(model.get("sigmoid_scale", 1.0))
    if raw_score >= 0.0:
        return 1.0 / (1.0 + math.exp(-raw_score))
    exp_score = math.exp(raw_score)
    return exp_score / (1.0 + exp_score)


def run_strategy_on_matched(
    matched_root: str,
    symbol: str = "BTCUSDT",
    tag: Optional[str] = None,
    model_path: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> list[dict]:
    """Run the full OFMR strategy on matched data and return trade records."""
    feat_result = compute_features_from_matched(matched_root, symbol=symbol, tag=tag)
    df = feat_result.frame
    strat = OfmrStrategy(model_path=model_path or DEFAULT_MODEL_PATH)
    records: list[dict] = []
    for row in df.iter_rows(named=True):
        features = {name: row[name] for name in FEATURE_COLUMNS}
        ts_val = row["datetime"]
        ts_ms = ts_val.timestamp() * 1000.0 if hasattr(ts_val, "timestamp") else float(ts_val)
        price = row["price"]
        bvs = row["bar_vol_std"]
        decision, info = strat.update(features, ts_ms, price, bar_vol_std=bvs)
        if decision != TradeDecision.HOLD:
            records.append({
                "datetime": row["datetime"],
                "price": price,
                "decision": decision.value,
                "probability": info.get("probability", 0.0),
                "position": info.get("position", "flat"),
                "reason": info.get("reason", ""),
                "return_r": info.get("return_r"),
                **{name: row[name] for name in FEATURE_COLUMNS},
            })
    if output_csv:
        import pandas as pd
        pd.DataFrame(records).to_csv(output_csv, index=False)
    return records


__all__ = [
    "OfmrStrategy",
    "TradeDecision",
    "Position",
    "Trade",
    "FEATURE_COLUMNS",
    "N_FEATURES",
    "run_strategy_on_matched",
    "DEFAULT_MODEL_PATH",
]