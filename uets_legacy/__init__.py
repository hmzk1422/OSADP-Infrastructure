"""
Intraday Momentum Composite — Live Execution Module
=====================================================
Production-grade strategy execution on Bybit using:
    - Real-time WebSocket trades + orderbook → 1-min bar features
    - LightGBM model for P(LONG) prediction
    - RSI(14) oversold filter (<30) as hunting zone
    - Short-fade strategy: enter SHORT when model confidence is high
    - Volatility-scaled barrier exits (profit/stop/timeout)

Modules:
    config.py      — All strategy parameters (mirrors research)
    features.py    — Streaming feature engine (1s micro + 1min bars)
    execution.py   — Order management & position tracking
    executor.py    — Main orchestrator (async event loop)
"""

from .config import *
from .features import LiveFeatureEngine
from .execution import QuantSimulator, PositionTracker