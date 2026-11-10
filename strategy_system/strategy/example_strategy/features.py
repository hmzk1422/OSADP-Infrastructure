from __future__ import annotations

from collections import deque

WINDOW_SIZE = 5


class ExampleFeatures:
    def __init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=WINDOW_SIZE)

    def compute(self, price: float) -> float:
        self._prices.append(price)
        return price - sum(self._prices) / len(self._prices)
