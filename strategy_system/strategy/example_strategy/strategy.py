from __future__ import annotations


class ExampleStrategy:
    def __init__(self) -> None:
        self.position = 0

    def decide(self, dz: float) -> tuple[int, int]:
        if self.position == 0:
            if dz > 1.0:
                self.position = 1
                return 1, self.position
            if dz < -1.0:
                self.position = -1
                return -1, self.position
        return 0, self.position
