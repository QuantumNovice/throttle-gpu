"""Test doubles shared by the test modules."""

from __future__ import annotations

from collections.abc import Sequence


class FakeSource:
    """Replays scripted temperature readings. The last reading repeats forever."""

    def __init__(self, *readings: Sequence[float]) -> None:
        self._readings: list[list[float]] = [list(r) for r in readings]
        self.closed = False

    def read(self) -> list[float]:
        if len(self._readings) > 1:
            return self._readings.pop(0)
        return list(self._readings[0])

    def close(self) -> None:
        self.closed = True


class BrokenSource:
    """A source that fails every read, like a driver that has gone away."""

    def read(self) -> list[float]:
        raise OSError("driver not responding")

    def close(self) -> None:
        pass
