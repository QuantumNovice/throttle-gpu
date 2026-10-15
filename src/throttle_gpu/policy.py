"""Thermal limits and the logic that decides when to trip."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .errors import GPUMonitorError, ThermalLimitExceeded


@dataclass(frozen=True)
class ThermalPolicy:
    """How hot a GPU may get and how quickly the guard reacts.

    Attributes:
        max_temp_c: Trip when a GPU is strictly hotter than this (°C).
        interval_s: Seconds between temperature samples.
        consecutive: Number of samples in a row above ``max_temp_c`` required to trip.
            Values above 1 filter out one-off spikes.
        warn_temp_c: Optional lower threshold that triggers a warning callback.
    """

    max_temp_c: float = 85.0
    interval_s: float = 1.0
    consecutive: int = 2
    warn_temp_c: float | None = None

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError("interval_s must be greater than zero")
        if self.consecutive < 1:
            raise ValueError("consecutive must be at least 1")
        if self.warn_temp_c is not None and self.warn_temp_c >= self.max_temp_c:
            raise ValueError("warn_temp_c must be lower than max_temp_c")


class ThermalTracker:
    """Feeds samples into a policy and reports when the limit is breached."""

    def __init__(self, policy: ThermalPolicy) -> None:
        self.policy = policy
        self.peak_c: float | None = None
        self._over_count = 0

    def update(self, temps: Sequence[float]) -> ThermalLimitExceeded | None:
        """Record one sample. Return an exception if the limit has been breached."""
        if not temps:
            raise GPUMonitorError("temperature source returned no GPU readings")

        hottest = max(temps)
        gpu_index = list(temps).index(hottest)
        self.peak_c = hottest if self.peak_c is None else max(self.peak_c, hottest)

        if hottest > self.policy.max_temp_c:
            self._over_count += 1
        else:
            self._over_count = 0

        if self._over_count >= self.policy.consecutive:
            return ThermalLimitExceeded(hottest, self.policy.max_temp_c, gpu_index)
        return None
