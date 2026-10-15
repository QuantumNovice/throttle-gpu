"""Exceptions raised by throttle-gpu."""

from __future__ import annotations


class GPUUnavailableError(RuntimeError):
    """No usable NVIDIA GPU or driver tooling could be found."""


class GPUMonitorError(RuntimeError):
    """Temperature monitoring failed, so the guard cannot vouch for the run.

    The guard fails closed: if it cannot read a temperature, it stops the work.
    """


class ThermalLimitExceeded(RuntimeError):
    """A GPU stayed above the configured temperature limit."""

    def __init__(self, temperature_c: float, limit_c: float, gpu_index: int = 0) -> None:
        self.temperature_c = temperature_c
        self.limit_c = limit_c
        self.gpu_index = gpu_index
        super().__init__(
            f"GPU {gpu_index} reached {temperature_c:.0f} °C, above the {limit_c:.0f} °C limit"
        )
