"""throttle-gpu: stop GPU-heavy Python code before the machine overheats.

Public API:
    ThermalGuard        context manager that watches temperature around a block of code
    run_isolated       run a function in a child process that is killed on overheat
    ThermalPolicy       the limits a guard or runner enforces
    default_source      pick the best available temperature source
    ThermalLimitExceeded, GPUMonitorError, GPUUnavailableError   exceptions
"""

from .errors import GPUMonitorError, GPUUnavailableError, ThermalLimitExceeded
from .guard import ThermalGuard
from .isolated import run_isolated
from .policy import ThermalPolicy
from .sources import NvidiaSmiSource, NvmlSource, TemperatureSource, default_source

__all__ = [
    "GPUMonitorError",
    "GPUUnavailableError",
    "NvidiaSmiSource",
    "NvmlSource",
    "TemperatureSource",
    "ThermalGuard",
    "ThermalLimitExceeded",
    "ThermalPolicy",
    "default_source",
    "run_isolated",
]

__version__ = "0.1.0"
