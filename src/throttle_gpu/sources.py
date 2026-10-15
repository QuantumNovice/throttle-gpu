"""Temperature sources: where GPU temperatures come from.

``default_source()`` prefers NVML (through ``nvidia-ml-py``) and falls back to
parsing ``nvidia-smi`` output. Tests and custom integrations can pass any object
with the same ``read()`` / ``close()`` methods.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import List, Protocol

from .errors import GPUUnavailableError


class TemperatureSource(Protocol):
    """Anything that can report the current temperature (°C) of each GPU."""

    def read(self) -> List[float]:
        """Return one temperature per GPU, in device index order."""

    def close(self) -> None:
        """Release any resources held by the source."""


class NvmlSource:
    """Reads GPU temperatures through NVML. Requires an NVIDIA driver."""

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception:
            pynvml.nvmlShutdown()
            raise
        if not self._handles:
            pynvml.nvmlShutdown()
            raise GPUUnavailableError("NVML reports no NVIDIA GPUs")

    def read(self) -> List[float]:
        gpu_sensor = self._nvml.NVML_TEMPERATURE_GPU
        return [
            float(self._nvml.nvmlDeviceGetTemperature(handle, gpu_sensor))
            for handle in self._handles
        ]

    def close(self) -> None:
        self._nvml.nvmlShutdown()


class NvidiaSmiSource:
    """Reads GPU temperatures by calling ``nvidia-smi``. Slower than NVML."""

    def __init__(self, binary: str = "nvidia-smi", timeout_s: float = 10.0) -> None:
        path = shutil.which(binary)
        if path is None:
            raise GPUUnavailableError(f"'{binary}' was not found on PATH")
        self._path = path
        self._timeout_s = timeout_s

    def read(self) -> List[float]:
        result = subprocess.run(
            [self._path, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
            check=True,
        )
        return [float(line) for line in result.stdout.splitlines() if line.strip()]

    def close(self) -> None:
        pass


def default_source() -> TemperatureSource:
    """Return the best available source, or raise ``GPUUnavailableError``."""
    try:
        return NvmlSource()
    except Exception:
        pass
    return NvidiaSmiSource()
