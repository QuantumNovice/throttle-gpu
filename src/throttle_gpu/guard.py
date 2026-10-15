"""``ThermalGuard``: watch GPU temperature while a block of code runs."""

from __future__ import annotations

import _thread
import threading
from typing import Callable, Optional

from .errors import GPUMonitorError, ThermalLimitExceeded
from .policy import ThermalPolicy, ThermalTracker
from .sources import TemperatureSource, default_source


class ThermalGuard:
    """Context manager that stops the ``with`` block when a GPU overheats.

    A background thread samples the GPU temperature. When the limit is breached, the
    main thread receives ``KeyboardInterrupt``, and the guard turns it into
    ``ThermalLimitExceeded`` when the block exits. Call ``check()`` inside long loops to
    stop at a known point instead.

    Limitation: Python only runs the interrupt between bytecodes. A call that blocks
    inside native code, such as a long CUDA kernel sync, is not interrupted until it
    returns. For a hard stop, use ``run_isolated()`` or the ``throttle-gpu run`` command,
    which terminate the work process.

    Example::

        with ThermalGuard(max_temp_c=83, consecutive=3):
            train_model()
    """

    def __init__(
        self,
        max_temp_c: float = 85.0,
        *,
        interval_s: float = 1.0,
        consecutive: int = 2,
        warn_temp_c: Optional[float] = None,
        on_warning: Optional[Callable[[float, float], None]] = None,
        source: Optional[TemperatureSource] = None,
    ) -> None:
        self.policy = ThermalPolicy(
            max_temp_c=max_temp_c,
            interval_s=interval_s,
            consecutive=consecutive,
            warn_temp_c=warn_temp_c,
        )
        self._on_warning = on_warning
        self._source = source
        self._owns_source = False
        self._tracker = ThermalTracker(self.policy)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._interrupt_main = False

    @property
    def peak_temp_c(self) -> Optional[float]:
        """Hottest reading seen so far, or ``None`` before the first sample."""
        return self._tracker.peak_c

    @property
    def tripped(self) -> bool:
        """``True`` once the guard has decided to stop the work."""
        return self._error is not None

    def check(self) -> None:
        """Raise the guard's error now if it has tripped. Cheap enough for hot loops."""
        if self._error is not None:
            raise self._error

    def __enter__(self) -> "ThermalGuard":
        if self._source is None:
            self._source = default_source()
            self._owns_source = True
        # Only the main thread can receive the interrupt. Other threads rely on check().
        self._interrupt_main = threading.current_thread() is threading.main_thread()
        self._error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._watch, name="throttle-gpu-monitor", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._owns_source and self._source is not None:
            self._source.close()
            self._source = None
            self._owns_source = False

        if self._error is not None:
            if exc_type is None or exc_type is KeyboardInterrupt:
                # The trip interrupt may have been delivered as KeyboardInterrupt.
                # Replace it with the real error.
                raise self._error from None
        return False

    def _watch(self) -> None:
        assert self._source is not None
        warned = False
        while not self._stop.is_set():
            try:
                temps = self._source.read()
                error: Optional[BaseException] = self._tracker.update(temps)
            except GPUMonitorError as exc:
                error = exc
            except Exception as exc:
                error = GPUMonitorError(f"could not read GPU temperature: {exc}")

            if error is None and self._on_warning and self.policy.warn_temp_c is not None:
                hottest = max(temps)
                if hottest >= self.policy.warn_temp_c and not warned:
                    warned = True
                    self._on_warning(hottest, self.policy.warn_temp_c)
                elif hottest < self.policy.warn_temp_c:
                    warned = False

            if error is not None:
                self._error = error
                if self._interrupt_main:
                    _thread.interrupt_main()
                return

            self._stop.wait(self.policy.interval_s)
