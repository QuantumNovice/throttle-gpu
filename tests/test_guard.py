import time

import pytest

from throttle_gpu import GPUMonitorError, ThermalGuard, ThermalLimitExceeded
from throttle_gpu.guard import ThermalGuard as _Guard  # noqa: F401  (import path check)

from .fakes import BrokenSource, FakeSource


def test_guard_stops_block_when_hot():
    source = FakeSource([95.0])
    start = time.monotonic()
    with pytest.raises(ThermalLimitExceeded) as info:
        with ThermalGuard(max_temp_c=80, interval_s=0.01, consecutive=2, source=source):
            while True:  # simulates a long GPU job
                time.sleep(0.01)
    assert info.value.temperature_c == 95.0
    assert time.monotonic() - start < 5
    assert source.closed is False  # caller-owned sources are not closed


def test_guard_check_raises_inside_loop():
    guard = ThermalGuard(max_temp_c=80, interval_s=0.01, consecutive=1, source=FakeSource([95.0]))
    with pytest.raises(ThermalLimitExceeded):
        with guard:
            while True:
                guard.check()
                time.sleep(0.01)
    assert guard.tripped is True


def test_guard_passes_when_cool():
    source = FakeSource([60.0])
    with ThermalGuard(max_temp_c=80, interval_s=0.01, source=source) as guard:
        time.sleep(0.05)
    assert guard.tripped is False
    assert guard.peak_temp_c == 60.0


def test_guard_does_not_mask_user_exceptions():
    source = FakeSource([60.0])
    with pytest.raises(ValueError, match="user bug"):
        with ThermalGuard(max_temp_c=80, interval_s=0.01, source=source):
            raise ValueError("user bug")


def test_guard_fails_closed_when_reading_fails():
    with pytest.raises(GPUMonitorError, match="driver not responding"):
        with ThermalGuard(max_temp_c=80, interval_s=0.01, source=BrokenSource()):
            while True:
                time.sleep(0.01)


def test_guard_warning_fires_once_per_crossing():
    warnings = []
    source = FakeSource([75.0], [75.0], [60.0], [75.0])
    with ThermalGuard(
        max_temp_c=90,
        warn_temp_c=70,
        interval_s=0.01,
        source=source,
        on_warning=lambda temp, limit: warnings.append((temp, limit)),
    ):
        time.sleep(0.2)
    assert warnings == [(75.0, 70.0), (75.0, 70.0)]
