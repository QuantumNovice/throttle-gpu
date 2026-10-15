import pytest

from throttle_gpu import GPUMonitorError, ThermalLimitExceeded, ThermalPolicy
from throttle_gpu.policy import ThermalTracker


def test_policy_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        ThermalPolicy(interval_s=0)


def test_policy_rejects_zero_consecutive():
    with pytest.raises(ValueError):
        ThermalPolicy(consecutive=0)


def test_policy_warn_must_be_below_max():
    with pytest.raises(ValueError):
        ThermalPolicy(max_temp_c=80, warn_temp_c=80)


def test_tracker_trips_only_after_consecutive_samples():
    tracker = ThermalTracker(ThermalPolicy(max_temp_c=80, consecutive=3))
    assert tracker.update([85.0]) is None
    assert tracker.update([85.0]) is None
    error = tracker.update([85.0])
    assert isinstance(error, ThermalLimitExceeded)
    assert error.temperature_c == 85.0
    assert error.limit_c == 80


def test_tracker_resets_streak_on_cool_sample():
    tracker = ThermalTracker(ThermalPolicy(max_temp_c=80, consecutive=2))
    assert tracker.update([85.0]) is None
    assert tracker.update([70.0]) is None
    assert tracker.update([85.0]) is None  # streak restarted, not tripped yet


def test_tracker_limit_is_strict():
    tracker = ThermalTracker(ThermalPolicy(max_temp_c=80, consecutive=1))
    assert tracker.update([80.0]) is None  # exactly at the limit is allowed
    assert tracker.update([80.5]) is not None


def test_tracker_uses_hottest_gpu_and_reports_its_index():
    tracker = ThermalTracker(ThermalPolicy(max_temp_c=80, consecutive=1))
    error = tracker.update([60.0, 91.0, 75.0])
    assert error.gpu_index == 1
    assert error.temperature_c == 91.0


def test_tracker_records_peak():
    tracker = ThermalTracker(ThermalPolicy(max_temp_c=100))
    tracker.update([50.0, 60.0])
    tracker.update([70.0])
    tracker.update([55.0])
    assert tracker.peak_c == 70.0


def test_tracker_rejects_empty_reading():
    tracker = ThermalTracker(ThermalPolicy())
    with pytest.raises(GPUMonitorError):
        tracker.update([])
