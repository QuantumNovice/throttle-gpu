import time

import pytest

from throttle_gpu import ThermalLimitExceeded, run_isolated

from .fakes import FakeSource


def add(a, b):
    return a + b


def fail():
    raise ValueError("child failed")


def sleep_forever():
    time.sleep(120)


def test_isolated_returns_result():
    assert run_isolated(add, (2, 3), source=FakeSource([50.0]), interval_s=0.05) == 5


def test_isolated_accepts_kwargs():
    assert (
        run_isolated(add, kwargs={"a": 4, "b": 1}, source=FakeSource([50.0]), interval_s=0.05) == 5
    )


def test_isolated_propagates_child_exception():
    with pytest.raises(ValueError, match="child failed"):
        run_isolated(fail, source=FakeSource([50.0]), interval_s=0.05)


def test_isolated_kills_child_when_hot():
    start = time.monotonic()
    with pytest.raises(ThermalLimitExceeded):
        run_isolated(
            sleep_forever,
            max_temp_c=80,
            interval_s=0.05,
            consecutive=1,
            source=FakeSource([95.0]),
        )
    assert time.monotonic() - start < 30  # the 120 s child was stopped early
