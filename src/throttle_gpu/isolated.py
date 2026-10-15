"""``run_isolated``: run a function in a child process and kill it if the GPU overheats."""

from __future__ import annotations

import multiprocessing
import pickle
import queue as queue_module
from typing import Any, Callable, Mapping, Optional, Sequence

from .errors import GPUMonitorError, ThermalLimitExceeded
from .policy import ThermalPolicy, ThermalTracker
from .sources import TemperatureSource, default_source

_KILL_GRACE_S = 5.0


def _child_main(target: Callable[..., Any], args: tuple, kwargs: dict, results: Any) -> None:
    try:
        payload: Any = ("ok", target(*args, **kwargs))
    except BaseException as exc:
        payload = ("error", exc)
    try:
        pickle.dumps(payload)
    except Exception:
        # Fall back to a picklable summary when the value or exception can't cross processes.
        payload = ("error", RuntimeError(f"unpicklable result or exception: {payload!r}"))
    results.put(payload)


def run_isolated(
    target: Callable[..., Any],
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    max_temp_c: float = 85.0,
    interval_s: float = 1.0,
    consecutive: int = 2,
    source: Optional[TemperatureSource] = None,
) -> Any:
    """Run ``target(*args, **kwargs)`` in a child process, stopping it on overheat.

    The child runs under the ``spawn`` start method, which is the safe choice for CUDA
    on every platform. ``target`` must be a module-level function, and its arguments
    and return value must be picklable.

    Returns the target's return value. Re-raises the target's exception if it fails.
    Raises ``ThermalLimitExceeded`` after killing the child if the limit is breached.

    Args:
        target: Module-level callable to run.
        args: Positional arguments for ``target``.
        kwargs: Keyword arguments for ``target``.
        max_temp_c: Trip when a GPU is hotter than this (°C).
        interval_s: Seconds between temperature samples.
        consecutive: Over-limit samples in a row required to trip.
        source: Temperature source. Defaults to ``default_source()``.
    """
    policy = ThermalPolicy(
        max_temp_c=max_temp_c, interval_s=interval_s, consecutive=consecutive
    )
    tracker = ThermalTracker(policy)
    owns_source = source is None
    if source is None:
        source = default_source()

    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    child = ctx.Process(
        target=_child_main, args=(target, tuple(args), dict(kwargs or {}), results)
    )
    child.start()
    try:
        while child.is_alive():
            child.join(policy.interval_s)
            if not child.is_alive():
                break
            try:
                error = tracker.update(source.read())
            except GPUMonitorError as exc:
                error = exc
            except Exception as exc:
                error = GPUMonitorError(f"could not read GPU temperature: {exc}")
            if error is not None:
                _stop_child(child)
                raise error
    finally:
        if owns_source:
            source.close()
        if child.is_alive():
            _stop_child(child)

    try:
        status, value = results.get(timeout=_KILL_GRACE_S)
    except queue_module.Empty:
        raise RuntimeError(
            f"isolated process exited without a result (exit code {child.exitcode})"
        ) from None
    finally:
        results.close()

    if status == "error":
        raise value
    return value


def _stop_child(child: "multiprocessing.Process") -> None:
    child.terminate()
    child.join(_KILL_GRACE_S)
    if child.is_alive():
        child.kill()
        child.join()
