"""Command-line interface: ``throttle-gpu status | watch | run``."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from .errors import GPUMonitorError, GPUUnavailableError, ThermalLimitExceeded
from .policy import ThermalPolicy, ThermalTracker
from .sources import default_source

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_THERMAL_LIMIT = 3
_KILL_GRACE_S = 5.0


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-temp",
        type=float,
        default=85.0,
        help="trip above this temperature in °C (default 85)",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="seconds between samples (default 1)"
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="over-limit samples in a row required to trip (default 2)",
    )


def _policy_from(args: argparse.Namespace) -> ThermalPolicy:
    return ThermalPolicy(
        max_temp_c=args.max_temp, interval_s=args.interval, consecutive=args.consecutive
    )


def _cmd_status(args: argparse.Namespace) -> int:
    source = default_source()
    try:
        for index, temp in enumerate(source.read()):
            print(f"GPU {index}: {temp:.0f} °C")
    finally:
        source.close()
    return EXIT_OK


def _cmd_watch(args: argparse.Namespace) -> int:
    policy = _policy_from(args)
    tracker = ThermalTracker(policy)
    source = default_source()
    try:
        while True:
            temps = source.read()
            print(
                f"{time.strftime('%H:%M:%S')}  "
                + "  ".join(f"GPU{i} {t:.0f}°C" for i, t in enumerate(temps)),
                flush=True,
            )
            error = tracker.update(temps)
            if error is not None:
                print(f"throttle-gpu: {error}", file=sys.stderr)
                return EXIT_THERMAL_LIMIT
            time.sleep(policy.interval_s)
    except KeyboardInterrupt:
        return EXIT_OK
    finally:
        source.close()


def _cmd_run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print(
            "throttle-gpu run: no command given (usage: throttle-gpu run -- CMD ...)",
            file=sys.stderr,
        )
        return EXIT_ERROR

    policy = _policy_from(args)
    tracker = ThermalTracker(policy)
    source = default_source()

    # Own process group on POSIX so the whole tree can be signalled together.
    popen_kwargs = {} if sys.platform == "win32" else {"start_new_session": True}
    child = subprocess.Popen(command, **popen_kwargs)
    try:
        while True:
            try:
                return_code = child.wait(timeout=policy.interval_s)
            except subprocess.TimeoutExpired:
                pass
            else:
                return return_code

            error = tracker.update(source.read())
            if error is not None:
                print(f"throttle-gpu: {error}; stopping process {child.pid}", file=sys.stderr)
                _terminate_tree(child)
                return EXIT_THERMAL_LIMIT
    finally:
        if child.poll() is None:
            _terminate_tree(child)
        source.close()


def _terminate_tree(child: subprocess.Popen) -> None:
    """Stop a process and its children. Graceful on POSIX, forceful on Windows."""
    if child.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(child.pid)],
            capture_output=True,
            check=False,
        )
        child.wait()
        return

    _signal_group(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        _signal_group(child.pid, signal.SIGKILL)
        child.wait()


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="throttle-gpu",
        description="Stop GPU-heavy work before the GPU overheats.",
    )
    sub = parser.add_subparsers(dest="command_name", required=True)

    sub.add_parser("status", help="print the current temperature of each GPU")

    watch = sub.add_parser("watch", help="print temperatures until interrupted or a limit trips")
    _add_policy_args(watch)

    run = sub.add_parser(
        "run",
        help="run a command and kill it if a GPU overheats",
        description="Run a command. If a GPU overheats, stop the command and its children.",
    )
    _add_policy_args(run)
    run.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"status": _cmd_status, "watch": _cmd_watch, "run": _cmd_run}
    try:
        return handlers[args.command_name](args)
    except GPUUnavailableError as exc:
        print(f"throttle-gpu: no usable GPU: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (GPUMonitorError, ThermalLimitExceeded) as exc:
        print(f"throttle-gpu: {exc}", file=sys.stderr)
        return EXIT_THERMAL_LIMIT if isinstance(exc, ThermalLimitExceeded) else EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
