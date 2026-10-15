import sys

import pytest

from throttle_gpu import cli

from .fakes import FakeSource


@pytest.fixture
def cool_gpu(monkeypatch):
    monkeypatch.setattr(cli, "default_source", lambda: FakeSource([55.0, 60.0]))


@pytest.fixture
def hot_gpu(monkeypatch):
    monkeypatch.setattr(cli, "default_source", lambda: FakeSource([99.0]))


def test_status_prints_each_gpu(cool_gpu, capsys):
    assert cli.main(["status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "GPU 0: 55 °C" in out
    assert "GPU 1: 60 °C" in out


def test_watch_exits_with_thermal_code_when_hot(hot_gpu, capsys):
    code = cli.main(["watch", "--max-temp", "80", "--interval", "0.01", "--consecutive", "2"])
    assert code == cli.EXIT_THERMAL_LIMIT
    assert "above the 80" in capsys.readouterr().err


def test_run_returns_child_exit_code(cool_gpu):
    code = cli.main(
        ["run", "--interval", "0.05", "--", sys.executable, "-c", "raise SystemExit(7)"]
    )
    assert code == 7


def test_run_kills_child_when_hot(hot_gpu):
    code = cli.main(
        [
            "run",
            "--max-temp",
            "80",
            "--interval",
            "0.05",
            "--consecutive",
            "1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(120)",
        ]
    )
    assert code == cli.EXIT_THERMAL_LIMIT


def test_run_without_command_is_an_error(cool_gpu, capsys):
    assert cli.main(["run", "--"]) == cli.EXIT_ERROR
    assert "no command given" in capsys.readouterr().err


def test_missing_gpu_is_reported(monkeypatch, capsys):
    from throttle_gpu import GPUUnavailableError

    def no_gpu():
        raise GPUUnavailableError("no driver")

    monkeypatch.setattr(cli, "default_source", no_gpu)
    assert cli.main(["status"]) == cli.EXIT_ERROR
    assert "no usable GPU" in capsys.readouterr().err
