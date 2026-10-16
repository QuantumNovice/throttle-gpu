"""Generate the GitHub Pages API reference from package docstrings."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DOCS = ROOT / "docs" / "api"
MODULES = (
    "throttle_gpu",
    "throttle_gpu.errors",
    "throttle_gpu.guard",
    "throttle_gpu.isolated",
    "throttle_gpu.policy",
    "throttle_gpu.sources",
    "throttle_gpu.cli",
)


def main() -> None:
    API_DOCS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(path for path in (source_path, existing_path) if path)
    subprocess.run(
        [sys.executable, "-m", "pydoc", "-w", *MODULES],
        cwd=API_DOCS,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
