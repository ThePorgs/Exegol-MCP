"""Run the project's own mypy, whichever tool made the environment (the pre-push hook in .pre-commit-config.yaml).

    python scripts/run_mypy.py [mypy arguments...]

A git hook runs with the user's shell PATH, and neither PDM nor uv activates .venv, so a bare `mypy` is not found
(review WR-04). This launcher finds the project environment itself, in this order:

1. the project's .venv mypy (.venv/bin/mypy, or .venv/Scripts/mypy.exe on Windows): the default in-project venv of
   both PDM and uv;
2. `pdm run mypy`, when pdm is on PATH (a PDM venv outside the project);
3. `uv run --no-sync mypy`, when uv is on PATH (UV_PROJECT_ENVIRONMENT pointing elsewhere).

It forwards its arguments, returns mypy's own exit status and never installs anything: `--no-sync` keeps `uv run`
from syncing the environment, and `pdm run` does not sync. Standard library only, Python 3.10.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _venv_mypy(root: Path) -> Path | None:
    candidate = root / ".venv" / ("Scripts/mypy.exe" if os.name == "nt" else "bin/mypy")
    return candidate if candidate.is_file() else None


def _command(root: Path) -> list[str] | None:
    venv_mypy = _venv_mypy(root)
    if venv_mypy is not None:
        return [str(venv_mypy)]
    pdm = shutil.which("pdm")
    if pdm is not None:
        return [pdm, "run", "mypy"]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "run", "--no-sync", "mypy"]
    return None


def main(argv: list[str]) -> int:
    command = _command(PROJECT_ROOT)
    if command is None:
        print(
            "run_mypy: mypy not found: create the project's .venv (pdm sync -G dev or uv sync), "
            "or put pdm (pdm run mypy) or uv (uv run mypy) on PATH",
            file=sys.stderr,
        )
        return 1
    # pre-commit's language: python points VIRTUAL_ENV at the hook's own empty environment; PDM and uv must resolve
    # the project environment instead.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.call([*command, *argv], cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
