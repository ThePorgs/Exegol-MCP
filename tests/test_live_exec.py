"""Opt-in live re-run of the 02-UAT scenarios against a real Exegol container (gaps G-02-1, G-02-3, G-02-4).

The whole module is skipped unless EXEGOL_MCP_LIVE_CONTAINER names an Exegol container (its SDK/tool name, e.g.
`mcpuat`; the docker name is `exegol-<name>`). When set, every test drives the REAL stdio entry point
(`python -m exegol_mcp --type stdio`) with the user's own HOME and Docker, through the SDK's own MCP `Client` —
never tests.fake_server, so the real adapter, the real setsid -w launcher and the real /proc kill/sweep run.

These scenarios cannot run under the Docker-free gate; they are the backstop that re-verifies the live truths the
UAT found broken. Run them with:

    EXEGOL_MCP_LIVE_CONTAINER=mcpuat .venv/bin/python -m pytest tests/test_live_exec.py -q
"""

import os
import secrets
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, StdioServerParameters

from exegol_mcp.adapter.port import JOB_TAG_ENV

REPO_ROOT = Path(__file__).resolve().parent.parent

_CONTAINER = os.environ.get("EXEGOL_MCP_LIVE_CONTAINER")
_DOCKER_NAME = f"exegol-{_CONTAINER}" if _CONTAINER else ""

pytestmark = pytest.mark.skipif(
    not _CONTAINER,
    reason="opt-in live suite: set EXEGOL_MCP_LIVE_CONTAINER=<exegol container name> to run",
)


def _server_parameters() -> StdioServerParameters:
    """SDK stdio-client parameters for the REAL entry point (the user's own HOME, adapter and Docker)."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "exegol_mcp", "--type", "stdio", "--no-update-check"],
        env=dict(os.environ),
        cwd=str(REPO_ROOT),
    )


@asynccontextmanager
async def _session() -> AsyncIterator[Client]:
    """Open a fresh MCP client session against a freshly launched real server over stdio."""
    async with Client(_server_parameters()) as client:
        yield client


async def _call(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool, assert it did not error, and return its structured content."""
    result = await client.call_tool(name, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None, (name, result.content)
    return result.structured_content


def _docker_running(name: str) -> bool:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


def _marker_present(marker: str) -> bool:
    """True if any process inside the container still runs a command carrying the unique marker.

    The grep pattern is bracketed so the grep process itself never matches (its own argv contains `[s]leep`,
    which does not match the literal `sleep`).
    """
    bracketed = f"[{marker[0]}]{marker[1:]}"
    out = subprocess.run(
        ["docker", "exec", _DOCKER_NAME, "sh", "-c", f"ps -eo args | grep -c -- '{bracketed}'"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.returncode == 0 and out.stdout.strip() not in ("", "0")


def _seed_orphan(tag: str, duration: str) -> None:
    """Start a tagged, detached setsid -w process in the container, as a crashed server would leave behind."""
    subprocess.run(
        ["docker", "exec", "-d", "-e", f"{JOB_TAG_ENV}={tag}", _DOCKER_NAME, "setsid", "-w", "sleep", duration],
        check=True,
    )


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


def _marker_gone(marker: str, timeout: float = 20.0) -> bool:
    """True once no in-container process runs the unique marker (give SIGTERM+grace+SIGKILL time to land)."""
    return _wait_until(lambda: not _marker_present(marker), timeout)


@pytest.fixture(scope="module", autouse=True)
def _live_container() -> Iterator[None]:
    """Ensure the container is running for the module (via the real start_container tool); restore its prior state."""
    assert _CONTAINER
    was_running = _docker_running(_DOCKER_NAME)

    async def _start() -> None:
        async with _session() as client:
            await _call(client, "start_container", {"container_name": _CONTAINER})

    anyio.run(_start)
    yield
    if not was_running:

        async def _stop() -> None:
            async with _session() as client:
                await _call(client, "stop_container", {"container_name": _CONTAINER})

        anyio.run(_stop)


# --- Task 1: the foreground path keeps its late output and real exit code (G-02-3) ----------------------------


async def test_foreground_keeps_late_output_and_real_exit_code() -> None:
    async with _session() as client:
        started = time.perf_counter()
        payload = await _call(
            client,
            "exec_command",
            {"container_name": _CONTAINER, "command": "echo hi; sleep 3; echo bye; exit 3", "timeout": 30},
        )
        elapsed = time.perf_counter() - started

    # The exec stayed attached for the full command: both lines and the real exit code 3, not 'hi' and 0 at ~2 s.
    assert payload["stdout"] == "hi\nbye\n"
    assert payload["exit_code"] == 3
    assert payload["timed_out"] is False
    assert payload["job_id"] is None
    assert elapsed >= 3.0


# --- Task 3: the UAT scenarios, re-run live (G-02-1, G-02-3, G-02-4) ------------------------------------------
#
# Each long-running command carries a unique `sleep <marker>` so a follow-up check (a bracketed ps grep that
# never matches itself) can confirm no process with that marker survives. The tick-loop markers are distinct
# short sleeps so at least two ticks print inside the 4 s timeout; the plain-sleep markers are distinct large
# integers.


async def test_on_timeout_kill_terminates_and_returns_partial_output() -> None:
    marker = "sleep 1.7"
    command = f"for i in $(seq 1 30); do echo tick$i; {marker}; done"
    async with _session() as client:
        payload = await _call(
            client,
            "exec_command",
            {"container_name": _CONTAINER, "command": command, "timeout": 4, "on_timeout": "kill"},
        )

    # The foreground timeout fired and the real in-container process was killed, with partial output kept.
    assert payload["timed_out"] is True
    assert payload["exit_code"] is None
    assert payload["job_id"] is None
    assert payload["stdout"].count("tick") >= 2
    assert _marker_gone(marker)


async def test_on_timeout_background_demotes_to_a_pollable_job() -> None:
    marker = "sleep 1.8"
    command = f"for i in $(seq 1 30); do echo tick$i; {marker}; done"
    async with _session() as client:
        payload = await _call(
            client,
            "exec_command",
            {"container_name": _CONTAINER, "command": command, "timeout": 4, "on_timeout": "background"},
        )
        assert payload["timed_out"] is True
        job_id = payload["job_id"]
        assert isinstance(job_id, str) and job_id

        running = await _call(client, "get_job", {"job_id": job_id})
        assert running["status"] == "running"

        stopped = await _call(client, "stop_job", {"job_id": job_id})
        assert stopped["status"] == "killed"

    assert _marker_gone(marker)


async def test_background_job_runs_until_stopped() -> None:
    marker = "sleep 303"
    async with _session() as client:
        payload = await _call(
            client, "exec_command", {"container_name": _CONTAINER, "command": marker, "background": True}
        )
        job_id = payload["job_id"]
        assert isinstance(job_id, str) and job_id
        assert payload["exit_code"] is None

        # The UAT saw the job reported completed/exit 0 within 1.5 s while the sleep still ran; it must stay running.
        await anyio.sleep(3)
        running = await _call(client, "get_job", {"job_id": job_id})
        assert running["status"] == "running"
        assert _marker_present(marker)

        stopped = await _call(client, "stop_job", {"job_id": job_id})
        assert stopped["status"] == "killed"

    assert _marker_gone(marker)


async def test_term_ignoring_background_job_is_killed_after_the_grace() -> None:
    marker = "sleep 305"
    # The shell ignores SIGTERM and restarts the sleep, so only the SIGKILL after the grace ends it (D-16).
    command = f"trap '' TERM; while :; do {marker}; done"
    async with _session() as client:
        payload = await _call(
            client, "exec_command", {"container_name": _CONTAINER, "command": command, "background": True}
        )
        job_id = payload["job_id"]
        assert isinstance(job_id, str) and job_id
        assert _wait_until(lambda: _marker_present(marker), timeout=5)

        stopped = await _call(client, "stop_job", {"job_id": job_id})
        assert stopped["status"] == "killed"

    assert _marker_gone(marker)


async def test_startup_and_shutdown_orphan_sweep() -> None:
    # Startup sweep: seed a tagged orphan OUTSIDE the server (as a crashed server would leave), then a fresh
    # server session must sweep it during lifespan startup.
    startup_marker = "sleep 304"
    _seed_orphan(secrets.token_urlsafe(16), "304")
    assert _wait_until(lambda: _marker_present(startup_marker), timeout=5)

    async with _session():
        pass  # opening a session runs the startup orphan sweep in the lifespan

    assert _marker_gone(startup_marker)

    # Shutdown sweep: start a background job, then close the session; the shielded shutdown scope sweeps it (D-11).
    shutdown_marker = "sleep 306"
    async with _session() as client:
        await _call(
            client, "exec_command", {"container_name": _CONTAINER, "command": shutdown_marker, "background": True}
        )
        assert _wait_until(lambda: _marker_present(shutdown_marker), timeout=5)

    assert _marker_gone(shutdown_marker)


async def test_remote_command_on_timeout_background_becomes_a_job() -> None:
    # netexec against an unroutable host hangs connecting; the foreground timeout demotes it to a pollable job.
    async with _session() as client:
        payload = await _call(
            client,
            "exec_remote_command",
            {
                "container_name": _CONTAINER,
                "protocol": "smb",
                "target_ip": "10.255.255.1",
                "username": "dummy",
                "password": "dummy",
                "command": "whoami",
                "timeout": 3,
                "on_timeout": "background",
            },
        )
        assert payload["timed_out"] is True
        job_id = payload["job_id"]
        assert isinstance(job_id, str) and job_id

        stopped = await _call(client, "stop_job", {"job_id": job_id})
        assert stopped["status"] == "killed"
