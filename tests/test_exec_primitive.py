"""Docker-free contract and Linux-only real-process tests for the tagged job primitive (exec.py).

Two defects diagnosed live in 02-UAT.md are locked here (gaps G-02-1 and G-02-3):
- the launcher must be `setsid -w` (not bare `setsid`), so the exec stays attached until the command really
  ends and its output and exit code belong to the command, not to a setsid fork that returns 0 at once;
- the kill and sweep scripts must run as a plain `/bin/sh` argv carrying no job tag, not through the SDK's zsh
  wrapper (where unquoted parameters are not word-split, so nothing was killed), and must see the tag on the
  environ.

The contract tests run on every host against docker-py call-shape doubles. The Linux-only tests drive real
session-leader processes and `/proc`, so they are skipped where util-linux setsid or `/proc` is unavailable
(macOS included); the opt-in live suite in tests/test_live_exec.py covers the real container path.
"""

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from exegol_mcp.adapter.exec import (
    _SETSID_WAIT,
    _kill_script,
    _sweep_script,
    kill_tagged_job,
    start_tagged_job,
    sweep_tagged_jobs,
)
from exegol_mcp.adapter.port import JOB_TAG_ENV

_NEEDS_LINUX = not (sys.platform.startswith("linux") and shutil.which("setsid") and shutil.which("ps"))
_LINUX_ONLY = pytest.mark.skipif(_NEEDS_LINUX, reason="needs Linux with util-linux setsid, ps and /proc")


# --- Docker-py call-shape doubles ------------------------------------------------------------------------------


class _FakeApi:
    """Records the low-level exec_create / exec_start calls _create_and_start_job makes."""

    def __init__(self) -> None:
        self.exec_create_calls: list[SimpleNamespace] = []
        self.exec_start_calls: list[SimpleNamespace] = []
        self._generator = iter([(b"", None)])

    def exec_create(self, container_id: str, cmd: str, environment: Any = None, tty: Any = None) -> dict[str, str]:
        self.exec_create_calls.append(
            SimpleNamespace(container_id=container_id, cmd=cmd, environment=environment, tty=tty)
        )
        return {"Id": "exec-id"}

    def exec_start(self, exec_id: str, stream: Any = None, demux: Any = None) -> Any:
        self.exec_start_calls.append(SimpleNamespace(exec_id=exec_id, stream=stream, demux=demux))
        return self._generator

    def exec_inspect(self, exec_id: str) -> dict[str, int]:
        return {"ExitCode": 0}


class _FakeDockerContainer:
    """A docker-py Container double: .id, .client.api recorders, and an exec_run recorder (for kill/sweep)."""

    def __init__(self, exec_run_result: Any = None) -> None:
        self.id = "container-id"
        self.api = _FakeApi()
        self.client = SimpleNamespace(api=self.api)
        self.exec_run_calls: list[tuple[Any, dict[str, Any]]] = []
        self._exec_run_result = exec_run_result or SimpleNamespace(output=(b"", None), exit_code=0)

    def exec_run(self, cmd: Any, **kwargs: Any) -> Any:
        self.exec_run_calls.append((cmd, kwargs))
        return self._exec_run_result


def _exegol_container(docker_container: _FakeDockerContainer) -> SimpleNamespace:
    # ExegolContainer keeps the docker-py container in a private, name-mangled attribute.
    return SimpleNamespace(_ExegolContainer__container=docker_container)


# --- Task 1: the launcher is `setsid -w`, with the exact environment and a streamed demuxed exec ---------------


async def test_start_tagged_job_launches_under_setsid_wait_with_the_tag_on_the_environment() -> None:
    docker_container = _FakeDockerContainer()

    await start_tagged_job(_exegol_container(docker_container), "scan", "nmap -p-", "tag-123", cap_bytes=1024)

    [create] = docker_container.api.exec_create_calls
    # The command shlex-splits to the launcher prefix, then the SDK zsh wrapper, unchanged (G-02-3).
    assert shlex.split(create.cmd)[:4] == ["setsid", "-w", "zsh", "-c"]
    assert create.container_id == "container-id"
    assert create.tty is False
    # The command rides CMD and the unique tag rides EXEGOL_MCP_JOB, exactly as before.
    assert create.environment == {"CMD": "nmap -p-", "DISABLE_AUTO_UPDATE": "true", JOB_TAG_ENV: "tag-123"}
    [start] = docker_container.api.exec_start_calls
    assert start.stream is True
    assert start.demux is True


def test_setsid_wait_is_the_exact_prefix_the_adapter_sends() -> None:
    # The Linux test below runs this same literal, so the real-process check exercises the adapter's prefix.
    assert _SETSID_WAIT == "setsid -w"


@_LINUX_ONLY
def test_setsid_wait_keeps_the_commands_late_output_and_real_exit_code() -> None:
    # docker exec gives the command its own session, so start_new_session=True mirrors that here.
    script = "sleep 1; echo late; exit 7"

    started = time.perf_counter()
    waited = subprocess.run(
        [*_SETSID_WAIT.split(), "sh", "-c", script],
        start_new_session=True,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    # With -w the setsid parent waits: the stream, the output and the exit code all belong to the real command.
    assert waited.stdout.strip() == "late"
    assert waited.returncode == 7
    assert elapsed >= 1.0


@_LINUX_ONLY
def test_bare_setsid_forks_and_loses_the_exit_code_the_original_defect() -> None:
    script = "sleep 1; echo late; exit 7"

    started = time.perf_counter()
    forked = subprocess.run(
        ["setsid", "sh", "-c", script],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    elapsed = time.perf_counter() - started

    # Without -w the parent forks the command into the background and exits 0 at once: the real exit code 7 and
    # the late output are lost. This is exactly what broke every foreground exec, background job and remote run.
    assert forked.returncode == 0
    assert elapsed < 1.0


# --- Task 2: kill and sweep run as a plain /bin/sh argv, carry no tag, and see the environ --------------------


async def test_kill_tagged_job_runs_a_plain_posix_argv_with_no_tag() -> None:
    docker_container = _FakeDockerContainer()

    await kill_tagged_job(_exegol_container(docker_container), "tag-xyz", 1.0)

    [(argv, kwargs)] = docker_container.exec_run_calls
    # A fixed /bin/sh argv, not the SDK zsh wrapper, so unquoted parameters word-split as POSIX sh (G-02-1 (b)).
    assert argv == ["/bin/sh", "-c", _kill_script("tag-xyz", 1.0)]
    # No environment, so the script carries no CMD and no job tag: it can never match itself (T-02-19).
    assert "environment" not in kwargs
    assert 'eval "$CMD"' not in argv[2]
    assert kwargs == {"detach": False, "stream": False, "demux": True, "tty": False}
    # The low-level exec_create/exec_start path (used only for streamed jobs) is never touched for kill.
    assert docker_container.api.exec_create_calls == []


async def test_sweep_tagged_jobs_runs_a_plain_posix_argv_and_parses_the_count() -> None:
    docker_container = _FakeDockerContainer(SimpleNamespace(output=(b"2\n", None), exit_code=0))

    killed = await sweep_tagged_jobs(_exegol_container(docker_container), 1.0)

    assert killed == 2
    [(argv, kwargs)] = docker_container.exec_run_calls
    assert argv == ["/bin/sh", "-c", _sweep_script(1.0)]
    assert "environment" not in kwargs
    assert 'eval "$CMD"' not in argv[2]
    assert kwargs == {"detach": False, "stream": False, "demux": True, "tty": False}
    assert docker_container.api.exec_create_calls == []


@pytest.mark.parametrize("output", [(b"", None), (b"not-a-number\n", None), (None, None), (b"\n\n", None)])
async def test_sweep_count_is_zero_on_empty_or_unparsable_output(output: Any) -> None:
    docker_container = _FakeDockerContainer(SimpleNamespace(output=output, exit_code=0))
    assert await sweep_tagged_jobs(_exegol_container(docker_container), 1.0) == 0


def test_the_kill_script_no_longer_matches_the_command_line() -> None:
    # The old command-line-matching fallback matched argv, not the environ, so it never saw the tag; it is gone.
    assert "pkill" not in _kill_script("tag", 1.0)


@pytest.mark.parametrize(
    "script", [_kill_script("some-tag", 1.0), _sweep_script(1.0)], ids=["kill", "sweep"]
)
def test_scripts_are_valid_posix_sh(script: str) -> None:
    checked = 0
    for shell in ("sh", "dash", "bash"):
        path = shutil.which(shell)
        if path is None:
            continue
        result = subprocess.run([path, "-n"], input=script, capture_output=True, text=True, check=False)
        assert result.returncode == 0, (shell, result.stderr)
        checked += 1
    assert checked, "no POSIX shell available to syntax-check the scripts"


def test_kill_script_returns_at_once_when_nothing_is_tagged() -> None:
    # Nothing carries this random tag, so the grace sleep must be skipped: a 10 s grace cannot cost 10 s.
    started = time.perf_counter()
    result = subprocess.run(
        ["/bin/sh", "-c", _kill_script("no-such-tag-here", 10.0)], capture_output=True, text=True, check=False
    )
    elapsed = time.perf_counter() - started

    assert result.returncode == 0
    assert elapsed < 5.0


def test_sweep_script_prints_zero_and_returns_at_once_when_nothing_is_tagged() -> None:
    started = time.perf_counter()
    result = subprocess.run(
        ["/bin/sh", "-c", _sweep_script(10.0)], capture_output=True, text=True, check=False
    )
    elapsed = time.perf_counter() - started

    assert result.returncode == 0
    assert result.stdout.strip().splitlines()[-1] == "0"
    assert elapsed < 5.0


# --- Task 2: Linux-only real-process proof of the kill and sweep scripts --------------------------------------


def _tagged_pids(tag: str) -> list[int]:
    """Every PID whose /proc/<pid>/environ holds the EXACT EXEGOL_MCP_JOB=<tag> entry (T-02-02)."""
    entry = f"{JOB_TAG_ENV}={tag}".encode()
    pids: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            data = (proc / "environ").read_bytes()
        except OSError:
            continue
        if entry in data.split(b"\0"):
            pids.append(int(proc.name))
    return pids


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@_LINUX_ONLY
def test_kill_and_sweep_end_tagged_groups_while_a_prefix_tag_and_an_untagged_control_survive() -> None:
    target = "job-" + os.urandom(6).hex()
    suffix = target + "X"  # a prefix-sharing tag that the exact-match kill must NOT hit (T-02-02)
    spawned: list[subprocess.Popen[bytes]] = []

    def _spawn(tag: str | None, command: str) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        if tag is not None:
            env[JOB_TAG_ENV] = tag
        proc = subprocess.Popen(
            [*_SETSID_WAIT.split(), "sh", "-c", command],
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        spawned.append(proc)
        return proc

    try:
        _spawn(target, "sleep 30")
        _spawn(target, "trap '' TERM; while :; do sleep 1; done")  # ignores SIGTERM: only SIGKILL ends it
        _spawn(suffix, "sleep 30")
        control = _spawn(None, "sleep 30")

        # All tagged processes are visible before the kill.
        assert _wait_until(lambda: len(_tagged_pids(target)) >= 2, timeout=5.0)
        assert _tagged_pids(suffix) and control.poll() is None

        subprocess.run(["/bin/sh", "-c", _kill_script(target, 1.0)], check=False)

        # The exact-tag groups are gone (the TERM-ignoring one included); the prefix-tag job and the control live.
        assert _wait_until(lambda: _tagged_pids(target) == [], timeout=5.0)
        assert _tagged_pids(suffix) and control.poll() is None

        # The sweep now finds one orphaned job (the suffix-tagged one): with setsid -w it spans two process
        # groups but one distinct tag, so the count is 1, not 2.
        result = subprocess.run(
            ["/bin/sh", "-c", _sweep_script(1.0)], capture_output=True, text=True, check=False
        )
        assert result.stdout.strip().splitlines()[-1] == "1"
        assert _wait_until(lambda: _tagged_pids(suffix) == [], timeout=5.0)
        assert control.poll() is None
    finally:
        for proc in spawned:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
