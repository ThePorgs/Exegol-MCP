"""D-18 container start/stop internal timeout and the real SDK startup-wait bound. Docker-free.

Task 1 covers the service/tool bound: a slow start or stop (over a cooperative slow fake and a
blocked-worker-thread fake) is cut at container_op_timeout_s with a clear error; a fast start/stop and a
no-op in the target state return true; the bound is separate from the exec timeout keys; and image
downloads are never cut by the container timeout.

Task 2 covers the real path: patch_exegol_sdk() installs a deadline-aware ContainerLogStream subclass that
ends the SDK's otherwise-unbounded READY wait before the enclosing anyio deadline, mirroring the SDK's own
Ctrl+C skip path, without lengthening the SDK's 2 s legacy-image detection or changing behavior when no
deadline encloses it.
"""

import logging
import threading
import time
import types
from datetime import datetime
from typing import Any

import anyio
import pytest
from mcp import Client

from exegol_mcp.adapter import sdk_headless
from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.app import build_server
from exegol_mcp.models.config import ExecutionLimits, OperatorConfig
from exegol_mcp.services.containers import start_container, stop_container
from exegol_mcp.services.images import download_image as download_image_service
from tests.fakes import FakeExegolAdapter

# --- Slow fakes (Docker-free): one cooperative, one blocking a worker thread ----------------------------------


class _SleepFake(FakeExegolAdapter):
    """start_container/stop_container await a cooperative anyio.sleep before delegating (slow async task)."""

    def __init__(self, slow_s: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._slow_s = slow_s

    async def start_container(self, name: str) -> bool:
        await anyio.sleep(self._slow_s)
        return await super().start_container(name)

    async def stop_container(self, name: str) -> bool:
        await anyio.sleep(self._slow_s)
        return await super().stop_container(name)


class _BlockedThreadFake(FakeExegolAdapter):
    """start/stop block a worker thread via run_blocking, mirroring the real adapter's offloaded SDK calls.

    run_blocking abandons the worker thread on cancellation, so the service deadline can still fire.
    """

    def __init__(self, slow_s: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._slow_s = slow_s

    async def start_container(self, name: str) -> bool:
        await run_blocking(threading.Event().wait, self._slow_s)
        return await super().start_container(name)

    async def stop_container(self, name: str) -> bool:
        await run_blocking(threading.Event().wait, self._slow_s)
        return await super().stop_container(name)


class _SlowDownloadFake(FakeExegolAdapter):
    """download_image awaits a cooperative sleep before delegating, to show downloads are never bounded."""

    def __init__(self, download_s: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._download_s = download_s

    async def download_image(self, name: str, version: str, progress: Any) -> bool:
        await anyio.sleep(self._download_s)
        return await super().download_image(name, version, progress)


def _stopped(fake: FakeExegolAdapter, name: str = "exegol-lab") -> FakeExegolAdapter:
    """Flip a container to stopped so a start call reaches the adapter."""
    fake.running[name] = False
    return fake


# --- Service bound: slow start/stop is cut at the deadline with a clear error (D-18) --------------------------


async def test_slow_start_is_cut_at_the_deadline_with_a_clear_error() -> None:
    fake = _stopped(_SleepFake(slow_s=5.0))
    started = time.perf_counter()
    with pytest.raises(RuntimeError) as exc:
        await start_container(fake, "exegol-lab", timeout_s=0.2)
    assert time.perf_counter() - started < 2.0  # cut at the deadline, well under slow_s
    message = str(exc.value)
    assert "exegol-lab" in message
    assert "0.2" in message
    assert "list_containers" in message
    assert ("start" in message)  # names starting, not stopping


async def test_slow_stop_is_cut_at_the_deadline_with_a_clear_error() -> None:
    fake = _SleepFake(slow_s=5.0)  # default containers are running, so stop reaches the adapter
    started = time.perf_counter()
    with pytest.raises(RuntimeError) as exc:
        await stop_container(fake, "exegol-lab", timeout_s=0.2)
    assert time.perf_counter() - started < 2.0
    message = str(exc.value)
    assert "exegol-lab" in message
    assert "0.2" in message
    assert "list_containers" in message
    assert "stop" in message


async def test_a_blocked_worker_thread_start_is_cut_at_the_deadline() -> None:
    fake = _stopped(_BlockedThreadFake(slow_s=5.0))
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="did not finish starting"):
        await start_container(fake, "exegol-lab", timeout_s=0.2)
    assert time.perf_counter() - started < 2.0


async def test_a_blocked_worker_thread_stop_is_cut_at_the_deadline() -> None:
    fake = _BlockedThreadFake(slow_s=5.0)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="did not finish stopping"):
        await stop_container(fake, "exegol-lab", timeout_s=0.2)
    assert time.perf_counter() - started < 2.0


# --- Boundary edge: a fast start/stop and a no-op in the target state return true -----------------------------


async def test_a_fast_start_returns_true_unchanged() -> None:
    fake = _stopped(FakeExegolAdapter())
    assert await start_container(fake, "exegol-lab", timeout_s=5.0) is True
    assert ("start", "exegol-lab") in fake.state_changes


async def test_a_fast_stop_returns_true_unchanged() -> None:
    fake = FakeExegolAdapter()
    assert await stop_container(fake, "exegol-lab", timeout_s=5.0) is True
    assert ("stop", "exegol-lab") in fake.state_changes


async def test_already_running_start_is_a_no_op_even_with_a_tiny_timeout() -> None:
    fake = FakeExegolAdapter()  # exegol-lab is running by default
    assert await start_container(fake, "exegol-lab", timeout_s=0.001) is True
    assert fake.state_changes == []  # the adapter was never asked to start it


async def test_already_stopped_stop_is_a_no_op_even_with_a_tiny_timeout() -> None:
    fake = _stopped(FakeExegolAdapter())
    assert await stop_container(fake, "exegol-lab", timeout_s=0.001) is True
    assert fake.state_changes == []


async def test_an_unknown_container_still_raises_a_value_error_not_a_timeout() -> None:
    fake = FakeExegolAdapter()
    with pytest.raises(ValueError, match="not found"):
        await start_container(fake, "no-such-container", timeout_s=0.001)


# --- Through the in-memory Client: the tool reads container_op_timeout_s, separate from the exec keys ----------


async def test_tool_slow_start_returns_is_error_mentioning_list_containers() -> None:
    config = OperatorConfig(execution=ExecutionLimits(container_op_timeout_s=0.2))
    fake = _stopped(_SleepFake(slow_s=5.0, config=config))
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool("start_container", {"container_name": "exegol-lab"})
    assert result.is_error
    assert "list_containers" in str(result.content)


async def test_separation_a_start_slower_than_the_exec_default_but_faster_than_the_container_timeout_succeeds() -> None:
    # A tiny exec default must not touch start/stop; the large container timeout is the only bound that applies.
    config = OperatorConfig(execution=ExecutionLimits(timeout_default_s=0.01, container_op_timeout_s=5.0))
    fake = _stopped(_SleepFake(slow_s=0.3, config=config))
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool("start_container", {"container_name": "exegol-lab"})
    assert not result.is_error, result.content
    assert result.structured_content is not None and result.structured_content["result"] is True


async def test_a_slow_download_is_not_bounded_by_the_container_timeout() -> None:
    # container_op_timeout_s is tiny, but download_image has no fixed timeout (D-18), so a slow pull completes.
    config = OperatorConfig(execution=ExecutionLimits(container_op_timeout_s=0.1))
    fake = _SlowDownloadFake(download_s=0.3, config=config)
    assert await download_image_service(fake, "full") is True
    assert ("full", "latest") in fake.pulls


# --- Task 2: the deadline-aware SDK startup-status (ContainerLogStream) bound (D-18 real path) ----------------


class _StubDockerContainer:
    """Minimal docker container whose logs() yields scripted byte chunks per poll, then silence forever.

    Each logs() call returns the next scripted poll (a list of byte chunks); once the script is exhausted it
    returns an empty iterator, standing in for a container that has gone silent (never logs READY).
    """

    def __init__(self, polls: list[list[bytes]]) -> None:
        self._polls = [list(poll) for poll in polls]
        self.logs_calls = 0

    def logs(self, stream: bool, follow: bool, since: Any, until: Any) -> Any:
        self.logs_calls += 1
        if self._polls:
            return iter(self._polls.pop(0))
        return iter(())  # silence


@pytest.fixture
def fast_log_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the SDK ContainerLogStream poll loop fast: a 1 s time.sleep becomes a 10 ms real sleep."""
    from exegol.utils import ContainerLogStream as log_stream_module

    monkeypatch.setattr(log_stream_module, "time", types.SimpleNamespace(sleep=lambda _s: time.sleep(0.01)))


@pytest.fixture
def sdk_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore every SDK global patch_exegol_sdk() mutates, including the ContainerLogStream patch point."""
    import exegol.model.ExegolContainer as container_module
    from docker.models.containers import ContainerCollection
    from exegol.console.ExegolPrompt import ExegolRich
    from exegol.console.TUI import ExegolTUI
    from exegol.utils.ExeLog import ExeLog, logger
    from exegol.utils.MetaSingleton import MetaSingleton

    monkeypatch.setattr(MetaSingleton, "_MetaSingleton__instances", dict(MetaSingleton._MetaSingleton__instances))
    monkeypatch.setattr(ExeLog, "console", ExeLog.console)
    monkeypatch.setattr(logger, "_ExeLog__critical_method", logger._ExeLog__critical_method)
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "console"):
            monkeypatch.setattr(handler, "console", handler.console)
    for name in ("Ask", "Confirm", "Acknowledge"):
        monkeypatch.setattr(ExegolRich, name, ExegolRich.__dict__[name])
    monkeypatch.setattr(ExegolTUI, "downloadDockerLayer", ExegolTUI.__dict__["downloadDockerLayer"])
    monkeypatch.setattr(container_module, "ContainerLogStream", container_module.ContainerLogStream)
    monkeypatch.setattr(container_module, "secure_remove", container_module.secure_remove)
    monkeypatch.setattr(ContainerCollection, "create", ContainerCollection.__dict__["create"])
    for name in sdk_headless._PATCHED_CONTAINER_METHODS:
        monkeypatch.setattr(container_module.ExegolContainer, name, container_module.ExegolContainer.__dict__[name])


def _consume_until_ready(stream: Any) -> list[str]:
    """Drain a log stream the way the SDK's __start_container does: stop at the READY line."""
    lines: list[str] = []
    for line in stream:
        lines.append(line)
        if line == "READY":
            break
    return lines


@pytest.mark.usefixtures("sdk_restore")
def test_patch_installs_the_deadline_aware_log_stream_without_touching_the_utils_module() -> None:
    import exegol.model.ExegolContainer as container_module
    from exegol.utils import ContainerLogStream as utils_module
    from exegol.utils.ContainerLogStream import ContainerLogStream as Base

    sdk_headless.patch_exegol_sdk()

    installed = container_module.ContainerLogStream
    assert installed is not Base
    assert issubclass(installed, Base)
    # The utils module the base lives in is left untouched for any other SDK user.
    assert utils_module.ContainerLogStream is Base


@pytest.mark.usefixtures("fast_log_stream")
async def test_inside_a_deadline_the_wait_ends_before_the_deadline_after_logs_flowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdk_headless, "_DEADLINE_MARGIN_S", 0.3)
    container = _StubDockerContainer(polls=[[b"[PROGRESS] building\n"]])
    started = time.perf_counter()
    with anyio.fail_after(1.0):  # noqa: ASYNC100 - the synchronous stream must bound itself; no await inside
        stream = sdk_headless._DeadlineContainerLogStream(container, start_date=datetime.now(), timeout=2)
        lines = list(stream)
    elapsed = time.perf_counter() - started
    # One line was read, then the silent container's wait ended at the stop time (~0.7 s), before the deadline.
    assert lines == ["[PROGRESS] building"]
    assert elapsed < 1.0


@pytest.mark.usefixtures("fast_log_stream")
async def test_inside_a_deadline_a_partial_line_then_silence_still_ends_before_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdk_headless, "_DEADLINE_MARGIN_S", 0.3)
    # A chunk with no trailing newline (a partial line) flows, then the container goes silent.
    container = _StubDockerContainer(polls=[[b"[PROGRESS] partial-no-newline"]])
    started = time.perf_counter()
    with anyio.fail_after(1.0):  # noqa: ASYNC100 - the synchronous stream must bound itself; no await inside
        stream = sdk_headless._DeadlineContainerLogStream(container, start_date=datetime.now(), timeout=2)
        lines = list(stream)
    elapsed = time.perf_counter() - started
    assert lines == []  # the partial line is never completed, and the wait still ends
    assert elapsed < 1.0


@pytest.mark.usefixtures("fast_log_stream")
async def test_with_no_enclosing_deadline_the_subclass_matches_the_base_class() -> None:
    script = [[b"[INFO] starting\n", b"READY\n"]]
    base_stream = sdk_headless.ContainerLogStream(_StubDockerContainer(polls=script), start_date=datetime.now(), timeout=2)
    deadline_stream = sdk_headless._DeadlineContainerLogStream(
        _StubDockerContainer(polls=script), start_date=datetime.now(), timeout=2
    )
    # No anyio.fail_after wraps these: current_effective_deadline() is math.inf, so the subclass behaves like
    # the SDK class and yields exactly the same lines up to READY.
    assert _consume_until_ready(deadline_stream) == _consume_until_ready(base_stream)
    assert _consume_until_ready(sdk_headless._DeadlineContainerLogStream(
        _StubDockerContainer(polls=[[b"[INFO] starting\n", b"READY\n"]]), start_date=datetime.now(), timeout=2
    )) == ["[INFO] starting", "READY"]


async def test_the_two_second_legacy_detection_is_not_lengthened_by_a_later_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stub that never logs anything, inside a long (10 s) deadline. The SDK's own 2 s legacy-image detection
    # must still fire at ~2 s, not be stretched to the deadline.
    monkeypatch.setattr(sdk_headless, "_DEADLINE_MARGIN_S", 0.3)
    container = _StubDockerContainer(polls=[])
    started = time.perf_counter()
    with anyio.fail_after(10.0):  # noqa: ASYNC100 - the synchronous stream must bound itself; no await inside
        stream = sdk_headless._DeadlineContainerLogStream(container, start_date=datetime.now(), timeout=2)
        lines = list(stream)
    elapsed = time.perf_counter() - started
    assert lines == []
    assert 1.5 < elapsed < 5.0  # the SDK's 2 s detection, not the 10 s deadline (nor the ~9.7 s stop time)


def test_the_sdk_log_stream_still_exposes_both_name_mangled_timeout_attributes() -> None:
    # The subclass overrides these two name-mangled attributes as properties; if a future exegol release
    # renamed them, the properties would silently stop intercepting the SDK. This pins the names.
    stream = sdk_headless.ContainerLogStream(_StubDockerContainer(polls=[]), start_date=datetime.now(), timeout=2)
    assert hasattr(stream, "_ContainerLogStream__enable_timeout")
    assert hasattr(stream, "_ContainerLogStream__timeout_date")
