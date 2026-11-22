"""INT-01: container start, stop, create post-setup and removal keep the event loop free, without Docker.

These tests drive the REAL, patched ExegolContainer methods over a stub docker-py handle whose calls block a real
worker thread, like tests/test_responsiveness.py and the CR-01 removal tests in tests/test_lifecycle_sdk.py. A
concurrent MCP client (tools/list, list_containers) is the probe: while a docker call blocks, it must keep answering
in under MAX_CONCURRENT_LATENCY. Each responsiveness test has an inversion that puts Exegol's own method back and
shows the probe detects the old on-loop behaviour (QUAL-02).
"""

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from exegol.exceptions.ExegolExceptions import ObjectNotFound
from mcp import Client

from exegol_mcp.adapter import real, sdk_headless
from exegol_mcp.adapter.port import MANAGED_LABEL, MANAGED_LABEL_VALUE, RUN_ID_LABEL
from exegol_mcp.adapter.real import RealExegolAdapter
from exegol_mcp.app import build_server
from exegol_mcp.services import containers as containers_service
from tests.fakes import FakeExegolAdapter
from tests.test_lifecycle_sdk import _wait_for_thread_event
from tests.test_responsiveness import BLOCK_SECONDS, MAX_CONCURRENT_LATENCY

_CALL_NAMES = ("start", "stop")

# What a container prints while it boots: one startup line, then the READY marker the SDK's start waits for.
_BOOT_LOG = [b"[INFO] booting\n", b"READY\n"]


class _BlockingHandle:
    """The docker-py Container behind a real ExegolContainer: every call recorded, start and stop able to block.

    start() and stop() record their thread and keyword arguments, set their `entered` Event, wait on their `release`
    Event for at most their block seconds (like a long docker stop), then flip the container's Status and set
    `done`. A test therefore knows when the worker thread is inside the blocking call, can let it go, and can wait
    until it finished, so no thread outlives its test.
    """

    def __init__(self, name: str = "lab", status: str = "exited") -> None:
        self.name = name
        self.id = f"{name}-id"
        self.labels = {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "abc"}
        self.attrs: dict[str, Any] = {"State": {"Status": status}}
        # The state-changing docker calls, in call order.
        self.calls: list[str] = []
        self.threads: dict[str, list[int]] = {}
        self.kwargs: dict[str, list[dict[str, Any]]] = {}
        self.block_seconds: dict[str, float] = {name_: 0.0 for name_ in _CALL_NAMES}
        self.entered = {name_: threading.Event() for name_ in _CALL_NAMES}
        self.release = {name_: threading.Event() for name_ in _CALL_NAMES}
        self.done = {name_: threading.Event() for name_ in _CALL_NAMES}
        # perf_counter timestamps, so a test can tell whether a concurrent request answered during the block.
        self.started_at: dict[str, float] = {}
        self.finished_at: dict[str, float] = {}
        # Every logs() read, and the chunks the successive reads return (the start tests read the READY line).
        self.log_reads = 0
        self.log_script: list[list[bytes]] = [list(_BOOT_LOG)]
        self.log_threads: list[int] = []
        # Every docker call and recorded SDK step in order, logs() reads included (calls holds only state changes).
        self.events: list[str] = []
        # Raised by start() before it blocks, like docker refusing the start (an APIError for an allocated port).
        self.start_error: Exception | None = None

    def reload(self) -> None:
        return None

    def _blocking_call(self, name: str, status_after: str, kwargs: dict[str, Any]) -> None:
        self.started_at[name] = time.perf_counter()
        self.threads.setdefault(name, []).append(threading.get_ident())
        self.kwargs.setdefault(name, []).append(kwargs)
        self.events.append(name)
        self.entered[name].set()
        self.release[name].wait(self.block_seconds[name])
        self.attrs["State"]["Status"] = status_after
        self.calls.append(name)
        self.finished_at[name] = time.perf_counter()
        self.done[name].set()

    def stop(self, **kwargs: Any) -> None:
        self._blocking_call("stop", "exited", kwargs)

    def start(self, **kwargs: Any) -> None:
        if self.start_error is not None:
            raise self.start_error
        self._blocking_call("start", "running", kwargs)

    def logs(self, stream: bool = False, follow: bool = False, since: Any = None, until: Any = None) -> Iterator[bytes]:
        self.log_reads += 1
        self.log_threads.append(threading.get_ident())
        self.events.append("logs")
        if self.log_script:
            return iter(self.log_script.pop(0))
        return iter(())

    def release_all(self) -> None:
        """Let every blocked call go, then wait for each call that started, so no worker thread outlives the test."""
        for name in _CALL_NAMES:
            self.release[name].set()
        for name in _CALL_NAMES:
            if self.entered[name].is_set():
                assert self.done[name].wait(5.0), f"the docker {name} never finished"


class _OffloadConfig:
    """The ContainerConfig getters the SDK's start, stop and post-setup read."""

    legacy_entrypoint = False

    def getVpnConfigPath(self) -> None:
        return None

    def getVolumes(self) -> list[Any]:
        return []

    def isGUIEnable(self) -> bool:
        return False


def _offload_container(handle: _BlockingHandle, config: Any = None) -> Any:
    """A real ExegolContainer over the stub handle, built without its docker-reading __init__."""
    from exegol.model.ExegolContainer import ExegolContainer

    container = object.__new__(ExegolContainer)
    container._ExegolContainer__container = handle  # type: ignore[attr-defined]
    container.config = _OffloadConfig() if config is None else config
    container.name = handle.name
    # Both are set by the SDK's __init__ and read by its start path (post-setup runs once per object).
    container._ExegolContainer__post_start_applied = False  # type: ignore[attr-defined]
    container._ExegolContainer__xhost_applied = False  # type: ignore[attr-defined]
    return container


@pytest.fixture
def sdk_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore every SDK global patch_exegol_sdk() mutates, including each patched ExegolContainer method."""
    import exegol.model.ExegolContainer as container_module
    import exegol.profile.ProfileAskUser as ask_module
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
    monkeypatch.setattr(ask_module, "stdinCanAnswer", ask_module.stdinCanAnswer)
    monkeypatch.setattr(ContainerCollection, "create", ContainerCollection.__dict__["create"])
    for name in sdk_headless._PATCHED_CONTAINER_METHODS:
        monkeypatch.setattr(container_module.ExegolContainer, name, container_module.ExegolContainer.__dict__[name])
    monkeypatch.setattr(sdk_headless, "_parameters", sdk_headless._parameters)


@pytest.fixture
def patched(sdk_restore: None) -> Any:
    """The headless SDK with the operator's --accept-eula given, as the server runs it."""
    sdk_headless.patch_exegol_sdk()
    sdk_headless.set_accept_eula(True)
    return sdk_headless._parameters


@pytest.fixture(autouse=True)
def fresh_console_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own console lock: a contended asyncio.Lock binds to the loop of the test that used it."""
    import exegol.console.ExegolStatus as status_module

    monkeypatch.setattr(status_module, "ConsoleLock", asyncio.Lock())


@pytest.fixture
def fast_log_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the SDK ContainerLogStream poll loop fast: a 1 s time.sleep becomes a 10 ms real sleep."""
    from exegol.utils import ContainerLogStream as log_stream_module

    monkeypatch.setattr(log_stream_module, "time", SimpleNamespace(sleep=lambda _s: time.sleep(0.01)))


@pytest.fixture
def docker_containers(patched: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The containers the adapter's DockerUtils finds, by name: a test adds its ExegolContainer to the dict."""
    containers: dict[str, Any] = {}

    def get_container(tag: str) -> Any:
        if tag in containers:
            return containers[tag]
        raise ObjectNotFound

    monkeypatch.setattr(real, "DockerUtils", lambda: SimpleNamespace(getContainer=get_container))
    return containers


async def _stop_through_the_service(timeout_s: float | None = None) -> bool:
    return await containers_service.stop_container(RealExegolAdapter(), "lab", timeout_s=timeout_s)


# --- stop_container: the docker stop runs off the loop (INT-01) --------------------------------------------------


async def test_stop_container_keeps_tools_list_fast_while_docker_stops(docker_containers: dict[str, Any]) -> None:
    handle = _BlockingHandle(status="running")
    handle.block_seconds["stop"] = BLOCK_SECONDS
    docker_containers["lab"] = _offload_container(handle)
    loop_thread = threading.get_ident()
    result: dict[str, Any] = {}

    async def stop() -> None:
        result["stopped"] = await _stop_through_the_service()

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(stop)
            await _wait_for_thread_event(handle.entered["stop"])
            await anyio.sleep(0.05)  # let the worker thread settle into its blocking wait
            started = time.perf_counter()
            await client.list_tools()
            list_tools_latency = time.perf_counter() - started
            started = time.perf_counter()
            listed = await client.call_tool("list_containers", {})
            list_containers_latency = time.perf_counter() - started
            answered_during_the_stop = not handle.done["stop"].is_set()
            handle.release["stop"].set()
    finally:
        handle.release_all()

    assert not listed.is_error, listed.content
    assert list_tools_latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, list_tools_latency
    assert list_containers_latency < MAX_CONCURRENT_LATENCY, list_containers_latency
    assert answered_during_the_stop
    # Exegol's stop ran its docker call in a worker thread, with exactly Exegol's default timeout as a keyword.
    [stop_thread] = handle.threads["stop"]
    assert stop_thread != loop_thread
    assert handle.kwargs["stop"] == [{"timeout": 10}]
    assert result["stopped"] is True
    assert handle.attrs["State"]["Status"] == "exited"


async def test_a_blocked_stop_returns_the_timeout_error_on_time(docker_containers: dict[str, Any]) -> None:
    handle = _BlockingHandle(status="running")
    handle.block_seconds["stop"] = 5.0
    docker_containers["lab"] = _offload_container(handle)

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match=r"did not finish stopping within 0\.3 s") as error:
            await _stop_through_the_service(timeout_s=0.3)
        elapsed = time.perf_counter() - started
        # The abandoned worker thread is still inside the docker stop: it is never killed, it finishes on its own.
        still_blocked = not handle.done["stop"].is_set()
    finally:
        handle.release_all()

    assert elapsed < 2.0, elapsed
    assert "list_containers" in str(error.value)
    assert still_blocked


async def test_the_sdk_stop_on_the_loop_is_detected(
    docker_containers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inversion: with Exegol's own stop back, the docker stop blocks the loop and tools/list waits for it."""
    assert sdk_headless._SDK_STOP is not None
    from exegol.model.ExegolContainer import ExegolContainer

    monkeypatch.setattr(ExegolContainer, "stop", sdk_headless._SDK_STOP)
    handle = _BlockingHandle(status="running")
    handle.block_seconds["stop"] = BLOCK_SECONDS
    docker_containers["lab"] = _offload_container(handle)

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(_stop_through_the_service)
            await _wait_for_thread_event(handle.entered["stop"])
            await client.list_tools()
            assert handle.started_at["stop"] is not None
            since_stop_start = time.perf_counter() - handle.started_at["stop"]
            stalled_until_the_stop_finished = handle.done["stop"].is_set()
    finally:
        handle.release_all()

    assert stalled_until_the_stop_finished
    assert since_stop_start >= MAX_CONCURRENT_LATENCY, since_stop_start


async def test_stopping_a_stopped_container_makes_no_docker_call(patched: Any) -> None:
    handle = _BlockingHandle(status="exited")
    container = _offload_container(handle)

    await container.stop()

    assert handle.calls == []
    assert handle.kwargs == {}


def test_patch_installs_the_stop_mirror_idempotently(patched: Any) -> None:
    import inspect

    from exegol.model.ExegolContainer import ExegolContainer

    sdk_headless.patch_exegol_sdk()
    sdk_headless.patch_exegol_sdk()

    assert ExegolContainer.stop is sdk_headless._stop_offloaded
    # Exegol's own stop is kept at import, so the drift pins below read the SDK's source, not the mirror's.
    assert sdk_headless._SDK_STOP is not None
    assert sdk_headless._SDK_STOP.__qualname__ == "ExegolContainer.stop"
    assert "stop" in sdk_headless._PATCHED_CONTAINER_METHODS
    source = inspect.getsource(sdk_headless._SDK_STOP)
    assert "self.__container.stop(timeout=timeout)" in source
    assert 'ExegolStatus(f"Waiting to stop ({timeout}s timeout)", spinner_style="blue")' in source
    assert 'logger.info(f"Stopping container {self.name}")' in source


# --- start_container: the docker start and the startup log polls run off the loop (INT-01) -----------------------


async def _start_through_the_service(timeout_s: float | None = None) -> bool:
    return await containers_service.start_container(RealExegolAdapter(), "lab", timeout_s=timeout_s)


async def test_start_container_keeps_tools_list_fast_while_docker_starts(docker_containers: dict[str, Any]) -> None:
    handle = _BlockingHandle(status="exited")
    handle.block_seconds["start"] = BLOCK_SECONDS
    docker_containers["lab"] = _offload_container(handle)
    loop_thread = threading.get_ident()
    result: dict[str, Any] = {}

    async def start() -> None:
        result["started"] = await _start_through_the_service()

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(start)
            await _wait_for_thread_event(handle.entered["start"])
            await anyio.sleep(0.05)  # let the worker thread settle into its blocking wait
            started = time.perf_counter()
            await client.list_tools()
            list_tools_latency = time.perf_counter() - started
            started = time.perf_counter()
            listed = await client.call_tool("list_containers", {})
            list_containers_latency = time.perf_counter() - started
            answered_during_the_start = not handle.done["start"].is_set()
            handle.release["start"].set()
    finally:
        handle.release_all()

    assert not listed.is_error, listed.content
    assert list_tools_latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, list_tools_latency
    assert list_containers_latency < MAX_CONCURRENT_LATENCY, list_containers_latency
    assert answered_during_the_start
    # Exegol's start ran its docker call and every startup log poll in worker threads, never on the loop.
    [start_thread] = handle.threads["start"]
    assert start_thread != loop_thread
    assert handle.log_reads >= 1
    assert loop_thread not in handle.log_threads
    assert result["started"] is True
    assert handle.attrs["State"]["Status"] == "running"


async def test_a_blocked_start_returns_the_timeout_error_on_time(docker_containers: dict[str, Any]) -> None:
    handle = _BlockingHandle(status="exited")
    handle.block_seconds["start"] = 5.0
    docker_containers["lab"] = _offload_container(handle)

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match=r"did not finish starting within 0\.3 s") as error:
            await _start_through_the_service(timeout_s=0.3)
        elapsed = time.perf_counter() - started
        # The abandoned worker thread is still inside the docker start: it is never killed, it finishes on its own.
        still_blocked = not handle.done["start"].is_set()
    finally:
        handle.release_all()

    assert elapsed < 2.0, elapsed
    assert "list_containers" in str(error.value)
    assert still_blocked


async def test_the_sdk_start_on_the_loop_is_detected(
    docker_containers: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inversion: with Exegol's own __start_container back, the docker start blocks the loop and tools/list waits."""
    assert sdk_headless._SDK_START_CONTAINER is not None
    from exegol.model.ExegolContainer import ExegolContainer

    monkeypatch.setattr(ExegolContainer, "_ExegolContainer__start_container", sdk_headless._SDK_START_CONTAINER)
    handle = _BlockingHandle(status="exited")
    handle.block_seconds["start"] = BLOCK_SECONDS
    docker_containers["lab"] = _offload_container(handle)

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(_start_through_the_service)
            await _wait_for_thread_event(handle.entered["start"])
            await client.list_tools()
            assert handle.started_at["start"] is not None
            since_start_began = time.perf_counter() - handle.started_at["start"]
            stalled_until_the_start_finished = handle.done["start"].is_set()
    finally:
        handle.release_all()

    assert stalled_until_the_start_finished
    assert since_start_began >= MAX_CONCURRENT_LATENCY, since_start_began


def test_patch_installs_the_start_mirrors_idempotently(patched: Any) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    sdk_headless.patch_exegol_sdk()
    sdk_headless.patch_exegol_sdk()

    assert ExegolContainer.start is sdk_headless._start_offloaded
    installed_start_container = ExegolContainer._ExegolContainer__start_container  # type: ignore[attr-defined]
    assert installed_start_container is sdk_headless._start_container_offloaded
    # Exegol's own methods are kept at import, so the drift pins below read the SDK's source, not the mirrors'.
    assert sdk_headless._SDK_START_CONTAINER is not None
    assert sdk_headless._SDK_START is not None
    assert sdk_headless._SDK_START_CONTAINER.__qualname__ == "ExegolContainer.__start_container"
    assert sdk_headless._SDK_START.__qualname__ == "ExegolContainer.start"
    assert {"start", "_ExegolContainer__start_container"} <= set(sdk_headless._PATCHED_CONTAINER_METHODS)


def test_the_sdk_start_still_matches_the_mirrored_lines() -> None:
    import inspect

    from exegol.model.ExegolContainer import ExegolContainer

    assert sdk_headless._SDK_START_CONTAINER is not None
    assert sdk_headless._SDK_START is not None
    start_container = inspect.getsource(sdk_headless._SDK_START_CONTAINER)
    for line in (
        "self.__container.start()",
        "for line in ContainerLogStream(self.__container, start_date=start_date, timeout=2):",
        'if line == "READY":',
        "if not self.config.legacy_entrypoint:",
        'async with ExegolStatus(f"Waiting to start {self.name}", spinner_style="blue") as progress:',
        'progress.update(status=f"[blue][Startup][/blue] {line}")',
        "except APIError as e:",
        "except ReadTimeout:",
        'logger.verbose(f"Container started in {(datetime.now() - start_date).seconds} seconds")',
        "Docker raised a critical error when starting the container [green]{self.name}[/green], error message is:",
        "Received a timeout error from docker... Unable to start the container, retry later.",
        *(
            f"line.startswith('[{prefix}]')"
            for prefix in ("INFO", "VERBOSE", "ADVANCED", "DEBUG", "WARNING", "ERROR", "SUCCESS", "PROGRESS")
        ),
    ):
        assert line in start_container, line

    start = inspect.getsource(sdk_headless._SDK_START)
    ordered = (
        "if not self.isRunning():",
        "self.__preStartChecks()",
        'logger.info(f"Starting container {self.name}")',
        "await self.__start_container()",
        "await self.__postStartSetup()",
    )
    positions = [start.index(line) for line in ordered]
    assert positions == sorted(positions)

    x11 = inspect.getsource(ExegolContainer._ExegolContainer__applyX11ACLs)  # type: ignore[attr-defined]
    assert 'await self.exec(f"xauth add {xauthEntry}", as_daemon=False, quiet=True)' in x11


# --- auto_start and the SDK's startup edge cases (INT-01, LIFE-07) ------------------------------------------------


async def test_auto_start_keeps_tools_list_fast_while_docker_starts(docker_containers: dict[str, Any]) -> None:
    handle = _BlockingHandle(status="exited")
    handle.block_seconds["start"] = BLOCK_SECONDS
    docker_containers["lab"] = _offload_container(handle)

    async def auto_start() -> None:
        await containers_service.ensure_running(RealExegolAdapter(), "lab", auto_start=True)

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(auto_start)
            await _wait_for_thread_event(handle.entered["start"])
            await anyio.sleep(0.05)  # let the worker thread settle into its blocking wait
            started = time.perf_counter()
            await client.list_tools()
            latency = time.perf_counter() - started
            answered_during_the_start = not handle.done["start"].is_set()
            handle.release["start"].set()
    finally:
        handle.release_all()

    assert latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, latency
    assert answered_during_the_start
    assert handle.attrs["State"]["Status"] == "running"


async def test_start_reads_the_log_stream_until_ready_off_the_loop(patched: Any, fast_log_polls: None) -> None:
    handle = _BlockingHandle(status="exited")
    # READY arrives on the second poll; the third chunk must never be read.
    handle.log_script = [[b"[INFO] one\n"], [b"[PROGRESS] two\n", b"READY\n"], [b"[INFO] never\n"]]
    container = _offload_container(handle)

    await container.start()

    assert handle.log_reads == 2
    assert handle.log_script == [[b"[INFO] never\n"]]
    assert threading.get_ident() not in handle.log_threads
    assert handle.calls == ["start"]


async def test_an_image_that_logs_nothing_ends_the_wait_at_the_sdk_timeout(patched: Any) -> None:
    handle = _BlockingHandle(status="exited")
    handle.log_script = []  # the image supports no startup status: every poll is empty
    container = _offload_container(handle)

    started = time.perf_counter()
    await container.start()
    elapsed = time.perf_counter() - started

    # The SDK's own 2 s legacy-image detection ends the wait, neither at once nor stretched.
    assert 1.5 < elapsed < 5.0, elapsed
    assert handle.log_reads >= 1
    assert handle.attrs["State"]["Status"] == "running"


async def test_a_legacy_entrypoint_container_reads_no_logs(patched: Any) -> None:
    handle = _BlockingHandle(status="exited")
    config = _OffloadConfig()
    config.legacy_entrypoint = True
    container = _offload_container(handle, config)

    await container.start()

    assert handle.calls == ["start"]
    assert handle.log_reads == 0


async def test_a_docker_api_error_on_start_raises_runtime_error(patched: Any) -> None:
    import docker.errors

    handle = _BlockingHandle(status="exited")
    handle.start_error = docker.errors.APIError("boom", explanation="port is already allocated")
    container = _offload_container(handle)

    with pytest.raises(RuntimeError, match="port is already allocated"):
        await container.start()

    assert handle.log_reads == 0
    assert handle.calls == []


async def test_post_start_setup_runs_with_exec_offloaded_then_restored(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    seen: list[bool] = []

    async def post_start_setup(self: Any) -> None:
        seen.append("exec" in vars(self))

    monkeypatch.setattr(ExegolContainer, "_ExegolContainer__postStartSetup", post_start_setup)
    container = _offload_container(_BlockingHandle(status="exited"))

    await container.start()

    # The setup ran once, with the offloaded exec shadow bound, and the shadow is gone afterwards.
    assert seen == [True]
    assert "exec" not in vars(container)


async def test_start_runs_the_sdk_steps_in_order(patched: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    handle = _BlockingHandle(status="exited")

    def pre_start_checks(self: Any) -> None:
        handle.events.append("pre_start_checks")

    async def post_start_setup(self: Any) -> None:
        handle.events.append("post_start_setup")

    monkeypatch.setattr(ExegolContainer, "_ExegolContainer__preStartChecks", pre_start_checks)
    monkeypatch.setattr(ExegolContainer, "_ExegolContainer__postStartSetup", post_start_setup)
    container = _offload_container(handle)

    await container.start()

    # Running-state read (no event), pre-start checks, docker start, log polls until READY, post-start setup.
    assert handle.events[:2] == ["pre_start_checks", "start"]
    assert "logs" in handle.events[2:-1]
    assert handle.events[-1] == "post_start_setup"


async def test_an_abandoned_log_poll_ends_after_cancellation(patched: Any, fast_log_polls: None) -> None:
    handle = _BlockingHandle(status="exited")
    handle.log_script = [[b"[INFO] one\n"]]  # one line, then silence: with no deadline the stream polls forever
    container = _offload_container(handle)
    scope = anyio.CancelScope()  # no deadline: only stop_now can end the abandoned poll thread

    async def start() -> None:
        with scope:
            await container.start()

    async with anyio.create_task_group() as tg:
        tg.start_soon(start)
        with anyio.fail_after(5.0):
            while handle.log_reads < 3:
                await anyio.sleep(0.01)
        scope.cancel()
    reads_at_cancel = handle.log_reads
    await anyio.sleep(1.0)

    # At 10 ms a poll, an unstopped thread would read about 100 more times in that second.
    assert handle.log_reads - reads_at_cancel <= 3, handle.log_reads - reads_at_cancel


async def test_a_start_and_a_stop_of_two_containers_run_together(docker_containers: dict[str, Any]) -> None:
    starting = _BlockingHandle(name="lab", status="exited")
    starting.block_seconds["start"] = 0.3
    stopping = _BlockingHandle(name="lab2", status="running")
    stopping.block_seconds["stop"] = 0.3
    docker_containers["lab"] = _offload_container(starting)
    docker_containers["lab2"] = _offload_container(stopping)
    results: dict[str, bool] = {}

    async def start() -> None:
        results["started"] = await containers_service.start_container(RealExegolAdapter(), "lab")

    async def stop() -> None:
        results["stopped"] = await containers_service.stop_container(RealExegolAdapter(), "lab2")

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(start)
            tg.start_soon(stop)
            started = time.perf_counter()
            await client.list_tools()
            latency = time.perf_counter() - started
    finally:
        starting.release_all()
        stopping.release_all()

    assert results == {"started": True, "stopped": True}
    assert latency < MAX_CONCURRENT_LATENCY, latency
    # Both status displays hold the SDK's one console lock for their whole docker wait, so they run in turn.
    first, second = sorted(
        [
            (starting.started_at["start"], starting.finished_at["start"]),
            (stopping.started_at["stop"], stopping.finished_at["stop"]),
        ]
    )
    assert first[1] <= second[0], (first, second)


async def test_starting_a_running_container_makes_no_docker_call(patched: Any) -> None:
    handle = _BlockingHandle(status="running")
    container = _offload_container(handle)

    await container.start()

    assert handle.calls == []
    assert handle.kwargs == {}
    assert handle.log_reads == 0


async def test_an_unknown_name_is_refused_before_any_docker_call(docker_containers: dict[str, Any]) -> None:
    other = _BlockingHandle(name="other", status="exited")
    docker_containers["other"] = _offload_container(other)

    with pytest.raises(ValueError, match="not found"):
        await _start_through_the_service()  # asks for "lab", which does not exist

    assert other.events == []
    assert other.calls == []
    assert other.log_reads == 0
