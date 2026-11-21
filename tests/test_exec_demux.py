"""Demuxed exec decoding (D-04) and per-request download progress (D-25), without Docker."""

import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from exegol_mcp.adapter import download_progress
from exegol_mcp.adapter.exec import build_exec_result, exec_demuxed
from exegol_mcp.adapter.offload import pull_limiter
from exegol_mcp.context.request import RequestState, bind_request
from exegol_mcp.models.exec import ExecResult

# --- build_exec_result: the D-04 decode contract ---------------------------------------------------------------


def test_stdout_and_stderr_stay_separate_with_the_real_exit_code() -> None:
    result = build_exec_result((b"out\n", b"err\n"), 3)

    assert result == ExecResult(stdout="out\n", stderr="err\n", exit_code=3)


def test_a_missing_exit_code_stays_none() -> None:
    result = build_exec_result((b"", None), None)

    assert result.exit_code is None
    assert result == ExecResult(stdout="", stderr="", exit_code=None)


def test_a_command_with_no_stdout_keeps_its_exit_code() -> None:
    result = build_exec_result((None, b"not found\n"), 127)

    assert result == ExecResult(stdout="", stderr="not found\n", exit_code=127)


def test_no_output_at_all_decodes_to_empty_streams() -> None:
    assert build_exec_result(None, 0) == ExecResult(stdout="", stderr="", exit_code=0)


def test_invalid_utf8_is_replaced_not_fatal() -> None:
    result = build_exec_result((b"ok \xff\xfe", b"bad \xc3"), 1)

    assert result.stdout == "ok ��"
    assert result.stderr == "bad �"


def test_whitespace_is_preserved_exactly() -> None:
    result = build_exec_result((b"  padded  \n\n", b"\tindented \r\n"), 0)

    assert result.stdout == "  padded  \n\n"
    assert result.stderr == "\tindented \r\n"


# --- exec_demuxed: the docker exec call shape -------------------------------------------------------------------


class _FakeDockerContainer:
    """Records the exec_run call and returns a docker-py style (exit_code, output) result."""

    def __init__(self, exit_code: int | None, output: Any) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.thread_ids: list[int] = []
        self._result = SimpleNamespace(exit_code=exit_code, output=output)

    def exec_run(self, cmd: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((cmd, kwargs))
        self.thread_ids.append(threading.get_ident())
        return self._result


def _exegol_container(docker_container: _FakeDockerContainer) -> SimpleNamespace:
    # ExegolContainer keeps the docker-py container in a private, name-mangled attribute.
    return SimpleNamespace(_ExegolContainer__container=docker_container)


async def test_exec_runs_a_demuxed_non_tty_exec_off_the_event_loop() -> None:
    docker_container = _FakeDockerContainer(exit_code=2, output=(b"partial\n", b"boom\n"))

    result = await exec_demuxed(_exegol_container(docker_container), "id; false")

    assert result == ExecResult(stdout="partial\n", stderr="boom\n", exit_code=2)
    [(cmd, kwargs)] = docker_container.calls
    assert kwargs["demux"] is True
    assert kwargs["stream"] is False
    assert kwargs["tty"] is False
    assert kwargs["detach"] is False
    # The command travels in the CMD variable, wrapped by the SDK's own zsh pre-routine.
    assert kwargs["environment"] == {"CMD": "id; false", "DISABLE_AUTO_UPDATE": "true"}
    assert cmd.startswith("zsh -c ")
    assert docker_container.thread_ids != [threading.get_ident()]


async def test_exec_reports_a_missing_exit_code_as_none() -> None:
    docker_container = _FakeDockerContainer(exit_code=None, output=(None, None))

    result = await exec_demuxed(_exegol_container(docker_container), "true")

    assert result == ExecResult(stdout="", stderr="", exit_code=None)


async def test_exec_names_the_sdk_internal_when_the_docker_handle_moved() -> None:
    with pytest.raises(RuntimeError, match="_ExegolContainer__container"):
        await exec_demuxed(SimpleNamespace(), "id")


# --- download progress: per-request sink, offloaded iteration, no division by zero ------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.reports: list[tuple[float, float | None, str | None]] = []

    async def __call__(self, progress: float, total: float | None, message: str | None) -> None:
        self.reports.append((progress, total, message))


def _pull(*layers: str) -> list[dict[str, str]]:
    lines = [{"status": "Pulling from nwodtuhs/exegol"}]
    lines += [{"status": "Pulling fs layer", "id": layer} for layer in layers]
    lines += [{"status": "Downloading", "id": layer} for layer in layers]
    lines += [{"status": "Download complete", "id": layer} for layer in layers]
    lines += [{"status": "Pull complete", "id": layer} for layer in layers]
    lines += [{"status": "Status: Downloaded newer image for nwodtuhs/exegol:free"}]
    return lines


async def test_progress_goes_to_the_requests_own_sink() -> None:
    recorder = _Recorder()

    with bind_request(RequestState(request_id="r1", progress=recorder)):
        await download_progress.download_docker_layer(iter(_pull("a", "b")))

    assert (25, 100, "Downloading layers: 1 / 2") in recorder.reports
    assert (75, 100, "Extracting layers: 1 / 2") in recorder.reports
    assert recorder.reports[-1] == (100, 100, "Download complete")
    values = [progress for progress, _, _ in recorder.reports]
    assert values == sorted(values)


async def test_a_layer_completing_before_any_layer_is_announced_does_not_divide_by_zero() -> None:
    recorder = _Recorder()
    stream = [{"status": "Download complete", "id": "orphan"}, {"status": "Pull complete", "id": "orphan"}]

    with bind_request(RequestState(progress=recorder)):
        await download_progress.download_docker_layer(iter(stream))

    assert recorder.reports[-1] == (100, 100, "Download complete")


async def test_a_docker_error_line_raises() -> None:
    stream = [{"status": "Pulling fs layer", "id": "a"}, {"error": "unauthorized: authentication required"}]

    with pytest.raises(RuntimeError, match="unauthorized: authentication required"):
        await download_progress.download_docker_layer(iter(stream))


async def test_a_download_without_a_bound_request_reports_nothing() -> None:
    await download_progress.download_docker_layer(iter(_pull("a")))


async def test_each_line_is_pulled_off_the_loop_on_the_pull_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    limiters: list[Any] = []
    real_run_blocking = download_progress.run_blocking

    async def recording_run_blocking(fn: Any, *args: Any, limiter: Any = None) -> Any:
        limiters.append(limiter)
        return await real_run_blocking(fn, *args, limiter=limiter)

    monkeypatch.setattr(download_progress, "run_blocking", recording_run_blocking)
    loop_thread = threading.get_ident()
    pulled_in: list[int] = []

    def stream() -> Iterator[dict[str, str]]:
        for line in _pull("a"):
            pulled_in.append(threading.get_ident())
            yield line

    await download_progress.download_docker_layer(stream())

    assert limiters and all(limiter is pull_limiter for limiter in limiters)
    assert pulled_in and loop_thread not in pulled_in


async def test_concurrent_downloads_each_report_only_to_their_own_caller() -> None:
    async def download(recorder: _Recorder, layers: tuple[str, ...]) -> None:
        with bind_request(RequestState(progress=recorder)):
            await download_progress.download_docker_layer(iter(_pull(*layers)))

    # What each pull reports when it runs alone.
    alone_first, alone_second = _Recorder(), _Recorder()
    await download(alone_first, ("a",))
    await download(alone_second, ("b", "c", "d"))

    # The same pulls interleaved (every line crosses a worker thread, so the two tasks alternate).
    first, second = _Recorder(), _Recorder()
    async with anyio.create_task_group() as tg:
        tg.start_soon(download, first, ("a",))
        tg.start_soon(download, second, ("b", "c", "d"))

    assert first.reports == alone_first.reports
    assert second.reports == alone_second.reports
