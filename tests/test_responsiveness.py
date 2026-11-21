"""Responsiveness regression for the EXEC-01 offload fix (QUAL-02), on all three transports (D-24).

While a ~2s blocking exec is in flight, the server must keep answering other requests. The fake's exec does a
synchronous wait, like a docker exec, and goes through run_blocking exactly as the real adapter does.

The three legs:
- in-memory: the in-process Client against build_server(); the long call awaits the fake's exec_command
  directly on the same event loop, so an un-offloaded call would freeze that loop and tools/list with it;
- streamable HTTP: the real HTTP app (SDK streamable-HTTP app behind the bearer gate) under uvicorn on
  127.0.0.1, in its own thread and event loop, with the long call made through the exec_command tool;
- stdio: the real entry point as a subprocess (tests.fake_server, Docker-free), driven by the SDK stdio client,
  with the long call made through the exec_command tool.

Gate meaning: if the blocking work runs on the event loop instead of through run_blocking, every concurrent
response waits for the rest of the block and the < 1.0s assertions fail. Each leg keeps that inversion permanent:
the same measurement, run with the exec on the loop (offload_exec=False / EXEGOL_MCP_TEST_OFFLOAD=0), must report
the stall.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from exegol_mcp.app import build_server
from exegol_mcp.bootstrap.cli import HTTP_HOST, MCP_PATH
from exegol_mcp.bootstrap.transports import build_http_app
from tests.fake_server import stdio_server_parameters
from tests.fakes import FakeExegolAdapter
from tests.http_harness import serve_in_thread

BLOCK_SECONDS = 2.0
# Generous against CI jitter, but half the block duration: an un-offloaded exec pushes a concurrent response
# to ~BLOCK_SECONDS, so it cannot pass.
MAX_CONCURRENT_LATENCY = 1.0


async def test_concurrent_requests_stay_fast_while_exec_blocks() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=BLOCK_SECONDS)
    async with Client(build_server(None, adapter)) as client, anyio.create_task_group() as tg:
        tg.start_soon(adapter.exec_command, "exegol-lab", "sleep 2")
        await adapter.exec_started.wait()
        await anyio.sleep(0.05)  # let the worker thread enter its blocking wait
        assert adapter.exec_started_at is not None

        started = time.perf_counter()
        tools = await client.list_tools()
        list_tools_latency = time.perf_counter() - started
        assert "list_containers" in {tool.name for tool in tools.tools}

        started = time.perf_counter()
        result = await client.call_tool("list_containers", {})
        list_containers_latency = time.perf_counter() - started
        assert not result.is_error, result.content

        # Both responses arrived while the exec was still blocked, not after it.
        assert adapter.exec_finished_at is None
        since_exec_start = time.perf_counter() - adapter.exec_started_at
        adapter.exec_release.set()

    assert list_tools_latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, list_tools_latency
    assert list_containers_latency < MAX_CONCURRENT_LATENCY, list_containers_latency
    assert since_exec_start < MAX_CONCURRENT_LATENCY, since_exec_start


async def test_blocking_exec_on_the_loop_is_detected() -> None:
    """Inversion check: an un-offloaded exec stalls a concurrent tools/list for the whole block."""
    adapter = FakeExegolAdapter(exec_block_seconds=BLOCK_SECONDS, offload_exec=False)
    async with Client(build_server(None, adapter)) as client, anyio.create_task_group() as tg:
        tg.start_soon(adapter.exec_command, "exegol-lab", "sleep 2")
        await adapter.exec_started.wait()
        await client.list_tools()
        assert adapter.exec_started_at is not None
        since_exec_start = time.perf_counter() - adapter.exec_started_at
        stalled_until_exec_finished = adapter.exec_finished_at is not None

    assert stalled_until_exec_finished
    assert since_exec_start >= MAX_CONCURRENT_LATENCY, since_exec_start


# --- Transport legs (HTTP and stdio): the long call goes through the exec_command tool ----------------------------

ERAS = ["legacy", "2026-07-28"]
SECRET = "responsiveness-test-secret"
# How long the test waits after sending the long call before measuring, so the exec is already blocking.
SETTLE_SECONDS = 0.4


@dataclass
class Timings:
    list_tools_latency: float
    list_containers_latency: float
    list_sent_at: float
    answered_at: float
    exec_done_at: float
    exec_sent_at: float


async def _measure_during_exec(client: Client) -> Timings:
    """Start a blocking exec_command call, then time tools/list and list_containers while it is in flight."""
    done: dict[str, float] = {}

    async def long_call() -> None:
        result = await client.call_tool("exec_command", {"container_name": "exegol-lab", "command": "sleep 2"})
        assert not result.is_error, result.content
        done["exec"] = time.perf_counter()

    # Warm up first: the first request of a session can carry one-off client work (tool-list caching for output
    # validation, era discovery), which would otherwise delay the exec past the moment tools/list is measured.
    await client.list_tools()
    await client.call_tool("server_info", {})

    async with anyio.create_task_group() as tg:
        sent_at = time.perf_counter()
        tg.start_soon(long_call)
        await anyio.sleep(SETTLE_SECONDS)

        started = list_sent_at = time.perf_counter()
        tools = await client.list_tools()
        list_tools_latency = time.perf_counter() - started
        assert "exec_command" in {tool.name for tool in tools.tools}

        started = time.perf_counter()
        result = await client.call_tool("list_containers", {})
        list_containers_latency = time.perf_counter() - started
        assert not result.is_error, result.content
        answered_at = time.perf_counter()

    return Timings(list_tools_latency, list_containers_latency, list_sent_at, answered_at, done["exec"], sent_at)


def _assert_responsive(timings: Timings) -> None:
    assert timings.list_tools_latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, timings
    assert timings.list_containers_latency < MAX_CONCURRENT_LATENCY, timings
    # The exec was already blocking when tools/list was sent: it cannot have started later than exec_done -
    # BLOCK_SECONDS. Both answers arrived before it finished, and it really blocked for its full duration.
    assert timings.list_sent_at >= timings.exec_done_at - BLOCK_SECONDS, timings
    assert timings.answered_at < timings.exec_done_at, timings
    assert timings.exec_done_at - timings.exec_sent_at >= BLOCK_SECONDS, timings


def _assert_stalled(timings: Timings) -> None:
    """Inversion: with the exec on the loop, tools/list waits for the rest of the block."""
    assert timings.list_tools_latency >= MAX_CONCURRENT_LATENCY, timings


async def _measure_over_http(adapter: FakeExegolAdapter, mode: str) -> Timings:
    app = build_http_app(build_server(None, adapter), auth="bearer", secret=SECRET)
    with serve_in_thread(app) as port:
        url = f"http://{HTTP_HOST}:{port}{MCP_PATH}"
        async with (
            httpx2.AsyncClient(headers={"Authorization": f"Bearer {SECRET}"}) as http_client,
            Client(streamable_http_client(url, http_client=http_client), mode=mode) as client,
        ):
            return await _measure_during_exec(client)


@pytest.mark.parametrize("mode", ERAS)
async def test_http_requests_stay_fast_while_exec_blocks(mode: str) -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=BLOCK_SECONDS)
    timings = await _measure_over_http(adapter, mode)
    _assert_responsive(timings)
    assert adapter.exec_commands == ["sleep 2"]


async def test_http_blocking_exec_on_the_loop_is_detected() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=BLOCK_SECONDS, offload_exec=False)
    _assert_stalled(await _measure_over_http(adapter, "legacy"))


async def _measure_over_stdio(home: Path, mode: str, offload: bool = True) -> Timings:
    params = stdio_server_parameters(
        home, EXEGOL_MCP_TEST_EXEC_BLOCK=str(BLOCK_SECONDS), EXEGOL_MCP_TEST_OFFLOAD="1" if offload else "0"
    )
    async with Client(params, mode=mode) as client:
        return await _measure_during_exec(client)


@pytest.mark.parametrize("mode", ERAS)
async def test_stdio_requests_stay_fast_while_exec_blocks(tmp_path: Path, mode: str) -> None:
    _assert_responsive(await _measure_over_stdio(tmp_path, mode))


async def test_stdio_blocking_exec_on_the_loop_is_detected(tmp_path: Path) -> None:
    _assert_stalled(await _measure_over_stdio(tmp_path, "legacy", offload=False))
