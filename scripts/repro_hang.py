"""One-off reproduction of the v1 event-loop hang, measured before/after the v2 offload fix (D-23).

    uv run python scripts/repro_hang.py [--block 3] [--runs 3]

What it measures: while one exec_command call runs a command that takes --block seconds, how long a concurrent
tools/list takes to answer. It does this twice, through the same v2 server, the same exec_command tool and the
same transport, changing only the call that performs the exec:

- before (v1 shape): `await ExegolContainer.exec_raw(...)`, the real exegol 6.0.2 method v1 relied on. It is
  `async def` but calls docker-py's synchronous `exec_run(stream=False)` without awaiting anything, so the whole
  event loop waits for the command (RESEARCH Pitfall 1);
- after (v2): `await exec_demuxed(...)`, the real v2 adapter function, which offloads `exec_run` to a worker
  thread through run_blocking().

Both are real code. Only the Docker side is a stand-in: each ExegolContainer wraps a stub docker-py container
whose exec_run() sleeps for --block seconds, like a real `docker exec` of a long command would block. The script
never contacts the Docker daemon, never touches a real container or image, and reads no Exegol config, so it is
safe to run on any machine; a live-Docker variant was deliberately not run (it would exec in the user's real
containers). Each scenario is measured over the in-memory client and over streamable HTTP (uvicorn on 127.0.0.1,
the v1 default transport).
"""

import argparse
import platform
import socket
import statistics
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import httpx2
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exegol.model.ExegolContainer import ExegolContainer

from exegol_mcp.adapter.exec import exec_demuxed
from exegol_mcp.adapter.port import ContainerState
from exegol_mcp.app import build_server
from exegol_mcp.bootstrap.transports import UVICORN_LOG_CONFIG, build_http_app
from exegol_mcp.models.exec import ExecResult

CONTAINER = "exegol-repro"
SETTLE_SECONDS = 0.3


class BlockingDockerContainer:
    """Stand-in for a docker-py Container whose exec_run blocks like a long `docker exec`."""

    def __init__(self, block_seconds: float) -> None:
        self.block_seconds = block_seconds

    def exec_run(self, cmd: Any, **kwargs: Any) -> SimpleNamespace:
        time.sleep(self.block_seconds)
        if kwargs.get("demux"):
            return SimpleNamespace(exit_code=0, output=(b"done\n", b""))
        return SimpleNamespace(exit_code=0, output=b"done\n")


def stub_exegol_container(block_seconds: float) -> ExegolContainer:
    """A real ExegolContainer instance (so exec_raw is the SDK's own code) wired to the blocking stub."""
    container = object.__new__(ExegolContainer)
    container._ExegolContainer__container = BlockingDockerContainer(block_seconds)  # type: ignore[attr-defined]
    container.isRunning = lambda: True  # type: ignore[method-assign]
    return container


class ReproPort:
    """The two ExegolPort methods exec_command uses; the exec itself is the v1 or the v2 call."""

    def __init__(self, exec_call: Callable[[str], Awaitable[ExecResult]]) -> None:
        self._exec_call = exec_call

    async def get_container_state(self, name: str) -> ContainerState | None:
        return ContainerState(running=True) if name == CONTAINER else None

    async def exec_command(self, name: str, command: str) -> ExecResult:
        return await self._exec_call(command)


def v1_port(block_seconds: float) -> ReproPort:
    container = stub_exegol_container(block_seconds)

    async def exec_raw(command: str) -> ExecResult:
        exit_code, output = await container.exec_raw(command)  # the v1 call: blocks the event loop
        return ExecResult(stdout=output, stderr="", exit_code=exit_code)

    return ReproPort(exec_raw)


def v2_port(block_seconds: float) -> ReproPort:
    container = stub_exegol_container(block_seconds)

    async def offloaded(command: str) -> ExecResult:
        return await exec_demuxed(container, command)  # the v2 adapter call: exec_run in a worker thread

    return ReproPort(offloaded)


async def measure(client: Client, block_seconds: float) -> tuple[float, float]:
    """Return (tools/list latency while the exec runs, exec duration).

    The latency is counted from the moment tools/list was due (SETTLE_SECONDS after the exec call was sent), not
    from when it was actually sent. With the in-memory client, client and server share one event loop: a blocked
    loop also delays the client's own sleep, so the request leaves late and a send-to-answer timer would miss the
    stall entirely.
    """
    await client.list_tools()  # warm-up: the client caches the tool list on first use
    out: dict[str, float] = {}

    async def long_call() -> None:
        started = time.perf_counter()
        result = await client.call_tool(
            "exec_command", {"container_name": CONTAINER, "command": f"sleep {block_seconds}"}
        )
        assert not result.is_error, result.content
        out["exec"] = time.perf_counter() - started

    async with anyio.create_task_group() as tg:
        due = time.perf_counter() + SETTLE_SECONDS
        tg.start_soon(long_call)
        await anyio.sleep(SETTLE_SECONDS)
        await client.list_tools()
        out["list"] = time.perf_counter() - due
    return out["list"], out["exec"]


async def measure_in_memory(port: ReproPort, block_seconds: float) -> tuple[float, float]:
    async with Client(build_server(None, port), mode="legacy") as client:  # type: ignore[arg-type]
        return await measure(client, block_seconds)


async def measure_http(port: ReproPort, block_seconds: float) -> tuple[float, float]:
    app = build_http_app(build_server(None, port), auth="none", secret=None)  # type: ignore[arg-type]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    http_port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=http_port, log_config=UVICORN_LOG_CONFIG))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        while not server.started:
            await anyio.sleep(0.02)
        url = f"http://127.0.0.1:{http_port}/mcp"
        async with (
            httpx2.AsyncClient() as http_client,
            Client(streamable_http_client(url, http_client=http_client), mode="legacy") as client,
        ):
            return await measure(client, block_seconds)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def summary(samples: list[float]) -> str:
    return f"median {statistics.median(samples):.3f}s (min {min(samples):.3f}s, max {max(samples):.3f}s)"


async def main(block_seconds: float, runs: int) -> None:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"mcp {version('mcp')}, exegol {version('exegol')}, uvicorn {version('uvicorn')}, anyio {version('anyio')}")
    print(f"exec blocks {block_seconds}s; tools/list due {SETTLE_SECONDS}s after the exec; {runs} runs per cell")
    print("tools/list latency = answer time - due time (a healthy server answers in milliseconds)")
    print()
    print(f"{'scenario':<34} {'transport':<10} tools/list latency during the exec          exec duration")
    for label, factory in (
        ("before: v1 SDK exec_raw (on loop)", v1_port),
        ("after: v2 exec_demuxed (offload)", v2_port),
    ):
        for transport, runner in (("in-memory", measure_in_memory), ("http", measure_http)):
            lists, execs = [], []
            for _ in range(runs):
                list_latency, exec_duration = await runner(factory(block_seconds), block_seconds)
                lists.append(list_latency)
                execs.append(exec_duration)
            print(f"{label:<34} {transport:<10} {summary(lists):<42} {summary(execs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", type=float, default=3.0, help="seconds the exec blocks (default: 3)")
    parser.add_argument("--runs", type=int, default=3, help="runs per scenario and transport (default: 3)")
    options = parser.parse_args()
    anyio.run(main, options.block, options.runs)
