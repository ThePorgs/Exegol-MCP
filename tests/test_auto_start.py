"""auto_start on exec_command and exec_remote_command (LIFE-07, D-15), Docker-free over the fake adapter.

A stopped container is started first when the agent passes auto_start=true, bounded by
execution.container_op_timeout_s (D-18), and then the command runs. Without it the clear not-running error
stays and names both start_container and auto_start. A running container is never restarted, an unknown one is
still not-found, and a slow start keeps the existing "did not finish starting" error. exec_remote_command checks
and starts the container before the credential-bearing backend command is built, and a cancellation that lands
while the container is still stopped is never replaced by the kill-by-tag's "not running" error (EXEC-07).
"""

from types import SimpleNamespace
from typing import Any

import anyio
import pytest
import requests
from mcp import Client

from exegol_mcp.app import build_server
from exegol_mcp.models.config import ExecutionLimits, OperatorConfig
from exegol_mcp.services import containers as containers_service
from exegol_mcp.services import execution as execution_service
from exegol_mcp.services.jobs import JobStore
from exegol_mcp.tools import exec as exec_tool
from exegol_mcp.tools import remote as remote_tool
from tests.fakes import FakeExegolAdapter


class _SlowStartFake(FakeExegolAdapter):
    """start_container awaits a cooperative anyio.sleep before delegating (a slow container start)."""

    def __init__(self, slow_s: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._slow_s = slow_s

    async def start_container(self, name: str) -> bool:
        await anyio.sleep(self._slow_s)
        return await super().start_container(name)


def _stopped(fake: FakeExegolAdapter, name: str = "exegol-lab") -> FakeExegolAdapter:
    """Flip a container to stopped and forget the setup flip, so state_changes only shows what the call did."""
    fake.running[name] = False
    fake.state_changes.clear()
    return fake


# --- exec_command over the in-memory MCP Client (Task 1) -----------------------------------------------------


async def test_a_stopped_container_without_auto_start_names_start_container_and_auto_start() -> None:
    fake = _stopped(FakeExegolAdapter())
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool("exec_command", {"container_name": "exegol-lab", "command": "id"})
    assert result.is_error
    text = str(result.content)
    assert "start it first" in text
    assert "auto_start" in text
    assert fake.state_changes == []
    assert fake.exec_commands == []


async def test_auto_start_starts_a_stopped_container_then_runs_the_command() -> None:
    fake = _stopped(FakeExegolAdapter())
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool(
            "exec_command", {"container_name": "exegol-lab", "command": "id", "auto_start": True}
        )
    assert not result.is_error, result.content
    assert result.structured_content is not None
    assert result.structured_content["exit_code"] == 0
    assert fake.state_changes[-1] == ("start", "exegol-lab")
    assert fake.running["exegol-lab"] is True
    assert "id" in fake.exec_commands


async def test_auto_start_on_a_running_container_starts_nothing_and_runs_the_command() -> None:
    fake = FakeExegolAdapter()
    assert fake.running["exegol-lab"] is True
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool(
            "exec_command", {"container_name": "exegol-lab", "command": "id", "auto_start": True}
        )
    assert not result.is_error, result.content
    assert ("start", "exegol-lab") not in fake.state_changes
    assert fake.state_changes == []
    assert "id" in fake.exec_commands


async def test_auto_start_on_an_unknown_container_is_not_found_and_starts_nothing() -> None:
    fake = FakeExegolAdapter()
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool(
            "exec_command", {"container_name": "ghost", "command": "id", "auto_start": True}
        )
    assert result.is_error
    assert "list_containers" in str(result.content)
    assert fake.state_changes == []
    assert fake.exec_commands == []


async def test_auto_start_that_overruns_the_container_op_timeout_runs_no_command() -> None:
    config = OperatorConfig(execution=ExecutionLimits(container_op_timeout_s=0.2))
    fake = _stopped(_SlowStartFake(slow_s=5.0, config=config))
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool(
            "exec_command", {"container_name": "exegol-lab", "command": "id", "auto_start": True}
        )
    assert result.is_error
    text = str(result.content)
    assert "did not finish starting" in text
    assert "list_containers" in text
    assert "id" not in fake.exec_commands


# --- Concurrency (edge LIFE-07): two auto-starts on the same stopped container both succeed ------------------


async def test_two_concurrent_auto_start_execs_on_one_stopped_container_both_succeed() -> None:
    fake = _stopped(_SlowStartFake(slow_s=0.05))
    limits = ExecutionLimits()
    results: dict[str, Any] = {}
    async with anyio.create_task_group() as store_tg:
        store = JobStore(fake, store_tg, limits)

        async def run(command: str) -> None:
            results[command] = await execution_service.exec_command(
                fake, "exegol-lab", command, auto_start=True, start_timeout_s=5.0, limits=limits, jobs=store
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(run, "id")
            tg.start_soon(run, "whoami")
        store_tg.cancel_scope.cancel()

    assert results["id"].exit_code == 0
    assert results["whoami"].exit_code == 0
    assert "id" in fake.exec_commands
    assert "whoami" in fake.exec_commands
    assert fake.running["exegol-lab"] is True
    assert ("stop", "exegol-lab") not in fake.state_changes
    assert 1 <= fake.state_changes.count(("start", "exegol-lab")) <= 2


async def test_two_concurrent_ensure_running_calls_on_one_stopped_container_both_return_none() -> None:
    fake = _stopped(_SlowStartFake(slow_s=0.05))
    returned: list[object] = []

    async def ensure() -> None:
        returned.append(await containers_service.ensure_running(fake, "exegol-lab", auto_start=True, timeout_s=5.0))

    async with anyio.create_task_group() as tg:
        tg.start_soon(ensure)
        tg.start_soon(ensure)
    assert returned == [None, None]
    assert fake.running["exegol-lab"] is True


# --- ensure_running directly ----------------------------------------------------------------------------------


async def test_ensure_running_without_auto_start_raises_the_not_running_message() -> None:
    fake = _stopped(FakeExegolAdapter())
    with pytest.raises(RuntimeError, match="start it first") as raised:
        await containers_service.ensure_running(fake, "exegol-lab", auto_start=False)
    assert "auto_start=true" in str(raised.value)
    assert fake.state_changes == []


async def test_ensure_running_on_an_unknown_container_raises_value_error() -> None:
    fake = FakeExegolAdapter()
    with pytest.raises(ValueError, match="list_containers"):
        await containers_service.ensure_running(fake, "ghost", auto_start=True)
    assert fake.state_changes == []


# --- exec_remote_command auto_start (Task 2) -----------------------------------------------------------------

_REMOTE_ARGS: dict[str, Any] = {
    "container_name": "exegol-lab",
    "protocol": "ssh",
    "target_ip": "10.0.0.1",
    "username": "root",
    "password": "pw",
    "command": "id",
}


def _nxc_commands(fake: FakeExegolAdapter) -> list[str]:
    return [c for c in fake.exec_commands if c.startswith("nxc ")]


async def test_remote_auto_start_starts_a_stopped_container_then_runs_the_backend_command() -> None:
    fake = _stopped(FakeExegolAdapter())
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool("exec_remote_command", {**_REMOTE_ARGS, "auto_start": True})
    assert not result.is_error, result.content
    assert ("start", "exegol-lab") in fake.state_changes
    assert fake.state_changes.count(("start", "exegol-lab")) == 1  # only one start is ever attempted
    assert len(_nxc_commands(fake)) == 1


async def test_remote_without_auto_start_on_a_stopped_container_builds_no_backend_command() -> None:
    fake = _stopped(FakeExegolAdapter())
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool("exec_remote_command", dict(_REMOTE_ARGS))
    assert result.is_error
    text = str(result.content)
    assert "start it first" in text
    assert "auto_start" in text
    assert _nxc_commands(fake) == []
    assert fake.state_changes == []


@pytest.mark.parametrize("override", [{"protocol": "telnet"}, {"port": 70000}])
async def test_remote_invalid_input_with_auto_start_is_rejected_before_any_start(override: dict[str, Any]) -> None:
    fake = _stopped(FakeExegolAdapter())
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool("exec_remote_command", {**_REMOTE_ARGS, **override, "auto_start": True})
    assert result.is_error
    assert fake.state_changes == []
    assert fake.exec_commands == []


# --- Cancel path: the kill-by-tag is best-effort, the cancellation always propagates (EXEC-07) ------------------


# The kill errors the guard must swallow: the real adapter's not-running RuntimeError, an OSError from the Docker
# socket, and a docker APIError (a requests HTTPError) when the container stops between the check and the exec.
_KILL_ERRORS = {
    "runtime-error": lambda name: RuntimeError(f"Container '{name}' is not running"),
    "os-error": lambda name: ConnectionResetError(104, "Connection reset by peer"),
    "http-error": lambda name: requests.exceptions.HTTPError("409 Client Error: Conflict"),
}


class _KillRaisesFake(FakeExegolAdapter):
    """kill_job raises, like the real adapter does for a container that is not running or a Docker failure."""

    def __init__(self, kill_error: str = "runtime-error", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._kill_error = _KILL_ERRORS[kill_error]

    async def kill_job(self, name: str, tag: str, grace_s: float) -> None:
        raise self._kill_error(name)


class _SlowStartKillRaisesFake(_SlowStartFake):
    """A slow start plus a failing kill_job: it raises while the container is still stopped."""

    def __init__(self, slow_s: float, kill_error: str = "runtime-error", **kwargs: Any) -> None:
        super().__init__(slow_s, **kwargs)
        self._kill_error = _KILL_ERRORS[kill_error]

    async def kill_job(self, name: str, tag: str, grace_s: float) -> None:
        if not self.running.get(name, False):
            raise self._kill_error(name)
        await super().kill_job(name, tag, grace_s)


def _ctx(adapter: FakeExegolAdapter) -> SimpleNamespace:
    """A minimal stand-in for the MCP Context: tools only read request_context.lifespan_context."""
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=adapter))


async def _cancel_mid_call(adapter: FakeExegolAdapter, call: Any, started: Any) -> dict[str, object]:
    """Run the tool call, cancel it once ``started`` is set, and report how the call ended."""
    outcome: dict[str, object] = {}
    async with anyio.create_task_group() as store_tg:
        adapter.jobs = JobStore(adapter, store_tg, ExecutionLimits())

        async def run() -> None:
            try:
                outcome["result"] = await call(_ctx(adapter))
            except anyio.get_cancelled_exc_class():
                outcome["cancelled"] = True
                raise
            except BaseException as exc:  # noqa: BLE001 - the test reports what replaced the cancellation
                outcome["other"] = exc

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await started()
            tg.cancel_scope.cancel()
        store_tg.cancel_scope.cancel()
    return outcome


@pytest.mark.parametrize("kill_error", sorted(_KILL_ERRORS))
async def test_cancelling_exec_command_propagates_even_when_the_kill_raises(kill_error: str) -> None:
    fake = _KillRaisesFake(kill_error, exec_block_seconds=30.0)

    async def call(ctx: Any) -> Any:
        return await exec_tool.exec_command("exegol-lab", "long-scan", ctx=ctx)

    outcome = await _cancel_mid_call(fake, call, fake.exec_started.wait)
    assert outcome == {"cancelled": True}


@pytest.mark.parametrize("kill_error", sorted(_KILL_ERRORS))
async def test_cancelling_exec_remote_command_propagates_even_when_the_kill_raises(kill_error: str) -> None:
    fake = _KillRaisesFake(kill_error, exec_block_seconds=30.0)

    async def call(ctx: Any) -> Any:
        return await remote_tool.exec_remote_command(**_REMOTE_ARGS, ctx=ctx)

    outcome = await _cancel_mid_call(fake, call, fake.exec_started.wait)
    assert outcome == {"cancelled": True}


@pytest.mark.parametrize("kill_error", sorted(_KILL_ERRORS))
@pytest.mark.parametrize("tool", ["exec_command", "exec_remote_command"])
async def test_cancelling_during_a_slow_auto_start_propagates_and_runs_no_command(tool: str, kill_error: str) -> None:
    fake = _stopped(_SlowStartKillRaisesFake(slow_s=5.0, kill_error=kill_error))

    async def call(ctx: Any) -> Any:
        if tool == "exec_command":
            return await exec_tool.exec_command("exegol-lab", "id", auto_start=True, ctx=ctx)
        return await remote_tool.exec_remote_command(**_REMOTE_ARGS, auto_start=True, ctx=ctx)

    async def started() -> None:
        await anyio.sleep(0.05)  # the call is inside the slow start, the container still stopped

    outcome = await _cancel_mid_call(fake, call, started)
    assert outcome == {"cancelled": True}
    assert fake.exec_commands == []
