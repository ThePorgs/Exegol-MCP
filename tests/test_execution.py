"""Bounded in-container exec service (EXEC-03, EXEC-08, D-04): running result, output cap, exit codes, errors.

Docker-free: the fake routes exec_command through the tagged job primitive (start_job/pump_job/read_job)."""

import anyio
import pytest
from mcp import Client

from exegol_mcp.app import build_server
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import BoundedExecResult, ExecResult
from exegol_mcp.services import execution as execution_service
from tests.fakes import FakeExegolAdapter


def _stopped(adapter: FakeExegolAdapter, name: str) -> FakeExegolAdapter:
    adapter.running[name] = False
    return adapter


async def test_exec_command_returns_the_bounded_demuxed_result_for_a_running_container() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="hello\n", stderr="warn\n", exit_code=0))
    result = await execution_service.exec_command(adapter, "exegol-lab", "echo hello")
    # A foreground run: the three D-04 fields plus job_id None, timed_out False and no drops.
    assert result == BoundedExecResult(stdout="hello\n", stderr="warn\n", exit_code=0)
    assert isinstance(result, BoundedExecResult)
    assert result.stdout == "hello\n"
    assert result.stderr == "warn\n"
    assert result.exit_code == 0
    assert result.job_id is None
    assert result.timed_out is False
    assert result.stdout_bytes_dropped == 0
    assert result.stderr_bytes_dropped == 0


async def test_exec_command_keeps_stdout_and_stderr_separate() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="out only\n", stderr="err only\n", exit_code=0))
    result = await execution_service.exec_command(adapter, "exegol-lab", "cmd")
    assert result.stdout == "out only\n"
    assert result.stderr == "err only\n"


async def test_exec_command_keeps_a_none_exit_code_as_none() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="", stderr="", exit_code=None))
    result = await execution_service.exec_command(adapter, "exegol-lab", "true")
    assert result.exit_code is None


async def test_exec_command_keeps_a_nonzero_exit_code() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="", stderr="no\n", exit_code=127))
    result = await execution_service.exec_command(adapter, "exegol-lab", "missing-binary")
    assert result.exit_code == 127
    assert result.stderr == "no\n"


async def test_exec_command_over_the_cap_keeps_the_tail_and_reports_the_dropped_count() -> None:
    big = "0123456789" * 10  # 100 ASCII bytes
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout=big, stderr="", exit_code=0))
    result = await execution_service.exec_command(
        adapter, "exegol-lab", "noisy", limits=ExecutionLimits(output_cap_bytes=8)
    )
    assert result.stdout == big[-8:]  # the tail is kept (EXEC-03, D-21)
    assert result.stdout_bytes_dropped == 100 - 8  # the exact front-dropped byte count
    assert result.stderr_bytes_dropped == 0


async def test_exec_command_at_the_cap_drops_nothing() -> None:
    exact = "01234567"  # exactly 8 bytes
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout=exact, stderr="", exit_code=0))
    result = await execution_service.exec_command(
        adapter, "exegol-lab", "cmd", limits=ExecutionLimits(output_cap_bytes=8)
    )
    assert result.stdout == exact
    assert result.stdout_bytes_dropped == 0


async def test_exec_command_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Container 'ghost' not found"):
        await execution_service.exec_command(FakeExegolAdapter(), "ghost", "echo hi")


async def test_exec_command_stopped_container_raises_a_clear_start_it_first_error() -> None:
    adapter = _stopped(FakeExegolAdapter(), "exegol-lab")
    with pytest.raises(RuntimeError, match="start it first") as raised:
        await execution_service.exec_command(adapter, "exegol-lab", "echo hi")
    assert "exegol-lab" in str(raised.value)


async def test_exec_command_never_offers_to_start_a_stopped_container() -> None:
    # The v1 tool elicited the user to start a stopped container; v2 must only ever return the clear error.
    adapter = _stopped(FakeExegolAdapter(), "exegol-lab")
    with pytest.raises(RuntimeError):
        await execution_service.exec_command(adapter, "exegol-lab", "echo hi")
    assert adapter.state_changes == []
    assert adapter.exec_commands == []


# --- background=true (EXEC-04) -----------------------------------------------------------------------------


async def test_exec_command_background_returns_a_job_id_and_empty_output_then_get_job_reads_it() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="scan done\n", stderr="warn\n", exit_code=0))
    async with Client(build_server(None, adapter)) as client:
        started = await client.call_tool(
            "exec_command", {"container_name": "exegol-lab", "command": "nmap -p-", "background": True}
        )
        assert not started.is_error, started.content
        payload = started.structured_content
        assert payload is not None
        job_id = payload["job_id"]
        # EXEC-04: returns at once with an opaque id and no output; the real exit code is null, not 0.
        assert isinstance(job_id, str) and job_id
        assert payload["stdout"] == "" and payload["stderr"] == ""
        assert payload["exit_code"] is None
        assert payload["timed_out"] is False

        # The job runs under the store; poll it (offset 0 each time) until it finishes.
        info: dict[str, object] = {}
        for _ in range(500):
            got = await client.call_tool("get_job", {"job_id": job_id, "stdout_offset": 0, "stderr_offset": 0})
            assert not got.is_error, got.content
            assert got.structured_content is not None
            info = got.structured_content
            if info["status"] != "running":
                break
            await anyio.sleep(0.01)
    # EXEC-08 preserved: separate stdout/stderr and the real exit code survive the job read.
    assert info["stdout"] == "scan done\n"
    assert info["stderr"] == "warn\n"
    assert info["exit_code"] == 0
    assert info["status"] == "completed"


async def test_exec_command_background_on_a_stopped_container_is_a_clear_error() -> None:
    adapter = _stopped(FakeExegolAdapter(), "exegol-lab")
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool(
            "exec_command", {"container_name": "exegol-lab", "command": "nmap", "background": True}
        )
    assert result.is_error
    assert "start it first" in str(result.content)


async def test_exec_command_background_without_a_store_raises_a_clear_error() -> None:
    adapter = FakeExegolAdapter()
    with pytest.raises(RuntimeError, match="[Bb]ackground"):
        await execution_service.exec_command(adapter, "exegol-lab", "nmap", background=True, jobs=None)
