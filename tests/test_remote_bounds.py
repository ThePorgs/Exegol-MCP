"""Remote-exec parity: netexec bounded/background/cancellable like exec_command (D-17, EXEC-02/03/07).

exec_remote_command routes the built netexec command through the same bounded exec path as exec_command, so a long
sweep or spray inherits the per-call timeout, the operator clamp, the on_timeout demotion, the background flag, the
output cap and the kill-on-cancel. These tests drive the remote service (and, for cancellation, the remote tool
handler) Docker-free against FakeExegolAdapter, exactly as the lifespan wires the store, and confirm the netexec
password and the built command never leak into logs or progress (the v1 invariant), while the command output stays
unredacted (SEC-02).
"""

import logging
from types import SimpleNamespace

import anyio
import pytest

from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import BoundedExecResult, ExecResult
from exegol_mcp.services import remote as remote_service
from exegol_mcp.services.jobs import JobStore
from exegol_mcp.tools import remote as remote_tool
from tests.fakes import FakeExegolAdapter

# A password that must never show up in a log line or a progress message, and a built command that must not either.
_PASSWORD = "SuperSecret123!"

# A raw netexec capture: the "[+] Executed command" marker, then one prefixed output line to strip.
_NETEXEC_OUTPUT = "SMB  10.0.0.1  445  10.0.0.1  [+] Executed command\nSMB  10.0.0.1  445  10.0.0.1  uid=0(root)\n"


def _ctx(adapter: FakeExegolAdapter) -> SimpleNamespace:
    """A minimal stand-in for the MCP Context: tools only read request_context.lifespan_context."""
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=adapter))


# --- Timeout / on_timeout parity (D-17, EXEC-02) ----------------------------------------------------------


async def test_a_slow_remote_command_is_timed_out_and_kill_returns_a_null_exit_with_the_process_gone() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await remote_service.exec_remote_command(
            adapter, "exegol-lab", "ssh", "10.0.0.1", "root", _PASSWORD, "sleep 100",
            timeout=0.05, on_timeout="kill", limits=limits, jobs=store,
        )
        assert isinstance(result, BoundedExecResult)
        assert result.timed_out is True
        assert result.exit_code is None  # a killed command's exit code is null, never coerced to 0 (EXEC-08)
        assert result.job_id is None  # kill hands out no job id
        assert adapter.job_processes == {}  # the netexec process group is gone
        tg.cancel_scope.cancel()


async def test_a_slow_remote_command_with_on_timeout_background_becomes_a_pollable_job() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await remote_service.exec_remote_command(
            adapter, "exegol-lab", "smb", "10.0.0.1", "admin", _PASSWORD, "sleep 100",
            timeout=0.05, on_timeout="background", limits=limits, jobs=store,
        )
        assert result.timed_out is True
        assert result.exit_code is None
        assert result.job_id is not None  # demoted to a pollable background job (D-12)
        info = await store.read(result.job_id, stdout_offset=0, stderr_offset=0)
        assert info.status == "running"
        await store.stop(result.job_id)  # release the running job so the task group can close
        tg.cancel_scope.cancel()


async def test_a_background_remote_command_returns_a_job_id_at_once() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await remote_service.exec_remote_command(
            adapter, "exegol-lab", "ssh", "10.0.0.1", "root", _PASSWORD, "sleep 100",
            background=True, limits=limits, jobs=store,
        )
        assert result.job_id is not None
        assert result.exit_code is None
        assert result.stdout == ""  # empty output up front; poll the job for the raw netexec output
        await store.stop(result.job_id)
        tg.cancel_scope.cancel()


async def test_a_fast_remote_command_returns_the_parsed_output_with_its_real_exit_code() -> None:
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=_NETEXEC_OUTPUT, stderr="", exit_code=0))
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await remote_service.exec_remote_command(
            adapter, "exegol-lab", "smb", "10.0.0.1", "admin", _PASSWORD, "id",
            timeout=5.0, limits=limits, jobs=store,
        )
        assert result.timed_out is False
        assert result.job_id is None
        assert result.exit_code == 0
        assert result.stdout == "uid=0(root)"  # a synchronous result keeps the netexec log-line parse
        tg.cancel_scope.cancel()


async def test_a_nonzero_remote_exit_is_a_result_not_an_exception_through_the_bounded_path() -> None:
    # EXEC-08: a backend failure is a result carrying the backend's own exit code, never an exception, even bounded.
    unreachable = ExecResult(stdout="SSH  10.0.0.1  22  HOST  [-] Connection refused", stderr="", exit_code=1)
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: unreachable)
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await remote_service.exec_remote_command(
            adapter, "exegol-lab", "ssh", "10.0.0.1", "root", _PASSWORD, "id",
            timeout=5.0, limits=limits, jobs=store,
        )
        assert result.timed_out is False
        assert result.exit_code == 1
        tg.cancel_scope.cancel()


# --- Cancellation parity (EXEC-07): the netexec process group is killed on client cancellation ------------


async def test_client_cancellation_kills_the_netexec_process_group() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits()
    result_holder: dict[str, object] = {}
    async with anyio.create_task_group() as store_tg:
        store = JobStore(adapter, store_tg, limits)
        adapter.jobs = store
        ctx = _ctx(adapter)

        async def run() -> None:
            result_holder["r"] = await remote_tool.exec_remote_command(
                "exegol-lab", "ssh", "10.0.0.1", "root", _PASSWORD, "sleep 100", ctx=ctx
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await adapter.exec_started.wait()  # the netexec command is launched and blocking
            tg.cancel_scope.cancel()  # the client cancels the call mid-run

        # A shielded finally killed the tagged process group (EXEC-07); no result was delivered (D-13).
        assert adapter.job_processes == {}
        assert "r" not in result_holder
        store_tg.cancel_scope.cancel()


# --- The never-log invariant and the unredacted-output rule (SEC-02) --------------------------------------


async def test_the_password_and_the_built_command_never_appear_in_logs_or_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="done", stderr="", exit_code=0))
    limits = ExecutionLimits()
    with caplog.at_level(logging.DEBUG):
        async with anyio.create_task_group() as tg:
            store = JobStore(adapter, tg, limits)
            result = await remote_service.exec_remote_command(
                adapter, "exegol-lab", "smb", "10.0.0.1", "admin", _PASSWORD, "whoami",
                domain="CORP", timeout=5.0, limits=limits, jobs=store,
            )
            tg.cancel_scope.cancel()
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert _PASSWORD not in logged  # the password is never logged, on any path
    assert "nxc" not in logged  # the built netexec command is never logged either
    assert _PASSWORD not in result.stdout
    assert "nxc" not in result.stdout


async def test_output_the_remote_command_prints_is_returned_unredacted_through_the_bounded_path() -> None:
    # SEC-02: tool output is a finding for the operator, so a credential the remote command prints comes back as is.
    raw = "SMB  10.0.0.1  445  10.0.0.1  [+] Executed command\nSMB  10.0.0.1  445  10.0.0.1  CORP\\admin:SuperSecret123\n"
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=raw, stderr="", exit_code=0))
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await remote_service.exec_remote_command(
            adapter, "exegol-lab", "smb", "10.0.0.1", "admin", _PASSWORD, "type creds.txt",
            domain="CORP", timeout=5.0, limits=limits, jobs=store,
        )
        assert result.stdout == "CORP\\admin:SuperSecret123"  # the printed credential is kept, not redacted
        tg.cancel_scope.cancel()
