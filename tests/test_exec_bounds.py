"""Bounded and cancellable foreground exec (EXEC-02/07, EXEC-08 preserved), Docker-free over the fake.

The fake's exec_block_seconds makes a command overrun a tiny timeout and its process table lets a test assert the
tagged in-container process is gone after a kill or a cancellation. Every test drives the execution service (and,
for cancellation, the exec tool handler) against a JobStore built in a task group, exactly as the lifespan does.
"""

from types import SimpleNamespace

import anyio
import pytest

from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.services import execution as execution_service
from exegol_mcp.services.jobs import JobStore
from exegol_mcp.tools import exec as exec_tool
from tests.fakes import FakeExegolAdapter


def _ctx(adapter: FakeExegolAdapter) -> SimpleNamespace:
    """A minimal stand-in for the MCP Context: tools only read request_context.lifespan_context."""
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=adapter))


# --- Task 1: per-call timeout, operator clamp and the on_timeout branch (EXEC-02) --------------------------


async def test_default_timeout_is_applied_when_none_given() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits(timeout_default_s=0.05)
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        # No timeout given: the operator default (here 0.05s) applies, so the slow command times out.
        result = await execution_service.exec_command(adapter, "exegol-lab", "long", limits=limits, jobs=store)
        assert result.timed_out is True
        assert result.job_id is not None  # on_timeout defaults to background (D-12)
        await store.stop(result.job_id)
        tg.cancel_scope.cancel()


async def test_a_timeout_below_the_max_is_applied_verbatim() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await execution_service.exec_command(
            adapter, "exegol-lab", "long", timeout=0.05, on_timeout="kill", limits=limits, jobs=store
        )
        assert result.timed_out is True
        tg.cancel_scope.cancel()


async def test_a_timeout_above_the_max_is_clamped_to_the_operator_maximum() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits(timeout_max_s=0.05)
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        # A huge requested timeout must be clamped to timeout_max_s (0.05), so the slow command still times out.
        result = await execution_service.exec_command(
            adapter, "exegol-lab", "long", timeout=30.0, on_timeout="background", limits=limits, jobs=store
        )
        assert result.timed_out is True
        await store.stop(result.job_id)
        tg.cancel_scope.cancel()


async def test_a_non_positive_timeout_is_rejected() -> None:
    adapter = FakeExegolAdapter()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        with pytest.raises(ValueError, match="timeout"):
            await execution_service.exec_command(adapter, "exegol-lab", "x", timeout=0, jobs=store)
        with pytest.raises(ValueError, match="timeout"):
            await execution_service.exec_command(adapter, "exegol-lab", "x", timeout=-5, jobs=store)
        tg.cancel_scope.cancel()


async def test_an_invalid_on_timeout_is_rejected() -> None:
    adapter = FakeExegolAdapter()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        with pytest.raises(ValueError, match="on_timeout"):
            await execution_service.exec_command(
                adapter, "exegol-lab", "x", on_timeout="explode", jobs=store  # type: ignore[arg-type]
            )
        tg.cancel_scope.cancel()


async def test_on_timeout_kill_terminates_and_returns_a_null_exit_code() -> None:
    adapter = FakeExegolAdapter(
        exec_block_seconds=30.0, exec_result=ExecResult(stdout="partial\n", stderr="", exit_code=0)
    )
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await execution_service.exec_command(
            adapter, "exegol-lab", "long-scan", timeout=0.05, on_timeout="kill", limits=limits, jobs=store
        )
        assert result.timed_out is True
        assert result.exit_code is None  # a killed command's exit code is null, never coerced to 0 (EXEC-08)
        assert result.job_id is None  # kill hands out no job id
        assert result.stdout == "partial\n"  # the partial output is returned
        assert adapter.job_processes == {}  # the tagged in-container process is gone
        tg.cancel_scope.cancel()


async def test_on_timeout_background_promotes_to_a_pollable_job() -> None:
    adapter = FakeExegolAdapter(
        exec_block_seconds=30.0, exec_result=ExecResult(stdout="partial\n", stderr="", exit_code=0)
    )
    limits = ExecutionLimits()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await execution_service.exec_command(
            adapter, "exegol-lab", "long-scan", timeout=0.05, on_timeout="background", limits=limits, jobs=store
        )
        assert result.timed_out is True
        assert result.exit_code is None  # still running when demoted
        assert result.job_id is not None
        # The promoted job continues the SAME stream: its partial output is already pollable via the store.
        info = await store.read(result.job_id, stdout_offset=0, stderr_offset=0)
        assert info.stdout == "partial\n"
        assert info.status == "running"
        await store.stop(result.job_id)  # release the running job so the task group can close
        tg.cancel_scope.cancel()


async def test_a_promotion_past_the_cap_kills_instead_of_promoting() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits(max_concurrent_jobs=1)
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        # Fill the single background slot with a real background job.
        bg = await store.start("exegol-lab", "existing-job")
        assert store.running_count() == 1
        # A foreground command overruns; promotion is refused (cap full) so the command is killed instead (D-10).
        result = await execution_service.exec_command(
            adapter, "exegol-lab", "overrun", timeout=0.05, on_timeout="background", limits=limits, jobs=store
        )
        assert result.timed_out is True
        assert result.exit_code is None
        assert result.job_id is None  # killed, not promoted
        await store.stop(bg)
        tg.cancel_scope.cancel()


async def test_a_demoted_job_past_max_job_runtime_is_auto_killed() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits(timeout_default_s=0.05, max_job_runtime_s=0.05)
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, limits)
        result = await execution_service.exec_command(
            adapter, "exegol-lab", "runaway", on_timeout="background", limits=limits, jobs=store
        )
        assert result.timed_out is True
        assert result.job_id is not None
        # The watchdog (max_job_runtime_s) kills the demoted runaway even though nobody stops it (D-15).
        info = None
        for _ in range(500):
            info = await store.read(result.job_id)
            if info.status != "running":
                break
            await anyio.sleep(0.01)
        assert info is not None and info.status == "killed"
        tg.cancel_scope.cancel()


async def test_a_fast_command_returns_a_full_result_with_timed_out_false() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="done\n", stderr="e\n", exit_code=0))
    adapter.exec_release.set()  # release the fake exec immediately so the command finishes before any timeout
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        result = await execution_service.exec_command(
            adapter, "exegol-lab", "quick", timeout=5.0, limits=ExecutionLimits(), jobs=store
        )
        assert result.timed_out is False
        assert result.job_id is None
        assert result.stdout == "done\n"
        assert result.stderr == "e\n"
        assert result.exit_code == 0
        tg.cancel_scope.cancel()


async def test_exec08_preserved_separate_streams_and_the_real_exit_code() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="out\n", stderr="err\n", exit_code=3))
    adapter.exec_release.set()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        result = await execution_service.exec_command(adapter, "exegol-lab", "cmd", jobs=store)
        assert result.stdout == "out\n"
        assert result.stderr == "err\n"
        assert result.exit_code == 3  # the real exit code, kept separate from the streams
        assert result.timed_out is False
        tg.cancel_scope.cancel()


# --- Task 2: client cancellation kills the in-container command (EXEC-07) ----------------------------------


async def test_client_cancellation_kills_the_in_container_process() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    limits = ExecutionLimits()
    result_holder: dict[str, object] = {}
    async with anyio.create_task_group() as store_tg:
        store = JobStore(adapter, store_tg, limits)
        adapter.jobs = store
        ctx = _ctx(adapter)

        async def run() -> None:
            result_holder["r"] = await exec_tool.exec_command("exegol-lab", "long-scan", ctx=ctx)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await adapter.exec_started.wait()  # the exec is launched and blocking
            tg.cancel_scope.cancel()  # the client cancels the call mid-run

        # A shielded finally killed the tagged process group (EXEC-07); no result was delivered (D-13).
        assert adapter.job_processes == {}
        assert "r" not in result_holder
        store_tg.cancel_scope.cancel()


async def test_a_normal_run_is_not_cancelled_and_kills_nothing() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="ok\n", stderr="", exit_code=0))
    adapter.exec_release.set()
    limits = ExecutionLimits()
    async with anyio.create_task_group() as store_tg:
        store = JobStore(adapter, store_tg, limits)
        adapter.jobs = store
        ctx = _ctx(adapter)
        result = await exec_tool.exec_command("exegol-lab", "quick", ctx=ctx)
        assert result.timed_out is False
        assert result.exit_code == 0
        assert adapter.job_processes == {}  # finished on its own
        assert adapter._killed_tags == set()  # nothing was killed
        store_tg.cancel_scope.cancel()
