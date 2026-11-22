"""In-memory JobStore and lifespan orphan sweep (EXEC-04/05/06, D-08/D-10/D-11), Docker-free over the fake.

The store talks only to an ExegolPort (the fake's simulated process table) and mints opaque ids with
secrets.token_urlsafe. These tests drive it directly inside a task group (as the lifespan does) and through
build_server()'s lifespan to prove the startup/shutdown sweep.
"""

import re

import anyio
import pytest
from mcp import Client

from exegol_mcp.app import build_server
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.models.job import JobInfo, JobSummary
from exegol_mcp.services.jobs import JobStore
from tests.fakes import FakeExegolAdapter


async def _wait_done(store: JobStore) -> None:
    """Wait until no job in the store is still running (the fake's pump finishes fast at block 0)."""
    for _ in range(500):
        if store.running_count() == 0:
            return
        await anyio.sleep(0.01)
    raise AssertionError("a job never finished")


async def test_start_returns_an_opaque_secrets_id() -> None:
    adapter = FakeExegolAdapter()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        job_id = await store.start("exegol-lab", "sleep 1")
        assert isinstance(job_id, str)
        assert len(job_id) >= 16
        # secrets.token_urlsafe alphabet only: no guessable sequential counter.
        assert re.fullmatch(r"[A-Za-z0-9_-]+", job_id)
        tg.cancel_scope.cancel()


async def test_ids_are_distinct_across_many_starts() -> None:
    adapter = FakeExegolAdapter()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits(max_concurrent_jobs=100))
        ids = [await store.start("exegol-lab", f"cmd {i}") for i in range(25)]
        assert len(set(ids)) == 25
        tg.cancel_scope.cancel()


async def test_read_is_incremental_with_offsets_and_resumes_when_omitted() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="0123456789abcdef", stderr="", exit_code=0))
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits(output_cap_bytes=4))
        job_id = await store.start("exegol-lab", "noisy")
        await _wait_done(store)
        first = await store.read(job_id)
        assert first.stdout == "0123"
        assert first.stdout_next_offset == 4
        # Omitting the offset resumes from where the last read ended (D-08).
        second = await store.read(job_id)
        assert second.stdout == "4567"
        assert second.stdout_next_offset == 8
        # An explicit offset re-reads from there.
        again = await store.read(job_id, stdout_offset=0)
        assert again.stdout == "0123"
        tg.cancel_scope.cancel()


async def test_stop_kills_the_process_and_marks_the_job_killed() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        job_id = await store.start("exegol-lab", "long-scan")
        assert store.running_count() == 1
        info = await store.stop(job_id)
        # After stop, no in-container process carries the tag (EXEC-06, success criterion 3).
        assert adapter.job_processes == {}
        assert info.status == "killed"
        assert info.exit_code is None
        tg.cancel_scope.cancel()


async def test_stop_on_a_finished_job_returns_its_final_state() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="done\n", stderr="", exit_code=0))
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        job_id = await store.start("exegol-lab", "quick")
        await _wait_done(store)
        info = await store.stop(job_id)
        assert info.status == "completed"
        assert info.exit_code == 0
        tg.cancel_scope.cancel()


async def test_a_start_past_the_cap_raises_a_clear_error_naming_the_limit() -> None:
    adapter = FakeExegolAdapter(exec_block_seconds=30.0)
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits(max_concurrent_jobs=1))
        first = await store.start("exegol-lab", "job-one")
        with pytest.raises(RuntimeError, match=r"\(1\)"):
            await store.start("exegol-lab", "job-two")
        await store.stop(first)  # release the running job so the task group can close
        tg.cancel_scope.cancel()


async def test_an_unknown_id_is_a_value_error_not_a_crash() -> None:
    adapter = FakeExegolAdapter()
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits())
        with pytest.raises(ValueError, match="ghost"):
            await store.read("ghost")
        with pytest.raises(ValueError, match="ghost"):
            await store.stop("ghost")
        tg.cancel_scope.cancel()


async def test_list_jobs_returns_summaries_of_this_servers_jobs() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="x", stderr="", exit_code=0))
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits(max_concurrent_jobs=10))
        a = await store.start("exegol-lab", "alpha command")
        b = await store.start("exegol-lab", "beta command")
        await _wait_done(store)
        summaries = store.list()
        assert all(isinstance(s, JobSummary) for s in summaries)
        ids = {s.id for s in summaries}
        assert {a, b} <= ids
        by_id = {s.id: s for s in summaries}
        assert by_id[a].command.startswith("alpha")
        tg.cancel_scope.cancel()


async def test_a_finished_job_older_than_the_ttl_is_swept_on_access() -> None:
    adapter = FakeExegolAdapter(exec_result=ExecResult(stdout="bye\n", stderr="", exit_code=0))
    async with anyio.create_task_group() as tg:
        store = JobStore(adapter, tg, ExecutionLimits(finished_job_ttl_s=0.01))
        job_id = await store.start("exegol-lab", "quick")
        await _wait_done(store)
        assert isinstance(await store.read(job_id), JobInfo)  # present right after it finishes
        await anyio.sleep(0.05)  # let the TTL lapse
        with pytest.raises(ValueError, match="ghost|" + re.escape(job_id)):
            await store.read(job_id)
        assert store.list() == []
        tg.cancel_scope.cancel()


async def test_the_lifespan_sweeps_tagged_orphans_at_startup_and_shutdown() -> None:
    adapter = FakeExegolAdapter()
    # A tagged process left behind by a prior (crashed) server run.
    adapter.job_processes["orphan-tag"] = "exegol-lab"
    async with Client(build_server(None, adapter)):
        # The startup sweep (D-11) killed the orphan before serving.
        assert "orphan-tag" not in adapter.job_processes
        assert adapter.jobs is not None  # the store was attached to the context
    # Swept once at startup and again in the shielded shutdown path.
    assert adapter.sweep_calls >= 2
