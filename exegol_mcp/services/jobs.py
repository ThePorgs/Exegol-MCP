"""In-memory background-job store (EXEC-04/05/06, D-08/D-09/D-10/D-11).

SDK-free: the store talks only to an ExegolPort (never exegol/docker directly). It mints an opaque job id with
secrets.token_urlsafe (unguessable, T-02-01), launches the command through the tagged streamed primitive
(start_job), drains it with pump_job in a server-lifetime task group so the job outlives the request that
started it (D-09), serves offset-based incremental reads capped per read (D-08/D-22), and stops a job by killing
its tagged process group (EXEC-06). A start past max_concurrent_jobs is a clear RuntimeError naming the limit
(D-10); finished jobs older than finished_job_ttl_s are swept on access. No file is ever written in the
container: the only crash-recovery mechanism is the lifespan tag sweep (D-11, adapter.sweep_orphans).
"""

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anyio

from exegol_mcp.adapter.port import JOB_CAP_REACHED_MESSAGE, ExegolPort, JobHandle, JobRead
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.job import JobInfo, JobStatus, JobSummary

# How long a command preview is before it is truncated with an ellipsis in list_jobs.
_COMMAND_PREVIEW_CHARS = 80


def _now_iso() -> str:
    """UTC start timestamp as a stable ISO-8601 string (seconds precision, trailing Z)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _preview(command: str) -> str:
    """A short, single-line preview of a command for list_jobs."""
    collapsed = " ".join(command.split())
    if len(collapsed) <= _COMMAND_PREVIEW_CHARS:
        return collapsed
    return collapsed[: _COMMAND_PREVIEW_CHARS - 1] + "…"


@dataclass
class _JobState:
    """One tracked job: its handle, the command, start time, the last-read offsets and whether it was stopped."""

    handle: JobHandle
    command: str
    started_iso: str
    stdout_offset: int = 0
    stderr_offset: int = 0
    killed: bool = False
    # Monotonic time the job was first observed finished, stamped lazily; drives the finished-job TTL sweep.
    finished_at: float | None = field(default=None)


class JobStore:
    """The server's in-memory background-job registry (one per lifespan)."""

    def __init__(self, adapter: ExegolPort, task_group: anyio.abc.TaskGroup, limits: ExecutionLimits) -> None:
        """Build the store.

        Args:
            adapter: The Exegol port the store launches, drains, reads and kills jobs through.
            task_group: A server-lifetime task group (created in the lifespan) the reader loops run in, so a
                background job outlives the request that started it (D-09).
            limits: The operator execution limits (buffer/output caps, concurrency cap, TTL, kill grace).
        """
        self._adapter = adapter
        self._task_group = task_group
        self._limits = limits
        self._jobs: dict[str, _JobState] = {}
        # Provisional foreground records (02-06): a running command whose reader is already supervised by the
        # store but which is not yet a visible job and does not count against max_concurrent_jobs. It is resolved
        # synchronously in the execution service — discarded on completion, killed on on_timeout=kill, or
        # promoted into _jobs on on_timeout=background. Invisible to list_jobs until promoted.
        self._provisional: dict[str, _JobState] = {}

    # --- Lifecycle -------------------------------------------------------------------------------------------

    async def start(self, name: str, command: str) -> str:
        """Start a background job and return its opaque id (EXEC-04).

        Raises:
            RuntimeError: If the server is already running max_concurrent_jobs visible jobs (D-10), or the
                container is not running (surfaced by the adapter).
            ValueError: If no container has that name (surfaced by the adapter).
        """
        self._sweep_finished()
        cap = self._limits.max_concurrent_jobs
        if self.running_count() >= cap:
            raise RuntimeError(JOB_CAP_REACHED_MESSAGE.format(cap=cap))
        job_id = secrets.token_urlsafe(16)
        await self._launch(job_id, name, command)
        return job_id

    async def seed(self, job_id: str, name: str, command: str) -> None:
        """Register a job under a caller-chosen id, bypassing the concurrency cap.

        Test-only hook used by the contract/protocol-era "call every tool" suites to make a known job id
        resolvable through the lifespan. Production never calls it: the lifespan only seeds adapters that expose
        a seed_jobs list, which the ServerContext does not.
        """
        await self._launch(job_id, name, command)

    async def _launch(self, job_id: str, name: str, command: str) -> None:
        handle = await self._adapter.start_job(name, command, job_id, cap_bytes=self._limits.job_buffer_cap_bytes)
        # The reader runs in the server-lifetime task group, so the job keeps draining after the request returns.
        self._task_group.start_soon(self._adapter.pump_job, handle)
        self._jobs[job_id] = _JobState(handle=handle, command=command, started_iso=_now_iso())
        # D-15: a watchdog in the same task group auto-kills a visible job that outruns max_job_runtime_s, even
        # when nobody ever polls it.
        self._task_group.start_soon(self._watchdog, job_id)

    async def read(self, job_id: str, stdout_offset: int | None = None, stderr_offset: int | None = None) -> JobInfo:
        """Return the job's status and the new output since the offsets (D-08/D-22).

        Omitted offsets resume from where the last read for this job ended. Each stream returns at most the
        operator per-read cap (output_cap_bytes); call again with the returned next offsets to page forward.

        Raises:
            ValueError: If no job has that id.
        """
        self._sweep_finished()
        state = self._require(job_id)
        so = state.stdout_offset if stdout_offset is None else stdout_offset
        eo = state.stderr_offset if stderr_offset is None else stderr_offset
        read = await self._adapter.read_job(state.handle, so, eo, max_bytes=self._limits.output_cap_bytes)
        state.stdout_offset = read.stdout_next_offset
        state.stderr_offset = read.stderr_next_offset
        self._stamp(state)
        return self._info(job_id, state, read)

    async def stop(self, job_id: str) -> JobInfo:
        """Kill a running job's tagged process group and return its final state (EXEC-06); idempotent.

        A job that already finished is left as-is and its final state is returned. Afterwards no in-container
        process carries the job's tag.

        Raises:
            ValueError: If no job has that id.
        """
        self._sweep_finished()
        state = self._require(job_id)
        if state.handle.running:
            await self._adapter.kill_job(state.handle.name, state.handle.tag, grace_s=self._limits.kill_grace_s)
            state.killed = True
            # Wait for the reader to observe the kill and set the completion event, bounded so stop never hangs.
            with anyio.move_on_after(self._limits.kill_grace_s + 5):
                await state.handle.done.wait()
        self._stamp(state)
        read = await self._adapter.read_job(
            state.handle, state.stdout_offset, state.stderr_offset, max_bytes=self._limits.output_cap_bytes
        )
        return self._info(job_id, state, read)

    def list(self) -> list[JobSummary]:
        """Return a summary of every job this server currently tracks (EXEC-05)."""
        self._sweep_finished()
        return [
            JobSummary(id=jid, status=self._status(s), started=s.started_iso, command=_preview(s.command))
            for jid, s in self._jobs.items()
        ]

    def running_count(self) -> int:
        """How many tracked jobs are still running (counts against max_concurrent_jobs; used by 02-06)."""
        return sum(1 for s in self._jobs.values() if s.handle.running and not s.killed)

    # --- Provisional foreground records (02-06 timeout / on_timeout promotion) --------------------------------

    async def start_provisional(self, name: str, command: str, job_id: str | None = None) -> tuple[str, JobHandle]:
        """Launch a foreground command as a provisional (not-yet-visible) job and return its id and handle.

        The pump runs in the server-lifetime task group, never inside the caller's deadline, so if the command
        overruns and is promoted a later get_job continues the SAME stream with no re-attach (RESEARCH Open
        Question 3). A provisional record is not counted against max_concurrent_jobs and is invisible to
        list_jobs until promoted (EXEC-02/D-12).

        Args:
            name: Container name.
            command: Command to run.
            job_id: The id/env tag to launch under (the tool mints it up front so a client cancellation can kill
                by tag, EXEC-07); a fresh opaque id is minted when omitted.

        Returns:
            The provisional job id (its env tag) and its JobHandle.
        """
        job_id = job_id or secrets.token_urlsafe(16)
        handle = await self._adapter.start_job(name, command, job_id, cap_bytes=self._limits.job_buffer_cap_bytes)
        self._task_group.start_soon(self._adapter.pump_job, handle)
        self._provisional[job_id] = _JobState(handle=handle, command=command, started_iso=_now_iso())
        return job_id, handle

    def promote(self, job_id: str) -> str | None:
        """Promote a provisional record to a visible job on on_timeout=background (D-12), or refuse when full.

        A promotion counts against max_concurrent_jobs exactly like a background start (D-10): when the server is
        already running its maximum, the record is left provisional and None is returned so the caller kills the
        command instead of letting a demoted job breach the cap.

        Raises:
            ValueError: If no provisional record has that id.

        Returns:
            The now-visible job id, or None when the concurrency cap is already full.
        """
        state = self._provisional.get(job_id)
        if state is None:
            raise ValueError(f"No provisional job with id '{job_id}'.")
        self._sweep_finished()
        if self.running_count() >= self._limits.max_concurrent_jobs:
            return None
        del self._provisional[job_id]
        self._jobs[job_id] = state
        # Now visible, so it is bounded by max_job_runtime_s like any background job (D-15).
        self._task_group.start_soon(self._watchdog, job_id)
        return job_id

    async def kill_provisional(self, job_id: str) -> None:
        """Kill a provisional job's tagged process group and drop the record (on_timeout=kill, or cap-full promotion).

        Afterwards no in-container process carries the job's tag (EXEC-06). A record that is not provisional
        (already resolved) is a no-op.
        """
        state = self._provisional.pop(job_id, None)
        if state is None:
            return
        if state.handle.running:
            await self._adapter.kill_job(state.handle.name, state.handle.tag, grace_s=self._limits.kill_grace_s)
            # Wait for the reader to observe the kill and set the completion event, bounded so it never hangs.
            with anyio.move_on_after(self._limits.kill_grace_s + 5):
                await state.handle.done.wait()

    def discard_provisional(self, job_id: str) -> None:
        """Drop a provisional record whose command completed before the deadline (nothing to kill)."""
        self._provisional.pop(job_id, None)

    # --- Internals -------------------------------------------------------------------------------------------

    async def _watchdog(self, job_id: str) -> None:
        """Auto-kill a visible job that outruns max_job_runtime_s, even if nobody polls it (D-15)."""
        await anyio.sleep(self._limits.max_job_runtime_s)
        state = self._jobs.get(job_id)
        if state is not None and state.handle.running and not state.killed:
            await self._adapter.kill_job(state.handle.name, state.handle.tag, grace_s=self._limits.kill_grace_s)
            state.killed = True

    def _require(self, job_id: str) -> _JobState:
        state = self._jobs.get(job_id)
        if state is None:
            raise ValueError(f"No job with id '{job_id}'. Call list_jobs to see the current jobs.")
        return state

    def _status(self, state: _JobState) -> JobStatus:
        if state.killed:
            return "killed"
        if state.handle.running:
            return "running"
        return "completed" if state.handle.exit_code == 0 else "failed"

    def _stamp(self, state: _JobState) -> None:
        """Record when a job was first seen finished, so the TTL sweep can expire it later."""
        if not state.handle.running and state.finished_at is None:
            state.finished_at = time.monotonic()

    def _sweep_finished(self) -> None:
        """Drop finished jobs older than finished_job_ttl_s (D-10), opportunistically on each access."""
        now = time.monotonic()
        ttl = self._limits.finished_job_ttl_s
        for job_id in list(self._jobs):
            state = self._jobs[job_id]
            self._stamp(state)
            if state.finished_at is not None and (now - state.finished_at) > ttl:
                del self._jobs[job_id]

    def _info(self, job_id: str, state: _JobState, read: JobRead) -> JobInfo:
        return JobInfo(
            id=job_id,
            status=self._status(state),
            exit_code=read.exit_code,
            stdout=read.stdout,
            stderr=read.stderr,
            stdout_next_offset=read.stdout_next_offset,
            stderr_next_offset=read.stderr_next_offset,
            stdout_gap=read.stdout_gap,
            stderr_gap=read.stderr_gap,
            stdout_bytes_dropped=read.stdout_bytes_dropped,
            stderr_bytes_dropped=read.stderr_bytes_dropped,
            started=state.started_iso,
        )
