"""MCP tool handlers for background jobs. Thin wrappers: the store comes from the lifespan, logic in the store.

get_job / stop_job / list_jobs are the agent-facing half of the background-job model (EXEC-05/06). They read the
JobStore the build_server lifespan attached to the lifespan context and run its calls inside tool_errors(), so an
unknown job id reaches the agent as a readable error, never a crash.
"""

from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.job import JobInfo, JobSummary
from exegol_mcp.services.jobs import JobStore
from exegol_mcp.tools.errors import tool_errors


def _job_store(adapter: object) -> JobStore:
    """Return the background-job store the lifespan attached to the context, or raise a clear error."""
    store = getattr(adapter, "jobs", None)
    if store is None:
        raise RuntimeError("Background jobs are not available on this server; run commands in the foreground.")
    return store


# EXEC-05: background-job polling.
async def get_job(
    job_id: str,
    stdout_offset: int | None = None,
    stderr_offset: int | None = None,
    *,
    ctx: Context[ExegolPort, Any],
) -> JobInfo:
    """Poll a background job and read its new output since the given offsets.

    IMPORTANT: Start a background job with exec_command(background=true) (or exec_remote_command), which returns a
    job_id at once. Each call returns only the NEW output since the offsets you pass; to continue reading, pass
    back the stdout_next_offset / stderr_next_offset this returned, or omit the offsets to resume where your last
    read ended. exit_code is null while the job is still running (status "running") and is the real exit code once
    it finishes; it is never coerced to 0. A non-zero stdout_gap/stderr_gap means that many bytes were dropped by
    the live buffer cap before you read them (a size effect, not redaction); stdout/stderr output is returned
    unredacted.

    RELATED TOOLS:
    - exec_command: start a command (set background=true to get a job_id to poll here)
    - exec_remote_command: start a remote command in the background the same way
    - stop_job: stop a running job (kills the whole in-container process group)
    - list_jobs: see every background job on this server

    Args:
        job_id: The job id returned when the background command was started.
        stdout_offset: Byte offset to read stdout from; omit to resume from your last read.
        stderr_offset: Byte offset to read stderr from; omit to resume from your last read.

    Returns:
        The job status, the real exit code (null while running), the new stdout/stderr since the offsets, the next
        offsets to pass back, the skipped-gap counts and the total bytes dropped to the buffer cap.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await _job_store(adapter).read(job_id, stdout_offset, stderr_offset)


# EXEC-06: background-job stop.
async def stop_job(job_id: str, *, ctx: Context[ExegolPort, Any]) -> JobInfo:
    """Stop a background job, terminating the whole in-container process group.

    IMPORTANT: This sends SIGTERM to the job's process group, then SIGKILL after the operator grace period, so the
    command (and anything it spawned) is really gone from the container afterwards. It is idempotent: stopping a job
    that has already finished just returns its final state. Read any remaining output with get_job first if you need
    it. An unknown job id returns a clear error.

    RELATED TOOLS:
    - get_job: read a job's output and status before stopping it
    - list_jobs: find the job id to stop
    - exec_command: start a command (background=true) that this can later stop

    Args:
        job_id: The job id to stop.

    Returns:
        The job's final state: status "killed" (or its completed/failed state if it had already finished) and the
        exit code (null when it was killed before exiting).
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await _job_store(adapter).stop(job_id)


# EXEC-05: background-job listing.
async def list_jobs(*, ctx: Context[ExegolPort, Any]) -> list[JobSummary]:
    """List the background jobs this server is tracking.

    IMPORTANT: This only reads state. It returns every job this server currently knows about (running and recently
    finished, until they are swept after the operator retention period), each with its id, status, start time and a
    short command preview. Use get_job with an id here to read a job's output, or stop_job to stop it.

    RELATED TOOLS:
    - get_job: read the status and output of one job from this list
    - stop_job: stop one of the listed jobs
    - exec_command: start a new background command (background=true)

    Returns:
        One summary per tracked job (id, status, start time, command preview); an empty list when there are none.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return _job_store(adapter).list()
