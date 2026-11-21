"""Published result shapes of the background-job tools (EXEC-04/05/06, D-07/D-08).

Tasks-shaped so LONG-01 can later swap native MCP Tasks behind the same tool surface. JobInfo is what get_job
and stop_job return (status, the real exit code, the new output since an offset, the next offsets to page with,
and the gap/dropped counts); JobSummary is one row of list_jobs. These reach the LLM (FastMCP derives the tools'
output schema from them), so every field carries a description. The gap and dropped counts are bytes lost to the
buffer cap for size, never redaction (SEC-02): command output is returned unredacted.
"""

from typing import Literal

from pydantic import BaseModel, Field

# A job's lifecycle state. "running" while the command is in flight; "completed" on exit 0; "failed" on a nonzero
# or unknown exit; "killed" after stop_job (or cancellation); "timed_out" is added by 02-06's on_timeout path.
JobStatus = Literal["running", "completed", "failed", "killed", "timed_out"]


# EXEC-05
class JobInfo(BaseModel):
    """Status and one incremental, offset-based read of a background job's two output streams."""

    id: str = Field(description="Opaque job id (unguessable); pass it to get_job and stop_job")
    status: JobStatus = Field(
        description="running, completed (exit 0), failed (nonzero/unknown exit), killed (stopped), or timed_out",
    )
    exit_code: int | None = Field(
        description="Real exit code once the job finished; null while it is still running and never coerced to 0",
    )
    stdout: str = Field(
        description="New standard output since stdout_offset (at most the operator per-read cap), never redacted",
    )
    stderr: str = Field(description="New standard error since stderr_offset, kept separate from stdout")
    stdout_next_offset: int = Field(
        description="Pass this back as stdout_offset on the next get_job to continue reading stdout where this left off",
    )
    stderr_next_offset: int = Field(description="Pass this back as stderr_offset to continue reading stderr")
    stdout_gap: int = Field(
        description=(
            "Bytes of stdout skipped because the requested offset was behind the retained buffer window. A size "
            "effect of the live buffer cap, not redaction. 0 when nothing was skipped"
        ),
    )
    stderr_gap: int = Field(description="Bytes of stderr skipped because the requested offset was behind the window")
    stdout_bytes_dropped: int = Field(
        description="Total stdout bytes dropped from the FRONT (oldest) to fit the buffer cap; a size marker, not redaction",
    )
    stderr_bytes_dropped: int = Field(description="Total stderr bytes dropped from the front to fit the buffer cap")
    started: str = Field(description="UTC timestamp (ISO 8601) when the job was started")


# EXEC-05
class JobSummary(BaseModel):
    """One row of list_jobs: a job's id, status, start time and a short command preview."""

    id: str = Field(description="Opaque job id; pass it to get_job or stop_job")
    status: JobStatus = Field(description="running, completed, failed, killed or timed_out")
    started: str = Field(description="UTC timestamp (ISO 8601) when the job was started")
    command: str = Field(description="Short preview of the command the job runs (truncated with an ellipsis)")
