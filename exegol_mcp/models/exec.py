"""Published result shape of a command run inside an Exegol container (D-04)."""

from pydantic import BaseModel, Field


class ExecResult(BaseModel):
    """Result of a shell command run inside an Exegol container"""

    stdout: str = Field(
        description=(
            "Standard output of the command, never redacted. exec_command returns it exactly as produced; "
            "exec_remote_command returns the remote command's output without netexec's log lines"
        ),
    )
    stderr: str = Field(description="Standard error of the command, kept separate from stdout")
    exit_code: int | None = Field(
        description="Real exit code of the command. null means the exit code is unknown; it is never coerced to 0",
    )


# EXEC-03 (Phase 2): execution-bounds fields around the EXEC-08 published contract.
class BoundedExecResult(ExecResult):
    """A command result with the execution-bounds fields.

    Subclasses ExecResult: its three fields (stdout, stderr, exit_code) are inherited unchanged, so the
    published result contract is preserved and only new, defaulted fields are added around it.
    """

    timed_out: bool = Field(
        default=False,
        description="Whether the command hit its timeout. False for a foreground command that completed in time",
    )
    job_id: str | None = Field(
        default=None,
        description="Set when the command became a background job (e.g. on timeout); null for a plain foreground run",
    )
    stdout_bytes_dropped: int = Field(
        default=0,
        description=(
            "Bytes removed from the FRONT (oldest) of stdout because it exceeded the operator output cap, keeping "
            "the tail. A size marker, not redaction: output is returned unredacted. 0 means nothing was dropped"
        ),
    )
    stderr_bytes_dropped: int = Field(
        default=0,
        description="Bytes removed from the front of stderr for the same per-stream cap; 0 means nothing was dropped",
    )
