"""MCP tool handler for in-container command execution. Thin wrapper: adapter from the lifespan, logic in the service."""

import logging
import secrets
from typing import Any, Literal

import anyio
from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import BoundedExecResult
from exegol_mcp.services import execution as execution_service
from exegol_mcp.tools.containers import _container_op_timeout
from exegol_mcp.tools.errors import tool_errors

_log = logging.getLogger(__name__)

# Fixed debug text for a failed kill-on-cancel: it never carries the command, the tag or any argument.
_KILL_ON_CANCEL_FAILED = "kill-by-tag after a cancelled call failed; the cancellation still propagates"


async def exec_command(
    container_name: str,
    command: str,
    background: bool = False,
    timeout: float | None = None,
    on_timeout: Literal["kill", "background"] = "background",
    auto_start: bool = False,
    *,
    ctx: Context[ExegolPort, Any],
) -> BoundedExecResult:
    """Run a shell command inside a running Exegol container and return its output and exit code.

    IMPORTANT: The container must be running. Start it with start_container first, or pass auto_start=true to
    have this tool start a stopped container before running the command (the start is bounded by the server's
    container start timeout and does not count against the command timeout). Without auto_start a stopped
    container returns a clear error asking you to start it. For remote network operations (SSH, WinRM, SMB,
    MSSQL, WMI, RDP) use exec_remote_command instead of running a client by hand here. If the output of either
    stream exceeds the operator's size cap, it is cut from the FRONT (oldest) to keep the tail, and the result
    reports how many bytes were dropped per stream in stdout_bytes_dropped / stderr_bytes_dropped. This is a
    size limit, not redaction: command output (credentials a tool prints included) is returned unredacted.

    A foreground command is bounded by a timeout (seconds): give `timeout` or let the operator default apply;
    any value is clamped to the operator maximum. When it expires, `on_timeout` decides the outcome:
    on_timeout="background" (the default) turns the overrunning command into a background job you then poll with
    get_job(job_id) — the result comes back with timed_out=true, exit_code null and that job_id; on_timeout="kill"
    stops it and returns the partial output with timed_out=true and exit_code null. A timed-out or killed command
    reports exit code null, never 0.

    Set background=true up front for a command you already know is long (a full-port scan, a long brute force):
    the tool returns at once with a job_id and empty output instead of waiting. Poll the job and read its output
    incrementally with get_job(job_id), stop it with stop_job(job_id), and see all jobs with list_jobs. Use the
    foreground default (background=false) for commands whose output you want straight back.

    RELATED TOOLS:
    - exec_remote_command: run a command on a remote host over a network protocol (more specific than this tool)
    - start_container: start the container before running anything in it
    - get_job: poll a background job (started here or demoted on timeout) and read its output incrementally
    - stop_job: stop a background job started here
    - list_jobs: list the background jobs running on this server
    - list_installed_tools: discover which tools are available in the container
    - get_tool_help: read the help of a specific tool before using it

    Args:
        container_name: Name of the container, as returned by list_containers.
        command: Command to run inside the container.
        background: Start the command in the background and return a job_id at once instead of waiting for it.
        timeout: Foreground timeout in seconds. Omit to apply the operator default; any value is clamped to the
            operator maximum. Must be positive.
        on_timeout: What to do when a foreground command overruns its timeout: "background" (default, demote it
            to a job you poll with get_job) or "kill" (stop it and return the partial output).
        auto_start: Start the container first if it is stopped (default false). A running container is left as
            is; an unknown one is still an error.

    Returns:
        A foreground run that finished in time: separate stdout and stderr, the real exit code (null when
        unknown), the per-stream bytes dropped to fit the output cap, timed_out false and job_id null. A
        foreground run that timed out: the partial output, timed_out true, exit_code null, and a job_id when it
        was demoted to a background job (null when killed). A background run: empty output, exit_code null and
        the job_id to poll with get_job.
    """
    # 1. The adapter is injected through the server lifespan; the tool never imports Exegol or Docker.
    adapter = ctx.request_context.lifespan_context
    # 2. Read the operator execution limits from the lifespan context; built-in defaults when it carries none.
    config = getattr(adapter, "config", None)
    limits = config.execution if config is not None else ExecutionLimits()
    # 3. The background-job store the lifespan attached (None on an ad-hoc context without one).
    jobs = getattr(adapter, "jobs", None)
    # 4. Mint the foreground tag up front so a client cancellation mid-exec can kill the in-container process
    #    group by tag (EXEC-07). The service launches the provisional foreground command under this exact tag.
    tag = secrets.token_urlsafe(16)
    # 5. Delegate to the service; anticipated errors reach the client through tool_errors(). A client
    #    cancellation is delivered as scope-cancellation of this handler (mcp 2.x, no Context flag): catch it,
    #    kill the tagged process group from a shielded scope, then re-raise. Per the 2026-07-28 spec a cancelled
    #    request gets no response, so no job id is handed out on cancel (D-13). This lives inside tool_errors:
    #    its __aexit__ passes BaseExceptions (cancellation) through untouched, so the re-raise is safe.
    async with tool_errors(adapter):
        try:
            return await execution_service.exec_command(
                adapter,
                container_name,
                command,
                timeout=timeout,
                on_timeout=on_timeout,
                limits=limits,
                background=background,
                jobs=jobs,
                tag=tag,
                auto_start=auto_start,
                start_timeout_s=_container_op_timeout(adapter),
            )
        except anyio.get_cancelled_exc_class():
            # A background start owns its process through the store and returns at once, so it is not subject to
            # this foreground kill-on-cancel; only a foreground command's tagged process must be terminated.
            if not background:
                with anyio.CancelScope(shield=True):
                    # Best-effort: a cancel can land while auto_start is still starting the container, and a
                    # kill-by-tag on a container that is not running raises, as can the Docker socket (an OSError)
                    # or the API (an APIError when the container stops mid-call). Any such kill error is swallowed:
                    # it must never replace the cancellation (EXEC-07), so it is logged with a fixed message and no
                    # arguments. BaseException (the cancellation itself) is never caught here.
                    try:
                        await adapter.kill_job(container_name, tag, grace_s=limits.kill_grace_s)
                    except Exception:  # noqa: BLE001 - a failed kill must never replace the cancellation (EXEC-07)
                        _log.debug(_KILL_ON_CANCEL_FAILED)
            raise
