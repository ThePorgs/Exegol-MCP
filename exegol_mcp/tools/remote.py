"""MCP tool handler for netexec remote execution. Thin wrapper: adapter from the lifespan, logic in the service."""

import logging
import secrets
from typing import Any, Literal

import anyio
from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import BoundedExecResult
from exegol_mcp.services import remote as remote_service
from exegol_mcp.tools.containers import _container_op_timeout
from exegol_mcp.tools.errors import tool_errors

_log = logging.getLogger(__name__)

# Fixed debug text for a failed kill-on-cancel: it never carries the command, the tag or any argument.
_KILL_ON_CANCEL_FAILED = "kill-by-tag after a cancelled call failed; the cancellation still propagates"


async def exec_remote_command(
    container_name: str,
    protocol: str,
    target_ip: str,
    username: str,
    password: str,
    command: str,
    domain: str | None = None,
    port: int | None = None,
    use_powershell: bool = False,
    background: bool = False,
    timeout: float | None = None,
    on_timeout: Literal["kill", "background"] = "background",
    auto_start: bool = False,
    *,
    ctx: Context[ExegolPort, Any],
) -> BoundedExecResult:
    """Run a command on a remote host over a network protocol (SSH, WinRM, SMB, MSSQL, WMI, RDP) using the
    container's installed remote-execution tooling (netexec is the current backend).

    This runs the command without a persistent session. It is more specific than exec_command: use it for every
    remote network connection instead of running a client by hand. A connection or authentication failure comes
    back as a result, not an error. exit_code is the backend's own exit code, passed through unchanged, and it
    does not tell you whether the login worked: netexec, the current backend, exits 0 even when authentication
    fails. Read the output instead: a rejected login shows as a line containing `[-]`. The marker is not always at
    the front: it can follow the backend's protocol/host prefix, as in
    `SMB 10.0.0.1 445 DC01 [-] CORP\\user:password STATUS_LOGON_FAILURE`, and this holds for the parsed result
    and for the raw output get_job returns, on every supported protocol. Empty output with exit code 0 usually
    means the backend never reached the service (wrong host or port).

    IMPORTANT: The container must be running. Start it with start_container first, or pass auto_start=true to have
    this tool start a stopped container before the remote command runs (the start is bounded by the server's
    container start timeout and does not count against the command timeout). The output is not
    redacted: it can contain credentials the remote tool prints, including the password you passed. If either
    stream exceeds the operator's size cap, it is cut from the FRONT (oldest) to keep the tail and the result
    reports how many bytes were dropped per stream — a size limit, not redaction. Credentials you pass persist
    outside this server: in your client's transcript and logs, as tool state inside the container, and on the
    backend's argv where in-container `ps` can read them — an accepted exposure, because the container is your own
    disposable toolbox and the server's own logs never record the command or password (no-log policy). The full
    guidance on limiting this persistence is on the Exegol MCP Security page: https://docs.exegol.com/mcp/security

    A foreground remote command is bounded exactly like exec_command. Give `timeout` (seconds) or let the operator
    default apply; any value is clamped to the operator maximum. When it expires, `on_timeout` decides the outcome:
    on_timeout="background" (the default) turns the overrunning sweep into a background job you then poll with
    get_job(job_id) — the result comes back with timed_out=true, exit_code null and that job_id; on_timeout="kill"
    stops it and returns the partial output with timed_out=true and exit_code null. A timed-out or killed command
    reports exit code null, never 0.

    Set background=true up front for a remote command you already know is long (a wide sweep or a password spray):
    the tool returns at once with a job_id and empty output instead of waiting. Poll it with get_job(job_id) and
    stop it with stop_job(job_id). A background or timed-out remote command returns the RAW netexec output (its log
    lines included) through get_job — the log-line parsing that a synchronous result gets cannot run incrementally
    on those reads. A synchronous foreground result returns the parsed command output, netexec's log lines removed.

    RELATED TOOLS:
    - exec_command: run a command locally inside the container (not over the network)
    - get_job: poll a backgrounded or timed-out remote command and read its (raw) output incrementally
    - stop_job: stop a backgrounded remote command
    - list_jobs: list the background jobs running on this server
    - list_containers: choose the container to run netexec from
    - start_container: start the container before remote operations
    - list_installed_tools: discover the network and security tools available in the container

    SUPPORTED PROTOCOLS:
    - ssh: Secure Shell connections
    - winrm: Windows Remote Management
    - smb: Server Message Block (Windows file sharing)
    - mssql: Microsoft SQL Server
    - wmi: Windows Management Instrumentation
    - rdp: Remote Desktop Protocol

    Args:
        container_name: Name of the Exegol container to run netexec from.
        protocol: Protocol to use (ssh, winrm, smb, mssql, wmi, rdp).
        target_ip: Target IP address or hostname.
        username: Username for authentication.
        password: Password for authentication.
        command: Command to run on the target.
        domain: Optional domain name (for Windows protocols like smb and winrm).
        port: Port of the remote service (1-65535). Omit it to use the protocol's default port.
        use_powershell: Use the PowerShell flag (-X) instead of -x. Useful for SMB or WinRM PowerShell commands.
        background: Start the remote command in the background and return a job_id at once instead of waiting.
        timeout: Foreground timeout in seconds. Omit to apply the operator default; any value is clamped to the
            operator maximum. Must be positive.
        on_timeout: What to do when a foreground remote command overruns its timeout: "background" (default,
            demote it to a job you poll with get_job) or "kill" (stop it and return the partial output).
        auto_start: Start the container first if it is stopped (default false). A running container is left as
            is; an unknown one is still an error.

    Returns:
        A foreground run that finished in time: the parsed command output, the invocation's stderr, the backend's
        real exit code (null when unknown; 0 does not mean the login succeeded), the per-stream bytes dropped to fit
        the output cap, timed_out false and job_id null. A foreground run that timed out: the raw partial output,
        timed_out true, exit_code null, and a job_id when it was demoted to a background job (null when killed). A
        background run: empty output, exit_code null and the job_id to poll with get_job.
    """
    # 1. The adapter is injected through the server lifespan; the tool never imports Exegol or Docker.
    adapter = ctx.request_context.lifespan_context
    # 2. Read the operator execution limits from the lifespan context; built-in defaults when it carries none.
    config = getattr(adapter, "config", None)
    limits = config.execution if config is not None else ExecutionLimits()
    # 3. The background-job store the lifespan attached (None on an ad-hoc context without one).
    jobs = getattr(adapter, "jobs", None)
    # 4. Mint the foreground tag up front so a client cancellation mid-run can kill the netexec process group by
    #    tag (EXEC-07). The service launches the provisional foreground command under this exact tag.
    tag = secrets.token_urlsafe(16)
    # 5. Delegate to the service; anticipated errors reach the client through tool_errors(). A client cancellation
    #    is delivered as scope-cancellation of this handler (mcp 2.x, no Context flag): catch it, kill the tagged
    #    netexec process group from a shielded scope, then re-raise. Per the spec a cancelled request gets no
    #    response, so no job id is handed out on cancel (D-13). This mirrors exec_command's kill-on-cancel and
    #    lives inside tool_errors, whose __aexit__ passes BaseExceptions (cancellation) through untouched.
    async with tool_errors(adapter):
        try:
            return await remote_service.exec_remote_command(
                adapter,
                container_name,
                protocol,
                target_ip,
                username,
                password,
                command,
                domain=domain,
                port=port,
                use_powershell=use_powershell,
                timeout=timeout,
                on_timeout=on_timeout,
                background=background,
                limits=limits,
                jobs=jobs,
                tag=tag,
                auto_start=auto_start,
                start_timeout_s=_container_op_timeout(adapter),
            )
        except anyio.get_cancelled_exc_class():
            # A background start owns its process through the store and returns at once, so it is not subject to
            # this foreground kill-on-cancel; only a foreground command's tagged netexec process must be terminated.
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
