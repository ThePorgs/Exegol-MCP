"""In-container exec service (SDK-free: it only talks to an ExegolPort)."""

import secrets

import anyio

from exegol_mcp.adapter.port import ContainerState, ExegolPort, JobRead
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import BoundedExecResult
from exegol_mcp.services.containers import ensure_running
from exegol_mcp.services.jobs import JobStore


def _bounded(read: JobRead, *, timed_out: bool, exit_code: int | None, job_id: str | None = None) -> BoundedExecResult:
    """Build a BoundedExecResult around the unchanged D-04 fields (EXEC-08): exit_code is passed verbatim and a
    null is never coerced to 0; the per-stream dropped-byte counts come from the tail buffers (EXEC-03)."""
    return BoundedExecResult(
        stdout=read.stdout,
        stderr=read.stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        job_id=job_id,
        stdout_bytes_dropped=read.stdout_bytes_dropped,
        stderr_bytes_dropped=read.stderr_bytes_dropped,
    )


async def require_running(adapter: ExegolPort, name: str) -> ContainerState:
    """Return the live state of a container that exists and is running.

    Shared guard for every service that runs something inside a container.

    Args:
        adapter: The Exegol port.
        name: Name of the container.

    Returns:
        The live ContainerState (always running on success).

    Raises:
        ValueError: If no container has that name.
        RuntimeError: If the container exists but is not running.
    """
    state = await adapter.get_container_state(name)
    if state is None:
        raise ValueError(f"Container '{name}' not found. Call list_containers to see the existing containers.")
    if not state.running:
        raise RuntimeError(
            f"Container '{name}' is not running. Please start it first using the 'start_container' tool."
        )
    return state


async def exec_command(
    adapter: ExegolPort,
    name: str,
    command: str,
    *,
    timeout: float | None = None,
    on_timeout: str = "background",
    limits: ExecutionLimits | None = None,
    background: bool = False,
    jobs: JobStore | None = None,
    tag: str | None = None,
    auto_start: bool = False,
    start_timeout_s: float | None = None,
) -> BoundedExecResult:
    """Run a shell command inside a running container, bounded, foreground or background (EXEC-02/03/04, EXEC-08).

    Foreground (background=False): the command runs through the tagged, streamed job primitive as a provisional
    job whose reader is supervised by the store, bounded by a per-call timeout. ``timeout`` is a float number of
    seconds; when omitted the operator default (execution.timeout_default_s, 50s) applies, and any value is
    clamped to the operator maximum (execution.timeout_max_s, 10 min) so a huge request cannot pin a worker
    thread past the cap (T-02-05). On expiry the agent-chosen ``on_timeout`` decides the outcome (D-12):
    ``background`` (the default) promotes the still-running command to a visible job carrying the partial output
    plus timed_out=True and a job_id to poll or stop; ``kill`` terminates it and returns the partial output with
    exit_code=None. A promotion counts against max_concurrent_jobs like a background start (D-10): when the cap is
    already full the command is killed instead and the result carries no job_id. Output over the per-stream cap
    keeps the tail and reports how many bytes were dropped (EXEC-03); the separate stdout/stderr and the real
    exit code are preserved and a null exit code is never coerced to 0 (EXEC-08).

    Background (background=True): the command is started under the job store and the call returns at once with an
    opaque job_id, empty output and a null exit code (EXEC-04); poll it with get_job and stop it with stop_job.

    Args:
        adapter: The Exegol port.
        name: Name of the container, as returned by list_containers.
        command: Command to run inside the container.
        timeout: Per-call foreground timeout in seconds; None applies the operator default, any value is clamped
            to the operator maximum. Must be positive when given.
        on_timeout: What to do when the foreground timeout expires: "background" (default, demote to a job) or
            "kill" (terminate and return the partial output).
        limits: Execution limits (timeout default/max, the per-stream output cap, concurrency cap, kill grace,
            max job runtime). None uses the built-in defaults.
        background: Start the command in the background and return a job id immediately (EXEC-04).
        jobs: The background-job store. Required when background is True; the foreground timeout/promotion path
            also uses it (a foreground call without a store runs inline and unbounded, for ad-hoc service use).
        tag: The env tag/job id to launch a foreground command under; the tool mints it up front so a client
            cancellation can kill the process group by tag (EXEC-07). A fresh opaque id is minted when omitted.
        auto_start: Start the container first when it is stopped (LIFE-07, D-15). The start is bounded by
            start_timeout_s and completes before the exec timeout begins, so the timeout keeps its meaning.
        start_timeout_s: Container start timeout in seconds for auto_start; None uses the built-in
            container_op_timeout_s default (D-18). The tool passes the operator-configured value.

    Returns:
        Foreground completed in time: a BoundedExecResult with separate stdout/stderr tail-capped at
        limits.output_cap_bytes, the real exit code and the per-stream dropped-byte counts (job_id None,
        timed_out False). Foreground timed out: partial output with timed_out True, exit_code None, and a job_id
        when demoted to a background job (None when killed). Background: empty output, exit_code None and the new
        job's job_id.

    Raises:
        ValueError: If no container has that name, the timeout is not positive, or on_timeout is not
            "kill"/"background".
        RuntimeError: If the container is not running and auto_start is false, an auto_start does not finish
            within start_timeout_s, background is requested without a job store, or the background concurrency
            cap is reached.
    """
    effective = limits if limits is not None else ExecutionLimits()
    # 0. Validate the bounded-exec inputs (V5): a non-positive timeout or an unknown on_timeout is the agent's to
    #    fix (ValueError), not a host-state problem. Clamping (not rejection) handles an over-large timeout.
    if on_timeout not in ("kill", "background"):
        raise ValueError(f"on_timeout must be 'kill' or 'background', not {on_timeout!r}.")
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, not {timeout!r}.")
    # 1. The container must exist and run. A stopped one is started first only on an explicit auto_start (D-15),
    #    bounded by its own container-op timeout and finished before the exec deadline below begins.
    await ensure_running(adapter, name, auto_start=auto_start, timeout_s=start_timeout_s)
    # 2. Background: hand the command to the job store and return the job id at once (EXEC-04).
    if background:
        if jobs is None:
            raise RuntimeError(
                "Background execution is not available on this server; run the command in the foreground instead."
            )
        job_id = await jobs.start(name, command)
        return BoundedExecResult(stdout="", stderr="", exit_code=None, job_id=job_id)
    # 3. Foreground without a job store (ad-hoc service call): run inline and unbounded, as in 02-02.
    if jobs is None:
        job_tag = tag or secrets.token_urlsafe(16)
        handle = await adapter.start_job(name, command, job_tag, cap_bytes=effective.output_cap_bytes)
        await adapter.pump_job(handle)
        read = await adapter.read_job(handle, 0, 0, None)
        return _bounded(read, timed_out=False, exit_code=read.exit_code)
    # 4. Foreground with a store (EXEC-02): launch a provisional record whose pump runs in the store's task group
    #    (never inside the deadline, so a promoted job's later reads continue the same stream — RESEARCH Open
    #    Question 3), then wait on its completion under the operator-clamped timeout with anyio.fail_after.
    deadline = min(timeout if timeout is not None else effective.timeout_default_s, effective.timeout_max_s)
    job_id, handle = await jobs.start_provisional(name, command, tag)
    try:
        with anyio.fail_after(deadline):
            await handle.done.wait()
    except TimeoutError:
        # Read the partial output already buffered; the exit code is null (the command did not finish).
        read = await adapter.read_job(handle, 0, 0, None)
        if on_timeout == "kill":
            await jobs.kill_provisional(job_id)
            return _bounded(read, timed_out=True, exit_code=None)
        promoted = jobs.promote(job_id)
        if promoted is None:
            # The visible-job cap is full: a demoted job would breach D-10, so kill instead of promoting.
            await jobs.kill_provisional(job_id)
            return _bounded(read, timed_out=True, exit_code=None)
        return _bounded(read, timed_out=True, exit_code=None, job_id=promoted)
    # 5. Completed in time: discard the provisional record and return the full bounded result (timed_out False).
    read = await adapter.read_job(handle, 0, 0, None)
    jobs.discard_provisional(job_id)
    return _bounded(read, timed_out=False, exit_code=read.exit_code)
