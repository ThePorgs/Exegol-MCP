"""Demuxed command execution inside an Exegol container (D-04).

ExegolContainer.exec_raw is not used: it merges stdout and stderr, decodes them as strict UTF-8, and makes a
synchronous docker call from an async method, which blocked the whole v1 server for the length of the command.
Here the docker-py exec runs with demux on in a worker thread, so the result keeps the two streams apart and the
real exit code, and the event loop stays free.
"""

from typing import Any

from exegol.model.ExegolContainer import ExegolContainer

from exegol_mcp.adapter.offload import jobs_limiter, run_blocking
from exegol_mcp.adapter.port import JOB_TAG_ENV, JobHandle, JobRead
from exegol_mcp.adapter.tail_buffer import TailBuffer, decode_output
from exegol_mcp.models.exec import ExecResult

# ExegolContainer keeps its docker-py Container in this private (name-mangled) attribute and has no public
# accessor for it.
_DOCKER_HANDLE = "_ExegolContainer__container"

# Returned by next() once the exec output generator is exhausted (the stream-reader sentinel).
_END_OF_STREAM: Any = object()


def _decode(stream: bytes | None) -> str:
    # Undecodable bytes become U+FFFD instead of failing the call; whitespace is kept exactly as produced.
    # Delegates to the one shared SDK-free decoder so the real adapter, the fake and the services agree.
    return "" if stream is None else decode_output(stream)


def build_exec_result(output: tuple[bytes | None, bytes | None] | None, exit_code: int | None) -> ExecResult:
    """Turn a demuxed docker exec result into an ExecResult.

    Args:
        output: The (stdout, stderr) pair docker-py returns with demux=True. Either side is None when the command
            wrote nothing to it; the whole pair may be None when there was no output at all.
        exit_code: The exit code docker reported, or None when it is unknown.

    Returns:
        The two streams decoded separately and the exit code as reported: None is never turned into 0.
    """
    stdout, stderr = output if output is not None else (None, None)
    return ExecResult(stdout=_decode(stdout), stderr=_decode(stderr), exit_code=exit_code)


def _run_exec(docker_container: Any, command: str) -> ExecResult:
    # Reuse the SDK's zsh wrapping (aliases and ~/.zshrc loaded, the command passed through $CMD) so a command
    # behaves as it does from the exegol wrapper.
    payload, str_cmd = ExegolContainer.formatShellCommand(command, quiet=True)
    result = docker_container.exec_run(
        payload,
        environment={"CMD": str_cmd, "DISABLE_AUTO_UPDATE": "true"},
        detach=False,
        stream=False,
        demux=True,
        tty=False,
    )
    return build_exec_result(result.output, result.exit_code)


def _docker_handle(container: Any) -> Any:
    """Return the docker-py Container behind an ExegolContainer, or raise if the SDK moved the private handle."""
    docker_container = getattr(container, _DOCKER_HANDLE, None)
    if docker_container is None:
        raise RuntimeError(
            f"Cannot reach the docker container behind the Exegol container: the installed exegol SDK no longer "
            f"exposes {_DOCKER_HANDLE}. This exegol release is not compatible with this exegol-mcp version."
        )
    return docker_container


async def exec_demuxed(container: Any, command: str) -> ExecResult:
    """Run a shell command in a running Exegol container without blocking the event loop.

    Args:
        container: The ExegolContainer to run the command in. The caller checks that it is running.
        command: The shell command line.

    Returns:
        Separate stdout and stderr and the real exit code (D-04).

    Raises:
        RuntimeError: If the installed exegol SDK no longer exposes the container's docker handle.
    """
    return await run_blocking(_run_exec, _docker_handle(container), command)


# --- Tagged, process-grouped, streamed job primitive (EXEC-03/04/06, D-09/D-16/D-20..22) --------------------------

# Launcher prefix for every tagged job. See _create_and_start_job for why the wait flag is load-bearing.
_SETSID_WAIT = "setsid -w"


def _create_and_start_job(docker_container: Any, command: str, tag: str) -> tuple[str, Any, Any]:
    """Create and start a tagged, process-grouped, streamed demuxed exec.

    Reuses the SDK's zsh wrapping (as the foreground exec does) and rides the unique EXEGOL_MCP_JOB tag on the
    environment alongside the existing CMD / DISABLE_AUTO_UPDATE. The low-level exec_create/exec_start keep the
    exec id so pump_job can read the real exit code from exec_inspect once the stream ends.

    The launcher is `setsid -w`, and the wait flag is load-bearing. Inside docker exec the process already leads
    its own process group, so util-linux setsid forks the command into a child and, without -w, the parent exits
    0 immediately: the exec stream ends about two seconds later, exec_inspect reports ExitCode 0, and the real
    command keeps running untracked (G-02-1 (a), G-02-3, G-02-4). With -w the parent waits for the child, so the
    stream, the output and the exit code all belong to the real command. The child still gets its own process
    group, which kill-by-tag targets (D-16).

    Returns:
        (exec_id, output generator of (stdout_chunk|None, stderr_chunk|None) tuples, the docker APIClient).
    """
    payload, str_cmd = ExegolContainer.formatShellCommand(command, quiet=True)
    api = docker_container.client.api
    exec_id = api.exec_create(
        docker_container.id,
        f"{_SETSID_WAIT} {payload}",
        environment={"CMD": str_cmd, "DISABLE_AUTO_UPDATE": "true", JOB_TAG_ENV: tag},
        tty=False,
    )["Id"]
    # docker-py disables the socket read timeout for a streamed exec, so a long job does not trip it.
    generator = api.exec_start(exec_id, stream=True, demux=True)
    return exec_id, generator, api


async def start_tagged_job(container: Any, name: str, command: str, tag: str, cap_bytes: int) -> JobHandle:
    """Launch the tagged, streamed exec and return a JobHandle with two bounded TailBuffers (running, no exit)."""
    docker_container = _docker_handle(container)
    handle = JobHandle(name=name, tag=tag, stdout=TailBuffer(cap_bytes), stderr=TailBuffer(cap_bytes))
    handle.stream = await run_blocking(_create_and_start_job, docker_container, command, tag)
    return handle


async def pump_tagged_job(handle: JobHandle) -> None:
    """Drain the demuxed output into the handle's buffers, then record the exit code and mark it done."""
    exec_id, generator, api = handle.stream
    lines = iter(generator)
    try:
        while True:
            # Each next() is a blocking socket read: pull it off the loop on the dedicated jobs limiter so job
            # readers never draw from the default pool (RESEARCH Open Question 2, resolved).
            chunk = await run_blocking(next, lines, _END_OF_STREAM, limiter=jobs_limiter)
            if chunk is _END_OF_STREAM:
                break
            stdout_chunk, stderr_chunk = chunk
            if stdout_chunk:
                handle.stdout.append(stdout_chunk)
            if stderr_chunk:
                handle.stderr.append(stderr_chunk)
    finally:
        # The exit code is valid only once the stream ends; exec_inspect reports None while Running and is
        # never coerced to 0 (D-04). A failure here must still finish the handle, not leave a reader hung.
        try:
            info = await run_blocking(api.exec_inspect, exec_id)
            handle.exit_code = info.get("ExitCode")
        except Exception:  # noqa: BLE001  # completing the handle must not mask the exit-code lookup failing
            handle.exit_code = None
        handle.running = False
        handle.done.set()


async def read_tagged_job(
    handle: JobHandle, stdout_offset: int, stderr_offset: int, max_bytes: int | None
) -> JobRead:
    """Read both bounded buffers forward from the given offsets and build the JobRead (D-08/D-22)."""
    out_data, out_next, out_gap = handle.stdout.read(stdout_offset, limit=max_bytes)
    err_data, err_next, err_gap = handle.stderr.read(stderr_offset, limit=max_bytes)
    return JobRead(
        stdout=decode_output(out_data),
        stderr=decode_output(err_data),
        stdout_next_offset=out_next,
        stderr_next_offset=err_next,
        stdout_gap=out_gap,
        stderr_gap=err_gap,
        stdout_bytes_dropped=handle.stdout.dropped,
        stderr_bytes_dropped=handle.stderr.dropped,
        exit_code=handle.exit_code,
        running=handle.running,
    )


def _run_posix(docker_container: Any, script: str) -> ExecResult:
    """Run a fixed /bin/sh script as an explicit argv, with NO environment.

    Unlike _run_exec, this bypasses ExegolContainer.formatShellCommand: the kill and sweep scripts rely on POSIX
    word splitting of unquoted parameters, which the SDK's zsh wrapper (`eval "$CMD"`) does not do, so the kill
    loops signalled nothing (G-02-1 (b)). Passing no environment also means the script carries no job tag and can
    never match itself (T-02-19).
    """
    result = docker_container.exec_run(
        ["/bin/sh", "-c", script],
        detach=False,
        stream=False,
        demux=True,
        tty=False,
    )
    return build_exec_result(result.output, result.exit_code)


def _kill_script(tag: str, grace_s: float) -> str:
    """Shell that kills every process group carrying EXEGOL_MCP_JOB=<tag> (SIGTERM, then SIGKILL after grace_s).

    Runs as a fixed ['/bin/sh', '-c', script] argv with no environment (see _run_posix / kill_tagged_job), so it
    carries no job tag and never matches itself, and it does not go through the SDK's zsh wrapper. That wrapper
    was the bug (G-02-1 (b)): zsh does not word-split an unquoted parameter, so the `for G in $PGIDS` loops ran
    once with the whole list as a single word and killed nothing. Under /bin/sh the parameters split as POSIX
    intends.

    Scans each process's NUL-delimited environ for the exact tag entry: grep -z with ^...$ anchors the whole
    entry, so only EXEGOL_MCP_JOB=<tag> matches, never a prefix (T-02-02). With setsid -w a job spans two tagged
    process groups (the waiting launcher's and the command's); both are matched and signalled. A matched process
    whose group cannot be resolved is signalled by PID. When nothing matched, the grace sleep is skipped and the
    script exits at once. A final pass re-reads the environ and SIGKILLs by PID anything that still carries the
    exact entry, covering a tagged process that left its group.
    """
    entry = f"{JOB_TAG_ENV}={tag}"
    return (
        f'ENTRY="{entry}"; PGIDS=""; PIDS=""; '
        "for E in /proc/[0-9]*/environ; do "
        'if grep -qz -- "^$ENTRY$" "$E" 2>/dev/null; then '
        "P=${E#/proc/}; P=${P%/environ}; "
        'G=$(ps -o pgid= -p "$P" 2>/dev/null | tr -d " "); '
        'if [ -n "$G" ]; then PGIDS="$PGIDS $G"; else PIDS="$PIDS $P"; fi; '
        "fi; done; "
        '[ -z "$PGIDS$PIDS" ] && exit 0; '
        'for G in $PGIDS; do kill -TERM -"$G" 2>/dev/null; done; '
        'for P in $PIDS; do kill -TERM "$P" 2>/dev/null; done; '
        f"sleep {grace_s}; "
        'for G in $PGIDS; do kill -KILL -"$G" 2>/dev/null; done; '
        'for P in $PIDS; do kill -KILL "$P" 2>/dev/null; done; '
        "for E in /proc/[0-9]*/environ; do "
        'if grep -qz -- "^$ENTRY$" "$E" 2>/dev/null; then '
        "P=${E#/proc/}; P=${P%/environ}; "
        'kill -KILL "$P" 2>/dev/null; '
        "fi; done; "
        "exit 0"
    )


def _sweep_script(grace_s: float) -> str:
    """Shell that kills every tagged process group and echoes how many DISTINCT jobs (tags) it found.

    Runs as a fixed ['/bin/sh', '-c', script] argv with no environment, for the same reason as the kill script:
    the SDK's zsh wrapper does not word-split unquoted parameters, so the kill loops signalled nothing (G-02-1
    (b)); under /bin/sh they split as POSIX intends, and the untagged script never matches itself.

    Matches any process whose environ holds an EXEGOL_MCP_JOB= entry (anchored prefix, T-02-02) and signals its
    group (SIGTERM, grace, SIGKILL; a process whose group cannot be resolved is signalled by PID; the grace sleep
    is skipped when nothing matched). The printed count is the number of DISTINCT EXEGOL_MCP_JOB values, i.e. the
    number of orphaned jobs, not the raw process-group count: setsid -w doubles the latter because each job spans
    the waiting launcher's group and the command's group under one tag.
    """
    prefix = f"{JOB_TAG_ENV}="
    return (
        f'PREFIX="{prefix}"; PGIDS=""; PIDS=""; TAGS=""; '
        "for E in /proc/[0-9]*/environ; do "
        'V=$(tr "\\0" "\\n" < "$E" 2>/dev/null | grep -- "^$PREFIX"); '
        '[ -n "$V" ] || continue; '
        "P=${E#/proc/}; P=${P%/environ}; "
        'TAGS=$(printf "%s\\n%s" "$TAGS" "$V"); '
        'G=$(ps -o pgid= -p "$P" 2>/dev/null | tr -d " "); '
        'if [ -n "$G" ]; then PGIDS="$PGIDS $G"; else PIDS="$PIDS $P"; fi; '
        "done; "
        'N=$(printf "%s\\n" "$TAGS" | sort -u | grep -c -- "^$PREFIX"); '
        'if [ -n "$PGIDS$PIDS" ]; then '
        'for G in $PGIDS; do kill -TERM -"$G" 2>/dev/null; done; '
        'for P in $PIDS; do kill -TERM "$P" 2>/dev/null; done; '
        f"sleep {grace_s}; "
        'for G in $PGIDS; do kill -KILL -"$G" 2>/dev/null; done; '
        'for P in $PIDS; do kill -KILL "$P" 2>/dev/null; done; '
        "fi; "
        'echo "$N"'
    )


async def kill_tagged_job(container: Any, tag: str, grace_s: float) -> None:
    """Run the kill-by-tag shell in the container (off the loop) as a plain /bin/sh argv with no job tag. No file
    is written in the container (D-09)."""
    await run_blocking(_run_posix, _docker_handle(container), _kill_script(tag, grace_s))


async def sweep_tagged_jobs(container: Any, grace_s: float) -> int:
    """Run the sweep shell in the container as a plain /bin/sh argv and return how many distinct tagged jobs it
    killed (one per EXEGOL_MCP_JOB value, not the process-group count that setsid -w doubles)."""
    result = await run_blocking(_run_posix, _docker_handle(container), _sweep_script(grace_s))
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return 0
    try:
        return int(lines[-1].strip())
    except ValueError:
        return 0
