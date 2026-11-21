"""Container file transfer through exec (FILE-01, FILE-02, D-16, D-18, D-20).

One transfer path per operation, the same for every path in the container (rootfs, /proc, /sys, /dev, tmpfs and
bind mounts). There is no archive-first branch with a fallback, because a live probe (RESEARCH Pattern 5) measured:
- get_archive answers 404 for /proc, /sys and tmpfs files, so reads would need a fallback anyway;
- put_archive into a tmpfs reports success but the file never appears, and put_archive onto a symlink replaces the
  link instead of writing its target;
- a demuxed, non-TTY exec returns stdout byte for byte;
- the exec umask is 0000, so anything an exec creates is world-writable unless the script sets its own umask.

So a read is a probe exec plus one bounded `head -c` exec, and the raw stdout bytes are kept as they are: file bytes
never pass through decode_output or build_exec_result. A write stages the payload with put_archive into a fresh
/.exegol-mcp-<16 hex>/ directory on the container rootfs (where put_archive is reliable), then one exec copies it
to the target with a shell redirection, applies an optional mode and checks the written size. A tmpfs target
therefore really receives the bytes, a symlink's target is updated, and the staging directory is always removed.

Every command is a fixed `/bin/sh -c <script> sh <path> ...` argv with no environment of its own: paths, sizes and
modes are positional parameters ("$1", "$2"...), never formatted into the script text, so a path such as
`/tmp/$(id)` is only ever a file name (T-04-29). Every script starts with `umask 022` (Pitfall 15) and runs its
potentially blocking command under coreutils `timeout -s KILL`, so a blocking pseudo file can never pin a worker
thread (Pitfall 15, T-04-32). File contents and payloads are never logged or copied into an error message.
"""

import contextlib
import io
import logging
import math
import secrets
import tarfile
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import anyio
import docker.errors

from exegol_mcp.adapter.exec import _docker_handle
from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.port import DirListing, DirRecord, FileRead, FileWrite
from exegol_mcp.adapter.tail_buffer import decode_output

_log = logging.getLogger(__name__)

# The exit status of a command coreutils `timeout -s KILL` stopped at its limit (128 + SIGKILL).
_TIME_LIMIT_EXIT = 137
# The copy script's exit status when the target does not hold exactly the bytes that were sent.
_SIZE_MISMATCH_EXIT = 7
# The write prep script's exit status when the target's parent directory cannot be created.
_NO_PARENT_EXIT = 5

# Resolve "$1" (symlinks followed, "." and ".." removed, missing components allowed) into P. The `printf x` guard
# keeps a trailing newline of the name through the command substitution; the two expansions then strip the guard
# and the newline realpath printed.
_RESOLVE = 'P=$(timeout -s KILL "{t}" realpath -m -- "$1" && printf x) || exit $?; P=${{P%x}}; P=${{P%?}}; '

# Probe ($1 path, $2 seconds): print "<resolved>\0<kind>\0<size>\0", kind one of missing|directory|fifo|socket|file.
# The kind and size are taken on the original path, the one the read uses: a per-process link such as /proc/self
# resolves to the probe's own, short-lived process, so its resolved path is only reported, never tested.
_PROBE_SCRIPT = (
    "umask 022; "
    + _RESOLVE.format(t="$2")
    + 'if [ -d "$1" ]; then K=directory; elif [ -p "$1" ]; then K=fifo; elif [ -S "$1" ]; then K=socket; '
    'elif [ -e "$1" ]; then K=file; else K=missing; fi; '
    'S=0; if [ "$K" != missing ]; then S=$(timeout -s KILL "$2" stat -L -c %s -- "$1") || exit $?; fi; '
    r"printf '%s\0%s\0%s\0' " + '"$P" "$K" "$S"'
)

# Read ($1 path, $2 byte count, $3 seconds): the first $2 bytes of the file, raw. The original path is read (the
# probe already refused directories, FIFOs and sockets), so per-process links such as /proc/self keep their meaning.
_READ_SCRIPT = 'umask 022; exec timeout -s KILL "$3" head -c "$2" -- "$1"'

# Write prep ($1 path, $2 "1" to overwrite or "0", $3 seconds): print "<resolved>\0<status>\0<existed>\0" with status
# one of ok|directory|fifo|socket|exists. Every refusal is decided before anything is created; only for "ok" is the
# parent created (mode 0755 under umask 022). A parent that cannot be created exits 5.
_WRITE_PREP_SCRIPT = (
    "umask 022; "
    + _RESOLVE.format(t="$3")
    + 'E=0; if [ -d "$P" ]; then R=directory; elif [ -p "$P" ]; then R=fifo; elif [ -S "$P" ]; then R=socket; '
    'elif [ -e "$P" ]; then E=1; if [ "$2" = 0 ]; then R=exists; else R=ok; fi; else R=ok; fi; '
    'if [ "$R" = ok ]; then D=${P%/*}; [ -n "$D" ] || D=/; '
    'timeout -s KILL "$3" mkdir -p -- "$D"; S=$?; '
    'if [ "$S" -ne 0 ]; then [ "$S" -eq 137 ] || S=5; exit "$S"; fi; fi; '
    r"printf '%s\0%s\0%s\0' " + '"$P" "$R" "$E"'
)

# Copy ($1 target, $2 staging directory, $3 mode or "", $4 expected size, $5 seconds). The trap removes the staging
# directory on every exit. The redirection and the size read run in an inner shell under the timeout, so even
# opening the target is bounded. Exit 7 when the target does not hold exactly $4 bytes.
_COPY_SCRIPT = (
    "umask 022; "
    "trap 'rm -rf -- \"$2\"' EXIT; "
    'timeout -s KILL "$5" sh -c \'cat -- "$1" > "$2"\' sh "$2/data" "$1" || exit $?; '
    'if [ -n "$3" ]; then chmod "$3" -- "$1" || exit $?; fi; '
    'N=$(timeout -s KILL "$5" sh -c \'wc -c < "$1"\' sh "$1") || exit $?; '
    '[ "$N" -eq "$4" ] || exit 7'
)

# Cleanup ($1 staging directory): the fallback removal when the copy exec never ran to completion.
_CLEANUP_SCRIPT = 'umask 022; rm -rf -- "$1"'

# List ($1 path, $2 depth, $3 NUL-terminated field cap, $4 seconds): print "<resolved>\0", then six NUL-terminated
# fields per entry: kind letter, size, octal mode, mtime, symlink target, relative path. Missing exits 3, not a
# directory exits 4. find -H follows a symlinked starting point only; entries are never followed (D-19). head -z
# caps the output; the fd 3/4 plumbing carries find's own exit status past the pipe so a timeout still exits 137.
_LIST_SCRIPT = (
    "umask 022; "
    + _RESOLVE.format(t="$4")
    + '[ -e "$1" ] || exit 3; [ -d "$1" ] || exit 4; '
    r"printf '%s\0' " + '"$P"; '
    "exec 4>&1; "
    'R=$( { { timeout -s KILL "$4" find -H "$1" -mindepth 1 -maxdepth "$2" '
    r"-printf '%y\0%s\0%m\0%T@\0%l\0%P\0' "
    '2>/dev/null; printf %s "$?" >&3; } | head -z -n "$3" >&4; } 3>&1 ); '
    '[ "$R" = 137 ] && exit 137; exit 0'
)

# Exit statuses of the list script.
_LIST_MISSING_EXIT = 3
_LIST_NOT_A_DIRECTORY_EXIT = 4
# The fields find prints per entry.
_LIST_FIELDS = 6
_KINDS: dict[str, Literal["file", "dir", "symlink"]] = {"f": "file", "d": "dir", "l": "symlink"}

# The in-container staging directory lives at the rootfs root, where put_archive is reliable.
_STAGING_PREFIX = ".exegol-mcp-"

# One copy at a time per (container, resolved target) in this process: [lock, number of writes holding or waiting].
_TARGET_LOCKS: dict[tuple[str, str], list[Any]] = {}


def _run_posix_raw(docker_container: Any, script: str, *args: str) -> tuple[bytes, bytes, int | None]:
    """Run a fixed /bin/sh script with positional arguments and return raw (stdout, stderr, exit code).

    Same argv shape as exec._run_posix (no environment, demuxed, non-TTY), but stdout is never decoded: it may hold
    file bytes. The arguments reach the script as "$1", "$2"... and are never part of the script text.
    """
    result = docker_container.exec_run(
        ["/bin/sh", "-c", script, "sh", *args],
        detach=False,
        stream=False,
        demux=True,
        tty=False,
    )
    stdout, stderr = result.output if result.output is not None else (None, None)
    return stdout or b"", stderr or b"", result.exit_code


def staging_tar(token: str, payload: bytes) -> bytes:
    """Build the staging archive: a directory .exegol-mcp-<token> (0700) holding the file data (0600), root-owned.

    mtime is set explicitly: an unset tar mtime shows up as 1970 in the container.
    """
    buf = io.BytesIO()
    now = int(time.time())
    with tarfile.open(fileobj=buf, mode="w") as tar:
        directory = tarfile.TarInfo(f"{_STAGING_PREFIX}{token}")
        directory.type, directory.mode, directory.mtime = tarfile.DIRTYPE, 0o700, now
        directory.uid = directory.gid = 0
        tar.addfile(directory)
        data = tarfile.TarInfo(f"{_STAGING_PREFIX}{token}/data")
        data.size, data.mode, data.mtime = len(payload), 0o600, now
        data.uid = data.gid = 0
        tar.addfile(data, io.BytesIO(payload))
    return buf.getvalue()


def _seconds(timeout_s: float) -> str:
    # Whole seconds, rounded up and at least 1: `timeout 0` would mean no limit at all.
    return str(max(1, math.ceil(timeout_s)))


def _fields(stdout: bytes, count: int, what: str) -> list[bytes]:
    fields = stdout.split(b"\0")
    if len(fields) < count + 1:
        raise RuntimeError(f"Unexpected output from the in-container {what} (are GNU coreutils installed?)")
    return fields[:count]


def _command_error(action: str, path: str, exit_code: int | None, stderr: bytes, timeout_s: float) -> RuntimeError:
    """Map a failed in-container command to a RuntimeError. stderr is a command's error text, never file bytes."""
    if exit_code == _TIME_LIMIT_EXIT:
        return RuntimeError(
            f"{action} '{path}' did not finish within the {_seconds(timeout_s)}s in-container time limit"
        )
    detail = decode_output(stderr).strip()[:500]
    reason = f": {detail}" if detail else ""
    return RuntimeError(f"{action} '{path}' failed in the container (exit code {exit_code}){reason}")


def _refuse_special(kind: str, path: str) -> None:
    if kind == "fifo":
        raise ValueError(f"'{path}' is a FIFO (named pipe): file transfer refuses it, since it could block")
    if kind == "socket":
        raise ValueError(f"'{path}' is a socket: file transfer refuses it")


async def read_container_file(container: Any, path: str, max_bytes: int, timeout_s: float) -> FileRead:
    """Read at most max_bytes + 1 raw bytes of path in a running container (FILE-01).

    Args:
        container: The running ExegolContainer. The caller checks that it exists and runs.
        path: The in-container path; any mount works.
        max_bytes: The size limit. One more byte is read so a larger file is detected.
        timeout_s: The bound on each in-container command.

    Returns:
        The raw bytes, the size (the bytes read when within the limit; the stat size beyond it, or None for a pseudo
        file that reports 0), whether the file is over the limit, and the resolved path.

    Raises:
        ValueError: The path is missing, a directory, a FIFO or a socket.
        RuntimeError: An in-container command hit its time limit or failed.
    """
    docker_container = _docker_handle(container)
    seconds = _seconds(timeout_s)

    # 1. Probe: resolved path, kind and stat size, with the path as "$1".
    stdout, stderr, code = await run_blocking(_run_posix_raw, docker_container, _PROBE_SCRIPT, path, seconds)
    if code != 0:
        raise _command_error("Probing", path, code, stderr, timeout_s)
    raw_resolved, raw_kind, raw_size = _fields(stdout, 3, "file probe")
    resolved, kind = decode_output(raw_resolved), decode_output(raw_kind)
    if kind == "missing":
        raise ValueError(f"No such file in the container: '{path}'")
    if kind == "directory":
        raise ValueError(f"'{path}' is a directory: use list_directory to see its entries")
    _refuse_special(kind, path)
    probe_size = int(raw_size)

    # 2. Bounded raw read: one byte past the limit reveals a larger file, even a pseudo file whose stat size is 0.
    data, stderr, code = await run_blocking(
        _run_posix_raw, docker_container, _READ_SCRIPT, path, str(max_bytes + 1), seconds
    )
    if code != 0:
        raise _command_error("Reading", path, code, stderr, timeout_s)
    over_limit = len(data) > max_bytes
    if not over_limit:
        size: int | None = len(data)
    else:
        size = probe_size if probe_size > max_bytes else None
    return FileRead(data=data, size=size, over_limit=over_limit, path=resolved)


async def write_container_file(
    container: Any, path: str, data: bytes, overwrite: bool, mode: int | None, timeout_s: float
) -> FileWrite:
    """Write data to path in a running container through a staged archive and an exec copy (FILE-02, D-18).

    Args:
        container: The running ExegolContainer. The caller checks that it exists and runs.
        path: The in-container target path; any mount works and a symlink's target is written.
        data: The payload bytes.
        overwrite: When False, an existing target is refused.
        mode: Permission bits to apply after the write (0 to 0o7777), or None to keep the shell defaults (0644 for a
            new file, the existing mode otherwise).
        timeout_s: The bound on each in-container command.

    Returns:
        The resolved path written, the payload size and whether the file was created.

    Raises:
        ValueError: An invalid mode, a directory, FIFO or socket target, an existing target without overwrite, or a
            parent directory that cannot be created. Nothing is staged in these cases.
        RuntimeError: The write did not land (a refused staging archive, a size mismatch), or an in-container
            command hit its time limit or failed.
    """
    if mode is not None and not 0 <= mode <= 0o7777:
        raise ValueError(f"Invalid mode {mode!r}: give permission bits between 0 and 0o7777 (for example 0o755)")
    docker_container = _docker_handle(container)
    seconds = _seconds(timeout_s)

    # 1. Prep: every refusal before any byte is staged, then the parent directory (0755).
    stdout, stderr, code = await run_blocking(
        _run_posix_raw, docker_container, _WRITE_PREP_SCRIPT, path, "1" if overwrite else "0", seconds
    )
    if code == _NO_PARENT_EXIT:
        detail = decode_output(stderr).strip()[:500]
        raise ValueError(f"Cannot create the parent directory of '{path}' in the container: {detail}")
    if code != 0:
        raise _command_error("Preparing the write to", path, code, stderr, timeout_s)
    raw_resolved, raw_status, raw_existed = _fields(stdout, 3, "write prep")
    status = decode_output(raw_status)
    if status == "directory":
        raise ValueError(f"'{path}' is a directory: write_file writes files only")
    if status == "exists":
        raise ValueError(f"'{path}' already exists and overwrite is false: pass overwrite=true to replace it")
    _refuse_special(status, path)
    # Copy to the resolved path the prep checked. A name that is not valid UTF-8 (reached through a symlink) cannot
    # cross the exec argv unchanged, so the original path is used then: the redirection resolves it the same way.
    try:
        target = raw_resolved.decode("utf-8")
    except UnicodeDecodeError:
        target = path
    created = decode_output(raw_existed) != "1"

    # 2. Stage the payload on the rootfs in a fresh, private directory.
    token = secrets.token_hex(8)
    staging_dir = f"/{_STAGING_PREFIX}{token}"
    mode_text = "" if mode is None else f"{mode:04o}"
    copy_completed = False
    try:
        # docker-py 7.1 put_archive raises APIError on any refusal (read-only rootfs, full disk, 404) before it
        # returns, so its return value carries nothing. The finally still removes the staging directory (WR-07).
        try:
            await run_blocking(docker_container.put_archive, "/", staging_tar(token, data))
        except docker.errors.APIError as e:
            detail = e.explanation or str(e)
            raise RuntimeError(
                f"The write to '{path}' did not land: Docker refused the staging archive ({detail})"
            ) from None
        # 3. Copy with shell-redirection semantics, apply the mode, check the size. The script's trap removes the
        # staging directory whatever happens once it runs. Copies to one target are serialized: two redirections
        # truncating the same file at once would interleave and fail each other's size check.
        async with _target_lock((getattr(container, "name", None) or str(id(docker_container)), target)):
            _, stderr, code = await run_blocking(
                _run_posix_raw, docker_container, _COPY_SCRIPT, target, staging_dir, mode_text, str(len(data)), seconds
            )
        copy_completed = True
    finally:
        if not copy_completed:
            await _remove_staging(docker_container, staging_dir)
    if code == _SIZE_MISMATCH_EXIT:
        raise RuntimeError(
            f"The write to '{path}' did not land: the file does not hold the {len(data)} bytes that were sent "
            "(a pseudo file, a full filesystem or a concurrent writer)"
        )
    if code != 0:
        raise _command_error("Writing", path, code, stderr, timeout_s)
    return FileWrite(path=decode_output(raw_resolved), size=len(data), created=created)


@contextlib.asynccontextmanager
async def _target_lock(key: tuple[str, str]) -> AsyncIterator[None]:
    """Hold the per-(container, target) copy lock; the entry is dropped once no write waits on it.

    Only touched on the event loop, so the dict needs no lock of its own.
    """
    entry = _TARGET_LOCKS.get(key)
    if entry is None:
        entry = _TARGET_LOCKS[key] = [anyio.Lock(), 0]
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if entry[1] == 0:
            del _TARGET_LOCKS[key]


async def _remove_staging(docker_container: Any, staging_dir: str) -> None:
    """Best-effort removal of a staging directory the copy exec never cleaned up (T-04-34).

    Shielded so a cancelled write still removes it. A failure here is logged, never raised: it must not hide the
    error that ended the write.
    """
    with anyio.CancelScope(shield=True):
        try:
            await run_blocking(_run_posix_raw, docker_container, _CLEANUP_SCRIPT, staging_dir)
        except Exception as e:  # noqa: BLE001  # best effort: the original failure is what the caller must see
            _log.debug("Could not remove the staging directory %s: %s", staging_dir, e)


def _dir_record(fields: list[bytes]) -> DirRecord:
    kind_letter, size, mode, mtime, target, name = (decode_output(field) for field in fields)
    try:
        return DirRecord(
            path=name,
            kind=_KINDS.get(kind_letter, "other"),
            size=int(size),
            mode=int(mode, 8),
            mtime=float(mtime),
            target=target or None,
        )
    except ValueError:
        raise RuntimeError("Unexpected output from the in-container directory listing") from None


async def list_container_directory(
    container: Any, path: str, depth: int, max_entries: int, timeout_s: float
) -> DirListing:
    """List a directory in a running container with one bounded, NUL-separated find exec (FILE-03, D-19).

    Args:
        container: The running ExegolContainer. The caller checks that it exists and runs.
        path: The in-container directory.
        depth: How many levels to descend (1: the direct entries).
        max_entries: The most entries returned; one more is read to detect truncation.
        timeout_s: The bound on the in-container find.

    Returns:
        The resolved directory path, at most max_entries entries in find order (unsorted) and whether more existed.
        Entry names are decoded lossily (U+FFFD for bytes that are not UTF-8), for display only.

    Raises:
        ValueError: An invalid depth or max_entries, a missing path, or a path that is not a directory.
        RuntimeError: The find hit its time limit, failed, or printed output that does not parse.
    """
    if depth < 1 or max_entries < 1:
        raise ValueError("depth and max_entries must both be at least 1")
    docker_container = _docker_handle(container)
    cap = str(_LIST_FIELDS * (max_entries + 1))
    stdout, stderr, code = await run_blocking(
        _run_posix_raw, docker_container, _LIST_SCRIPT, path, str(depth), cap, _seconds(timeout_s)
    )
    if code == _LIST_MISSING_EXIT:
        raise ValueError(f"No such file or directory in the container: '{path}'")
    if code == _LIST_NOT_A_DIRECTORY_EXIT:
        raise ValueError(f"'{path}' is not a directory: use read_file to read a file")
    if code != 0:
        raise _command_error("Listing", path, code, stderr, timeout_s)
    [resolved] = _fields(stdout, 1, "directory listing")
    # The fields after the resolved path, without what follows the last NUL. A trailing partial record (fewer than
    # six fields) is dropped: only whole records count.
    fields = stdout.split(b"\0")[1:-1]
    count = len(fields) // _LIST_FIELDS
    entries = [
        _dir_record(fields[index * _LIST_FIELDS : (index + 1) * _LIST_FIELDS])
        for index in range(min(count, max_entries))
    ]
    return DirListing(path=decode_output(resolved), entries=entries, truncated=count > max_entries)
