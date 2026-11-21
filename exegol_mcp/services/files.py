"""Container file service: read_file, write_file and list_directory (FILE-01..04, D-16..D-20).

SDK-free: it only talks to an ExegolPort. It owns the rules the port does not: the container path rule (absolute,
no NUL, bounded length), the one raw-byte size limit (files.max_bytes), the UTF-8-or-base64 encoding of a read,
the length prechecks and decoding of a write payload, the refusal of per-process write targets, the octal mode
string, and the sorting, mtime clamping and paging of a directory listing. A file over the limit is an error, never
cut to fit (D-17). Nothing here logs: paths, sizes tied to paths and contents stay out of the server
logs (Phase 3 D-08), and contents reach the agent unredacted (Phase 3 D-09).
"""

import base64
import math
import posixpath
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from exegol_mcp.adapter.port import DirRecord, ExegolPort
from exegol_mcp.models.files import DirectoryPage, DirEntry, FileContent, FileWritten
from exegol_mcp.services.containers import ensure_running
from exegol_mcp.services.paging import page

# D-20: the longest container path accepted, in characters (Linux PATH_MAX).
CONTAINER_PATH_MAX_LEN = 4096

# D-17/FILE-04: the errors for a transfer over files.max_bytes. They state the exact raw-byte size when it is known
# (no rounding) and the limit, name the config key, and point to the routes that have no limit.
_BIGGER_TRANSFERS = (
    "It is never cut to fit. Move larger files through a host-mounted workspace (create_container's workspace) or "
    "with exec_command (for example split, gzip or head/tail), or ask the operator to raise files.max_bytes."
)
FILE_TOO_LARGE_MESSAGE = "{subject} is {size} bytes, over the {limit}-byte limit set by files.max_bytes. " + (
    _BIGGER_TRANSFERS
)
FILE_LARGER_THAN_MESSAGE = (
    "{subject} is larger than the {limit}-byte limit set by files.max_bytes (its exact size is unknown). "
    + _BIGGER_TRANSFERS
)

# D-18: a write mode is an octal permission string of 3 or 4 digits, with an optional leading 0 ("644", "0755",
# "4755", "04755").
MODE_RE = re.compile(r"^0?[0-7]{3,4}$")

# IN-05: per-process aliases. The write prep resolves a path inside its own short-lived helper shell, so these would
# name that shell's descriptors or /proc entry, never what the agent meant. Exact names, then whole subtrees.
_PER_PROCESS_NAMES = frozenset(
    {"/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/fd", "/proc/self", "/proc/thread-self"}
)
_PER_PROCESS_PREFIXES = ("/dev/fd/", "/proc/self/", "/proc/thread-self/")

# WR-05: the epoch, and the clamped mtimes for a value datetime cannot hold (past 9999, before year 1, inf or NaN).
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MTIME_MAX_TEXT = datetime.max.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
_MTIME_MIN_TEXT = datetime.min.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def validate_container_path(path: str) -> str:
    """Check a container path against the D-20 rule before it reaches the port.

    Any location is allowed (/proc, /sys, /dev, tmpfs included); only the shape is checked.

    Args:
        path: The in-container path the agent gave.

    Returns:
        The path, unchanged.

    Raises:
        ValueError: If the path is not absolute, contains a NUL character or is longer than CONTAINER_PATH_MAX_LEN.
    """
    if "\x00" in path:
        raise ValueError("The container path contains a NUL character, which no path can hold.")
    if len(path) > CONTAINER_PATH_MAX_LEN:
        raise ValueError(
            f"The container path is {len(path)} characters long, over the {CONTAINER_PATH_MAX_LEN}-character limit."
        )
    if not path.startswith("/"):
        raise ValueError(f"The container path '{path}' is not absolute: give a path starting with '/'.")
    return path


def _refuse_per_process_target(path: str) -> None:
    """Refuse a write target that is a per-process alias (IN-05), after normalizing slashes and dot segments.

    A real /proc/<pid>/... path is not refused: only the aliases that would resolve against the helper shell are.

    Raises:
        ValueError: If the path is /dev/stdin, /dev/stdout, /dev/stderr, /dev/fd, /proc/self or /proc/thread-self,
            or lies under /dev/fd, /proc/self or /proc/thread-self.
    """
    # posixpath.normpath keeps exactly two leading slashes (POSIX leaves "//" implementation-defined), so collapse
    # every leading slash into one first.
    normalized = posixpath.normpath("/" + path.lstrip("/"))
    if normalized in _PER_PROCESS_NAMES or normalized.startswith(_PER_PROCESS_PREFIXES):
        raise ValueError(
            f"'{path}' is a per-process path that would resolve against a short-lived helper shell, so write_file "
            "refuses it. Write to a regular path, or use exec_command for redirection."
        )


def parse_mode(mode: str | None) -> int | None:
    """Parse a write_file mode string (D-18).

    Args:
        mode: An octal permission string such as "0755" or "644", or None to keep the default.

    Returns:
        The permission bits, or None when no mode was given.

    Raises:
        ValueError: If the string is not 3 or 4 octal digits with an optional leading 0.
    """
    if mode is None:
        return None
    if not MODE_RE.fullmatch(mode):
        raise ValueError(
            f"Invalid mode '{mode}': give the permission bits as an octal string of 3 or 4 digits, "
            "for example \"0755\" or \"644\"."
        )
    return int(mode, 8)


async def read_file(
    adapter: ExegolPort,
    name: str,
    path: str,
    *,
    encoding: Literal["auto", "base64"],
    max_bytes: int,
    op_timeout_s: float,
    auto_start: bool,
    start_timeout_s: float | None = None,
) -> FileContent:
    """Read a whole container file, as UTF-8 text when it is valid UTF-8 and as base64 otherwise (FILE-01, D-16).

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        path: Absolute in-container path of the file.
        encoding: "auto" returns valid (strict) UTF-8 as text and anything else as base64; "base64" always
            returns base64.
        max_bytes: The raw-byte size limit (files.max_bytes). A file of exactly max_bytes is accepted.
        op_timeout_s: Bound of each in-container command of the read.
        auto_start: Start the container first when it is stopped (LIFE-07, D-15).
        start_timeout_s: Bound of that start; None uses the built-in container_op_timeout_s default.

    Returns:
        The file content, its encoding, its raw-byte size and the resolved path.

    Raises:
        ValueError: On a bad path, an unknown container, a missing file, a directory, or a file over max_bytes.
        RuntimeError: If the container is stopped without auto_start, the start overruns, or the read overruns.
    """
    # 1. The path rule, before any adapter call.
    validate_container_path(path)
    # 2. The container must run, or be started when the agent asked for it.
    await ensure_running(adapter, name, auto_start=auto_start, timeout_s=start_timeout_s)
    # 3. One bounded read: the port takes at most max_bytes + 1 bytes, so over_limit is exact.
    result = await adapter.read_file(name, path, max_bytes, op_timeout_s)
    data = result.data
    # 4. Over the limit is an error, never a truncated result (D-17).
    if result.over_limit or len(data) > max_bytes:
        subject = f"The file '{path}'"
        if result.size is not None and result.size > max_bytes:
            raise ValueError(FILE_TOO_LARGE_MESSAGE.format(subject=subject, size=result.size, limit=max_bytes))
        raise ValueError(FILE_LARGER_THAN_MESSAGE.format(subject=subject, limit=max_bytes))
    # 5. Encode: strict UTF-8 (an encoded surrogate or a cut character falls back to base64), or forced base64.
    if encoding == "auto":
        try:
            return FileContent(path=result.path, encoding="utf-8", content=data.decode("utf-8"), size=len(data))
        except UnicodeDecodeError:
            pass
    return FileContent(
        path=result.path, encoding="base64", content=base64.b64encode(data).decode("ascii"), size=len(data)
    )


def _decode_payload(path: str, content: str, encoding: Literal["utf-8", "base64"], max_bytes: int) -> bytes:
    """Turn a write payload into raw bytes, refusing anything over max_bytes before it is encoded or decoded."""
    subject = f"The content for '{path}'"
    # 1. Length prechecks bound the memory an oversized payload can take (T-04-30). A UTF-8 character is at least
    #    one byte, and max_bytes bytes encode to exactly 4 * ceil(max_bytes / 3) padded base64 characters.
    if encoding == "utf-8":
        if len(content) > max_bytes:
            raise ValueError(
                f"{subject} is {len(content)} characters, so more than the {max_bytes}-byte limit set by "
                f"files.max_bytes. {_BIGGER_TRANSFERS}"
            )
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(
                "The content holds a character that cannot be written as UTF-8 (a lone UTF-16 surrogate). "
                "Send the bytes with encoding=\"base64\" instead."
            ) from None
    else:
        longest = 4 * math.ceil(max_bytes / 3)
        if len(content) > longest:
            raise ValueError(
                f"{subject} is {len(content)} base64 characters, more than the {longest} that the {max_bytes}-byte "
                f"limit set by files.max_bytes allows. {_BIGGER_TRANSFERS}"
            )
        # binascii.Error is a ValueError; a non-ASCII character raises a plain ValueError (IN-02).
        try:
            data = base64.b64decode(content, validate=True)
        except ValueError:
            raise ValueError(
                "The content is not valid base64: give standard, padded base64 (A-Z, a-z, 0-9, '+', '/', '=') "
                "with no whitespace, or use encoding=\"utf-8\" for text."
            ) from None
    # 2. The raw-byte limit itself, with the exact size (D-17).
    if len(data) > max_bytes:
        raise ValueError(FILE_TOO_LARGE_MESSAGE.format(subject=subject, size=len(data), limit=max_bytes))
    return data


async def write_file(
    adapter: ExegolPort,
    name: str,
    path: str,
    content: str,
    *,
    encoding: Literal["utf-8", "base64"],
    overwrite: bool,
    mode: str | None,
    max_bytes: int,
    op_timeout_s: float,
    auto_start: bool,
    start_timeout_s: float | None = None,
) -> FileWritten:
    """Write a file into a container from UTF-8 text or base64, as the container's root (FILE-02, D-18).

    Missing parent directories are created (0755), an existing file is replaced unless overwrite is false and
    keeps its owner and mode unless mode is given, and a symlink's target is written.

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        path: Absolute in-container path of the file.
        content: The payload, as text ("utf-8") or base64 ("base64").
        encoding: How content is encoded.
        overwrite: Replace an existing file; False refuses one.
        mode: Optional octal permission string such as "0755".
        max_bytes: The raw-byte size limit (files.max_bytes), applied to the decoded bytes.
        op_timeout_s: Bound of each in-container command of the write.
        auto_start: Start the container first when it is stopped (LIFE-07, D-15).
        start_timeout_s: Bound of that start; None uses the built-in container_op_timeout_s default.

    Returns:
        The path written, the raw-byte size and whether the file was created.

    Raises:
        ValueError: On a bad path or mode, a payload over max_bytes or not base64, an unknown container, a
            directory target, or an existing file with overwrite false.
        RuntimeError: If the container is stopped without auto_start, the start or the write overruns, or the
            write did not land.
    """
    # 1. The path rule, the per-process refusal and the mode, before any adapter call.
    validate_container_path(path)
    _refuse_per_process_target(path)
    mode_bits = parse_mode(mode)
    # 2-4. Length prechecks, then encode or decode, then the raw-byte limit.
    data = _decode_payload(path, content, encoding, max_bytes)
    # 5. The container must run, or be started when the agent asked for it; then one verified port write.
    await ensure_running(adapter, name, auto_start=auto_start, timeout_s=start_timeout_s)
    result = await adapter.write_file(name, path, data, overwrite, mode_bits, op_timeout_s)
    return FileWritten(path=result.path, size=result.size, created=result.created)


def _mtime_text(mtime: float) -> str:
    """Format an epoch mtime as ISO-8601 UTC to the second, clamped to datetime's range (WR-05).

    The epoch plus a timedelta does not depend on the platform's C library, unlike datetime.fromtimestamp. A value
    datetime cannot hold (past year 9999, before year 1, +-inf, NaN) maps to datetime.max when positive and to
    datetime.min otherwise, so one planted mtime never fails a whole listing.
    """
    try:
        return (_EPOCH + timedelta(seconds=mtime)).isoformat(timespec="seconds")
    except (OverflowError, ValueError):
        return _MTIME_MAX_TEXT if mtime > 0 else _MTIME_MIN_TEXT


def _dir_entry(record: DirRecord) -> DirEntry:
    """Map one port record to the published entry shape: octal mode string and ISO-8601 UTC mtime."""
    return DirEntry(
        name=posixpath.basename(record.path),
        path=record.path,
        type=record.kind,
        size=record.size,
        mode=f"{record.mode:04o}",
        mtime=_mtime_text(record.mtime),
        symlink_target=record.target,
    )


async def list_directory(
    adapter: ExegolPort,
    name: str,
    path: str,
    *,
    depth: int,
    cursor: str | None,
    page_size: int,
    max_depth: int,
    max_entries: int,
    op_timeout_s: float,
    auto_start: bool,
    start_timeout_s: float | None = None,
) -> DirectoryPage:
    """List a container directory, sorted by relative path and paged by an opaque cursor (FILE-03, D-19).

    Symlinks are reported with their target and never followed or descended into. The port collects at most
    max_entries entries within depth and says when it dropped some; truncated carries that to the agent.

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        path: Absolute in-container path of the directory.
        depth: How many levels to list (1: the direct entries).
        cursor: The opaque cursor from a previous page's next_cursor, or None for the first page.
        page_size: Entries per page (execution.listing_page_size).
        max_depth: The largest depth accepted (files.max_list_depth).
        max_entries: The most entries collected before paging (files.max_list_entries).
        op_timeout_s: Bound of the in-container listing command.
        auto_start: Start the container first when it is stopped (LIFE-07, D-15).
        start_timeout_s: Bound of that start; None uses the built-in container_op_timeout_s default.

    Returns:
        One page of entries, the cursor of the next page, the resolved path and whether the listing was truncated.

    Raises:
        ValueError: On a bad path, a depth outside 1..max_depth, an unknown container, a missing path, a file
            path, or a forged cursor.
        RuntimeError: If the container is stopped without auto_start, the start overruns, or the listing overruns.
    """
    # 1. The path rule and the depth bounds, before any adapter call (T-04-37). A bool is not a depth.
    validate_container_path(path)
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= max_depth:
        raise ValueError(
            f"Invalid depth {depth!r}: give an integer from 1 to {max_depth}, the largest depth the server's "
            "files.max_list_depth allows."
        )
    # 2. The container must run, or be started when the agent asked for it.
    await ensure_running(adapter, name, auto_start=auto_start, timeout_s=start_timeout_s)
    # 3. One bounded listing: the port stops at max_entries and reports truncation.
    listing = await adapter.list_directory(name, path, depth, max_entries, op_timeout_s)
    # 4-5. Map, sort by relative path (stable paging on an unchanged tree), then page.
    entries = sorted((_dir_entry(record) for record in listing.entries), key=lambda entry: entry.path)
    items, next_cursor = page(entries, cursor, page_size)
    # 6. The resolved path and the port's truncation flag.
    return DirectoryPage(path=listing.path, items=items, next_cursor=next_cursor, truncated=listing.truncated)
