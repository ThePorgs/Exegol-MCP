"""MCP tool handlers for container files. Thin wrappers: adapter from the lifespan, rules in the file service."""

from typing import Any, Literal

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.config import ExecutionLimits, FilesSettings
from exegol_mcp.models.files import DirectoryPage, FileContent, FileWritten
from exegol_mcp.services import files as files_service
from exegol_mcp.tools.containers import _container_op_timeout
from exegol_mcp.tools.errors import tool_errors


def _files_settings(adapter: object) -> FilesSettings:
    config = getattr(adapter, "config", None)
    return config.files if config is not None else FilesSettings()


def _execution_limits(adapter: object) -> ExecutionLimits:
    config = getattr(adapter, "config", None)
    return config.execution if config is not None else ExecutionLimits()


def _op_timeout(adapter: object) -> float:
    """The bound of each in-container command of a transfer: the operator's default exec timeout."""
    return float(_execution_limits(adapter).timeout_default_s)


async def read_file(
    container_name: str,
    path: str,
    encoding: Literal["auto", "base64"] = "auto",
    auto_start: bool = False,
    *,
    ctx: Context[ExegolPort, Any],
) -> FileContent:
    """Read a whole file from an Exegol container and return its content as UTF-8 text or base64.

    IMPORTANT: Any absolute path in the container works, including /proc, /sys, /dev and tmpfs mounts. A file that
    is valid UTF-8 comes back as text with encoding "utf-8"; anything else (binary data, invalid UTF-8) comes back
    as base64 with encoding "base64". Pass encoding="base64" to always get the exact bytes as base64. The server's
    file size limit applies: a file larger than it is an error stating its size and the limit, and is never cut to
    fit. Move big files through a host-mounted workspace or with exec_command (for example head, tail, split or
    gzip) instead. Content is returned as the file holds it, unredacted (credentials in a file included). The
    container must be running: start it with start_container first, or pass auto_start=true to have this tool start
    a stopped container before the read.

    RELATED TOOLS:
    - write_file: write a file into the container
    - list_directory: see which files a container directory holds before reading one
    - exec_command: read part of a file (head, tail, grep) or handle files over the size limit
    - list_containers: check the container name and status
    - start_container: start the container before reading from it

    Args:
        container_name: Name of the container, as returned by list_containers.
        path: Absolute path of the file in the container (starts with '/').
        encoding: "auto" (default) returns UTF-8 text as text and anything else as base64; "base64" always returns
            base64.
        auto_start: Start the container first if it is stopped (default false). A running container is left as
            is; an unknown one is still an error.

    Returns:
        The resolved path, the encoding ("utf-8" or "base64"), the content in that encoding and the file size in
        raw bytes.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await files_service.read_file(
            adapter,
            container_name,
            path,
            encoding=encoding,
            max_bytes=_files_settings(adapter).max_bytes,
            op_timeout_s=_op_timeout(adapter),
            auto_start=auto_start,
            start_timeout_s=_container_op_timeout(adapter),
        )


async def write_file(
    container_name: str,
    path: str,
    content: str,
    encoding: Literal["utf-8", "base64"] = "utf-8",
    overwrite: bool = True,
    mode: str | None = None,
    auto_start: bool = False,
    *,
    ctx: Context[ExegolPort, Any],
) -> FileWritten:
    """Write a file into an Exegol container from UTF-8 text or base64.

    IMPORTANT: The write creates the file or overwrites an existing one, and creates missing parent directories
    (mode 0755). Give mode as an octal string such as "0755" (for a script) or "0644"; without it a new file gets
    0644 and an existing file keeps its owner and mode. The write runs as the container's root, at any absolute
    path, and goes through a symlink to its target. Pass overwrite=false to refuse a file that already exists. For
    binary data, pass the bytes base64-encoded with encoding="base64". The server's file size limit applies to the
    decoded bytes: a larger payload is an error stating its size and the limit; move big files through a
    host-mounted workspace or with exec_command instead. Two concurrent writes to the same path are not atomic: the
    last one wins. The content is never logged by the server. The container must be running: start it with
    start_container first, or pass auto_start=true to have this tool start a stopped container before the write.

    RELATED TOOLS:
    - read_file: read the file back, or read any other container file
    - list_directory: see which files a container directory holds
    - exec_command: run the written file, or move files over the size limit
    - list_containers: check the container name and status
    - start_container: start the container before writing to it

    Args:
        container_name: Name of the container, as returned by list_containers.
        path: Absolute path of the file in the container (starts with '/'). It must not be a directory.
        content: The file content: text for encoding "utf-8", standard padded base64 for encoding "base64".
        encoding: "utf-8" (default) writes content as UTF-8 text; "base64" decodes content to the exact bytes.
        overwrite: Replace an existing file (default true); false refuses one.
        mode: Permission bits as an octal string such as "0755"; omit it to keep the default or existing mode.
        auto_start: Start the container first if it is stopped (default false). A running container is left as
            is; an unknown one is still an error.

    Returns:
        The path the bytes landed at (a symlink's target), the number of raw bytes written and whether the file
        was created (false when an existing file was replaced).
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await files_service.write_file(
            adapter,
            container_name,
            path,
            content,
            encoding=encoding,
            overwrite=overwrite,
            mode=mode,
            max_bytes=_files_settings(adapter).max_bytes,
            op_timeout_s=_op_timeout(adapter),
            auto_start=auto_start,
            start_timeout_s=_container_op_timeout(adapter),
        )


async def list_directory(
    container_name: str,
    path: str,
    depth: int = 1,
    cursor: str | None = None,
    auto_start: bool = False,
    *,
    ctx: Context[ExegolPort, Any],
) -> DirectoryPage:
    """List a directory in an Exegol container: each entry's type, size, mode, modification time and symlink target.

    IMPORTANT: The listing is one level by default; pass depth to go deeper, up to the server's depth limit. Each
    entry has its name, its path relative to the listed directory, its type (file, dir, symlink or other), size,
    octal mode, modification time (ISO-8601 UTC) and, for a symlink, its target. Symlinks are reported with their
    target and never followed: a symlink to a directory is not descended into. Results are sorted by relative path
    and paged: when next_cursor is not null, pass it back as cursor to get the next page. truncated=true means the
    tree had more entries than the server's listing cap, so only the first ones are listed: narrow the path or lower
    depth. Names that are not valid UTF-8 show replacement characters. Listing a file is an error (use read_file).
    The container must be running: start it with start_container first, or pass auto_start=true to have this tool
    start a stopped container before the listing.

    RELATED TOOLS:
    - read_file: read one of the listed files
    - write_file: write a file into the container
    - exec_command: search the filesystem (find, grep) or list with other options
    - list_containers: check the container name and status
    - start_container: start the container before listing it

    Args:
        container_name: Name of the container, as returned by list_containers.
        path: Absolute path of the directory in the container (starts with '/').
        depth: How many levels to list (default 1: the directory's direct entries). It goes from 1 to the depth
            limit the operator set in the server config (5 by default); anything else is an error.
        cursor: The next_cursor of a previous page, to fetch the next page; omit it for the first page.
        auto_start: Start the container first if it is stopped (default false). A running container is left as
            is; an unknown one is still an error.

    Returns:
        The resolved directory path, the entries on this page, the cursor of the next page (null on the last page)
        and whether the listing was truncated at the server's entry cap.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        settings = _files_settings(adapter)
        limits = _execution_limits(adapter)
        return await files_service.list_directory(
            adapter,
            container_name,
            path,
            depth=depth,
            cursor=cursor,
            page_size=limits.listing_page_size,
            max_depth=settings.max_list_depth,
            max_entries=settings.max_list_entries,
            op_timeout_s=float(limits.timeout_default_s),
            auto_start=auto_start,
            start_timeout_s=_container_op_timeout(adapter),
        )
