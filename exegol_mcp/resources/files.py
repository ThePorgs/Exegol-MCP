"""MCP resource template for one file under an Exegol container's /workspace.

Thin: adapter from the lifespan, the read itself from the file service, so the size limit and the UTF-8 detection are
exactly read_file's.
"""

import base64
import posixpath
from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.resources.containers import require_running_for_read
from exegol_mcp.services import files as files_service
from exegol_mcp.tools.errors import resource_errors
from exegol_mcp.tools.files import _files_settings, _op_timeout

_WORKSPACE_ROOT = "/workspace"


async def workspace_file(name: str, path: str, *, ctx: Context[ExegolPort, Any]) -> str | bytes:
    """One file under the /workspace directory of an Exegol container, read whole through the container.

    The path after workspace/ is relative to /workspace and reaches /workspace only. A file that is valid UTF-8
    comes back as text (text/plain); anything else comes back as binary (application/octet-stream) holding the exact
    bytes. The server's file size limit (files.max_bytes) applies: a larger file is an error, never cut to fit.
    Content is returned as the file holds it, unredacted. Workspace files are not listed one by one: find their
    names with list_directory.

    IMPORTANT: The container must be running. Reading this resource never starts, stops or changes a container: a
    stopped container is an error that says to start it with start_container first. The name is the exact,
    case-sensitive container name.

    RELATED TOOLS:
    - read_file: read any path in the container, force base64, or start a stopped container first (auto_start)
    - list_directory: find the names of the files under /workspace, which are not listed as resources
    - write_file: write a file into the container
    - start_container: start a stopped container before reading this resource
    """
    adapter = ctx.request_context.lifespan_context
    async with resource_errors(adapter):
        # 1. Lexical containment, behind the SDK's own parameter checks (traversal, absolute path, NUL), which stay
        # at their defaults: the normalized path must be /workspace or lie under it.
        full = posixpath.normpath(posixpath.join(_WORKSPACE_ROOT, path))
        if full != _WORKSPACE_ROOT and not full.startswith(_WORKSPACE_ROOT + "/"):
            raise ValueError(
                f"The workspace resource reaches {_WORKSPACE_ROOT} only, and '{path}' leaves it. Use the read_file "
                "tool to read other paths in the container."
            )
        # 2. A resource read never starts the container.
        await require_running_for_read(adapter, name)
        # 3. The read runs inside the container, never through the host bind mount, so a symlink planted under
        # /workspace resolves in the container's own filesystem. files.max_bytes and strict UTF-8 apply unchanged.
        content = await files_service.read_file(
            adapter,
            name,
            full,
            encoding="auto",
            max_bytes=_files_settings(adapter).max_bytes,
            op_timeout_s=_op_timeout(adapter),
            auto_start=False,
        )
        # 4. Text stays text; anything else goes back as bytes, which the SDK sends as a blob.
        if content.encoding == "utf-8":
            return content.content
        return base64.b64decode(content.content)
