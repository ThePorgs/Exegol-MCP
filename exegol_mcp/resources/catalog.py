"""MCP resource template for the tools installed in one Exegol container.

Thin: adapter from the lifespan, data from the catalog service.
"""

import json
from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.resources.containers import require_running_for_read
from exegol_mcp.services import catalog as catalog_service
from exegol_mcp.tools.errors import resource_errors


async def installed_tools(name: str, *, ctx: Context[ExegolPort, Any]) -> str:
    """The whole catalog of tools installed in one Exegol container, as JSON, in a single read.

    The JSON object is {"tools": [...]}: every tool listed in the container's /.exegol/installed_tools.csv, in file
    order, each with its name, version, category and description (null when the catalog does not record it). The
    list is never paged and never filtered. An empty catalog reads as {"tools": []}.

    IMPORTANT: The container must be running. Reading this resource never starts, stops or changes a container: a
    stopped container is an error that says to start it with start_container first. The name is the exact,
    case-sensitive container name.

    RELATED TOOLS:
    - list_installed_tools: the same catalog, one page at a time
    - get_tool_help: read the help of one of the listed tools
    - exec_command: run one of the listed tools in the container
    - start_container: start a stopped container before reading this resource
    """
    adapter = ctx.request_context.lifespan_context
    async with resource_errors(adapter):
        await require_running_for_read(adapter, name)
        tools = await catalog_service.read_installed_tools(adapter, name)
        # An object rather than a bare array, so fields can be added later without breaking readers.
        return json.dumps({"tools": [tool.model_dump(mode="json") for tool in tools]})
