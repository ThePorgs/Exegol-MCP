"""MCP tool handler for server_info (D-13). Thin wrapper: data source from the lifespan, logic in services."""

from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.server_info import ServerInfo
from exegol_mcp.services import server_info as server_info_service
from exegol_mcp.tools.errors import tool_errors


async def server_info(ctx: Context[ExegolPort, Any]) -> ServerInfo:
    """Report the exegol-mcp version, the Exegol library and wrapper versions, Docker status and update notice.

    IMPORTANT: Call this to diagnose the setup or when a tool fails unexpectedly. A library/wrapper version
    mismatch is information only and does not stop the server from working. When update_notice is set, tell the
    user a newer exegol-mcp release is available. This tool only reads state.

    RELATED TOOLS:
    - list_containers: check the containers once the server is known to be healthy
    - list_images: check the installed images

    Returns:
        Versions, Docker status, configuration directories, compatibility state and update notice.
    """
    source = server_info_service.source_from(ctx.request_context.lifespan_context)
    async with tool_errors(ctx.request_context.lifespan_context):
        return await server_info_service.server_info(source)
