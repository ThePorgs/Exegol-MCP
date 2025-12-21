import pytest

# Importing assets registers all tools on the FastMCP server via decorators
import exegol_mcp.assets  # noqa: F401
from exegol_mcp.mcp_app import mcp_server


EXPECTED_TOOLS = [
    "start_container",
    "stop_container",
    "execute_command_in_container",
    "list_exegol_containers",
    "list_installed_images",
    "list_all_images",
    "download_image",
    "list_installed_tools",
    "get_tool_help",
    "execute_remote_command"
]


@pytest.mark.anyio
async def test_mcp_server_exposes_expected_tools():
    """Ensure the FastMCP server has all expected tools registered.

    This is a lightweight unit test that checks the server-side registry directly
    (no networking). It validates that tool decorators correctly exposed the tools
    on the FastMCP instance, which is the core contract for MCP servers.
    """

    tools = await mcp_server.list_tools()
    tool_names = {tool.name for tool in tools}

    missing = [name for name in EXPECTED_TOOLS if name not in tool_names]
    assert not missing, f"Missing expected tools: {missing}. Available: {sorted(tool_names)}"

