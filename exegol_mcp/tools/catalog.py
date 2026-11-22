"""MCP tool handlers for the Exegol tool/resource catalog. Thin wrappers: adapter from the lifespan, logic in the service."""

from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.catalog import InstalledToolsPage, OfflineResourcesPage
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.services import catalog as catalog_service
from exegol_mcp.tools.errors import tool_errors


def _page_size(adapter: ExegolPort) -> int:
    """Read the operator-configured listing page size from the lifespan context, or the built-in default.

    The lifespan context (ServerContext in production, the FakeExegolAdapter in tests) carries the resolved
    OperatorConfig as .config; an ad-hoc context without one falls back to the ExecutionLimits default (200).
    """
    config = getattr(adapter, "config", None)
    if config is None:
        return ExecutionLimits().listing_page_size
    return int(config.execution.listing_page_size)


async def list_installed_tools(
    container_name: str, cursor: str | None = None, *, ctx: Context[ExegolPort, Any]
) -> InstalledToolsPage:
    """List one page of the tools installed in an Exegol container, read from its /.exegol/installed_tools.csv.

    IMPORTANT: Call this to discover what is available before running or asking for help on a tool. The
    container must be running. This tool only reads state. The catalog can be large, so the result is one
    page: when next_cursor is not null, call this tool again passing cursor=next_cursor to fetch the next page;
    a null next_cursor means this was the last page. The page size is set by the operator config.

    RELATED TOOLS:
    - get_tool_help: read the help of a specific tool from this list
    - exec_command: run one of the listed tools in the container
    - exec_remote_command: run a listed network tool against a remote host
    - list_containers: choose which container to explore

    Args:
        container_name: Name of the container, as returned by list_containers.
        cursor: The next_cursor from a previous call, to fetch the next page. Omit for the first page.

    Returns:
        A page: items is one page of installed tools, each with its name, version, link and description (null when
        the CSV does not record them), and next_cursor is the cursor for the next page or null on the last page. An
        empty items list with a null next_cursor means the catalog file is empty.
    """
    adapter = ctx.request_context.lifespan_context
    page_size = _page_size(adapter)
    async with tool_errors(adapter):
        return await catalog_service.list_installed_tools(
            adapter, container_name, cursor=cursor, page_size=page_size
        )


async def get_tool_help(container_name: str, tool_name: str, *, ctx: Context[ExegolPort, Any]) -> ExecResult:
    """Get the help text of a tool installed in an Exegol container, trying --help, -h then help.

    IMPORTANT: The container must be running. This runs the tool with a help flag, so it reuses the exec path,
    but it is read-only by intent. Use list_installed_tools first to find the tool name.

    RELATED TOOLS:
    - list_installed_tools: discover the available tools before getting help
    - exec_command: run the tool once you understand its usage
    - list_containers: choose which container to query

    Args:
        container_name: Name of the container, as returned by list_containers.
        tool_name: Name of the tool to get help for.

    Returns:
        The help output of the first flag that succeeds, or of the last attempt when none succeeds (its real
        exit code is preserved, never coerced).
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await catalog_service.get_tool_help(adapter, container_name, tool_name)


async def list_offline_resources(
    container_name: str, target_os: str = "linux", cursor: str | None = None, *, ctx: Context[ExegolPort, Any]
) -> OfflineResourcesPage:
    """List one page of Exegol's offline resources under /opt/resources/<target_os> in a container.

    Exegol's offline resources are a curated set of standalone tools and scripts often used during
    penetration tests, CTFs and red teams, bundled so they do not have to be downloaded each time.

    IMPORTANT: The container must be running. This tool only reads state. The listing can be large, so the
    result is one page: when next_cursor is not null, call this tool again passing cursor=next_cursor to fetch
    the next page; a null next_cursor means this was the last page. The page size is set by the operator config.

    RELATED TOOLS:
    - list_installed_tools: list the tools installed in the container's PATH
    - exec_command: use one of the listed resources in the container
    - list_containers: choose which container to explore

    Args:
        container_name: Name of the container, as returned by list_containers.
        target_os: OS subdirectory to list, "linux" (default) or "windows".
        cursor: The next_cursor from a previous call, to fetch the next page. Omit for the first page.

    Returns:
        A page: items is one page of offline-resource paths, and next_cursor is the cursor for the next page or
        null on the last page. An empty items list with a null next_cursor means none was found.
    """
    adapter = ctx.request_context.lifespan_context
    page_size = _page_size(adapter)
    async with tool_errors(adapter):
        return await catalog_service.list_offline_resources(
            adapter, container_name, target_os, cursor=cursor, page_size=page_size
        )
