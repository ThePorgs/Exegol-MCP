import csv
import io
from typing import List

from mcp.server.fastmcp import Context

from exegol_mcp.utils.exegol_utils import get_container_by_name, check_exegol_readiness, read_installed_resources
from exegol_mcp.mcp_app import mcp_server
from exegol_mcp.models.container import InstalledTool, ExecutionResult


async def _read_installed_tools_csv(container_name: str, ctx: Context) -> str:
    """
    Helper function to read the installed_tools.csv file from an Exegol container.
    Args:
        container_name: Name of the Exegol container
        ctx: MCP context
    Returns:
        Content of the installed_tools.csv file
    """
    container = get_container_by_name(container_name)
    if not container:
        await ctx.error(f"Container '{container_name}' not found")
        raise ValueError(f"Container '{container_name}' not found")

    # Check if container is running
    if not container.isRunning():
        await ctx.error(f"Container '{container_name}' is not running")
        raise RuntimeError(f"Container '{container_name}' is not running")

    # Read the installed_tools.csv file from the container
    # The file is located at /.exegol/installed_tools.csv in each container
    exit_code, output = await container.exec_raw("cat /.exegol/installed_tools.csv")

    if exit_code != 0:
        await ctx.error(f"Failed to read installed_tools.csv from container '{container_name}'")
        raise RuntimeError(f"Failed to read installed_tools.csv: {output}")

    return output


@mcp_server.tool()
async def list_installed_tools(
        container_name: str,
        ctx: Context
) -> List[InstalledTool]:
    """
    List all installed tools in an Exegol container by reading the installed_tools.csv file.
    IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

    RELATED TOOLS:
    - get_tool_help: To get detailed help for specific tools from this list
    - execute_command_in_container: To actually use the tools listed here
    - list_exegol_containers: To choose which container to explore
    - start_container: To ensure container is running before listing tools

    Args:
        container_name: Name of the Exegol container
    Returns:
        List of installed tools with their information (name, category, description)
    """
    await ctx.info(f"Listing installed tools in container {container_name}")
    await check_exegol_readiness(ctx)

    # Read the CSV file directly
    csv_content = await _read_installed_tools_csv(container_name, ctx)

    # Parse the CSV content
    tools = []
    csv_reader = csv.DictReader(io.StringIO(csv_content))

    for row in csv_reader:
        tool = InstalledTool(
            name=row.get('name', row.get('Tool', row.get('tool', ''))).strip(),
            category=row.get('category', row.get('Category', '')).strip() or None,
            description=row.get('description', row.get('Description', '')).strip() or None
        )
        tools.append(tool)

    await ctx.info(f"Found {len(tools)} installed tools")
    return tools


@mcp_server.tool()
async def get_tool_help(
        container_name: str,
        tool_name: str,
        ctx: Context
) -> ExecutionResult:
    """
    Get help/documentation for a specific tool installed in an Exegol container.
    Tries common help flags: -h, --help, help
    IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

    RELATED TOOLS:
    - list_installed_tools: To discover available tools before getting help
    - execute_command_in_container: To test tools after reading their help
    - list_exegol_containers: To choose which container to get help from

    Args:
        container_name: Name of the Exegol container
        tool_name: Name of the tool to get help for
    Returns:
        ExecutionResult with the help output
    """
    await ctx.info(f"Getting help for tool '{tool_name}' in container {container_name}")
    await check_exegol_readiness(ctx)

    container = get_container_by_name(container_name)
    if not container:
        await ctx.error(f"Container '{container_name}' not found")
        raise ValueError(f"Container '{container_name}' not found")

    if not container.isRunning():
        await ctx.error(f"Container '{container_name}' is not running")
        raise RuntimeError(f"Container '{container_name}' is not running")

    # Try different help flags
    help_flags = ["--help", "-h", "help"]
    last_exit_code = None
    last_error = None

    for flag in help_flags:
        command = f"{tool_name} {flag}"
        exit_code, output = await container.exec_raw(command)

        if exit_code == 0:
            await ctx.info(f"Successfully retrieved help using '{flag}' flag")
            return ExecutionResult(exit_code=exit_code, output=output)

        # If exit code is not 0, try next flag
        last_exit_code = exit_code
        last_error = output

    # If all flags failed, return the last error
    await ctx.warning(f"Could not get help for '{tool_name}' using standard flags. Last attempt output:")
    return ExecutionResult(exit_code=last_exit_code or 1, output=last_error or "Tool help not available")


@mcp_server.tool()
async def list_installed_exegol_resources(
        ctx: Context,
        container_name: str,
        target_os: str = "linux"
) -> List[str]:
    """
    Exegol's "offline resources" are a neat choice of standalone tools and scripts that are often used during penetration tests,
    CTFs and red-teams. While many penetration testers download those resources again every time they need them, Exegol users don't have to.
    This tool lists all installed resources in an Exegol container by listing files in /opt/resources/<target_os>.

    Args:
        container_name: Name of the Exegol container
        target_os: OS type of the target
    Returns:
        List of full path tools and script present in the container
    """
    await ctx.info(f"Listing installed exegol resources in container {container_name}")
    await check_exegol_readiness(ctx)
    resources = await read_installed_resources(ctx, container_name, target_os)
    return resources
