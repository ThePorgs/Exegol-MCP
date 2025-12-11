from mcp import ServerSession
from mcp.server.fastmcp import Context

from src.exegol_utils import get_container_by_name
from src.mcp_app import mcp_server
from src.models.container import ExecutionResult


@mcp_server.tool()
async def start_container(
        container_name: str,
        ctx: Context[ServerSession, None]
) -> str:
    """
    Start an Exegol container if it is not running yet.
    Args:
        container_name: Name of the container
    Returns:
        Command output (stdout + stderr)
    """
    await ctx.info(f"Starting container {container_name}")

    # 1. Check if container exists
    container = await get_container_by_name(container_name)
    if not container:
        raise ValueError(f"Container '{container_name}' not found")

    if not container.isRunning():
        await container.start()

    if container.isRunning():
        return "Container started successfully"
    else:
        return "Container failed to start"

@mcp_server.tool()
async def execute_command_in_container(
        container_name: str,
        command: str,
        ctx: Context[ServerSession, None]
) -> ExecutionResult:
    """
    Execute a command in an Exegol container.
    Args:
        container_name: Name of the container
        command: Command to execute
    Returns:
        Command output (stdout + stderr)
    """
    await ctx.info(f"Executing command '{command}' in container {container_name}")

    # 1. Check if container exists
    container = await get_container_by_name(container_name)
    if not container:
        raise ValueError(f"Container '{container_name}' not found")

    # 2. Check if container is running
    if not container.isRunning():
        # TODO add elicit when supported by clients
        raise RuntimeError(f"Container '{container_name}' is not running")

    # 3. Execute the command
    exit_code, result = await container.exec_raw(command)

    await ctx.info(f"Command executed successfully")
    return ExecutionResult(exit_code=exit_code, output=result)
