from mcp.server.fastmcp import Context

from src.exegol_utils import get_container_by_name, check_exegol_readiness
from src.mcp_app import mcp_server
from src.models.container import ExecutionResult


@mcp_server.tool()
async def execute_command_in_container(
        container_name: str,
        command: str,
        ctx: Context
) -> ExecutionResult:
    """
    Execute a command in an Exegol container.
    Args:
        container_name: Name of the container
        command: Command to execute
    Returns:
        exit_code: Exit code of the command
        output: Command output (stdout + stderr)
    """
    await ctx.info(f"Executing command '{command}' in container {container_name}")
    await check_exegol_readiness(ctx)

    # 1. Check if container exists
    container = get_container_by_name(container_name)
    if not container:
        await ctx.error(f"Container '{container_name}' not found")
        raise ValueError(f"Container '{container_name}' not found")

    # 2. Check if container is running
    if not container.isRunning():
        error_msg = (
            f"Container '{container_name}' is not running. "
            f"Please start it first using the 'start_container' tool."
        )
        await ctx.error(error_msg)
        raise RuntimeError(error_msg)

    # 3. Execute the command
    exit_code, result = await container.exec_raw(command)

    if exit_code == 0:
        await ctx.info(f"Command executed successfully")
    else:
        await ctx.warning(f"Command execution failed with exit code {exit_code}")
    return ExecutionResult(exit_code=exit_code, output=result)
