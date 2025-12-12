from mcp.server.fastmcp import Context

from src.exegol_utils import get_container_by_name, check_exegol_readiness
from src.mcp_app import mcp_server
from src.models.container import ExecutionResult


@mcp_server.tool()
async def start_container(
        container_name: str,
        ctx: Context
) -> bool:
    """
    Start an Exegol container if it is not running yet.
    Args:
        container_name: Name of the container
    Returns:
        return true if the container is running
    """
    await ctx.info(f"Starting container {container_name}")
    await check_exegol_readiness(ctx)

    # 1. Check if container exists
    container = get_container_by_name(container_name)
    if not container:
        await ctx.error(f"Container '{container_name}' not found")
        raise ValueError(f"Container '{container_name}' not found")

    if not container.isRunning():
        await container.start()

    return container.isRunning()

@mcp_server.tool()
async def stop_container(
        container_name: str,
        ctx: Context
) -> bool:
    """
    Stop an Exegol container if it is running.
    Args:
        container_name: Name of the container
    Returns:
        return true if the container is stopped
    """
    await ctx.info(f"Stopping container {container_name}")
    await check_exegol_readiness(ctx)

    # 1. Check if container exists
    container = get_container_by_name(container_name)
    if not container:
        await ctx.error(f"Container '{container_name}' not found")
        raise ValueError(f"Container '{container_name}' not found")

    if container.isRunning():
        await container.stop()

    return not container.isRunning()

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
        # Use elicitation to ask user if they want to start the container
        try:
            response = await ctx.elicit(
                mode="form",
                message=f"Container '{container_name}' is not running. Would you like to start it now?",
                requested_schema={
                    "type": "object",
                    "properties": {
                        "start_container": {
                            "type": "boolean",
                            "title": "Start Container",
                            "description": f"Start the container '{container_name}' before executing the command",
                            "default": True
                        }
                    },
                    "required": ["start_container"]
                }
            )
            
            if response.get("start_container", False):
                await ctx.info(f"Starting container {container_name} as requested")
                await container.start()
                if not container.isRunning():
                    raise RuntimeError(f"Failed to start container '{container_name}'")
            else:
                error_msg = f"Container '{container_name}' is not running and user declined to start it"
                await ctx.error(error_msg)
                raise RuntimeError(error_msg)
        except AttributeError:
            # Fallback if elicitation is not supported by the client
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
