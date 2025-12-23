from mcp.server.fastmcp import Context

from exegol_mcp.utils.exegol_utils import get_container_by_name, check_exegol_readiness
from exegol_mcp.mcp_app import mcp_server
from exegol_mcp.models.container import ExecutionResult
from exegol_mcp.models.elicit_forms import UserConfirmation
from exegol_mcp.utils.client_checks import is_elicitation_form_supported


@mcp_server.tool()
async def execute_command_in_container(
        container_name: str,
        command: str,
        ctx: Context
) -> ExecutionResult:
    """
    Execute a command in an Exegol container.
    IMPORTANT: Always check if there isn't a more specific tool created for this task (like execute_remote_command for remote connections) before using this tool.

    RELATED TOOLS:
    - execute_remote_command: For remote network connections (SSH, WinRM, SMB, etc.)
    - list_installed_tools: To discover available tools before executing commands
    - get_tool_help: To get help documentation for specific tools
    - list_exegol_containers: To check container status before execution
    - start_container/stop_container: To manage container lifecycle

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
        if is_elicitation_form_supported(ctx):
            # TODO to test in a client that supports elicitation
            result = await ctx.elicit(
                message=f"Container '{container_name}' is not running. Do you want to start it?",
                schema=UserConfirmation)
            if result.action == "accept" and result.data:
                if result.data.confirmation:
                    await container.start()
            else:
                error_msg = f"Container '{container_name}' is not running and user declined to start it"
                await ctx.error(error_msg)
                raise RuntimeError(error_msg)
            if not container.isRunning():
                raise RuntimeError(f"Failed to start container '{container_name}'")
        else:
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
