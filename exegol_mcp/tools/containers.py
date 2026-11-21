"""MCP tool handlers for Exegol containers. Thin wrappers: adapter from the lifespan, logic in services."""

from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.services import containers as containers_service
from exegol_mcp.tools.errors import tool_errors


def _container_op_timeout(adapter: ExegolPort) -> float:
    """Read the operator-configured container start/stop timeout from the lifespan context (D-18).

    The lifespan context carries the resolved OperatorConfig as .config; an ad-hoc context without one falls
    back to the ExecutionLimits default (50 s). This is the dedicated container_op_timeout_s key, never the
    exec timeout keys.
    """
    config = getattr(adapter, "config", None)
    if config is None:
        return ExecutionLimits().container_op_timeout_s
    return float(config.execution.container_op_timeout_s)


async def list_containers(managed_only: bool = False, *, ctx: Context[ExegolPort, Any]) -> list[ContainerInfo]:
    """List all Exegol containers on this host with their status and configuration.

    IMPORTANT: Call this first to discover container names before acting on a container. It only reads
    state and never starts, stops or changes a container. Each entry reports managed (true when exegol-mcp
    created the container), run_id (the server run that created it) and this_run (true when this server process
    created it). Containers from earlier or crashed server runs still show as managed, with their own run_id and
    this_run false. Only managed containers can be removed or cleaned up by the lifecycle tools, unless the
    operator allows otherwise. Pass managed_only=true to list only exegol-mcp's containers.

    RELATED TOOLS:
    - create_container: create a new container, which then lists here as managed and this_run
    - start_container / stop_container: change the state of a container listed here
    - exec_command: run a shell command inside a running container listed here
    - list_installed_tools: see which tools are available inside a listed container
    - list_images: see which Exegol images containers can be created from

    Args:
        managed_only: List only the containers exegol-mcp created (from this or an earlier run).

    Returns:
        One entry per container (name, status, image name and version, network, features, devices,
        VPN, environment, privileges, capabilities, comment, managed, run_id, this_run). An empty list means no
        container matches.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await containers_service.list_containers(
            adapter, run_id=getattr(adapter, "run_id", None), managed_only=managed_only
        )


async def start_container(container_name: str, ctx: Context[ExegolPort, Any]) -> bool:
    """Start an Exegol container if it is not running yet.

    IMPORTANT: Starting a container that already runs is a no-op that returns true, so it is safe to call before
    every use. A container must be running before exec_command can run anything in it. The server gives a start
    at most the operator's configured container start/stop timeout; if it has not finished by then the call
    returns an error and the container may still be starting, so check list_containers.

    RELATED TOOLS:
    - list_containers: check the container name and status before starting it
    - exec_command: run a shell command in the started container
    - list_installed_tools: see which tools are available in the started container
    - stop_container: stop the container when done

    Args:
        container_name: Name of the container, as returned by list_containers.

    Returns:
        true once the container is running.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await containers_service.start_container(
            adapter, container_name, timeout_s=_container_op_timeout(adapter)
        )


async def stop_container(container_name: str, ctx: Context[ExegolPort, Any]) -> bool:
    """Stop a running Exegol container.

    IMPORTANT: Stopping ends every process running in the container, including running commands and shells.
    Stopping a container that is already stopped is a no-op that returns true. The server gives a stop at most
    the operator's configured container start/stop timeout; if it has not finished by then the call returns an
    error and the container may still be stopping, so check list_containers.

    RELATED TOOLS:
    - list_containers: check the container name and status before stopping it
    - start_container: start the container again later
    - exec_command: save or clean up work in the container before stopping it

    Args:
        container_name: Name of the container, as returned by list_containers.

    Returns:
        true once the container is stopped.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await containers_service.stop_container(
            adapter, container_name, timeout_s=_container_op_timeout(adapter)
        )
