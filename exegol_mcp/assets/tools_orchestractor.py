from typing import List

from mcp.server.fastmcp import Context

from exegol_mcp.models.image import ImageInfo
from exegol_mcp.utils.exegol_utils import get_exegol_container, check_exegol_readiness, get_container_by_name, list_images
from exegol_mcp.mcp_app import mcp_server
from exegol_mcp.models.container import ContainerInfo


@mcp_server.tool()
async def list_exegol_containers(ctx: Context) -> List[ContainerInfo]:
    """List all available Exegol containers with their status.
        Returns a list of Exegol containers configured on the system
        with their detailed information (name, status, image, network_driver etc...) in a chart.
        Returns:
            List of Exegol containers with their metadata
        Raises:
            RuntimeError: If Exegol is not ready or configured
    """
    await ctx.info("Starting action: Listing Exegol containers")
    await check_exegol_readiness(ctx)
    return await get_exegol_container()


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
async def list_installed_images(ctx: Context) -> List[ImageInfo]:
    """List all installed Exegol images with their status.
        Returns a list of Exegol images installed on the system
        with their detailed information (name, version, up-to-date status, etc...) in a chart.
        Returns:
            List of Exegol images with their metadata
        Raises:
            RuntimeError: If Exegol is not ready or configured
    """
    await ctx.info("Starting action: Listing Exegol installed images")
    await check_exegol_readiness(ctx)
    return await list_images(installed_only=True)
