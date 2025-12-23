from typing import List

from mcp.server.fastmcp import Context

from exegol_mcp.models.image import ImageInfo
from exegol_mcp.utils.exegol_utils import get_exegol_container, check_exegol_readiness, get_container_by_name, \
    list_images, download_exegol_image
from exegol_mcp.mcp_app import mcp_server
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.utils.mocking.TUI_MCP import TUI_MCP


@mcp_server.tool()
async def list_exegol_containers(ctx: Context) -> List[ContainerInfo]:
    """List all available Exegol containers with their status.
        IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

        RELATED TOOLS:
        - start_container/stop_container: To manage specific containers after listing
        - execute_command_in_container: To execute commands in listed containers
        - list_installed_tools: To see tools available in specific containers
        - list_installed_images: To check what images are available for containers

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
    IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

    RELATED TOOLS:
    - list_exegol_containers: To check container status before starting
    - execute_command_in_container: To execute commands in the started container
    - list_installed_tools: To discover tools available in the container
    - stop_container: To stop the container when done

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
    IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

    RELATED TOOLS:
    - list_exegol_containers: To check container status before stopping
    - start_container: To restart the container later if needed
    - execute_command_in_container: Use before stopping to save work/cleanup

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
        IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

        RELATED TOOLS:
        - list_all_images: To see both installed and available images
        - download_image: To install/update images from this list
        - list_exegol_containers: To see which images are used by containers

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


@mcp_server.tool()
async def list_all_images(ctx: Context) -> List[ImageInfo]:
    """List all Exegol images with their status.
        IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

        RELATED TOOLS:
        - list_installed_images: To see only installed images (faster)
        - download_image: To download and install images from this list
        - list_exegol_containers: To understand current container setup

        Returns a list of Exegol images installed on the system and available to download
        with their detailed information (name, version, up-to-date status, etc...).
        Returns:
            List of Exegol images with their metadata
        Raises:
            RuntimeError: If Exegol is not ready or configured
    """
    await ctx.info("Starting action: Listing Exegol images")
    return await list_images()


@mcp_server.tool()
async def download_image(ctx: Context, image_name: str, image_version: str = "latest") -> bool:
    """Download and install an Exegol image, this can be a new image or
        update an already installed but outdated one.
        IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

        RELATED TOOLS:
        - list_all_images: To discover available images before downloading
        - list_installed_images: To check current installation status
        - list_exegol_containers: To see how images are used in practice

        Args:
            image_name: Name of the image to download
            image_version: Version of the image to download. All Pro images can download a specific legacy version except nightly
        Returns:
            return true if the image is installed
        Raises:
            RuntimeError: If Exegol is not ready or configured
    """
    await ctx.info("Starting action: Downloading Exegol image")
    await check_exegol_readiness(ctx)
    if image_name in ["nightly", "free"] and image_version != "latest":
        raise ValueError(f"{image_name} image can only be downloaded with version 'latest'")

    TUI_MCP.DOWNLOAD_CONTEXT = ctx
    try:
        result = await download_exegol_image(image_name, image_version)
    finally:
        TUI_MCP.DOWNLOAD_CONTEXT = None

    return result
