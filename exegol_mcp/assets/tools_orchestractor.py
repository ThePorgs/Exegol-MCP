from typing import List, Optional

from mcp.server.fastmcp import Context

from exegol_mcp.models.image import ImageInfo
from exegol_mcp.utils.exegol_utils import get_exegol_container, check_exegol_readiness, get_container_by_name, \
    list_images, download_exegol_image, create_exegol_container
from exegol_mcp.mcp_app import mcp_server
from exegol_mcp.models.container import ContainerInfo, ContainerCreationResult
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


@mcp_server.tool()
async def create_container(
        container_name: str,
        image_name: str,
        network_mode: str = "host",
        vpn_path: Optional[str] = None,
        vpn_auth: Optional[str] = None,
        enable_gui: bool = True,
        enable_desktop: bool = False,
        desktop_config: Optional[str] = None,
        share_timezone: bool = True,
        enable_my_resources: bool = True,
        enable_exegol_resources: bool = True,
        enable_shell_logging: bool = False,
        shell_logging_method: str = "asciinema",
        shell_logging_compress: Optional[bool] = None,
        privileged: bool = False,
        hostname: Optional[str] = None,
        shell: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        devices: Optional[list[str]] = None,
        envs: Optional[dict] = None,
        volumes: Optional[list[dict]] = None,
        ports: Optional[list[dict]] = None,
        workspace_path: Optional[str] = None,
        comment: Optional[str] = None,
        ctx: Context = None
) -> ContainerCreationResult:
    """
    Create a new Exegol container with the specified configuration.
    The container will be started automatically after creation.
    Covers all features available in `exegol start`.
    IMPORTANT: Always check if there isn't a more specific tool created for this task before using this tool.

    RELATED TOOLS:
    - list_all_images: To discover available images before creating a container
    - list_installed_images: To check which images are already installed
    - download_image: To install an image before creating a container
    - list_exegol_containers: To check existing containers before creation
    - start_container/stop_container: To manage the container after creation

    NETWORK MODES:
    - host: Share host network stack (default, recommended for pentest)
    - docker: Docker bridge network (isolated, with VPN support)
    - nat: NAT network with port forwarding
    - disabled: No network

    Args:
        container_name: Name for the new container (alphanumeric, hyphens, dots, underscores)
        image_name: Exegol image to use (e.g. "free", "ad", "web", "full", "osint", "light", "nightly")
        network_mode: Network mode (host, docker, nat, disabled)
        vpn_path: Host path to an OpenVPN config file (.ovpn)
        vpn_auth: Host path to VPN auth credentials file
        enable_gui: Enable console GUI with X11 + Wayland forwarding
        enable_desktop: Enable remote desktop access
        desktop_config: Desktop configuration string (e.g. "localhost:3389")
        share_timezone: Share host timezone with the container
        enable_my_resources: Mount /opt/my-resources in the container
        enable_exegol_resources: Mount /opt/resources (offline pentest resources)
        enable_shell_logging: Enable shell session recording
        shell_logging_method: Shell logging method: "asciinema" (default) or "script"
        shell_logging_compress: Compress shell logs (None = use default)
        privileged: Run in privileged mode (use with caution)
        hostname: Custom hostname for the container (default: container name)
        shell: Default shell to use (e.g. "bash", "zsh", "fish")
        capabilities: Additional Linux capabilities (e.g. ["NET_ADMIN", "SYS_PTRACE"])
        devices: Host devices to passthrough (e.g. ["/dev/ttyUSB0", "/dev/bus/usb"])
        envs: Extra environment variables as key-value pairs (e.g. {"MY_VAR": "value"})
        volumes: Extra volume mounts as list of {"host": "/path", "container": "/path", "read_only": false}
        ports: Port mappings as list of {"host": 8080, "container": 80, "protocol": "tcp"}
        workspace_path: Custom workspace path on host (default: ~/.exegol/workspaces/<name>)
        comment: Optional comment describing the container purpose
    Returns:
        ContainerCreationResult with name, image, status and workspace path
    """
    await ctx.info(f"Creating container '{container_name}' with image '{image_name}'")
    await check_exegol_readiness(ctx)

    # Check name is not already taken
    existing = get_container_by_name(container_name)
    if existing:
        raise ValueError(f"Container '{container_name}' already exists")

    container = await create_exegol_container(
        name=container_name,
        image_name=image_name,
        network_mode=network_mode,
        vpn_path=vpn_path,
        vpn_auth=vpn_auth,
        enable_gui=enable_gui,
        enable_desktop=enable_desktop,
        desktop_config=desktop_config,
        share_timezone=share_timezone,
        enable_my_resources=enable_my_resources,
        enable_exegol_resources=enable_exegol_resources,
        enable_shell_logging=enable_shell_logging,
        shell_logging_method=shell_logging_method,
        shell_logging_compress=shell_logging_compress,
        privileged=privileged,
        hostname=hostname,
        shell=shell,
        capabilities=capabilities,
        devices=devices,
        envs=envs,
        volumes=volumes,
        ports=ports,
        workspace_path=workspace_path,
        comment=comment,
    )

    import os
    workspace_path = os.path.join(
        os.path.expanduser("~"), ".exegol", "workspaces", container_name
    )

    await ctx.info(f"Container '{container_name}' created and started successfully")
    return ContainerCreationResult(
        name=container.name,
        image=f"{container.image.getName()} {container.image.getImageVersion()}",
        status=container.getRawStatus(),
        workspace_path=workspace_path,
    )
