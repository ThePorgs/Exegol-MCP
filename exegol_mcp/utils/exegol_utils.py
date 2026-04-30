import logging
import os
from typing import List, Optional

from exegol.exceptions.ExegolExceptions import ObjectNotFound
from exegol.model.ExegolContainer import ExegolContainer
from exegol.model.ExegolImage import ExegolImage
from exegol.utils.DockerUtils import DockerUtils
from mcp.server.fastmcp import Context

from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.image import ImageInfo


async def check_exegol_readiness(ctx: Optional[Context] = None) -> bool:
    """
    Check if Exegol is ready and DockerUtils can be used.
    Verifies that Docker/Exegol is available by attempting to instantiate DockerUtils.
    Raises RuntimeError if Exegol is not ready.
    
    Args:
        ctx: MCP context for logging
    Returns:
        True if Exegol is ready
    Raises:
        RuntimeError: If Exegol/Docker is not available or not properly configured
    """
    try:
        # Try to instantiate DockerUtils to verify Exegol/Docker is available
        # This will fail if Docker is not running, not installed, or if Exegol is not properly configured
        _ = DockerUtils()  # Instantiating to verify availability
        # If instantiation succeeds, Exegol is ready
        if ctx:
            await ctx.info("Exegol readiness check passed")
        else:
            logging.info("Exegol readiness check passed")
        return True
    except Exception as e:
        # If any exception occurs, Exegol is not ready
        # Common causes: Docker not running, Docker not installed, Exegol not configured
        error_msg = (
            f"Exegol is not ready yet. Please run exegol first with `exegol info` "
            f"and make sure it works. Error: {str(e)}"
        )
        if ctx:
            await ctx.error(error_msg)
            raise RuntimeError(error_msg) from e
        else:
            logging.error(error_msg)
            exit(1)

async def get_exegol_container() -> List[ContainerInfo]:
    DockerUtils().clearCache()
    containers: List[ExegolContainer] = await DockerUtils().listContainers()
    results: List[ContainerInfo] = []
    for container in containers:
        network_drivers, network_name = container.config.getNetwork()
        features: List[str] = []
        if container.config.isGUIEnable():
            features.append("Console GUI")
        if container.config.isDesktopEnabled():
            features.append("Remote-Desktop")
        if container.config.isExegolResourcesEnable():
            features.append("Exegol-Resources")
        if container.config.isMyResourcesEnable():
            features.append("My-Resources")
        if container.config.isShellLoggingEnable():
            features.append("shell-logging")
        if container.config.isTimezoneShared():
            features.append("Shared-timezone")
        if container.config.isWrapperStartShared():
            features.append("Exegol-Resources")
        vpn = None
        if container.config.getVpnConfigPath():
            vpn = container.config.getVpnConfigPath().name
        results.append(ContainerInfo(
            name=container.name,
            creation_date=container.config.getCreationDate(),
            image_name=container.image.getName(),
            image_version=container.image.getImageVersion(),
            status=container.getRawStatus(),
            network_driver=network_drivers,
            network_name=network_name,
            comment=container.config.getComment(),
            features=features,
            vpn=vpn,
            env=[e for e in container.config.getTextEnvs().split(os.linesep) if e],
            devices=container.config.getDevices(),
            is_privileged=container.config.getPrivileged(),
            capabilities=["Docker default capabilities"] + container.config.getCapabilities()
        ))
    return results

async def create_exegol_container(
        name: str,
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
        capabilities: Optional[List[str]] = None,
        devices: Optional[List[str]] = None,
        envs: Optional[dict] = None,
        volumes: Optional[List[dict]] = None,
        ports: Optional[List[dict]] = None,
        workspace_path: Optional[str] = None,
        comment: Optional[str] = None,
) -> ExegolContainer:
    """Create a new Exegol container using the SDK.

    Covers all `exegol start` creation parameters.

    Args:
        name: Container name
        image_name: Exegol image to use (e.g. "free", "ad", "web")
        network_mode: Network mode (host, bridge, docker, nat, disabled)
        vpn_path: Path to VPN config file on the host
        vpn_auth: Path to VPN auth credentials file
        enable_gui: Enable console GUI (X11 + Wayland)
        enable_desktop: Enable remote desktop
        desktop_config: Desktop configuration string (e.g. "localhost:3389")
        share_timezone: Share host timezone with the container
        enable_my_resources: Mount /opt/my-resources
        enable_exegol_resources: Mount /opt/resources
        enable_shell_logging: Enable shell logging
        shell_logging_method: Logging method: "asciinema" or "script"
        shell_logging_compress: Compress shell logs (None = default)
        privileged: Run container in privileged mode
        hostname: Custom hostname (default: container name)
        shell: Default shell (e.g. "bash", "zsh", "fish")
        capabilities: Additional Linux capabilities (e.g. ["NET_ADMIN", "SYS_PTRACE"])
        devices: Host devices to passthrough (e.g. ["/dev/ttyUSB0"])
        envs: Environment variables as key-value pairs (e.g. {"MY_VAR": "value"})
        volumes: Volume mounts as list of {host, container, read_only?} dicts
        ports: Port mappings as list of {host, container, protocol?} dicts
        workspace_path: Custom workspace path on host (default: dedicated)
        comment: Optional comment describing the container purpose
    Returns:
        The created ExegolContainer
    """
    import re
    from exegol.model.ContainerConfig import ContainerConfig
    from exegol.model.ExegolContainerTemplate import ExegolContainerTemplate
    from exegol.model.ExegolNetwork import ExegolNetworkMode

    # Validate name
    if not name or not name.strip():
        raise ValueError("Container name cannot be empty")
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', name):
        raise ValueError(f"Invalid container name '{name}'. Use only letters, digits, hyphens, dots, and underscores.")

    # Validate network mode
    mode_map = {
        "host": ExegolNetworkMode.host,
        "bridge": ExegolNetworkMode.docker,
        "docker": ExegolNetworkMode.docker,
        "nat": ExegolNetworkMode.nat,
        "disabled": ExegolNetworkMode.disabled,
    }
    if network_mode.lower() not in mode_map:
        raise ValueError(f"Invalid network mode '{network_mode}'. Valid: {', '.join(mode_map.keys())}")

    # Resolve image
    selected_image = await DockerUtils().getOfficialImageFromList(image_name)
    if selected_image is None:
        raise ValueError(f"Image '{image_name}' not found")
    if not selected_image.isInstall():
        raise ValueError(f"Image '{image_name}' is not installed. Download it first with download_image.")

    # Build config
    config = ContainerConfig(container_name=name, hostname=hostname)

    # Network
    net_mode = mode_map[network_mode.lower()]
    await config.setNetworkMode(net_mode)

    # Core features
    if enable_gui:
        await config.enableGUI()
    if enable_desktop:
        await config.enableDesktop(desktop_config or "")
    if share_timezone:
        config.enableSharedTimezone()
    if enable_my_resources:
        config.enableMyResources()
    if enable_exegol_resources:
        await config.enableExegolResources()
    if enable_shell_logging:
        config.enableShellLogging(shell_logging_method, shell_logging_compress)
    if privileged:
        config.setPrivileged(True)
    if vpn_path:
        # VPN auth is read from ParametersManager inside enableVPN
        if vpn_auth:
            from exegol.console.cli.ParametersManager import ParametersManager
            ParametersManager().set_parameter("vpn_auth", vpn_auth)
        await config.enableVPN(vpn_path)
        if vpn_auth:
            ParametersManager().set_parameter("vpn_auth", None)
    if comment:
        config.setComment(comment)

    # Shell override
    if shell:
        config.addEnv(ContainerConfig.ExegolEnv.user_shell.value, shell)

    # Capabilities
    if capabilities:
        for cap in capabilities:
            config.addCapability(cap.upper())

    # Devices
    if devices:
        for dev in devices:
            config.addUserDevice(dev)

    # Environment variables
    if envs:
        for key, value in envs.items():
            config.addEnv(key, str(value))

    # Volumes
    if volumes:
        for vol in volumes:
            config.addVolume(
                host_path=vol["host"],
                container_path=vol["container"],
                read_only=vol.get("read_only", False),
            )

    # Ports
    if ports:
        for port in ports:
            await config.addPort(
                port_host=int(port["host"]),
                port_container=int(port["container"]),
                protocol=port.get("protocol", "tcp"),
            )

    # Custom workspace path
    if workspace_path:
        config.setWorkspaceShare(workspace_path)

    # Create template and container
    template = ExegolContainerTemplate(name=name, image=selected_image, config=config)
    container = DockerUtils().createContainer(template)

    # Post-creation setup (entrypoint patching, workspace deployment)
    await container.postCreateSetup()
    await container.start()

    # Clear DockerUtils cache so list_exegol_containers sees the new container
    DockerUtils().clearCache()

    return container


def get_container_by_name(name: str) -> Optional[ExegolContainer]:
    """
    Get an Exegol container by its name.
    Args:
        name: Name of the container
    Returns:
        ExegolContainer if found, None otherwise
    """
    try:
        return DockerUtils().getContainer(name)
    except ObjectNotFound:
        return None

async def list_images(installed_only: bool = False) -> List[ImageInfo]:
    """List Exegol images from the exegol sdk"""
    DockerUtils().clearCache()
    images: List[ExegolImage] = await DockerUtils().listImages(include_custom=True)
    DockerUtils().clearCache()
    results: List[ImageInfo] = []
    for i in images:
        if installed_only and not i.isInstall():
            continue
        results.append(ImageInfo(
            name=i.getName(),
            version=i.getImageVersion(),
            lastest_version=i.getLatestVersion(),
            is_installed=i.isInstall(),
            is_built_locally=i.isLocal(),
            is_up_to_date=i.isUpToDate(),
            build_date=i.getBuildDate(),
            license=i.getDisplayLicense()
        ))
    return results

async def download_exegol_image(image_name: str, image_version: str = "latest") -> bool:
    """Download an Exegol image from the exegol sdk"""
    image_tag = f"{image_name}-{image_version}" if image_version != "latest" else image_name
    try:
        # Find image by name
        selected_image = await DockerUtils().getOfficialImageFromList(image_tag)
        DockerUtils().clearCache()
    except ObjectNotFound:
        raise RuntimeError(f"Image '{image_tag}' doesn't exist") from None

    if await DockerUtils().downloadImage(selected_image, install_mode=not selected_image.isInstall()):
        if selected_image.isVersionSpecific():
            # Install latest tag if not already installed
            try:
                result = await DockerUtils().getOfficialImageFromList(image_name)
                if result is None or not result.isInstall():
                    raise ObjectNotFound
            except ObjectNotFound:
                DockerUtils().createLocalLastestImageTag(selected_image)
        elif selected_image.hasVersionTag():
            # Install version specific tag
            result = await DockerUtils().downloadVersionTag(selected_image)
            if type(result) is str:
                raise RuntimeError(f"Error while downloading version tag, '{image_tag}': {result}")
        return True
    return False


async def read_installed_resources(ctx: Context, container_name: str, target_os: str) -> List[str]:
    """
    Helper function to read the installed_tools.csv file from an Exegol container.
    Args:
        container_name: Name of the Exegol container
        target_os: OS type of the target
        ctx: MCP context
    Returns:
        Content of the installed_tools.csv file
    """
    if target_os not in ["linux", "windows"]:
        raise ValueError(f"Invalid target OS '{target_os}'. Only 'linux' and 'windows' are supported.")
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
    exit_code, output = await container.exec_raw(f"find /opt/resources/{target_os} -type f | grep -iv '/.git' | grep -ivE '\\.(md|png|jpg|txt|py|c|cpp|cs|h|sln|vcxproj(.filters)?|xml|csproj|yml|pdb|dll|idl|sys|chm|desktop|json|pssproj|rst|spec|xsd|config|asm)$' | grep -ivE '/(LICEN[SC]E|makefile|bootstrap|dockerfile)$'")

    if exit_code != 0:
        await ctx.error(f"Failed to read installed_tools.csv from container '{container_name}'")
        raise RuntimeError(f"Failed to read installed_tools.csv: {output}")

    return output.splitlines()
