import os
from typing import List, Optional

from exegol.exceptions.ExegolExceptions import ObjectNotFound
from exegol.model.ExegolContainer import ExegolContainer
from exegol.model.ExegolImage import ExegolImage
from exegol.utils.DockerUtils import DockerUtils

from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.image import ImageInfo


async def check_exegol_readiness(ctx) -> bool:
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
        await ctx.info("Exegol readiness check passed")
        return True
    except Exception as e:
        # If any exception occurs, Exegol is not ready
        # Common causes: Docker not running, Docker not installed, Exegol not configured
        error_msg = (
            f"Exegol is not ready yet. Please run exegol first with `exegol info` "
            f"and make sure it works. Error: {str(e)}"
        )
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e

async def get_exegol_container() -> List[ContainerInfo]:
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
        if not selected_image.isVersionSpecific() and selected_image.hasVersionTag():
            result = await DockerUtils().downloadVersionTag(selected_image)
            if type(result) is str:
                raise RuntimeError(f"Error while downloading version tag, '{image_tag}': {result}")
        return True
    return False
