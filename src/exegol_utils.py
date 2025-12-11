import os
from typing import List, Optional

from exegol.exceptions.ExegolExceptions import ObjectNotFound
from exegol.model.ExegolContainer import ExegolContainer
from exegol.utils.DockerUtils import DockerUtils

from src.models.container import ContainerInfo

def is_exegol_ready() -> bool:
    # TODO add check if exegol is ready
    return True

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

async def get_container_by_name(name: str) -> Optional[ExegolContainer]:
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
