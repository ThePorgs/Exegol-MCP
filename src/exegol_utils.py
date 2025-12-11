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
    results = []
    for container in containers:
        network_drivers, network_name = container.config.getNetwork()
        results.append({
            "name": container.name,
            "image_name": container.image.getName(),
            "image_version": container.image.getImageVersion(),
            "status": container.getRawStatus(),
            "network_driver": network_drivers,
            "network_name": network_name
        })
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
