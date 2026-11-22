from pydantic import BaseModel, Field


class ContainerInfo(BaseModel):
    """Container information from Exegol SDK"""

    name: str = Field(description="Exegol container name, used by every tool that acts on a container")
    creation_date: str = Field(description="Date the container was created, as reported by Exegol")
    image_name: str = Field(description="Name of the Exegol image the container was created from")
    image_version: str = Field(description="Version of the Exegol image the container was created from")
    status: str = Field(description="Current container status as reported by Docker (for example Running or Stopped)")
    network_driver: str | None = Field(description="Docker network driver (host, bridge, ...), null when unknown")
    network_name: str | None = Field(description="Name of the Docker network the container is attached to, if any")
    features: list[str] = Field(description="Exegol features enabled on the container (resources, shell logging, ...)")
    devices: list[str] = Field(description="Host devices shared with the container")
    vpn: str | None = Field(description="VPN configuration file name, null when the container has no VPN")
    env: list[str] = Field(description="Custom environment variables set on the container, as KEY=VALUE strings")
    is_privileged: bool = Field(description="True when the container runs in Docker privileged mode")
    capabilities: list[str] = Field(description="Linux capabilities granted to the container")
    comment: str | None = Field(description="Free-text comment attached to the container, null when none")
    managed: bool = Field(
        default=False, description="True when the container was created by exegol-mcp (carries its label)"
    )
    run_id: str | None = Field(
        default=None, description="ID of the exegol-mcp server run that created it, null when not managed"
    )
    this_run: bool = Field(default=False, description="True when this server process created it")


class InstalledTool(BaseModel):
    """Information about an installed tool in an Exegol container"""

    name: str = Field(description="Name of the tool")
    version: str | None = Field(default=None, description="Installed tool version when recorded")
    link: str | None = Field(
        default=None,
        description="Upstream URL of the tool (its project page or repository) when the catalog records it",
    )
    description: str | None = Field(default=None, description="Description of the tool")
