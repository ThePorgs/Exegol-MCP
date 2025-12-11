from typing import List

from mcp.server.fastmcp import Context

from src.exegol_utils import get_exegol_container, is_exegol_ready
from src.mcp_app import mcp_server
from src.models.container import ContainerInfo

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
    if is_exegol_ready():
        await ctx.error("Exegol is not ready yet. Please run exegol first with `exegol info` and make sure it works.")
    return await get_exegol_container()
