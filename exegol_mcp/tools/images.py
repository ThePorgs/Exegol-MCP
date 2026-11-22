"""MCP tool handlers for Exegol images. Thin wrappers: adapter from the lifespan, logic in services."""

from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.image import ImageInfo, ImageScope
from exegol_mcp.services import images as images_service
from exegol_mcp.tools.errors import tool_errors


async def list_images(scope: ImageScope = "installed", *, ctx: Context[ExegolPort, Any]) -> list[ImageInfo]:
    """List Exegol images with their version and update status.

    IMPORTANT: scope "installed" (the default) lists only the images on this host. scope "all" also lists the
    images available for download, which needs the Exegol registry. This tool only reads state.

    RELATED TOOLS:
    - download_image: install or update an image listed here
    - list_containers: see which images existing containers use

    Args:
        scope: "installed" for the images on this host, "all" for installed plus available for download.

    Returns:
        One entry per image (name, installed version, latest version, installed / up-to-date flags, build date,
        license). An empty list means no image matches the scope.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await images_service.list_images(adapter, scope)


async def download_image(
    image_name: str,
    image_version: str = "latest",
    *,
    ctx: Context[ExegolPort, Any],
) -> bool:
    """Download and install an Exegol image, or update an installed image that is outdated.

    IMPORTANT: Downloads are large and can take many minutes. Progress is reported to clients that ask for it.
    An image that is already installed and up to date is not downloaded again: the call returns true at once.
    A download that Exegol refuses or that fails is an error, never false.
    The nightly and free images only exist as version "latest". Pro images can install a specific older version.

    RELATED TOOLS:
    - list_images: find the image names (scope "all") and check what is already installed
    - list_containers: see which images existing containers use

    Args:
        image_name: Name of the image to download, as returned by list_images.
        image_version: Version to install, "latest" by default.

    Returns:
        true once the image is installed and up to date, including when it already was.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await images_service.download_image(
            adapter,
            image_name,
            image_version,
            progress=ctx.report_progress,
            request_id=ctx.request_id,
        )
