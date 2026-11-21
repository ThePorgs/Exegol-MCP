"""Image services (SDK-free: they only talk to an ExegolPort)."""

from typing import get_args

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.context.request import ProgressCallback, RequestState, bind_request
from exegol_mcp.models.image import ImageInfo, ImageScope

# Images that only exist as their latest build: Exegol publishes no versioned tags for them.
LATEST_ONLY_IMAGES = ("nightly", "free")


async def list_images(adapter: ExegolPort, scope: ImageScope = "installed") -> list[ImageInfo]:
    """List Exegol images.

    Both scopes take the same path: v1 skipped the readiness check for its "all images" tool, v2 does not
    special-case either scope.

    Args:
        adapter: The Exegol port.
        scope: "installed" for the images on this host, "all" for installed plus available for download.

    Returns:
        The images in the adapter's order, [] when there is none.

    Raises:
        ValueError: If scope is not "installed" or "all".
    """
    if scope not in get_args(ImageScope):
        raise ValueError(f"Invalid scope '{scope}'. Valid scopes: {', '.join(get_args(ImageScope))}")
    return await adapter.list_images(scope)


async def download_image(
    adapter: ExegolPort,
    name: str,
    version: str = "latest",
    progress: ProgressCallback | None = None,
    request_id: str | None = None,
) -> bool:
    """Download or update an Exegol image.

    Args:
        adapter: The Exegol port.
        name: Name of the image (for example "full").
        version: Version to install, "latest" by default. nightly and free only exist as "latest".
        progress: Progress sink of the calling request, None when the caller did not ask for progress.
        request_id: Id of the calling MCP request, kept with the per-request state.

    Returns:
        True once the image is installed and up to date, including when it already was (the adapter then pulls
        nothing).

    Raises:
        ValueError: If the name or version is empty, or a versioned download is asked for nightly or free.
    """
    # 1. Validate the arguments before any Docker call
    if not name.strip():
        raise ValueError("image_name must not be empty. Call list_images with scope 'all' to see the image names.")
    if not version.strip():
        raise ValueError("image_version must not be empty. Use 'latest' for the latest build.")
    if name in LATEST_ONLY_IMAGES and version != "latest":
        raise ValueError(f"{name} image can only be downloaded with version 'latest'")

    # 2. Download with this request's progress sink bound. bind_request resets it in a finally block, so a
    # concurrent download never reports to another request's client (v1 used a class-global context).
    with bind_request(RequestState(request_id=request_id, progress=progress)):
        return await adapter.download_image(name, version, progress)
