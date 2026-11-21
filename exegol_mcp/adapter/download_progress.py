"""Docker image pull progress, reported to the request that asked for the pull (D-25).

patch_exegol_sdk() installs download_docker_layer() as ExegolTUI.downloadDockerLayer, so DockerUtils.downloadImage
hands it docker-py's pull stream instead of drawing a rich progress bar. Two v1 problems are fixed here:

- v1 read the progress target from a class attribute shared by every call, so two concurrent pulls reported to
  whichever request set it last. The target now comes from the request's own ContextVar state.
- v1 iterated the pull stream on the event loop. Each next() on it is a blocking read from the Docker daemon, so
  every line is now pulled in a worker thread on the dedicated pull limiter.
"""

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from exegol_mcp.adapter.offload import pull_limiter, run_blocking
from exegol_mcp.context.request import request_ctx

_log = logging.getLogger(__name__)

# Returned by next() once the pull stream is exhausted.
_END_OF_STREAM: Any = object()

# Statuses docker sends many times per layer; they carry no state change.
_IGNORED_STATUSES = ("Waiting", "Verifying Checksum", "Downloading", "Extracting")


async def report_progress(progress: float, total: float | None, message: str | None) -> None:
    """Send a progress update to the current request's progress sink, if it bound one."""
    state = request_ctx.get()
    if state is not None and state.progress is not None:
        await state.progress(progress, total, message)


async def download_docker_layer(stream: Iterable[dict[str, Any]], quick_exit: bool = False) -> None:
    """Follow a docker pull stream and report layer progress (0-100) to the current request.

    Progress counts each layer twice, once downloaded and once extracted, as v1 did.

    Args:
        stream: docker-py's decoded pull stream (one dict per status line).
        quick_exit: Accepted for signature compatibility with ExegolTUI.downloadDockerLayer; unused.

    Raises:
        RuntimeError: If docker reports an error line.
    """
    lines: Iterator[dict[str, Any]] = iter(stream)
    layers: set[Any] = set()
    layers_downloaded: set[Any] = set()
    layers_extracted: set[Any] = set()
    total = 0

    while True:
        line = await run_blocking(next, lines, _END_OF_STREAM, limiter=pull_limiter)
        if line is _END_OF_STREAM:
            break
        status = line.get("status", "")
        error = line.get("error", "")
        layer_id = line.get("id")
        update_needed = False
        if error != "":
            _log.error(f"Docker download error: {error}")
            raise RuntimeError(f"Docker download error: {error}")

        if status == "Pulling fs layer":  # a new layer to download
            layers.add(layer_id)
            total = len(layers) * 2
            update_needed = True
        elif status == "Already exists":
            layers.add(layer_id)
            total = len(layers) * 2
            layers_downloaded.add(layer_id)
            layers_extracted.add(layer_id)
            update_needed = True
        elif status == "Download complete":
            layers_downloaded.add(layer_id)
            update_needed = True
        elif status == "Pull complete":
            layers_extracted.add(layer_id)
            update_needed = True
        elif "Image is up to date" in status or "Status: Downloaded newer image for" in status:
            _log.info(status)
        elif status in _IGNORED_STATUSES or "Pulling from " in status:
            continue
        else:
            _log.info(f"Unknown docker download status: {status}")

        if update_needed:
            if total == 0:
                # A layer finished before docker announced any layer: there is no ratio to report yet.
                continue
            if len(layers_downloaded) < len(layers):
                message = f"Downloading layers: {len(layers_downloaded)} / {len(layers)}"
            else:
                message = f"Extracting layers: {len(layers_extracted)} / {len(layers)}"
            current = len(layers_downloaded) + len(layers_extracted)
            await report_progress(int(current * 100 / total), 100, message)

    await report_progress(100, 100, "Download complete")
