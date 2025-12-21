import logging
from typing import Set, Dict, Generator, Optional

from mcp.server.fastmcp import Context


class TUI_MCP:

    DOWNLOAD_CONTEXT: Optional[Context] = None

    @classmethod
    async def downloadDockerLayer(cls, stream: Generator, quick_exit: bool = False) -> None:
        if cls.DOWNLOAD_CONTEXT is None:
            raise RuntimeError("No download context available.")
        ctx: Context = cls.DOWNLOAD_CONTEXT
        layers: Set[str] = set()
        layers_downloaded: Set[str] = set()
        layers_extracted: Set[str] = set()
        total = len(layers) * 2

        for line in stream:  # Receiving stream from docker API
            status = line.get("status", '')
            error = line.get("error", '')
            layer_id = line.get("id")
            update_needed = False
            if error != "":
                logging.error(f"Docker download error: {error}")
                await ctx.error(f"Docker download error: {error}")
                raise RuntimeError(f"Docker download error: {error}")

            if status == "Pulling fs layer":  # Identify new layer to download
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
            elif status == "Pull complete":  # Mark task as complete and remove it from the pool
                layers_extracted.add(layer_id)
                update_needed = True
            elif "Image is up to date" in status or "Status: Downloaded newer image for" in status:
                await ctx.info(status)
            elif status in ["Waiting", "Verifying Checksum", "Downloading", "Extracting"] or "Pulling from " in status:
                # Ignore these status messages
                continue
            else:
                logging.info(f"Unknown docker download status: {status}")

            if update_needed:
                message = f"Downloading layers: {len(layers_downloaded)} / {len(layers)}" if len(layers_downloaded) < len(layers) else f"Extracting layers: {len(layers_extracted)} / {len(layers)}"
                current = len(layers_downloaded) + len(layers_extracted)
                await ctx.report_progress(progress=int(current*100/total),
                                                           total=100,
                                                           message=message)
        await ctx.report_progress(progress=100, total=100, message="Download complete")

