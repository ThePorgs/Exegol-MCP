"""Result models for container removal and cleanup (LIFE-03, LIFE-04, D-12, D-13).

These shapes reach the LLM (FastMCP derives the tools' output schema from them), so every field carries a
description.
"""

from typing import Literal

from pydantic import BaseModel, Field

CleanupScope = Literal["this_run", "all_managed"]


# LIFE-03, D-12
class RemovedContainer(BaseModel):
    """One removed container and what is still on the host after its removal"""

    name: str = Field(description="Name of the removed container")
    workspace_kept: bool = Field(
        description=(
            "True when the container's workspace is still on the host after the removal: kept on request, a "
            "host-mounted workspace that still exists (never deleted), or one Exegol could not delete (a WSL "
            "workspace, or files it lacked the rights to remove). False once it is gone, including an empty private "
            "workspace Exegol deletes even when keeping it was asked (nothing is lost then). False when the "
            "container had no workspace. When the host cannot be checked, the workspace is reported as kept. Shell "
            "logs under /workspace/logs follow it."
        )
    )
    logs_kept: bool = Field(
        description=(
            "True when a Sentinel audit history directory is still on the host after the removal. False when the "
            "container had no Sentinel logging, or the history is gone (Exegol deletes a Sentinel directory that "
            "holds no history even when keeping it was asked). When the host cannot be checked, the history is "
            "reported as kept."
        )
    )


# LIFE-04
class CleanupFailure(BaseModel):
    """One container cleanup_containers could not remove"""

    name: str = Field(description="Name of the container that was not removed")
    error: str = Field(
        description=(
            "Why the removal failed. The container is usually still on the host: call list_containers to check (one "
            "removed concurrently by another process is reported here too)."
        )
    )


# LIFE-04, D-13
class CleanupResult(BaseModel):
    """What cleanup_containers removed and what it could not"""

    scope: CleanupScope = Field(
        description="this_run: containers this server process created; all_managed: every exegol-mcp container"
    )
    removed: list[str] = Field(description="Names of the removed containers, sorted by name; empty when none matched")
    failed: list[CleanupFailure] = Field(
        description="Containers whose removal failed, sorted by name; the others were still removed"
    )
