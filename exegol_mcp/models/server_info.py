"""Published result shape of the server_info tool (D-13)."""

from typing import Literal

from pydantic import BaseModel, Field

# How the bundled Exegol library relates to the Exegol wrapper installed on the host. A mismatch is information,
# never a failure (D-07/D-12): startup only fails when the Exegol config cannot be read at all.
CompatibilityState = Literal["match", "mismatch", "unknown"]


class ServerInfo(BaseModel):
    """Versions, Docker status and compatibility state of this exegol-mcp server"""

    exegol_mcp_version: str = Field(description="Version of the exegol-mcp server")
    exegol_library_version: str | None = Field(
        description="Version of the Exegol library bundled with exegol-mcp, null when it cannot be determined",
    )
    exegol_wrapper_version: str | None = Field(
        default=None,
        description="Version of the Exegol wrapper installed on the host, null when it is not detectable",
    )
    docker_status: str = Field(description="Docker daemon status as seen by the server (for example reachable)")
    config_dirs: list[str] = Field(description="Exegol and exegol-mcp configuration directories in use")
    compatibility_state: CompatibilityState = Field(
        description=(
            "Exegol library vs wrapper versions: 'match', 'mismatch' (information only, the server still works) "
            "or 'unknown' when the wrapper version is not detectable"
        ),
    )
    update_notice: str | None = Field(
        default=None,
        description="Message telling the user a newer exegol-mcp release is available, null when up to date or unknown",
    )
