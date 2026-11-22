"""MCP resources: read-only views a client reads by URI.

These are MCP protocol resources, not the "Exegol resources" (the offline resource bundle inside a container) that
the list_offline_resources tool lists.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

from exegol_mcp.models.config import ToolGroup
from exegol_mcp.resources.catalog import installed_tools
from exegol_mcp.resources.containers import container_info
from exegol_mcp.resources.files import workspace_file

# The resource URI contract (05-01 Task 1, maintainer choice option-a). A published contract once v2 ships: the
# plural "containers" segment names the collection and {name} picks one container, REST-style.
CONTAINER_INFO_URI = "exegol://containers/{name}/info"
INSTALLED_TOOLS_URI = "exegol://containers/{name}/tools"
WORKSPACE_FILE_URI = "exegol://containers/{name}/workspace/{+path}"


class ResourceEntry(NamedTuple):
    """One registry entry: the template function, its URI template, client-facing metadata and its tool group."""

    fn: Callable[..., Any]
    uri_template: str
    name: str
    title: str
    mime_type: str
    group: ToolGroup  # never None: every resource belongs to a tool group (D-04)
    listed: bool  # True when resources/list enumerates one concrete resource per container


# Explicit resource registry consumed by build_server(). Gating filters on the group at registration, as for tools:
# build_server registers only the entries whose group is not in tool_groups.disabled (D-04), so a disabled resource
# is absent from resources/templates/list and resources/list, and a read of it fails as not found. Container info
# and the installed-tools catalog follow the containers group; workspace files follow the files group.
RESOURCE_REGISTRY: list[ResourceEntry] = [
    ResourceEntry(
        container_info,
        CONTAINER_INFO_URI,
        "container_info",
        "Exegol container info",
        "application/json",
        group="containers",
        listed=True,
    ),
    ResourceEntry(
        installed_tools,
        INSTALLED_TOOLS_URI,
        "installed_tools",
        "Installed tools catalog",
        "application/json",
        group="containers",
        listed=True,
    ),
    # Template-only: workspace files are never enumerated in resources/list (D-03).
    ResourceEntry(
        workspace_file,
        WORKSPACE_FILE_URI,
        "workspace_file",
        "Workspace file",
        "application/octet-stream",
        group="files",
        listed=False,
    ),
]
