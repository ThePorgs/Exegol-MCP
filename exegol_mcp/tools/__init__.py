from collections.abc import Callable
from typing import Any, NamedTuple

from mcp_types import ToolAnnotations

from exegol_mcp.models.config import ToolGroup
from exegol_mcp.tools.catalog import get_tool_help, list_installed_tools, list_offline_resources
from exegol_mcp.tools.containers import list_containers, start_container, stop_container
from exegol_mcp.tools.exec import exec_command
from exegol_mcp.tools.files import list_directory, read_file, write_file
from exegol_mcp.tools.images import download_image, list_images
from exegol_mcp.tools.jobs import get_job, list_jobs, stop_job
from exegol_mcp.tools.lifecycle import cleanup_containers, create_container, remove_container
from exegol_mcp.tools.remote import exec_remote_command
from exegol_mcp.tools.server_info import server_info


class ToolEntry(NamedTuple):
    """One registry entry: the tool function, its explicit annotations and its tool group."""

    fn: Callable[..., Any]
    annotations: ToolAnnotations
    group: ToolGroup | None  # None only for server_info, which belongs to no tool group and is never gateable


# Explicit tool registry consumed by build_server(). A tool reaches the server only through this list,
# never through an import side effect. Every entry sets read_only_hint and destructive_hint explicitly.
# Gating filters on the group at registration: build_server skips every entry whose group the operator listed in
# tool_groups.disabled (CONF-02, D-23), so a disabled tool is absent from tools/list.
TOOL_REGISTRY: list[ToolEntry] = [
    ToolEntry(
        list_containers,
        ToolAnnotations(title="List Exegol containers", read_only_hint=True, destructive_hint=False),
        group="containers",
    ),
    ToolEntry(
        start_container,
        ToolAnnotations(
            title="Start an Exegol container",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        group="containers",
    ),
    ToolEntry(
        stop_container,
        ToolAnnotations(
            title="Stop an Exegol container",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
        ),
        group="containers",
    ),
    ToolEntry(
        list_images,
        ToolAnnotations(
            title="List Exegol images",
            read_only_hint=True,
            destructive_hint=False,
            # scope "all" queries the Exegol registry
            open_world_hint=True,
        ),
        group="images",
    ),
    ToolEntry(
        download_image,
        ToolAnnotations(
            title="Download an Exegol image",
            read_only_hint=False,
            destructive_hint=False,
            open_world_hint=True,
        ),
        group="images",
    ),
    ToolEntry(
        server_info,
        ToolAnnotations(
            title="Exegol MCP server information",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        group=None,
    ),
    ToolEntry(
        exec_command,
        ToolAnnotations(
            title="Run a command in an Exegol container",
            read_only_hint=False,
            destructive_hint=True,
            open_world_hint=True,
        ),
        group="execution",
    ),
    ToolEntry(
        exec_remote_command,
        ToolAnnotations(
            title="Run a command on a remote host via netexec",
            read_only_hint=False,
            destructive_hint=True,
            open_world_hint=True,
        ),
        group="execution",
    ),
    ToolEntry(
        list_installed_tools,
        ToolAnnotations(
            title="List installed tools in an Exegol container",
            read_only_hint=True,
            destructive_hint=False,
        ),
        group="containers",
    ),
    ToolEntry(
        get_tool_help,
        ToolAnnotations(
            title="Get help for a tool in an Exegol container",
            read_only_hint=True,
            destructive_hint=False,
        ),
        group="execution",
    ),
    ToolEntry(
        list_offline_resources,
        ToolAnnotations(
            title="List Exegol offline resources in a container",
            read_only_hint=True,
            destructive_hint=False,
        ),
        group="containers",
    ),
    ToolEntry(
        get_job,
        ToolAnnotations(
            title="Poll a background job and read its output",
            read_only_hint=True,
            destructive_hint=False,
        ),
        group="execution",
    ),
    ToolEntry(
        stop_job,
        ToolAnnotations(
            title="Stop a background job",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
        ),
        group="execution",
    ),
    ToolEntry(
        list_jobs,
        ToolAnnotations(
            title="List background jobs on this server",
            read_only_hint=True,
            destructive_hint=False,
        ),
        group="execution",
    ),
    ToolEntry(
        create_container,
        ToolAnnotations(
            title="Create an Exegol container",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
        group="containers",
    ),
    ToolEntry(
        remove_container,
        ToolAnnotations(
            title="Remove an Exegol container",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        ),
        group="containers",
    ),
    ToolEntry(
        cleanup_containers,
        ToolAnnotations(
            title="Remove exegol-mcp's own containers",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        group="containers",
    ),
    ToolEntry(
        read_file,
        ToolAnnotations(
            title="Read a file from an Exegol container",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        group="files",
    ),
    ToolEntry(
        write_file,
        ToolAnnotations(
            title="Write a file into an Exegol container",
            read_only_hint=False,
            destructive_hint=True,
            # Not idempotent: a repeated overwrite=false call fails, and `created` differs between calls (IN-06).
            idempotent_hint=False,
            open_world_hint=False,
        ),
        group="files",
    ),
    ToolEntry(
        list_directory,
        ToolAnnotations(
            title="List a directory in an Exegol container",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        group="files",
    ),
]
