"""MCP resource template for one Exegol container's info, and the running-container guard every other resource uses.

Thin: adapter from the lifespan, data from the service.
"""

from typing import Any

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.services import containers as containers_service
from exegol_mcp.tools.errors import resource_errors

# The stopped-container error of every resource that reads inside a container. Resources take no start argument and
# a read never changes a container, so the wording names start_container only.
RESOURCE_NOT_RUNNING_MESSAGE = (
    "Container '{name}' is not running, and reading a resource never starts it. Start it with start_container, then "
    "read the resource again."
)


async def require_running_for_read(adapter: ExegolPort, name: str) -> None:
    """Check that a container exists and runs before a resource reads inside it. Never starts it.

    Args:
        adapter: The Exegol port.
        name: Name of the container.

    Raises:
        ValueError: If no container has that name (a not-found resource, -32602).
        RuntimeError: If the container exists but is stopped (RESOURCE_NOT_RUNNING_MESSAGE, -32603).
    """
    state = await adapter.get_container_state(name)
    if state is None:
        raise ValueError(f"Container '{name}' not found. Call list_containers to see the existing containers.")
    if not state.running:
        raise RuntimeError(RESOURCE_NOT_RUNNING_MESSAGE.format(name=name))


async def container_info(name: str, *, ctx: Context[ExegolPort, Any]) -> str:
    """Information about one Exegol container, as JSON: the same fields list_containers returns for it.

    The JSON object holds the container's name, creation date, image name and version, status, network, enabled
    features, shared devices, VPN, custom environment variables, privileged mode, capabilities and comment, plus
    whether exegol-mcp created it (managed), the ID of the server run that created it (run_id) and whether this
    server process created it (this_run).

    IMPORTANT: Reading this resource only reads the container listing. It works for stopped containers and never
    starts, stops or changes a container. The name is the exact, case-sensitive container name.

    RELATED TOOLS:
    - list_containers: list every container with the same information
    - start_container: start a stopped container before running commands in it
    - the installed-tools resource of the same container: the catalog of tools installed in it
    """
    # RES-03: info is read from the listing alone (no exec, no state check), so a stopped container is readable and
    # nothing is ever started. The content is exactly what list_containers returns, never filtered.
    adapter = ctx.request_context.lifespan_context
    async with resource_errors(adapter):
        infos = await containers_service.list_containers(adapter, run_id=getattr(adapter, "run_id", None))
        for info in infos:
            if info.name == name:
                return info.model_dump_json()
        raise ValueError(f"Container '{name}' not found. Call list_containers to see the existing containers.")
