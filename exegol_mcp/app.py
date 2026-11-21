"""Server factory: builds the MCPServer and registers the tools and resources of every enabled tool group.

Tools come from TOOL_REGISTRY and MCP resources from RESOURCE_REGISTRY. Both are gated at registration by
tool_groups.disabled (D-23, D-04): an entry of a disabled group never exists on the server.
"""

import inspect
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from mcp.server.mcpserver import MCPServer
from mcp_types import Icon

from exegol_mcp import __version__
from exegol_mcp.adapter import offload
from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.config import OperatorConfig
from exegol_mcp.resources import RESOURCE_REGISTRY
from exegol_mcp.resources.listing import install_resource_handlers
from exegol_mcp.services.jobs import JobStore
from exegol_mcp.tools import TOOL_REGISTRY

_log = logging.getLogger(__name__)

EXEGOL_ICON = Icon(src="https://docs.exegol.com/images/Exegol_Symbol_DarkVersion.svg", mime_type="image/svg+xml")


async def _sweep_orphans(adapter: ExegolPort, phase: str) -> None:
    """Run the D-11 tag sweep, best-effort: log the count, swallow a failure so it never blocks serving."""
    try:
        swept = await adapter.sweep_orphans()
        _log.debug("%s orphan sweep killed %d tagged process(es)", phase, swept)
    except Exception:  # noqa: BLE001  # crash recovery is best-effort; a failed sweep must not stop the server
        _log.warning("%s orphan sweep failed; continuing", phase, exc_info=True)


def build_server(config: OperatorConfig | None, adapter: ExegolPort) -> MCPServer[ExegolPort]:
    """Build the Exegol MCP server around an injected adapter.

    Args:
        config: The resolved operator config (CONF-03). None falls back to the adapter's own config, or to
            OperatorConfig() built-in defaults, so build_server(None, FakeExegolAdapter()) keeps working.
        adapter: The ExegolPort implementation tools reach through ctx.request_context.lifespan_context.

    Returns:
        A configured MCPServer that registers only the TOOL_REGISTRY entries of enabled tool groups: an entry whose
        group is listed in tool_groups.disabled is never added, so it is absent from tools/list and a call to it
        fails as an unknown tool (CONF-02, D-23). server_info has no group and is always registered. Resources
        follow the same rule (D-04): a RESOURCE_REGISTRY entry of a disabled group is never registered, so it is
        absent from resources/templates/list and a read of it fails as not found.
    """
    # Resolved once, before the server exists, so registration-time gating and the lifespan see the same config.
    effective = config if config is not None else (getattr(adapter, "config", None) or OperatorConfig())

    @asynccontextmanager
    async def lifespan(_server: MCPServer[ExegolPort]) -> AsyncIterator[ExegolPort]:
        # The lifespan context carries the resolved config so tools read it from
        # ctx.request_context.lifespan_context.config (CONF-03).
        adapter.config = effective  # type: ignore[attr-defined]  # ExegolPort gains .config in Phase 2 wiring
        # D-09: this server process's run id, stamped on every container create_container makes so later listings
        # tell this run's containers from earlier runs'. Never logged.
        adapter.run_id = secrets.token_hex(6)  # type: ignore[attr-defined]  # the context gains .run_id in 04-03
        # Size the jobs limiter to the concurrency cap plus the default pool so capped job readers can never
        # starve a foreground exec's reader (RESEARCH Open Question 2, resolved).
        offload.size_jobs_limiter(effective.execution.max_concurrent_jobs)
        # D-11 crash recovery: kill any tagged process a previous (crashed) run left behind before serving.
        # Best-effort: a sweep that cannot run (Docker hiccup) must never stop the server from serving.
        await _sweep_orphans(adapter, "startup")
        # The reader loops of every background job run in this server-lifetime task group, so a job outlives
        # the request that started it (D-09).
        async with anyio.create_task_group() as task_group:
            store = JobStore(adapter, task_group, effective.execution)
            adapter.jobs = store  # type: ignore[attr-defined]  # the context gains .jobs in Phase 2 wiring
            # Test-only: seed known job ids for the contract/era "call every tool" suites. The production
            # ServerContext exposes no seed_jobs, so this is a no-op off the fake.
            for seed_id, seed_name, seed_command in getattr(adapter, "seed_jobs", ()):
                try:
                    await store.seed(seed_id, seed_name, seed_command)
                except Exception:  # noqa: BLE001  # a seed is a test fixture; never let it break startup
                    _log.debug("job seed %r skipped", seed_id, exc_info=True)
            try:
                yield adapter
            finally:
                # D-14: no container is ever removed at shutdown; only tagged job processes are swept here.
                # Sweep again at shutdown (D-11) in a shielded scope so the kill is never abandoned, then
                # cancel the reader task group so a still-running job does not hold the lifespan open.
                with anyio.CancelScope(shield=True):
                    await _sweep_orphans(adapter, "shutdown")
                task_group.cancel_scope.cancel()

    server = MCPServer(
        "Exegol",
        version=__version__,
        website_url="https://exegol.com",
        icons=[EXEGOL_ICON],
        lifespan=lifespan,
    )
    # D-23: gating happens at registration, so a disabled tool never exists on the server. Gating hides tools; it is
    # not a security boundary between groups (exec_command can still move files when `files` is disabled).
    disabled = set(effective.tool_groups.disabled)
    withheld = 0
    for entry in TOOL_REGISTRY:
        if entry.group is None or entry.group not in disabled:
            server.add_tool(entry.fn, annotations=entry.annotations)
        else:
            withheld += 1
    # D-04: resources are gated by the same groups, at registration too. Like tool gating, this hides surface; it does
    # not confine an agent that still has exec_command.
    enabled_resources = [resource for resource in RESOURCE_REGISTRY if resource.group not in disabled]
    withheld_resources = len(RESOURCE_REGISTRY) - len(enabled_resources)
    for resource in enabled_resources:
        server.resource(
            resource.uri_template,
            name=resource.name,
            title=resource.title,
            description=inspect.cleandoc(resource.fn.__doc__ or ""),
            mime_type=resource.mime_type,
        )(resource.fn)
    # D-03: a live, paginated resources/list over the enabled entries. Installed even when none is enabled, so the
    # SDK's static, cursor-ignoring handler is never what answers.
    install_resource_handlers(server, enabled_resources)
    if disabled:
        _log.debug(
            "tool groups disabled: %s; %d tool(s) and %d resource template(s) withheld",
            ", ".join(sorted(disabled)),
            withheld,
            withheld_resources,
        )
    return server
