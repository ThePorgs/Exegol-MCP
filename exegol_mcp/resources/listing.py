"""A live, paginated resources/list for the per-container MCP resources (D-03).

MCPServer's own resources/list handler returns only the static resources registered at build time and ignores the
cursor. Exegol resources are per container and containers come and go, so this module replaces that handler on the
low-level server: each call lists the containers live, builds one concrete resource per container for every listed
registry entry, and pages the result with the shared opaque cursor helpers. Nothing is cached at build time.

It also wraps resources/read so a text read of a template declared application/octet-stream (workspace files) is
labelled text/plain, while its binary reads keep application/octet-stream.

No resources list-changed notification is ever sent.
"""

import inspect
from collections.abc import Sequence
from typing import Any

from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.shared.uri_template import UriTemplate
from mcp_types import (
    ListResourcesResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextResourceContents,
)

from exegol_mcp.resources import ResourceEntry
from exegol_mcp.services import containers as containers_service
from exegol_mcp.services import paging
from exegol_mcp.tools.catalog import _page_size
from exegol_mcp.tools.errors import protocol_errors

_OCTET_STREAM = "application/octet-stream"
_TEXT_PLAIN = "text/plain"


def _summary(entry: ResourceEntry) -> str:
    """The first paragraph of the entry's template description: the one-line summary shown per resource."""
    return inspect.cleandoc(entry.fn.__doc__ or "").split("\n\n", 1)[0]


def install_resource_handlers(server: MCPServer[Any], entries: Sequence[ResourceEntry]) -> None:
    """Replace the server's resources/list handler with a live, paginated per-container listing, and wrap its
    resources/read handler so text reads of application/octet-stream templates are labelled text/plain.

    Args:
        server: The MCPServer build_server is configuring.
        entries: The RESOURCE_REGISTRY entries of enabled groups. Only those with listed=True are enumerated, so a
            resource of a disabled group never appears. An empty sequence still installs the handler, which then
            returns an empty list rather than falling back to the SDK's static one.
    """
    # Parsed once here: a malformed template fails at startup, not on the first list request.
    listed = [(entry, UriTemplate.parse(entry.uri_template), _summary(entry)) for entry in entries if entry.listed]

    async def list_resources(
        ctx: ServerRequestContext[Any, Any], params: PaginatedRequestParams | None
    ) -> ListResourcesResult:
        adapter = ctx.lifespan_context
        cursor = params.cursor if params is not None else None
        # protocol_errors maps a bad cursor (ValueError) to -32602 and Docker loss, SDK guard or readiness failures
        # to -32603 with readable text; anything else still propagates as a crash.
        async with protocol_errors(adapter):
            items: list[Resource] = []
            if listed:
                infos = await containers_service.list_containers(adapter, run_id=getattr(adapter, "run_id", None))
                for info in infos:
                    for entry, template, description in listed:
                        items.append(
                            Resource(
                                uri=template.expand({"name": info.name}),
                                name=f"{entry.name}:{info.name}",
                                title=f"{entry.title}: {info.name}",
                                description=description,
                                mime_type=entry.mime_type,
                            )
                        )
            page_items, next_cursor = paging.page(items, cursor, _page_size(adapter))
        return ListResourcesResult(resources=page_items, next_cursor=next_cursor)

    server._lowlevel_server.add_request_handler("resources/list", PaginatedRequestParams, list_resources)

    _install_read_relabel(server, entries)


def _install_read_relabel(server: MCPServer[Any], entries: Sequence[ResourceEntry]) -> None:
    """Wrap resources/read so text reads of application/octet-stream templates are labelled text/plain.

    The SDK stamps every read with the single mime_type of its template, so a template that returns UTF-8 text for
    some files and bytes for others would label its text application/octet-stream. Blob contents keep that type;
    reads of any other template pass through untouched. Nothing is installed when no such template is enabled.
    """
    templates = [UriTemplate.parse(entry.uri_template) for entry in entries if entry.mime_type == _OCTET_STREAM]
    if not templates:
        return
    original = server._lowlevel_server.get_request_handler("resources/read")
    if original is None or original.params_type is not ReadResourceRequestParams:
        raise RuntimeError("The MCP SDK registered no resources/read handler of the expected shape to wrap.")
    original_handler = original.handler

    async def read_resource(ctx: ServerRequestContext[Any, Any], params: ReadResourceRequestParams) -> Any:
        result = await original_handler(ctx, params)
        if not isinstance(result, ReadResourceResult):
            return result
        uri = str(params.uri)
        if not any(template.match(uri) is not None for template in templates):
            return result
        contents = [
            content.model_copy(update={"mime_type": _TEXT_PLAIN})
            if isinstance(content, TextResourceContents)
            else content
            for content in result.contents
        ]
        return result.model_copy(update={"contents": contents})

    server._lowlevel_server.add_request_handler("resources/read", ReadResourceRequestParams, read_resource)
