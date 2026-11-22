"""MCP resources (RES-01..RES-04). Docker-free, over the in-memory MCP Client, in both protocol eras.

Resources are read-only views a client reads by URI. Each RESOURCE_REGISTRY entry belongs to one tool group and is
gated at registration exactly like tools (D-04, remapped at UAT G-05-1a): container info and the installed-tools catalog
follow ``containers`` and workspace files follow ``files``. A resource of a disabled group is never registered, so it
is absent from resources/templates/list and resources/list, and a read fails as not found (-32602).
"""

import base64
import json
import re
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError
from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS, BlobResourceContents, TextResourceContents

from exegol_mcp.adapter.errors import HeadlessPromptError, PatchPointError, UnknownParameterError
from exegol_mcp.app import build_server
from exegol_mcp.models.config import TOOL_GROUPS, OperatorConfig
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.resources import (
    CONTAINER_INFO_URI,
    INSTALLED_TOOLS_URI,
    RESOURCE_REGISTRY,
    WORKSPACE_FILE_URI,
    ResourceEntry,
)
from exegol_mcp.services import containers as containers_service
from exegol_mcp.services import paging
from exegol_mcp.tools import TOOL_REGISTRY
from exegol_mcp.tools.errors import protocol_errors, resource_errors, tool_errors
from tests.fakes import (
    DEFAULT_CONTAINERS,
    SEEDED_MANAGED_CONTAINER,
    SEEDED_MANAGED_RUN_ID,
    FakeExegolAdapter,
    catalog_router,
)

# mode passed to the Client -> the protocol version the session must negotiate (as in tests/test_protocol_eras.py).
ERAS: dict[str, str] = {"legacy": "2025-11-25", "2026-07-28": "2026-07-28"}

# The mapping of every resource URI template to the tool group that gates it (D-04, remapped at UAT G-05-1a: info and
# the tools catalog follow containers, workspace files follow files), spelled out here so this table, not the
# implementation constant, is the lock.
EXPECTED_RESOURCE_GROUPS: dict[str, str] = {
    "exegol://containers/{name}/info": "containers",
    "exegol://containers/{name}/tools": "containers",
    "exegol://containers/{name}/workspace/{+path}": "files",
}

# Planning traceability IDs that must never leak into a client-visible description.
_PLANNING_ID = re.compile(r"\b(D|RES|QUAL|SEC|EXEC|LIFE|FILE|CONF)-\d")

# The stopped-container wording of every resource that needs a running container, spelled out as the lock: a
# resource read never starts anything, so it names start_container only.
_NOT_RUNNING = (
    "Container '{name}' is not running, and reading a resource never starts it. Start it with start_container, then "
    "read the resource again."
)

_INFO_URI = "exegol://containers/exegol-lab/info"
_TOOLS_URI = "exegol://containers/exegol-lab/tools"
_READINESS_DIAGNOSIS = "READINESS: docker gone"


def _server(
    disabled: list[Any] | None = None,
    config: OperatorConfig | None = None,
    adapter: FakeExegolAdapter | None = None,
) -> MCPServer[Any]:
    if config is None:
        # Validated the same way config.yml is, so the in-memory and file-loaded paths share one model.
        config = OperatorConfig.model_validate({"tool_groups": {"disabled": disabled or []}})
    if adapter is None:
        adapter = FakeExegolAdapter(config=config)
    else:
        adapter.config = config
    return build_server(None, adapter)


def _paged_config(page_size: int, disabled: list[str] | None = None) -> OperatorConfig:
    return OperatorConfig.model_validate(
        {"execution": {"listing_page_size": page_size}, "tool_groups": {"disabled": disabled or []}}
    )


def _named_containers(*names: str) -> list[Any]:
    return [DEFAULT_CONTAINERS[0].model_copy(update={"name": name}) for name in names]


async def _walk(client: Client, cursor: str | None = None) -> list[list[str]]:
    """Every page of resources/list as lists of URIs, following next_cursor to the end."""
    pages: list[list[str]] = []
    while True:
        result = await client.list_resources(cursor=cursor, cache_mode="bypass")
        pages.append([str(resource.uri) for resource in result.resources])
        cursor = result.next_cursor
        if cursor is None:
            return pages


def _stopped_adapter() -> FakeExegolAdapter:
    stopped = DEFAULT_CONTAINERS[0].model_copy(update={"name": "stopped-lab", "status": "Stopped"})
    return FakeExegolAdapter(containers=[stopped])


class _DaemonGoneAdapter(FakeExegolAdapter):
    """Every container port method fails as if the Docker daemon died after startup; diagnose() says so (D-08)."""

    async def list_containers(self) -> list[Any]:
        raise ConnectionError("daemon gone")

    async def get_container_state(self, name: str) -> Any:
        raise ConnectionError("daemon gone")

    async def diagnose(self) -> str | None:
        return _READINESS_DIAGNOSIS


class _Diagnoser:
    """A ReadinessDiagnoser returning a fixed answer."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer

    async def diagnose(self) -> str | None:
        return self.answer


# --- The registry and its D-04 group lock ---------------------------------------------------------------------------


def test_registry_entries_match_the_resource_group_lock() -> None:
    assert RESOURCE_REGISTRY, "the registry must hold at least the container info resource"
    for entry in RESOURCE_REGISTRY:
        assert isinstance(entry, ResourceEntry)
        assert entry.group in TOOL_GROUPS, entry
        assert EXPECTED_RESOURCE_GROUPS.get(entry.uri_template) == entry.group, entry


def test_registry_holds_the_container_info_entry() -> None:
    (entry,) = [e for e in RESOURCE_REGISTRY if e.uri_template == CONTAINER_INFO_URI]
    assert CONTAINER_INFO_URI == "exegol://containers/{name}/info"
    assert entry.name == "container_info"
    assert entry.mime_type == "application/json"
    assert entry.group == "containers"
    assert entry.listed is True


# --- The info template ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(ERAS))
async def test_template_is_listed_with_a_clean_description(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
    (info,) = [t for t in templates if t.uri_template == CONTAINER_INFO_URI]
    assert info.name == "container_info"
    assert info.mime_type == "application/json"
    assert info.description
    assert not _PLANNING_ID.search(info.description), info.description


# --- Reading container info -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(ERAS))
async def test_info_read_returns_the_list_containers_entry(mode: str) -> None:
    adapter = FakeExegolAdapter()
    async with Client(_server(adapter=adapter), mode=mode) as client:
        result = await client.read_resource(_INFO_URI)
        expected = await containers_service.list_containers(adapter, run_id=adapter.run_id)
    (content,) = result.contents
    assert isinstance(content, TextResourceContents)
    assert content.mime_type == "application/json"
    (want,) = [info for info in expected if info.name == "exegol-lab"]
    assert json.loads(content.text) == want.model_dump(mode="json")


@pytest.mark.parametrize("mode", list(ERAS))
async def test_info_read_carries_the_managed_fields(mode: str) -> None:
    adapter = FakeExegolAdapter(seed_managed_container=True)
    async with Client(_server(adapter=adapter), mode=mode) as client:
        result = await client.read_resource(f"exegol://containers/{SEEDED_MANAGED_CONTAINER}/info")
    (content,) = result.contents
    assert isinstance(content, TextResourceContents)
    data = json.loads(content.text)
    assert data["managed"] is True
    assert data["run_id"] == SEEDED_MANAGED_RUN_ID
    assert data["this_run"] is False


@pytest.mark.parametrize("mode", list(ERAS))
async def test_info_read_of_a_stopped_container_never_starts_it(mode: str) -> None:
    adapter = _stopped_adapter()
    async with Client(_server(adapter=adapter), mode=mode) as client:
        result = await client.read_resource("exegol://containers/stopped-lab/info")
    (content,) = result.contents
    assert isinstance(content, TextResourceContents)
    assert json.loads(content.text)["status"] == "Stopped"
    assert adapter.state_changes == []
    assert adapter.exec_commands == []


@pytest.mark.parametrize("mode", list(ERAS))
async def test_info_read_of_an_unknown_container_is_invalid_params(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource("exegol://containers/nope/info")
    assert raised.value.code == INVALID_PARAMS
    assert "nope" in str(raised.value)
    assert "list_containers" in str(raised.value)


@pytest.mark.parametrize("mode", list(ERAS))
async def test_info_read_reports_the_readiness_diagnosis(mode: str) -> None:
    async with Client(_server(adapter=_DaemonGoneAdapter()), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_INFO_URI)
    assert raised.value.code == INTERNAL_ERROR
    assert _READINESS_DIAGNOSIS in str(raised.value)


# --- Gating by the containers group (D-04) --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_containers_hides_the_info_template_and_its_reads(mode: str) -> None:
    async with Client(_server(["containers"]), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_INFO_URI)
        tools = [tool.name for tool in (await client.list_tools()).tools]
    assert CONTAINER_INFO_URI not in [t.uri_template for t in templates]
    assert INSTALLED_TOOLS_URI not in [t.uri_template for t in templates]
    assert raised.value.code == INVALID_PARAMS
    assert tools == [entry.fn.__name__ for entry in TOOL_REGISTRY if entry.group != "containers"]


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_every_other_group_keeps_the_info_resource(mode: str) -> None:
    others = [group for group in TOOL_GROUPS if group != "containers"]
    async with Client(_server(others), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
        result = await client.read_resource(_INFO_URI)
    assert CONTAINER_INFO_URI in [t.uri_template for t in templates]
    assert len(result.contents) == 1


# --- The boundary: one ladder, three raised-type families ------------------------------------------------------------


async def test_boundary_resource_errors_maps_bad_input_to_not_found() -> None:
    original = ValueError("Container 'x' not found.")
    with pytest.raises(ResourceNotFoundError) as raised:
        async with resource_errors():
            raise original
    assert str(raised.value) == str(original)
    assert raised.value.__cause__ is original


async def test_boundary_resource_errors_maps_a_bad_state_to_resource_error() -> None:
    original = RuntimeError("Container 'x' is not running")
    with pytest.raises(ResourceError) as raised:
        async with resource_errors():
            raise original
    assert type(raised.value) is ResourceError
    assert str(raised.value) == str(original)
    assert raised.value.__cause__ is original


async def test_boundary_resource_errors_maps_sdk_guard_errors() -> None:
    for guard in (UnknownParameterError("p"), HeadlessPromptError("prompt"), PatchPointError("point")):
        with pytest.raises(ResourceError) as raised:
            async with resource_errors(_Diagnoser(None)):
                raise guard
        assert type(raised.value) is ResourceError
        assert str(raised.value) == str(guard)
        assert raised.value.__cause__ is guard


async def test_boundary_resource_errors_prefers_the_diagnosis() -> None:
    original = RuntimeError("low-level text")
    with pytest.raises(ResourceError) as raised:
        async with resource_errors(_Diagnoser(_READINESS_DIAGNOSIS)):
            raise original
    assert str(raised.value) == _READINESS_DIAGNOSIS
    assert raised.value.__cause__ is original


async def test_boundary_resource_errors_passes_resource_errors_and_cancellation_through() -> None:
    already = ResourceError("already mapped")
    with pytest.raises(ResourceError) as raised:
        async with resource_errors(_Diagnoser(_READINESS_DIAGNOSIS)):
            raise already
    assert raised.value is already
    cancelled = anyio.get_cancelled_exc_class()
    with pytest.raises(cancelled):
        async with resource_errors(_Diagnoser(_READINESS_DIAGNOSIS)):
            raise cancelled()
    with pytest.raises(KeyError):
        async with resource_errors():
            raise KeyError("crash")


async def test_boundary_protocol_errors_maps_to_json_rpc_codes() -> None:
    with pytest.raises(MCPError) as bad_input:
        async with protocol_errors():
            raise ValueError("Invalid pagination cursor '!!!'.")
    assert bad_input.value.code == INVALID_PARAMS
    assert "Invalid pagination cursor" in str(bad_input.value)
    with pytest.raises(MCPError) as bad_state:
        async with protocol_errors(_Diagnoser(_READINESS_DIAGNOSIS)):
            raise ConnectionError("daemon gone")
    assert bad_state.value.code == INTERNAL_ERROR
    assert str(bad_state.value) == _READINESS_DIAGNOSIS
    already = MCPError(code=INVALID_PARAMS, message="kept")
    with pytest.raises(MCPError) as passed:
        async with protocol_errors():
            raise already
    assert passed.value is already


def test_boundary_factories_keep_one_context_parameter() -> None:
    import inspect

    for factory in (tool_errors, resource_errors, protocol_errors):
        assert list(inspect.signature(factory).parameters) == ["context"], factory


# --- resources/list: live, ordered, paged (D-03) -------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(ERAS))
async def test_list_returns_one_info_resource_per_container_in_order(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        result = await client.list_resources()
    assert result.next_cursor is None
    assert [str(r.uri) for r in result.resources] == [
        "exegol://containers/exegol-lab/info",
        "exegol://containers/exegol-lab/tools",
        "exegol://containers/exegol-lab2/info",
        "exegol://containers/exegol-lab2/tools",
    ]
    first = result.resources[0]
    assert first.name == "container_info:exegol-lab"
    assert first.title == "Exegol container info: exegol-lab"
    assert first.mime_type == "application/json"
    assert first.description
    assert not _PLANNING_ID.search(first.description), first.description


@pytest.mark.parametrize("mode", list(ERAS))
async def test_list_pages_walk_rebuilds_the_full_ordered_list(mode: str) -> None:
    adapter = FakeExegolAdapter(containers=_named_containers("a", "b", "c"))
    async with Client(_server(config=_paged_config(1), adapter=adapter), mode=mode) as client:
        pages = await _walk(client)
    assert pages == [
        ["exegol://containers/a/info"],
        ["exegol://containers/a/tools"],
        ["exegol://containers/b/info"],
        ["exegol://containers/b/tools"],
        ["exegol://containers/c/info"],
        ["exegol://containers/c/tools"],
    ]


@pytest.mark.parametrize("mode", list(ERAS))
async def test_list_exact_multiple_of_the_page_size_has_no_trailing_empty_page(mode: str) -> None:
    # Four containers list eight resources (info and tools each), so a page size of 4 fills exactly two pages.
    adapter = FakeExegolAdapter(containers=_named_containers("a", "b", "c", "d"))
    async with Client(_server(config=_paged_config(4), adapter=adapter), mode=mode) as client:
        pages = await _walk(client)
    assert [len(page) for page in pages] == [4, 4]


@pytest.mark.parametrize("mode", list(ERAS))
@pytest.mark.parametrize("cursor", ["!!!", paging.encode_cursor(99)])
async def test_list_bad_cursor_is_invalid_params(mode: str, cursor: str) -> None:
    async with Client(_server(), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.list_resources(cursor=cursor, cache_mode="bypass")
    assert raised.value.code == INVALID_PARAMS
    assert "cursor" in str(raised.value)


@pytest.mark.parametrize("mode", list(ERAS))
async def test_list_on_an_empty_host_is_empty(mode: str) -> None:
    async with Client(_server(adapter=FakeExegolAdapter(containers=[])), mode=mode) as client:
        result = await client.list_resources()
    assert result.resources == []
    assert result.next_cursor is None


@pytest.mark.parametrize("mode", list(ERAS))
async def test_list_with_every_listed_group_disabled_is_empty(mode: str) -> None:
    async with Client(_server(["containers"]), mode=mode) as client:
        result = await client.list_resources()
    assert result.resources == []
    assert result.next_cursor is None


@pytest.mark.parametrize("mode", list(ERAS))
async def test_list_is_live_and_sees_a_container_created_after_startup(mode: str) -> None:
    adapter = FakeExegolAdapter()
    async with Client(_server(adapter=adapter), mode=mode) as client:
        before = await client.list_resources()
        late = _named_containers("late-lab")[0]
        adapter.containers.append(late)
        adapter.running[late.name] = True
        after = await client.list_resources(cache_mode="bypass")
    assert "exegol://containers/late-lab/info" not in [str(r.uri) for r in before.resources]
    assert [str(r.uri) for r in after.resources][-2:] == [
        "exegol://containers/late-lab/info",
        "exegol://containers/late-lab/tools",
    ]


@pytest.mark.parametrize("mode", list(ERAS))
async def test_list_reports_the_readiness_diagnosis(mode: str) -> None:
    async with Client(_server(adapter=_DaemonGoneAdapter()), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.list_resources(cache_mode="bypass")
    assert raised.value.code == INTERNAL_ERROR
    assert str(raised.value) == _READINESS_DIAGNOSIS


async def test_list_is_identical_in_both_eras() -> None:
    listings = []
    for mode in ERAS:
        adapter = FakeExegolAdapter(containers=_named_containers("a", "b", "c"))
        async with Client(_server(config=_paged_config(3), adapter=adapter), mode=mode) as client:
            first = await client.list_resources()
            second = await client.list_resources(cursor=first.next_cursor)
        listings.append(
            [r.model_dump(mode="json", exclude_none=True) for r in [*first.resources, *second.resources]]
        )
    assert listings[0] == listings[1]
    assert len(listings[0]) == 6


# --- The installed-tools catalog resource (RES-02) -------------------------------------------------------------------


_CATALOG_HEADER = "Tool,Version,Link,Description\n"


def _catalog_adapter(csv_text: str, **kwargs: Any) -> FakeExegolAdapter:
    def router(command: str) -> ExecResult | None:
        if "installed_tools.csv" in command:
            return ExecResult(stdout=csv_text, stderr="", exit_code=0)
        return None

    return FakeExegolAdapter(exec_router=router, **kwargs)


def _large_catalog(count: int) -> str:
    return _CATALOG_HEADER + "".join(f"tool{i},1.{i},https://example.com/tool{i},desc{i}\n" for i in range(count))


async def _read_tools(client: Client, uri: str = _TOOLS_URI) -> Any:
    result = await client.read_resource(uri)
    (content,) = result.contents
    assert isinstance(content, TextResourceContents)
    assert content.mime_type == "application/json"
    return json.loads(content.text)


def test_registry_holds_the_installed_tools_entry() -> None:
    (entry,) = [e for e in RESOURCE_REGISTRY if e.uri_template == INSTALLED_TOOLS_URI]
    assert INSTALLED_TOOLS_URI == "exegol://containers/{name}/tools"
    assert entry.name == "installed_tools"
    assert entry.mime_type == "application/json"
    assert entry.group == "containers"
    assert entry.listed is True


@pytest.mark.parametrize("mode", list(ERAS))
async def test_tools_template_is_listed_with_a_clean_description(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
    (tools,) = [t for t in templates if t.uri_template == INSTALLED_TOOLS_URI]
    assert tools.name == "installed_tools"
    assert tools.mime_type == "application/json"
    assert tools.description
    assert "list_installed_tools" in tools.description
    assert not _PLANNING_ID.search(tools.description), tools.description


@pytest.mark.parametrize("mode", list(ERAS))
async def test_tools_read_returns_the_whole_catalog_as_json(mode: str) -> None:
    adapter = FakeExegolAdapter(exec_router=catalog_router)
    async with Client(_server(adapter=adapter), mode=mode) as client:
        data = await _read_tools(client)
    assert data == {
        "tools": [{"name": "nmap", "version": "7.94", "link": "https://nmap.org", "description": "Network scanner"}]
    }


@pytest.mark.parametrize("mode", list(ERAS))
async def test_tools_read_is_never_paged_and_equals_every_tool_page(mode: str) -> None:
    adapter = _catalog_adapter(_large_catalog(450))
    async with Client(_server(config=_paged_config(200), adapter=adapter), mode=mode) as client:
        data = await _read_tools(client)
        paged: list[Any] = []
        cursor: str | None = None
        while True:
            arguments: dict[str, Any] = {"container_name": "exegol-lab"}
            if cursor is not None:
                arguments["cursor"] = cursor
            page = await client.call_tool("list_installed_tools", arguments)
            assert page.structured_content is not None
            paged += page.structured_content["items"]
            cursor = page.structured_content["next_cursor"]
            if cursor is None:
                break
    assert len(data["tools"]) == 450
    assert data["tools"] == paged
    assert [tool["name"] for tool in data["tools"]] == [f"tool{i}" for i in range(450)]


@pytest.mark.parametrize("mode", list(ERAS))
@pytest.mark.parametrize("csv_text", [_CATALOG_HEADER, ""], ids=["header-only", "empty-file"])
async def test_tools_read_of_an_empty_catalog_is_an_empty_list(mode: str, csv_text: str) -> None:
    async with Client(_server(adapter=_catalog_adapter(csv_text)), mode=mode) as client:
        data = await _read_tools(client)
    assert data == {"tools": []}


@pytest.mark.parametrize("mode", list(ERAS))
async def test_tools_read_keeps_short_rows_and_duplicates_in_file_order(mode: str) -> None:
    csv_text = _CATALOG_HEADER + "nmap,7.94,https://nmap.org,scan\nshort\nnmap,7.95,https://nmap.org,scan\n"
    async with Client(_server(adapter=_catalog_adapter(csv_text)), mode=mode) as client:
        data = await _read_tools(client)
    assert [(t["name"], t["version"]) for t in data["tools"]] == [("nmap", "7.94"), ("short", None), ("nmap", "7.95")]


@pytest.mark.parametrize("mode", list(ERAS))
async def test_tools_read_of_a_stopped_container_never_starts_it(mode: str) -> None:
    adapter = _stopped_adapter()
    adapter.exec_router = catalog_router
    async with Client(_server(adapter=adapter), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource("exegol://containers/stopped-lab/tools")
    assert raised.value.code == INTERNAL_ERROR
    assert str(raised.value) == _NOT_RUNNING.format(name="stopped-lab")
    assert adapter.state_changes == []
    assert adapter.exec_commands == []


@pytest.mark.parametrize("mode", list(ERAS))
async def test_tools_read_of_an_unknown_container_is_invalid_params(mode: str) -> None:
    async with Client(_server(adapter=FakeExegolAdapter(exec_router=catalog_router)), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource("exegol://containers/nope/tools")
    assert raised.value.code == INVALID_PARAMS
    assert "nope" in str(raised.value)
    assert "list_containers" in str(raised.value)


@pytest.mark.parametrize("mode", list(ERAS))
async def test_tools_list_entries_are_named_per_container(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        result = await client.list_resources()
    tools = [r for r in result.resources if str(r.uri).endswith("/tools")]
    assert [str(r.uri) for r in tools] == [
        "exegol://containers/exegol-lab/tools",
        "exegol://containers/exegol-lab2/tools",
    ]
    assert [r.name for r in tools] == ["installed_tools:exegol-lab", "installed_tools:exegol-lab2"]
    assert all(r.mime_type == "application/json" for r in tools)
    assert all(r.description and not _PLANNING_ID.search(r.description) for r in tools)


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_execution_keeps_the_tools_template_its_entries_and_reads(mode: str) -> None:
    # The tools catalog follows containers, not execution: get_tool_help goes while the catalog resource stays.
    adapter = FakeExegolAdapter(exec_router=catalog_router)
    async with Client(_server(["execution"], adapter=adapter), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
        listed = (await client.list_resources()).resources
        data = await _read_tools(client)
        tools = [tool.name for tool in (await client.list_tools()).tools]
    assert INSTALLED_TOOLS_URI in [t.uri_template for t in templates]
    assert "exegol://containers/exegol-lab/tools" in [str(r.uri) for r in listed]
    assert "exegol://containers/exegol-lab2/tools" in [str(r.uri) for r in listed]
    assert [tool["name"] for tool in data["tools"]] == ["nmap"]
    assert "get_tool_help" not in tools
    assert "list_installed_tools" in tools


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_containers_hides_the_tools_template_its_entries_and_reads(mode: str) -> None:
    adapter = FakeExegolAdapter(exec_router=catalog_router)
    async with Client(_server(["containers"], adapter=adapter), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
        listed = (await client.list_resources()).resources
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_TOOLS_URI)
    assert INSTALLED_TOOLS_URI not in [t.uri_template for t in templates]
    assert not [r for r in listed if str(r.uri).endswith("/tools")]
    assert raised.value.code == INVALID_PARAMS
    assert adapter.exec_commands == []


async def test_tools_read_is_identical_in_both_eras() -> None:
    reads = []
    for mode in ERAS:
        async with Client(_server(adapter=FakeExegolAdapter(exec_router=catalog_router)), mode=mode) as client:
            reads.append(await _read_tools(client))
    assert reads[0] == reads[1]


# --- Workspace files (RES-01) ----------------------------------------------------------------------------------------


_WORKSPACE = "exegol://containers/exegol-lab/workspace/"
_WORKSPACE_FILES: dict[str, bytes] = {
    "/workspace/notes.txt": b"hello\n",
    "/workspace/sub/a.txt": b"A",
    "/workspace/b.txt": b"B",
    "/workspace/bin.dat": b"\xff\x00\x01",
    "/workspace/eight.txt": b"12345678",
    "/workspace/nine.txt": b"123456789",
    "/etc/shadow": b"root:x",
}


class _SpyFilesAdapter(FakeExegolAdapter):
    """Records every read_file call, and can hold a read for a while to make concurrent reads overlap."""

    def __init__(self, delays: dict[str, float] | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("files", {"exegol-lab": dict(_WORKSPACE_FILES)})
        super().__init__(**kwargs)
        self.read_calls: list[tuple[str, str]] = []
        self.delays = delays or {}

    async def read_file(self, name: str, path: str, max_bytes: int, timeout_s: float) -> Any:
        self.read_calls.append((name, path))
        await anyio.sleep(self.delays.get(path, 0))
        return await super().read_file(name, path, max_bytes, timeout_s)


def _files_config(max_bytes: int | None = None, disabled: list[str] | None = None) -> OperatorConfig:
    data: dict[str, Any] = {"tool_groups": {"disabled": disabled or []}}
    if max_bytes is not None:
        data["files"] = {"max_bytes": max_bytes}
    return OperatorConfig.model_validate(data)


async def _read_text(client: Client, uri: str) -> TextResourceContents:
    result = await client.read_resource(uri)
    (content,) = result.contents
    assert isinstance(content, TextResourceContents), content
    return content


def test_registry_holds_the_workspace_entry_last_and_unlisted() -> None:
    assert WORKSPACE_FILE_URI == "exegol://containers/{name}/workspace/{+path}"
    assert [(e.uri_template, e.group, e.listed) for e in RESOURCE_REGISTRY] == [
        (CONTAINER_INFO_URI, "containers", True),
        (INSTALLED_TOOLS_URI, "containers", True),
        (WORKSPACE_FILE_URI, "files", False),
    ]
    (entry,) = [e for e in RESOURCE_REGISTRY if e.uri_template == WORKSPACE_FILE_URI]
    assert entry.name == "workspace_file"
    assert entry.mime_type == "application/octet-stream"


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_templates_list_holds_the_three_templates_in_order(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
    assert [(t.uri_template, t.mime_type) for t in templates] == [
        (CONTAINER_INFO_URI, "application/json"),
        (INSTALLED_TOOLS_URI, "application/json"),
        (WORKSPACE_FILE_URI, "application/octet-stream"),
    ]
    workspace = templates[2]
    assert workspace.name == "workspace_file"
    assert workspace.description
    assert "/workspace" in workspace.description
    assert "read_file" in workspace.description
    assert "list_directory" in workspace.description
    assert not _PLANNING_ID.search(workspace.description), workspace.description


@pytest.mark.parametrize("mode", list(ERAS))
@pytest.mark.parametrize(
    ("path", "expected"),
    [("notes.txt", "hello\n"), ("sub/a.txt", "A"), ("a/../b.txt", "B")],
    ids=["file", "subdirectory", "dot-dot-inside"],
)
async def test_workspace_text_read_is_text_plain(mode: str, path: str, expected: str) -> None:
    adapter = _SpyFilesAdapter()
    async with Client(_server(adapter=adapter), mode=mode) as client:
        content = await _read_text(client, _WORKSPACE + path)
    assert content.text == expected
    assert content.mime_type == "text/plain"
    assert str(content.uri) == _WORKSPACE + path


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_read_goes_through_the_container_under_workspace(mode: str) -> None:
    adapter = _SpyFilesAdapter()
    async with Client(_server(adapter=adapter), mode=mode) as client:
        await _read_text(client, _WORKSPACE + "a/../b.txt")
    assert adapter.read_calls == [("exegol-lab", "/workspace/b.txt")]


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_binary_read_is_an_octet_stream_blob(mode: str) -> None:
    async with Client(_server(adapter=_SpyFilesAdapter()), mode=mode) as client:
        result = await client.read_resource(_WORKSPACE + "bin.dat")
    (content,) = result.contents
    assert isinstance(content, BlobResourceContents), content
    assert content.mime_type == "application/octet-stream"
    assert base64.b64decode(content.blob) == b"\xff\x00\x01"


@pytest.mark.parametrize("mode", list(ERAS))
@pytest.mark.parametrize(
    "path",
    ["../etc/shadow", "%2E%2E/etc/shadow", "/etc/shadow", "a%00b", "sub/../../etc/shadow"],
    ids=["dot-dot", "encoded-dot-dot", "absolute", "nul", "nested-escape"],
)
async def test_workspace_escapes_are_refused_before_any_read(mode: str, path: str) -> None:
    adapter = _SpyFilesAdapter()
    async with Client(_server(adapter=adapter), mode=mode) as client:
        # The in-memory client sends a %00 URI as is, so every refusal here is the server's -32602.
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_WORKSPACE + path)
    assert raised.value.code == INVALID_PARAMS
    assert "root:x" not in str(raised.value)
    assert adapter.read_calls == []


@pytest.mark.parametrize(
    "path", ["../etc/shadow", "/etc/shadow", "sub/../../etc/shadow", "../workspace-other/x"]
)
async def test_workspace_lexical_check_refuses_escapes_behind_the_sdk(path: str) -> None:
    from exegol_mcp.resources.files import workspace_file

    adapter = _SpyFilesAdapter()
    ctx: Any = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=adapter))
    with pytest.raises(ResourceNotFoundError) as raised:
        await workspace_file("exegol-lab", path, ctx=ctx)
    assert "/workspace" in str(raised.value)
    assert "read_file" in str(raised.value)
    assert adapter.read_calls == []


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_directory_itself_is_invalid_params(mode: str) -> None:
    async with Client(_server(adapter=_SpyFilesAdapter()), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_WORKSPACE)
    assert raised.value.code == INVALID_PARAMS


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_read_honours_files_max_bytes_without_truncating(mode: str) -> None:
    adapter = _SpyFilesAdapter()
    async with Client(_server(config=_files_config(max_bytes=8), adapter=adapter), mode=mode) as client:
        exact = await _read_text(client, _WORKSPACE + "eight.txt")
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_WORKSPACE + "nine.txt")
    assert exact.text == "12345678"
    assert raised.value.code == INVALID_PARAMS
    assert "files.max_bytes" in str(raised.value)


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_read_of_a_stopped_container_never_starts_it(mode: str) -> None:
    stopped = DEFAULT_CONTAINERS[0].model_copy(update={"name": "stopped-lab", "status": "Stopped"})
    adapter = _SpyFilesAdapter(containers=[stopped], files={"stopped-lab": {"/workspace/notes.txt": b"x"}})
    async with Client(_server(adapter=adapter), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource("exegol://containers/stopped-lab/workspace/notes.txt")
    assert raised.value.code == INTERNAL_ERROR
    assert str(raised.value) == _NOT_RUNNING.format(name="stopped-lab")
    assert adapter.state_changes == []
    assert adapter.read_calls == []


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_read_of_an_unknown_container_is_invalid_params(mode: str) -> None:
    adapter = _SpyFilesAdapter()
    async with Client(_server(adapter=adapter), mode=mode) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource("exegol://containers/nope/workspace/notes.txt")
    assert raised.value.code == INVALID_PARAMS
    assert "list_containers" in str(raised.value)
    assert adapter.read_calls == []


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_files_are_never_listed(mode: str) -> None:
    async with Client(_server(adapter=_SpyFilesAdapter()), mode=mode) as client:
        listed = (await client.list_resources()).resources
    assert listed
    assert not [r for r in listed if "/workspace" in str(r.uri)]


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_files_hides_the_workspace_template_and_its_reads(mode: str) -> None:
    adapter = _SpyFilesAdapter(exec_router=catalog_router)
    async with Client(_server(adapter=adapter, config=_files_config(disabled=["files"])), mode=mode) as client:
        templates = (await client.list_resource_templates()).resource_templates
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_WORKSPACE + "notes.txt")
        info = await _read_text(client, _INFO_URI)
        tools = await _read_text(client, _TOOLS_URI)
    assert [t.uri_template for t in templates] == [CONTAINER_INFO_URI, INSTALLED_TOOLS_URI]
    assert raised.value.code == INVALID_PARAMS
    assert adapter.read_calls == []
    assert info.mime_type == "application/json"
    assert tools.mime_type == "application/json"


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_other_reads_keep_their_json_type(mode: str) -> None:
    async with Client(_server(adapter=_SpyFilesAdapter(exec_router=catalog_router)), mode=mode) as client:
        info = await _read_text(client, _INFO_URI)
        tools = await _read_text(client, _TOOLS_URI)
    assert info.mime_type == "application/json"
    assert tools.mime_type == "application/json"


@pytest.mark.parametrize("mode", list(ERAS))
async def test_workspace_concurrent_reads_each_get_their_own_file(mode: str) -> None:
    # notes.txt is held longer than b.txt, so the two reads overlap and finish in the reverse order.
    adapter = _SpyFilesAdapter(delays={"/workspace/notes.txt": 0.05})
    results: dict[str, str] = {}
    async with Client(_server(adapter=adapter), mode=mode) as client:

        async def read(path: str) -> None:
            results[path] = (await _read_text(client, _WORKSPACE + path)).text

        async with anyio.create_task_group() as group:
            group.start_soon(read, "notes.txt")
            group.start_soon(read, "b.txt")
    assert results == {"notes.txt": "hello\n", "b.txt": "B"}
    assert sorted(adapter.read_calls) == [("exegol-lab", "/workspace/b.txt"), ("exegol-lab", "/workspace/notes.txt")]


async def test_workspace_reads_are_identical_in_both_eras() -> None:
    reads = []
    for mode in ERAS:
        async with Client(_server(adapter=_SpyFilesAdapter()), mode=mode) as client:
            text = await client.read_resource(_WORKSPACE + "notes.txt")
            blob = await client.read_resource(_WORKSPACE + "bin.dat")
        reads.append([c.model_dump(mode="json") for c in [*text.contents, *blob.contents]])
    assert reads[0] == reads[1]


# --- Resource gating per non-container group (G-05-1a) --------------------------------------------------------------


async def _surface(client: Client) -> tuple[list[str], list[str]]:
    """The resources/list URIs and the resources/templates/list URI templates, in order."""
    listed = [str(r.uri) for r in (await client.list_resources(cache_mode="bypass")).resources]
    templates = [t.uri_template for t in (await client.list_resource_templates()).resource_templates]
    return listed, templates


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_images_leaves_every_resource_and_template_as_the_default(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        default = await _surface(client)
    async with Client(_server(["images"]), mode=mode) as client:
        gated = await _surface(client)
    assert gated == default
    assert default[1] == [CONTAINER_INFO_URI, INSTALLED_TOOLS_URI, WORKSPACE_FILE_URI]


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_execution_keeps_both_listed_resources_of_every_container(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        default = await _surface(client)
    adapter = FakeExegolAdapter(exec_router=catalog_router)
    async with Client(_server(["execution"], adapter=adapter), mode=mode) as client:
        listed, templates = await _surface(client)
        data = await _read_tools(client)
        info = await client.read_resource(_INFO_URI)
        tools = [tool.name for tool in (await client.list_tools()).tools]
    assert (listed, templates) == default
    assert listed == [
        "exegol://containers/exegol-lab/info",
        "exegol://containers/exegol-lab/tools",
        "exegol://containers/exegol-lab2/info",
        "exegol://containers/exegol-lab2/tools",
    ]
    assert [tool["name"] for tool in data["tools"]] == ["nmap"]
    assert len(info.contents) == 1
    assert "get_tool_help" not in tools and "exec_remote_command" not in tools


@pytest.mark.parametrize("mode", list(ERAS))
async def test_gating_files_hides_only_the_workspace_template(mode: str) -> None:
    async with Client(_server(), mode=mode) as client:
        default_listed, _ = await _surface(client)
    adapter = _SpyFilesAdapter(exec_router=catalog_router)
    async with Client(_server(["files"], adapter=adapter), mode=mode) as client:
        listed, templates = await _surface(client)
        with pytest.raises(MCPError) as raised:
            await client.read_resource(_WORKSPACE + "notes.txt")
        info = await _read_text(client, _INFO_URI)
        catalog = await _read_tools(client)
    assert templates == [CONTAINER_INFO_URI, INSTALLED_TOOLS_URI]
    assert listed == default_listed
    assert raised.value.code == INVALID_PARAMS
    assert adapter.read_calls == []
    assert info.mime_type == "application/json"
    assert [tool["name"] for tool in catalog["tools"]] == ["nmap"]
