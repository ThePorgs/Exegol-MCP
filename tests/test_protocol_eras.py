"""Both protocol eras over one server (PROTO-01/PROTO-02).

A 2025-11-25 client (the initialize handshake, mode="legacy") and a 2026-07-28 client (the stateless per-request
envelope) connect to the same build_server(...) and must see the same tool surface and be able to call every tool.
The edges covered: the two eras side by side on one server object (adjacency), an empty Exegol host (empty), and
the order and full definitions of the tool list (ordering).
"""

from typing import Any

import pytest
from mcp import Client
from mcp_types import Tool
from mcp_types.version import LATEST_HANDSHAKE_VERSION, LATEST_MODERN_VERSION

from exegol_mcp.app import build_server
from tests.fakes import MINIMAL_TOOL_ARGUMENTS, V2_TOOL_NAMES, FakeExegolAdapter, catalog_router

# mode passed to the Client -> the protocol version the session must negotiate.
ERAS: dict[str, str] = {"legacy": "2025-11-25", "2026-07-28": "2026-07-28"}

# Tools whose result is a bare list wrapped as {"result": [...]}, not a scalar.
STRUCTURED_LIST_INFO_TOOLS = {"list_containers", "list_images"}
# The paginated listings return a page object {"items": [...], "next_cursor": ...}: the catalog listings (EXEC-10,
# D-23) and the container directory listing (FILE-03, D-19).
PAGED_LIST_TOOLS = {"list_installed_tools", "list_offline_resources", "list_directory"}


def _adapter() -> FakeExegolAdapter:
    # seed_contract_job makes SEEDED_JOB_ID resolvable so the "call every tool" suite can call get_job/stop_job;
    # seed_managed_container gives remove_container a managed container to remove.
    return FakeExegolAdapter(exec_router=catalog_router, seed_contract_job=True, seed_managed_container=True)


def _definition(tool: Tool) -> dict[str, Any]:
    """The parts of a tool definition a client relies on, comparable across eras."""
    return tool.model_dump(mode="json", exclude_none=True)


def test_the_eras_under_test_are_the_sdk_latest_handshake_and_modern_versions() -> None:
    assert ERAS["legacy"] == LATEST_HANDSHAKE_VERSION == "2025-11-25"
    assert ERAS["2026-07-28"] == LATEST_MODERN_VERSION == "2026-07-28"


@pytest.mark.parametrize("mode", list(ERAS))
async def test_each_era_negotiates_its_version_and_lists_the_full_v2_surface(mode: str) -> None:
    async with Client(build_server(None, _adapter()), mode=mode) as client:
        assert client.protocol_version == ERAS[mode]
        names = [tool.name for tool in (await client.list_tools()).tools]
    assert len(names) == len(set(names)), names
    assert set(names) == V2_TOOL_NAMES


@pytest.mark.parametrize("mode", list(ERAS))
async def test_each_era_can_call_every_tool(mode: str) -> None:
    adapter = _adapter()
    async with Client(build_server(None, adapter), mode=mode) as client:
        listed = {tool.name for tool in (await client.list_tools()).tools}
        assert listed == set(MINIMAL_TOOL_ARGUMENTS), "every listed tool needs a canned call"
        for name, arguments in MINIMAL_TOOL_ARGUMENTS.items():
            result = await client.call_tool(name, arguments)
            assert not result.is_error, (mode, name, result.content)
            assert result.structured_content is not None, (mode, name)
            if name in STRUCTURED_LIST_INFO_TOOLS:
                assert isinstance(result.structured_content.get("result"), list), (mode, name)
            if name in PAGED_LIST_TOOLS:
                assert isinstance(result.structured_content.get("items"), list), (mode, name)
                assert "next_cursor" in result.structured_content, (mode, name)
    # The calls really reached the adapter: a stop, a download, and the exec-backed tools all ran.
    assert ("stop", "exegol-lab2") in adapter.state_changes
    assert adapter.downloads == [("full", "latest")]
    assert "id" in adapter.exec_commands


async def test_both_eras_side_by_side_on_one_server_see_identical_tool_definitions() -> None:
    """Adjacency + ordering: one server object, both eras connected at once, same tools in the same order."""
    server = build_server(None, _adapter())
    async with Client(server, mode="legacy") as legacy, Client(server, mode="2026-07-28") as modern:
        legacy_tools = (await legacy.list_tools()).tools
        modern_tools = (await modern.list_tools()).tools
        # Calls interleaved across the two sessions answer independently.
        legacy_result = await legacy.call_tool("list_containers", {})
        modern_result = await modern.call_tool("list_containers", {})

    assert [tool.name for tool in legacy_tools] == [tool.name for tool in modern_tools]
    assert [_definition(tool) for tool in legacy_tools] == [_definition(tool) for tool in modern_tools]
    assert legacy_result.structured_content == modern_result.structured_content


async def test_the_tool_order_is_stable_across_sessions_and_eras() -> None:
    orders = []
    for mode in [*ERAS, *ERAS]:
        async with Client(build_server(None, _adapter()), mode=mode) as client:
            orders.append([tool.name for tool in (await client.list_tools()).tools])
    assert all(order == orders[0] for order in orders), orders


@pytest.mark.parametrize("mode", list(ERAS))
async def test_an_empty_exegol_host_still_exposes_the_full_surface_and_empty_lists(mode: str) -> None:
    """Empty edge: no containers and no images change nothing about the surface; the lists come back empty."""
    empty = FakeExegolAdapter(containers=[], images=[])
    async with Client(build_server(None, empty), mode=mode) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        containers = await client.call_tool("list_containers", {})
        images = await client.call_tool("list_images", {"scope": "all"})
    assert names == V2_TOOL_NAMES
    assert containers.structured_content == {"result": []}
    assert images.structured_content == {"result": []}
