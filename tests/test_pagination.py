"""Opaque-cursor pagination tests for the catalog listings (EXEC-10, D-23). Docker-free.

Covers the pure _page helper (walk/boundary/empty/single/forged-cursor) and the two paged service
functions (large catalog walk, default page size, order preserved, malformed cursor -> ValueError).
Tool-level walks over the in-memory Client live in the second half (Task 2).
"""

import pytest
from mcp import Client

from exegol_mcp.app import build_server
from exegol_mcp.models.catalog import InstalledToolsPage, OfflineResourcesPage
from exegol_mcp.models.config import ExecutionLimits, OperatorConfig
from exegol_mcp.services import catalog as catalog_service
from exegol_mcp.services.catalog import _encode_cursor, _page
from tests.fakes import FakeExegolAdapter


def _large_csv(count: int) -> str:
    header = "name,version,category,description\n"
    rows = "".join(f"tool{i},1.{i},cat{i % 3},desc{i}\n" for i in range(count))
    return header + rows


def _large_find(count: int) -> str:
    return "".join(f"/opt/resources/linux/res{i}\n" for i in range(count))


def _catalog_adapter(csv_text: str = "", find_text: str = "") -> FakeExegolAdapter:
    def router(command: str) -> object:
        if "installed_tools.csv" in command:
            from exegol_mcp.models.exec import ExecResult

            return ExecResult(stdout=csv_text, stderr="", exit_code=0)
        if command.startswith("find /opt/resources/"):
            from exegol_mcp.models.exec import ExecResult

            return ExecResult(stdout=find_text, stderr="", exit_code=0)
        return None

    return FakeExegolAdapter(exec_router=router)  # type: ignore[arg-type]


# --- The pure _page helper ------------------------------------------------------------------------------------


def test_page_walks_a_list_in_order_with_no_duplicate_or_skip() -> None:
    items = list(range(10))
    seen: list[int] = []
    cursor: str | None = None
    while True:
        page, cursor = _page(items, cursor, 4)
        assert len(page) <= 4
        seen += page
        if cursor is None:
            break
    assert seen == items


def test_page_exact_multiple_ends_with_null_cursor_and_no_trailing_empty_page() -> None:
    items = list(range(10))  # exact multiple of page_size 5
    pages: list[list[int]] = []
    cursor: str | None = None
    while True:
        page, cursor = _page(items, cursor, 5)
        pages.append(page)
        if cursor is None:
            break
    assert pages == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # exactly two pages, no third empty page
    assert cursor is None


def test_page_single_item_returns_one_page_with_null_cursor() -> None:
    page, cursor = _page(["only"], None, 200)
    assert page == ["only"]
    assert cursor is None


def test_page_empty_list_returns_one_empty_page_with_null_cursor() -> None:
    page, cursor = _page([], None, 200)
    assert page == []
    assert cursor is None


def test_page_boundary_item_appears_exactly_once() -> None:
    items = list(range(10))
    first, cursor = _page(items, None, 4)
    assert cursor is not None
    second, _ = _page(items, cursor, 4)
    # The first item of page 2 is the next item after the last item of page 1 (no dup, no skip).
    assert first[-1] == 3
    assert second[0] == 4


def test_page_malformed_cursor_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _page(list(range(3)), "!!not-a-cursor!!", 2)


def test_page_forged_out_of_range_offset_raises_value_error() -> None:
    forged = _encode_cursor(10_000)
    with pytest.raises(ValueError):
        _page(list(range(3)), forged, 2)


def test_page_negative_offset_is_rejected() -> None:
    forged = _encode_cursor(-5)
    with pytest.raises(ValueError):
        _page(list(range(3)), forged, 2)


def test_page_rejects_a_non_positive_page_size() -> None:
    with pytest.raises(ValueError):
        _page(list(range(3)), None, 0)


# --- The paged service functions ------------------------------------------------------------------------------


async def _walk_tools(adapter: FakeExegolAdapter, page_size: int) -> list[str]:
    names: list[str] = []
    cursor: str | None = None
    while True:
        page = await catalog_service.list_installed_tools(adapter, "exegol-lab", cursor=cursor, page_size=page_size)
        assert isinstance(page, InstalledToolsPage)
        assert len(page.items) <= page_size
        names += [t.name for t in page.items]
        cursor = page.next_cursor
        if cursor is None:
            break
    return names


async def _walk_resources(adapter: FakeExegolAdapter, page_size: int) -> list[str]:
    paths: list[str] = []
    cursor: str | None = None
    while True:
        page = await catalog_service.list_offline_resources(
            adapter, "exegol-lab", cursor=cursor, page_size=page_size
        )
        assert isinstance(page, OfflineResourcesPage)
        assert len(page.items) <= page_size
        paths += page.items
        cursor = page.next_cursor
        if cursor is None:
            break
    return paths


async def test_list_installed_tools_walks_a_large_catalog_in_order() -> None:
    adapter = _catalog_adapter(csv_text=_large_csv(450))
    names = await _walk_tools(adapter, page_size=50)
    assert names == [f"tool{i}" for i in range(450)]  # full list, in order, no dup/skip


async def test_list_offline_resources_walks_a_large_find_output_in_order() -> None:
    adapter = _catalog_adapter(find_text=_large_find(450))
    paths = await _walk_resources(adapter, page_size=50)
    assert paths == [f"/opt/resources/linux/res{i}" for i in range(450)]


async def test_list_installed_tools_default_page_size_is_two_hundred() -> None:
    adapter = _catalog_adapter(csv_text=_large_csv(450))
    first = await catalog_service.list_installed_tools(adapter, "exegol-lab")
    assert len(first.items) == 200  # built-in default page size (D-23)
    assert first.next_cursor is not None
    # Walking from the default-size first page still reconstructs the whole catalog in order.
    names = [t.name for t in first.items]
    cursor = first.next_cursor
    while cursor is not None:
        page = await catalog_service.list_installed_tools(adapter, "exegol-lab", cursor=cursor)
        names += [t.name for t in page.items]
        cursor = page.next_cursor
    assert names == [f"tool{i}" for i in range(450)]


async def test_list_installed_tools_empty_catalog_is_one_empty_page() -> None:
    adapter = _catalog_adapter(csv_text="")
    page = await catalog_service.list_installed_tools(adapter, "exegol-lab")
    assert page.items == []
    assert page.next_cursor is None


async def test_list_installed_tools_malformed_cursor_raises_value_error() -> None:
    adapter = _catalog_adapter(csv_text=_large_csv(10))
    with pytest.raises(ValueError):
        await catalog_service.list_installed_tools(adapter, "exegol-lab", cursor="!!forged!!")


# --- Tool-level walks over the in-memory Client (Task 2) ------------------------------------------------------


def _client_adapter(page_size: int, csv_count: int = 0, find_count: int = 0) -> FakeExegolAdapter:
    def router(command: str) -> object:
        from exegol_mcp.models.exec import ExecResult

        if "installed_tools.csv" in command:
            return ExecResult(stdout=_large_csv(csv_count), stderr="", exit_code=0)
        if command.startswith("find /opt/resources/"):
            return ExecResult(stdout=_large_find(find_count), stderr="", exit_code=0)
        return None

    config = OperatorConfig(execution=ExecutionLimits(listing_page_size=page_size))
    return FakeExegolAdapter(exec_router=router, config=config)  # type: ignore[arg-type]


async def test_tool_list_installed_tools_walks_pages_disjoint_and_in_order() -> None:
    adapter = _client_adapter(page_size=50, csv_count=120)
    async with Client(build_server(None, adapter)) as client:
        first = await client.call_tool("list_installed_tools", {"container_name": "exegol-lab"})
        assert not first.is_error, first.content
        assert first.structured_content is not None
        assert len(first.structured_content["items"]) == 50  # the operator-configured page size (D-23)
        cursor = first.structured_content["next_cursor"]
        assert cursor is not None

        second = await client.call_tool(
            "list_installed_tools", {"container_name": "exegol-lab", "cursor": cursor}
        )
        assert not second.is_error, second.content
        assert second.structured_content is not None

    first_names = [t["name"] for t in first.structured_content["items"]]
    second_names = [t["name"] for t in second.structured_content["items"]]
    assert first_names == [f"tool{i}" for i in range(50)]
    assert second_names == [f"tool{i}" for i in range(50, 100)]
    assert set(first_names).isdisjoint(second_names)


async def test_tool_list_offline_resources_pages_over_the_client() -> None:
    adapter = _client_adapter(page_size=50, find_count=120)
    async with Client(build_server(None, adapter)) as client:
        first = await client.call_tool("list_offline_resources", {"container_name": "exegol-lab"})
        assert not first.is_error, first.content
        assert first.structured_content is not None
        assert len(first.structured_content["items"]) == 50
        cursor = first.structured_content["next_cursor"]
        assert cursor is not None
        second = await client.call_tool(
            "list_offline_resources", {"container_name": "exegol-lab", "cursor": cursor}
        )
        assert not second.is_error, second.content
        assert second.structured_content is not None
    assert first.structured_content["items"] == [f"/opt/resources/linux/res{i}" for i in range(50)]
    assert second.structured_content["items"] == [f"/opt/resources/linux/res{i}" for i in range(50, 100)]


async def test_tool_page_size_follows_the_operator_config() -> None:
    adapter = _client_adapter(page_size=5, csv_count=12)
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool("list_installed_tools", {"container_name": "exegol-lab"})
    assert result.structured_content is not None
    assert len(result.structured_content["items"]) == 5  # override changed the page size from the default 200


async def test_tool_default_page_size_without_an_override_is_two_hundred() -> None:
    # A FakeExegolAdapter built with the default OperatorConfig() carries listing_page_size 200.
    def router(command: str) -> object:
        from exegol_mcp.models.exec import ExecResult

        if "installed_tools.csv" in command:
            return ExecResult(stdout=_large_csv(250), stderr="", exit_code=0)
        return None

    adapter = FakeExegolAdapter(exec_router=router)  # type: ignore[arg-type]
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool("list_installed_tools", {"container_name": "exegol-lab"})
    assert result.structured_content is not None
    assert len(result.structured_content["items"]) == 200
    assert result.structured_content["next_cursor"] is not None


async def test_tool_forged_cursor_is_a_tool_error_not_a_crash() -> None:
    adapter = _client_adapter(page_size=50, csv_count=120)
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool(
            "list_installed_tools", {"container_name": "exegol-lab", "cursor": "!!forged!!"}
        )
    assert result.is_error  # a forged cursor reaches the agent as a ToolError (T-02-04), never an unhandled crash
