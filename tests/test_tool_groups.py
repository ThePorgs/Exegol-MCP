"""Tool-group gating (CONF-02, D-22/D-23, the four groups of UAT G-05-1a). Docker-free, over the in-memory MCP Client.

A group listed in ``tool_groups.disabled`` is filtered out at registration in build_server, so its tools are absent
from tools/list and a call to one fails as an unknown tool. server_info has no group and is never gated. Gating hides
tools; it is not a security boundary (exec_command can still move files when ``files`` is disabled, D-23).
"""

from typing import Any

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError

from exegol_mcp.app import build_server
from exegol_mcp.bootstrap import secret_store
from exegol_mcp.bootstrap.config import ConfigError, load_operator_config
from exegol_mcp.models.config import OperatorConfig
from exegol_mcp.tools import TOOL_REGISTRY
from tests.fakes import FakeExegolAdapter

# The four tool group names the maintainer chose at UAT (G-05-1a), in order, spelled out here so this table, not the
# implementation constant, is the lock.
ALL_GROUPS = ["containers", "images", "execution", "files"]

# The binding G-05-1a mapping of every registered tool to its group. None marks server_info, the one tool that is
# never gateable.
EXPECTED_GROUPS: dict[str, str | None] = {
    "list_containers": "containers",
    "start_container": "containers",
    "stop_container": "containers",
    "create_container": "containers",
    "remove_container": "containers",
    "cleanup_containers": "containers",
    "list_installed_tools": "containers",
    "list_offline_resources": "containers",
    "list_images": "images",
    "download_image": "images",
    "exec_command": "execution",
    "get_job": "execution",
    "stop_job": "execution",
    "list_jobs": "execution",
    "get_tool_help": "execution",
    "exec_remote_command": "execution",
    "read_file": "files",
    "write_file": "files",
    "list_directory": "files",
    "server_info": None,
}

# The Phase 4 group names superseded at UAT (G-05-1a). No release shipped them, so each is an unknown name that stops
# startup like any other: refused, never aliased to one of the four.
RETIRED_GROUPS = ["lifecycle", "exec", "remote", "catalog"]

# The execution group's tools, derived from the lock table.
_EXECUTION_TOOLS = {name for name, group in EXPECTED_GROUPS.items() if group == "execution"}


@pytest.fixture
def config_bytes(monkeypatch: pytest.MonkeyPatch):
    """Make read_user_config_file('config.yml') return the given bytes (or None for 'file absent')."""

    def _set(raw: bytes | None) -> None:
        def _reader(name: str) -> bytes | None:
            assert name == "config.yml", name
            return raw

        monkeypatch.setattr(secret_store, "read_user_config_file", _reader)

    return _set


def _registry_names() -> list[str]:
    return [entry.fn.__name__ for entry in TOOL_REGISTRY]


def _server(disabled: list[Any] | None = None, config: OperatorConfig | None = None) -> MCPServer[Any]:
    if config is None:
        # Validated the same way config.yml is, so the in-memory and file-loaded paths share one model.
        config = OperatorConfig.model_validate({"tool_groups": {"disabled": disabled or []}})
    return build_server(None, FakeExegolAdapter(config=config))


async def _listed(server: MCPServer[Any]) -> list[str]:
    async with Client(server) as client:
        return [tool.name for tool in (await client.list_tools()).tools]


# --- The four-group mapping lock --------------------------------------------------------------------------------------


def test_every_registry_entry_is_a_tool_entry_tagged_with_its_group() -> None:
    from exegol_mcp.tools import ToolEntry

    assert TOOL_REGISTRY, "the registry is empty"
    for entry in TOOL_REGISTRY:
        assert isinstance(entry, ToolEntry), entry
        name = entry.fn.__name__
        assert name in EXPECTED_GROUPS, f"{name} has no tool group in EXPECTED_GROUPS"
        assert entry.group == EXPECTED_GROUPS[name], name


def test_server_info_is_the_only_ungated_tool() -> None:
    assert [entry.fn.__name__ for entry in TOOL_REGISTRY if entry.group is None] == ["server_info"]


def test_the_four_groups_are_exactly_the_maintainer_names() -> None:
    from exegol_mcp.models.config import TOOL_GROUPS

    assert list(TOOL_GROUPS) == ALL_GROUPS
    assert {group for group in EXPECTED_GROUPS.values() if group is not None} == set(ALL_GROUPS)


def test_registry_matches_expected_groups() -> None:
    """The surface lock: the registry holds exactly EXPECTED_GROUPS's tools, each in its group."""
    groups = {entry.fn.__name__: entry.group for entry in TOOL_REGISTRY}
    assert set(groups) == set(EXPECTED_GROUPS)
    assert groups == EXPECTED_GROUPS


def test_every_tool_group_has_a_tool() -> None:
    from exegol_mcp.models.config import TOOL_GROUPS

    names = _registry_names()
    assert len(names) == len(set(names)) == 20, names
    populated = {entry.group for entry in TOOL_REGISTRY}
    for group in TOOL_GROUPS:
        assert group in populated, f"tool group {group!r} has no registered tool"


# --- Gating at registration -----------------------------------------------------------------------------------------


async def test_disabling_images_removes_only_list_images_and_download_image() -> None:
    names = await _listed(_server(["images"]))
    assert "list_images" not in names and "download_image" not in names
    assert names == [name for name in _registry_names() if name not in {"list_images", "download_image"}]


async def test_disabling_execution_removes_exactly_its_six_tools() -> None:
    names = await _listed(_server(["execution"]))
    assert _EXECUTION_TOOLS == {
        "exec_command",
        "get_job",
        "stop_job",
        "list_jobs",
        "get_tool_help",
        "exec_remote_command",
    }
    assert names == [name for name in _registry_names() if name not in _EXECUTION_TOOLS]


async def test_calling_a_disabled_tool_fails_as_an_unknown_tool() -> None:
    async with Client(_server(["execution"])) as client:
        result = await client.call_tool(
            "exec_remote_command",
            {"container_name": "c", "protocol": "ssh", "target": "10.0.0.1", "username": "u", "command": "id"},
        )
    assert result.is_error
    text = " ".join(getattr(block, "text", "") for block in result.content)
    assert "Unknown tool" in text, text


async def test_the_other_tools_stay_callable_when_execution_is_disabled() -> None:
    async with Client(_server(["execution"])) as client:
        result = await client.call_tool("list_containers", {})
    assert not result.is_error, result.content


async def test_disabling_two_groups_removes_exactly_their_union_in_registry_order() -> None:
    removed = {"list_images", "download_image", "read_file", "write_file", "list_directory"}
    names = await _listed(_server(["images", "files"]))
    assert names == [name for name in _registry_names() if name not in removed]


async def test_listing_a_group_twice_equals_listing_it_once() -> None:
    assert await _listed(_server(["images", "images"])) == await _listed(_server(["images"]))


async def test_disabling_every_group_leaves_only_server_info() -> None:
    assert await _listed(_server(ALL_GROUPS)) == ["server_info"]


# --- All on by default (D-22) ---------------------------------------------------------------------------------------


async def test_the_default_config_lists_the_full_surface() -> None:
    assert await _listed(_server(config=OperatorConfig())) == _registry_names()


async def test_an_explicit_empty_disabled_list_lists_the_full_surface() -> None:
    assert await _listed(_server([])) == _registry_names()


@pytest.mark.parametrize(
    "raw",
    [None, b"", b"tool_groups:\n", b"tool_groups:\n  disabled: []\n"],
    ids=["no-file", "empty-file", "bare-key", "empty-list"],
)
async def test_a_config_file_without_disabled_groups_lists_the_full_surface(config_bytes, raw: bytes | None) -> None:
    config_bytes(raw)
    cfg = load_operator_config()
    assert cfg.tool_groups.disabled == []
    assert await _listed(_server(config=cfg)) == _registry_names()


# --- Loaded from config.yml (the tracer path) -----------------------------------------------------------------------


async def test_a_config_file_disabling_execution_matches_the_in_memory_config(config_bytes) -> None:
    config_bytes(b"tool_groups:\n  disabled: [execution]\n")
    cfg = load_operator_config()
    assert cfg.tool_groups.disabled == ["execution"]
    names = await _listed(_server(config=cfg))
    assert names == await _listed(_server(["execution"]))
    assert names == [name for name in _registry_names() if name not in _EXECUTION_TOOLS]
    async with Client(_server(config=cfg)) as client:
        remote = await client.call_tool(
            "exec_remote_command",
            {"container_name": "c", "protocol": "ssh", "target": "10.0.0.1", "username": "u", "command": "id"},
        )
        listing = await client.call_tool("list_containers", {})
    assert remote.is_error
    assert "Unknown tool" in " ".join(getattr(block, "text", "") for block in remote.content)
    assert not listing.is_error, listing.content


def test_an_unknown_group_name_is_a_config_error_naming_the_entry(config_bytes) -> None:
    config_bytes(b"tool_groups:\n  disabled: [remote_exec]\n")
    with pytest.raises(ConfigError, match=r"tool_groups\.disabled\.0"):
        load_operator_config()


def test_an_unknown_tool_groups_key_is_a_named_error(config_bytes) -> None:
    config_bytes(b"tool_groups:\n  enabled: [images]\n")
    with pytest.raises(ConfigError, match="enabled"):
        load_operator_config()


# --- Retired group names are refused, never aliased (G-05-1a) -------------------------------------------------------


@pytest.mark.parametrize("name", RETIRED_GROUPS)
def test_a_retired_group_name_stops_startup_naming_the_entry(config_bytes, name: str) -> None:
    config_bytes(f"tool_groups:\n  disabled: [{name}]\n".encode())
    with pytest.raises(ConfigError, match=r"tool_groups\.disabled\.0"):
        load_operator_config()


@pytest.mark.parametrize("name", RETIRED_GROUPS)
def test_a_retired_group_name_after_a_valid_one_names_its_own_index(config_bytes, name: str) -> None:
    config_bytes(f"tool_groups:\n  disabled: [images, {name}]\n".encode())
    with pytest.raises(ConfigError, match=r"tool_groups\.disabled\.1"):
        load_operator_config()


@pytest.mark.parametrize("name", RETIRED_GROUPS)
def test_the_in_memory_config_refuses_a_retired_group_name(name: str) -> None:
    with pytest.raises(ValidationError):
        OperatorConfig.model_validate({"tool_groups": {"disabled": [name]}})
