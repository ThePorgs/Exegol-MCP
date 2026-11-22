"""Catalog service unit tests (CORE-05): CSV parse incl. short-row/empty-file, order, help quoting. Docker-free."""

import shlex

import pytest
from mcp import Client

from exegol_mcp.app import build_server
from exegol_mcp.models.container import InstalledTool
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.resources import INSTALLED_TOOLS_URI
from exegol_mcp.services import catalog as catalog_service
from tests.fakes import FakeExegolAdapter

# The header Exegol images write: Tool,Version,Link,Description.
NORMAL_CSV = (
    "Tool,Version,Link,Description\n"
    "nmap,7.94,https://nmap.org,Network scanner\n"
    "nmap,7.95,https://nmap.org,Newer network scanner\n"
    "crackmapexec,5.4,https://github.com/byt3bl33d3r/CrackMapExec,Swiss army knife\n"
)

# Older and free images write no Version column: Tool,Link,Description.
FREE_IMAGE_CSV = "Tool,Link,Description\nnmap,https://nmap.org,Network scanner\n"

# Rows with fewer fields than the header: csv.DictReader fills the rest with None (the v1 crash trigger).
SHORT_ROW_CSV = "Tool,Version,Link,Description\nnmap,7.94\nhydra\n"


def _csv_adapter(csv_text: str) -> FakeExegolAdapter:
    def router(command: str) -> ExecResult | None:
        if "installed_tools.csv" in command:
            return ExecResult(stdout=csv_text, stderr="", exit_code=0)
        return None

    return FakeExegolAdapter(exec_router=router)


def _stopped(adapter: FakeExegolAdapter, name: str) -> FakeExegolAdapter:
    adapter.running[name] = False
    return adapter


async def test_list_installed_tools_parses_a_normal_csv() -> None:
    tools = (await catalog_service.list_installed_tools(_csv_adapter(NORMAL_CSV), "exegol-lab")).items
    assert [t.name for t in tools] == ["nmap", "nmap", "crackmapexec"]
    assert tools[0].version == "7.94"
    assert tools[0].link == "https://nmap.org"
    assert tools[0].description == "Network scanner"


async def test_list_installed_tools_preserves_csv_order_and_keeps_same_name_versions_distinct() -> None:
    tools = (await catalog_service.list_installed_tools(_csv_adapter(NORMAL_CSV), "exegol-lab")).items
    assert [(t.name, t.version) for t in tools] == [
        ("nmap", "7.94"),
        ("nmap", "7.95"),
        ("crackmapexec", "5.4"),
    ]


async def test_list_installed_tools_returns_empty_list_for_an_empty_file() -> None:
    assert (await catalog_service.list_installed_tools(_csv_adapter(""), "exegol-lab")).items == []


async def test_list_installed_tools_does_not_crash_on_short_rows() -> None:
    # The v1 (row.get(k)).strip() raised AttributeError on a None field; here it must return partial items.
    tools = (await catalog_service.list_installed_tools(_csv_adapter(SHORT_ROW_CSV), "exegol-lab")).items
    assert [t.name for t in tools] == ["nmap", "hydra"]
    assert tools[0].version == "7.94"
    assert tools[0].link is None
    assert tools[0].description is None
    assert tools[1].version is None
    assert tools[1].link is None
    assert tools[1].description is None


async def test_list_installed_tools_parses_a_free_image_csv_without_a_version_column() -> None:
    tools = (await catalog_service.list_installed_tools(_csv_adapter(FREE_IMAGE_CSV), "exegol-lab")).items
    assert [t.name for t in tools] == ["nmap"]
    assert tools[0].version is None
    assert tools[0].link == "https://nmap.org"
    assert tools[0].description == "Network scanner"


@pytest.mark.parametrize("link_header", ["Link", "link", "URL", "url"])
def test_parse_installed_tools_reads_the_link_from_any_header_variant(link_header: str) -> None:
    tools = catalog_service.parse_installed_tools(
        f"Tool,Version,{link_header},Description\nnmap,7.94,https://nmap.org,Network scanner\n"
    )
    assert [(t.name, t.version, t.link, t.description) for t in tools] == [
        ("nmap", "7.94", "https://nmap.org", "Network scanner")
    ]


def test_installed_tool_has_exactly_the_four_catalog_keys() -> None:
    expected = {"name", "version", "link", "description"}
    assert set(InstalledTool.model_fields) == expected
    (tool,) = catalog_service.parse_installed_tools(NORMAL_CSV)[:1]
    assert set(tool.model_dump(mode="json")) == expected


async def test_list_installed_tools_unknown_container_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not found"):
        await catalog_service.list_installed_tools(_csv_adapter(NORMAL_CSV), "ghost")


async def test_list_installed_tools_stopped_container_raises_start_it_first() -> None:
    adapter = _stopped(_csv_adapter(NORMAL_CSV), "exegol-lab")
    with pytest.raises(RuntimeError, match="start it first"):
        await catalog_service.list_installed_tools(adapter, "exegol-lab")


async def test_get_tool_help_rejects_a_metacharacter_tool_name_before_shelling() -> None:
    # v2 pre-SEC-07 relied on shlex.quote alone to neutralise a metacharacter name; SEC-07 now rejects such a
    # name outright with a ValueError before any shell command is built (shlex.quote stays as defense-in-depth
    # for the now always-charset-valid name).
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="usage: ...", stderr="", exit_code=0))
    with pytest.raises(ValueError):
        await catalog_service.get_tool_help(adapter, "exegol-lab", "my tool; rm -rf /")
    assert adapter.exec_commands == []


@pytest.mark.parametrize(
    "bad_name",
    [
        "",  # empty
        "   ",  # whitespace-only
        "\t",  # whitespace-only (tab)
        "a" * 129,  # over the 128-char cap
        "nmapé",  # non-ASCII letter
        "a b",  # embedded space
        "x;y",  # shell metacharacter
        "x$(touch /tmp/x)",  # command-substitution payload
        "a/b",  # path separator, outside the charset
    ],
)
async def test_get_tool_help_rejects_an_invalid_tool_name_before_shelling(bad_name: str) -> None:
    # SEC-07/D-07: a malformed tool name is a ValueError (-> ToolError) raised before require_running and before
    # any adapter.exec_command, so an invalid name is never built into a shell command.
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="x", stderr="", exit_code=0))
    with pytest.raises(ValueError):
        await catalog_service.get_tool_help(adapter, "exegol-lab", bad_name)
    assert adapter.exec_commands == []


@pytest.mark.parametrize("good_name", ["nmap", "crackmapexec", "python3.11"])
async def test_get_tool_help_accepts_a_valid_tool_name(good_name: str) -> None:
    # The charset allows letters, digits, '.', '_', '+' and '-'; a valid name still reaches the happy path.
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="usage", stderr="", exit_code=0))
    result = await catalog_service.get_tool_help(adapter, "exegol-lab", good_name)
    assert result.exit_code == 0
    # shlex.quote stays as defense-in-depth (a no-op for a charset-valid name, but still applied).
    assert adapter.exec_commands == [f"{shlex.quote(good_name)} --help"]


async def test_get_tool_help_returns_the_first_successful_flag() -> None:
    def router(command: str) -> ExecResult | None:
        if command.endswith(" -h"):
            return ExecResult(stdout="help via -h", stderr="", exit_code=0)
        return ExecResult(stdout="", stderr="unknown flag", exit_code=2)

    adapter = FakeExegolAdapter(exec_router=router)
    result = await catalog_service.get_tool_help(adapter, "exegol-lab", "nmap")
    assert result.stdout == "help via -h"
    assert result.exit_code == 0
    # --help tried first (failed), then -h (succeeded); "help" never tried.
    assert adapter.exec_commands == ["nmap --help", "nmap -h"]


async def test_get_tool_help_returns_the_last_result_without_coercing_the_exit_code() -> None:
    # Every flag fails with a None exit code: v1 coerced it to 1; v2 must preserve None.
    adapter = FakeExegolAdapter(
        exec_router=lambda _cmd: ExecResult(stdout="", stderr="no such tool", exit_code=None)
    )
    result = await catalog_service.get_tool_help(adapter, "exegol-lab", "ghosttool")
    assert result.exit_code is None
    assert len(adapter.exec_commands) == 3


async def test_list_offline_resources_rejects_an_unknown_target_os() -> None:
    adapter = FakeExegolAdapter()
    with pytest.raises(ValueError, match="Invalid target OS 'bsd'"):
        await catalog_service.list_offline_resources(adapter, "exegol-lab", "bsd")


async def test_list_offline_resources_runs_the_verbatim_find_and_splits_lines() -> None:
    captured: list[str] = []

    def router(command: str) -> ExecResult | None:
        captured.append(command)
        return ExecResult(stdout="/opt/resources/linux/a\n/opt/resources/linux/b\n", stderr="", exit_code=0)

    adapter = FakeExegolAdapter(exec_router=router)
    resources = (await catalog_service.list_offline_resources(adapter, "exegol-lab", "windows")).items
    assert resources == ["/opt/resources/linux/a", "/opt/resources/linux/b"]
    assert captured[0].startswith("find /opt/resources/windows -type f")
    assert "LICEN[SC]E" in captured[0]


# --- The catalog contract served to the model ------------------------------------------------------------------------

_FIELDS_PHRASE = "name, version, link and description (null when the CSV does not record them)"


async def test_served_list_installed_tools_description_names_the_real_fields() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tools = (await client.list_tools()).tools
    (tool,) = [t for t in tools if t.name == "list_installed_tools"]
    assert _FIELDS_PHRASE in " ".join((tool.description or "").split())


async def test_served_installed_tools_template_description_names_the_real_fields() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        templates = (await client.list_resource_templates()).resource_templates
    (template,) = [t for t in templates if t.uri_template == INSTALLED_TOOLS_URI]
    assert _FIELDS_PHRASE in " ".join((template.description or "").split())


async def test_served_list_installed_tools_output_schema_has_exactly_the_four_keys() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tools = (await client.list_tools()).tools
    (tool,) = [t for t in tools if t.name == "list_installed_tools"]
    assert tool.output_schema is not None
    assert set(tool.output_schema["$defs"]["InstalledTool"]["properties"]) == {
        "name",
        "version",
        "link",
        "description",
    }
