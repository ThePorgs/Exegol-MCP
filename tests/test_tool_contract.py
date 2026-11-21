"""Tool contract tests over the in-memory MCP Client (successor of the v1 registration test).

Plan 03 seeded it; plan 06 turns it into the full PROTO-03/04 contract test over the final 11-tool v2 surface.
"""

import importlib.metadata
from typing import Any

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import Tool, ToolAnnotations

import exegol_mcp
from exegol_mcp.adapter.errors import HeadlessPromptError, PatchPointError, UnknownParameterError
from exegol_mcp.adapter.port import IMAGE_UP_TO_DATE_MESSAGE, ExegolPort
from exegol_mcp.app import build_server
from exegol_mcp.context.request import ProgressCallback, RequestState, request_ctx
from exegol_mcp.models.image import ImageScope
from exegol_mcp.models.server_info import ServerInfo
from exegol_mcp.services import containers as containers_service
from exegol_mcp.services import images as images_service
from exegol_mcp.services import server_info as server_info_service
from exegol_mcp.services.server_info import VersionReport
from exegol_mcp.tools.errors import ToolBoundary, tool_errors
from tests.fakes import (
    DEFAULT_CONTAINERS,
    DEFAULT_IMAGES,
    MINIMAL_TOOL_ARGUMENTS,
    V2_TOOL_NAMES,
    FakeExegolAdapter,
    catalog_router,
)


async def _call_list_containers(adapter: ExegolPort) -> list[dict[str, Any]]:
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool("list_containers", {})
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content["result"]


async def test_list_containers_is_listed_read_only_with_output_schema() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert "list_containers" in tools, sorted(tools)
    tool = tools["list_containers"]
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    assert tool.output_schema is not None


async def test_list_containers_returns_fake_containers_as_structured_content() -> None:
    adapter = FakeExegolAdapter()
    containers = await _call_list_containers(adapter)
    assert containers == [c.model_dump(mode="json") for c in adapter.containers]


async def test_list_containers_empty_adapter_returns_empty_list() -> None:
    assert await _call_list_containers(FakeExegolAdapter(containers=[])) == []


async def test_list_containers_order_is_stable_across_calls() -> None:
    adapter = FakeExegolAdapter()
    async with Client(build_server(None, adapter)) as client:
        first = (await client.call_tool("list_containers", {})).structured_content
        second = (await client.call_tool("list_containers", {})).structured_content
    assert first is not None and second is not None
    assert [c["name"] for c in first["result"]] == [c["name"] for c in second["result"]]
    assert first == second


async def test_list_containers_keeps_near_identical_containers_distinct() -> None:
    containers = await _call_list_containers(FakeExegolAdapter())
    assert len(containers) == len(DEFAULT_CONTAINERS) == 2
    assert [c["name"] for c in containers] == ["exegol-lab", "exegol-lab2"]


# --- Container, image and server_info services (plan 03) ---------------------------------------------------


def _stopped(adapter: FakeExegolAdapter, name: str) -> FakeExegolAdapter:
    adapter.running[name] = False
    return adapter


async def test_start_container_starts_a_stopped_container() -> None:
    adapter = _stopped(FakeExegolAdapter(), "exegol-lab")
    assert await containers_service.start_container(adapter, "exegol-lab") is True
    assert adapter.running["exegol-lab"] is True
    assert adapter.state_changes == [("start", "exegol-lab")]


async def test_start_container_is_a_noop_when_already_running() -> None:
    adapter = FakeExegolAdapter()
    assert await containers_service.start_container(adapter, "exegol-lab") is True
    assert adapter.state_changes == []


async def test_start_container_unknown_name_raises_value_error() -> None:
    adapter = FakeExegolAdapter()
    with pytest.raises(ValueError, match="Container 'ghost' not found"):
        await containers_service.start_container(adapter, "ghost")
    assert adapter.state_changes == []


async def test_stop_container_stops_a_running_container() -> None:
    adapter = FakeExegolAdapter()
    assert await containers_service.stop_container(adapter, "exegol-lab") is True
    assert adapter.running["exegol-lab"] is False
    assert adapter.state_changes == [("stop", "exegol-lab")]


async def test_stop_container_is_a_noop_when_already_stopped() -> None:
    adapter = _stopped(FakeExegolAdapter(), "exegol-lab2")
    assert await containers_service.stop_container(adapter, "exegol-lab2") is True
    assert adapter.state_changes == []


async def test_stop_container_unknown_name_raises_value_error() -> None:
    adapter = FakeExegolAdapter()
    with pytest.raises(ValueError, match="Container 'ghost' not found"):
        await containers_service.stop_container(adapter, "ghost")
    assert adapter.state_changes == []


async def test_list_images_installed_scope_returns_only_installed_images() -> None:
    images = await images_service.list_images(FakeExegolAdapter(), "installed")
    assert [i.name for i in images] == ["free"]
    assert all(i.is_installed for i in images)


async def test_list_images_all_scope_returns_installed_and_available_images() -> None:
    images = await images_service.list_images(FakeExegolAdapter(), "all")
    assert [(i.name, i.is_installed) for i in images] == [("free", True), ("full", False), ("nightly", False)]


async def test_list_images_keeps_an_installed_image_and_its_available_counterpart_distinct() -> None:
    installed = DEFAULT_IMAGES[1].model_copy(update={"version": "5.0.0", "is_installed": True, "is_up_to_date": False})
    available = DEFAULT_IMAGES[1].model_copy()
    adapter = FakeExegolAdapter(images=[installed, available])
    assert await images_service.list_images(adapter, "all") == [installed, available]
    assert await images_service.list_images(adapter, "installed") == [installed]


@pytest.mark.parametrize("scope", ["installed", "all"])
async def test_list_images_empty_adapter_returns_empty_list(scope: ImageScope) -> None:
    assert await images_service.list_images(FakeExegolAdapter(images=[]), scope) == []


async def test_list_images_order_is_stable_across_calls() -> None:
    adapter = FakeExegolAdapter()
    assert await images_service.list_images(adapter, "all") == await images_service.list_images(adapter, "all")


async def test_list_images_rejects_an_unknown_scope() -> None:
    with pytest.raises(ValueError, match="Invalid scope 'remote'"):
        await images_service.list_images(FakeExegolAdapter(), "remote")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["nightly", "free"])
async def test_download_image_rejects_a_versioned_nightly_or_free(name: str) -> None:
    adapter = FakeExegolAdapter()
    with pytest.raises(ValueError, match=f"{name} image can only be downloaded with version 'latest'"):
        await images_service.download_image(adapter, name, "5.0.0")
    assert adapter.downloads == []


@pytest.mark.parametrize(("name", "version"), [("", "latest"), ("  ", "latest"), ("full", "")])
async def test_download_image_rejects_empty_arguments(name: str, version: str) -> None:
    adapter = FakeExegolAdapter()
    with pytest.raises(ValueError, match="must not be empty"):
        await images_service.download_image(adapter, name, version)
    assert adapter.downloads == []


async def test_download_image_unknown_image_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Image 'ghost' not found"):
        await images_service.download_image(FakeExegolAdapter(), "ghost")


async def test_download_image_installs_and_reports_progress_to_the_request_sink() -> None:
    adapter = FakeExegolAdapter()
    seen: list[tuple[float, float | None, str | None]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    assert await images_service.download_image(adapter, "full", "latest", progress=sink, request_id="7") is True
    assert adapter.downloads == [("full", "latest")]
    assert seen[-1] == (100, 100, "Download complete")
    assert [i.name for i in await images_service.list_images(adapter, "installed")] == ["free", "full"]


async def test_download_image_binds_the_request_state_only_for_the_call() -> None:
    bound: list[RequestState | None] = []

    class RecordingAdapter(FakeExegolAdapter):
        async def download_image(self, name: str, version: str, progress: ProgressCallback | None) -> bool:
            bound.append(request_ctx.get())
            return await super().download_image(name, version, progress)

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        return None

    await images_service.download_image(RecordingAdapter(), "full", progress=sink, request_id="42")
    assert bound == [RequestState(request_id="42", progress=sink)]
    assert request_ctx.get() is None


class _FakeInfoSource:
    def __init__(self, library: str | None, wrapper: str | None, notice: str | None = None) -> None:
        self.report = VersionReport(
            exegol_mcp_version="2.0.0",
            exegol_library_version=library,
            exegol_wrapper_version=wrapper,
            docker_status="reachable",
            config_dirs=["/home/user/.exegol", "/home/user/.config/exegol-mcp"],
            update_notice=notice,
        )

    async def version_report(self) -> VersionReport:
        return self.report


@pytest.mark.parametrize(
    ("library", "wrapper", "state"),
    [
        ("6.0.2", "6.0.2", "match"),
        ("6.0.2", "v6.0.2", "match"),
        ("6.0.2", "5.1.13", "mismatch"),
        ("6.0.2", None, "unknown"),
    ],
)
async def test_server_info_builds_a_server_info_from_the_source(library: str, wrapper: str | None, state: str) -> None:
    info = await server_info_service.server_info(_FakeInfoSource(library, wrapper, notice="exegol-mcp 2.1.0 is out"))
    assert isinstance(info, ServerInfo)
    assert info.exegol_mcp_version == "2.0.0"
    assert info.exegol_library_version == library
    assert info.exegol_wrapper_version == wrapper
    assert info.docker_status == "reachable"
    assert info.config_dirs == ["/home/user/.exegol", "/home/user/.config/exegol-mcp"]
    assert info.compatibility_state == state
    assert info.update_notice == "exegol-mcp 2.1.0 is out"


async def test_server_info_falls_back_to_package_metadata_without_a_source() -> None:
    source = server_info_service.source_from(FakeExegolAdapter())
    assert isinstance(source, server_info_service.DefaultServerInfoSource)
    info = await server_info_service.server_info(source)
    assert info.exegol_mcp_version == exegol_mcp.__version__
    assert info.exegol_library_version == importlib.metadata.version("exegol")
    assert info.compatibility_state == "unknown"
    assert info.update_notice is None


async def test_server_info_uses_a_lifespan_context_that_is_a_source() -> None:
    source = _FakeInfoSource("6.0.2", "6.0.2")
    assert server_info_service.source_from(source) is source


# --- PROTO-03/04 registry contract: the exact 11-tool v2 surface -------------------------------------------------

# Tools that list or describe state: each must declare an output schema and return structured content (PROTO-04).
LIST_INFO_TOOLS = {
    "list_containers",
    "list_images",
    "list_installed_tools",
    "list_offline_resources",
    "server_info",
    "list_jobs",
    "list_directory",
}


def _contract_violations(tools: list[Tool]) -> list[str]:
    """Return every PROTO-03/04 breach in a tools/list result, [] when the surface honours the contract."""
    violations: list[str] = []
    for tool in tools:
        if tool.annotations is None:
            violations.append(f"{tool.name}: no annotations")
            continue
        if tool.annotations.read_only_hint is None:
            violations.append(f"{tool.name}: read_only_hint not set")
        if tool.annotations.destructive_hint is None:
            violations.append(f"{tool.name}: destructive_hint not set")
    for tool in tools:
        if tool.name in LIST_INFO_TOOLS and tool.output_schema is None:
            violations.append(f"{tool.name}: list/info tool without output_schema")
    return violations


async def _list_tools(server: MCPServer[Any]) -> list[Tool]:
    async with Client(server) as client:
        return (await client.list_tools()).tools


async def test_registry_exposes_exactly_the_v2_tool_surface() -> None:
    names = [tool.name for tool in await _list_tools(build_server(None, FakeExegolAdapter()))]
    assert len(names) == len(set(names)) == 20, names
    assert (
        set(names)
        == V2_TOOL_NAMES
        == {
            "list_containers",
            "start_container",
            "stop_container",
            "list_images",
            "download_image",
            "exec_command",
            "exec_remote_command",
            "list_installed_tools",
            "get_tool_help",
            "list_offline_resources",
            "server_info",
            "get_job",
            "stop_job",
            "list_jobs",
            "create_container",
            "remove_container",
            "cleanup_containers",
            "read_file",
            "write_file",
            "list_directory",
        }
    )
    assert LIST_INFO_TOOLS <= V2_TOOL_NAMES


async def test_registry_every_tool_is_annotated_and_every_list_info_tool_has_an_output_schema() -> None:
    assert _contract_violations(await _list_tools(build_server(None, FakeExegolAdapter()))) == []


async def test_registry_annotations_follow_the_v1_to_v2_mapping() -> None:
    tools = {tool.name: tool for tool in await _list_tools(build_server(None, FakeExegolAdapter()))}
    expected = {
        "list_containers": (True, False, None, None),
        "start_container": (False, False, True, None),
        "stop_container": (False, True, True, None),
        "list_images": (True, False, None, True),
        "download_image": (False, False, None, True),
        "server_info": (True, False, True, None),
        "exec_command": (False, True, None, True),
        "exec_remote_command": (False, True, None, True),
        "list_installed_tools": (True, False, None, None),
        "get_tool_help": (True, False, None, None),
        "list_offline_resources": (True, False, None, None),
        "get_job": (True, False, None, None),
        "list_jobs": (True, False, None, None),
        "stop_job": (False, True, True, None),
        "create_container": (False, False, False, False),
        "remove_container": (False, True, False, False),
        "cleanup_containers": (False, True, True, False),
        "read_file": (True, False, True, False),
        "write_file": (False, True, False, False),
        "list_directory": (True, False, True, False),
    }
    assert set(expected) == set(tools) == V2_TOOL_NAMES
    for name, hints in expected.items():
        annotations = tools[name].annotations
        assert annotations is not None, name
        actual = (
            annotations.read_only_hint,
            annotations.destructive_hint,
            annotations.idempotent_hint,
            annotations.open_world_hint,
        )
        assert actual == hints, name


async def test_registry_contract_check_catches_a_missing_annotation_or_schema() -> None:
    """The contract check must fail on an unannotated tool and on a list/info tool without an output schema."""

    async def unannotated_tool() -> str:
        return "x"

    async def list_containers():  # type: ignore[no-untyped-def]  # no return annotation: no output schema
        return "x"

    server = build_server(None, FakeExegolAdapter())
    server.add_tool(unannotated_tool)
    assert _contract_violations(await _list_tools(server)) == ["unannotated_tool: no annotations"]

    schemaless: MCPServer[Any] = MCPServer("schemaless")
    schemaless.add_tool(list_containers, annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False))
    assert _contract_violations(await _list_tools(schemaless)) == [
        "list_containers: list/info tool without output_schema"
    ]

    half: MCPServer[Any] = MCPServer("half")
    half.add_tool(unannotated_tool, annotations=ToolAnnotations(read_only_hint=True))
    assert _contract_violations(await _list_tools(half)) == ["unannotated_tool: destructive_hint not set"]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("list_images", {"scope": "all"}), *((name, MINIMAL_TOOL_ARGUMENTS[name]) for name in sorted(LIST_INFO_TOOLS))],
)
async def test_registry_list_info_tools_return_structured_content(name: str, arguments: dict[str, Any]) -> None:
    async with Client(build_server(None, FakeExegolAdapter(exec_router=catalog_router))) as client:
        tool = {tool.name: tool for tool in (await client.list_tools()).tools}[name]
        result = await client.call_tool(name, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    assert tool.output_schema is not None
    # The structured content is an object matching the declared schema's top-level properties.
    assert set(result.structured_content) <= set(tool.output_schema.get("properties", {})), name


# --- End-to-end tool calls over the in-memory Client ---------------------------------------------------------


async def test_start_container_tool_starts_then_lists_the_container_running() -> None:
    adapter = _stopped(FakeExegolAdapter(), "exegol-lab")
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool("start_container", {"container_name": "exegol-lab"})
        again = await client.call_tool("start_container", {"container_name": "exegol-lab"})
    assert result.structured_content == {"result": True}
    assert again.structured_content == {"result": True}
    assert adapter.state_changes == [("start", "exegol-lab")]


async def test_stop_container_tool_unknown_name_returns_a_clear_error() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        result = await client.call_tool("stop_container", {"container_name": "ghost"})
    assert result.is_error
    assert "Container 'ghost' not found" in str(result.content)


async def test_list_images_tool_defaults_to_installed_scope() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        installed = await client.call_tool("list_images", {})
        everything = await client.call_tool("list_images", {"scope": "all"})
    assert installed.structured_content is not None and everything.structured_content is not None
    assert [i["name"] for i in installed.structured_content["result"]] == ["free"]
    assert [i["name"] for i in everything.structured_content["result"]] == ["free", "full", "nightly"]
    assert "latest_version" in installed.structured_content["result"][0]


async def test_list_images_tool_rejects_an_unknown_scope() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        result = await client.call_tool("list_images", {"scope": "remote"})
    assert result.is_error


async def test_download_image_tool_reports_progress_to_the_calling_client() -> None:
    adapter = FakeExegolAdapter()
    seen: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool("download_image", {"image_name": "full"}, progress_callback=on_progress)
    assert result.structured_content == {"result": True}
    assert adapter.downloads == [("full", "latest")]
    assert adapter.pulls == [("full", "latest")]
    assert seen and seen[-1] == (100, 100, "Download complete")


async def test_download_image_tool_already_up_to_date_returns_true_and_says_so_over_the_fake() -> None:
    adapter = FakeExegolAdapter()
    seen: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool("download_image", {"image_name": "free"}, progress_callback=on_progress)
    assert result.structured_content == {"result": True}
    assert adapter.pulls == []
    assert adapter.downloads == [("free", "latest")]
    assert seen == [(100, 100, IMAGE_UP_TO_DATE_MESSAGE.format(image="free"))]


async def test_download_image_tool_rejects_a_versioned_nightly() -> None:
    adapter = FakeExegolAdapter()
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool("download_image", {"image_name": "nightly", "image_version": "5.0.0"})
    assert result.is_error
    assert "only be downloaded with version 'latest'" in str(result.content)
    assert adapter.downloads == []


async def test_server_info_tool_returns_a_server_info() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        result = await client.call_tool("server_info", {})
    assert result.structured_content is not None
    info = ServerInfo.model_validate(result.structured_content)
    assert info.exegol_mcp_version == exegol_mcp.__version__


async def test_tool_errors_passes_anticipated_errors_to_the_client_and_leaves_crashes_alone() -> None:
    for anticipated in (ValueError("bad name"), RuntimeError("Container 'x' is not running")):
        with pytest.raises(ToolError, match=str(anticipated)) as raised:
            async with tool_errors():
                raise anticipated
        assert raised.value.__cause__ is anticipated
    with pytest.raises(KeyError):
        async with tool_errors():
            raise KeyError("crash")


# --- SDK guard errors (D-09, D-15): anticipated failures the agent must read, never the opaque crash --------------

def _sdk_guard_errors() -> list[Exception]:
    return [UnknownParameterError("p"), HeadlessPromptError("prompt"), PatchPointError("point")]


class _GuardTrippingAdapter(FakeExegolAdapter):
    """list_containers trips the SDK's unset-parameter guard, as a renamed exegol option would (D-09)."""

    async def list_containers(self) -> list[Any]:
        raise UnknownParameterError("The Exegol SDK read the CLI parameter 'probe_param'")


class _HealthyDiagnoser:
    """A ReadinessDiagnoser whose host is still ready; counts how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def diagnose(self) -> str | None:
        self.calls += 1
        return None


async def test_sdk_guard_error_reaches_the_client_as_a_readable_error() -> None:
    async with Client(build_server(None, _GuardTrippingAdapter())) as client:
        result = await client.call_tool("list_containers", {})
    assert result.is_error
    text = " ".join(getattr(block, "text", "") for block in result.content)
    assert "probe_param" in text, text
    assert text != "Error executing tool list_containers", text


async def test_tool_errors_maps_sdk_guard_errors_without_a_diagnoser() -> None:
    for guard in _sdk_guard_errors():
        with pytest.raises(ToolError) as raised:
            async with tool_errors():
                raise guard
        assert str(raised.value) == str(guard)
        assert raised.value.__cause__ is guard


async def test_tool_errors_maps_sdk_guard_errors_after_a_healthy_diagnosis() -> None:
    diagnoser = _HealthyDiagnoser()
    guards = _sdk_guard_errors()
    for guard in guards:
        with pytest.raises(ToolError) as raised:
            async with tool_errors(diagnoser):
                raise guard
        assert str(raised.value) == str(guard)
        assert raised.value.__cause__ is guard
    assert diagnoser.calls == len(guards)


# --- D-08 boundary contract: no sync bypass, a failing diagnosis never hides the error, every tool diagnoses ------


def test_tool_boundary_has_no_sync_form() -> None:
    # A sync "with" would skip the awaited readiness re-check, so the boundary must not offer one (D-08).
    assert not hasattr(ToolBoundary, "__enter__")


class _BrokenDiagnoser:
    """A ReadinessDiagnoser whose own probe fails."""

    async def diagnose(self) -> str | None:
        raise OSError("probe broke")


async def test_tool_errors_a_failing_diagnosis_keeps_the_original_error() -> None:
    original = RuntimeError("original")
    with pytest.raises(ToolError) as raised:
        async with tool_errors(_BrokenDiagnoser()):
            raise original
    assert str(raised.value) == "original"
    assert raised.value.__cause__ is original


_READINESS_DIAGNOSIS = "READINESS: docker gone"


class _DaemonGoneAdapter(FakeExegolAdapter):
    """Every port method fails as if the Docker daemon died after startup; diagnose() says so (D-08)."""

    async def list_containers(self) -> list[Any]:
        raise ConnectionError("daemon gone")

    async def get_container_state(self, name: str) -> Any:
        raise ConnectionError("daemon gone")

    async def start_container(self, name: str) -> bool:
        raise ConnectionError("daemon gone")

    async def stop_container(self, name: str) -> bool:
        raise ConnectionError("daemon gone")

    async def list_images(self, scope: Any) -> list[Any]:
        raise ConnectionError("daemon gone")

    async def download_image(self, name: str, version: str, progress: ProgressCallback | None) -> bool:
        raise ConnectionError("daemon gone")

    async def exec_command(self, name: str, command: str) -> Any:
        raise ConnectionError("daemon gone")

    async def diagnose(self) -> str | None:
        return _READINESS_DIAGNOSIS


async def test_every_docker_tool_reports_the_readiness_diagnosis() -> None:
    # server_info never calls a port method: it reports Docker status instead of failing on it. The job tools
    # (get_job/stop_job/list_jobs) act on in-memory store state, not a live Docker round-trip, so an unknown id
    # is a plain ValueError rather than a daemon-gone readiness diagnosis; they are not Docker tools here.
    docker_tools = sorted(V2_TOOL_NAMES - {"server_info", "get_job", "stop_job", "list_jobs"})
    offenders: list[str] = []
    async with Client(build_server(None, _DaemonGoneAdapter())) as client:
        for name in docker_tools:
            result = await client.call_tool(name, MINIMAL_TOOL_ARGUMENTS[name])
            text = " ".join(getattr(block, "text", "") for block in result.content)
            if not result.is_error or _READINESS_DIAGNOSIS not in text:
                offenders.append(name)
    assert offenders == [], f"tools that did not report the readiness diagnosis: {offenders}"
