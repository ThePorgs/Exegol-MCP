"""create, list, remove and clean up containers (LIFE-01, LIFE-03..06), Docker-free over the in-memory MCP Client.

create_container goes from the tool through the SDK-free policy service to the fake port: the D-04 safe set only,
the D-06 name rules, a required installed image (D-07), the D-24 workspace allowlist, KEY=VALUE env entries
(Pitfall 5) and Exegol's port syntax, with this server process's run id stamped next to the managed label (D-09).
remove_container removes only managed containers unless the operator allows otherwise (D-11), with keep options that
fall back to the operator defaults, which keep everything (D-12). cleanup_containers removes this run's or every
managed container and never an unlabeled one (D-13), and the server removes nothing at startup or shutdown (D-14).
"""

import inspect
import re
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp_types import CallToolResult

from exegol_mcp import app as app_module
from exegol_mcp.adapter.port import MANAGED_LABEL, MANAGED_LABEL_VALUE, RUN_ID_LABEL, ContainerState, RemovalOutcome
from exegol_mcp.app import build_server
from exegol_mcp.models.config import ExecutionLimits, LifecycleSettings, OperatorConfig, ProfilesSettings
from exegol_mcp.services import containers as containers_service
from exegol_mcp.services import lifecycle as lifecycle_service
from exegol_mcp.tools import lifecycle as lifecycle_tool
from tests.fakes import (
    DEFAULT_CONTAINERS,
    MINIMAL_TOOL_ARGUMENTS,
    SEEDED_MANAGED_CONTAINER,
    SEEDED_MANAGED_RUN_ID,
    FakeExegolAdapter,
    _container,
)

_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_GENERATED_NAME_RE = re.compile(r"^mcp-[0-9a-f]{8}$")


async def _create(adapter: FakeExegolAdapter, **arguments: Any) -> CallToolResult:
    async with Client(build_server(None, adapter)) as client:
        return await client.call_tool("create_container", arguments)


def _ok(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


def _error_text(result: CallToolResult) -> str:
    assert result.is_error, result.structured_content
    return str(result.content)


def _allowlisted(tmp_path: Path, *entries: Path) -> FakeExegolAdapter:
    config = OperatorConfig(lifecycle=LifecycleSettings(workspace_allowlist=[str(e) for e in entries]))
    return FakeExegolAdapter(config=config, invoking_home=tmp_path)


# --- Task 1: create_container -----------------------------------------------------------------------------------


async def test_create_without_a_name_returns_a_running_managed_container_of_this_run() -> None:
    fake = FakeExegolAdapter()
    info = _ok(await _create(fake, image="free"))
    assert info["managed"] is True
    assert info["this_run"] is True
    assert fake.run_id is not None and _RUN_ID_RE.match(fake.run_id), fake.run_id
    assert info["run_id"] == fake.run_id
    assert _GENERATED_NAME_RE.match(info["name"]), info["name"]
    assert info["status"] == "Running"


async def test_the_spec_carries_the_managed_and_run_id_labels() -> None:
    fake = FakeExegolAdapter()
    _ok(await _create(fake, image="free"))
    assert len(fake.created_specs) == 1
    assert fake.created_specs[0].labels == {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: fake.run_id}


async def test_two_server_lifespans_mint_different_run_ids() -> None:
    first, second = FakeExegolAdapter(), FakeExegolAdapter()
    _ok(await _create(first, image="free"))
    _ok(await _create(second, image="free"))
    assert first.run_id and second.run_id
    assert first.run_id != second.run_id


async def test_a_leading_exegol_prefix_is_stripped_from_the_name() -> None:
    fake = FakeExegolAdapter()
    info = _ok(await _create(fake, image="free", container_name="exegol-web01"))
    assert info["name"] == "web01"
    assert fake.created_specs[0].name == "web01"


@pytest.mark.parametrize("name", ["Web01", "wéb", "", "-x", "exegol-", "a" * 64, "web 01", "web/01"])
async def test_an_invalid_name_is_refused_naming_the_rule(name: str) -> None:
    fake = FakeExegolAdapter()
    text = _error_text(await _create(fake, image="free", container_name=name))
    assert "lowercase" in text
    assert fake.created_specs == []


async def test_a_missing_image_points_to_list_images() -> None:
    fake = FakeExegolAdapter()
    text = _error_text(await _create(fake))
    assert "list_images" in text
    assert fake.created_specs == []


async def test_an_empty_image_points_to_list_images() -> None:
    fake = FakeExegolAdapter()
    assert "list_images" in _error_text(await _create(fake, image=""))
    assert fake.created_specs == []


async def test_an_image_that_is_not_installed_names_download_image_and_is_never_pulled() -> None:
    fake = FakeExegolAdapter()
    text = _error_text(await _create(fake, image="full"))
    assert "download_image" in text
    assert fake.created_specs == []
    assert fake.downloads == [] and fake.pulls == []


async def test_creating_the_same_name_twice_leaves_the_first_container_untouched() -> None:
    fake = FakeExegolAdapter()
    async with Client(build_server(None, fake)) as client:
        first = _ok(await client.call_tool("create_container", {"image": "free", "container_name": "web01"}))
        second = await client.call_tool("create_container", {"image": "free", "container_name": "web01"})
    assert "already exists" in _error_text(second)
    assert "web01" in _error_text(second)
    assert len(fake.created_specs) == 1
    assert next(c for c in fake.containers if c.name == "web01").model_dump(mode="json") == {**first, "this_run": False}


# --- workspace (D-24, D-08) ---


async def test_a_host_workspace_with_an_empty_allowlist_names_the_config_key() -> None:
    fake = FakeExegolAdapter()
    text = _error_text(await _create(fake, image="free", workspace="/srv/x"))
    assert "lifecycle.workspace_allowlist" in text
    assert fake.created_specs == []


async def test_an_allowlisted_workspace_is_accepted_absolute_and_in_the_tilde_form(tmp_path: Path) -> None:
    eng = tmp_path / "eng"
    eng.mkdir()
    resolved = eng.resolve()
    fake = _allowlisted(tmp_path, resolved)
    _ok(await _create(fake, image="free", container_name="abs", workspace=str(eng)))
    _ok(await _create(fake, image="free", container_name="tilde", workspace="~/eng"))
    assert [spec.workspace for spec in fake.created_specs] == [str(resolved), str(resolved)]


async def test_a_subdirectory_of_an_allowlisted_workspace_is_refused(tmp_path: Path) -> None:
    eng = tmp_path / "eng"
    (eng / "sub").mkdir(parents=True)
    fake = _allowlisted(tmp_path, eng.resolve())
    text = _error_text(await _create(fake, image="free", workspace=str(eng / "sub")))
    assert str(eng.resolve()) in text
    assert fake.created_specs == []


async def test_an_allowlisted_home_is_still_refused_at_create(tmp_path: Path) -> None:
    # Built directly, bypassing the loader that would refuse a home entry at startup: defense in depth.
    fake = _allowlisted(tmp_path, tmp_path.resolve())
    text = _error_text(await _create(fake, image="free", workspace="~"))
    assert "home directory" in text
    assert fake.created_specs == []


async def test_omitting_the_workspace_gives_a_private_one() -> None:
    fake = FakeExegolAdapter()
    _ok(await _create(fake, image="free"))
    assert fake.created_specs[0].workspace is None


# --- env, ports, hostname, comment, network (D-04, Pitfall 5) ---


async def test_key_value_env_entries_are_accepted() -> None:
    fake = FakeExegolAdapter()
    _ok(await _create(fake, image="free", env=["A=1", "B="]))
    assert fake.created_specs[0].env == ("A=1", "B=")


@pytest.mark.parametrize("entry", ["AWS_SECRET_ACCESS_KEY", "=x", "1A=x", "A=\x00"])
async def test_an_env_entry_without_key_equals_is_refused(entry: str) -> None:
    fake = FakeExegolAdapter()
    text = _error_text(await _create(fake, image="free", env=[entry]))
    assert "KEY=VALUE" in text
    assert fake.created_specs == []


@pytest.mark.parametrize(
    "port",
    [
        "8080:80",
        "127.0.0.1:2222:22:tcp",
        "8000-8010:9000-9010",
        "53:53:udp",
        "8080",
        # Exegol's documented optional parts on their own (IN-03).
        "8080:tcp",
        "127.0.0.1:8080",
        "8000-8010:9000-9010:udp",
    ],
)
async def test_a_port_in_exegol_syntax_is_accepted(port: str) -> None:
    fake = FakeExegolAdapter()
    _ok(await _create(fake, image="free", ports=[port]))
    assert fake.created_specs[0].ports == (port,)


@pytest.mark.parametrize(
    "port", ["127.0.0.1:2222:22/tcp", "70000:80", "80:70000", "9000-8000:80", "256.0.0.1:8080:80", "abc", ""]
)
async def test_a_malformed_or_out_of_range_port_is_refused(port: str) -> None:
    fake = FakeExegolAdapter()
    text = _error_text(await _create(fake, image="free", ports=[port]))
    assert "port" in text
    assert fake.created_specs == []


@pytest.mark.parametrize("port", ["8080:", "80::tcp", "1-2-3", "8080::", ":80", "80:80:", "80-:90", "80:udp:tcp"])
async def test_malformed_port_specs_are_refused(port: str) -> None:
    # Exegol's own permissive regex accepts these and then misparses them (IN-03): every separator needs its part.
    fake = FakeExegolAdapter()
    text = _error_text(await _create(fake, image="free", ports=[port]))
    assert f"Invalid port '{port}'" in text
    assert fake.created_specs == []


async def test_too_many_list_entries_are_refused() -> None:
    fake = FakeExegolAdapter()
    _error_text(await _create(fake, image="free", env=[f"K{i}=v" for i in range(65)]))
    _error_text(await _create(fake, image="free", ports=[str(1000 + i) for i in range(65)]))
    assert fake.created_specs == []


async def test_a_valid_hostname_and_comment_reach_the_spec() -> None:
    fake = FakeExegolAdapter()
    _ok(await _create(fake, image="free", hostname="web-01.lab", comment="engagement A", network_mode="nat"))
    spec = fake.created_specs[0]
    assert (spec.hostname, spec.comment, spec.network_mode) == ("web-01.lab", "engagement A", "nat")


@pytest.mark.parametrize("hostname", ["bad_host!", "-lead", "a" * 64, "", "x." * 127 + "y"])
async def test_an_invalid_hostname_is_refused(hostname: str) -> None:
    fake = FakeExegolAdapter()
    assert "hostname" in _error_text(await _create(fake, image="free", hostname=hostname))
    assert fake.created_specs == []


async def test_a_comment_over_512_characters_is_refused() -> None:
    fake = FakeExegolAdapter()
    assert "comment" in _error_text(await _create(fake, image="free", comment="c" * 513))
    _ok(await _create(fake, image="free", comment="c" * 512))
    assert len(fake.created_specs) == 1


async def test_an_attached_network_mode_is_refused_by_the_schema() -> None:
    fake = FakeExegolAdapter()
    _error_text(await _create(fake, image="free", network_mode="attached"))
    assert fake.created_specs == []


# --- profiles (04-08 Task 2: LIFE-02, CONF-01, D-01, D-03, D-05, D-07) ---


def _profiled(
    *allowed: str,
    require_profile: bool = False,
    profiles: dict[str, str | None] | None = None,
    workspace_allowlist: list[str] | None = None,
    invoking_home: Path | None = None,
    profile_licence: bool = True,
) -> FakeExegolAdapter:
    config = OperatorConfig(
        profiles=ProfilesSettings(allowed=list(allowed), require_profile=require_profile),
        lifecycle=LifecycleSettings(workspace_allowlist=workspace_allowlist or []),
    )
    return FakeExegolAdapter(
        config=config,
        profiles={"web": "free"} if profiles is None else profiles,
        profile_licence=profile_licence,
        invoking_home=invoking_home,
    )


async def test_an_allowlisted_profile_creates_without_an_image_argument() -> None:
    fake = _profiled("web")
    info = _ok(await _create(fake, profile="web"))
    assert info["managed"] is True
    assert info["this_run"] is True
    assert info["image_name"] == "free"
    [spec] = fake.created_specs
    assert spec.profile == "web"
    assert spec.image is None


async def test_an_unlisted_profile_is_refused_listing_the_allowed_ones() -> None:
    fake = _profiled("web", profiles={"web": "free", "ops": "free"})
    text = _error_text(await _create(fake, profile="ops"))
    assert "ops" in text and "web" in text
    assert fake.created_specs == []


async def test_with_an_empty_allowlist_every_profile_is_refused_saying_none_are_offered() -> None:
    fake = _profiled()
    text = _error_text(await _create(fake, profile="web"))
    assert "No container profiles are offered" in text
    assert "profiles.allowed" in text
    assert fake.created_specs == []


@pytest.mark.parametrize("profile", ["", "Web", "web ", "WEB"])
async def test_an_empty_or_differently_cased_profile_is_refused(profile: str) -> None:
    fake = _profiled("web")
    text = _error_text(await _create(fake, profile=profile, image="free"))
    assert "web" in text
    assert fake.created_specs == []


async def test_require_profile_refuses_a_create_without_a_profile_naming_the_allowed_ones() -> None:
    fake = _profiled("web", require_profile=True)
    text = _error_text(await _create(fake, image="free"))
    assert "web" in text
    assert "profile" in text
    assert fake.created_specs == []


async def test_require_profile_accepts_an_allowlisted_profile() -> None:
    fake = _profiled("web", require_profile=True)
    _ok(await _create(fake, profile="web"))
    assert [spec.profile for spec in fake.created_specs] == ["web"]


async def test_the_profile_checks_run_before_any_adapter_call() -> None:
    fake = _profiled("web", require_profile=True)
    calls: list[str] = []
    original = fake.get_container_state

    async def recording(name: str) -> Any:
        calls.append(name)
        return await original(name)

    fake.get_container_state = recording  # type: ignore[method-assign]
    _error_text(await _create(fake, image="free"))
    _error_text(await _create(fake, profile="ops"))
    assert calls == []


async def test_without_a_profile_or_an_image_the_error_points_to_list_images() -> None:
    fake = _profiled("web")
    assert "list_images" in _error_text(await _create(fake))
    assert fake.created_specs == []


async def test_a_profile_that_names_no_image_and_no_image_argument_points_to_list_images() -> None:
    fake = _profiled("web", profiles={"web": None})
    assert "list_images" in _error_text(await _create(fake, profile="web"))
    assert fake.containers == DEFAULT_CONTAINERS


async def test_safe_set_overrides_on_a_profile_are_validated_and_reach_the_spec(tmp_path: Path) -> None:
    workspace = tmp_path / "engagement"
    workspace.mkdir()
    fake = _profiled("web", workspace_allowlist=[str(workspace)], invoking_home=tmp_path)
    _ok(
        await _create(
            fake,
            profile="web",
            image="free",
            container_name="web01",
            workspace=str(workspace),
            network_mode="nat",
            ports=["127.0.0.1:8080:80"],
            env=["A=1"],
            hostname="box",
            comment="on top of web",
        )
    )
    [spec] = fake.created_specs
    assert (spec.profile, spec.image, spec.name) == ("web", "free", "web01")
    assert spec.workspace == str(workspace)
    assert (spec.network_mode, spec.ports, spec.env) == ("nat", ("127.0.0.1:8080:80",), ("A=1",))
    assert (spec.hostname, spec.comment) == ("box", "on top of web")


@pytest.mark.parametrize(
    "arguments",
    [{"env": ["AWS_SECRET_ACCESS_KEY"]}, {"ports": ["70000:80"]}, {"workspace": "/srv"}, {"hostname": "bad_host!"}],
)
async def test_overrides_on_a_profile_go_through_the_same_validation(arguments: dict[str, Any]) -> None:
    fake = _profiled("web")
    _error_text(await _create(fake, profile="web", **arguments))
    assert fake.created_specs == []


async def test_the_same_profile_twice_gives_two_containers() -> None:
    fake = _profiled("web")
    async with Client(build_server(None, fake)) as client:
        first = _ok(await client.call_tool("create_container", {"profile": "web", "container_name": "one"}))
        second = _ok(await client.call_tool("create_container", {"profile": "web", "container_name": "two"}))
    assert (first["name"], second["name"]) == ("one", "two")
    assert [spec.profile for spec in fake.created_specs] == ["web", "web"]


async def test_without_a_pro_licence_an_allowlisted_profile_is_refused_and_nothing_is_added() -> None:
    fake = _profiled("web", profile_licence=False)
    text = _error_text(await _create(fake, profile="web"))
    assert "Pro" in text
    assert fake.containers == DEFAULT_CONTAINERS


def test_the_tool_offers_profile_but_no_profile_only_capability() -> None:
    parameters = set(inspect.signature(lifecycle_tool.create_container).parameters)
    assert "profile" in parameters
    assert not parameters & {"privileged", "devices", "capabilities", "volumes", "vpn", "desktop", "gui"}


def test_the_create_container_docstring_explains_profiles_and_the_merge() -> None:
    doc = inspect.getdoc(lifecycle_tool.create_container) or ""
    assert "profile" in doc
    assert "require" in doc
    assert "added to the profile's" in doc


async def test_create_container_is_listed_with_its_annotations_and_an_output_schema() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tool = {tool.name: tool for tool in (await client.list_tools()).tools}["create_container"]
    assert tool.annotations is not None
    hints = (
        tool.annotations.read_only_hint,
        tool.annotations.destructive_hint,
        tool.annotations.idempotent_hint,
        tool.annotations.open_world_hint,
    )
    assert hints == (False, False, False, False)
    assert tool.output_schema is not None
    properties = set(tool.input_schema["properties"])
    # 04-08 added profile; the safe set is otherwise unchanged and no profile-only option is an argument (D-05).
    assert properties == {
        "image",
        "profile",
        "container_name",
        "workspace",
        "network_mode",
        "ports",
        "env",
        "hostname",
        "comment",
    }


# --- Task 2: list_containers shows managed, run_id and this_run (LIFE-04, LIFE-06, D-10) ------------------------


def _with_earlier_run() -> FakeExegolAdapter:
    # The two default unmanaged containers plus one an earlier (or crashed) server run created.
    return FakeExegolAdapter(
        containers=[
            *(c.model_copy() for c in DEFAULT_CONTAINERS),
            _container("exegol-old", "earlier", managed=True, run_id="earlier-run"),
        ]
    )


async def _list(client: Client, **arguments: Any) -> list[dict[str, Any]]:
    result = await client.call_tool("list_containers", arguments)
    return _ok(result)["result"]


async def test_a_container_from_an_earlier_run_shows_as_managed_but_not_this_run() -> None:
    async with Client(build_server(None, _with_earlier_run())) as client:
        listed = {c["name"]: c for c in await _list(client)}
    assert (listed["exegol-old"]["managed"], listed["exegol-old"]["run_id"]) == (True, "earlier-run")
    assert listed["exegol-old"]["this_run"] is False
    for name in ("exegol-lab", "exegol-lab2"):
        assert (listed[name]["managed"], listed[name]["run_id"], listed[name]["this_run"]) == (False, None, False)


async def test_a_container_created_by_this_server_lists_as_this_run() -> None:
    fake = _with_earlier_run()
    async with Client(build_server(None, fake)) as client:
        created = _ok(await client.call_tool("create_container", {"image": "free", "container_name": "web01"}))
        listed = {c["name"]: c for c in await _list(client)}
    assert created["this_run"] is True
    assert listed["web01"]["this_run"] is True
    assert listed["web01"]["run_id"] == fake.run_id
    assert listed["exegol-old"]["this_run"] is False


async def test_managed_only_lists_only_the_managed_containers_in_the_adapter_order() -> None:
    fake = _with_earlier_run()
    async with Client(build_server(None, fake)) as client:
        _ok(await client.call_tool("create_container", {"image": "free", "container_name": "web01"}))
        managed = await _list(client, managed_only=True)
        everything = await _list(client)
    assert [c["name"] for c in managed] == ["exegol-old", "web01"]
    assert all(c["managed"] for c in managed)
    assert len(everything) == 4


async def test_managed_only_with_no_managed_container_is_an_empty_list() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        assert await _list(client, managed_only=True) == []
        assert len(await _list(client)) == 2


async def test_this_run_compares_the_run_id_by_exact_string_equality() -> None:
    fake = FakeExegolAdapter(containers=[])
    async with Client(build_server(None, fake)) as client:
        # Seeded after the lifespan minted this run's id: one exact match, one that differs only in case.
        assert fake.run_id is not None
        # A minted token_hex id can be all digits, so .upper() would not change it; pin one with letters.
        fake.run_id = "a1b2c3d4e5f6"
        fake.containers = [
            _container("exact", None, managed=True, run_id=fake.run_id),
            _container("upper", None, managed=True, run_id=fake.run_id.upper()),
        ]
        listed = {c["name"]: c for c in await _list(client)}
    assert listed["exact"]["this_run"] is True
    assert listed["upper"]["this_run"] is False


async def test_the_service_never_marks_an_unmanaged_container_as_this_run() -> None:
    fake = FakeExegolAdapter(containers=[_container("hand", None, run_id="r1")])
    infos = await containers_service.list_containers(fake, run_id="r1")
    assert [(i.managed, i.this_run) for i in infos] == [(False, False)]
    assert await containers_service.list_containers(fake, run_id="r1", managed_only=True) == []


async def test_the_service_without_a_run_id_marks_nothing_as_this_run() -> None:
    infos = await containers_service.list_containers(_with_earlier_run())
    assert [i.this_run for i in infos] == [False, False, False]


# --- Plan 04-06 Task 1: remove_container (LIFE-03, D-11, D-12) ----------------------------------------------------


def _lifecycle_config(**lifecycle: Any) -> OperatorConfig:
    return OperatorConfig(lifecycle=LifecycleSettings(**lifecycle))


async def _remove(adapter: FakeExegolAdapter, **arguments: Any) -> CallToolResult:
    async with Client(build_server(None, adapter)) as client:
        return await client.call_tool("remove_container", arguments)


async def test_a_created_container_is_removed_keeping_both_by_default_and_is_no_longer_listed() -> None:
    fake = FakeExegolAdapter()
    async with Client(build_server(None, fake)) as client:
        name = _ok(await client.call_tool("create_container", {"image": "free"}))["name"]
        removed = _ok(await client.call_tool("remove_container", {"container_name": name}))
        listed = [c["name"] for c in await _list(client)]
    assert removed == {"name": name, "workspace_kept": True, "logs_kept": True}
    assert name not in listed
    assert fake.removals == [(name, True, True)]


async def test_an_unmanaged_container_is_refused_naming_the_operator_toggle() -> None:
    fake = FakeExegolAdapter()
    async with Client(build_server(None, fake)) as client:
        text = _error_text(await client.call_tool("remove_container", {"container_name": "exegol-lab"}))
        listed = [c["name"] for c in await _list(client)]
    assert "lifecycle.allow_remove_unmanaged" in text
    assert "not created by exegol-mcp" in text
    assert "exegol-lab" in listed
    assert fake.removals == []


async def test_the_operator_toggle_lets_remove_container_remove_an_unmanaged_container() -> None:
    fake = FakeExegolAdapter(config=_lifecycle_config(allow_remove_unmanaged=True))
    removed = _ok(await _remove(fake, container_name="exegol-lab"))
    assert removed["name"] == "exegol-lab"
    assert fake.removals == [("exegol-lab", True, True)]
    assert [c.name for c in fake.containers] == ["exegol-lab2"]


async def test_per_call_keep_options_reach_the_port() -> None:
    fake = FakeExegolAdapter(seed_managed_container=True)
    removed = _ok(await _remove(fake, container_name=SEEDED_MANAGED_CONTAINER, keep_workspace=False, keep_logs=False))
    assert (removed["workspace_kept"], removed["logs_kept"]) == (False, False)
    assert fake.removals == [(SEEDED_MANAGED_CONTAINER, False, False)]


async def test_an_omitted_keep_option_falls_back_to_the_config_default() -> None:
    fake = FakeExegolAdapter(config=_lifecycle_config(keep_workspace=False), seed_managed_container=True)
    _ok(await _remove(fake, container_name=SEEDED_MANAGED_CONTAINER))
    assert fake.removals == [(SEEDED_MANAGED_CONTAINER, False, True)]


async def test_an_explicit_keep_option_overrides_the_config_default() -> None:
    config = _lifecycle_config(keep_workspace=False, keep_logs=False)
    fake = FakeExegolAdapter(config=config, seed_managed_container=True)
    _ok(await _remove(fake, container_name=SEEDED_MANAGED_CONTAINER, keep_workspace=True))
    assert fake.removals == [(SEEDED_MANAGED_CONTAINER, True, False)]


async def test_removing_the_same_container_twice_names_list_containers() -> None:
    fake = FakeExegolAdapter(seed_managed_container=True)
    async with Client(build_server(None, fake)) as client:
        _ok(await client.call_tool("remove_container", {"container_name": SEEDED_MANAGED_CONTAINER}))
        text = _error_text(await client.call_tool("remove_container", {"container_name": SEEDED_MANAGED_CONTAINER}))
    assert "list_containers" in text
    assert fake.removals == [(SEEDED_MANAGED_CONTAINER, True, True)]


async def test_two_concurrent_removals_of_one_name_remove_it_once() -> None:
    fake = FakeExegolAdapter(seed_managed_container=True)
    outcomes: list[str] = []

    async def _one() -> None:
        try:
            await lifecycle_service.remove_container(
                fake, SEEDED_MANAGED_CONTAINER, keep_workspace=None, keep_logs=None, settings=LifecycleSettings()
            )
            outcomes.append("removed")
        except ValueError as e:
            outcomes.append(str(e))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_one)
        task_group.start_soon(_one)
    assert len(outcomes) == 2 and outcomes.count("removed") == 1, outcomes
    assert "list_containers" in next(o for o in outcomes if o != "removed")
    assert fake.removals == [(SEEDED_MANAGED_CONTAINER, True, True)]


def test_the_managed_seed_is_opt_in_and_minimal_arguments_remove_it() -> None:
    seeded = FakeExegolAdapter(seed_managed_container=True)
    seed = next(c for c in seeded.containers if c.name == SEEDED_MANAGED_CONTAINER)
    assert (seed.managed, seed.run_id) == (True, SEEDED_MANAGED_RUN_ID) == (True, "earlier-run")
    assert seeded.labels[SEEDED_MANAGED_CONTAINER] == {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "earlier-run"}
    assert [c.name for c in FakeExegolAdapter().containers] == [c.name for c in DEFAULT_CONTAINERS]
    assert MINIMAL_TOOL_ARGUMENTS["remove_container"] == {"container_name": "mcp-seed"}


async def test_remove_container_is_listed_destructive_with_an_output_schema() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tool = {tool.name: tool for tool in (await client.list_tools()).tools}["remove_container"]
    assert tool.annotations is not None
    hints = (
        tool.annotations.read_only_hint,
        tool.annotations.destructive_hint,
        tool.annotations.idempotent_hint,
        tool.annotations.open_world_hint,
    )
    assert hints == (False, True, False, False)
    assert tool.output_schema is not None
    assert set(tool.input_schema["properties"]) == {"container_name", "keep_workspace", "keep_logs"}


# --- Plan 04-06 Task 2: cleanup_containers (LIFE-04, D-10, D-13) and no shutdown cleanup (LIFE-05, D-14) ----------


async def _cleanup(client: Client, **arguments: Any) -> dict[str, Any]:
    return _ok(await client.call_tool("cleanup_containers", arguments))


async def test_cleanup_this_run_removes_only_the_containers_this_server_created() -> None:
    fake = FakeExegolAdapter(seed_managed_container=True)
    async with Client(build_server(None, fake)) as client:
        created = _ok(await client.call_tool("create_container", {"image": "free"}))["name"]
        result = await _cleanup(client)
        listed = [c["name"] for c in await _list(client)]
    assert result == {"scope": "this_run", "removed": [created], "failed": []}
    assert listed == ["exegol-lab", "exegol-lab2", SEEDED_MANAGED_CONTAINER]
    assert fake.removals == [(created, True, True)]


@pytest.mark.parametrize("allow_remove_unmanaged", [False, True])
async def test_cleanup_all_managed_also_removes_earlier_runs_but_never_unmanaged(allow_remove_unmanaged: bool) -> None:
    config = _lifecycle_config(allow_remove_unmanaged=allow_remove_unmanaged)
    fake = FakeExegolAdapter(config=config, seed_managed_container=True)
    async with Client(build_server(None, fake)) as client:
        created = _ok(await client.call_tool("create_container", {"image": "free"}))["name"]
        result = await _cleanup(client, scope="all_managed")
        listed = [c["name"] for c in await _list(client)]
    assert result == {"scope": "all_managed", "removed": sorted([created, SEEDED_MANAGED_CONTAINER]), "failed": []}
    assert listed == ["exegol-lab", "exegol-lab2"]


@pytest.mark.parametrize("scope", ["this_run", "all_managed"])
async def test_cleanup_with_nothing_to_remove_is_empty_and_running_it_twice_too(scope: str) -> None:
    fake = FakeExegolAdapter(seed_managed_container=True)
    async with Client(build_server(None, fake)) as client:
        _ok(await client.call_tool("create_container", {"image": "free"}))
        await _cleanup(client, scope=scope)
        second = await _cleanup(client, scope=scope)
    assert second == {"scope": scope, "removed": [], "failed": []}
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        assert await _cleanup(client, scope=scope) == {"scope": scope, "removed": [], "failed": []}


class _FailingRemovals(FakeExegolAdapter):
    """A fake whose removal of the named containers fails like a Docker error."""

    def __init__(self, failing: set[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.failing = failing

    async def remove_container(self, name: str, keep_workspace: bool, keep_logs: bool, **kwargs: Any) -> RemovalOutcome:
        if name in self.failing:
            raise RuntimeError(f"boom {name}")
        return await super().remove_container(name, keep_workspace, keep_logs, **kwargs)


async def test_cleanup_reports_failures_in_name_order_and_still_removes_the_others() -> None:
    names = ["zeta", "mid", "alpha", "beta"]
    fake = _FailingRemovals(
        {"mid", "beta"}, containers=[_container(n, None, managed=True, run_id="earlier-run") for n in names]
    )
    async with Client(build_server(None, fake)) as client:
        result = await _cleanup(client, scope="all_managed")
    assert result["removed"] == ["alpha", "zeta"]
    assert result["failed"] == [{"name": "beta", "error": "boom beta"}, {"name": "mid", "error": "boom mid"}]
    assert [c.name for c in fake.containers] == ["mid", "beta"]


class _SwappedAfterTheCheck(FakeExegolAdapter):
    """A fake whose state lookup still reports the managed container the name used to hold, while the container now
    behind the name carries the labels in self.labels (another process removed and recreated it in between)."""

    async def get_container_state(self, name: str) -> ContainerState | None:
        state = await super().get_container_state(name)
        if state is None:
            return None
        return ContainerState(running=state.running, managed=True, run_id=SEEDED_MANAGED_RUN_ID)


async def test_a_swap_to_an_unmanaged_container_after_the_service_check_is_refused_by_the_adapter() -> None:
    """WR-02: the adapter re-applies the managed gate to the container it actually removes (D-11)."""
    fake = _SwappedAfterTheCheck()
    assert fake.labels.get("exegol-lab", {}) == {}

    with pytest.raises(ValueError, match="lifecycle.allow_remove_unmanaged"):
        await lifecycle_service.remove_container(
            fake, "exegol-lab", keep_workspace=None, keep_logs=None, settings=LifecycleSettings()
        )

    assert fake.removals == []
    assert "exegol-lab" in [c.name for c in fake.containers]


async def test_cleanup_this_run_refuses_a_container_swapped_in_from_another_run() -> None:
    """WR-02 and D-13: this_run cleanup removes only a container whose own run id is exactly this run's."""
    fake = FakeExegolAdapter(containers=[_container(n, None, managed=True, run_id="this-run") for n in ("a", "swap")])
    # The listing still reports this run's id, but the container now behind "swap" is another run's.
    fake.labels["swap"] = {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "other-run"}

    result = await lifecycle_service.cleanup_containers(
        fake, scope="this_run", run_id="this-run", keep_workspace=None, keep_logs=None, settings=LifecycleSettings()
    )

    assert result.removed == ["a"]
    [failure] = result.failed
    assert failure.name == "swap"
    assert "is no longer one this server run created" in failure.error
    assert "list_containers" in failure.error
    assert [c.name for c in fake.containers] == ["swap"]


class _EmptyMessageOSError(OSError):
    """An OSError subclass raised with no message, as some socket and filesystem errors are."""


class _RaisingRemovals(FakeExegolAdapter):
    """A fake whose removal of the named containers raises the given exception, of any kind (WR-01)."""

    def __init__(self, errors: dict[str, Exception], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.errors = errors

    async def remove_container(self, name: str, keep_workspace: bool, keep_logs: bool, **kwargs: Any) -> RemovalOutcome:
        if name in self.errors:
            raise self.errors[name]
        return await super().remove_container(name, keep_workspace, keep_logs, **kwargs)


async def test_cleanup_reports_an_os_error_subclass_and_other_failures_and_still_removes_the_others() -> None:
    """WR-01: one container's OSError subclass or AttributeError-based error never aborts the cleanup (LIFE-04)."""
    from exegol_mcp.adapter.errors import UnknownParameterError

    names = ["zeta", "mid", "alpha", "beta", "gamma"]
    errors: dict[str, Exception] = {
        "mid": ConnectionResetError("reset"),
        "beta": _EmptyMessageOSError(),
        "gamma": UnknownParameterError("p"),
    }
    fake = _RaisingRemovals(errors, containers=[_container(n, None, managed=True, run_id="earlier-run") for n in names])
    async with Client(build_server(None, fake)) as client:
        result = await _cleanup(client, scope="all_managed")
    assert result["removed"] == ["alpha", "zeta"]
    # Edge LIFE-04 ordering: name order whatever mix of failures happened; an empty message gives the type name.
    assert result["failed"] == [
        {"name": "beta", "error": "_EmptyMessageOSError"},
        {"name": "gamma", "error": "p"},
        {"name": "mid", "error": "reset"},
    ]
    assert [c.name for c in fake.containers] == ["mid", "beta", "gamma"]


class _SuspendingRemoval(FakeExegolAdapter):
    """A fake whose removal of one container suspends until the caller is cancelled."""

    def __init__(self, suspending: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.suspending = suspending
        self.suspended = anyio.Event()

    async def remove_container(self, name: str, keep_workspace: bool, keep_logs: bool, **kwargs: Any) -> RemovalOutcome:
        if name == self.suspending:
            self.suspended.set()
            await anyio.sleep_forever()
        return await super().remove_container(name, keep_workspace, keep_logs, **kwargs)


async def test_a_cancelled_cleanup_propagates_and_returns_no_result() -> None:
    names = ["a", "b", "c"]
    fake = _SuspendingRemoval("b", containers=[_container(n, None, managed=True, run_id="earlier-run") for n in names])
    seen: dict[str, Any] = {}

    async def cleanup() -> None:
        with anyio.CancelScope() as scope:
            seen["scope"] = scope
            seen["result"] = await lifecycle_service.cleanup_containers(
                fake,
                scope="all_managed",
                run_id=None,
                keep_workspace=None,
                keep_logs=None,
                settings=LifecycleSettings(),
            )
        seen["cancelled"] = scope.cancelled_caught

    async with anyio.create_task_group() as tg:
        tg.start_soon(cleanup)
        await fake.suspended.wait()
        seen["scope"].cancel()

    # Cancellation is a BaseException: the per-container isolation never swallows it.
    assert seen["cancelled"] is True
    assert "result" not in seen
    assert [removal[0] for removal in fake.removals] == ["a"]
    assert [c.name for c in fake.containers] == ["b", "c"]


async def test_cleanup_keep_options_reach_every_removal_with_the_config_defaults_when_omitted() -> None:
    names = ["b", "a"]
    managed = [_container(n, None, managed=True, run_id="earlier-run") for n in names]
    fake = FakeExegolAdapter(containers=list(managed))
    async with Client(build_server(None, fake)) as client:
        await _cleanup(client, scope="all_managed", keep_workspace=False)
    assert fake.removals == [("a", False, True), ("b", False, True)]
    defaults = FakeExegolAdapter(config=_lifecycle_config(keep_logs=False), containers=list(managed))
    async with Client(build_server(None, defaults)) as client:
        await _cleanup(client, scope="all_managed")
    assert defaults.removals == [("a", True, False), ("b", True, False)]


async def test_the_container_op_timeout_reaches_every_removal() -> None:
    """CR-01: the operator's execution.container_op_timeout_s bounds each removal, single and cleanup alike."""
    names = ["a", "b", "c"]
    config = OperatorConfig(execution=ExecutionLimits(container_op_timeout_s=7))
    fake = FakeExegolAdapter(
        config=config, containers=[_container(n, None, managed=True, run_id="earlier-run") for n in names]
    )
    async with Client(build_server(None, fake)) as client:
        _ok(await client.call_tool("remove_container", {"container_name": "a"}))
        result = await _cleanup(client, scope="all_managed")
    assert result["removed"] == ["b", "c"]
    assert fake.removal_timeouts == [7.0, 7.0, 7.0]


async def test_the_cleanup_service_selects_nothing_for_this_run_without_a_run_id() -> None:
    fake = FakeExegolAdapter(seed_managed_container=True)
    result = await lifecycle_service.cleanup_containers(
        fake, scope="this_run", run_id=None, keep_workspace=None, keep_logs=None, settings=LifecycleSettings()
    )
    assert (result.removed, result.failed) == ([], [])
    assert fake.removals == []


async def test_the_cleanup_service_refuses_an_unknown_scope() -> None:
    fake = FakeExegolAdapter(seed_managed_container=True)
    with pytest.raises(ValueError, match="this_run"):
        await lifecycle_service.cleanup_containers(
            fake, scope="everything", run_id="r", keep_workspace=None, keep_logs=None, settings=LifecycleSettings()
        )
    assert fake.removals == []


async def test_cleanup_containers_is_listed_destructive_idempotent_with_an_output_schema() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tool = {tool.name: tool for tool in (await client.list_tools()).tools}["cleanup_containers"]
    assert tool.annotations is not None
    hints = (
        tool.annotations.read_only_hint,
        tool.annotations.destructive_hint,
        tool.annotations.idempotent_hint,
        tool.annotations.open_world_hint,
    )
    assert hints == (False, True, True, False)
    assert tool.output_schema is not None
    assert set(tool.input_schema["properties"]) == {"scope", "keep_workspace", "keep_logs"}
    assert MINIMAL_TOOL_ARGUMENTS["cleanup_containers"] == {}


async def test_two_server_lifespans_remove_no_container_at_startup_or_shutdown() -> None:
    # LIFE-05 / D-14: managed containers of this and earlier runs, plus unmanaged ones, all survive two full
    # startup/shutdown cycles; the only lifespan cleanup is the job-process orphan sweep.
    fake = FakeExegolAdapter(seed_managed_container=True)
    async with Client(build_server(None, fake)) as client:
        created = _ok(await client.call_tool("create_container", {"image": "free"}))["name"]
    expected = ["exegol-lab", "exegol-lab2", SEEDED_MANAGED_CONTAINER, created]
    assert [c.name for c in fake.containers] == expected
    async with Client(build_server(None, fake)) as client:
        assert [c["name"] for c in await _list(client)] == expected
    assert fake.removals == []
    assert [c.name for c in fake.containers] == expected
    assert fake.sweep_calls == 4  # startup and shutdown sweeps of both lifespans


def test_the_lifespan_never_calls_into_container_removal() -> None:
    source = inspect.getsource(app_module)
    assert '_sweep_orphans(adapter, "shutdown")' in source
    for forbidden in ("remove_container", "cleanup_containers", "lifecycle_service"):
        assert forbidden not in source, forbidden
