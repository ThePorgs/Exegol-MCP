"""Operator config file (config.yml) model, loader and server wiring (CONF-03/CONF-04, D-01..D-06). Docker-free.

Every test feeds bytes (or None) through a monkeypatched secret_store.read_user_config_file, so nothing here touches
the user's real config dir. The loader never writes config.yml (D-03): it is pure override on top of code defaults.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from exegol_mcp.bootstrap import config as config_module
from exegol_mcp.bootstrap import secret_store
from exegol_mcp.bootstrap.cli import DEFAULT_PORT, parse_args
from exegol_mcp.bootstrap.config import ConfigError, load_operator_config, resolve
from exegol_mcp.models.config import ExecutionLimits, OperatorConfig, SecuritySettings
from exegol_mcp.services import host_paths

# The Phase-4 sections (CONF-01 config side). Imported lazily in the tests so a missing model fails the test that
# needs it, not the whole module.
PHASE4_SECTIONS = ("ProfilesSettings", "LifecycleSettings", "FilesSettings")


@pytest.fixture
def config_bytes(monkeypatch: pytest.MonkeyPatch):
    """Make read_user_config_file('config.yml') return the given bytes (or None for 'file absent')."""

    def _set(raw: bytes | None) -> None:
        def _reader(name: str) -> bytes | None:
            assert name == "config.yml", name
            return raw

        monkeypatch.setattr(secret_store, "read_user_config_file", _reader)

    return _set


# --- Defaults and the full D-04 skeleton ----------------------------------------------------------------------------


def test_operator_config_defaults_match_the_objective_values() -> None:
    c = OperatorConfig()
    e = c.execution
    assert e.timeout_default_s == 50
    assert e.timeout_max_s == 600
    assert e.output_cap_bytes == 1_048_576
    assert e.job_buffer_cap_bytes == 1_048_576
    assert e.max_concurrent_jobs == 16
    assert e.finished_job_ttl_s == 3600
    assert e.max_job_runtime_s == 86_400
    assert e.kill_grace_s == 5
    assert e.listing_page_size == 200
    assert e.container_op_timeout_s == 50
    # D-06 mirrored keys are optional overrides, unset by default.
    assert c.transport is None
    assert c.port is None
    assert c.auth is None
    assert c.update_check is None


def test_both_sections_forbid_unknown_keys() -> None:
    assert OperatorConfig.model_config["extra"] == "forbid"
    assert ExecutionLimits.model_config["extra"] == "forbid"


def test_phase4_section_defaults_keep_everything() -> None:
    c = OperatorConfig()
    # D-01 / D-03: no profiles offered and none required.
    assert c.profiles.allowed == []
    assert c.profiles.require_profile is False
    # D-11: hand-made containers are protected; D-12: workspace and logs are kept; D-24: no host-path workspace.
    assert c.lifecycle.allow_remove_unmanaged is False
    assert c.lifecycle.keep_workspace is True
    assert c.lifecycle.keep_logs is True
    assert c.lifecycle.workspace_allowlist == []
    # D-17 / D-19: one 10 MiB raw-byte limit, a depth cap and an entry cap for listings.
    assert c.files.max_bytes == 10_485_760
    assert c.files.max_list_depth == 5
    assert c.files.max_list_entries == 10_000


def test_phase4_sections_forbid_unknown_keys() -> None:
    from exegol_mcp.models import config as models

    for name in PHASE4_SECTIONS:
        assert getattr(models, name).model_config.get("extra") == "forbid", name


# --- Loading a file -------------------------------------------------------------------------------------------------


def test_a_valid_file_overrides_only_what_it_sets(config_bytes) -> None:
    config_bytes(b"execution:\n  timeout_default_s: 12.5\n")

    c = load_operator_config()

    assert c.execution.timeout_default_s == 12.5
    # Everything it did not set keeps the built-in default.
    assert c.execution.timeout_max_s == 600
    assert c.execution.listing_page_size == 200


def test_a_missing_file_yields_built_in_defaults(config_bytes) -> None:
    config_bytes(None)
    assert load_operator_config() == OperatorConfig()


def test_an_empty_file_yields_built_in_defaults(config_bytes) -> None:
    config_bytes(b"")
    assert load_operator_config() == OperatorConfig()


def test_a_whitespace_only_file_yields_built_in_defaults(config_bytes) -> None:
    config_bytes(b"   \n\n  \n")
    assert load_operator_config() == OperatorConfig()


# --- Fail-fast validation (CONF-04) ---------------------------------------------------------------------------------


def test_an_unknown_top_level_section_is_a_named_error(config_bytes) -> None:
    config_bytes(b"nonsense:\n  foo: 1\n")
    with pytest.raises(ConfigError, match="nonsense"):
        load_operator_config()


def test_an_unknown_nested_execution_key_is_a_named_error(config_bytes) -> None:
    # Adjacency edge: a near-miss typo of a real key must be rejected, not silently ignored.
    config_bytes(b"execution:\n  timeout_defalut_s: 5\n")
    with pytest.raises(ConfigError, match="timeout_defalut_s"):
        load_operator_config()


def test_a_zero_value_for_a_gt0_field_is_a_named_error(config_bytes) -> None:
    config_bytes(b"execution:\n  timeout_default_s: 0\n")
    with pytest.raises(ConfigError, match="timeout_default_s"):
        load_operator_config()


def test_a_negative_value_for_a_gt0_field_is_a_named_error(config_bytes) -> None:
    config_bytes(b"execution:\n  output_cap_bytes: -1\n")
    with pytest.raises(ConfigError, match="output_cap_bytes"):
        load_operator_config()


def test_a_non_numeric_value_for_a_numeric_field_is_a_named_error(config_bytes) -> None:
    config_bytes(b"execution:\n  timeout_default_s: not-a-number\n")
    with pytest.raises(ConfigError, match="timeout_default_s"):
        load_operator_config()


def test_kill_grace_s_accepts_zero_but_rejects_negative(config_bytes) -> None:
    config_bytes(b"execution:\n  kill_grace_s: 0\n")
    assert load_operator_config().execution.kill_grace_s == 0

    config_bytes(b"execution:\n  kill_grace_s: -1\n")
    with pytest.raises(ConfigError, match="kill_grace_s"):
        load_operator_config()


def test_a_positive_boundary_value_loads(config_bytes) -> None:
    config_bytes(b"execution:\n  timeout_default_s: 1\n")
    assert load_operator_config().execution.timeout_default_s == 1


def test_a_misspelled_lifecycle_key_is_a_named_error(config_bytes) -> None:
    config_bytes(b"lifecycle:\n  keep_workspce: true\n")
    with pytest.raises(ConfigError, match="keep_workspce"):
        load_operator_config()


@pytest.mark.parametrize("section", ["profiles", "files"])
def test_an_unknown_key_in_a_phase4_section_is_a_named_error(config_bytes, section: str) -> None:
    config_bytes(f"{section}:\n  not_a_key: 1\n".encode())
    with pytest.raises(ConfigError, match=rf"{section}\.not_a_key"):
        load_operator_config()


@pytest.mark.parametrize("field", ["max_bytes", "max_list_depth", "max_list_entries"])
def test_a_zero_files_limit_is_a_named_error(config_bytes, field: str) -> None:
    config_bytes(f"files:\n  {field}: 0\n".encode())
    with pytest.raises(ConfigError, match=rf"files\.{field}"):
        load_operator_config()


@pytest.mark.parametrize("section", ["profiles", "lifecycle", "files", "tool_groups"])
def test_a_bare_phase4_section_key_yields_the_section_defaults(config_bytes, section: str) -> None:
    config_bytes(f"{section}:\n".encode())
    assert load_operator_config() == OperatorConfig()


def test_lifecycle_and_files_values_load(config_bytes) -> None:
    config_bytes(
        b"lifecycle:\n  allow_remove_unmanaged: true\n  keep_workspace: false\n  keep_logs: false\n"
        b"files:\n  max_bytes: 1024\n  max_list_depth: 2\n  max_list_entries: 50\n"
        b"profiles:\n  require_profile: true\n"
    )
    c = load_operator_config()
    assert c.lifecycle.allow_remove_unmanaged is True
    assert c.lifecycle.keep_workspace is False
    assert c.lifecycle.keep_logs is False
    assert (c.files.max_bytes, c.files.max_list_depth, c.files.max_list_entries) == (1024, 2, 50)
    assert c.profiles.require_profile is True


def test_valid_profile_names_are_accepted(config_bytes) -> None:
    config_bytes(b"profiles:\n  allowed: [local.recon, web, team_a.web-01]\n")
    assert load_operator_config().profiles.allowed == ["local.recon", "web", "team_a.web-01"]


@pytest.mark.parametrize(
    "entries",
    ['[""]', '["a b"]', '["x/y"]', "[web, web]", '[".hidden"]', '["' + "a" * 129 + '"]'],
    ids=["empty", "whitespace", "separator", "duplicate", "leading-dot", "too-long"],
)
def test_a_bad_profile_entry_is_a_named_error(config_bytes, entries: str) -> None:
    config_bytes(f"profiles:\n  allowed: {entries}\n".encode())
    with pytest.raises(ConfigError, match=r"profiles\.allowed"):
        load_operator_config()


# --- D-24 workspace allowlist: resolved and checked at startup ------------------------------------------------------


@pytest.fixture
def invoking_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the sudo-aware home seam at a tmp_path home so `~` never resolves to the real user's home."""
    home = tmp_path / "home" / "u"
    home.mkdir(parents=True)
    monkeypatch.setattr(config_module, "_invoking_home", lambda: home)
    return home


def _allowlist_yaml(*entries: str) -> bytes:
    # JSON strings are YAML double-quoted scalars: `~` stays a string (bare, it is YAML null) and NUL survives.
    items = "".join(f"    - {json.dumps(entry)}\n" for entry in entries)
    return f"lifecycle:\n  workspace_allowlist:\n{items}".encode()


def test_allowlist_entries_are_stored_as_absolute_realpaths(config_bytes, invoking_home: Path, tmp_path: Path) -> None:
    config_bytes(_allowlist_yaml("~/eng", f"{tmp_path}/shared"))
    allowlist = load_operator_config().lifecycle.workspace_allowlist
    assert allowlist == [
        os.path.realpath(invoking_home / "eng"),
        os.path.realpath(tmp_path / "shared"),
    ]
    assert all(Path(entry).is_absolute() for entry in allowlist)


def test_an_empty_allowlist_loads_unchanged(config_bytes, invoking_home: Path) -> None:
    config_bytes(b"lifecycle:\n  workspace_allowlist: []\n")
    assert load_operator_config().lifecycle.workspace_allowlist == []


@pytest.mark.parametrize("which", ["home", "tilde", "root", "home-parent"])
def test_a_dangerous_allowlist_entry_stops_startup_naming_its_index(
    config_bytes, invoking_home: Path, tmp_path: Path, which: str
) -> None:
    dangerous = {"home": str(invoking_home), "tilde": "~", "root": "/", "home-parent": str(invoking_home.parent)}
    config_bytes(_allowlist_yaml(f"{tmp_path}/ok", dangerous[which]))
    with pytest.raises(ConfigError, match=r"lifecycle\.workspace_allowlist\.1"):
        load_operator_config()


@pytest.mark.parametrize("raw", ["eng", "~root/x", "/opt/a\x00b"], ids=["relative", "tilde-user", "nul"])
def test_a_malformed_allowlist_entry_stops_startup_naming_its_index(
    config_bytes, invoking_home: Path, raw: str
) -> None:
    config_bytes(_allowlist_yaml(raw))
    with pytest.raises(ConfigError, match=r"lifecycle\.workspace_allowlist\.0"):
        load_operator_config()


def test_an_allowlist_entry_inside_another_stops_startup_naming_both_indexes(
    config_bytes, invoking_home: Path, tmp_path: Path
) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    config_bytes(_allowlist_yaml(f"{tmp_path}/a", f"{tmp_path}/a/b"))
    with pytest.raises(ConfigError, match=r"workspace_allowlist\.1.*workspace_allowlist\.0"):
        load_operator_config()
    # The reverse order names index 0 as the inner entry.
    config_bytes(_allowlist_yaml(f"{tmp_path}/a/b", f"{tmp_path}/a"))
    with pytest.raises(ConfigError, match=r"workspace_allowlist\.0.*workspace_allowlist\.1"):
        load_operator_config()


def test_an_allowlist_entry_inside_another_through_a_symlinked_parent_stops_startup(
    config_bytes, invoking_home: Path, tmp_path: Path
) -> None:
    # /link -> /a, so /link/b is /a/b once resolved: the comparison uses resolved paths.
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "a")
    config_bytes(_allowlist_yaml(f"{tmp_path}/a", f"{tmp_path}/link/b"))
    with pytest.raises(ConfigError, match=r"workspace_allowlist\.1.*workspace_allowlist\.0"):
        load_operator_config()


def test_sibling_and_duplicate_allowlist_entries_still_load(config_bytes, invoking_home: Path, tmp_path: Path) -> None:
    config_bytes(_allowlist_yaml(f"{tmp_path}/a", f"{tmp_path}/ab"))
    assert load_operator_config().lifecycle.workspace_allowlist == [
        os.path.realpath(tmp_path / "a"),
        os.path.realpath(tmp_path / "ab"),
    ]
    config_bytes(_allowlist_yaml(f"{tmp_path}/a", f"{tmp_path}/a"))
    assert load_operator_config().lifecycle.workspace_allowlist == [os.path.realpath(tmp_path / "a")] * 2


# IN-08: realpath keeps the operator's spelling, so on a case-insensitive host (macOS, Windows) /Work and /work/sub
# are nested on disk. Every test forces the host rule explicitly, so Linux CI and a macOS laptop agree. The
# directories are never created: on a case-insensitive filesystem Work and work would be one directory.


def test_a_case_differing_nested_allowlist_pair_stops_startup_on_a_case_insensitive_host(
    config_bytes, invoking_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_paths, "CASE_INSENSITIVE_HOST", True)
    config_bytes(_allowlist_yaml(f"{tmp_path}/Work", f"{tmp_path}/work/sub"))
    with pytest.raises(ConfigError, match=r"workspace_allowlist\.1.*workspace_allowlist\.0"):
        load_operator_config()


def test_a_case_differing_allowlist_pair_loads_on_a_case_sensitive_host(
    config_bytes, invoking_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_paths, "CASE_INSENSITIVE_HOST", False)
    config_bytes(_allowlist_yaml(f"{tmp_path}/Work", f"{tmp_path}/work/sub"))
    assert load_operator_config().lifecycle.workspace_allowlist == [
        os.path.realpath(tmp_path / "Work"),
        os.path.realpath(tmp_path / "work" / "sub"),
    ]


def test_resolve_carries_the_phase4_sections_through(config_bytes, invoking_home: Path) -> None:
    config_bytes(b"files:\n  max_bytes: 4096\n" + _allowlist_yaml("~/eng"))
    resolved = resolve(parse_args([]), load_operator_config())
    assert resolved.files.max_bytes == 4096
    assert resolved.lifecycle.workspace_allowlist == [os.path.realpath(invoking_home / "eng")]


def test_the_loader_uses_the_safe_yaml_loader() -> None:
    # Config injection (T-02-06): only yaml.safe_load is used, never yaml.load / yaml.unsafe_load.
    import inspect

    source = inspect.getsource(config_module)
    assert "yaml.safe_load" in source
    assert "yaml.load(" not in source
    assert "unsafe_load" not in source


# --- main() fails fast on an invalid file (CONF-04) -----------------------------------------------------------------


@pytest.fixture
def restore_root_logging():
    """Save and restore the root/package logger state main() mutates via configure_stderr_logging."""
    root = logging.getLogger()
    package = logging.getLogger("exegol_mcp")
    saved = (list(root.handlers), root.level, package.level)
    yield
    handlers, root_level, package_level = saved
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(root_level)
    package.setLevel(package_level)


@pytest.mark.usefixtures("restore_root_logging")
def test_main_with_an_invalid_file_returns_1_with_one_stderr_line_and_clean_stdout(
    config_bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    from exegol_mcp.main import main

    config_bytes(b"execution:\n  timeout_default_s: 0\n")

    status = main(["--type", "stdio"])

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # stdio keeps stdout clean (SEC-08)
    assert "timeout_default_s" in captured.err  # the message names the offending field
    assert "Traceback" not in captured.err  # a clean error, never a raised traceback


# --- CLI > file > default resolution (D-05/D-06) --------------------------------------------------------------------


def test_mirrored_flags_default_to_none_so_the_file_can_supply_them() -> None:
    # A bare launch leaves type/port/auth unset so resolve() can tell "set on the CLI" from "left to the file".
    args = parse_args([])
    assert args.type is None
    assert args.port is None
    assert args.auth is None
    # Explicit flags still parse to their values.
    explicit = parse_args(["-t", "stdio", "-p", "8080", "--auth", "bearer"])
    assert (explicit.type, explicit.port, explicit.auth) == ("stdio", 8080, "bearer")


def test_resolve_a_bare_launch_keeps_the_stable_http_2187_bearer_defaults() -> None:
    resolved = resolve(parse_args([]), OperatorConfig())
    assert resolved.transport == "http"
    assert resolved.port == DEFAULT_PORT == 2187
    assert resolved.auth == "bearer"
    assert resolved.update_check is True


def test_resolve_cli_beats_file_and_file_beats_default_for_port() -> None:
    assert resolve(parse_args(["-p", "9000"]), OperatorConfig(port=2187)).port == 9000  # CLI wins
    assert resolve(parse_args([]), OperatorConfig(port=8080)).port == 8080  # file wins over default
    assert resolve(parse_args([]), OperatorConfig()).port == 2187  # built-in default


def test_resolve_transport_and_auth_follow_the_same_precedence() -> None:
    assert resolve(parse_args(["-t", "stdio"]), OperatorConfig(transport="http")).transport == "stdio"
    assert resolve(parse_args([]), OperatorConfig(transport="stdio")).transport == "stdio"
    # auth: a bare launch resolves to bearer; a CLI --auth bearer wins over a file auth key.
    assert resolve(parse_args([]), OperatorConfig()).auth == "bearer"
    assert resolve(parse_args(["--auth", "bearer"]), OperatorConfig(auth="none")).auth == "bearer"


# --- SEC-04 / D-12 / D-13: --auth none removed from the CLI; no-auth fails closed to bearer ------------------------


def test_cli_auth_none_exits_non_zero_with_guidance(capsys: pytest.CaptureFixture[str]) -> None:
    # D-12: the removed `--auth none` form exits non-zero and the guidance names the config key and bearer flow.
    for argv in (["--auth", "none"], ["--auth=none"]):
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert "security.allow_no_auth" in err
        assert "bearer" in err


def test_resolve_no_auth_fails_closed_to_bearer_without_the_opt_in() -> None:
    # D-13: a file auth="none" without the opt-in coerces to bearer (fail closed).
    assert resolve(parse_args([]), OperatorConfig(auth="none")).auth == "bearer"


def test_resolve_no_auth_survives_only_with_the_explicit_opt_in() -> None:
    # D-13: file auth="none" + allow_no_auth=True is the one deliberate route to no-auth.
    cfg = OperatorConfig(auth="none", security=SecuritySettings(allow_no_auth=True))
    assert resolve(parse_args([]), cfg).auth == "none"


def test_resolve_cli_bearer_beats_allow_no_auth() -> None:
    # D-13: CLI --auth bearer wins over the config opt-in (CLI > config, the safe direction).
    cfg = OperatorConfig(auth="none", security=SecuritySettings(allow_no_auth=True))
    assert resolve(parse_args(["--auth", "bearer"]), cfg).auth == "bearer"


def test_security_settings_defaults_closed_and_forbids_extra() -> None:
    assert OperatorConfig().security.allow_no_auth is False
    assert SecuritySettings.model_config.get("extra") == "forbid"


def test_resolve_update_check_precedence_with_and_without_the_flag() -> None:
    # --no-update-check can only force-disable, and it wins over a file that enables it.
    assert resolve(parse_args(["--no-update-check"]), OperatorConfig(update_check=True)).update_check is False
    # Absent flag: the file value applies.
    assert resolve(parse_args([]), OperatorConfig(update_check=False)).update_check is False
    # Absent from both: enabled by default.
    assert resolve(parse_args([]), OperatorConfig()).update_check is True


def test_resolve_carries_the_file_execution_section_through_unchanged() -> None:
    file_cfg = load_operator_config_from(b"execution:\n  timeout_default_s: 7\n")
    resolved = resolve(parse_args([]), file_cfg)
    assert resolved.execution.timeout_default_s == 7


def load_operator_config_from(raw: bytes) -> OperatorConfig:
    """Parse raw config bytes without a monkeypatch fixture (used where a direct model is handier)."""
    import yaml

    return OperatorConfig.model_validate(yaml.safe_load(raw.decode("utf-8")) or {})
