"""Bootstrap tests: readiness gate and EULA flow, versions/banner/update check, server assembly (plan 01-05).

Docker-free and network-free: the host probe, the PyPI fetch and the EULA datastore are fakes, so nothing here
touches the real Docker daemon, ~/.exegol or the bearer secret.
"""

import errno
import io
import json
import os
import re
import sys
import sysconfig
import time
import types
from argparse import Namespace
from pathlib import Path

import pytest

from exegol_mcp import __version__
from exegol_mcp.adapter import mount_paths, selfcheck
from exegol_mcp.adapter.port import ConfigStatus, ExegolMountDir, HostProbe
from exegol_mcp.bootstrap import eula, readiness, secret_store, versions
from exegol_mcp.bootstrap.cli import parse_args

# --- Fakes ----------------------------------------------------------------------------------------------------------


class FakeHostProbe:
    """HostProbe with every host state settable, recording EULA writes and the order of the checks."""

    def __init__(
        self,
        *,
        eula_accepted: bool = True,
        docker_error: Exception | None = None,
        config_refused: bool = False,
        config_error: Exception | None = None,
        patch_point_check=None,
        profile_problems: list[str] | None = None,
        mount_dirs: list[ExegolMountDir] | None = None,
        mount_error: Exception | None = None,
    ) -> None:
        self.eula_accepted = eula_accepted
        self.mount_dirs = list(mount_dirs or [])
        self.mount_error = mount_error
        self.mount_profiles: list[list[str]] = []
        self.profile_problems = list(profile_problems or [])
        self.validated_profiles: list[list[str]] = []
        self.docker_error = docker_error
        self.config_refused = config_refused
        self.config_error = config_error
        self.patch_point_check = patch_point_check
        self.eula_writes: list[bool] = []
        self.calls: list[str] = []

    def assert_patch_points(self) -> None:
        self.calls.append("patch-points")
        if self.patch_point_check is not None:
            self.patch_point_check()

    def probe_docker(self) -> str:
        self.calls.append("docker")
        if self.docker_error is not None:
            raise self.docker_error
        return "reachable (Docker 29.8.2)"

    def ping_docker(self) -> str:
        self.calls.append("ping")
        if self.docker_error is not None:
            raise self.docker_error
        return "reachable (Docker 29.8.2)"

    def load_config(self) -> ConfigStatus:
        self.calls.append("config")
        if self.config_error is not None:
            raise self.config_error
        return ConfigStatus(path="/home/user/.exegol/config.yml", refused=self.config_refused)

    def exegol_config_dir(self) -> str:
        return "/home/user/.exegol"

    def validate_profiles(self, names) -> list[str]:
        self.calls.append("profiles")
        self.validated_profiles.append(list(names))
        return list(self.profile_problems)

    def exegol_mount_dirs(self, profiles) -> list[ExegolMountDir]:
        self.calls.append("mounts")
        self.mount_profiles.append(list(profiles))
        if self.mount_error is not None:
            raise self.mount_error
        return list(self.mount_dirs)

    def is_eula_accepted(self) -> bool:
        self.calls.append("eula")
        return self.eula_accepted

    def update_eula(self, accepted: bool) -> None:
        self.eula_writes.append(accepted)
        self.eula_accepted = accepted


def _args(*argv: str) -> Namespace:
    return parse_args(list(argv))


def _never_prompt() -> bool:
    raise AssertionError("the EULA prompt must not run here")


@pytest.fixture
def tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend stdin is a terminal (pytest's stdin is not)."""

    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _Tty())


@pytest.fixture
def no_tty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO())


# --- Task 1: readiness gate ----------------------------------------------------------------------------------------


def test_readiness_the_fake_probe_matches_the_host_probe_protocol() -> None:
    assert isinstance(FakeHostProbe(), HostProbe)


def test_readiness_the_real_probe_matches_the_host_probe_protocol() -> None:
    from exegol_mcp.adapter.host import RealHostProbe

    assert isinstance(RealHostProbe(), HostProbe)


async def test_readiness_passes_and_reports_what_it_checked() -> None:
    probe = FakeHostProbe()

    report = await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert report.docker_status == "reachable (Docker 29.8.2)"
    assert report.exegol_config_dir == "/home/user/.exegol"
    assert report.exegol_config_file == "/home/user/.exegol/config.yml"
    assert report.eula == "accepted"
    assert probe.calls == ["patch-points", "docker", "config", "eula"]


async def test_readiness_a_missing_patch_point_fails_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from exegol.console.TUI import ExegolTUI

    monkeypatch.delattr(ExegolTUI, "downloadDockerLayer")
    probe = FakeHostProbe(patch_point_check=selfcheck.assert_patch_points)

    with pytest.raises(readiness.ReadinessError, match=re.escape("ExegolTUI.downloadDockerLayer")) as raised:
        await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert raised.value.check == "patch-points"
    # The gate stops at the first failure: Docker is never probed with a broken SDK.
    assert probe.calls == ["patch-points"]


async def test_readiness_unreachable_docker_fails_with_the_exegol_info_guidance() -> None:
    probe = FakeHostProbe(docker_error=RuntimeError("Unable to connect to docker (from env config)"))

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert raised.value.check == "docker"
    message = str(raised.value)
    assert "Docker is not reachable" in message
    assert "exegol info" in message
    assert "Unable to connect to docker" in message
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "config" not in probe.calls


async def test_readiness_an_unreadable_config_fails_with_the_upgrade_message() -> None:
    probe = FakeHostProbe(config_refused=True)

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert raised.value.check == "config"
    message = str(raised.value)
    assert "/home/user/.exegol/config.yml" in message
    assert "upgrade the Exegol wrapper and run it once" in message
    assert "never migrates" in message
    assert "eula" not in probe.calls


async def test_readiness_a_config_loader_error_fails_with_the_upgrade_message() -> None:
    probe = FakeHostProbe(config_error=RuntimeError("Invalid value for network_default_netmask"))

    with pytest.raises(readiness.ReadinessError, match="upgrade the Exegol wrapper") as raised:
        await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert raised.value.check == "config"
    assert "network_default_netmask" in str(raised.value)


async def test_readiness_errors_are_distinct_per_check() -> None:
    failures = {
        "docker": FakeHostProbe(docker_error=OSError("socket")),
        "config": FakeHostProbe(config_refused=True),
        "eula": FakeHostProbe(eula_accepted=False),
    }
    messages = set()
    for check, probe in failures.items():
        with pytest.raises(readiness.ReadinessError) as raised:
            await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)
        assert raised.value.check == check
        messages.add(str(raised.value).split(".")[0])
    assert len(messages) == 3


async def test_readiness_a_profile_problem_stops_startup_after_config_before_eula() -> None:
    probe = FakeHostProbe(profile_problems=["Profile 'ops': not found"])

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt, profiles=["web", "ops"])

    assert raised.value.check == "profiles"
    message = str(raised.value)
    assert "ops" in message
    assert "not found" in message
    assert "profiles.allowed" in message
    assert probe.validated_profiles == [["web", "ops"]]
    # After the config step and before the EULA step; the EULA is never reached.
    assert probe.calls == ["patch-points", "docker", "config", "profiles"]


async def test_readiness_resolvable_profiles_pass_and_run_between_config_and_eula() -> None:
    probe = FakeHostProbe()

    await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt, profiles=("web",))

    assert probe.calls == ["patch-points", "docker", "config", "profiles", "eula"]
    assert probe.validated_profiles == [["web"]]


async def test_readiness_an_empty_allowlist_never_validates_profiles() -> None:
    probe = FakeHostProbe(profile_problems=["must not be asked"])

    await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt, profiles=[])
    await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert "profiles" not in probe.calls
    assert probe.validated_profiles == []


async def test_readiness_recheck_never_revalidates_profiles() -> None:
    probe = FakeHostProbe(profile_problems=["Profile 'ops': not found"])

    await readiness.recheck(probe, accept_eula=True)

    assert "profiles" not in probe.calls


async def test_readiness_the_error_is_a_runtime_error_for_the_tool_boundary() -> None:
    probe = FakeHostProbe(docker_error=ConnectionError("daemon gone"))

    with pytest.raises(RuntimeError, match="Docker is not reachable"):
        await readiness.recheck(probe, accept_eula=False)


async def test_readiness_recheck_reports_a_revoked_eula_and_never_prompts() -> None:
    probe = FakeHostProbe(eula_accepted=False)

    with pytest.raises(readiness.ReadinessError, match="no longer accepted") as raised:
        await readiness.recheck(probe, accept_eula=False)
    assert raised.value.check == "eula"

    # With --accept-eula the run stays accepted whatever the datastore says.
    await readiness.recheck(probe, accept_eula=True)
    assert probe.eula_writes == []


async def test_readiness_recheck_passes_when_the_host_is_still_ready() -> None:
    probe = FakeHostProbe()
    await readiness.recheck(probe, accept_eula=False)
    assert probe.calls == ["ping", "eula"]


# --- Task 1: EULA flow (D-05/D-06) ---------------------------------------------------------------------------------


@pytest.mark.usefixtures("no_tty_stdin")
async def test_eula_no_tty_not_accepted_no_flag_fails_at_startup_with_the_d05_message() -> None:
    probe = FakeHostProbe(eula_accepted=False)

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(probe, _args(), prompt=_never_prompt)

    assert raised.value.check == "eula"
    message = str(raised.value)
    assert "Run `exegol-mcp` or `exegol` once in a terminal" in message
    assert "--accept-eula" in message
    assert eula.EULA_URL in message
    assert probe.eula_writes == []


@pytest.mark.usefixtures("tty_stdin")
async def test_eula_a_stdio_launch_never_prompts_even_on_a_terminal() -> None:
    probe = FakeHostProbe(eula_accepted=False)

    with pytest.raises(readiness.ReadinessError, match="--accept-eula") as raised:
        await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert raised.value.check == "eula"
    assert probe.eula_writes == []


@pytest.mark.usefixtures("no_tty_stdin")
async def test_eula_accept_eula_flag_passes_headless_without_writing_the_datastore() -> None:
    probe = FakeHostProbe(eula_accepted=False)

    report = await readiness.check_readiness(probe, _args("--type", "stdio", "--accept-eula"), prompt=_never_prompt)

    assert report.eula == "flag"
    assert probe.eula_writes == []
    assert "eula" not in probe.calls


@pytest.mark.usefixtures("no_tty_stdin")
async def test_eula_a_prior_acceptance_passes_headless() -> None:
    probe = FakeHostProbe(eula_accepted=True)

    report = await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt)

    assert report.eula == "accepted"


@pytest.mark.usefixtures("tty_stdin")
async def test_eula_a_terminal_acceptance_is_recorded_with_update_eula() -> None:
    probe = FakeHostProbe(eula_accepted=False)
    asked: list[bool] = []

    def accept() -> bool:
        asked.append(True)
        return True

    report = await readiness.check_readiness(probe, _args("--print-config"), prompt=accept)

    assert asked == [True]
    assert report.eula == "prompted"
    assert probe.eula_writes == [True]


@pytest.mark.usefixtures("tty_stdin")
async def test_eula_a_declined_terminal_prompt_fails_and_records_nothing() -> None:
    probe = FakeHostProbe(eula_accepted=False)

    with pytest.raises(readiness.ReadinessError, match="was not accepted") as raised:
        await readiness.check_readiness(probe, _args(), prompt=lambda: False)

    assert raised.value.check == "eula"
    assert probe.eula_writes == []


def test_eula_the_prompt_writes_to_stderr_only(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))

    assert eula.ask_on_stderr() is True

    captured = capsys.readouterr()
    assert captured.out == ""
    assert eula.EULA_URL in captured.err
    assert "accepted the EULA" in captured.err


def test_eula_the_prompt_defaults_to_no(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))

    assert eula.ask_on_stderr() is False
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("argv", "tty", "expected"),
    [
        (["--type", "stdio"], True, False),
        (["--type", "http"], True, True),
        (["--type", "http"], False, False),
        (["--print-config"], True, True),
    ],
)
def test_eula_a_prompt_needs_a_terminal_and_never_runs_in_stdio(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], tty: bool, expected: bool
) -> None:
    class _Stdin(io.StringIO):
        def isatty(self) -> bool:
            return tty

    monkeypatch.setattr(sys, "stdin", _Stdin())
    assert eula.can_prompt(_args(*argv)) is expected


# --- Task 2: versions, banner, update check (D-10..D-13) -----------------------------------------------------------


@pytest.fixture
def config_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point exegol-mcp's config dir at a temp dir: the real one (and the bearer secret) is never touched."""
    directory = tmp_path / "exegol-mcp"
    monkeypatch.setattr(secret_store, "user_config_dir", lambda: directory)
    return directory


def _versions(wrapper: str | None = "6.0.2", library: str | None = "6.0.2") -> versions.Versions:
    return versions.Versions(
        exegol_mcp="2.0.0",
        exegol_library=library,
        exegol_wrapper=wrapper,
        docker_status="reachable (Docker 29.8.2)",
        config_dirs=["/home/user/.exegol", "/home/user/.config/exegol-mcp"],
    )


def test_versions_gather_reports_versions_docker_and_config_dirs(config_home, monkeypatch) -> None:
    monkeypatch.setattr(versions, "detect_wrapper_version", lambda: "6.0.1")

    gathered = versions.gather_versions("reachable (Docker 29.8.2)", "/home/user/.exegol")

    assert gathered.exegol_mcp == __version__
    assert gathered.exegol_library == versions.library_version()
    assert gathered.exegol_wrapper == "6.0.1"
    assert gathered.docker_status == "reachable (Docker 29.8.2)"
    assert gathered.config_dirs == ["/home/user/.exegol", str(config_home)]


def test_versions_a_wrapper_mismatch_is_information_only() -> None:
    assert _versions(wrapper="6.0.1").compatibility == "mismatch"
    assert _versions(wrapper="6.0.2").compatibility == "match"
    assert _versions(wrapper=None).compatibility == "unknown"


def test_versions_server_info_report_carries_the_banner_facts_and_the_notice() -> None:
    report = versions.server_info_report(_versions(wrapper="6.0.1"), "reachable", "exegol-mcp 9.9.9 is available")

    assert report.exegol_mcp_version == "2.0.0"
    assert report.exegol_library_version == "6.0.2"
    assert report.exegol_wrapper_version == "6.0.1"
    assert report.docker_status == "reachable"
    assert report.config_dirs == ["/home/user/.exegol", "/home/user/.config/exegol-mcp"]
    assert report.update_notice == "exegol-mcp 9.9.9 is available"


def _fake_environment(root, version: str):
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    site = root / "lib" / "python3.12" / "site-packages" / f"exegol-{version}.dist-info"
    site.mkdir(parents=True)
    (site / "METADATA").write_text(f"Metadata-Version: 2.1\nName: exegol\nVersion: {version}\n")
    # exegol-mcp's own metadata must never be mistaken for the wrapper's.
    other = root / "lib" / "python3.12" / "site-packages" / "exegol_mcp-2.0.0.dist-info"
    other.mkdir(parents=True)
    (other / "METADATA").write_text("Name: exegol-mcp\nVersion: 2.0.0\n")
    return bin_dir


def test_versions_the_wrapper_version_is_read_from_its_environment(tmp_path) -> None:
    bin_dir = _fake_environment(tmp_path / "pipx" / "venvs" / "exegol", "6.0.1")
    python = bin_dir / "python"
    python.write_text("")
    script = bin_dir / "exegol"
    script.write_text(f"#!{python}\nfrom exegol.manager.ExegolController import main\n")

    assert versions.detect_wrapper_version(which=lambda _name: str(script)) == "6.0.1"


def test_versions_the_wrapper_version_is_read_from_a_git_clone(tmp_path) -> None:
    (tmp_path / "exegol" / "config").mkdir(parents=True)
    (tmp_path / "exegol" / "config" / "ConstantConfig.py").write_text('__version__ = "6.1.0b1"\n')
    script = tmp_path / "exegol.py"
    script.write_text("#!/usr/bin/env python3\n")

    assert versions.detect_wrapper_version(which=lambda _name: str(script)) == "6.1.0b1"


def test_versions_no_wrapper_on_the_path_is_none(tmp_path) -> None:
    assert versions.detect_wrapper_version(which=lambda _name: None) is None
    assert versions.detect_wrapper_version(which=lambda _name: str(tmp_path / "missing")) is None


def test_banner_stdio_mode_writes_to_stderr_only(capsys: pytest.CaptureFixture) -> None:
    versions.print_banner(_args("--type", "stdio"), _versions(wrapper="6.0.1"))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exegol-mcp 2.0.0" in captured.err
    assert "6.0.1" in captured.err
    assert "information only" in captured.err
    assert "reachable (Docker 29.8.2)" in captured.err
    assert "/home/user/.exegol" in captured.err
    assert "stdio" in captured.err


def test_banner_stdio_console_is_stderr_and_http_console_is_stdout() -> None:
    assert versions.banner_console(_args("--type", "stdio")).stderr is True
    assert versions.banner_console(_args("--type", "http")).stderr is False


def test_banner_http_mode_prints_to_the_console_with_the_endpoint(capsys: pytest.CaptureFixture) -> None:
    # main() writes the resolved transport/port/auth back onto the Namespace before the banner, so the banner
    # always sees concrete values (the mirrored flags now default to None).
    versions.print_banner(_args("--type", "http", "--port", "4242", "--auth", "bearer", "--no-update-check"), _versions())

    out = capsys.readouterr().out
    assert "http://127.0.0.1:4242/mcp (bearer auth)" in out
    assert "disabled (--no-update-check)" in out


def test_banner_client_template_redacts_the_secret_unless_asked_or_new() -> None:
    # A resolved HTTP/bearer Namespace (what main() hands these functions after CLI>file>default resolution).
    http_bearer = ("--type", "http", "--port", "2187", "--auth", "bearer")
    redacted = versions.client_config_template(_args(*http_bearer), "s3cr3t", created=False)
    assert "s3cr3t" not in redacted
    assert "<redacted> (use --print-config to reveal)" in redacted
    assert '"url": "http://127.0.0.1:2187/mcp"' in redacted

    assert "Bearer s3cr3t" in versions.client_config_template(
        _args(*http_bearer, "--print-config"), "s3cr3t", created=False
    )
    assert "Bearer s3cr3t" in versions.client_config_template(_args(*http_bearer), "s3cr3t", created=True)


def test_banner_client_template_without_auth_and_for_stdio() -> None:
    # No CLI path to no-auth now (D-12); it is resolved from the security.allow_no_auth config key, so build the
    # no-auth Namespace by parsing bare flags then setting auth, simulating that resolution.
    no_auth_args = _args()
    no_auth_args.auth = "none"
    no_auth = versions.client_config_template(no_auth_args, None, created=False)
    assert "Authorization" not in no_auth

    stdio = versions.client_config_template(_args("--type", "stdio"), None, created=False)
    command, args = versions.stdio_launch()
    assert json.loads(stdio.split(": ", 1)[1]) == {"type": "stdio", "command": command, "args": args}


# --- CR-04: the stdio launch exegol-mcp prints keeps the client's working directory off the import path -----------


def _stdio_entry() -> dict:
    template = versions.client_config_template(_args("--type", "stdio"), None, created=False)
    return json.loads(template.split(": ", 1)[1])


def _scripts_dir(monkeypatch: pytest.MonkeyPatch, scripts: Path) -> None:
    """Make sysconfig.get_path("scripts") return scripts; every other name keeps the interpreter's own answer."""
    original = sysconfig.get_path

    def _get_path(name: str, *args, **kwargs) -> str:
        return str(scripts) if name == "scripts" else original(name, *args, **kwargs)

    monkeypatch.setattr(sysconfig, "get_path", _get_path)


def _console_script_file(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / ("exegol-mcp.exe" if os.name == "nt" else "exegol-mcp")
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    return script


def _planted_client(tmp_path: Path) -> Path:
    """A client working directory holding a planted rich.py and exegol_mcp package; each exits 42 on import."""
    client = tmp_path / "client"
    (client / "exegol_mcp").mkdir(parents=True)
    for module, name in ((client / "rich.py", "rich.py"), (client / "exegol_mcp" / "__init__.py", "exegol_mcp")):
        module.write_text(f"import sys\nsys.stderr.write('PLANTED {name} ran on the host\\n')\nsys.exit(42)\n")
    return client


def _child_env(**overrides: str) -> dict[str, str]:
    """os.environ without PYTHONSAFEPATH and PYTHONPATH, updated with overrides, for the planted-module subprocesses
    (IN-27).

    PYTHONSAFEPATH on 3.11+ keeps the working directory off sys.path whatever the launch, so a guard test would pass
    with -P or the -c filter removed. An empty PYTHONPATH component puts the absolute working directory on sys.path,
    which no printed launch drops, so a planted module would run for an environmental reason.
    """
    env = {name: value for name, value in os.environ.items() if name not in ("PYTHONSAFEPATH", "PYTHONPATH")}
    env.update(overrides)
    return env


def _cpython_enables_the_user_site(executable: str, cwd: Path) -> bool:
    """Whether a normal start of that interpreter (no -s, -S, -I or PYTHONNOUSERSITE) reads the user site, as CPython
    decides it, so a test never takes that value from the code under test (WR-18).

    The child runs in cwd with _child_env(), PYTHONNOUSERSITE and __PYVENV_LAUNCHER__ removed, and reports
    site.ENABLE_USER_SITE.
    """
    dropped = ("PYTHONNOUSERSITE", "__PYVENV_LAUNCHER__")
    env = {name: value for name, value in _child_env().items() if name not in dropped}
    result = subprocess.run(
        [executable, "-c", "import site; print(site.ENABLE_USER_SITE)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
        env=env,
    )
    reported = result.stdout.strip()
    assert reported in ("True", "False", "None"), result
    return reported == "True"


def test_the_stdio_template_child_env_drops_pythonsafepath_and_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setenv("PYTHONPATH", os.pathsep)
    monkeypatch.setenv("EXEGOL_MCP_PROBE", "kept")

    env = _child_env(HOME="h")

    assert "PYTHONSAFEPATH" not in env
    assert "PYTHONPATH" not in env
    assert env["EXEGOL_MCP_PROBE"] == "kept"
    assert env["HOME"] == "h"


def test_the_stdio_template_launches_this_environments_console_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _console_script_file(_real_root(tmp_path) / "scripts")
    _scripts_dir(monkeypatch, script.parent)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    assert _stdio_entry() == {"type": "stdio", "command": str(script), "args": ["--type", "stdio"]}
    assert '"-m"' not in versions.client_config_template(_args("--type", "stdio"), None, created=False)


def test_the_stdio_template_prefers_the_running_console_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _real_root(tmp_path)
    _scripts_dir(monkeypatch, _console_script_file(root / "scripts").parent)
    running = _console_script_file(root / "other")
    monkeypatch.chdir(root)
    # A relative sys.argv[0] is made absolute.
    monkeypatch.setattr(sys, "argv", [os.path.join("other", running.name), "--type", "stdio"])

    assert _stdio_entry() == {"type": "stdio", "command": str(running), "args": ["--type", "stdio"]}


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks")
@pytest.mark.parametrize("link_in", ["argv0-link", "scripts-dir-link"])
def test_the_stdio_template_resolves_a_symlinked_console_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, link_in: str
) -> None:
    # WR-15: a pipx or uv tool link in ~/.local/bin prints the script inside the tool's environment.
    root = _real_root(tmp_path)
    target = _console_script_file(root / "tool" / "venv" / "bin")
    (root / "localbin").mkdir()
    link = root / "localbin" / "exegol-mcp"
    link.symlink_to(target)
    if link_in == "argv0-link":
        empty = root / "scripts"
        empty.mkdir()
        _scripts_dir(monkeypatch, empty)
        monkeypatch.setattr(sys, "argv", [str(link), "--type", "stdio"])
    else:
        _scripts_dir(monkeypatch, root / "localbin")
        monkeypatch.setattr(sys, "argv", ["pytest"])

    entry = _stdio_entry()
    assert entry == {"type": "stdio", "command": str(target), "args": ["--type", "stdio"]}
    assert entry["command"] != str(link)


def _without_a_console_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, safe_path_flag: bool) -> None:
    """No console script in sys.argv[0] or the scripts directory, and the fallback branch forced (WR-16)."""
    empty = _real_root(tmp_path) / "scripts"
    empty.mkdir()
    _scripts_dir(monkeypatch, empty)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(versions, "_safe_path_flag_supported", lambda: safe_path_flag)


_FALLBACK_BRANCHES = [
    pytest.param(True, id="safe-path-flag"),
    pytest.param(False, id="cwd-free-launcher"),
]


@pytest.mark.parametrize("safe_path_flag", _FALLBACK_BRANCHES)
def test_the_stdio_template_fallback_drops_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, safe_path_flag: bool
) -> None:
    assert versions._safe_path_flag_supported() is (sys.version_info >= (3, 11))
    _without_a_console_script(monkeypatch, tmp_path, safe_path_flag)

    if safe_path_flag:
        expected = ["-P", "-m", "exegol_mcp", "--type", "stdio"]
    else:
        expected = ["-c", versions._CWD_FREE_LAUNCHER, "--type", "stdio"]
    assert _stdio_entry() == {"type": "stdio", "command": sys.executable, "args": expected}
    # Idempotency: stdio_launch only reads the process state.
    assert versions.stdio_launch() == versions.stdio_launch()


@pytest.mark.skipif(sys.platform == "win32", reason="the planted modules are POSIX-shaped")
@pytest.mark.parametrize(
    "safe_path_flag",
    [
        pytest.param(
            True,
            id="safe-path-flag",
            marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="-P exists from Python 3.11"),
        ),
        pytest.param(False, id="cwd-free-launcher"),
    ],
)
def test_the_stdio_template_fallback_launch_never_imports_from_the_client_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, safe_path_flag: bool
) -> None:
    import subprocess

    client = _planted_client(tmp_path)
    _without_a_console_script(monkeypatch, tmp_path, safe_path_flag)
    entry = _stdio_entry()
    assert entry["command"] == sys.executable

    launched = subprocess.run(
        [entry["command"], *entry["args"], "--help"],
        cwd=client,
        env=_child_env(HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "cfg")),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert launched.returncode == 0, launched.stderr
    assert "usage: exegol-mcp" in launched.stdout
    assert "PLANTED" not in launched.stdout and "PLANTED" not in launched.stderr


def test_the_stdio_template_cwd_free_launcher_drops_only_the_working_directory(tmp_path: Path) -> None:
    import subprocess

    clean = tmp_path / "clean"
    clean.mkdir()
    # A 3.11+ host setting PYTHONSAFEPATH would hide the '' entry the baseline must show.
    env = _child_env()

    def _sys_path(code: str) -> list[str]:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=clean,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    baseline = _sys_path("import json, sys; print(json.dumps(sys.path))")
    launched = _sys_path(versions._DROP_WORKING_DIRECTORY + "; import json; print(json.dumps(sys.path))")
    assert baseline[0] == ""
    # Empty and ordering edges: only the empty entries go, every other entry keeps its place.
    assert launched == [entry for entry in baseline if entry]

    assert versions._CWD_FREE_LAUNCHER == (
        versions._DROP_WORKING_DIRECTORY + "; from exegol_mcp.main import main; sys.exit(main())"
    )
    assert versions._DROP_WORKING_DIRECTORY.startswith("import sys;")
    # Encoding edge: the printed JSON shows the launcher verbatim.
    assert '"' not in versions._CWD_FREE_LAUNCHER and "\\" not in versions._CWD_FREE_LAUNCHER
    assert json.dumps(versions._CWD_FREE_LAUNCHER) == f'"{versions._CWD_FREE_LAUNCHER}"'


@pytest.mark.skipif(sys.platform == "win32", reason="the planted modules are POSIX-shaped")
def test_the_stdio_template_cwd_free_launcher_exits_with_mains_status(tmp_path: Path) -> None:
    import subprocess

    client = _planted_client(tmp_path)
    home = tmp_path / "home"
    (home / ".exegol-mcp").mkdir(parents=True)
    (home / ".exegol-mcp" / "config.yml").write_text("bogus_key: 1\n")

    launched = subprocess.run(
        [sys.executable, "-c", versions._CWD_FREE_LAUNCHER, "--type", "stdio"],
        cwd=client,
        env=_child_env(HOME=str(home), XDG_CONFIG_HOME=str(home / "cfg")),
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # main() returns 1 for an invalid config.yml, and the launcher exits with it; stdout stays clean (SEC-08).
    assert launched.returncode == 1, launched.stderr
    assert launched.stdout == ""
    assert "config.yml" in launched.stderr
    assert "PLANTED" not in launched.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="the planted module and console script are POSIX-shaped")
def test_the_stdio_template_launch_never_imports_from_the_client_working_directory(tmp_path: Path) -> None:
    import subprocess

    client = tmp_path / "client"
    client.mkdir()
    (client / "rich.py").write_text(
        "import sys\nsys.stderr.write('PLANTED rich.py ran on the host\\n')\nsys.exit(42)\n"
    )
    env = _child_env(HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "cfg"))
    entry = _stdio_entry()
    # This environment's console script, installed by uv sync, not the interpreter.
    assert entry["command"] != sys.executable
    assert os.path.isfile(entry["command"])

    generated = subprocess.run(
        [entry["command"], *entry["args"], "--help"],
        cwd=client,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert generated.returncode == 0, generated.stderr
    assert "usage:" in generated.stdout
    assert "PLANTED" not in generated.stdout and "PLANTED" not in generated.stderr

    # The mechanism the verifier reproduced: under python -m, the client's working directory is first on sys.path.
    module_launch = subprocess.run(
        [sys.executable, "-m", "exegol_mcp", "--help"],
        cwd=client,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert module_launch.returncode == 42
    assert "PLANTED" in module_launch.stderr


def test_comp03_print_config_emits_working_stdio_and_http_configs() -> None:
    # COMP-03: --print-config emits BOTH a working stdio entry and an HTTP entry whose Authorization header carries
    # the exact honest secret (connects as-is), while the redacted path still hides it.
    http = versions.client_config_template(
        _args("--type", "http", "--port", "2187", "--auth", "bearer", "--print-config"),
        "HONEST-SECRET-xyz",
        created=False,
    )
    assert "Bearer HONEST-SECRET-xyz" in http
    assert '"url": "http://127.0.0.1:2187/mcp"' in http

    stdio = versions.client_config_template(_args("--type", "stdio", "--print-config"), None, created=False)
    assert '"type": "stdio"' in stdio

    redacted = versions.client_config_template(
        _args("--type", "http", "--port", "2187", "--auth", "bearer"), "HONEST-SECRET-xyz", created=False
    )
    assert "HONEST-SECRET-xyz" not in redacted


def test_comp03_no_auth_warning_names_the_config_key(caplog: pytest.LogCaptureFixture) -> None:
    # D-13: no-auth is reachable only via security.allow_no_auth now; the warning must name that key, not --auth none.
    no_auth_args = _args("--type", "http", "--port", "2187")
    no_auth_args.auth = "none"
    with caplog.at_level("WARNING"):
        versions.print_client_config(no_auth_args, None, created=False)
    assert "security.allow_no_auth" in caplog.text
    assert "mode: none" not in caplog.text


def test_banner_print_client_config_is_a_plain_copyable_block(capsys: pytest.CaptureFixture) -> None:
    versions.print_client_config(
        _args("--type", "http", "--port", "2187", "--auth", "bearer", "--print-config"), "s3cr3t", created=False
    )

    out = capsys.readouterr().out
    assert out.startswith("MCP client template:\n")
    assert '"Authorization": "Bearer s3cr3t"' in out
    assert "\x1b[" not in out


def test_update_check_a_failing_fetch_never_raises_and_gives_no_notice(config_home) -> None:
    def offline(_timeout: float) -> str | None:
        raise OSError("Network is unreachable")

    checker = versions.UpdateChecker(enabled=True, fetch=offline)

    assert checker.check() is None
    assert checker.notice is None
    assert not (config_home / versions.UPDATE_CACHE_FILE).exists()


@pytest.mark.parametrize("error", [ValueError("bad json"), KeyError("info"), TimeoutError("timed out")])
def test_update_check_swallows_every_fetch_error(config_home, error: Exception) -> None:
    def failing(_timeout: float) -> str | None:
        raise error

    assert versions.UpdateChecker(enabled=True, fetch=failing).check() is None


def test_update_check_opt_out_short_circuits_without_fetching(config_home) -> None:
    def must_not_fetch(_timeout: float) -> str | None:
        raise AssertionError("the opt-out must skip the network entirely")

    checker = versions.UpdateChecker(enabled=False, fetch=must_not_fetch)

    assert checker.check() is None
    assert checker.start() is None
    assert not config_home.exists()


def test_update_check_reports_a_newer_release_and_caches_it(config_home) -> None:
    timeouts: list[float] = []

    def pypi(timeout: float) -> str | None:
        timeouts.append(timeout)
        return "2.1.0"

    checker = versions.UpdateChecker(enabled=True, current="2.0.0", fetch=pypi, now=lambda: 1000.0)

    notice = checker.check()
    assert notice is not None and "exegol-mcp 2.1.0 is available" in notice
    assert timeouts == [versions.UPDATE_CHECK_TIMEOUT_SECONDS]
    cached = json.loads((config_home / versions.UPDATE_CACHE_FILE).read_text())
    assert cached == {"checked_at": 1000.0, "latest_version": "2.1.0"}


def test_update_check_a_fresh_cache_answers_without_the_network(config_home) -> None:
    config_home.mkdir()
    (config_home / versions.UPDATE_CACHE_FILE).write_text(json.dumps({"checked_at": 1000.0, "latest_version": "2.0.5"}))

    def must_not_fetch(_timeout: float) -> str | None:
        raise AssertionError("a fresh cache must not hit PyPI")

    checker = versions.UpdateChecker(enabled=True, current="2.0.0", fetch=must_not_fetch, now=lambda: 1060.0)
    assert "2.0.5" in (checker.check() or "")


def test_update_check_an_expired_cache_is_refreshed(config_home) -> None:
    config_home.mkdir()
    (config_home / versions.UPDATE_CACHE_FILE).write_text(json.dumps({"checked_at": 0.0, "latest_version": "2.0.5"}))

    checker = versions.UpdateChecker(
        enabled=True, current="2.0.0", fetch=lambda _t: "2.0.0", now=lambda: versions.UPDATE_CACHE_TTL_SECONDS + 1
    )

    assert checker.check() is None


def test_update_check_no_notice_when_up_to_date_or_unparseable(config_home) -> None:
    assert versions.UpdateChecker(enabled=True, current="2.0.0", fetch=lambda _t: "2.0.0").check() is None
    assert versions.UpdateChecker(enabled=True, current="2.0.0", fetch=lambda _t: "1.0.2").check() is None
    assert versions.UpdateChecker(enabled=True, current="2.0.0", fetch=lambda _t: None).check() is None
    assert versions.is_newer("garbage", "2.0.0") is False
    assert versions.is_newer("2.0.10", "2.0.9") is True


def test_update_check_a_pre_release_ranks_below_its_final_release(config_home) -> None:
    # A 2.0.0b1 beta tester must be told when 2.0.0 ships, and never told an older stable release is newer.
    assert versions.is_newer("2.0.0", "2.0.0b1") is True
    assert versions.is_newer("2.0.0rc1", "2.0.0b2") is True
    assert versions.is_newer("2.0.0b2", "2.0.0b1") is True
    assert versions.is_newer("2.0.0b1", "2.0.0") is False
    assert versions.is_newer("1.0.2", "2.0.0b1") is False
    assert versions.is_newer("2.0.0b1", "2.0.0b1") is False
    assert "2.0.0" in (versions.UpdateChecker(enabled=True, current="2.0.0b1", fetch=lambda _t: "2.0.0").check() or "")


def test_update_check_runs_in_a_daemon_thread_without_blocking_startup(config_home) -> None:
    import threading

    release = threading.Event()

    def slow_pypi(_timeout: float) -> str | None:
        release.wait(5)
        return "9.9.9"

    checker = versions.UpdateChecker(enabled=True, current="2.0.0", fetch=slow_pypi)
    started = time.monotonic()
    thread = checker.start()
    assert time.monotonic() - started < 0.5
    assert thread is not None and thread.daemon
    assert checker.notice is None
    assert checker.start() is thread  # once per checker

    release.set()
    thread.join(5)
    assert checker.notice is not None and "9.9.9" in checker.notice


def test_update_check_an_unwritable_cache_never_raises(config_home, monkeypatch) -> None:
    def unwritable(_name: str, _data: bytes):
        raise PermissionError("read-only home")

    monkeypatch.setattr(secret_store, "write_user_config_file", unwritable)

    assert "2.1.0" in (versions.UpdateChecker(enabled=True, current="2.0.0", fetch=lambda _t: "2.1.0").check() or "")


def test_update_check_the_config_file_writer_is_private(config_home) -> None:
    import os
    import stat

    path = secret_store.write_user_config_file("probe.json", b"{}")

    assert path == config_home / "probe.json"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(config_home.stat().st_mode) == 0o700
    with pytest.raises(ValueError):
        secret_store.write_user_config_file("../escape", b"")


# --- Task 3: lifespan context, transports, assembly (COMP-01/02, PROTO-01/02, Pitfall 8) ---------------------------

import subprocess

from mcp import Client

from exegol_mcp.app import build_server
from exegol_mcp.bootstrap import transports
from exegol_mcp.bootstrap.lifecycle import ServerContext
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.services.server_info import ServerInfoSource
from tests.fakes import DEFAULT_CONTAINERS, FakeExegolAdapter


def _context(
    docker_status: str = "reachable (Docker 29.8.2)",
    notice: str | None = None,
    *,
    adapter: FakeExegolAdapter | None = None,
    probe: FakeHostProbe | None = None,
    accept_eula: bool = False,
) -> ServerContext:
    checker = versions.UpdateChecker(enabled=False)
    if notice is not None:
        checker._notice = notice  # the background check already found a newer release
    return ServerContext(
        adapter=adapter if adapter is not None else FakeExegolAdapter(),
        versions=_versions(),
        update_checker=checker,
        probe=probe if probe is not None else FakeHostProbe(),
        startup_docker_status=docker_status,
        accept_eula=accept_eula,
    )


def test_assembly_the_context_is_both_an_exegol_port_and_a_server_info_source() -> None:
    context = _context()
    from exegol_mcp.adapter.port import ExegolPort

    assert isinstance(context, ExegolPort)
    assert isinstance(context, ServerInfoSource)


async def test_assembly_version_report_carries_live_docker_status_and_the_update_notice() -> None:
    report = await _context(notice="exegol-mcp 9.9.9 is available").version_report()

    # The probe pings reachable, so server_info shows a live status, not "not checked".
    assert "reachable" in report.docker_status
    assert report.exegol_mcp_version == "2.0.0"
    assert report.exegol_library_version == "6.0.2"
    assert report.update_notice == "exegol-mcp 9.9.9 is available"


async def test_assembly_version_report_reports_a_daemon_that_went_down() -> None:
    context = ServerContext(
        adapter=FakeExegolAdapter(),
        versions=_versions(),
        update_checker=versions.UpdateChecker(enabled=False),
        probe=FakeHostProbe(docker_error=ConnectionError("daemon gone")),
        startup_docker_status="reachable (Docker 29.8.2)",
        accept_eula=False,
    )

    report = await context.version_report()
    assert "unreachable" in report.docker_status


async def test_assembly_the_context_forwards_every_port_call_to_the_adapter() -> None:
    context = _context()
    assert await context.list_containers() == DEFAULT_CONTAINERS
    assert await context.start_container("exegol-lab") is True
    assert await context.stop_container("exegol-lab") is False


async def test_assembly_the_context_forwards_create_container_to_the_adapter() -> None:
    from exegol_mcp.adapter.port import CreateSpec

    fake = FakeExegolAdapter()
    info = await _context(adapter=fake).create_container(CreateSpec(name="mcp-1", image="free"))

    assert info.name == "mcp-1"
    assert [spec.name for spec in fake.created_specs] == ["mcp-1"]


async def test_assembly_the_context_forwards_the_file_calls_to_the_adapter() -> None:
    fake = FakeExegolAdapter()
    context = _context(adapter=fake)

    written = await context.write_file("exegol-lab", "/tmp/a", b"hi", True, None, 30.0)
    read = await context.read_file("exegol-lab", "/tmp/a", 100, 30.0)

    assert (written.path, written.created, read.data) == ("/tmp/a", True, b"hi")
    listing = await context.list_directory("exegol-lab", "/tmp", 1, 10, 30.0)
    assert [entry.path for entry in listing.entries] == ["a"]


def test_assembly_the_context_has_no_run_id_until_the_lifespan_sets_one() -> None:
    assert _context().run_id is None


def test_assembly_the_context_keeps_the_given_invoking_home(tmp_path: Path) -> None:
    context = ServerContext(
        adapter=FakeExegolAdapter(),
        versions=_versions(),
        update_checker=versions.UpdateChecker(enabled=False),
        probe=FakeHostProbe(),
        startup_docker_status="reachable (Docker 29.8.2)",
        accept_eula=False,
        invoking_home=tmp_path,
    )

    assert context.invoking_home == tmp_path


def test_assembly_the_context_defaults_the_invoking_home_to_the_invoking_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exegol_mcp.bootstrap import lifecycle

    monkeypatch.setattr(lifecycle, "invoking_user_home", lambda: tmp_path / "operator")

    assert _context().invoking_home == tmp_path / "operator"


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
async def test_assembly_both_protocol_eras_list_and_call_tools(mode: str) -> None:
    server = build_server(None, _context())
    async with Client(server, mode=mode) as client:
        tool_names = {tool.name for tool in (await client.list_tools()).tools}
        assert "list_containers" in tool_names

        result = await client.call_tool("list_containers", {})
        listed = result.structured_content["result"]
        assert [c["name"] for c in listed] == ["exegol-lab", "exegol-lab2"]


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
async def test_assembly_both_eras_read_server_info_from_the_lifespan_context(mode: str) -> None:
    server = build_server(None, _context(notice="exegol-mcp 9.9.9 is available"))
    async with Client(server, mode=mode) as client:
        result = await client.call_tool("server_info", {})
        info = result.structured_content

        # Not the DefaultServerInfoSource fallback: the lifespan context answered.
        assert info["docker_status"] != "not checked"
        assert "reachable" in info["docker_status"]
        assert info["exegol_mcp_version"] == "2.0.0"
        assert info["update_notice"] == "exegol-mcp 9.9.9 is available"



# --- Tool-time readiness re-check at the tool boundary (D-08, WR-01) ------------------------------------------------


class _FailingListAdapter(FakeExegolAdapter):
    """FakeExegolAdapter whose list_containers raises the given error, standing in for a call that hit a dead host."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def list_containers(self) -> list[ContainerInfo]:
        raise self.error


# The readiness wording without the "{error}" placeholder, so the test does not retype the message.
_DOCKER_LOST_WORDING = readiness.DOCKER_UNREACHABLE_MESSAGE.split("{error}")[0].strip()


# What mcp 2.x sends for a tool that raised anything but ToolError: the name, no message.
_OPAQUE_CRASH = "Error executing tool list_containers"


def _text(result) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


# Distinct texts for the failing adapter and the readiness ping, so the test can tell which one the client saw.
_ADAPTER_ERROR = "list_containers lost its Docker socket mid-call"
_PING_ERROR = "ping: no Docker daemon at unix:///var/run/docker.sock"


async def test_boundary_docker_lost_after_startup_reaches_the_client() -> None:
    probe = FakeHostProbe()  # healthy when the server starts
    context = _context(adapter=_FailingListAdapter(ConnectionError(_ADAPTER_ERROR)), probe=probe)
    probe.docker_error = ConnectionError(_PING_ERROR)  # the daemon goes away after startup

    async with Client(build_server(None, context)) as client:
        result = await client.call_tool("list_containers", {})

    assert result.is_error
    text = _text(result)
    assert _DOCKER_LOST_WORDING in text
    assert text != _OPAQUE_CRASH  # not the SDK's message-less crash text
    # The boundary replaces the tool's error with the readiness diagnosis (tools/errors.py step 2); the tool's
    # own error stays server-side as the ToolError cause and never reaches the client.
    assert _PING_ERROR in text
    assert _ADAPTER_ERROR not in text


async def test_boundary_revoked_eula_reaches_the_client() -> None:
    probe = FakeHostProbe(eula_accepted=False)  # the EULA was revoked after startup
    context = _context(adapter=_FailingListAdapter(ConnectionError("refused")), probe=probe, accept_eula=False)

    async with Client(build_server(None, context)) as client:
        result = await client.call_tool("list_containers", {})

    assert result.is_error
    assert readiness.EULA_REVOKED_MESSAGE in _text(result)
    assert probe.eula_writes == []  # the re-check never writes the EULA datastore

    # With --accept-eula the operator accepted it for this run: the datastore is not consulted at all.
    accepted_probe = FakeHostProbe(eula_accepted=False)
    accepted = _context(adapter=_FailingListAdapter(ConnectionError("refused")), probe=accepted_probe, accept_eula=True)
    async with Client(build_server(None, accepted)) as client:
        result = await client.call_tool("list_containers", {})

    assert result.is_error
    assert "ping" in accepted_probe.calls
    assert "eula" not in accepted_probe.calls
    assert readiness.EULA_REVOKED_MESSAGE not in _text(result)


async def test_boundary_bad_input_does_not_ping_docker() -> None:
    probe = FakeHostProbe()
    async with Client(build_server(None, _context(probe=probe))) as client:
        result = await client.call_tool("start_container", {"container_name": "no-such-container"})

    assert result.is_error
    assert "not found" in _text(result)
    assert "ping" not in probe.calls  # bad input is the agent's to fix; no Docker round-trip


async def test_boundary_unexpected_crash_with_a_healthy_host_is_unchanged() -> None:
    probe = FakeHostProbe()  # Docker reachable and EULA accepted
    context = _context(adapter=_FailingListAdapter(KeyError("crash")), probe=probe)

    async with Client(build_server(None, context)) as client:
        result = await client.call_tool("list_containers", {})

    assert result.is_error
    assert "ping" in probe.calls  # the boundary did ask, and the host answered "still ready"
    assert _text(result) == _OPAQUE_CRASH  # still a crash: no readiness wording, no ToolError mapping


def test_assembly_the_http_app_is_the_sdk_app_wrapped_only_by_the_bearer_gate() -> None:
    from exegol_mcp.bootstrap.security import BearerGate

    server = build_server(None, _context())

    gated = transports.build_http_app(server, auth="bearer", secret="s3cr3t")
    assert isinstance(gated, BearerGate)
    # The wrapped app is the SDK Starlette app itself, not a re-mount (Pitfall 8).
    from starlette.applications import Starlette

    assert isinstance(gated.app, Starlette)

    # No-auth fails closed unless the explicit opt-in is passed (SEC-04/D-13).
    with pytest.raises(RuntimeError, match="allow_no_auth"):
        transports.build_http_app(server, auth="none", secret=None)
    plain = transports.build_http_app(server, auth="none", secret=None, allow_no_auth=True)
    assert isinstance(plain, Starlette)


def test_assembly_bearer_mode_requires_a_secret() -> None:
    server = build_server(None, _context())
    with pytest.raises(ValueError, match="no secret"):
        transports.build_http_app(server, auth="bearer", secret=None)


def test_assembly_transports_do_not_remount_the_sdk_app() -> None:
    from pathlib import Path

    source = Path("exegol_mcp/bootstrap/transports.py").read_text()
    assert "Mount(" not in source


def test_assembly_run_stdio_calls_server_run_with_stdio() -> None:
    calls: list[str] = []

    class _Server:
        def run(self, transport: str) -> None:
            calls.append(transport)

    transports.run_stdio(_Server())  # type: ignore[arg-type]
    assert calls == ["stdio"]


def test_assembly_the_entry_point_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "exegol_mcp", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "exegol-mcp" in result.stdout
    assert "--accept-eula" in result.stdout
    assert "--no-update-check" in result.stdout


def test_assembly_the_console_script_target_is_unchanged() -> None:
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text()
    assert 'exegol-mcp = "exegol_mcp.main:main"' in pyproject


# --- Container profiles at startup (D-02, CONF-01, CONF-04) --------------------------------------------------------


@pytest.fixture
def restore_root_logging():
    """Save and restore the root/package logger state main() mutates via configure_stderr_logging."""
    import logging

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
def test_main_a_bad_allowlisted_profile_exits_1_with_one_stderr_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import exegol_mcp.adapter.host as host_module
    import exegol_mcp.main as main_module
    from exegol_mcp.adapter import sdk_headless
    from exegol_mcp.main import main

    def _no_server(*args, **kwargs):
        raise AssertionError("a bad allowlisted profile must stop startup before any server is built")

    monkeypatch.setattr(main_module, "build_server", _no_server)

    monkeypatch.setattr(secret_store, "read_user_config_file", lambda name: b"profiles:\n  allowed: [web, ops]\n")
    # The real SDK patch is process-wide; this test only drives main()'s wiring, so keep it out.
    monkeypatch.setattr(sdk_headless, "patch_exegol_sdk", lambda: None)
    monkeypatch.setattr(sdk_headless, "set_accept_eula", lambda accepted: None)
    probe = FakeHostProbe(profile_problems=["Profile 'ops': Container profile 'ops' not found."])
    monkeypatch.setattr(host_module, "RealHostProbe", lambda: probe)

    status = main(["--type", "stdio", "--accept-eula", "--no-update-check"])

    assert status == 1
    assert probe.validated_profiles == [["web", "ops"]]
    captured = capsys.readouterr()
    assert captured.out == ""  # stdio keeps stdout clean (SEC-08)
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert "Profile 'ops'" in lines[0]
    assert "profiles.allowed" in lines[0]
    assert "Traceback" not in captured.err


@pytest.mark.usefixtures("restore_root_logging")
def test_main_an_import_time_original_missing_exits_1_with_one_stderr_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # WR-12: a missing SDK internal stops startup with the named D-15 refusal on one line, never a traceback.
    import exegol_mcp.adapter.host as host_module
    import exegol_mcp.main as main_module
    from exegol_mcp.adapter import sdk_headless
    from exegol_mcp.main import main

    def _no_server(*args, **kwargs):
        raise AssertionError("a missing patch point must stop startup before any server is built")

    def _no_probe():
        raise AssertionError("a missing patch point must stop startup before the readiness gate")

    def _broken_patch() -> None:
        raise selfcheck._broken("exegol.utils.FsUtils.secure_remove", "is missing")

    monkeypatch.setattr(main_module, "build_server", _no_server)
    monkeypatch.setattr(secret_store, "read_user_config_file", lambda name: b"")
    monkeypatch.setattr(sdk_headless, "patch_exegol_sdk", _broken_patch)
    monkeypatch.setattr(sdk_headless, "set_accept_eula", lambda accepted: None)
    monkeypatch.setattr(host_module, "RealHostProbe", _no_probe)

    status = main(["--type", "stdio", "--accept-eula", "--no-update-check"])

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # stdio keeps stdout clean (SEC-08)
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert "exegol.utils.FsUtils.secure_remove" in lines[0]
    assert "install a supported exegol 6.x release" in lines[0]
    assert "Traceback" not in captured.err


# --- Allowlist entries nested with directories Exegol mounts (WR-10) ------------------------------------------------

_MY_RESOURCES = ExegolMountDir(
    "/x/.exegol/my-resources", "Exegol's my-resources directory (customization.my_resources_path)", False
)
_WORKSPACES = ExegolMountDir(
    "/x/.exegol/workspaces", "Exegol's private workspaces directory (volumes.private_workspace_path)", True
)
_EXEGOL_MOUNTS = [_MY_RESOURCES, _WORKSPACES]


def test_an_entry_inside_an_exegol_mount_is_refused_naming_its_index_and_the_directory() -> None:
    problem = readiness.workspace_allowlist_mount_problem(
        ["/srv/engagements", "/x/.exegol/my-resources/shared"], _EXEGOL_MOUNTS, fold_case=False
    )

    assert problem is not None
    assert "config.yml" in problem
    assert "lifecycle.workspace_allowlist.1" in problem
    assert "/x/.exegol/my-resources/shared" in problem
    assert "'/x/.exegol/my-resources'" in problem
    assert "Exegol's my-resources directory (customization.my_resources_path)" in problem
    assert "read-write" in problem
    assert problem.endswith(readiness._MOUNT_PROBLEM_FIX)
    assert readiness._MOUNT_PROBLEM_FIX.startswith(
        "Allow a directory outside every directory Exegol or an allowlisted profile mounts"
    )


def test_an_entry_inside_a_private_workspace_is_refused_as_an_exegol_mount_problem() -> None:
    problem = readiness.workspace_allowlist_mount_problem(
        ["/x/.exegol/workspaces/scan1/out"], _EXEGOL_MOUNTS, fold_case=False
    )

    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert "lies inside Exegol's private workspaces directory" in problem


def test_an_entry_that_is_or_contains_a_mounted_parent_is_an_exegol_mount_problem() -> None:
    same = readiness.workspace_allowlist_mount_problem(["/x/.exegol/workspaces"], [_WORKSPACES], fold_case=False)
    parent = readiness.workspace_allowlist_mount_problem(["/x/.exegol"], [_WORKSPACES], fold_case=False)

    assert same is not None
    assert "'/x/.exegol/workspaces' is Exegol's private workspaces directory" in same
    assert "subdirectories" in same
    assert parent is not None
    assert "'/x/.exegol' contains Exegol's private workspaces directory" in parent


def test_an_entry_containing_a_mounted_directory_is_an_exegol_mount_problem() -> None:
    problem = readiness.workspace_allowlist_mount_problem(["/x/.exegol"], [_MY_RESOURCES], fold_case=False)

    assert problem is not None
    assert "'/x/.exegol' contains Exegol's my-resources directory" in problem
    assert "('/x/.exegol/my-resources')" in problem


def test_an_entry_equal_to_a_mounted_directory_or_a_prefix_sibling_has_no_exegol_mount_problem() -> None:
    assert (
        readiness.workspace_allowlist_mount_problem(["/x/.exegol/my-resources"], _EXEGOL_MOUNTS, fold_case=False)
        is None
    )
    assert (
        readiness.workspace_allowlist_mount_problem(["/x/.exegol/workspaces-old"], _EXEGOL_MOUNTS, fold_case=False)
        is None
    )


def test_a_case_differing_entry_inside_an_exegol_mount_follows_the_host_case_rule() -> None:
    entry = ["/x/.Exegol/My-Resources/shared"]

    folded = readiness.workspace_allowlist_mount_problem(entry, _EXEGOL_MOUNTS, fold_case=True)

    assert folded is not None
    assert "lies inside Exegol's my-resources directory" in folded
    assert readiness.workspace_allowlist_mount_problem(entry, _EXEGOL_MOUNTS, fold_case=False) is None


async def test_readiness_an_allowlist_entry_nested_with_an_exegol_mount_stops_startup_before_eula() -> None:
    probe = FakeHostProbe(mount_dirs=_EXEGOL_MOUNTS)

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            probe,
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=["/x/.exegol/my-resources/shared"],
        )

    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in str(raised.value)
    assert probe.calls[-1] == "mounts"
    assert "eula" not in probe.calls


async def test_readiness_an_empty_allowlist_never_asks_for_exegol_mounts() -> None:
    probe = FakeHostProbe(mount_error=AssertionError("must not be asked"))

    await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt, workspace_allowlist=[])
    await readiness.check_readiness(probe, _args("--type", "stdio"), prompt=_never_prompt, profiles=("web",))

    assert "mounts" not in probe.calls
    assert probe.mount_profiles == []


async def test_readiness_unreadable_exegol_mounts_stop_startup_fail_closed() -> None:
    probe = FakeHostProbe(mount_error=RuntimeError("boom"))

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            probe, _args("--type", "stdio"), prompt=_never_prompt, workspace_allowlist=["/srv/engagements"]
        )

    assert raised.value.check == "workspace-allowlist"
    assert "boom" in str(raised.value)
    assert "lifecycle.workspace_allowlist" in str(raised.value)
    # IN-23: the failure may come from the import path, not only from Exegol's mounts.
    assert "imports code from" in str(raised.value)
    # IN-25: or from walking the user site, which the message names too.
    assert "user site" in str(raised.value)
    assert "eula" not in probe.calls


async def test_readiness_a_clean_allowlist_passes_the_exegol_mount_check() -> None:
    probe = FakeHostProbe(mount_dirs=_EXEGOL_MOUNTS)

    await readiness.check_readiness(
        probe,
        _args("--type", "stdio"),
        prompt=_never_prompt,
        profiles=("web",),
        workspace_allowlist=["/srv/engagements", "/x/.exegol/my-resources", "/x/.exegol/workspaces-old"],
    )

    assert probe.calls == ["patch-points", "docker", "config", "profiles", "mounts", "eula"]
    assert probe.mount_profiles == [["web"]]


@pytest.mark.usefixtures("restore_root_logging")
def test_main_an_allowlist_entry_inside_an_exegol_mount_exits_1_with_one_stderr_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import os

    import exegol_mcp.adapter.host as host_module
    import exegol_mcp.main as main_module
    from exegol_mcp.adapter import sdk_headless
    from exegol_mcp.main import main

    def _no_server(*args, **kwargs):
        raise AssertionError("an allowlist entry inside an Exegol mount must stop startup before any server is built")

    root = Path(os.path.realpath(tmp_path))
    my_resources = root / "my-resources"
    entry = my_resources / "shared"
    monkeypatch.setattr(main_module, "build_server", _no_server)
    monkeypatch.setattr(
        secret_store,
        "read_user_config_file",
        lambda name: f"lifecycle:\n  workspace_allowlist: ['{entry}']\n".encode(),
    )
    monkeypatch.setattr(sdk_headless, "patch_exegol_sdk", lambda: None)
    monkeypatch.setattr(sdk_headless, "set_accept_eula", lambda accepted: None)
    probe = FakeHostProbe(mount_dirs=[ExegolMountDir(str(my_resources), "Exegol's my-resources directory (x)", False)])
    monkeypatch.setattr(host_module, "RealHostProbe", lambda: probe)

    status = main(["--type", "stdio", "--accept-eula", "--no-update-check"])

    assert status == 1
    assert probe.mount_profiles == [[]]
    captured = capsys.readouterr()
    assert captured.out == ""  # stdio keeps stdout clean (SEC-08)
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert "lifecycle.workspace_allowlist.0" in lines[0]
    assert str(entry) in lines[0]
    assert "Traceback" not in captured.err


@pytest.mark.usefixtures("restore_root_logging")
def test_main_allowlisting_exegol_mcps_config_dir_exits_1_with_one_stderr_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home: Path
) -> None:
    # T-05-50 end to end: the real config.yml in a temporary home's ~/.exegol-mcp allowlists that directory. main()
    # reads it, resolves the entry, and the readiness gate refuses it through the real _exegol_mcp_config_dirs, before
    # any server is built or container touched. Only Docker and Exegol's own directories are faked (no Docker here).
    import exegol_mcp.adapter.host as host_module
    import exegol_mcp.main as main_module
    from exegol_mcp.adapter import sdk_headless
    from exegol_mcp.main import main

    def _no_server(*args, **kwargs):
        raise AssertionError("allowlisting exegol-mcp's config directory must stop startup before any server is built")

    real_home = Path(os.path.realpath(home))
    monkeypatch.setattr(secret_store, "invoking_user_home", lambda: real_home)
    monkeypatch.setenv("HOME", str(real_home))
    config_dir = real_home / ".exegol-mcp"
    config_dir.mkdir(mode=0o700)
    config_yml = f"lifecycle:\n  workspace_allowlist: ['{config_dir}']\n"
    (config_dir / "config.yml").write_text(config_yml)
    monkeypatch.setattr(main_module, "build_server", _no_server)
    monkeypatch.setattr(sdk_headless, "patch_exegol_sdk", lambda: None)
    monkeypatch.setattr(sdk_headless, "set_accept_eula", lambda accepted: None)
    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_python_install_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    probe = _RealMountsProbe()
    monkeypatch.setattr(host_module, "RealHostProbe", lambda: probe)

    status = main(["--type", "stdio", "--accept-eula", "--no-update-check"])

    assert status == 1
    assert probe.calls == ["patch-points", "docker", "config", "mounts"]
    captured = capsys.readouterr()
    assert captured.out == ""  # stdio keeps stdout clean (SEC-08)
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert "lifecycle.workspace_allowlist.0" in lines[0]
    assert f"'{config_dir}' is exegol-mcp's config directory" in lines[0]
    assert "exegol-mcp reads its config.yml (the operator's restrictions) and its bearer secret from there" in lines[0]
    assert readiness._SERVER_CONFIG_PROBLEM_FIX in lines[0]
    assert "Traceback" not in captured.err
    # The check only resolved the paths: config.yml is untouched and the old directory was never created.
    assert (config_dir / "config.yml").read_text() == config_yml
    assert not secret_store._platform_config_dir().exists()


def test_the_real_probe_reports_exegols_own_mount_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    import exegol_mcp.adapter.host as host_module

    names = {
        host_module.OptionKey.MY_RESOURCES_PATH: "my-resources",
        host_module.OptionKey.EXEGOL_RESOURCES_PATH: "exegol-resources",
        host_module.OptionKey.PRIVATE_WORKSPACE_PATH: "workspaces",
        host_module.OptionKey.SENTINEL_LOGS_HOST_PATH: "sentinel",
    }

    class _Resolver:
        def get(self, key):
            return tmp_path / names[key]

    monkeypatch.setattr(host_module, "OptionResolver", _Resolver)
    _without_gui_or_settings_dirs(monkeypatch)

    dirs = host_module.RealHostProbe().exegol_mount_dirs([])

    assert [d.path for d in dirs] == [os.path.realpath(tmp_path / name) for name in names.values()]
    assert [d.children_mounted for d in dirs] == [False, False, True, True]
    for d, option in zip(
        dirs,
        (
            "customization.my_resources_path",
            "volumes.exegol_resources_path",
            "volumes.private_workspace_path",
            "sentinel.sentinel_logs_host_path",
        ),
    ):
        assert option in d.label


# --- CR-02: every host path Docker's bind resolution walks for a mount ---------------------------------------------

_MOUNT_LABEL = "Exegol's my-resources directory (customization.my_resources_path)"


def _real_root(tmp_path: Path) -> Path:
    """tmp_path as config load stores an entry: its realpath (on macOS tmp_path sits under /var -> private/var)."""
    return Path(os.path.realpath(tmp_path))


def _stub_exegol_dirs(monkeypatch: pytest.MonkeyPatch, values: dict) -> None:
    """Make host.OptionResolver resolve the given OptionKey values, and every other key to None."""
    import exegol_mcp.adapter.host as host_module

    class _Resolver:
        def get(self, key):
            return values.get(key)

    monkeypatch.setattr(host_module, "OptionResolver", _Resolver)


def test_a_symlinked_mount_path_inside_an_entry_is_refused_by_the_bind_walk() -> None:
    walked = ExegolMountDir(
        path="/mnt/big/my-resources",
        label=_MOUNT_LABEL,
        children_mounted=False,
        lexical="/srv/pentest/my-resources",
        candidates=(
            "/srv/pentest/my-resources",
            "/srv",
            "/srv/pentest",
            "/mnt",
            "/mnt/big",
            "/mnt/big/my-resources",
        ),
    )

    problem = readiness.workspace_allowlist_mount_problem(["/srv/pentest"], [walked], fold_case=False)

    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert "'/srv/pentest'" in problem
    assert "/srv/pentest/my-resources" in problem
    assert _MOUNT_LABEL in problem
    assert problem.endswith(readiness._MOUNT_PROBLEM_FIX)
    # The realpath alone (04-20's comparison) misses it: the hole CR-02 reported.
    realpath_only = ExegolMountDir("/mnt/big/my-resources", _MOUNT_LABEL, False)
    assert readiness.workspace_allowlist_mount_problem(["/srv/pentest"], [realpath_only], fold_case=False) is None


def test_a_mount_reached_through_a_symlink_equal_to_the_entry_is_refused_by_the_bind_walk() -> None:
    linked = ExegolMountDir(
        path="/srv/pentest",
        label=_MOUNT_LABEL,
        children_mounted=False,
        lexical="/home/u/link",
        candidates=("/home/u/link", "/home", "/home/u", "/srv", "/srv/pentest"),
    )

    problem = readiness.workspace_allowlist_mount_problem(["/srv/pentest"], [linked], fold_case=False)

    assert problem is not None
    assert "symlink" in problem
    assert "/home/u/link" in problem
    relation = readiness._mount_relation(Path("/srv/pentest"), linked, fold_case=False)
    assert relation is not None and relation[0] == "contains"
    # Equal to a directly mounted directory reached without a symlink still loads (04-20).
    direct = ExegolMountDir("/srv/pentest", _MOUNT_LABEL, False)
    assert readiness.workspace_allowlist_mount_problem(["/srv/pentest"], [direct], fold_case=False) is None


def test_the_bind_walk_records_the_lexical_path_each_looked_up_entry_and_prefix_realpaths(tmp_path: Path) -> None:
    root = _real_root(tmp_path)
    (root / "big" / "my-resources").mkdir(parents=True)
    (root / "entry").mkdir()
    os.symlink(root / "big" / "my-resources", root / "entry" / "my-resources")
    os.symlink("big", root / "rel")

    linked = mount_paths.mount_dir(root / "entry" / "my-resources", _MOUNT_LABEL, False)

    assert linked.lexical == str(root / "entry" / "my-resources")
    assert linked.path == os.path.realpath(root / "big" / "my-resources")
    assert linked.candidates[0] == linked.lexical
    for expected in (root / "entry", root / "entry" / "my-resources", root / "big", root / "big" / "my-resources"):
        assert str(expected) in linked.candidates

    relative = mount_paths.mount_dir(root / "rel" / "my-resources", _MOUNT_LABEL, False)

    assert str(root / "rel") in relative.candidates
    assert str(root / "big" / "my-resources") in relative.candidates
    assert relative.path == os.path.realpath(root / "big" / "my-resources")

    dotted = mount_paths.mount_dir(f"{root}/entry/../big/my-resources", _MOUNT_LABEL, False)

    assert dotted.lexical == str(root / "big" / "my-resources")
    assert str(root / "entry") in dotted.candidates
    assert str(root / "big" / "my-resources") in dotted.candidates
    lexical = Path(dotted.lexical)
    for prefix in (*lexical.parents, lexical):
        assert os.path.realpath(prefix) in dotted.candidates


def test_the_bind_walk_follows_a_symlink_chain_through_an_entry(tmp_path: Path) -> None:
    root = _real_root(tmp_path)
    for name in ("entry", "other", "outside/t"):
        (root / name).mkdir(parents=True)
    os.symlink(root / "outside" / "t", root / "entry" / "s")
    os.symlink(root / "entry" / "s", root / "other" / "l")

    chained = mount_paths.mount_dir(root / "other" / "l", _MOUNT_LABEL, False)

    entry = root / "entry"
    for path in (chained.lexical, chained.path):
        assert not path.startswith(str(entry) + os.sep) and not str(entry).startswith(path + os.sep)
    assert str(entry / "s") in chained.candidates
    problem = readiness.workspace_allowlist_mount_problem([str(entry)], [chained], fold_case=False)
    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert str(entry / "s") in problem


def test_the_bind_walk_records_missing_components_lexically(tmp_path: Path) -> None:
    root = _real_root(tmp_path)
    (root / "entry").mkdir()

    missing = mount_paths.mount_dir(root / "entry" / "not-yet" / "my-resources", _MOUNT_LABEL, False)

    assert str(root / "entry" / "not-yet") in missing.candidates
    assert str(root / "entry" / "not-yet" / "my-resources") in missing.candidates
    problem = readiness.workspace_allowlist_mount_problem([str(root / "entry")], [missing], fold_case=False)
    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and permissions")
def test_the_bind_walk_fails_closed_on_a_symlink_loop_or_an_unreadable_component(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _real_root(tmp_path)
    os.symlink(root / "b", root / "a")
    os.symlink(root / "a", root / "b")

    with pytest.raises(OSError) as looped:
        mount_paths.mount_dir(root / "a" / "my-resources", _MOUNT_LABEL, False)
    assert looped.value.errno == errno.ELOOP

    def _denied(path, *args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", os.fspath(path))

    monkeypatch.setattr(mount_paths.os, "lstat", _denied)
    with pytest.raises(PermissionError):
        mount_paths.mount_dir(root / "entry" / "my-resources", _MOUNT_LABEL, False)


def test_the_bind_walk_is_repeatable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    (root / "big" / "my-resources").mkdir(parents=True)
    (root / "entry").mkdir()
    os.symlink(root / "big" / "my-resources", root / "entry" / "my-resources")
    _stub_exegol_dirs(
        monkeypatch,
        {
            host_module.OptionKey.MY_RESOURCES_PATH: root / "entry" / "my-resources",
            host_module.OptionKey.PRIVATE_WORKSPACE_PATH: root / "workspaces",
        },
    )
    _without_gui_or_settings_dirs(monkeypatch)

    first = mount_paths.mount_dir(root / "entry" / "my-resources", _MOUNT_LABEL, False)
    second = mount_paths.mount_dir(root / "entry" / "my-resources", _MOUNT_LABEL, False)
    probed = host_module.RealHostProbe().exegol_mount_dirs([])
    probed_again = host_module.RealHostProbe().exegol_mount_dirs([])

    assert first == second
    assert first.candidates
    assert probed == probed_again
    assert len(probed) == 2


def test_a_case_differing_symlinked_mount_follows_the_host_case_rule_in_the_bind_walk() -> None:
    walked = ExegolMountDir(
        path="/mnt/big/my-resources",
        label=_MOUNT_LABEL,
        children_mounted=False,
        lexical="/Srv/Pentest/my-resources",
        candidates=("/Srv/Pentest/my-resources", "/Srv", "/Srv/Pentest", "/mnt", "/mnt/big", "/mnt/big/my-resources"),
    )

    folded = readiness.workspace_allowlist_mount_problem(["/srv/pentest"], [walked], fold_case=True)

    assert folded is not None
    assert "lifecycle.workspace_allowlist.0" in folded
    assert readiness.workspace_allowlist_mount_problem(["/srv/pentest"], [walked], fold_case=False) is None


async def test_readiness_refuses_a_symlinked_my_resources_under_an_entry_bind_walk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    (root / "big" / "my-resources").mkdir(parents=True)
    (root / "entry").mkdir()
    os.symlink(root / "big" / "my-resources", root / "entry" / "my-resources")
    _stub_exegol_dirs(monkeypatch, {host_module.OptionKey.MY_RESOURCES_PATH: root / "entry" / "my-resources"})
    _without_gui_or_settings_dirs(monkeypatch)

    mounts = host_module.RealHostProbe().exegol_mount_dirs([])

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=mounts),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(root / "entry")],
        )

    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in str(raised.value)
    assert str(root / "entry" / "my-resources") in str(raised.value)


def test_every_real_probe_mount_dir_carries_its_bind_walk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _settings_tree(monkeypatch, root, install=True, server_config=True)
    monkeypatch.setattr(
        host_module,
        "GuiUtils",
        _gui_stub(x11=str(root / "tmp" / ".X11-unix"), wayland_available=True, wayland=root / "run" / "wayland-0"),
    )

    dirs = host_module.RealHostProbe().exegol_mount_dirs([])

    # Exegol's four directories, both GUI sockets, the five settings directories, exegol-mcp's config directory and
    # the old one, the four install roots, two import-path entries (the venv's bin directory and the base stdlib
    # directory; site-packages repeats Exegol's install directory and is listed once) and the seven read-only bind
    # sources (timezone files included): every host.py producer.
    assert len(dirs) == 26
    for mount in dirs:
        assert mount.lexical
        assert mount.candidates[0] == mount.lexical
        assert mount.path == os.path.realpath(mount.lexical)
        _assert_one_role(mount)


def _assert_one_role(mount: ExegolMountDir) -> None:
    """The role invariant: at most one of holds_mount_settings, holds_server_config, holds_host_code and contains_only
    is set, and children_mounted only when none is (04-23 assumption delta)."""
    roles = [mount.holds_mount_settings, mount.holds_server_config, mount.holds_host_code, mount.contains_only]
    assert sum(roles) <= 1, mount
    assert not (mount.children_mounted and any(roles)), mount


# --- IN-14 premise: directories Exegol reads its mount settings from -----------------------------------------------

_SETTINGS_LABEL = "Exegol's config directory"


def test_an_entry_nested_with_a_mount_settings_dir_is_refused() -> None:
    settings = ExegolMountDir("/srv/exegol-config", _SETTINGS_LABEL, False, holds_mount_settings=True)

    for entry in ("/srv/exegol-config/work", "/srv", "/srv/exegol-config"):
        problem = readiness.workspace_allowlist_mount_problem([entry], [settings], fold_case=False)
        assert problem is not None, entry
        assert "lifecycle.workspace_allowlist.0" in problem
        assert _SETTINGS_LABEL in problem
        assert "Exegol reads the settings" in problem
        assert readiness._SETTINGS_PROBLEM_FIX in problem
        assert readiness._MOUNT_PROBLEM_FIX not in problem

    plain = ExegolMountDir("/srv/exegol-config", _SETTINGS_LABEL, False)
    assert readiness.workspace_allowlist_mount_problem(["/srv/exegol-config"], [plain], fold_case=False) is None


def test_a_symlinked_mount_settings_dir_inside_an_entry_is_refused() -> None:
    linked = ExegolMountDir(
        path="/mnt/profiles",
        label="the container profile source 'team'",
        children_mounted=False,
        lexical="/srv/pentest/profiles",
        candidates=("/srv/pentest/profiles", "/srv", "/srv/pentest", "/mnt", "/mnt/profiles"),
        holds_mount_settings=True,
    )

    problem = readiness.workspace_allowlist_mount_problem(["/srv/pentest"], [linked], fold_case=False)

    assert problem is not None
    assert "/srv/pentest/profiles" in problem
    assert "Exegol reads the settings" in problem
    assert problem.endswith(readiness._SETTINGS_PROBLEM_FIX)


# --- WR-11: the GUI socket mounts; IN-14 premise: Exegol's settings directories -----------------------------------


def _gui_stub(*, x11=None, wayland_available: bool = False, wayland=None) -> type:
    """A GuiUtils stand-in: each getter returns its scripted value, or raises it when it is an exception."""

    class _Gui:
        @classmethod
        def getX11SocketPath(cls):
            if isinstance(x11, BaseException):
                raise x11
            return x11

        @classmethod
        def isWaylandGuiAvailable(cls) -> bool:
            return wayland_available

        @classmethod
        def getWaylandSocketPath(cls):
            if isinstance(wayland, BaseException):
                raise wayland
            return wayland

    return _Gui


def _no_gui() -> type:
    return _gui_stub(wayland=AssertionError("getWaylandSocketPath must not be called without Wayland"))


def _without_gui_or_settings_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a real-probe test off the host's GUI environment, Exegol config and Python installation (hermetic)."""
    import exegol_mcp.adapter.host as host_module

    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_python_install_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)


def _install_tree(monkeypatch: pytest.MonkeyPatch, root: Path, *, linux: bool = False) -> dict[str, Path]:
    """The Python installation exegol-mcp runs from, under root (CR-03): sys.prefix is the venv root/py/proj/.venv,
    sys.base_prefix is root/py/base, Exegol's install directory (ConstantConfig.src_root_path_obj) is the venv's
    site-packages and exegol-mcp's package directory is root/py/proj/exegol_mcp. root/py/proj/tests sits beside them.
    Exegol's read-only bind sources sit in site/exegol/utils/imgsync, derived as ConstantConfig derives them. The
    ConstantConfig stub keeps Exegol's config directory at root/x/cfg; the EnvInfo stub is a Linux shell with linux,
    a macOS shell without.

    host.sys is a namespace (executable, prefix, base_prefix, path, platform), so neither the real interpreter state
    nor the host's import path reaches the probe, and the real sys module is never replaced. The default path is a
    console-script launch of a non-editable venv: the venv's bin directory, the base stdlib directory and
    site-packages (CR-04). A test that needs other import-path entries sets host.sys.path. __PYVENV_LAUNCHER__ is
    removed from the environment, so the executable site.venv() starts from is host.sys.executable.

    host.site is a namespace too (ENABLE_USER_SITE, getusersitepackages), so no test depends on the host's user site
    (WR-17). The user site is root/home/.local/lib/python3.10/site-packages, returned as "user_site" and never
    created; it is disabled by default, and a test enables it by setting host.site.ENABLE_USER_SITE.

    Three stand-ins, decided by the pyvenv.cfg CPython's site.venv() reads (IN-25, WR-18, IN-28):
    - the venv (the default): host.sys.executable is venv/bin/python, so site.venv() reads venv/pyvenv.cfg (returned
      as "pyvenv_cfg"), which excludes system site packages as a standard venv's does, and a disabled user site is
      not listed. venv/bin holds no pyvenv.cfg; "exe_dir_cfg" is that path, not created, which site.venv() reads
      first when a test writes it;
    - a non-venv interpreter: a test sets host.sys.executable to base/bin/python3.10 and host.sys.prefix to
      host.sys.base_prefix. Neither base/bin nor base holds a pyvenv.cfg;
    - a -S start of a plain venv on 3.10-3.12: a test sets host.sys.prefix to host.sys.base_prefix only, with the
      executable unchanged, because under -S, site.venv() never runs, and on 3.10-3.12 sys.prefix then stays the
      base installation's even in a venv (IN-28, IN-32)."""
    import exegol_mcp.adapter.host as host_module

    venv = root / "py" / "proj" / ".venv"
    site = venv / "lib" / "python3.10" / "site-packages"
    base = root / "py" / "base"
    pkg = root / "py" / "proj" / "exegol_mcp"
    imgsync = site / "exegol" / "utils" / "imgsync"
    for directory in (
        site,
        venv / "bin",
        base / "lib" / "python3.10",
        pkg,
        root / "py" / "proj" / "tests",
        imgsync / "shell_script",
        imgsync / "sentinel",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for file in (pkg / "__init__.py", imgsync / "spawn.sh", imgsync / "entrypoint.sh"):
        file.touch()
    for file in (imgsync / "shell_script" / "zsh_hooks.sh", imgsync / "shell_script" / "bash_hooks.sh"):
        file.touch()
    pyvenv_cfg = venv / "pyvenv.cfg"
    pyvenv_cfg.write_text(f"home = {base / 'bin'}\ninclude-system-site-packages = false\n", encoding="utf-8")
    import_path = [str(venv / "bin"), str(base / "lib" / "python3.10"), str(site)]
    monkeypatch.setattr(
        host_module,
        "sys",
        types.SimpleNamespace(
            executable=str(venv / "bin" / "python"),
            prefix=str(venv),
            base_prefix=str(base),
            path=import_path,
            platform=sys.platform,
        ),
    )
    monkeypatch.delenv("__PYVENV_LAUNCHER__", raising=False)
    monkeypatch.setattr(host_module, "_EXEGOL_MCP_PACKAGE_DIR", pkg)
    user_site = root / "home" / ".local" / "lib" / "python3.10" / "site-packages"
    monkeypatch.setattr(
        host_module, "site", types.SimpleNamespace(ENABLE_USER_SITE=False, getusersitepackages=lambda: str(user_site))
    )

    class _Constants:
        exegol_config_path = root / "x" / "cfg"
        src_root_path_obj = site
        entrypoint_context_path_obj = src_root_path_obj / "exegol/utils/imgsync/entrypoint.sh"
        spawn_context_path_obj = src_root_path_obj / "exegol/utils/imgsync/spawn.sh"
        sentinel_zsh_context_path_obj = src_root_path_obj / "exegol/utils/imgsync/shell_script/zsh_hooks.sh"
        sentinel_bash_context_path_obj = src_root_path_obj / "exegol/utils/imgsync/shell_script/bash_hooks.sh"
        sentinel_context_path_obj = src_root_path_obj / "exegol/utils/imgsync/sentinel"

    class _Env:
        is_windows_shell = False
        is_mac_shell = not linux

    monkeypatch.setattr(host_module, "ConstantConfig", _Constants)
    monkeypatch.setattr(host_module, "EnvInfo", _Env)
    return {
        "venv": venv,
        "site": site,
        "base": base,
        "pkg": pkg,
        "imgsync": imgsync,
        "user_site": user_site,
        "pyvenv_cfg": pyvenv_cfg,
        "exe_dir_cfg": venv / "bin" / "pyvenv.cfg",
    }


def _settings_tree(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    sources: bool = True,
    install: bool = False,
    server_config: bool = False,
) -> None:
    """Exegol's settings sources under root: the config directory root/x/cfg, its profile component directory, and
    path:, git:, ~ and symlinked profile sources. Exegol's four directories sit in root/elsewhere; the GUI is off.
    With install, the Python installation and Exegol's read-only binds are _install_tree's under root (a Linux shell);
    without, the probe lists none of them. With server_config, exegol-mcp's config directory and the old one are
    _server_config_dirs' under root; without, the probe lists neither."""
    import exegol_mcp.adapter.host as host_module

    key = host_module.OptionKey
    component = root / "x" / "cfg" / "components" / "profiles"
    values: dict = {
        key.MY_RESOURCES_PATH: root / "elsewhere" / "my-resources",
        key.EXEGOL_RESOURCES_PATH: root / "elsewhere" / "exegol-resources",
        key.PRIVATE_WORKSPACE_PATH: root / "elsewhere" / "workspaces",
        key.SENTINEL_LOGS_HOST_PATH: root / "elsewhere" / "sentinel",
    }
    if sources:
        values[key.PROFILE_COMPONENT_PATH] = component
        values[key.PROFILE_SOURCES] = {
            "local": {"path": str(component / "local")},
            "team": {"git": "https://example.invalid/profiles.git"},
            "home": {"path": "~/profiles"},
            "linked": {"path": str(root / "link")},
            "odd": "not-a-mapping",
        }
    _stub_exegol_dirs(monkeypatch, values)

    class _Constants:
        exegol_config_path = root / "x" / "cfg"

    monkeypatch.setattr(host_module, "ConstantConfig", _Constants)
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setenv("HOME", str(root / "home"))
    (root / "entry" / "profiles").mkdir(parents=True, exist_ok=True)
    if not (root / "link").is_symlink():
        os.symlink(root / "entry" / "profiles", root / "link")
    if install:
        _install_tree(monkeypatch, root, linux=True)
    else:
        monkeypatch.setattr(host_module, "_python_install_dirs", list)
        monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    if server_config:
        _server_config_dirs(monkeypatch, root)
    else:
        monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)


def test_the_real_probe_reports_the_gui_socket_mounts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_python_install_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    monkeypatch.setattr(
        host_module,
        "GuiUtils",
        _gui_stub(x11=str(root / "tmp" / ".X11-unix"), wayland_available=True, wayland=root / "run" / "wayland-0"),
    )

    dirs = host_module.RealHostProbe().exegol_mount_dirs([])

    assert len(dirs) == 2
    x11, wayland = dirs
    assert x11.lexical == str(root / "tmp" / ".X11-unix")
    assert wayland.lexical == str(root / "run" / "wayland-0")
    assert "X11" in x11.label
    assert "Wayland" in wayland.label
    for mount in dirs:
        assert mount.children_mounted is False
        assert mount.holds_mount_settings is False
        assert mount.candidates[0] == mount.lexical


def test_the_gui_socket_mounts_are_skipped_when_exegol_shares_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from exegol.exceptions.ExegolExceptions import CancelOperation

    import exegol_mcp.adapter.host as host_module

    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_python_install_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    no_wayland = AssertionError("getWaylandSocketPath must not be called without Wayland")

    for x11 in (CancelOperation("no WSL distro yet"), None, "\\\\wsl.localhost\\d\\mnt\\wslg\\.X11-unix"):
        monkeypatch.setattr(host_module, "GuiUtils", _gui_stub(x11=x11, wayland=no_wayland))
        assert host_module.RealHostProbe().exegol_mount_dirs([]) == [], x11

    monkeypatch.setattr(host_module, "GuiUtils", _gui_stub(wayland_available=True, wayland=None))
    assert host_module.RealHostProbe().exegol_mount_dirs([]) == []


def test_an_entry_containing_the_x11_gui_socket_directory_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    socket_dir = root / "tmp" / ".X11-unix"  # does not exist yet
    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_python_install_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    monkeypatch.setattr(host_module, "GuiUtils", _gui_stub(x11=str(socket_dir), wayland=AssertionError("no")))

    mounts = host_module.RealHostProbe().exegol_mount_dirs([])
    problem = readiness.workspace_allowlist_mount_problem([str(root / "tmp")], mounts, fold_case=False)

    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert "X11" in problem
    assert readiness.workspace_allowlist_mount_problem([str(socket_dir)], mounts, fold_case=False) is None


async def test_an_unexpected_gui_socket_getter_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import exegol_mcp.adapter.host as host_module

    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_python_install_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    monkeypatch.setattr(host_module, "GuiUtils", _gui_stub(x11=RuntimeError("boom"), wayland=AssertionError("no")))

    with pytest.raises(RuntimeError, match="boom"):
        host_module.RealHostProbe().exegol_mount_dirs([])

    class _RealMountsProbe(FakeHostProbe):
        def exegol_mount_dirs(self, profiles) -> list[ExegolMountDir]:
            self.calls.append("mounts")
            return host_module.RealHostProbe().exegol_mount_dirs(profiles)

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            _RealMountsProbe(), _args("--type", "stdio"), prompt=_never_prompt, workspace_allowlist=["/srv/engagements"]
        )
    assert raised.value.check == "workspace-allowlist"
    assert "boom" in str(raised.value)


def test_exegols_gui_socket_mounts_are_still_read_write_volumes() -> None:
    import inspect

    from exegol.model.ContainerConfig import ContainerConfig

    source = inspect.getsource(ContainerConfig.enableGUI)

    for call in (
        "GuiUtils.getX11SocketPath()",
        "GuiUtils.getWaylandSocketPath()",
        "GuiUtils.isWaylandGuiAvailable()",
        "self.addVolume(host_path",
    ):
        assert call in source, call
    assert "read_only" not in source


def test_the_gui_socket_getters_are_self_checked() -> None:
    for pair in (
        ("exegol.utils.GuiUtils", "GuiUtils.getX11SocketPath"),
        ("exegol.utils.GuiUtils", "GuiUtils.getWaylandSocketPath"),
        ("exegol.utils.GuiUtils", "GuiUtils.isWaylandGuiAvailable"),
        ("exegol.exceptions.ExegolExceptions", "CancelOperation"),
    ):
        assert pair in selfcheck._CALLABLE_POINTS, pair
    selfcheck.assert_patch_points()


def test_the_real_probe_reports_exegols_settings_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _settings_tree(monkeypatch, root)

    dirs = host_module.RealHostProbe().exegol_mount_dirs([])

    assert len(dirs) == 9
    assert not any(mount.holds_mount_settings for mount in dirs[:4])
    settings = dirs[4:]
    component = root / "x" / "cfg" / "components" / "profiles"
    assert [mount.lexical for mount in settings] == [
        str(root / "x" / "cfg"),
        str(component),
        str(component / "local"),
        str(root / "home" / "profiles"),
        str(root / "link"),
    ]
    for mount, name in zip(settings, ("config directory", "profile_component_path", "'local'", "'home'", "'linked'")):
        assert name in mount.label
        assert mount.holds_mount_settings is True
        assert mount.children_mounted is False
        assert mount.candidates[0] == mount.lexical
    assert not any("'team'" in mount.label or "'odd'" in mount.label for mount in settings)

    _settings_tree(monkeypatch, root, sources=False)
    only_config = host_module.RealHostProbe().exegol_mount_dirs([])[4:]

    assert [mount.lexical for mount in only_config] == [str(root / "x" / "cfg")]


@pytest.mark.parametrize(
    ("entry", "names"),
    [
        (("x", "cfg"), ("config directory",)),
        (("x",), ("config directory",)),
        (("home", "profiles", "sub"), ("'home'",)),
        (("entry",), ("'linked'", "entry/profiles")),
    ],
    ids=["is-the-config-dir", "contains-the-config-dir", "inside-a-path-source", "contains-a-symlinked-source"],
)
async def test_an_entry_nested_with_a_settings_dir_stops_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: tuple[str, ...], names: tuple[str, ...]
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _settings_tree(monkeypatch, root)
    mounts = host_module.RealHostProbe().exegol_mount_dirs([])

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=mounts),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(root.joinpath(*entry))],
        )

    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "Exegol reads the settings" in message
    for name in names:
        assert name in message, name


async def test_an_entry_apart_from_every_settings_dir_passes_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _settings_tree(monkeypatch, root)
    probe = FakeHostProbe(mount_dirs=host_module.RealHostProbe().exegol_mount_dirs([]))

    await readiness.check_readiness(
        probe, _args("--type", "stdio"), prompt=_never_prompt, workspace_allowlist=[str(root / "work")]
    )

    assert probe.calls == ["patch-points", "docker", "config", "mounts", "eula"]


def test_exegols_profile_roots_still_match_the_settings_dirs() -> None:
    import inspect

    from exegol.profile.ContainerProfileManager import ContainerProfileManager

    enumerate_roots = inspect.getsource(ContainerProfileManager._ContainerProfileManager__enumerate_roots)

    assert "OptionKey.PROFILE_SOURCES" in inspect.getsource(ContainerProfileManager.__init__)
    assert "OptionKey.PROFILE_COMPONENT_PATH" in inspect.getsource(ContainerProfileManager._default_component_path)
    assert 'spec.get("path")' in enumerate_roots
    assert "SudoInfo.expandUser(local_path)" in enumerate_roots
    assert "self._component_path / key" in enumerate_roots
    for pair in (("PROFILE_COMPONENT_PATH", "profile_component_path"), ("PROFILE_SOURCES", "profile_sources")):
        assert pair in selfcheck._OPTION_KEYS, pair
    selfcheck.assert_patch_points()


def test_exegols_settings_dir_refusal_keeps_its_exact_wording() -> None:
    settings = ExegolMountDir("/srv/exegol-config", _SETTINGS_LABEL, False, holds_mount_settings=True)

    problem = readiness.workspace_allowlist_mount_problem(["/srv/exegol-config"], [settings], fold_case=False)

    assert problem == (
        "Invalid config.yml: field 'lifecycle.workspace_allowlist.0': '/srv/exegol-config' is Exegol's config "
        "directory ('/srv/exegol-config'): Exegol reads the settings that decide what containers mount from there, so "
        "an allowlisted directory may not lie inside it, contain it or be it, and the entry is refused (D-24, IN-14). "
        "Allow a directory apart from Exegol's config directory and its container profile sources."
    )


# --- T-05-50: exegol-mcp's own config directory (config.yml, the bearer secret) and the old one it moves from ------

_PLANNING_ID = re.compile(r"\b(?:AR|CR|D|IN|T|WR)-\d+")


def _server_config_dirs(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[Path, Path]:
    """Point exegol-mcp's config directory at root/home/.exegol-mcp (created) and the old platform directory at
    root/home/Library/Application Support/exegol-mcp (not created), so no test resolves the developer's own."""
    current = root / "home" / ".exegol-mcp"
    old = root / "home" / "Library" / "Application Support" / "exegol-mcp"
    current.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(secret_store, "user_config_dir", lambda: current)
    monkeypatch.setattr(secret_store, "_platform_config_dir", lambda: old)
    return current, old


def test_the_real_probe_lists_exegol_mcps_config_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    current, old = _server_config_dirs(monkeypatch, root)

    dirs = host_module._exegol_mcp_config_dirs()

    assert [mount.lexical for mount in dirs] == [str(current), str(old)]
    assert "exegol-mcp's config directory" in dirs[0].label
    assert "bearer secret" in dirs[0].label
    assert "earlier exegol-mcp versions" in dirs[1].label
    for mount in dirs:
        assert mount.holds_server_config is True
        assert mount.holds_mount_settings is False
        assert mount.children_mounted is False
        assert mount.candidates[0] == mount.lexical
        _assert_one_role(mount)
    # The check only resolves the paths: the missing old directory is walked lexically and never created.
    assert not old.exists()
    assert not (root / "home" / "Library").exists()


@pytest.mark.parametrize(
    ("entry", "witness", "phrase"),
    [
        (("home", ".exegol-mcp"), ("home", ".exegol-mcp"), "is exegol-mcp's config directory"),
        (("home", ".exegol-mcp", "work"), ("home", ".exegol-mcp"), "lies inside exegol-mcp's config directory"),
        (("home",), ("home", ".exegol-mcp"), "contains exegol-mcp's config directory"),
        (
            ("home", "Library"),
            ("home", "Library", "Application Support", "exegol-mcp"),
            "contains the config directory earlier",
        ),
        (
            ("home", "Library", "Application Support", "exegol-mcp"),
            ("home", "Library", "Application Support", "exegol-mcp"),
            "is the config directory earlier",
        ),
    ],
    ids=["is-the-config-dir", "inside-the-config-dir", "contains-the-config-dir", "contains-the-old-dir", "is-the-old"],
)
def test_an_entry_nested_with_exegol_mcps_config_dir_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: tuple[str, ...], witness: tuple[str, ...], phrase: str
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _server_config_dirs(monkeypatch, root)
    raw = str(root.joinpath(*entry))

    problem = readiness.workspace_allowlist_mount_problem(
        [str(root / "engagements"), raw], host_module._exegol_mcp_config_dirs(), fold_case=False
    )

    assert problem is not None
    assert problem.startswith(f"Invalid config.yml: field 'lifecycle.workspace_allowlist.1': '{raw}' {phrase}")
    assert f"('{root.joinpath(*witness)}')" in problem
    assert "exegol-mcp reads its config.yml (the operator's restrictions) and its bearer secret from there" in problem
    assert problem.endswith(readiness._SERVER_CONFIG_PROBLEM_FIX)
    assert "Exegol reads the settings" not in problem
    assert readiness._SETTINGS_PROBLEM_FIX not in problem
    assert _PLANNING_ID.search(problem) is None, problem


def test_the_server_config_fix_names_both_config_directories() -> None:
    fix = readiness._SERVER_CONFIG_PROBLEM_FIX

    assert "exegol-mcp's config directory" in fix
    assert "Exegol's config directory" in fix
    assert _PLANNING_ID.search(fix) is None


def test_an_entry_beside_exegol_mcps_config_dir_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _server_config_dirs(monkeypatch, root)
    mounts = host_module._exegol_mcp_config_dirs()

    for entry in (
        root / "home" / "engagements",
        root / "home" / ".exegol-mcp-old",
        root / "home" / "Library" / "Application Support" / "exegol-mcp-notes",
        root / "elsewhere",
    ):
        assert readiness.workspace_allowlist_mount_problem([str(entry)], mounts, fold_case=False) is None, entry


def _roots_v1_config_dir(monkeypatch: pytest.MonkeyPatch, root: Path, sudo: bool) -> Path:
    """Point the v1 secret a sudo start adopts at root/root/.config/exegol-mcp (not created), sudo on or off."""
    legacy = root / "root" / ".config" / "exegol-mcp"
    monkeypatch.setattr(secret_store, "is_sudo_context", lambda: sudo)
    monkeypatch.setattr(secret_store, "_legacy_secret_file_path", lambda: legacy / secret_store.SECRET_FILE_NAME)
    return legacy


def test_under_sudo_the_real_probe_also_lists_roots_v1_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """WR-09: a sudo start copies root's v1 bearer secret into ~/.exegol-mcp when the invoker has none, so a workspace
    reaching that directory could plant the secret the next sudo start adopts."""
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    current, old = _server_config_dirs(monkeypatch, root)
    legacy = _roots_v1_config_dir(monkeypatch, root, sudo=True)

    dirs = host_module._exegol_mcp_config_dirs()

    assert [mount.lexical for mount in dirs] == [str(current), str(old), str(legacy)]
    assert "v1" in dirs[2].label and "sudo" in dirs[2].label and "bearer secret" in dirs[2].label
    assert dirs[2].holds_server_config is True
    _assert_one_role(dirs[2])
    assert not (root / "root").exists()  # walked lexically, never created
    for entry in (root / "root" / ".config", legacy, legacy / "work"):
        problem = readiness.workspace_allowlist_mount_problem([str(entry)], dirs, fold_case=False)
        assert problem is not None, entry
        assert f"('{legacy}')" in problem
        assert "exegol-mcp reads its config.yml (the operator's restrictions) and its bearer secret from there" in problem
        assert problem.endswith(readiness._SERVER_CONFIG_PROBLEM_FIX)
        assert _PLANNING_ID.search(problem) is None, problem


def test_outside_sudo_roots_v1_config_dir_is_not_listed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only a sudo start adopts root's v1 secret, so only then does its directory restrict the allowlist."""
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    current, old = _server_config_dirs(monkeypatch, root)
    _roots_v1_config_dir(monkeypatch, root, sudo=False)

    assert [mount.lexical for mount in host_module._exegol_mcp_config_dirs()] == [str(current), str(old)]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
@pytest.mark.parametrize("entry", [("entry",), ("entry", "cfg")], ids=["holds-the-target", "is-the-target"])
def test_a_symlinked_exegol_mcp_config_dir_reaching_an_entry_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: tuple[str, ...]
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    target = root / "entry" / "cfg"
    target.mkdir(parents=True)
    link = root / "home" / ".exegol-mcp"
    link.parent.mkdir(parents=True)
    os.symlink(target, link)
    monkeypatch.setattr(secret_store, "user_config_dir", lambda: link)
    monkeypatch.setattr(secret_store, "_platform_config_dir", lambda: root / "home" / ".config" / "exegol-mcp")

    problem = readiness.workspace_allowlist_mount_problem(
        [str(root.joinpath(*entry))], host_module._exegol_mcp_config_dirs(), fold_case=False
    )

    assert problem is not None
    assert f"holds '{target}', which the path to exegol-mcp's config directory" in problem
    assert f"('{link}') goes through, following symlinks" in problem
    assert "exegol-mcp reads its config.yml" in problem
    assert problem.endswith(readiness._SERVER_CONFIG_PROBLEM_FIX)


async def test_an_entry_reaching_exegol_mcps_config_dir_stops_startup_with_exegols_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _settings_tree(monkeypatch, root, server_config=True)
    current, old = _server_config_dirs(monkeypatch, root)
    mounts = host_module.RealHostProbe().exegol_mount_dirs([])
    assert [mount.lexical for mount in mounts if mount.holds_server_config] == [str(current), str(old)]

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=mounts),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(root / "work"), str(current)],
        )

    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.1" in str(raised.value)
    assert "exegol-mcp reads its config.yml" in str(raised.value)


class _RealMountsProbe(FakeHostProbe):
    """FakeHostProbe whose mount set is the real probe's, so an error resolving a directory reaches the gate. The real
    probe is built at construction, so a test may then replace host.RealHostProbe with this probe."""

    def __init__(self, **kwargs) -> None:
        import exegol_mcp.adapter.host as host_module

        super().__init__(**kwargs)
        self._real = host_module.RealHostProbe()

    def exegol_mount_dirs(self, profiles) -> list[ExegolMountDir]:
        self.calls.append("mounts")
        return self._real.exegol_mount_dirs(profiles)


@pytest.mark.parametrize("failing", ["user_config_dir", "_platform_config_dir"])
async def test_an_unresolvable_exegol_mcp_config_dir_stops_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing: str
) -> None:
    root = _real_root(tmp_path)
    _settings_tree(monkeypatch, root, server_config=True)
    _server_config_dirs(monkeypatch, root)

    def _unresolved() -> Path:
        raise secret_store.UnresolvedSudoError("the sudo invoking user cannot be resolved")

    monkeypatch.setattr(secret_store, failing, _unresolved)

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            _RealMountsProbe(),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(root / "work")],
        )

    assert raised.value.check == "workspace-allowlist"
    assert "cannot be checked and exegol-mcp refuses to start" in str(raised.value)
    assert "the sudo invoking user cannot be resolved" in str(raised.value)


# --- CR-03: the Python installation exegol-mcp and Exegol run code from; Exegol's read-only binds -----------------

_VENV_LABEL = "the Python environment exegol-mcp runs from (sys.prefix)"


def test_an_entry_nested_with_an_install_root_is_refused_with_its_own_reason() -> None:
    venv = ExegolMountDir("/srv/py/.venv", _VENV_LABEL, False, holds_host_code=True)

    for entry in ("/srv/py/.venv/lib", "/srv/py", "/srv/py/.venv"):
        problem = readiness.workspace_allowlist_mount_problem([entry], [venv], fold_case=False)
        assert problem is not None, entry
        assert "lifecycle.workspace_allowlist.0" in problem
        assert _VENV_LABEL in problem
        assert "run code" in problem
        assert problem.endswith(readiness._INSTALL_PROBLEM_FIX)
        assert readiness._MOUNT_PROBLEM_FIX not in problem
        assert readiness._SETTINGS_PROBLEM_FIX not in problem

    assert readiness.workspace_allowlist_mount_problem(["/srv/py-other"], [venv], fold_case=False) is None
    # Encoding edge (IN-08): a case-differing spelling of a directory holding the root follows the host case rule.
    folded = readiness.workspace_allowlist_mount_problem(["/SRV/py"], [venv], fold_case=True)
    assert folded is not None and "run code" in folded
    assert readiness.workspace_allowlist_mount_problem(["/SRV/py"], [venv], fold_case=False) is None


@pytest.mark.parametrize(
    ("entry", "root_name"),
    [
        (("py", "proj"), "sys.prefix"),
        (("py", "proj", ".venv"), "sys.prefix"),
        (("py", "proj", ".venv", "lib", "python3.10", "site-packages", "somepkg"), "sys.prefix"),
        (("py", "proj", "exegol_mcp"), "package directory"),
        (("py", "base", "lib"), "sys.base_prefix"),
    ],
    ids=["parent-of-venv", "the-venv", "inside-the-venv", "the-package-dir", "inside-the-base"],
)
async def test_an_entry_holding_the_python_install_root_stops_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: tuple[str, ...], root_name: str
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _install_tree(monkeypatch, root)
    root.joinpath(*entry).mkdir(parents=True, exist_ok=True)
    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    mounts = host_module.RealHostProbe().exegol_mount_dirs([])

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=mounts),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(root.joinpath(*entry))],
        )

    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "run code" in message
    assert root_name in message


async def test_an_entry_apart_from_the_python_install_roots_passes_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _install_tree(monkeypatch, root)
    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    probe = FakeHostProbe(mount_dirs=host_module.RealHostProbe().exegol_mount_dirs([]))

    await readiness.check_readiness(
        probe,
        _args("--type", "stdio"),
        prompt=_never_prompt,
        workspace_allowlist=[str(root / "work"), str(root / "py" / "proj" / "tests")],
    )

    assert probe.calls == ["patch-points", "docker", "config", "mounts", "eula"]


def test_the_real_probe_reports_this_interpreters_install_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import site

    from exegol.config.ConstantConfig import ConstantConfig

    import exegol_mcp
    import exegol_mcp.adapter.host as host_module

    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)

    roots = [mount for mount in host_module.RealHostProbe().exegol_mount_dirs([]) if mount.holds_host_code]

    expected: list[str] = []
    for value in (sys.prefix, sys.base_prefix, ConstantConfig.src_root_path_obj, Path(exegol_mcp.__file__).parent):
        path = os.path.abspath(os.fspath(value))
        if path not in expected:
            expected.append(path)
    install_roots = len(expected)
    # CR-04: then every import-path entry, read at run time ('' is the working directory, a non-path is skipped).
    for entry in sys.path:
        if not isinstance(entry, (str, bytes, os.PathLike)):
            continue
        text = os.fsdecode(entry)
        path = os.path.abspath(text or os.getcwd())
        if path not in expected:
            expected.append(path)
    # WR-17, IN-25, WR-18: then the user site, even before it exists, whenever this start enables it or a normal start
    # of this interpreter reads it, as CPython itself reports in a child process (_cpython_enables_the_user_site),
    # never as the code under test decides it.
    if site.ENABLE_USER_SITE or _cpython_enables_the_user_site(sys.executable, tmp_path):
        user_site_value = site.getusersitepackages()
        user_site = os.path.abspath(user_site_value) if user_site_value else ""
        if user_site and user_site not in expected:
            expected.append(user_site)
    assert [mount.lexical for mount in roots] == expected
    for mount in roots[install_roots:]:
        assert "sys.path entry" in mount.label or "site.getusersitepackages()" in mount.label, mount
    for mount in roots:
        assert mount.holds_mount_settings is False
        assert mount.contains_only is False
        assert mount.children_mounted is False
        assert mount.candidates[0] == mount.lexical
    labels = " ".join(mount.label for mount in roots)
    for name in ("sys.prefix", "install directory", "package directory"):
        assert name in labels, name
    if os.path.abspath(sys.base_prefix) != os.path.abspath(sys.prefix):
        assert "sys.base_prefix" in labels



# --- CR-04: every directory on exegol-mcp's import path, the client's working directory included ------------------


@pytest.mark.parametrize(
    "case",
    ["cwd-equal", "cwd-parent", "cwd-inside", "cwd-empty-entry", "path-entry-inside"],
)
async def test_an_entry_nested_with_an_import_path_directory_stops_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    cwd = root / "engagements" / "acme"
    (cwd / "loot").mkdir(parents=True)
    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    # host._python_install_dirs and the real sys module stay: this is the verifier's repro on the real interpreter.
    if case == "path-entry-inside":
        # A PYTHONPATH or .pth entry (an editable checkout), appended after the interpreter's own entries.
        monkeypatch.setattr(sys, "path", [*sys.path, str(root / "extra")])
        entry = root / "extra" / "sub"
    else:
        monkeypatch.chdir(cwd)
        # python -m puts the absolute working directory first on sys.path; python -c puts ''.
        first = "" if case == "cwd-empty-entry" else str(cwd)
        monkeypatch.setattr(sys, "path", [first, *sys.path])
        entry = {"cwd-parent": root / "engagements", "cwd-inside": cwd / "loot"}.get(case, cwd)
    mounts = host_module.RealHostProbe().exegol_mount_dirs([])

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=mounts),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(entry)],
        )

    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "run code" in message
    assert "sys.path" in message
    assert message.endswith(readiness._INSTALL_PROBLEM_FIX)
    # The fix names the working directory for every case; the reason before it names the root that matched.
    reason = message.removesuffix(readiness._INSTALL_PROBLEM_FIX)
    if case == "path-entry-inside":
        assert "import path" in reason
        assert "working directory" not in reason
    else:
        assert "working directory" in reason


def test_every_import_path_entry_is_listed_once_as_an_install_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _install_tree(monkeypatch, root)
    (root / "cwd").mkdir()
    monkeypatch.chdir(root / "cwd")
    monkeypatch.setattr(
        host_module.sys,
        "path",
        ["", "rel", os.fsencode(str(root / "b")), root / "p", 42, str(tree["venv"])],
    )

    dirs = host_module._python_install_dirs()

    assert [mount.lexical for mount in dirs] == [
        str(tree["venv"]),
        str(tree["base"]),
        str(tree["site"]),
        str(tree["pkg"]),
        str(root / "cwd"),
        str(root / "cwd" / "rel"),
        str(root / "b"),
        str(root / "p"),
    ]
    labels = {mount.lexical: mount.label for mount in dirs}
    # Empty edge: '' is the working directory, never skipped; the int is skipped; the repeated venv keeps its label.
    assert "working directory" in labels[str(root / "cwd")]
    assert "sys.path entry ''" in labels[str(root / "cwd")]
    # IN-21: an empty or explicit PYTHONPATH entry puts it there too; no launch form is credited or blamed.
    assert "PYTHONPATH" in labels[str(root / "cwd")]
    assert "console script" not in labels[str(root / "cwd")]
    assert labels[str(tree["venv"])] == "the Python environment exegol-mcp runs from (sys.prefix)"
    # Encoding edge: a bytes entry is decoded with os.fsdecode, a pathlib.Path with os.fspath.
    for path, text in ((root / "cwd" / "rel", "rel"), (root / "b", str(root / "b")), (root / "p", str(root / "p"))):
        assert "import path" in labels[str(path)], path
        assert "working directory" not in labels[str(path)], path
        assert f"sys.path entry '{text}'" in labels[str(path)], path
    for mount in dirs:
        assert mount.holds_host_code is True
        assert mount.holds_mount_settings is False
        assert mount.contains_only is False
        assert mount.children_mounted is False
        assert mount.candidates[0] == mount.lexical
    # Idempotency edge: the listing only reads the host.
    assert host_module._python_install_dirs() == dirs


# --- WR-17: the user site, listed even before it exists ------------------------------------------------------------


@pytest.mark.parametrize("case", ["enabled-missing", "enabled-on-path", "disabled", "unknown"])
def test_the_user_site_is_listed_as_an_install_root_before_it_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _install_tree(monkeypatch, root)
    user_site = tree["user_site"]
    monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", {"disabled": False, "unknown": None}.get(case, True))
    if case == "enabled-on-path":
        user_site.mkdir(parents=True)
        monkeypatch.setattr(host_module.sys, "path", [*host_module.sys.path, str(user_site)])

    dirs = host_module._python_install_dirs()

    install = [tree["venv"], tree["base"], tree["site"], tree["pkg"], tree["venv"] / "bin"]
    expected = [str(path) for path in (*install, tree["base"] / "lib" / "python3.10")]
    labels = [mount.label for mount in dirs]
    if case in ("disabled", "unknown"):
        # Empty edge: _install_tree is a venv start without system site packages (its pyvenv.cfg says false), which
        # never reads the user site, so a disabled one adds nothing there (IN-25).
        assert [mount.lexical for mount in dirs] == expected
        assert not any("user site" in label for label in labels), labels
        return
    # Ordering edge: after the four install roots and every sys.path entry, listed once.
    assert [mount.lexical for mount in dirs] == [*expected, str(user_site)]
    last = dirs[-1]
    assert last.holds_host_code is True
    assert last.holds_mount_settings is False
    assert last.contains_only is False
    assert last.children_mounted is False
    assert last.candidates[0] == last.lexical
    if case == "enabled-on-path":
        # Already on sys.path: the earlier import-path label wins.
        assert "sys.path entry" in last.label
        assert "site.getusersitepackages()" not in last.label
        return
    assert "user site" in last.label
    assert "site.getusersitepackages()" in last.label
    # Encoding edge: the missing path is walked lexically, as site returns it, and never created.
    assert not user_site.exists()
    # Idempotency edge: a second call only reads.
    assert host_module._python_install_dirs() == dirs
    # Empty edge: an empty getusersitepackages() value adds nothing.
    monkeypatch.setattr(host_module.site, "getusersitepackages", lambda: "")
    assert [mount.lexical for mount in host_module._python_install_dirs()] == expected


# --- IN-25: the user site, listed on every start of an interpreter that can read it later -------------------------

# The pyvenv.cfg of a venv stand-in: None keeps _install_tree's (include-system-site-packages = false), _NO_PYVENV_CFG
# deletes it, and str or bytes rewrites it.
_NO_PYVENV_CFG = "<deleted>"


@pytest.mark.parametrize(
    ("non_venv", "enabled", "cfg", "listed"),
    [
        # -s, -I or PYTHONNOUSERSITE set ENABLE_USER_SITE False for one start; -S leaves it None.
        pytest.param(True, False, None, True, id="non-venv-no-user-site-flag"),
        pytest.param(True, None, None, True, id="non-venv-no-site-flag"),
        pytest.param(True, True, None, True, id="non-venv-enabled"),
        pytest.param(
            False, False, "home = /x/bin\ninclude-system-site-packages = true\n", True, id="venv-system-site-true"
        ),
        # Encoding edge: key and value are stripped and case-folded, as CPython's site.venv() does.
        pytest.param(
            False, False, "  Include-System-Site-Packages =  TRUE  \n", True, id="venv-system-site-mixed-case"
        ),
        # Empty edge: CPython defaults to true when the key is absent.
        pytest.param(False, False, "home = /x/bin\n", True, id="venv-system-site-key-absent"),
        # No pyvenv.cfg where site.venv() looks, so a normal start reads the user site.
        pytest.param(False, False, _NO_PYVENV_CFG, True, id="venv-cfg-missing"),
        # Fail closed: an undecodable pyvenv.cfg counts as including system site packages.
        pytest.param(
            False, False, b"include-system-site-packages = false\n\xff\xfe\n", True, id="venv-cfg-undecodable"
        ),
        # No over-refusal: a venv whose pyvenv.cfg says anything but true never reads the user site.
        pytest.param(False, False, None, False, id="venv-system-site-false"),
        # Ordering edge: the last line wins, and only "true" counts.
        pytest.param(
            False,
            False,
            "include-system-site-packages = true\ninclude-system-site-packages = yes\n",
            False,
            id="venv-last-value-not-true",
        ),
    ],
)
def test_the_user_site_is_listed_whenever_a_later_start_could_read_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    non_venv: bool,
    enabled: bool | None,
    cfg: str | bytes | None,
    listed: bool,
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _install_tree(monkeypatch, root)
    user_site = tree["user_site"]
    monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", enabled)
    if non_venv:
        monkeypatch.setattr(host_module.sys, "executable", str(tree["base"] / "bin" / "python3.10"))
        monkeypatch.setattr(host_module.sys, "prefix", host_module.sys.base_prefix)
    if cfg == _NO_PYVENV_CFG:
        tree["pyvenv_cfg"].unlink()
    elif isinstance(cfg, bytes):
        tree["pyvenv_cfg"].write_bytes(cfg)
    elif cfg is not None:
        tree["pyvenv_cfg"].write_text(cfg, encoding="utf-8")
    cfg_before = tree["pyvenv_cfg"].read_bytes() if tree["pyvenv_cfg"].exists() else None

    dirs = host_module._python_install_dirs()

    head = [tree["base"]] if non_venv else [tree["venv"], tree["base"]]
    expected = [str(path) for path in (*head, tree["site"], tree["pkg"], tree["venv"] / "bin")]
    expected.append(str(tree["base"] / "lib" / "python3.10"))
    labels = [mount.label for mount in dirs]
    if not listed:
        assert [mount.lexical for mount in dirs] == expected
        assert not any("user site" in label for label in labels), labels
        return
    # Ordering edge: the user site comes last, after every sys.path entry.
    assert [mount.lexical for mount in dirs] == [*expected, str(user_site)]
    last = dirs[-1]
    assert last.holds_host_code is True
    assert "user site" in last.label
    assert "site.getusersitepackages()" in last.label
    # Encoding edge: the missing path is walked lexically and never created.
    assert not user_site.exists()
    # Idempotency edge: a second call returns an equal list, and pyvenv.cfg is only read.
    assert host_module._python_install_dirs() == dirs
    assert (tree["pyvenv_cfg"].read_bytes() if tree["pyvenv_cfg"].exists() else None) == cfg_before
    if non_venv and enabled is False:
        # Empty edge: an empty or None getusersitepackages() value adds nothing, on a non-venv start too.
        for empty in ("", None):
            monkeypatch.setattr(host_module.site, "getusersitepackages", lambda value=empty: value)
            assert [mount.lexical for mount in host_module._python_install_dirs()] == expected


def _user_site_probe_stubs(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Path]:
    """_install_tree with the GUI, settings and read-only producers stubbed, so only the install roots are listed."""
    import exegol_mcp.adapter.host as host_module

    tree = _install_tree(monkeypatch, root)
    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_read_only_binds", list)
    return tree


@pytest.mark.parametrize("case", ["containing", "equal", "inside"])
async def test_an_entry_nested_with_a_missing_user_site_stops_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _user_site_probe_stubs(monkeypatch, root)
    user_site = tree["user_site"]
    monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", True)
    # Adjacency edge: equal is refused like containing and inside.
    entry = {"containing": root / "home" / ".local", "equal": user_site, "inside": user_site / "pkg"}[case]
    mounts = host_module.RealHostProbe().exegol_mount_dirs([])

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=mounts),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(entry)],
        )

    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "user site" in message
    assert "run code" in message
    assert message.endswith(readiness._INSTALL_PROBLEM_FIX)
    # The check only reads: the user site is still missing.
    assert not user_site.exists()


@pytest.mark.parametrize(
    "case",
    [
        "non-venv-refused",
        "system-site-venv-refused",
        "plain-venv-loads",
        "dual-layout-venv-refused",
        "no-site-flag-plain-venv-loads",
    ],
)
async def test_a_start_with_the_user_site_disabled_refuses_an_entry_containing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _user_site_probe_stubs(monkeypatch, root)
    user_site = tree["user_site"]
    # As under -s, -I or PYTHONNOUSERSITE: this start does not read the user site, a later one may (IN-25, G-04-5).
    monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", False)
    if case == "non-venv-refused":
        monkeypatch.setattr(host_module.sys, "executable", str(tree["base"] / "bin" / "python3.10"))
        monkeypatch.setattr(host_module.sys, "prefix", host_module.sys.base_prefix)
    elif case == "system-site-venv-refused":
        tree["pyvenv_cfg"].write_text("include-system-site-packages = true\n", encoding="utf-8")
    elif case == "dual-layout-venv-refused":
        # WR-18: site.venv() reads the pyvenv.cfg next to the interpreter first; it includes system site packages
        # while the one up a level excludes them, so a normal start reads the user site.
        tree["exe_dir_cfg"].write_text("include-system-site-packages = true\n", encoding="utf-8")
    elif case == "no-site-flag-plain-venv-loads":
        # IN-28, IN-32: under -S, site.venv() never runs, and on 3.10-3.12 sys.prefix then stays the base
        # installation's even in a venv (ENABLE_USER_SITE is None); the plain venv's pyvenv.cfg still excludes system
        # site packages, so no start reads the user site.
        monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", None)
        monkeypatch.setattr(host_module.sys, "prefix", host_module.sys.base_prefix)
    entry = root / "home" / ".local"
    probe = FakeHostProbe(mount_dirs=host_module.RealHostProbe().exegol_mount_dirs([]))

    async def start() -> None:
        await readiness.check_readiness(
            probe, _args("--type", "stdio"), prompt=_never_prompt, workspace_allowlist=[str(entry)]
        )

    if case in ("plain-venv-loads", "no-site-flag-plain-venv-loads"):
        # No over-refusal: a venv whose pyvenv.cfg excludes system site packages never reads the user site.
        await start()
        assert probe.calls == ["patch-points", "docker", "config", "mounts", "eula"]
        return
    with pytest.raises(readiness.ReadinessError) as raised:
        await start()

    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "user site" in message
    assert message.endswith(readiness._INSTALL_PROBLEM_FIX)
    assert not user_site.exists()


# --- WR-18, IN-28: the user site follows the pyvenv.cfg CPython's site.venv() reads -------------------------------


def _cfg_state(path: Path) -> bytes | str | None:
    """A pyvenv.cfg candidate as it stands: its bytes, "<directory>", or None when absent."""
    if path.is_dir():
        return "<directory>"
    return path.read_bytes() if path.exists() else None


@pytest.mark.parametrize(
    ("case", "listed"),
    [
        # WR-18: the pyvenv.cfg next to the interpreter wins over the one up a level.
        pytest.param("exe-dir-cfg-true-root-false", True, id="exe-dir-cfg-true-root-false"),
        # No over-refusal: the file site.venv() reads excludes system site packages, whatever the one up a level says.
        pytest.param("exe-dir-cfg-false-root-true", False, id="exe-dir-cfg-false-root-true"),
        pytest.param("exe-dir-cfg-only-false", False, id="exe-dir-cfg-only-false"),
        # Encoding edge, fail closed: the undecodable file is the one read, so the false one up a level is not.
        pytest.param("exe-dir-cfg-undecodable", True, id="exe-dir-cfg-undecodable"),
        # Ordering edge: os.path.isfile skips a directory, as CPython does, so the file up a level is read.
        pytest.param("exe-dir-cfg-directory-skipped", False, id="exe-dir-cfg-directory-skipped"),
        # IN-28, IN-32: under -S, site.venv() never runs, and on 3.10-3.12 sys.prefix then stays the base
        # installation's even in a venv, which plays no part.
        pytest.param("no-site-flag-plain-venv", False, id="no-site-flag-plain-venv"),
        # Empty edge, fail closed: an unknown executable lists the user site.
        pytest.param("executable-empty", True, id="executable-empty"),
        pytest.param("executable-none", True, id="executable-none"),
        # site.venv() starts from __PYVENV_LAUNCHER__ on macOS only.
        pytest.param("launcher-on-darwin", True, id="launcher-on-darwin"),
        pytest.param("launcher-ignored-off-darwin", False, id="launcher-ignored-off-darwin"),
    ],
)
def test_the_user_site_follows_the_pyvenv_cfg_cpython_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, listed: bool
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _install_tree(monkeypatch, root)
    user_site = tree["user_site"]
    exe_dir_cfg = tree["exe_dir_cfg"]
    root_cfg = tree["pyvenv_cfg"]
    other_cfg = root / "py" / "other" / "pyvenv.cfg"
    monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", False)
    true_cfg = "include-system-site-packages = true\n"
    false_cfg = "include-system-site-packages = false\n"
    if case == "exe-dir-cfg-true-root-false":
        exe_dir_cfg.write_text(true_cfg, encoding="utf-8")
    elif case == "exe-dir-cfg-false-root-true":
        root_cfg.write_text(true_cfg, encoding="utf-8")
        exe_dir_cfg.write_text(false_cfg, encoding="utf-8")
    elif case == "exe-dir-cfg-only-false":
        root_cfg.unlink()
        exe_dir_cfg.write_text(false_cfg, encoding="utf-8")
    elif case == "exe-dir-cfg-undecodable":
        exe_dir_cfg.write_bytes(b"\xff\xfe\n")
    elif case == "exe-dir-cfg-directory-skipped":
        exe_dir_cfg.mkdir()
    elif case == "no-site-flag-plain-venv":
        monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", None)
        monkeypatch.setattr(host_module.sys, "prefix", host_module.sys.base_prefix)
    elif case == "executable-empty":
        monkeypatch.setattr(host_module.sys, "executable", "")
    elif case == "executable-none":
        monkeypatch.setattr(host_module.sys, "executable", None)
    elif case in ("launcher-on-darwin", "launcher-ignored-off-darwin"):
        monkeypatch.setattr(host_module.sys, "platform", "darwin" if case == "launcher-on-darwin" else "linux")
        other_cfg.parent.mkdir(parents=True)
        other_cfg.write_text(true_cfg, encoding="utf-8")
        monkeypatch.setenv("__PYVENV_LAUNCHER__", str(root / "py" / "other" / "bin" / "python"))
    candidates = (exe_dir_cfg, root_cfg, other_cfg)
    before = [_cfg_state(path) for path in candidates]

    dirs = host_module._python_install_dirs()

    # IN-32: the venv site.venv() reads a pyvenv.cfg from is listed after sys.base_prefix unless it repeats sys.prefix.
    head = {
        "no-site-flag-plain-venv": [tree["base"], tree["venv"]],
        "launcher-on-darwin": [tree["venv"], tree["base"], root / "py" / "other"],
    }.get(case, [tree["venv"], tree["base"]])
    expected = [str(path) for path in (*head, tree["site"], tree["pkg"], tree["venv"] / "bin")]
    expected.append(str(tree["base"] / "lib" / "python3.10"))
    labels = [mount.label for mount in dirs]
    if not listed:
        assert [mount.lexical for mount in dirs] == expected
        assert not any("user site" in label for label in labels), labels
        return
    # Ordering edge: the user site comes last, after every sys.path entry.
    assert [mount.lexical for mount in dirs] == [*expected, str(user_site)]
    assert "user site" in dirs[-1].label
    assert "site.getusersitepackages()" in dirs[-1].label
    # Encoding edge: the missing path is walked lexically and never created.
    assert not user_site.exists()
    # Idempotency edge: a second call returns an equal list, and every pyvenv.cfg candidate is only read.
    assert host_module._python_install_dirs() == dirs
    assert [_cfg_state(path) for path in candidates] == before


@pytest.mark.skipif(sys.platform == "win32", reason="native Windows is not a tested platform")
@pytest.mark.parametrize("layout", ["plain", "system-site", "dual", "reverse", "exe-dir-only", "no-cfg"])
def test_the_user_site_rule_agrees_with_cpython_on_real_venv_layouts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, layout: str
) -> None:
    import venv

    import exegol_mcp.adapter.host as host_module

    v = tmp_path / "v"
    venv.create(v, with_pip=False, symlinks=True, system_site_packages=layout in ("system-site", "reverse"))
    root_cfg = v / "pyvenv.cfg"
    if layout in ("dual", "reverse"):
        # A second pyvenv.cfg next to the interpreter, which site.venv() reads first: true for dual, false for reverse.
        value = "true" if layout == "dual" else "false"
        lines = [
            f"include-system-site-packages = {value}" if line.startswith("include-system-site-packages") else line
            for line in root_cfg.read_text(encoding="utf-8").splitlines()
        ]
        (v / "bin" / "pyvenv.cfg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif layout == "exe-dir-only":
        root_cfg.rename(v / "bin" / "pyvenv.cfg")
    elif layout == "no-cfg":
        root_cfg.unlink()
    executable = str(v / "bin" / "python")

    normal = _cpython_enables_the_user_site(executable, tmp_path)
    dropped = ("PYTHONNOUSERSITE", "__PYVENV_LAUNCHER__")
    child = subprocess.run(
        [executable, "-s", "-c", "import json, sys; print(json.dumps([sys.executable, sys.prefix, sys.base_prefix]))"],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={name: value for name, value in _child_env().items() if name not in dropped},
    )
    state = json.loads(child.stdout)
    user_site = str(_real_root(tmp_path) / "home" / "site-packages")
    monkeypatch.setattr(
        host_module,
        "sys",
        types.SimpleNamespace(
            executable=state[0], prefix=state[1], base_prefix=state[2], path=[], platform=sys.platform
        ),
    )
    monkeypatch.setattr(
        host_module, "site", types.SimpleNamespace(ENABLE_USER_SITE=False, getusersitepackages=lambda: user_site)
    )
    monkeypatch.delenv("__PYVENV_LAUNCHER__", raising=False)

    # As under -s: this start disables the user site, so it is listed exactly when a normal start reads it.
    listed = user_site in [mount.lexical for mount in host_module._python_install_dirs()]

    assert listed is normal, (layout, normal, state)


# --- IN-32: the venv whose pyvenv.cfg site.venv() reads is protected even under -S ----------------------------------

_VENV_IT_STARTS_IN_LABEL = (
    "the virtual environment this interpreter starts in (the directory holding the pyvenv.cfg site reads)"
)


@pytest.mark.parametrize("cfg", ["cfg-one-level-up", "cfg-next-to-interpreter"])
async def test_a_no_site_start_refuses_an_entry_inside_the_venv_it_starts_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: str
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _user_site_probe_stubs(monkeypatch, root)
    # IN-32: a `python -S -m exegol_mcp` launch with PYTHONPATH at the venv's site-packages. Under -S, site.venv()
    # never runs, and on 3.10-3.12 sys.prefix then stays the base installation's, so no other root holds venv/bin.
    monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", None)
    monkeypatch.setattr(host_module.sys, "prefix", host_module.sys.base_prefix)
    (root / "cwd").mkdir()
    monkeypatch.chdir(root / "cwd")
    monkeypatch.setattr(
        host_module.sys, "path", [str(root / "cwd"), str(tree["site"]), str(tree["base"] / "lib" / "python3.10")]
    )
    if cfg == "cfg-next-to-interpreter":
        # The root is dirname(exe_dir) whichever candidate site.venv() finds.
        tree["pyvenv_cfg"].unlink()
        tree["exe_dir_cfg"].write_text("include-system-site-packages = false\n", encoding="utf-8")
    candidates = (tree["exe_dir_cfg"], tree["pyvenv_cfg"])
    before = [_cfg_state(path) for path in candidates]
    # A container mounting venv/bin could plant bin/pyvenv.cfg, which a later normal start reads first.
    entry = tree["venv"] / "bin"

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=host_module.RealHostProbe().exegol_mount_dirs([])),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(entry)],
        )

    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "virtual environment this interpreter starts in" in message
    assert "pyvenv.cfg site reads" in message
    assert "run code" in message
    assert message.endswith(readiness._INSTALL_PROBLEM_FIX)
    # The check only reads: both pyvenv.cfg candidates are unchanged.
    assert [_cfg_state(path) for path in candidates] == before


@pytest.mark.parametrize(
    "case", ["normal-start-listed-once", "no-site-start-listed", "launcher-on-darwin-listed", "no-cfg-not-listed"]
)
def test_the_venv_it_starts_in_is_an_install_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _install_tree(monkeypatch, root)
    venv_dir, base = str(tree["venv"]), tree["base"]
    other = root / "py" / "other"
    if case == "no-site-start-listed":
        # IN-32: the -S stand-in (sys.prefix is the base installation's on 3.10-3.12).
        monkeypatch.setattr(host_module.site, "ENABLE_USER_SITE", None)
        monkeypatch.setattr(host_module.sys, "prefix", host_module.sys.base_prefix)
    elif case == "launcher-on-darwin-listed":
        monkeypatch.setattr(host_module.sys, "platform", "darwin")
        other.mkdir(parents=True)
        (other / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
        monkeypatch.setenv("__PYVENV_LAUNCHER__", str(other / "bin" / "python"))
    elif case == "no-cfg-not-listed":
        monkeypatch.setattr(host_module.sys, "executable", str(base / "bin" / "python3.10"))
        monkeypatch.setattr(host_module.sys, "prefix", host_module.sys.base_prefix)
    cfg_before = tree["pyvenv_cfg"].read_bytes()

    dirs = host_module._python_install_dirs()

    lexicals = [mount.lexical for mount in dirs]
    labels = [mount.label for mount in dirs]
    if case == "normal-start-listed-once":
        # No duplicate root: the seen set drops it as a repeat of sys.prefix, whose label wins.
        assert lexicals.count(venv_dir) == 1
        assert dirs[lexicals.index(venv_dir)].label == "the Python environment exegol-mcp runs from (sys.prefix)"
        assert not any("virtual environment this interpreter starts in" in label for label in labels), labels
    elif case == "no-site-start-listed":
        tail = [tree["site"], tree["pkg"], tree["venv"] / "bin", base / "lib" / "python3.10"]
        assert lexicals == [str(base), venv_dir, *(str(path) for path in tail)]
        mount = dirs[1]
        assert mount.label == _VENV_IT_STARTS_IN_LABEL
        assert mount.holds_host_code is True
        assert mount.holds_mount_settings is False
        assert mount.contains_only is False
        assert mount.children_mounted is False
        # Idempotency edge: a second call returns an equal list, and pyvenv.cfg is only read.
        assert host_module._python_install_dirs() == dirs
        assert tree["pyvenv_cfg"].read_bytes() == cfg_before
    elif case == "launcher-on-darwin-listed":
        assert lexicals[:4] == [venv_dir, str(base), str(other), str(tree["site"])]
        assert dirs[2].label == _VENV_IT_STARTS_IN_LABEL
    else:
        # Empty edge: no pyvenv.cfg where site.venv() looks adds no root.
        assert not any("virtual environment this interpreter starts in" in label for label in labels), labels


def _start_state(executable: str, flag: str, cwd: Path) -> list[str]:
    """[sys.executable, sys.prefix, sys.base_prefix] a start of that interpreter with flag reports, from a child
    process, so a test never takes them from the code under test (IN-32)."""
    dropped = ("PYTHONNOUSERSITE", "__PYVENV_LAUNCHER__")
    child = subprocess.run(
        [executable, flag, "-c", "import json, sys; print(json.dumps([sys.executable, sys.prefix, sys.base_prefix]))"],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
        env={name: value for name, value in _child_env().items() if name not in dropped},
    )
    state = json.loads(child.stdout)
    assert isinstance(state, list) and len(state) == 3, child
    return [str(value) for value in state]


@pytest.mark.skipif(sys.platform == "win32", reason="native Windows is not a tested platform")
@pytest.mark.parametrize("flag", ["-S", "-s"])
def test_a_real_no_site_start_lists_the_venv_it_starts_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str
) -> None:
    import venv

    import exegol_mcp.adapter.host as host_module

    v = tmp_path / "v"
    venv.create(v, with_pip=False, symlinks=True)
    state = _start_state(str(v / "bin" / "python"), flag, tmp_path)
    user_site = str(_real_root(tmp_path) / "home" / "site-packages")
    monkeypatch.setattr(
        host_module,
        "sys",
        types.SimpleNamespace(
            executable=state[0], prefix=state[1], base_prefix=state[2], path=[], platform=sys.platform
        ),
    )
    monkeypatch.setattr(
        host_module,
        "site",
        types.SimpleNamespace(ENABLE_USER_SITE=None if flag == "-S" else False, getusersitepackages=lambda: user_site),
    )
    monkeypatch.delenv("__PYVENV_LAUNCHER__", raising=False)

    lexicals = [mount.lexical for mount in host_module._python_install_dirs()]

    # IN-32: on 3.10-3.12 a -S start reports the base installation as sys.prefix; the venv is still listed, once.
    assert lexicals.count(os.path.abspath(v)) == 1, (flag, state, lexicals)


@pytest.mark.skipif(sys.platform == "win32", reason="the working directory cannot be removed on Windows")
async def test_a_deleted_working_directory_stops_startup_naming_the_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _user_site_probe_stubs(monkeypatch, root)
    gone = root / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    with pytest.raises(OSError) as error:
        host_module.RealHostProbe().exegol_mount_dirs([])
    monkeypatch.chdir(root)

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_error=error.value),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(root / "work")],
        )

    # Fail-closed edge: a working directory deleted before startup stops it, and the message names the import path.
    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "imports code from" in message
    assert "sys.path" in message
    assert "lifecycle.workspace_allowlist" in message


_SENTINEL_BIND_LABEL = "Exegol's read-only bind of the Sentinel logger (ConstantConfig.sentinel_context_path_obj)"


def test_an_entry_inside_a_read_only_bind_source_loads_but_one_containing_it_is_refused() -> None:
    bind = ExegolMountDir("/srv/x/sentinel", _SENTINEL_BIND_LABEL, False, contains_only=True)

    # Inside a read-only mount: no container writes through it. Equal to it: directly mounted, "same" never applies.
    assert readiness.workspace_allowlist_mount_problem(["/srv/x/sentinel/sub"], [bind], fold_case=False) is None
    assert readiness.workspace_allowlist_mount_problem(["/srv/x/sentinel"], [bind], fold_case=False) is None
    problem = readiness.workspace_allowlist_mount_problem(["/srv/x"], [bind], fold_case=False)
    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert _SENTINEL_BIND_LABEL in problem
    assert "read-only" in problem
    assert problem.endswith(readiness._READ_ONLY_PROBLEM_FIX)

    linked = ExegolMountDir(
        path="/opt/real/sentinel",
        label=_SENTINEL_BIND_LABEL,
        children_mounted=False,
        lexical="/srv/entry/link/sentinel",
        candidates=(
            "/srv/entry/link/sentinel",
            "/srv",
            "/srv/entry",
            "/srv/entry/link",
            "/opt",
            "/opt/real",
            "/opt/real/sentinel",
        ),
        contains_only=True,
    )
    swapped = readiness.workspace_allowlist_mount_problem(["/srv/entry"], [linked], fold_case=False)
    assert swapped is not None
    assert "/srv/entry/link/sentinel" in swapped
    assert swapped.endswith(readiness._READ_ONLY_PROBLEM_FIX)


def _read_only_binds_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe only Exegol's read-only binds: no install root (it would pre-empt the rule), settings dir or GUI."""
    import exegol_mcp.adapter.host as host_module

    _stub_exegol_dirs(monkeypatch, {})
    monkeypatch.setattr(host_module, "GuiUtils", _no_gui())
    monkeypatch.setattr(host_module, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host_module, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host_module, "_python_install_dirs", list)


async def test_an_entry_holding_the_spawn_read_only_bind_source_stops_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    tree = _install_tree(monkeypatch, root)
    _read_only_binds_only(monkeypatch)
    mounts = host_module.RealHostProbe().exegol_mount_dirs([])

    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=mounts),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            workspace_allowlist=[str(tree["imgsync"])],
        )

    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "spawn.sh" in message
    assert "read-only" in message


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_a_symlinked_read_only_bind_source_under_an_entry_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    _install_tree(monkeypatch, root)
    _read_only_binds_only(monkeypatch)
    (root / "entry" / "imgsync").mkdir(parents=True)
    (root / "entry" / "imgsync" / "spawn.sh").touch()
    (root / "other").mkdir()
    os.symlink(root / "entry" / "imgsync", root / "other" / "imgsync")
    monkeypatch.setattr(host_module.ConstantConfig, "spawn_context_path_obj", root / "other" / "imgsync" / "spawn.sh")

    mounts = host_module.RealHostProbe().exegol_mount_dirs([])
    problem = readiness.workspace_allowlist_mount_problem([str(root / "entry")], mounts, fold_case=False)

    # Docker's walk reaches spawn.sh through root/entry/imgsync: a container mounting the entry could swap it.
    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert f"'{root / 'entry' / 'imgsync'}'" in problem
    assert "spawn.sh" in problem
    assert problem.endswith(readiness._READ_ONLY_PROBLEM_FIX)


def test_the_real_spawn_read_only_bind_source_under_an_entry_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from exegol.config.ConstantConfig import ConstantConfig

    import exegol_mcp.adapter.host as host_module

    _read_only_binds_only(monkeypatch)
    entry = os.path.realpath(ConstantConfig.spawn_context_path_obj.parent)

    problem = readiness.workspace_allowlist_mount_problem([entry], host_module.RealHostProbe().exegol_mount_dirs([]))

    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert "spawn.sh" in problem


def test_the_real_probe_reports_exegols_read_only_bind_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import exegol_mcp.adapter.host as host_module

    root = _real_root(tmp_path)
    names = ("spawn.sh", "entrypoint.sh", "zsh", "bash", "Sentinel logger")

    for linux in (True, False):
        tree = _install_tree(monkeypatch, root, linux=linux)
        _read_only_binds_only(monkeypatch)
        imgsync = tree["imgsync"]
        sources = [
            imgsync / "spawn.sh",
            imgsync / "entrypoint.sh",
            imgsync / "shell_script" / "zsh_hooks.sh",
            imgsync / "shell_script" / "bash_hooks.sh",
            imgsync / "sentinel",
        ]

        binds = host_module.RealHostProbe().exegol_mount_dirs([])

        if linux and sys.platform == "win32":
            continue  # the /etc timezone sources are POSIX paths
        expected = [str(source) for source in sources] + (["/etc/timezone", "/etc/localtime"] if linux else [])
        assert [mount.lexical for mount in binds] == expected, linux
        for mount, name in zip(binds, (*names, "/etc/timezone", "/etc/localtime")):
            assert name in mount.label
            assert mount.contains_only is True
            assert mount.children_mounted is False
            assert mount.holds_mount_settings is False
            assert mount.holds_host_code is False
            assert mount.candidates[0] == mount.lexical


def test_exegols_read_only_bind_calls_still_match_the_probe() -> None:
    import inspect

    import exegol.manager.ExegolManager as exegol_manager
    import exegol.model.ContainerConfig as container_config
    from exegol.config.ConstantConfig import ConstantConfig

    read_only = re.compile(r"addVolume\(([^,\n]+),[^\n]*read_only=True")

    # A new read-only bind in a future Exegol must be added to host._exegol_read_only_binds or profiles.py.
    assert set(read_only.findall(inspect.getsource(container_config))) == {
        "str(ConstantConfig.spawn_context_path_obj)",
        '"/etc/timezone"',
        '"/etc/localtime"',
        "ConstantConfig.sentinel_zsh_context_path_obj",
        "ConstantConfig.sentinel_bash_context_path_obj",
        "ConstantConfig.sentinel_context_path_obj",
        "vpn_path",
        "vpn_auth",
        "wireguard_path",
    }
    assert set(read_only.findall(inspect.getsource(exegol_manager))) == {"ConstantConfig.entrypoint_context_path_obj"}
    for attribute, relative in (
        ("entrypoint_context_path_obj", "exegol/utils/imgsync/entrypoint.sh"),
        ("spawn_context_path_obj", "exegol/utils/imgsync/spawn.sh"),
        ("sentinel_zsh_context_path_obj", "exegol/utils/imgsync/shell_script/zsh_hooks.sh"),
        ("sentinel_bash_context_path_obj", "exegol/utils/imgsync/shell_script/bash_hooks.sh"),
        ("sentinel_context_path_obj", "exegol/utils/imgsync/sentinel"),
    ):
        assert getattr(ConstantConfig, attribute) == ConstantConfig.src_root_path_obj / relative, attribute
    shared_timezone = inspect.getsource(container_config.ContainerConfig.enableSharedTimezone)
    assert "EnvInfo.is_windows_shell or EnvInfo.is_mac_shell" in shared_timezone


def test_the_install_root_and_read_only_bind_points_are_self_checked() -> None:
    for pair in (
        ("exegol.config.ConstantConfig", "ConstantConfig.src_root_path_obj"),
        ("exegol.config.ConstantConfig", "ConstantConfig.entrypoint_context_path_obj"),
        ("exegol.config.ConstantConfig", "ConstantConfig.spawn_context_path_obj"),
        ("exegol.config.ConstantConfig", "ConstantConfig.sentinel_zsh_context_path_obj"),
        ("exegol.config.ConstantConfig", "ConstantConfig.sentinel_bash_context_path_obj"),
        ("exegol.config.ConstantConfig", "ConstantConfig.sentinel_context_path_obj"),
        ("exegol.config.EnvInfo", "EnvInfo.is_windows_shell"),
        ("exegol.config.EnvInfo", "EnvInfo.is_mac_shell"),
    ):
        assert pair in selfcheck._ATTRIBUTE_POINTS, pair
    selfcheck.assert_patch_points()
