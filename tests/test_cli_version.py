"""The read-only `version` action, its only spelling (D-07, D-08, D-11, D-18, D-19, D-22, D-25). Docker-free.

The tracer runs the real entry point in a subprocess with a dead DOCKER_HOST and a scratch HOME that must stay empty;
the in-process tests call main() with every side-effecting step replaced by a function that fails the test.
"""

import importlib.metadata
import json
import logging
import os
import platform
import re
import stat
import subprocess
import sys
import threading
import time
import types
import urllib.error
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.text import Text

from exegol_mcp import __version__
from exegol_mcp import main as main_module
from exegol_mcp.adapter import diagnostics, selfcheck
from exegol_mcp.adapter.diagnostics import DockerFacts, SdkFacts
from exegol_mcp.bootstrap import logging_setup, secret_store, version_command, versions
from exegol_mcp.bootstrap.cli import parse_args
from exegol_mcp.main import main
from tests.fake_server import REPO_ROOT, server_env

DEAD_DOCKER = "unix:///nonexistent/docker.sock"
# What the base `version` report may load of the Exegol SDK (D-19); plan 05.2-04 reuses it.
LIGHT_SDK_MODULES = frozenset(
    {
        "exegol",
        "exegol.config",
        "exegol.config.ConstantConfig",
        "exegol.config.EnvInfo",
        "exegol.config.SudoInfo",
        "exegol.utils",
        "exegol.utils.ExeLog",
    }
)
BETA_NOTICE = "[!] You are currently using a Beta version of Exegol MCP, which may be unstable."


@pytest.fixture
def restore_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo what main() does to the process's logging and to the SDK's console and critical method."""
    from exegol.utils.ExeLog import ExeLog, logger

    root = logging.getLogger()
    package = logging.getLogger("exegol_mcp")
    saved = (list(root.handlers), root.level, package.level, logger.level, ExeLog.console)
    monkeypatch.setattr(logging_setup, "_console", logging_setup._console)
    monkeypatch.setattr(logger, "_ExeLog__critical_method", logger._ExeLog__critical_method)
    yield
    handlers, root_level, package_level, sdk_level, console = saved
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(root_level)
    package.setLevel(package_level)
    logger.setLevel(sdk_level)
    ExeLog.console = console


def _forbid(monkeypatch: pytest.MonkeyPatch, owner: object, name: str) -> None:
    """Make owner.name fail the test when called: `version` must never reach it."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(f"`version` must not call {name}")

    monkeypatch.setattr(owner, name, forbidden)


def _run_cli(home: Path, *argv: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "exegol_mcp", *argv],
        env=server_env(home, DOCKER_HOST=DEAD_DOCKER, **env),
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )


def test_version_runs_end_to_end_without_touching_home(tmp_path: Path) -> None:
    # The plan tracer: the real entry point, Docker down, a scratch HOME that must stay empty.
    home = tmp_path / "scratch-home"
    home.mkdir()

    result = _run_cli(home, "version")

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"[*] Exegol MCP is currently in version v{__version__}"
    assert f"[*] Exegol library: {importlib.metadata.version('exegol')}" in lines
    assert "[*] More about Exegol MCP at: https://docs.exegol.com/mcp/" in lines
    assert BETA_NOTICE in lines  # __version__ is a beta
    assert "Traceback" not in result.stderr
    assert list(home.rglob("*")) == []


def test_version_vvv_runs_end_to_end_without_touching_home(tmp_path: Path) -> None:
    # The plan tracer: the real entry point at -vvv, Docker down, a scratch HOME that must stay empty.
    home = tmp_path / "scratch-home"
    home.mkdir()

    result = _run_cli(home, "version", "-vvv", "--no-update-check")

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert f"[D] Exegol config directory: {home}/.exegol" in lines
    assert f"[D] Exegol config file: {home}/.exegol/config.yml (absent)" in lines
    assert f"[D] Datastore file: {home}/.exegol/.datastore" in lines
    assert "[D] EULA: not accepted (no datastore)" in lines
    assert any(line.startswith("[D] Docker: unreachable (") for line in lines)
    assert "[D] Docker engine: unknown" in lines
    assert "[D] Docker desktop: unknown" in lines
    assert any(line.startswith("[D] Host OS: ") for line in lines)
    assert any(line.startswith("[D] Arch: ") for line in lines)
    first_debug = next(index for index, line in enumerate(lines) if line.startswith("[D] "))
    assert lines.index(f"[*] Exegol MCP is currently in version v{__version__}") < first_debug
    assert lines[-1] == BETA_NOTICE
    assert "Traceback" not in result.stderr
    assert list(home.rglob("*")) == []


def test_version_vvv_stderr_carries_no_third_party_line(tmp_path: Path) -> None:
    # G-05.2-3 / D-27: the real entry point, a real docker-py probe against a dead socket, a scratch HOME. docker-py's
    # "Trying paths" / "No config file found" and keyring's "Loading <backend>" records never reach stderr.
    home = tmp_path / "scratch-home"
    home.mkdir()

    result = _run_cli(home, "version", "-vvv", "--no-update-check")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"[*] Exegol MCP is currently in version v{__version__}"
    assert any(line.startswith("[D] ") for line in result.stdout.splitlines())
    for line in result.stderr.splitlines():
        assert "Trying paths" not in line, line
        assert "No config file found" not in line, line
        assert not line.startswith("[D] Loading "), line
    assert list(home.rglob("*")) == []


def _log_third_party_noise(levels: tuple[int, ...] = (logging.DEBUG, logging.INFO)) -> None:
    """Emit records the way the libraries a `version` run loads do, each with a marker the tests look for."""
    for name in ("keyring.backend", "docker.utils.config", "urllib3.connectionpool", "asyncio", "httpx"):
        for level in levels:
            logging.getLogger(name).log(level, "noise %s %s", name, logging.getLevelName(level))


@pytest.fixture
def explicit_debug_logger() -> Iterator[logging.Logger]:
    """A third-party logger carrying a DEBUG level of its own, which the root level alone cannot silence."""
    logger = logging.getLogger("thirdparty.explicit")
    saved = logger.level
    logger.setLevel(logging.DEBUG)
    yield logger
    logger.setLevel(saved)


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_version_caps_third_party_records_at_warning(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    explicit_debug_logger: logging.Logger,
) -> None:
    def noisy_installation() -> str:
        _log_third_party_noise()
        explicit_debug_logger.debug("noise explicit debug")
        logging.getLogger("docker.api").warning("warning docker.api")
        logging.getLogger("exegol_mcp.tests.report").debug("own debug record")
        logging.getLogger("main").debug("sdk debug record")
        return "pip"

    monkeypatch.setattr(version_command, "_installation", noisy_installation)

    assert main(["version", "-vvv", "--no-update-check"]) == 0

    err = capsys.readouterr().err
    assert "noise" not in err
    assert "warning docker.api" in err
    assert "own debug record" in err
    assert "sdk debug record" in err


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
@pytest.mark.parametrize("verbosity", [[], ["-v"], ["-vv"], ["-vvv"]])
def test_version_caps_third_party_records_at_every_verbosity(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verbosity: list[str]
) -> None:
    def noisy_detect() -> str | None:
        _log_third_party_noise()
        logging.getLogger("httpx").warning("warning httpx")
        return None

    monkeypatch.setattr(version_command, "detect_wrapper_version", noisy_detect)

    assert main(["version", "--no-update-check", *verbosity]) == 0

    err = capsys.readouterr().err
    assert "noise" not in err
    assert "warning httpx" in err


@pytest.mark.usefixtures("restore_logging")
def test_the_version_cap_is_gone_after_a_server_logging_setup(capsys: pytest.CaptureFixture[str]) -> None:
    # D-27 / D-16: the cap belongs to `version` only; a server start rebuilds the handlers and third-party DEBUG prints.
    logging_setup.route_logging_to_stderr(3)
    logging_setup.cap_third_party_logs()
    logging.getLogger("httpx").debug("capped httpx debug")
    assert "capped httpx debug" not in capsys.readouterr().err

    logging_setup.route_logging_to_stderr(3)
    logging.getLogger("httpx").debug("server httpx debug")

    assert "server httpx debug" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "verbose", "action"),
    [
        ([], 0, None),
        (["version"], 0, "version"),
        (["version", "-vvv"], 3, "version"),
        (["-vvv", "version"], 3, "version"),
    ],
)
def test_version_parses_as_an_optional_action_without_changing_other_flags(
    argv: list[str], verbose: int, action: str | None
) -> None:
    args = parse_args(argv)

    assert (args.verbose, args.action) == (verbose, action)
    assert not hasattr(args, "show_version")
    assert (args.type, args.port, args.auth, args.print_config) == (None, None, None, False)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-t", "stdio", "version", "-v"], {"type": "stdio", "verbose": 1}),
        (["-p", "9", "version", "--no-update-check"], {"port": 9, "no_update_check": True}),
        (["--print-config", "version"], {"print_config": True}),
        (["version", "--accept-eula", "--auth", "bearer"], {"accept_eula": True, "auth": "bearer"}),
    ],
)
def test_version_combines_with_every_existing_flag(argv: list[str], expected: dict[str, object]) -> None:
    args = parse_args(argv)

    assert args.action == "version"
    for name, value in expected.items():
        assert getattr(args, name) == value, name


def test_an_unknown_action_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["bogus"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_auth_none_is_still_refused_with_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["--auth", "none", "version"])

    assert exc.value.code != 0
    assert "security.allow_no_auth" in capsys.readouterr().err


def test_version_has_one_spelling(capsys: pytest.CaptureFixture[str]) -> None:
    # D-25: `version` is the only spelling; --version is gone and --v, --ve and --ver mean --verbose again.
    with pytest.raises(SystemExit) as exc:
        parse_args(["--version"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --version" in capsys.readouterr().err

    for spelling in ("--v", "--ve", "--ver", "--verb", "--verbose"):
        assert parse_args([spelling]).verbose == 1, spelling
    assert parse_args(["--verbose", "--verbose"]).verbose == 2
    assert parse_args(["-vvv"]).verbose == 3
    assert parse_args(["version"]).action == "version"

    with pytest.raises(SystemExit) as exc:
        parse_args(["--vers"])  # no option starts with it any more
    assert exc.value.code == 2
    assert "unrecognized arguments: --vers" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("version", "wording"),
    [
        ("2.0.0a1", "an Alpha version"),
        ("2.0.0b1", "a Beta version"),
        ("2.0.0rc1", "a release candidate"),
    ],
)
def test_prerelease_notice_names_the_stage(version: str, wording: str) -> None:
    notice = version_command.prerelease_notice(version)

    assert isinstance(notice, Text)
    assert notice.plain.startswith("[!] You are currently using ")
    assert wording in notice.plain


def test_a_final_release_has_no_prerelease_notice() -> None:
    assert version_command.prerelease_notice("2.0.0") is None
    assert version_command.prerelease_notice("not a version") is None


@pytest.mark.usefixtures("restore_logging")
def test_version_starts_nothing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _forbid(monkeypatch, selfcheck, "assert_patch_points")
    _forbid(monkeypatch, main_module, "configure_stderr_logging")
    _forbid(monkeypatch, main_module, "load_operator_config")
    _forbid(monkeypatch, main_module, "check_readiness")
    _forbid(monkeypatch, main_module, "build_server")
    _forbid(monkeypatch, main_module, "run_http")
    _forbid(monkeypatch, main_module, "run_stdio")
    _forbid(monkeypatch, secret_store, "load_bearer_secret")
    _forbid(monkeypatch, versions, "fetch_latest_version")

    assert main(["version", "-t", "stdio"]) == 0
    action = capsys.readouterr()

    assert action.out.startswith(f"[*] Exegol MCP is currently in version v{__version__}\n")
    assert "Traceback" not in action.err


def _plant(directory: Path, name: str, content: str, mode: int = 0o644) -> Path:
    path = directory / name
    path.write_text(content)
    path.chmod(mode)
    return path


def _config_dir(home: Path) -> Path:
    """exegol-mcp's config dir under the fake home, created loosely (0755) so a hardening chmod would show."""
    directory = home / secret_store.CONFIG_DIR_NAME
    directory.mkdir()
    directory.chmod(0o755)
    return directory


def _fresh_newer_cache() -> str:
    return json.dumps({"checked_at": time.time(), "latest_version": "9.9.9"})


def _snapshot(*paths: Path) -> dict[Path, tuple[bytes | None, int]]:
    return {
        path: (path.read_bytes() if path.is_file() else None, stat.S_IMODE(path.stat().st_mode)) for path in paths
    }


@pytest.mark.usefixtures("restore_logging")
def test_version_reads_nothing_it_could_change(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _config_dir(home)
    sentinel = "SENTINEL-bearer-secret-value"
    files = [
        _plant(directory, "config.yml", "execution:\n  timeout_default_s: 30\n"),
        _plant(directory, "bearer.secret", sentinel + "\n"),
        _plant(directory, versions.UPDATE_CACHE_FILE, _fresh_newer_cache()),
    ]
    before = _snapshot(directory, *files)
    _forbid(monkeypatch, secret_store, "load_bearer_secret")
    _forbid(monkeypatch, secret_store, "read_user_config_file")
    _forbid(monkeypatch, secret_store, "write_user_config_file")
    _forbid(monkeypatch, secret_store, "_move_from_platform_dir")
    _forbid(monkeypatch, versions, "fetch_latest_version")

    assert main(["version"]) == 0

    captured = capsys.readouterr()
    assert "exegol-mcp 9.9.9 is available" in captured.out
    assert sentinel not in captured.out + captured.err
    assert _snapshot(directory, *files) == before
    assert sorted(path.name for path in directory.iterdir()) == sorted(path.name for path in files)


@pytest.mark.usefixtures("restore_logging")
def test_version_leaves_the_old_platform_dir_alone(
    home: Path, isolated_platform_config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = isolated_platform_config_dir
    old.mkdir()
    old_files = [
        _plant(old, "config.yml", "execution:\n  timeout_default_s: 30\n", 0o600),
        _plant(old, "bearer.secret", "old-secret\n", 0o600),
    ]
    before = _snapshot(old, *old_files)

    assert main(["version"]) == 0

    capsys.readouterr()
    assert _snapshot(old, *old_files) == before
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()


@pytest.mark.usefixtures("restore_logging")
def test_version_reports_an_invalid_config_and_exits_0(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = _config_dir(home)
    config = _plant(directory, "config.yml", "execution:\n  timeout_default_s: 0\n")
    cache = _plant(directory, versions.UPDATE_CACHE_FILE, _fresh_newer_cache())
    before = _snapshot(directory, config, cache)

    assert main(["version"]) == 0

    out = capsys.readouterr().out
    problem = [line for line in out.splitlines() if line.startswith("[!] ") and "timeout_default_s" in line]
    assert len(problem) == 1, out
    assert "is available" not in out
    assert out.startswith(f"[*] Exegol MCP is currently in version v{__version__}\n")
    assert _snapshot(directory, config, cache) == before


@pytest.mark.usefixtures("restore_logging")
@pytest.mark.parametrize(
    ("argv", "config_text"),
    [
        (["version", "--no-update-check"], None),
        (["version"], "update_check: false\n"),
    ],
    ids=["flag", "config-key"],
)
def test_version_honours_the_update_check_opt_outs(
    home: Path, capsys: pytest.CaptureFixture[str], argv: list[str], config_text: str | None
) -> None:
    directory = _config_dir(home)
    _plant(directory, versions.UPDATE_CACHE_FILE, _fresh_newer_cache())
    if config_text is not None:
        _plant(directory, "config.yml", config_text)

    assert main(argv) == 0

    assert "is available" not in capsys.readouterr().out


@pytest.mark.usefixtures("restore_logging")
def test_a_cached_notice_for_the_running_release_is_not_shown(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = _config_dir(home)
    _plant(directory, versions.UPDATE_CACHE_FILE, json.dumps({"checked_at": time.time(), "latest_version": "1.0.0"}))

    assert main(["version"]) == 0

    assert "is available" not in capsys.readouterr().out


def test_version_with_a_bad_exegol_home_exits_0_with_a_warning(tmp_path: Path) -> None:
    home = tmp_path / "scratch-home"
    home.mkdir()

    result = _run_cli(home, "version", EXEGOL_HOME="relative/dir")

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"[*] Exegol MCP is currently in version v{__version__}"
    warning = [line for line in lines[1:] if line.startswith("[!] ") and "stopped while loading" in line]
    assert len(warning) == 1, result.stdout
    assert "EXEGOL_HOME" in warning[0]
    assert "SystemExit" not in result.stdout
    assert "EXEGOL_HOME" in result.stderr
    assert "Traceback" not in result.stderr
    assert list(home.rglob("*")) == []


def test_version_vvv_without_the_sdk_prints_the_exegol_lines_as_unknown(tmp_path: Path) -> None:
    # D-18: the Exegol-side lines name the missing library instead of failing; the version report still exits 0.
    home = tmp_path / "scratch-home"
    home.mkdir()

    result = _run_cli(home, "version", "-vvv", "--no-update-check", EXEGOL_HOME="relative/dir")

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    for label in ("Exegol config directory", "EULA", "Docker", "Docker desktop"):
        assert f"[D] {label}: unknown (Exegol library not loaded)" in lines
    assert "Traceback" not in result.stderr
    assert list(home.rglob("*")) == []


@pytest.mark.usefixtures("restore_logging")
def test_version_turns_the_sdk_critical_into_a_raise(capsys: pytest.CaptureFixture[str]) -> None:
    from exegol.utils.ExeLog import logger

    assert main(["version", "--no-update-check"]) == 0
    capsys.readouterr()

    with pytest.raises(RuntimeError):
        logger.critical("x")
    capsys.readouterr()


@pytest.mark.parametrize("verbosity", [[], ["-vvv"]])
def test_version_imports_only_the_light_sdk_modules(tmp_path: Path, verbosity: list[str]) -> None:
    home = tmp_path / "scratch-home"
    home.mkdir()
    script = (
        "import json, sys\n"
        "from exegol_mcp.main import main\n"
        f"status = main(['version', '--no-update-check', *{verbosity!r}])\n"
        "loaded = sorted(m for m in sys.modules if m == 'exegol' or m.startswith('exegol.'))\n"
        "print(json.dumps([status, loaded]))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=server_env(home, DOCKER_HOST=DEAD_DOCKER),
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    status, loaded = json.loads(result.stdout.splitlines()[-1])
    assert status == 0
    assert loaded
    assert set(loaded) <= LIGHT_SDK_MODULES
    assert list(home.rglob("*")) == []


def test_peek_user_config_file_never_adopts_hardens_or_creates(
    home: Path, isolated_platform_config_dir: Path
) -> None:
    old = isolated_platform_config_dir
    old.mkdir()
    _plant(old, "config.yml", "old\n", 0o600)

    # No config dir: nothing to read, nothing created, nothing adopted from the old platform dir.
    assert secret_store.peek_user_config_file("config.yml") is None
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()
    assert (old / "config.yml").read_text() == "old\n"

    directory = _config_dir(home)
    planted = _plant(directory, "config.yml", "update_check: false\n")

    assert secret_store.peek_user_config_file("config.yml") == b"update_check: false\n"
    assert stat.S_IMODE(planted.stat().st_mode) == 0o644
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert secret_store.peek_user_config_file("update-check.json") is None
    assert not (directory / "update-check.json").exists()
    assert (old / "config.yml").read_text() == "old\n"
    with pytest.raises(ValueError):
        secret_store.peek_user_config_file("../config.yml")


def test_peek_refuses_a_symbolic_link(home: Path, tmp_path: Path) -> None:
    directory = _config_dir(home)
    target = tmp_path / "elsewhere.yml"
    target.write_text("update_check: false\n")
    os.symlink(target, directory / "config.yml")

    with pytest.raises(OSError):
        secret_store.peek_user_config_file("config.yml")


def test_a_read_only_update_checker_never_writes_the_cache_or_logs(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    checker = versions.UpdateChecker(
        enabled=True, current="2.0.0", fetch=lambda _timeout: "2.1.0", read_only=True
    )

    with caplog.at_level(logging.INFO):
        notice = checker.check()

    assert notice is not None and "2.1.0 is available" in notice
    assert caplog.records == []
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()


def test_cached_notice_is_none_when_disabled_or_uncached(home: Path) -> None:
    assert versions.UpdateChecker(enabled=False, read_only=True).cached_notice() is None
    assert versions.UpdateChecker(enabled=True, read_only=True).cached_notice() is None
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()


# ---- the exegol-mcp side of the -vvv section (plan 05.2-04, task 2) ----


@pytest.fixture
def fixed_facts(monkeypatch: pytest.MonkeyPatch) -> SdkFacts:
    """Replace the Exegol probes with fixed facts: no in-process test reads the real ~/.exegol or Docker."""
    facts = SdkFacts(
        exegol_config_dir="/exegol",
        exegol_config_file="/exegol/config.yml",
        datastore_file="/exegol/.datastore",
        eula="accepted",
        host_os="TestOS 1",
        arch="arm64",
        docker=DockerFacts("reachable (Docker 29.8.2)", "Docker Desktop (linux)", "On"),
    )
    monkeypatch.setattr(diagnostics, "collect_sdk_facts", lambda: facts)
    return facts


def _vvv(capsys: pytest.CaptureFixture[str], *extra: str) -> list[str]:
    assert main(["version", "-vvv", "--no-update-check", *extra]) == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    return captured.out.splitlines()


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_version_vvv_reports_the_exegol_mcp_state(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = home / secret_store.CONFIG_DIR_NAME

    lines = _vvv(capsys)

    assert f"[D] Exegol MCP config directory: {directory}" in lines
    assert f"[D] Config file: {directory / 'config.yml'} (absent, built-in defaults)" in lines
    assert f"[D] Bearer secret storage: file ({directory / 'bearer.secret'})" in lines
    assert not any(line.startswith("[D] Waiting to move") for line in lines)
    assert "[D] EULA: accepted" in lines
    assert "[D] Docker desktop: On" in lines
    assert any(line.startswith("[D] Installation: ") for line in lines)
    assert f"[D] Python: {platform.python_version()} ({sys.executable})" in lines
    assert f"[D] MCP SDK: {importlib.metadata.version('mcp')}" in lines
    assert not directory.exists()

    config_dir = _config_dir(home)
    _plant(config_dir, "config.yml", "execution:\n  timeout_default_s: 30\n")
    assert f"[D] Config file: {config_dir / 'config.yml'} (loaded)" in _vvv(capsys)

    _plant(config_dir, "config.yml", "execution:\n  timeout_default_s: 0\n")
    invalid = [line for line in _vvv(capsys) if line.startswith("[D] Config file: ")]
    assert len(invalid) == 1
    assert f"{config_dir / 'config.yml'} (invalid: Invalid config.yml: " in invalid[0]
    assert "timeout_default_s" in invalid[0]


class _FakeKeyring:
    """A keyring backend that must only ever be asked for its name."""


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_version_vvv_names_the_keyring_backend_without_reading_it(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("`version` must not read or write the keyring item")

    fake = types.SimpleNamespace(get_keyring=_FakeKeyring, get_password=forbidden, set_password=forbidden)
    monkeypatch.setattr(secret_store, "keyring", fake)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    lines = _vvv(capsys)

    assert "[D] Bearer secret storage: keyring (_FakeKeyring)" in lines


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_version_vvv_never_prints_the_secret(
    home: Path, isolated_platform_config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "SENTINEL-bearer-secret-value-9f3a"
    directory = _config_dir(home)
    _plant(directory, "bearer.secret", sentinel + "\n", 0o600)
    isolated_platform_config_dir.mkdir()
    _plant(isolated_platform_config_dir, "bearer.secret", sentinel + "-old\n", 0o600)

    assert main(["version", "-vvv", "--no-update-check"]) == 0

    captured = capsys.readouterr()
    assert "SENTINEL" not in captured.out + captured.err
    assert (directory / "bearer.secret").read_text() == sentinel + "\n"


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_version_vvv_reports_files_waiting_to_move_without_moving_them(
    home: Path, isolated_platform_config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = isolated_platform_config_dir
    old.mkdir()
    old_files = [
        _plant(old, "config.yml", "execution:\n  timeout_default_s: 30\n", 0o600),
        _plant(old, "bearer.secret", "old-secret\n", 0o600),
    ]
    before = _snapshot(old, *old_files)

    lines = _vvv(capsys)

    waiting = [line for line in lines if line.startswith("[D] Waiting to move into the config directory: ")]
    assert len(waiting) == 1
    assert all(str(path) in waiting[0] for path in old_files)
    assert "update-check.json" not in waiting[0]
    assert _snapshot(old, *old_files) == before
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()


@pytest.mark.parametrize(
    ("layout", "direct_url", "expected"),
    [
        ("pipx/venvs/exegol-mcp/lib/python3.12/site-packages/exegol_mcp", None, "pipx"),
        ("uv/tools/exegol-mcp/lib/python3.12/site-packages/exegol_mcp", None, "uv tool"),
        ("venv/lib/python3.12/site-packages/exegol_mcp", None, "pip"),
        (
            "src/exegol_mcp",
            '{"url": "file:///src/exegol-mcp", "dir_info": {"editable": true}}',
            "editable (/src/exegol-mcp)",
        ),
        ("src/exegol_mcp", '{"url": "file:///src/exegol-mcp", "dir_info": {}}', "unknown"),
        ("src/exegol_mcp", "not json", "unknown"),
        ("checkout/exegol_mcp", None, "source checkout"),
        ("elsewhere/exegol_mcp", None, "unknown"),
    ],
)
def test_install_method_follows_the_wrapper_rules(
    tmp_path: Path, layout: str, direct_url: str | None, expected: str
) -> None:
    package_dir = tmp_path / layout
    package_dir.mkdir(parents=True)
    if layout.startswith("checkout/"):
        (tmp_path / "checkout" / ".git").mkdir()

    assert version_command.install_method(package_dir, direct_url) == expected


# ---- the update check at -vvv (plan 05.2-04, task 3) ----


def _fake_fetch(monkeypatch: pytest.MonkeyPatch, answer: str | None | Exception) -> list[float]:
    """Replace the PyPI fetch; the returned list records each call's timeout (an exception is raised, not returned)."""
    calls: list[float] = []

    def fetch(timeout: float = 0.0) -> str | None:
        calls.append(timeout)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(versions, "fetch_latest_version", fetch)
    return calls


def _update_line(lines: list[str]) -> str:
    found = [line for line in lines if line.startswith("[D] Update check: ")]
    assert len(found) == 1, lines
    return found[0]


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_version_vvv_runs_the_update_check_read_only(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_fetch(monkeypatch, "9.9.9")

    assert main(["version", "-vvv"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert calls == [versions.UPDATE_CHECK_TIMEOUT_SECONDS]
    assert _update_line(lines) == "[D] Update check: update available: 9.9.9"
    assert sum(line.startswith("[*] exegol-mcp 9.9.9 is available") for line in lines) == 1
    assert lines[-1] == BETA_NOTICE
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()  # no update-check.json, no directory


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (__version__, f"no newer release on PyPI (latest stable: {__version__})"),
        (urllib.error.URLError("offline"), "failed (URLError)"),
        (None, "no answer"),
    ],
)
def test_version_vvv_reports_up_to_date_and_failed_checks(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str | None | Exception,
    expected: str,
) -> None:
    _fake_fetch(monkeypatch, answer)

    assert main(["version", "-vvv"]) == 0

    captured = capsys.readouterr()
    assert _update_line(captured.out.splitlines()) == f"[D] Update check: {expected}"
    assert "Traceback" not in captured.err


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_the_update_line_names_the_latest_stable_release(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # D-28: PyPI answers with the latest stable release; a beta running ahead of it has no newer release to move to.
    monkeypatch.setattr(version_command, "__version__", "2.0.0b1")
    _fake_fetch(monkeypatch, "1.0.2")

    assert main(["version", "-vvv"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert _update_line(lines) == "[D] Update check: no newer release on PyPI (latest stable: 1.0.2)"


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_a_slow_update_check_never_delays_version_past_its_timeout(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    release = threading.Event()

    def blocked(_timeout: float = 0.0) -> str | None:
        release.wait(30)
        return "9.9.9"

    monkeypatch.setattr(versions, "fetch_latest_version", blocked)
    monkeypatch.setattr(versions, "UPDATE_CHECK_TIMEOUT_SECONDS", 0.3)
    try:
        started = time.monotonic()
        assert main(["version", "-vvv"]) == 0
        elapsed = time.monotonic() - started
    finally:
        release.set()  # no thread outlives the test

    lines = capsys.readouterr().out.splitlines()
    assert elapsed < 2
    assert _update_line(lines) == "[D] Update check: no answer within 0.3 s"
    assert not any("is available" in line for line in lines)


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
@pytest.mark.parametrize(
    ("flag", "config", "expected"),
    [
        (True, None, "disabled (--no-update-check)"),
        (False, "update_check: false\n", "disabled (update_check: false in config.yml)"),
        (False, "execution:\n  timeout_default_s: 0\n", "not run (config.yml did not load)"),
    ],
)
def test_version_vvv_honours_the_update_check_opt_outs(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: bool,
    config: str | None,
    expected: str,
) -> None:
    calls = _fake_fetch(monkeypatch, AssertionError("an opted-out `version` must not call PyPI"))
    if config is not None:
        _plant(_config_dir(home), "config.yml", config)

    assert main(["version", "-vvv", *(["--no-update-check"] if flag else [])]) == 0

    assert calls == []
    assert _update_line(capsys.readouterr().out.splitlines()) == f"[D] Update check: {expected}"


@pytest.mark.usefixtures("restore_logging")
def test_version_without_vvv_never_fetches(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_fetch(monkeypatch, "9.9.9")

    assert main(["version"]) == 0
    assert main(["version", "-vv"]) == 0

    out = capsys.readouterr().out
    assert calls == []
    assert "[D] " not in out


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_version_vvv_shows_a_cached_release_once_without_fetching(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _config_dir(home)
    cache = _plant(directory, versions.UPDATE_CACHE_FILE, _fresh_newer_cache())
    before = _snapshot(directory, cache)
    calls = _fake_fetch(monkeypatch, AssertionError("a fresh cache must be enough"))

    assert main(["version", "-vvv"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert calls == []
    assert _update_line(lines) == "[D] Update check: update available: 9.9.9"
    assert sum("exegol-mcp 9.9.9 is available" in line for line in lines) == 1
    assert _snapshot(directory, cache) == before


# ---- a drifted SDK helper never breaks `version` (D-09, D-18; review WR-01) ----


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
@pytest.mark.parametrize("argv", [["version"], ["version", "-vvv", "--no-update-check"]], ids=["base", "vvv"])
def test_version_survives_a_moved_sdk_sudo_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    from exegol.config.SudoInfo import SudoInfo

    # Not the `home` fixture: it replaces the very lookup that drifts. The lookup fails before any file is touched,
    # and HOME is a scratch dir anyway.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delattr(SudoInfo, "getUserHome")

    assert main(argv) == 0

    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    assert captured.out.startswith(f"[*] Exegol MCP is currently in version v{__version__}\n")
    assert list(tmp_path.rglob("*")) == []
    if len(argv) > 1:
        assert "[D] Config file: unknown (AttributeError)" in captured.out.splitlines()


@pytest.mark.usefixtures("restore_logging")
def test_version_survives_a_diagnostics_module_that_cannot_be_imported(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import exegol_mcp.adapter as adapter_package

    # `from exegol_mcp.adapter import diagnostics` finds the attribute first, so remove it and block the import.
    monkeypatch.delattr(adapter_package, "diagnostics")
    monkeypatch.setitem(sys.modules, "exegol_mcp.adapter.diagnostics", None)

    assert main(["version", "-vvv", "--no-update-check"]) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert "Traceback" not in captured.out + captured.err
    for label in version_command._EXEGOL_LABELS:
        assert f"[D] {label}: unknown (ModuleNotFoundError)" in lines
    assert any(line.startswith("[D] Python: ") for line in lines)  # the rest of the section still prints


# ---- a config.yml still waiting in the old platform dir is honoured, read-only (D-10; review WR-03) ----


def test_peek_pending_adoption_reads_the_old_file_and_changes_nothing(
    home: Path, isolated_platform_config_dir: Path
) -> None:
    old = isolated_platform_config_dir
    old.mkdir()
    old.chmod(0o755)
    config = _plant(old, "config.yml", "update_check: false\n", 0o644)
    before = _snapshot(old, config)

    assert secret_store.peek_pending_adoption("config.yml") == (b"update_check: false\n", config)

    assert _snapshot(old, config) == before  # no chmod, no move
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()
    assert secret_store.peek_pending_adoption("update-check.json") is None  # not there
    assert secret_store.peek_pending_adoption("notes.txt") is None  # not an adopted name
    with pytest.raises(ValueError):
        secret_store.peek_pending_adoption("../config.yml")

    _plant(_config_dir(home), "config.yml", "execution:\n  timeout_default_s: 30\n")
    assert secret_store.peek_pending_adoption("config.yml") is None  # the new dir wins, as for a move


def test_peek_pending_adoption_refuses_a_symbolic_link(home: Path, isolated_platform_config_dir: Path) -> None:
    old = isolated_platform_config_dir
    old.mkdir()
    target = old.parent / "elsewhere.yml"
    target.write_text("update_check: false\n")
    os.symlink(target, old / "config.yml")

    with pytest.raises(OSError):
        secret_store.peek_pending_adoption("config.yml")


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
@pytest.mark.parametrize("argv", [["version"], ["version", "-vvv"]], ids=["base", "vvv"])
def test_version_honours_update_check_false_in_a_config_waiting_to_be_moved(
    home: Path,
    isolated_platform_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    old = isolated_platform_config_dir
    old.mkdir()
    old.chmod(0o755)
    old_files = [
        _plant(old, "config.yml", "update_check: false\n", 0o644),
        _plant(old, "bearer.secret", "old-secret\n", 0o600),
    ]
    before = _snapshot(old, *old_files)
    calls = _fake_fetch(monkeypatch, AssertionError("an opted-out `version` must not call PyPI"))
    _forbid(monkeypatch, secret_store, "_move_from_platform_dir")

    assert main(argv) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert calls == []
    assert "Traceback" not in captured.err
    assert _snapshot(old, *old_files) == before  # nothing moved, nothing chmod-ed
    assert sorted(path.name for path in old.iterdir()) == ["bearer.secret", "config.yml"]
    assert not (home / secret_store.CONFIG_DIR_NAME).exists()  # nothing created
    if "-vvv" in argv:
        assert _update_line(lines) == "[D] Update check: disabled (update_check: false in config.yml)"
        config_line = next(line for line in lines if line.startswith("[D] Config file: "))
        assert f"loaded from {old / 'config.yml'}" in config_line
        assert "built-in defaults" not in config_line


@pytest.mark.usefixtures("restore_logging")
def test_version_reports_an_invalid_config_waiting_to_be_moved(
    home: Path, isolated_platform_config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = isolated_platform_config_dir
    old.mkdir()
    config = _plant(old, "config.yml", "execution:\n  timeout_default_s: 0\n", 0o600)
    before = _snapshot(old, config)

    assert main(["version"]) == 0

    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("[!] ") and "timeout_default_s" in line]
    assert _snapshot(old, config) == before


# ---- what `version` shows: the wrapper only when it differs, every version in one colour (D-26) ----


def _library_version() -> str:
    return importlib.metadata.version("exegol")


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
@pytest.mark.parametrize(
    ("verbosity", "wrapper", "expected"),
    [
        ([], "library", None),
        ([], "6.0.1", "[*] Exegol wrapper: 6.0.1 (differs from the bundled library: information only)"),
        ([], None, "[*] Exegol wrapper: not detected"),
        (["-vvv", "--no-update-check"], "library", "library"),
    ],
    ids=["equal-hidden", "different", "missing", "equal-at-vvv"],
)
def test_the_wrapper_line_is_shown_only_when_it_differs(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verbosity: list[str],
    wrapper: str | None,
    expected: str | None,
) -> None:
    detected = _library_version() if wrapper == "library" else wrapper
    monkeypatch.setattr(version_command, "detect_wrapper_version", lambda: detected)
    if expected == "library":
        expected = f"[*] Exegol wrapper: {_library_version()}"

    assert main(["version", *verbosity]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert f"[*] Exegol library: {_library_version()}" in lines  # the library is always printed
    wrapper_lines = [line for line in lines if line.startswith("[*] Exegol wrapper: ")]
    assert wrapper_lines == ([] if expected is None else [expected])


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_the_wrapper_line_is_hidden_when_the_library_is_unknown_and_shown_with_one(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # compatibility is "unknown" when the library version is missing: nothing to match, so the wrapper is information.
    monkeypatch.setattr(version_command, "library_version", lambda: None)
    monkeypatch.setattr(version_command, "detect_wrapper_version", lambda: "6.0.1")

    assert main(["version"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "[*] Exegol library: unknown" in lines
    assert "[*] Exegol wrapper: 6.0.1" in lines


def _sgr_before(lines: list[str], label: str, value: str) -> str:
    """The escape sequence directly in front of a version value on the line carrying the label."""
    found = [line for line in lines if label in line]
    assert found, (label, lines)
    match = re.search(r"((?:\x1b\[[0-9;]*m)+)" + re.escape(value), found[0])
    assert match is not None, (label, value, found[0])
    return match.group(1)


@pytest.mark.usefixtures("restore_logging", "fixed_facts")
def test_every_version_value_has_the_version_colour(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(version_command, "detect_wrapper_version", lambda: "6.0.1")
    _fake_fetch(monkeypatch, "9.9.9")

    assert main(["version", "-vvv"]) == 0

    lines = capsys.readouterr().out.splitlines()
    sequences = {
        "headline": _sgr_before(lines, "is currently in version", f"v{__version__}"),
        "library": _sgr_before(lines, "Exegol library:", _library_version()),
        "wrapper": _sgr_before(lines, "Exegol wrapper:", "6.0.1"),
        "mcp sdk": _sgr_before(lines, "MCP SDK:", importlib.metadata.version("mcp")),
        "python": _sgr_before(lines, "Python:", platform.python_version()),
        "pypi": _sgr_before(lines, "Update check:", "9.9.9"),
    }
    assert len(set(sequences.values())) == 1, sequences
    assert next(iter(sequences.values())) == "\x1b[34m", sequences  # rich's SGR for blue
