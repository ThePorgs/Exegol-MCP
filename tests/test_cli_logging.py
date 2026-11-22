"""CLI flag parity with v1 (COMP-01) and stderr-only logging (SEC-08). Docker-free."""

import io
import logging
import sys
from collections.abc import Iterator

import pytest

from exegol_mcp.bootstrap import logging_setup
from exegol_mcp.bootstrap.cli import parse_args
from exegol_mcp.bootstrap.config import resolve
from exegol_mcp.bootstrap.logging_setup import ExeLogHandler, configure_stderr_logging
from exegol_mcp.models.config import OperatorConfig


def test_mirrored_flag_defaults_are_none_but_resolve_to_the_stable_launch() -> None:
    # The transport/port/auth flags now default to None so config.yml can supply them (D-06); the stable
    # http/2187/bearer launch comes from resolve(), not from argparse defaults.
    args = parse_args([])
    assert args.type is None
    assert args.port is None
    assert args.auth is None
    assert args.print_config is False
    assert args.verbose == 0
    assert args.accept_eula is False
    assert args.no_update_check is False

    resolved = resolve(args, OperatorConfig())
    assert (resolved.transport, resolved.port, resolved.auth) == ("http", 2187, "bearer")


def test_short_v1_flags_and_new_flags_parse_together() -> None:
    args = parse_args(
        ["-t", "stdio", "-p", "8080", "--auth", "bearer", "-pc", "-vvv", "--accept-eula", "--no-update-check"]
    )
    assert args.type == "stdio"
    assert args.port == 8080
    assert args.auth == "bearer"
    assert args.print_config is True
    assert args.verbose == 3
    assert args.accept_eula is True
    assert args.no_update_check is True


def test_cli_auth_none_is_rejected_with_guidance(capsys: pytest.CaptureFixture[str]) -> None:
    # D-12/SEC-04: `--auth none` is removed; it exits non-zero pointing at the config key, not a bare argparse error.
    with pytest.raises(SystemExit) as exc:
        parse_args(["--auth", "none"])
    assert exc.value.code != 0
    assert "security.allow_no_auth" in capsys.readouterr().err


def test_long_v1_flags_keep_their_names() -> None:
    args = parse_args(["--type", "http", "--port", "2188", "--auth", "bearer", "--print-config", "--verbose"])
    assert (args.type, args.port, args.auth, args.print_config, args.verbose) == ("http", 2188, "bearer", True, 1)


@pytest.mark.parametrize(
    "argv",
    [["--type", "sse"], ["--auth", "oauth"], ["--port", "not-a-port"]],
)
def test_v1_choices_still_reject_unknown_values(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.fixture
def restore_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


@pytest.mark.usefixtures("restore_logging")
def test_every_root_handler_writes_to_stderr() -> None:
    configure_stderr_logging(0)

    handlers = logging.getLogger().handlers
    assert handlers
    for handler in handlers:
        assert isinstance(handler, ExeLogHandler), handler
        assert handler.console.stderr is True, handler
        assert handler.console.file is sys.stderr, handler


@pytest.mark.usefixtures("restore_logging")
def test_handler_follows_a_replaced_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_stderr_logging(0)
    replacement = io.StringIO()
    monkeypatch.setattr(sys, "stderr", replacement)

    logging.getLogger("exegol_mcp.test").info("diagnostic line")

    assert "[*] diagnostic line" in replacement.getvalue()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "diagnostic line" not in captured.err


@pytest.mark.usefixtures("restore_logging")
def test_exegol_sdk_output_goes_to_stderr_in_the_wrapper_style(capsys: pytest.CaptureFixture[str]) -> None:
    from exegol.utils.ExeLog import ExeLog, logger

    configure_stderr_logging(0)
    assert ExeLog.console.stderr is True

    logger.warning("sdk warning")
    ExeLog.console.print("sdk console line")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[!] sdk warning" in captured.err
    assert "[bold" not in captured.err
    assert "sdk console line" in captured.err


@pytest.mark.usefixtures("restore_logging")
@pytest.mark.parametrize(
    ("verbose", "root_level", "package_level"),
    [
        (0, logging.WARNING, logging.INFO),
        (1, logging.INFO, 15),
        (2, logging.INFO, 13),
        (3, logging.DEBUG, logging.DEBUG),
    ],
)
def test_verbosity_sets_levels(verbose: int, root_level: int, package_level: int) -> None:
    from exegol.utils.ExeLog import logger

    configure_stderr_logging(verbose)
    configure_stderr_logging(verbose)  # idempotent: still exactly one handler

    root = logging.getLogger()
    assert root.level == root_level
    assert len(root.handlers) == 1
    assert logging.getLogger("exegol_mcp").level == package_level
    assert logger.level == package_level  # the SDK logger follows the wrapper's mapping too


@pytest.mark.usefixtures("restore_logging")
@pytest.mark.parametrize(("verbose", "shown"), [(1, True), (0, False)])
def test_the_bearer_secret_location_is_a_verbose_line(
    verbose: int, shown: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from exegol_mcp import main as main_module
    from exegol_mcp.bootstrap import secret_store

    monkeypatch.setattr(
        secret_store, "load_bearer_secret", lambda: ("SENTINEL-SECRET-1x", False, "/x/bearer.secret")
    )
    configure_stderr_logging(verbose)

    secret, created = main_module._load_secret(parse_args(["--type", "http", "--auth", "bearer"]))

    assert (secret, created) == ("SENTINEL-SECRET-1x", False)
    captured = capsys.readouterr()
    assert ("[V] Bearer secret stored at '/x/bearer.secret'" in captured.err) is shown
    assert "SENTINEL-SECRET-1x" not in captured.err
    assert captured.out == ""


@pytest.mark.usefixtures("restore_logging")
@pytest.mark.parametrize(
    ("verbose", "shown", "present", "wording"),
    [(1, True, False, "No operator config at '"), (1, True, True, "Loaded the operator config from '"), (0, False, True, "")],
)
def test_the_operator_config_file_is_a_verbose_line(
    verbose: int, shown: bool, present: bool, wording: str, home, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from exegol_mcp import main as main_module
    from exegol_mcp.adapter import sdk_headless, selfcheck
    from exegol_mcp.bootstrap import secret_store

    def _broken_patch() -> None:
        raise selfcheck._broken("exegol.utils.FsUtils.secure_remove", "is missing")

    def _no_server(*args, **kwargs):
        raise AssertionError("a missing patch point must stop startup before any server is built")

    monkeypatch.setattr(main_module, "build_server", _no_server)
    monkeypatch.setattr(secret_store, "read_user_config_file", lambda name: b"")
    monkeypatch.setattr(sdk_headless, "patch_exegol_sdk", _broken_patch)
    if present:
        config_dir = secret_store.user_config_dir()
        config_dir.mkdir(parents=True)
        (config_dir / "config.yml").write_text("")
    argv = ["--type", "stdio", "--accept-eula", "--no-update-check", *(["-v"] * verbose)]

    status = main_module.main(argv)

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    config_lines = [line for line in captured.err.splitlines() if "operator config" in line]
    assert bool(config_lines) is shown, captured.err
    if shown:
        assert config_lines[0].startswith(f"[V] {wording}")
        assert "config.yml" in config_lines[0]
