"""Docker-free tests of the wrapper-style diagnostics rendering (D-01..D-04, D-20, SEC-08).

Every record, whether from exegol-mcp, a third-party library or the Exegol SDK, renders on the standard error stream
as one line "<prefix> <text>" through one shared rich console. Markup is parsed only for the SDK's own logger.
"""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from exegol.utils.ExeLog import ExeLog, logger

from exegol_mcp.bootstrap import logging_setup
from exegol_mcp.bootstrap.logging_setup import (
    ExeLogHandler,
    configure_stderr_logging,
    route_logging_to_stderr,
    stderr_console,
)


@pytest.fixture(autouse=True)
def restore_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Save and restore every piece of process-wide logging state these tests change, with a known environment."""
    root = logging.getLogger()
    package = logging.getLogger("exegol_mcp")
    sse = logging.getLogger("sse_starlette")
    saved = (list(root.handlers), root.level, package.level, sse.level, logger.level)
    monkeypatch.setattr(ExeLog, "console", ExeLog.console)
    monkeypatch.setattr(logging_setup, "_console", logging_setup._console)
    monkeypatch.setattr(logger, "_ExeLog__critical_method", logger._ExeLog__critical_method)
    for name in ("FORCE_COLOR", "TTY_COMPATIBLE", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "20")
    yield
    handlers, root_level, package_level, sse_level, sdk_level = saved
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(root_level)
    package.setLevel(package_level)
    sse.setLevel(sse_level)
    logger.setLevel(sdk_level)


def _plain_logger(name: str) -> logging.Logger:
    """A stdlib Logger that is not an ExeLog (the SDK import made getLogger() return ExeLog instances)."""
    plain = logging.Logger(name)  # noqa: LOG001 - deliberately not from getLogger(), whose class is ExeLog
    plain.parent = logging.getLogger("exegol_mcp")
    plain.setLevel(logging.DEBUG)
    return plain


def _stderr_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err.splitlines()


def test_package_and_sdk_records_render_in_the_wrapper_style(capsys: pytest.CaptureFixture[str]) -> None:
    route_logging_to_stderr(0)
    configure_stderr_logging(0)

    logging.getLogger("exegol_mcp.tracer").info("package line")
    logger.success("sdk line [green]ok[/green]")
    ExeLog.console.print("console line")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == ["[*] package line", "[+] sdk line ok", "console line"]
    assert "[bold" not in captured.err


@pytest.mark.parametrize(
    ("levelno", "prefix"),
    [
        (logging.INFO, "[*]"),
        (logging.WARNING, "[!]"),
        (logging.ERROR, "[-]"),
        (logging.CRITICAL, "[!]"),
        (25, "[+]"),
        (15, "[V]"),
        (13, "[A]"),
        (logging.DEBUG, "[D]"),
    ],
)
def test_every_level_gets_the_wrapper_prefix(
    capsys: pytest.CaptureFixture[str], levelno: int, prefix: str
) -> None:
    route_logging_to_stderr(3)

    _plain_logger("exegol_mcp.levels").log(levelno, "text")

    assert _stderr_lines(capsys) == [f"{prefix} text"]


def test_markup_in_non_sdk_records_prints_literally(capsys: pytest.CaptureFixture[str]) -> None:
    route_logging_to_stderr(0)
    configure_stderr_logging(0)
    text = "[red]x[/red] [/oops]"

    adapter_logger = logging.getLogger("exegol_mcp.adapter.fake")
    assert isinstance(adapter_logger, ExeLog)  # exegol-mcp's own adapter loggers are ExeLog instances
    adapter_logger.info(text)
    _plain_logger("exegol_mcp.plain").info(text)

    assert _stderr_lines(capsys) == [f"[*] {text}", f"[*] {text}"]


def test_broken_sdk_markup_falls_back_to_literal_text(capsys: pytest.CaptureFixture[str]) -> None:
    configure_stderr_logging(0)

    logger.warning("container said [/bin/zsh] boom")

    assert _stderr_lines(capsys) == ["[!] container said [/bin/zsh] boom"]


def test_a_long_record_is_one_unpadded_line_at_a_narrow_width(capsys: pytest.CaptureFixture[str]) -> None:
    configure_stderr_logging(0)
    message = ("word " * 60).strip()
    assert len(message) == 299

    logging.getLogger("exegol_mcp.long").warning(message)
    logger.info(message)

    lines = _stderr_lines(capsys)
    assert len(lines) == 2
    assert not any(line.endswith(" ") for line in lines)
    assert lines[0] == f"[!] {message}"


def test_colour_only_on_a_terminal_and_never_with_no_color(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    colour_sgr = re.compile(r"\x1b\[[0-9;]*(3[0-7]|38;)")

    route_logging_to_stderr(0)  # not a terminal, no override: no escape byte at all
    logging.getLogger("exegol_mcp.colour").info("plain")
    assert "\x1b" not in capsys.readouterr().err

    monkeypatch.setenv("FORCE_COLOR", "1")
    route_logging_to_stderr(0)  # a fresh console re-reads the environment
    logging.getLogger("exegol_mcp.colour").info("coloured")
    assert "\x1b[" in capsys.readouterr().err

    monkeypatch.setenv("NO_COLOR", "1")
    route_logging_to_stderr(0)
    logging.getLogger("exegol_mcp.colour").info("no colour")
    assert colour_sgr.search(capsys.readouterr().err) is None


def test_the_handler_and_the_sdk_share_one_stderr_console() -> None:
    configure_stderr_logging(0)

    handler = logging.getLogger().handlers[0]
    assert isinstance(handler, ExeLogHandler)
    assert handler.console is ExeLog.console is stderr_console()
    assert stderr_console().stderr is True
    assert stderr_console().soft_wrap is True


def test_the_sdk_logger_name_matches() -> None:
    assert logger.name == logging_setup._SDK_LOGGER_NAME


def test_sdk_interpolated_markup_is_the_sdks_and_stays_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    """D-20 pinned on both sides.

    A value the SDK itself interpolates into its own message (a network or branch name) follows the SDK: the line
    renders as the wrapper would show it, without raising, as one line. This is the accepted residual (T-05.2-21).
    The same text in an exegol-mcp record prints with its brackets literal.
    """
    configure_stderr_logging(0)

    logger.info("network [bold red]x[/bold red]")
    sdk_lines = _stderr_lines(capsys)
    assert len(sdk_lines) == 1
    assert sdk_lines[0].startswith("[*] network ")

    logging.getLogger("exegol_mcp.tracer").info("network [bold red]x[/bold red]")
    assert _stderr_lines(capsys) == ["[*] network [bold red]x[/bold red]"]


def test_level_constants_match_the_installed_sdk() -> None:
    # bootstrap cannot import the SDK, so logging_setup duplicates ExeLog's level numbers; a renumbering lands here.
    assert ExeLog.SUCCESS == logging_setup.SUCCESS
    assert ExeLog.VERBOSE == logging_setup.VERBOSE
    assert ExeLog.ADVANCED == logging_setup.ADVANCED


def test_critical_raises_after_configure_and_never_exits() -> None:
    logger.setCriticalMethod("exit")  # the SDK's default, restored by the fixture
    configure_stderr_logging(0)

    with pytest.raises(RuntimeError, match="boom"):
        logger.critical("boom")


def test_no_exegol_mcp_module_logs_at_critical_on_its_module_logger() -> None:
    # exegol-mcp's adapter module loggers can be ExeLog instances whose own critical method is still "exit"
    # (Pitfall 8): a critical record through one would kill the server.
    package = Path(logging_setup.__file__).resolve().parent.parent
    offenders = [
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if re.search(r"_log\.critical\(", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("verbose", "root_level", "package_level"),
    [
        (0, logging.WARNING, logging.INFO),
        (1, logging.INFO, 15),
        (2, logging.INFO, 13),
        (3, logging.DEBUG, logging.DEBUG),
        (5, logging.DEBUG, logging.DEBUG),
    ],
)
def test_verbosity_follows_the_wrapper_and_sse_starlette_stays_capped(
    verbose: int, root_level: int, package_level: int
) -> None:
    configure_stderr_logging(verbose)

    assert logging.getLogger().level == root_level
    assert logging.getLogger("exegol_mcp").level == package_level
    assert logger.level == package_level
    # G-03-1, SEC-02: sse_starlette logs every streamed tool result body at DEBUG, so it never goes below INFO.
    assert logging.getLogger("sse_starlette.sse").getEffectiveLevel() == max(root_level, logging.INFO)
