"""`exegol-mcp --help` in the Exegol wrapper's style and the hidden --auth (D-32, D-33, SEC-08, QUAL-01). Docker-free.

The tracer runs the real entry point in a subprocess with a scratch HOME; colour is decided by rich alone (D-04), so
the same help is plain off a terminal and coloured with FORCE_COLOR, and equal once the colour is removed.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
from rich.text import Text

from exegol_mcp.bootstrap import cli, logging_setup
from exegol_mcp.bootstrap.cli import DOCS_URL, build_parser, parse_args
from tests.fake_server import REPO_ROOT, server_env

DEAD_DOCKER = "unix:///nonexistent/docker.sock"
SGR = re.compile(r"\x1b\[[0-9;]*m")
# A colour SGR (foreground or background, 8/16/256/true colour); bold and italic are styles and survive NO_COLOR.
COLOUR_SGR = re.compile(r"\x1b\[(?:[0-9;]*;)?(?:3[0-7]|9[0-7]|4[0-7]|10[0-7]|38|48)[;m]")
GROUPS = ("Action:", "Output options:", "Server options:", "Client config options:", "Startup options:")
EXAMPLE_COMMANDS = (
    "exegol-mcp",
    "exegol-mcp --print-config",
    "exegol-mcp --type stdio --print-config",
    "exegol-mcp version -vvv",
)
EVERY_OPTION = (
    "-h", "--help", "-v", "--verbose", "-t", "--type", "-p", "--port", "-pc", "--print-config",
    "--accept-eula", "--no-update-check", "{version}",
)  # fmt: skip


def _run_cli(home: Path, *argv: str, unset: tuple[str, ...] = (), **env: str) -> subprocess.CompletedProcess[str]:
    full_env = server_env(home, DOCKER_HOST=DEAD_DOCKER, **env)
    for name in unset:
        full_env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-m", "exegol_mcp", *argv],
        env=full_env,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )


def _plain_help(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    # No colour variable may leak in from the developer's shell, and the width must not wrap argparse's own layout.
    return _run_cli(tmp_path, "--help", unset=("FORCE_COLOR", "NO_COLOR", "TTY_COMPATIBLE"), COLUMNS="200")


def test_help_reads_like_the_wrappers(tmp_path: Path) -> None:
    # The plan tracer: the real entry point, a scratch HOME, stdout only.
    result = _plain_help(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    out = result.stdout
    assert out.startswith("usage: exegol-mcp")
    assert "\x1b" not in out
    for tag in ("[blue]", "[bright_blue]", "[/", "[bold", "[underline", "[i]"):
        assert tag not in out, tag
    assert f"Exegol MCP documentation: {DOCS_URL}" in out
    assert DOCS_URL == "https://docs.exegol.com/mcp/"
    lines = out.splitlines()
    for group in GROUPS:
        assert group in lines, group
    assert "Examples:" in lines
    for command in EXAMPLE_COMMANDS:
        assert any(line.rstrip().endswith(command) for line in lines), command
    for option in EVERY_OPTION:
        assert option in out, option
    assert "--auth" not in out


def test_help_keeps_every_usage_token(tmp_path: Path) -> None:
    usage = _plain_help(tmp_path).stdout.split("\n\n", 1)[0]

    # No token of the usage line may be swallowed as a markup tag.
    for token in ("[-h]", "[-v]", "[-t {http,stdio}]", "[-p PORT]", "[-pc]", "[--accept-eula]",
                  "[--no-update-check]", "[{version}]"):  # fmt: skip
        assert token in usage, token
    assert "--auth" not in usage


def test_help_is_coloured_only_on_a_terminal(tmp_path: Path) -> None:
    plain = _plain_help(tmp_path)
    coloured = _run_cli(
        tmp_path, "--help", unset=("NO_COLOR", "TTY_COMPATIBLE"), COLUMNS="200", FORCE_COLOR="1", TERM="xterm-256color"
    )

    assert coloured.returncode == 0, coloured.stderr
    assert coloured.stderr == ""
    assert "\x1b[" in coloured.stdout
    assert "\x1b" not in plain.stdout
    assert SGR.sub("", coloured.stdout) == plain.stdout
    # NO_COLOR beats FORCE_COLOR: a user who asked for no colour gets none.
    muted = _run_cli(tmp_path, "--help", unset=("TTY_COMPATIBLE",), COLUMNS="200", FORCE_COLOR="1", NO_COLOR="1")
    assert not COLOUR_SGR.search(muted.stdout)
    assert COLOUR_SGR.search(coloured.stdout)
    assert SGR.sub("", muted.stdout) == plain.stdout


def test_help_markup_is_well_formed() -> None:
    help_text = build_parser().format_help()

    rendered = Text.from_markup(help_text)  # raises MarkupError on a stray closing tag

    assert "Server options:" in rendered.plain
    assert "[/" not in rendered.plain


def test_auth_is_hidden_but_still_accepted(capsys: pytest.CaptureFixture[str]) -> None:
    assert parse_args(["--auth", "bearer"]).auth == "bearer"
    assert parse_args([]).auth is None
    assert "--auth" not in build_parser().format_help()
    # The refusal and the choices are unchanged (SEC-04).
    for argv in (["--auth", "none"], ["--auth=none"], ["--auth", "oauth"]):
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2, argv
    assert "security.allow_no_auth" in capsys.readouterr().err


def test_the_examples_are_aligned(tmp_path: Path) -> None:
    lines = _plain_help(tmp_path).stdout.splitlines()
    start = lines.index("Examples:")
    example_lines = [line for line in lines[start + 1 :] if line.strip()]

    assert len(example_lines) == 4
    columns = set()
    for line, command in zip(example_lines, EXAMPLE_COMMANDS, strict=True):
        assert line.startswith("  ") and ":" in line
        columns.add(line.index(command, line.index(":")))
    assert len(columns) == 1, example_lines


def test_the_docs_url_is_shared_with_the_version_report() -> None:
    from exegol_mcp.bootstrap import version_command, versions

    assert not hasattr(versions, "DOCS_URL")
    assert version_command.DOCS_URL is cli.DOCS_URL


# --- Errors, usage and the --auth none refusal through the stderr console (D-33, SEC-08) ----------------------------


@pytest.fixture
def forced_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the shared stderr console a fresh one that sees FORCE_COLOR, as on a terminal."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TTY_COMPATIBLE", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(logging_setup, "_console", logging_setup.make_stderr_console())


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--type", "[/oops]"], "argument -t/--type: invalid choice: '[/oops]'"),
        (["--port", "[red]x[/red]"], "argument -p/--port: invalid int value: '[red]x[/red]'"),
        (["[bold]x"], "argument action: invalid choice: '[bold]x'"),
        (["--type", "stdio", "--bogus"], "unrecognized arguments: --bogus"),
    ],
)
def test_command_line_text_in_errors_prints_literally(
    argv: list[str], expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert captured.out == ""
    assert captured.err.startswith("usage: exegol-mcp")
    assert f"exegol-mcp: error: {expected}" in captured.err


def test_errors_go_through_the_stderr_console_in_colour(
    forced_colour: None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--type", "[bold]sse"])
    coloured = capsys.readouterr()

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(logging_setup, "_console", logging_setup.make_stderr_console())
    with pytest.raises(SystemExit):
        parse_args(["--type", "[bold]sse"])
    muted = capsys.readouterr()

    assert coloured.out == "" and muted.out == ""
    assert COLOUR_SGR.search(coloured.err)
    assert not COLOUR_SGR.search(muted.err)
    # Colour changes nothing but the colour: the plain text is argparse's own wording, with the argument literal.
    assert SGR.sub("", coloured.err) == SGR.sub("", muted.err)
    assert "exegol-mcp: error: argument -t/--type: invalid choice: '[bold]sse'" in SGR.sub("", coloured.err)


@pytest.mark.parametrize(
    "argv",
    [
        ["--auth", "none"],
        ["--auth=none", "version"],
        ["--au", "none"],
        ["--aut", "none"],
        ["--au=none"],
        ["--aut=none", "version"],
    ],
)
def test_the_auth_none_refusal_goes_through_the_console(
    argv: list[str], forced_colour: None, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)

    captured = capsys.readouterr()
    plain = SGR.sub("", captured.err)
    assert exc.value.code == 2
    assert captured.out == ""
    assert COLOUR_SGR.search(captured.err)
    assert plain.startswith("exegol-mcp: error: --auth no longer accepts 'none'.")
    assert "security.allow_no_auth" in plain
    assert plain.endswith("there is no CLI flag for it.\n")


@pytest.mark.parametrize("argv", [["--au", "bearer"], ["--aut=bearer"], ["--au"], ["--aut", "oauth"]])
def test_an_abbreviated_auth_without_none_keeps_argparses_own_behaviour(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the value `none` earns the D-12 guidance; bearer still parses and other values keep argparse's error."""
    if argv[-1] in ("--au", "oauth"):
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2
        assert "no longer accepts" not in capsys.readouterr().err
    else:
        assert parse_args(argv).auth == "bearer"


def test_errors_never_reach_stdout_and_help_does(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["--type", "stdio", "--bogus"], ["--type", "stdio", "--auth", "none"], ["--port", "x"]):
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2, argv
        captured = capsys.readouterr()
        assert captured.out == "", argv
        assert captured.err != "", argv

    with pytest.raises(SystemExit) as exc:
        parse_args(["--type", "stdio", "-h"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert captured.err == ""
    assert captured.out.startswith("usage: exegol-mcp")
    assert "Examples:" in captured.out


def test_a_stdio_command_line_error_leaves_stdout_empty(tmp_path: Path) -> None:
    # SEC-08: stdout is the protocol stream once a stdio server runs, so a command-line error must never reach it.
    result = _run_cli(tmp_path, "--type", "stdio", "--bogus")

    assert result.returncode == 2
    assert result.stdout == ""
    assert "exegol-mcp: error: unrecognized arguments: --bogus" in result.stderr
