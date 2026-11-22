"""D-12/D-29 copy-safety of the client config block, Docker-free.

D-29 refines D-12: on a colour terminal the entry is syntax-highlighted, with colour as the only change. Stripped of
its SGR sequences it is byte-identical to the raw entry at any width; off a terminal and with NO_COLOR it is the raw
entry itself.

Whatever an operator copies from the terminal (the JSON entry and the bearer token in it) must arrive byte for byte:
no ExeLog prefix, escape byte, border, padding, wrapped line or trailing space, at any terminal width and with colour
forced on, and no '=' separator line around it. The block is highlighted on a colour terminal and raw elsewhere; only
the heading and the hint are console lines. The entry point test runs the real `main()` through the Docker-free
launcher at COLUMNS=20 with FORCE_COLOR=1, the in-process tests cover the transport split (D-31: an HTTP start prints
the block on stdout, a stdio start prints nothing at all), and an inversion shows a rich console would colour the
block, so the guards are not vacuous.
"""

import io
import json
import re
import secrets
import subprocess
from pathlib import Path

import pytest
from rich.console import Console
from rich.highlighter import JSONHighlighter
from rich.text import Text

from exegol_mcp.bootstrap import logging_setup, versions
from exegol_mcp.bootstrap.cli import parse_args
from tests.fake_server import REPO_ROOT, launcher_command, server_env

_SGR = re.compile(r"\x1b\[[0-9;]*m")
_HEADING = "[*] MCP client config:"
_HINT = "[*] The MCP client config can be printed any time with: exegol-mcp --print-config"


@pytest.fixture(autouse=True)
def narrow_coloured_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 20-column terminal with colour forced on: the harshest setting for a copied block."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "20")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TTY_COMPATIBLE", raising=False)
    # A fresh shared console re-reads the colour environment, as route_logging_to_stderr() does at startup.
    monkeypatch.setattr(logging_setup, "_console", logging_setup.make_stderr_console())


def _http_args(*extra: str):
    return parse_args(["--type", "http", "--port", "2187", "--auth", "bearer", *extra])


def _expected_block(token: str) -> str:
    entry = {"type": "http", "url": "http://127.0.0.1:2187/mcp", "headers": {"Authorization": f"Bearer {token}"}}
    return '"exegol-mcp": ' + json.dumps(entry, indent=4)


def _assert_clean_lines(raw: str) -> None:
    lines = raw.split("\n")
    assert not any(line.endswith(" ") for line in lines), lines
    assert not any(re.fullmatch(r"=+", line) for line in lines), lines


def _print_config(tmp_path: Path, token: str, columns: str, **env: str | None) -> subprocess.CompletedProcess[str]:
    """Run the real entry point with --print-config; a None value drops that variable from the environment."""
    knobs = {"EXEGOL_MCP_TEST_SECRET": token, "COLUMNS": columns, "FORCE_COLOR": "1", "TERM": "xterm-256color"}
    knobs.update({name: value for name, value in env.items() if value is not None})
    environment = server_env(tmp_path, **knobs)
    for name, value in env.items():
        if value is None:
            environment.pop(name, None)
    return subprocess.run(
        launcher_command("--type", "http", "--port", "2187", "--print-config"),
        env=environment,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("columns", ["20", "300"])
def test_print_config_is_highlighted_and_strips_to_the_raw_entry(tmp_path: Path, columns: str) -> None:
    token = secrets.token_urlsafe(32)
    assert len(token) == 43
    result = _print_config(tmp_path, token, columns)

    assert result.returncode == 0, result.stderr
    expected = _expected_block(token)
    body = result.stdout.split("\n", 1)[1]
    # D-29: colour only. The entry carries SGR sequences on a terminal ...
    assert _SGR.search(body), "the entry is not highlighted on a forced-colour terminal"
    # ... and with them removed it is byte for byte the raw entry at any width.
    stripped = _SGR.sub("", result.stdout)
    assert stripped == f"{_HEADING}\n{expected}\n"
    assert "\x1b" not in stripped
    _assert_clean_lines(stripped)
    authorization = [line for line in stripped.split("\n") if "Authorization" in line]
    assert len(authorization) == 1
    assert token in authorization[0]


def test_print_config_with_no_color_is_the_raw_entry(tmp_path: Path) -> None:
    """NO_COLOR wins over FORCE_COLOR: rich keeps bold under NO_COLOR, so the raw path is taken."""
    token = secrets.token_urlsafe(32)
    result = _print_config(tmp_path, token, "20", NO_COLOR="1")

    assert result.returncode == 0, result.stderr
    expected = _expected_block(token)
    assert result.stdout.endswith(expected + "\n")
    assert "\x1b" not in expected
    block_lines = result.stdout.split("\n")[1:]
    assert not any("\x1b" in line for line in block_lines)


def test_print_config_into_a_pipe_is_the_raw_entry(tmp_path: Path) -> None:
    token = secrets.token_urlsafe(32)
    result = _print_config(tmp_path, token, "20", FORCE_COLOR=None, TTY_COMPATIBLE=None, NO_COLOR=None)

    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout
    assert result.stdout == f"{_HEADING}\n{_expected_block(token)}\n"


def test_the_json_entry_round_trips() -> None:
    token = secrets.token_urlsafe(32)
    block = versions.client_config_template(_http_args("--print-config"), token, created=False)

    assert block == _expected_block(token)
    assert json.loads("{" + block + "}")["exegol-mcp"] == {
        "type": "http",
        "url": "http://127.0.0.1:2187/mcp",
        "headers": {"Authorization": f"Bearer {token}"},
    }


def test_a_stdio_start_prints_no_client_config(capsys: pytest.CaptureFixture[str]) -> None:
    """D-31: stdout is the protocol stream and a log panel is no place for the entry: nothing prints in stdio."""
    versions.print_startup_client_config(parse_args(["--type", "stdio"]), None, created=False)
    versions.print_startup_client_config(parse_args(["--type", "stdio"]), "s3cr3t", created=True)

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_http_startup_block_and_hint_go_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    token = secrets.token_urlsafe(32)

    versions.print_startup_client_config(_http_args(), token, created=True)

    captured = capsys.readouterr()
    assert captured.err == ""
    expected = _expected_block(token)
    assert _SGR.sub("", captured.out) == f"{_HEADING}\n{expected}\n\n{_HINT}\n"
    _assert_clean_lines(_SGR.sub("", captured.out))


def test_a_later_start_prints_only_the_hint(capsys: pytest.CaptureFixture[str]) -> None:
    versions.print_startup_client_config(_http_args(), "s3cr3t", created=False)

    captured = capsys.readouterr()
    assert _SGR.sub("", captured.out) == f"{_HINT}\n"
    assert "s3cr3t" not in captured.out + captured.err


def test_a_console_without_colour_gets_the_raw_entry() -> None:
    block = versions.client_config_template(_http_args("--print-config"), secrets.token_urlsafe(32), created=False)
    sink = io.StringIO()

    versions.write_copyable_block(Console(file=sink, force_terminal=True, color_system=None, width=20), block)

    assert sink.getvalue() == block + "\n"


def test_a_wrapping_console_would_break_the_entry() -> None:
    """Inversion: highlighting without soft wrap breaks the entry at a narrow width, so the guards can fail."""
    block = versions.client_config_template(_http_args("--print-config"), secrets.token_urlsafe(32), created=False)
    sink = io.StringIO()

    Console(file=sink, force_terminal=True, color_system="truecolor", width=20).print(JSONHighlighter()(Text(block)))

    assert _SGR.sub("", sink.getvalue()) != block + "\n"
