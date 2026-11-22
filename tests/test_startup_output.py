"""What a start prints (D-30, D-31, SEC-08), Docker-free and keychain-free.

main() runs in process with the secret store, the host probe, the adapter, the SDK patch and both transports faked:
no server starts, no socket opens, no Docker or keychain is touched. The assertions cover the stream split (stdout
for an HTTP start, stderr only for stdio), the first-run client config block against the later-start hint, the
bearer token appearing once and only on the start that created it, warnings still printing at the default level, and
the copy-safety of the first-run block at a narrow width with colour forced on. The serving line and the first-run
block print only once the HTTP port is bound (GR-IN-01): the faked transport calls the bound hook (or fails to bind),
and a few tests run the real run_http on an ephemeral 127.0.0.1 port, one of them on a port another socket holds.
"""

import json
import logging
import re
import secrets
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import exegol_mcp.adapter.host as host_module
import exegol_mcp.adapter.real as real_module
import exegol_mcp.main as main_module
from exegol_mcp.adapter import sdk_headless
from exegol_mcp.bootstrap import logging_setup, secret_store, transports, versions
from exegol_mcp.main import main
from tests.fakes import FakeExegolAdapter
from tests.http_harness import free_port
from tests.test_bootstrap import FakeHostProbe

_SGR = re.compile(r"\x1b\[[0-9;]*m")
_HEADING = "[*] MCP client config:"
_HINT = "[*] The MCP client config can be printed any time with: exegol-mcp --print-config"
_LOCATION = "the test secret store"


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


@pytest.fixture(autouse=True)
def plain_wide_terminal(monkeypatch: pytest.MonkeyPatch, restore_logging: None, home: Path) -> None:
    """No colour variable and 200 columns, so a line is never wrapped; tests that want colour set it themselves."""
    for name in ("FORCE_COLOR", "TTY_COMPATIBLE", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(logging_setup, "_console", logging_setup.make_stderr_console())


class _Run:
    """What the faked transports were asked to do, and the probe that served the start."""

    def __init__(self) -> None:
        self.http: list[dict[str, Any]] = []
        self.stdio = 0
        self.probe = FakeHostProbe()


def _install_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str,
    created: bool,
    config: bytes | None = None,
    bind: bool | None = True,
) -> _Run:
    """Fake everything main() would touch outside the process; return the recorder of what it started.

    `bind` decides what the faked HTTP transport does: True calls the bound hook (the port was bound), False exits
    the way uvicorn does after a bind error, and None leaves the real run_http in place.
    """
    run = _Run()
    monkeypatch.setattr(secret_store, "load_bearer_secret", lambda: (token, created, _LOCATION))
    monkeypatch.setattr(secret_store, "read_user_config_file", lambda name: config if name == "config.yml" else None)
    monkeypatch.setattr(sdk_headless, "patch_exegol_sdk", lambda: None)
    monkeypatch.setattr(sdk_headless, "set_accept_eula", lambda accepted: None)
    monkeypatch.setattr(host_module, "RealHostProbe", lambda: run.probe)
    monkeypatch.setattr(real_module, "RealExegolAdapter", FakeExegolAdapter)
    if bind is not None:

        def fake_run_http(server: object, **kwargs: Any) -> None:
            run.http.append(kwargs)
            if not bind:
                raise SystemExit(3)
            kwargs["on_bound"]()

        monkeypatch.setattr(main_module, "run_http", fake_run_http)
    monkeypatch.setattr(main_module, "run_stdio", lambda server: setattr(run, "stdio", run.stdio + 1))
    monkeypatch.setattr(versions, "detect_wrapper_version", lambda: versions.library_version())
    return run


def _block(token: str, port: int = 2187) -> str:
    entry = {"type": "http", "url": f"http://127.0.0.1:{port}/mcp", "headers": {"Authorization": f"Bearer {token}"}}
    return '"exegol-mcp": ' + json.dumps(entry, indent=4)


def _serving(port: int = 2187, auth: str = "bearer") -> str:
    return f"[*] Exegol MCP v{versions.__version__} serving on http://127.0.0.1:{port}/mcp ({auth} auth)"


def test_a_first_http_start_prints_the_block_once_then_the_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = secrets.token_urlsafe(32)
    run = _install_doubles(monkeypatch, token=token, created=True)

    assert main(["--accept-eula", "--no-update-check"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"{_serving()}\n{_HEADING}\n{_block(token)}\n\n{_HINT}\n"
    assert captured.out.count(token) == 1
    assert len(run.http) == 1
    assert run.http[0]["port"] == 2187
    assert run.http[0]["verbose"] == 0


def test_a_later_http_start_prints_one_line_and_the_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = secrets.token_urlsafe(32)
    run = _install_doubles(monkeypatch, token=token, created=False)

    assert main(["--accept-eula", "--no-update-check"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{_serving()}\n{_HINT}\n"
    assert captured.err == ""
    for stream in (captured.out, captured.err):
        assert token not in stream
        assert "MCP client config:" not in stream
        assert "redacted" not in stream
    assert len(run.http) == 1


def test_the_hint_names_a_non_default_port(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _install_doubles(monkeypatch, token="t", created=False)

    assert main(["--accept-eula", "--no-update-check", "-p", "9000"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{_serving(9000)}\n{_HINT} -p 9000\n"
    assert run.http[0]["port"] == 9000


def test_the_hint_is_coloured_like_the_wrappers_examples(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(logging_setup, "_console", logging_setup.make_stderr_console())
    _install_doubles(monkeypatch, token="t", created=False)

    assert main(["--accept-eula", "--no-update-check"]) == 0

    out = capsys.readouterr().out
    hint = next(line for line in out.splitlines() if "can be printed any time" in line)
    assert _SGR.sub("", hint) == _HINT
    sequences = _SGR.findall(hint)
    assert sequences, "the hint carries no colour on a forced terminal"
    # The command is italic (SGR 3), and --print-config is bright blue (94) like the wrapper's highlighted token.
    assert any(re.fullmatch(r"\x1b\[(?:[0-9;]*;)?3(?:;[0-9;]*)?m", sequence) for sequence in sequences), sequences
    assert any("94" in sequence.strip("\x1b[m").split(";") for sequence in sequences), sequences


def test_a_first_start_block_is_copy_safe_at_a_narrow_width(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "20")
    monkeypatch.setattr(logging_setup, "_console", logging_setup.make_stderr_console())
    token = secrets.token_urlsafe(32)
    _install_doubles(monkeypatch, token=token, created=True)

    assert main(["--accept-eula", "--no-update-check"]) == 0

    out = capsys.readouterr().out
    assert _SGR.search(out)
    stripped = _SGR.sub("", out)
    assert f"\n{_HEADING}\n{_block(token)}\n\n" in stripped
    assert not any(line.endswith(" ") for line in stripped.split("\n"))


def test_a_no_auth_start_prints_the_hint_and_the_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = b"auth: none\nsecurity:\n  allow_no_auth: true\n"
    run = _install_doubles(monkeypatch, token="t", created=False, config=config)

    assert main(["--accept-eula", "--no-update-check"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{_serving(auth='no')}\n{_HINT}\n"
    assert "security.allow_no_auth" in captured.err
    assert captured.err.startswith("[!] ")
    assert captured.err.count("Authentication is DISABLED") == 1
    assert run.http[0]["auth"] == "none"
    assert run.http[0]["allow_no_auth"] is True


def test_warnings_still_print_at_the_default_level(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _install_doubles(monkeypatch, token="t", created=False)

    def warn_through_the_sdk_logger() -> str:
        from exegol.utils.ExeLog import logger

        logger.warning("Docker is slow to answer")
        return "reachable (Docker 29.8.2)"

    monkeypatch.setattr(run.probe, "probe_docker", warn_through_the_sdk_logger)

    assert main(["--accept-eula", "--no-update-check"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{_serving()}\n{_HINT}\n"
    assert "[!] Docker is slow to answer" in captured.err.splitlines()


@pytest.mark.parametrize("verbosity", [[], ["-v"]], ids=["default", "verbose"])
def test_a_stdio_start_never_prints_the_client_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verbosity: list[str]
) -> None:
    run = _install_doubles(monkeypatch, token="t", created=True)

    assert main(["--type", "stdio", "--accept-eula", "--no-update-check", *verbosity]) == 0

    captured = capsys.readouterr()
    assert run.stdio == 1
    assert run.http == []
    assert captured.out == ""
    for text in ("MCP client config:", "--print-config", "Bearer"):
        assert text not in captured.err
    serving = f"[V] Exegol MCP v{versions.__version__} serving on stdio"
    if verbosity:
        assert serving in captured.err.splitlines()
    else:
        assert not any(line.startswith(("[*]", "[V]")) for line in captured.err.splitlines()), captured.err


def test_print_config_still_prints_the_full_entry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = secrets.token_urlsafe(32)
    run = _install_doubles(monkeypatch, token=token, created=False)

    assert main(["--accept-eula", "--no-update-check", "--print-config"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{_HEADING}\n{_block(token)}\n"
    assert "--print-config" not in captured.out
    assert run.http == [] and run.stdio == 0


# --- GR-IN-01: nothing is claimed or shown before the HTTP port is bound ----------------------------------------------

_HINT_UNSHOWN = "[!] The bearer secret was created but not shown, because the server did not start."


def test_a_first_start_that_cannot_bind_prints_no_serving_line_and_no_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = secrets.token_urlsafe(32)
    run = _install_doubles(monkeypatch, token=token, created=True, bind=False)

    with pytest.raises(SystemExit) as exc:
        main(["--accept-eula", "--no-update-check"])

    captured = capsys.readouterr()
    assert exc.value.code == 3
    assert len(run.http) == 1
    assert captured.out == ""
    assert "serving on" not in captured.err
    assert "MCP client config:" not in captured.err
    # The token was created but never shown: the failure path points at --print-config and never carries the token.
    assert token not in captured.err
    assert captured.err.splitlines() == [
        f"{_HINT_UNSHOWN} Print the MCP client config with: exegol-mcp --print-config"
    ]


def test_the_unshown_secret_hint_names_a_non_default_port(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_doubles(monkeypatch, token="t", created=True, bind=False)

    with pytest.raises(SystemExit):
        main(["--accept-eula", "--no-update-check", "-p", "9000"])

    assert capsys.readouterr().err.endswith("exegol-mcp --print-config -p 9000\n")


def test_a_later_start_that_cannot_bind_prints_nothing_of_its_own(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The secret already existed: no serving line, no token, and no unshown-secret hint either."""
    _install_doubles(monkeypatch, token=secrets.token_urlsafe(32), created=False, bind=False)

    with pytest.raises(SystemExit):
        main(["--accept-eula", "--no-update-check"])

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_the_next_start_after_a_failed_first_start_still_prints_the_block_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A first start that never bound leaves the secret created-but-unshown; --print-config is the way to it."""
    token = secrets.token_urlsafe(32)
    _install_doubles(monkeypatch, token=token, created=True, bind=False)
    with pytest.raises(SystemExit):
        main(["--accept-eula", "--no-update-check"])
    assert token not in capsys.readouterr().out

    assert main(["--accept-eula", "--no-update-check", "--print-config"]) == 0

    assert capsys.readouterr().out == f"{_HEADING}\n{_block(token)}\n"


@pytest.fixture
def real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real run_http in process: no dictConfig (it would shut pytest's handlers) and no serving loop."""
    monkeypatch.setattr(transports, "uvicorn_log_config", lambda verbose: None)

    async def stop_at_once(self: object) -> None:
        # The server is bound and up here; leaving the loop at once lets run_http return and the port close.
        return None

    monkeypatch.setattr(transports._AnnouncingServer, "main_loop", stop_at_once)


def _hold_a_port() -> tuple[socket.socket, int]:
    """Occupy an ephemeral port on 127.0.0.1 so that binding it fails."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    return holder, int(holder.getsockname()[1])


def test_a_real_port_clash_shows_the_bind_error_and_no_serving_line_or_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], real_http: None
) -> None:
    token = secrets.token_urlsafe(32)
    _install_doubles(monkeypatch, token=token, created=True, bind=None)
    holder, port = _hold_a_port()
    try:
        with pytest.raises(SystemExit) as exc:
            main(["--accept-eula", "--no-update-check", "-p", str(port)])
    finally:
        holder.close()

    captured = capsys.readouterr()
    assert exc.value.code == 3
    assert captured.out == ""
    assert token not in captured.err
    assert "serving on" not in captured.err
    assert "MCP client config:" not in captured.err
    assert str(port) in captured.err  # uvicorn's own bind error names the address
    assert "error while attempting to bind" in captured.err
    assert _HINT_UNSHOWN in captured.err


def test_a_real_bind_prints_the_serving_line_and_the_block_once_it_is_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], real_http: None
) -> None:
    token = secrets.token_urlsafe(32)
    _install_doubles(monkeypatch, token=token, created=True, bind=None)
    port = free_port()

    assert main(["--accept-eula", "--no-update-check", "-p", str(port)]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"{_serving(port)}\n{_HEADING}\n{_block(token, port)}\n\n{_HINT} -p {port}\n"
    assert captured.err == ""


def test_a_real_no_auth_start_warns_exactly_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], real_http: None
) -> None:
    config = b"auth: none\nsecurity:\n  allow_no_auth: true\n"
    _install_doubles(monkeypatch, token="t", created=False, config=config, bind=None)
    port = free_port()

    assert main(["--accept-eula", "--no-update-check", "-p", str(port)]) == 0

    captured = capsys.readouterr()
    assert captured.err.count("Authentication is DISABLED") == 1
    assert "security.allow_no_auth" in captured.err
    assert captured.out == f"{_serving(port, auth='no')}\n{_HINT} -p {port}\n"
