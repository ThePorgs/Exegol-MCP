"""Both transports connect, with the stable launch contract (COMP-01/COMP-02).

- The defaults stay what v1 clients are configured for: HTTP on 127.0.0.1, port 2187, path /mcp, bearer auth.
- Over streamable HTTP, a client with the right bearer token connects at 127.0.0.1:<port>/mcp and calls tools in
  both protocol eras; a missing, wrong or non-bearer token gets HTTP 400 (not 401/403, which some clients take as
  the start of an OAuth flow); nothing else is served besides /mcp.
- run_http() hands uvicorn exactly that app, bound to 127.0.0.1.
- The real entry point started as a process connects over HTTP (bearer) and over stdio.

Everything is Docker-free and keeps away from the user's real secret, keyring and Exegol config: the in-process
legs inject a secret, and the process legs go through tests.fake_server with a temporary HOME.
"""

import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from exegol_mcp.app import build_server
from exegol_mcp.bootstrap import transports
from exegol_mcp.bootstrap.cli import DEFAULT_PORT, HTTP_HOST, MCP_PATH, parse_args
from exegol_mcp.bootstrap.transports import build_http_app, run_http, uvicorn_log_config
from tests.fake_server import REPO_ROOT, launcher_command, server_env, stdio_server_parameters
from tests.fakes import V2_TOOL_NAMES, FakeExegolAdapter, catalog_router
from tests.http_harness import free_port, serve_in_thread
from tests.test_sdk_selfcheck import _run_python

SECRET = "transport-test-secret"
ERAS = ["legacy", "2026-07-28"]
PROCESS_TIMEOUT = 30.0


def _url(port: int, path: str = MCP_PATH) -> str:
    return f"http://{HTTP_HOST}:{port}{path}"


async def _connect_and_call(url: str, mode: str, secret: str = SECRET) -> tuple[set[str], dict[str, Any] | None]:
    """Connect with a bearer token, list the tools and call list_containers; return (tool names, structured)."""
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {secret}"}) as http_client,
        Client(streamable_http_client(url, http_client=http_client), mode=mode) as client,
    ):
        names = {tool.name for tool in (await client.list_tools()).tools}
        result = await client.call_tool("list_containers", {})
        assert not result.is_error, result.content
        return names, result.structured_content


async def _rejected_statuses(url: str) -> dict[str, tuple[int, Any]]:
    """POST an initialize request with no, a wrong, and a non-bearer Authorization header."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    }
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    attempts = {
        "missing": {},
        "wrong": {"Authorization": f"Bearer not-{SECRET}"},
        "empty": {"Authorization": "Bearer"},
        "basic": {"Authorization": f"Basic {SECRET}"},
    }
    statuses: dict[str, tuple[int, Any]] = {}
    async with httpx2.AsyncClient() as http_client:
        for label, auth in attempts.items():
            response = await http_client.post(url, json=body, headers={**headers, **auth})
            statuses[label] = (response.status_code, response.json())
    return statuses


def _assert_all_rejected_with_400(statuses: dict[str, tuple[int, Any]]) -> None:
    for label, (status, payload) in statuses.items():
        assert status == 400, (label, status, payload)
        assert payload == {"error": "Invalid bearer auth credentials"}, (label, payload)


# --- The stable launch contract --------------------------------------------------------------------------------------


def test_the_default_launch_is_http_on_localhost_2187_mcp_with_bearer_auth() -> None:
    from exegol_mcp.bootstrap.config import resolve
    from exegol_mcp.models.config import OperatorConfig

    args = parse_args([])
    # The mirrored flags default to None (D-06); the stable launch comes from CLI>file>default resolution.
    assert (args.type, args.port, args.auth) == (None, None, None)
    resolved = resolve(args, OperatorConfig())
    assert (resolved.transport, resolved.port, resolved.auth) == ("http", 2187, "bearer")
    assert (HTTP_HOST, DEFAULT_PORT, MCP_PATH) == ("127.0.0.1", 2187, "/mcp")
    # The v1 flags are all still accepted; --auth is bearer-only now (no-auth is config-only, D-12).
    legacy = parse_args(["-t", "stdio", "-p", "8000", "--auth", "bearer", "-pc", "-vv"])
    assert (legacy.type, legacy.port, legacy.auth, legacy.print_config, legacy.verbose) == (
        "stdio",
        8000,
        "bearer",
        True,
        2,
    )


@pytest.fixture
def recorded_http_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the uvicorn server run run_http() starts with a recorder of the Config it was built with.

    The Config's own logging step (dictConfig) is skipped: it would shut down pytest's live handlers.
    """
    captured: dict[str, Any] = {}
    monkeypatch.setattr(transports.uvicorn.Config, "configure_logging", lambda self: None)

    def fake_run(self: Any, sockets: Any = None) -> None:
        captured["app"] = self.config.app
        captured["host"] = self.config.host
        captured["port"] = self.config.port
        captured["log_config"] = self.config.log_config
        self.started = True

    monkeypatch.setattr(transports._AnnouncingServer, "run", fake_run)
    return captured


def test_run_http_serves_the_bearer_gated_app_on_localhost_only(recorded_http_run: dict[str, Any]) -> None:
    run_http(build_server(None, FakeExegolAdapter()), port=DEFAULT_PORT, auth="bearer", secret=SECRET)

    assert recorded_http_run["host"] == "127.0.0.1"
    assert recorded_http_run["port"] == 2187
    assert recorded_http_run["log_config"] == uvicorn_log_config(0)
    assert type(recorded_http_run["app"]).__name__ == "BearerGate"


@pytest.mark.parametrize(("verbose", "level"), [(0, "WARNING"), (1, "INFO"), (2, "INFO"), (3, "INFO")])
def test_uvicorn_logging_config_leaves_records_to_the_root_handler(verbose: int, level: str) -> None:
    config = uvicorn_log_config(verbose)

    assert "handlers" not in config
    assert "formatters" not in config
    assert config["disable_existing_loggers"] is False
    loggers = config["loggers"]
    assert set(loggers) == {"uvicorn", "uvicorn.error", "uvicorn.access"}
    for name, settings in loggers.items():
        assert settings == {"level": level, "propagate": True}, name


def test_run_http_hands_uvicorn_the_verbosity_it_was_given(recorded_http_run: dict[str, Any]) -> None:
    run_http(build_server(None, FakeExegolAdapter()), port=DEFAULT_PORT, auth="bearer", secret=SECRET, verbose=2)

    assert recorded_http_run["log_config"] == uvicorn_log_config(2)


def test_run_http_calls_the_bound_hook_only_once_the_port_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """GR-IN-01: the hook runs with the port already accepting connections, then the server leaves its loop."""
    monkeypatch.setattr(transports, "uvicorn_log_config", lambda verbose: None)

    async def stop_at_once(self: Any) -> None:
        return None

    monkeypatch.setattr(transports._AnnouncingServer, "main_loop", stop_at_once)
    port = free_port()
    reachable: list[bool] = []

    def on_bound() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(2)
            reachable.append(probe.connect_ex((HTTP_HOST, port)) == 0)

    run_http(build_server(None, FakeExegolAdapter()), port=port, auth="bearer", secret=SECRET, on_bound=on_bound)

    assert reachable == [True]


def test_run_http_does_not_call_the_bound_hook_when_the_port_is_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transports, "uvicorn_log_config", lambda verbose: None)
    calls: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind((HTTP_HOST, 0))
        holder.listen()
        port = int(holder.getsockname()[1])

        with pytest.raises(SystemExit) as exc:
            run_http(
                build_server(None, FakeExegolAdapter()),
                port=port,
                auth="bearer",
                secret=SECRET,
                on_bound=lambda: calls.append("bound"),
            )

    assert exc.value.code == 3
    assert calls == []


def test_run_http_warns_nothing_about_no_auth_itself(
    caplog: pytest.LogCaptureFixture, recorded_http_run: dict[str, Any]
) -> None:
    """GR-IN-02: the one no-auth warning is the startup output's, which names the config key."""
    with caplog.at_level("WARNING"):
        run_http(
            build_server(None, FakeExegolAdapter()), port=DEFAULT_PORT, auth="none", secret=None, allow_no_auth=True
        )

    assert "DISABLED" not in caplog.text
    assert type(recorded_http_run["app"]).__name__ != "BearerGate"


# The real configuration step uvicorn.run performs (logging.config.dictConfig), in an isolated interpreter: dictConfig
# shuts down the live handlers, which would break pytest's own in the test process. The -v count is argv[1].
_UVICORN_SCRIPT = """
import logging
import logging.config
import sys

from exegol_mcp.bootstrap.logging_setup import route_logging_to_stderr
from exegol_mcp.bootstrap.transports import uvicorn_log_config

verbose = int(sys.argv[1])
route_logging_to_stderr(verbose)
logging.config.dictConfig(uvicorn_log_config(verbose))
logging.getLogger("uvicorn.error").info("Uvicorn running on http://127.0.0.1:2187")
logging.getLogger("uvicorn.error").error("Could not bind port 2187")
logging.getLogger("uvicorn.access").debug("access-debug-hidden")
logging.getLogger("uvicorn.access").info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5555", "POST", "/mcp", "1.1", 200)
"""


@pytest.mark.parametrize("verbose", [0, 1, 3])
def test_uvicorn_lines_reach_the_root_handler_once(tmp_path: Path, verbose: int) -> None:
    result = _run_python(tmp_path, _UVICORN_SCRIPT, str(verbose))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    startup = "[*] Uvicorn running on http://127.0.0.1:2187"
    access = '[*] 127.0.0.1:5555 - "POST /mcp HTTP/1.1" 200'
    # A bind error is a warning-level problem: it prints at every verbosity, exactly once.
    assert sum("Could not bind port 2187" in line for line in lines) == 1, result.stderr
    # The startup and access lines are -v detail (D-30).
    assert lines.count(startup) == (1 if verbose >= 1 else 0), result.stderr
    assert lines.count(access) == (1 if verbose >= 1 else 0), result.stderr
    assert "access-debug-hidden" not in result.stderr


def test_http_app_keeps_the_streaming_response_for_cancellation_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-19 / RESEARCH Pitfall 4: build_http_app must never request json_response=True or stateless_http=True.

    Either would silently drop the SDK streaming response that the request cancellation (EXEC-07) and the
    per-client download progress (EXEC-09) this phase is built on depend on.
    """
    server = build_server(None, FakeExegolAdapter())
    original = server.streamable_http_app
    calls: list[dict[str, Any]] = []

    def spy(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(server, "streamable_http_app", spy)

    build_http_app(server, "none", None, allow_no_auth=True)
    build_http_app(server, "bearer", SECRET)

    assert len(calls) == 2  # one per auth mode
    for kwargs in calls:
        assert kwargs.get("json_response", False) is False  # streaming response kept (D-19)
        assert kwargs.get("stateless_http", False) is False
        assert kwargs["streamable_http_path"] == MCP_PATH
        assert kwargs["host"] == HTTP_HOST
        assert kwargs["transport_security"] is transports.LOCALHOST_TRANSPORT_SECURITY  # explicit allowlist (SEC-03)


def test_build_http_app_no_auth_fails_closed_unless_explicitly_allowed() -> None:
    """SEC-04/D-13: build_http_app refuses to return an unauthenticated app unless allow_no_auth is true."""
    from starlette.applications import Starlette

    server = build_server(None, FakeExegolAdapter())
    with pytest.raises(RuntimeError, match="allow_no_auth"):
        build_http_app(server, "none", None)
    plain = build_http_app(server, "none", None, allow_no_auth=True)
    assert isinstance(plain, Starlette)  # the SDK app itself, no BearerGate, no re-mount


# --- Streamable HTTP, in process -----------------------------------------------------------------------------------


async def test_http_foreign_host_and_origin_are_refused_but_localhost_bearer_connects() -> None:
    """SEC-03/D-11: an explicit TransportSecuritySettings refuses a foreign Host (421) and Origin (403)."""
    app = build_http_app(build_server(None, FakeExegolAdapter(exec_router=catalog_router)), "bearer", SECRET)
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    }
    base = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SECRET}",
    }
    with serve_in_thread(app) as port:
        url = _url(port)
        async with httpx2.AsyncClient() as http_client:
            # Foreign Host -> 421 even with a valid bearer token (DNS-rebinding protection, CVE-2025-66416).
            r_host = await http_client.post(url, json=body, headers={**base, "Host": "evil.com"})
            # Correct localhost Host but foreign Origin -> 403.
            r_origin = await http_client.post(url, json=body, headers={**base, "Origin": "http://evil.com"})
        assert r_host.status_code == 421, r_host.text
        assert r_origin.status_code == 403, r_origin.text
        # A correct localhost Host with a bearer token still connects and lists tools.
        names, _structured = await _connect_and_call(url, "legacy")
    assert names == V2_TOOL_NAMES


@pytest.mark.parametrize("mode", ERAS)
async def test_http_a_client_with_the_right_bearer_token_connects_and_calls_tools(mode: str) -> None:
    app = build_http_app(build_server(None, FakeExegolAdapter(exec_router=catalog_router)), "bearer", SECRET)
    with serve_in_thread(app) as port:
        names, structured = await _connect_and_call(_url(port), mode)
    assert names == V2_TOOL_NAMES
    assert structured is not None and [c["name"] for c in structured["result"]] == ["exegol-lab", "exegol-lab2"]


async def test_http_a_missing_wrong_or_non_bearer_token_gets_400() -> None:
    app = build_http_app(build_server(None, FakeExegolAdapter()), "bearer", SECRET)
    with serve_in_thread(app) as port:
        _assert_all_rejected_with_400(await _rejected_statuses(_url(port)))


async def test_http_only_the_mcp_path_is_served() -> None:
    app = build_http_app(build_server(None, FakeExegolAdapter()), "bearer", SECRET)
    with serve_in_thread(app) as port:
        async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {SECRET}"}) as http_client:
            response = await http_client.post(_url(port, "/"), json={})
    assert response.status_code == 404


async def test_http_the_app_run_http_builds_answers_mcp(recorded_http_run: dict[str, Any]) -> None:
    """The very app run_http() would hand to uvicorn, served for real, accepts a bearer client."""
    run_http(build_server(None, FakeExegolAdapter()), port=DEFAULT_PORT, auth="bearer", secret=SECRET)
    with serve_in_thread(recorded_http_run["app"]) as port:
        names, _structured = await _connect_and_call(_url(port), "legacy")
        _assert_all_rejected_with_400(await _rejected_statuses(_url(port)))
    assert names == V2_TOOL_NAMES


# --- The real entry point as a process --------------------------------------------------------------------------------


def _wait_for_port(port: int, process: subprocess.Popen[bytes], deadline: float) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server exited early with {process.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((HTTP_HOST, port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"nothing listening on {HTTP_HOST}:{port}")


@contextmanager
def _http_server_process(home: Path, port: int) -> Iterator[tuple[subprocess.Popen[bytes], Path]]:
    """Start `exegol-mcp --type http --port <port>` (bearer auth, injected secret) through the launcher."""
    log = home / "server.log"
    with log.open("wb") as log_file:
        process = subprocess.Popen(
            launcher_command("--type", "http", "--port", str(port), "--auth", "bearer", "--accept-eula"),
            env=server_env(home, EXEGOL_MCP_TEST_SECRET=SECRET),
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_port(port, process, time.monotonic() + PROCESS_TIMEOUT)
            yield process, log
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
                try:
                    process.wait(timeout=PROCESS_TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


async def test_process_http_connects_with_bearer_and_rejects_bad_tokens(tmp_path: Path) -> None:
    port = free_port()
    with _http_server_process(tmp_path, port) as (process, log):
        results = {mode: await _connect_and_call(_url(port), mode) for mode in ERAS}
        _assert_all_rejected_with_400(await _rejected_statuses(_url(port)))
        assert process.poll() is None, log.read_text()
    output = log.read_text()
    for names, structured in results.values():
        assert names == V2_TOOL_NAMES
        assert structured is not None and structured["result"]
    # The startup output names the endpoint, and the secret stays redacted in it (it was not just created).
    assert f"http://127.0.0.1:{port}/mcp" in output
    assert SECRET not in output


@pytest.mark.parametrize("mode", ERAS)
async def test_process_stdio_connects_and_calls_tools(tmp_path: Path, mode: str) -> None:
    async with Client(stdio_server_parameters(tmp_path), mode=mode) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        result = await client.call_tool("server_info", {})
    assert names == V2_TOOL_NAMES
    assert not result.is_error, result.content
    assert result.structured_content is not None
    assert result.structured_content["docker_status"] == "reachable (fake Docker)"
