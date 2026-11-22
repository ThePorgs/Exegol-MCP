"""stdio stdout purity (SEC-08): in stdio mode the server's stdout carries JSON-RPC messages and nothing else.

The real entry point is launched as a subprocess through tests.fake_server (Docker-free, isolated HOME) and driven
with raw JSON-RPC lines, so the test sees every byte the process writes. EXEGOL_MCP_TEST_NOISE=1 makes the startup
readiness probe and a mid-session tool call write marker lines through the Exegol SDK logger, the SDK console, an
mcp.* logger, the package logger and (during the session) a bare print(). The test then checks that:

- every stdout line parses as a JSON-RPC message (no banner, no client template, no rich output, no secret);
- every marker line is on stderr, so the paths that would have polluted stdout really ran.

The same launcher is the stdio leg of the responsiveness regression (tests/test_responsiveness.py, D-24).
"""

import json
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import Process
from mcp_types import jsonrpc_message_adapter
from mcp_types.version import LATEST_HANDSHAKE_VERSION

from tests.fake_server import NOISE_MARKER, REPO_ROOT, launcher_command, server_env, stdio_args

STEP_TIMEOUT = 30.0


class RawStdioSession:
    """A stdio MCP server subprocess driven with raw JSON-RPC lines, keeping every stdout line it wrote."""

    def __init__(self, process: Process) -> None:
        assert process.stdin is not None and process.stdout is not None
        self.process = process
        self.stdout_lines: list[str] = []
        self._buffer = b""
        self._next_id = 0

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._send(message)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and return its response, recording every line read on the way."""
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        await self._send(message)
        with anyio.fail_after(STEP_TIMEOUT):
            while True:
                line = await self._read_line()
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue  # recorded in stdout_lines; the purity assertion reports it
                if isinstance(parsed, dict) and parsed.get("id") == request_id:
                    return parsed

    async def close(self) -> tuple[int, str]:
        """Close stdin, collect the remaining stdout, and return (exit code, all of stderr)."""
        assert self.process.stdin is not None and self.process.stdout is not None
        assert self.process.stderr is not None
        await self.process.stdin.aclose()
        stderr = b""
        with anyio.fail_after(STEP_TIMEOUT):
            async for chunk in self.process.stdout:
                self._buffer += chunk
            async for chunk in self.process.stderr:
                stderr += chunk
            code = await self.process.wait()
        if self._buffer:
            self.stdout_lines.extend(self._buffer.decode("utf-8", errors="replace").splitlines())
            self._buffer = b""
        return code, stderr.decode("utf-8", errors="replace")

    async def _send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        await self.process.stdin.send((json.dumps(message) + "\n").encode())

    async def _read_line(self) -> str:
        assert self.process.stdout is not None
        while b"\n" not in self._buffer:
            self._buffer += await self.process.stdout.receive()
        raw, self._buffer = self._buffer.split(b"\n", 1)
        line = raw.decode("utf-8", errors="replace")
        self.stdout_lines.append(line)
        return line


def _non_json_rpc_lines(lines: list[str]) -> list[str]:
    """Every stdout line that is not one complete JSON-RPC message (blank lines included)."""
    bad: list[str] = []
    for line in lines:
        try:
            jsonrpc_message_adapter.validate_json(line)
        except ValueError:
            bad.append(line)
    return bad


async def _run_noisy_stdio_session(
    home: Path, *, noise: bool = True
) -> tuple[RawStdioSession, int, str, dict[str, dict[str, Any]]]:
    process = await anyio.open_process(
        launcher_command(*stdio_args()),
        env=server_env(home, **({"EXEGOL_MCP_TEST_NOISE": "1"} if noise else {})),
        cwd=REPO_ROOT,
    )
    session = RawStdioSession(process)
    try:
        responses = {
            "initialize": await session.request(
                "initialize",
                {
                    "protocolVersion": LATEST_HANDSHAKE_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "stdio-purity-test", "version": "0"},
                },
            )
        }
        await session.notify("notifications/initialized")
        responses["tools/list"] = await session.request("tools/list")
        # list_containers is where the launcher's adapter makes the SDK-style noise mid-session.
        responses["list_containers"] = await session.request("tools/call", {"name": "list_containers", "arguments": {}})
        responses["server_info"] = await session.request("tools/call", {"name": "server_info", "arguments": {}})
        code, stderr = await session.close()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    return session, code, stderr, responses


async def test_stdio_stdout_carries_only_json_rpc_messages(tmp_path: Path) -> None:
    session, code, stderr, responses = await _run_noisy_stdio_session(tmp_path)

    assert code == 0, stderr
    assert session.stdout_lines, "the server wrote nothing on stdout"
    assert _non_json_rpc_lines(session.stdout_lines) == [], stderr
    stdout = "\n".join(session.stdout_lines)
    for leak in (NOISE_MARKER, "MCP client config:", "serving on", "Bearer"):
        assert leak not in stdout, leak

    # The session really worked: a handshake, the tool list, and two tool calls answered on stdout.
    assert responses["initialize"]["result"]["protocolVersion"] == LATEST_HANDSHAKE_VERSION
    assert {tool["name"] for tool in responses["tools/list"]["result"]["tools"]} >= {"list_containers", "server_info"}
    assert responses["list_containers"]["result"]["isError"] is False
    assert responses["server_info"]["result"]["isError"] is False


async def test_stdio_sdk_and_logger_output_lands_on_stderr(tmp_path: Path) -> None:
    """The noisy paths ran, and every one of their lines went to stderr (proving the purity test is not vacuous)."""
    _session, code, stderr, _responses = await _run_noisy_stdio_session(tmp_path)

    assert code == 0, stderr
    channels = ["exegol-logger", "exegol-console", "mcp-logger", "package-logger"]
    for where in ("readiness", "tool-call"):
        for channel in channels:
            assert f"{NOISE_MARKER} {channel} {where}" in stderr, (channel, where, stderr)
    assert f"{NOISE_MARKER} bare-print tool-call" in stderr, stderr
    # D-30: a default stdio start prints no serving line (only -v does, and only on stderr).
    assert "serving on" not in stderr
    # D-31: a stdio start prints neither the client config nor the --print-config hint.
    assert "MCP client config:" not in stderr
    assert "--print-config" not in stderr


async def test_a_default_stdio_start_prints_only_warnings_on_stderr(tmp_path: Path) -> None:
    """D-30/D-31: without noise a default stdio session writes nothing on stderr but warnings and errors."""
    session, code, stderr, _responses = await _run_noisy_stdio_session(tmp_path, noise=False)

    assert code == 0, stderr
    assert _non_json_rpc_lines(session.stdout_lines) == []
    lines = [line for line in stderr.splitlines() if line.strip()]
    assert all(line.startswith(("[!] ", "[-] ")) for line in lines), stderr
    assert "serving on" not in stderr
    assert "MCP client config:" not in stderr


def test_the_purity_check_rejects_a_non_json_rpc_line() -> None:
    """Inversion: the line check flags a banner line, a blank line, and JSON that is not JSON-RPC."""
    good = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
    lines = [good, "exegol-mcp 2.0.0", "", json.dumps({"hello": "world"})]
    assert _non_json_rpc_lines(lines) == lines[1:]
