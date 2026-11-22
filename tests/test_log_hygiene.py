"""SEC-02 credential log-hygiene (D-08): a password never reaches any stderr line at any verbosity.

Docker-free: exec_remote_command runs against FakeExegolAdapter. The process stderr is redirected to an in-memory
buffer (the ExeLogHandler's console resolves sys.stderr on every write, so it follows the redirect),
configure_stderr_logging is run
at -v 0..3, and every captured stderr line is greped for the sentinel password. The command is handled twice to
cover the idempotency edge (a second handling must leak no more than the first).
"""

import io
import logging
import sys

import pytest

from exegol_mcp.bootstrap.logging_setup import configure_stderr_logging
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.services.remote import exec_remote_command
from tests.fakes import FakeExegolAdapter

# A unique, easily-greppable credential that would only appear on stderr if the server logged the command/password.
SENTINEL_PW = "SENTINEL-PW-9f3a"


@pytest.mark.parametrize("verbose", [0, 1, 2, 3])
async def test_password_never_appears_on_any_stderr_line_at_any_verbosity(
    verbose: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    configure_stderr_logging(verbose)

    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="done", stderr="", exit_code=0))
    # Twice: the no-log policy must hold on a repeat handling (SEC-02 idempotency edge).
    for _ in range(2):
        await exec_remote_command(
            adapter, "exegol-lab", "smb", "10.0.0.1", "admin", SENTINEL_PW, "id", domain="CORP"
        )

    leaking = [line for line in buffer.getvalue().splitlines() if SENTINEL_PW in line]
    assert leaking == [], f"password leaked on stderr at -v{verbose}: {leaking!r}"


# G-03-1: over streamable HTTP, sse_starlette logs every outgoing SSE chunk (every tool result body) at DEBUG.
@pytest.mark.parametrize("verbose", [0, 1, 2, 3])
def test_sse_starlette_is_capped_at_info_or_root_level(verbose: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    configure_stderr_logging(verbose)

    expected = max(logging.getLogger().getEffectiveLevel(), logging.INFO)
    assert logging.getLogger("sse_starlette.sse").getEffectiveLevel() == expected


def test_sse_starlette_cap_is_targeted_at_debug_verbosity(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    configure_stderr_logging(3)

    logging.getLogger("sse_starlette.sse").debug("chunk: %s", SENTINEL_PW)
    logging.getLogger("exegol_mcp.services.remote").debug("exegol-mcp debug kept")
    logging.getLogger("mcp.server.streamable_http").debug("mcp sdk debug kept")

    output = buffer.getvalue()
    assert SENTINEL_PW not in output
    assert "exegol-mcp debug kept" in output
    assert "mcp sdk debug kept" in output


def test_a_debug_line_is_printed_once_and_verbatim_at_vvv(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    configure_stderr_logging(3)

    # Square brackets in operator text stay literal (no markup in exegol-mcp's own records, D-01).
    logging.getLogger("exegol_mcp.services.remote").debug("kept [x] line")

    assert buffer.getvalue().splitlines().count("[D] kept [x] line") == 1, buffer.getvalue()
