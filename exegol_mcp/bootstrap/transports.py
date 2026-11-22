"""The stdio and HTTP transports (COMP-01/COMP-02, PROTO-01/02, Pitfall 8).

One MCPServer serves both protocol eras over both transports:
- stdio: server.run("stdio"). The SDK diverts fd 1 to stderr during the session; logging is already on the
  standard error stream (configure_stderr_logging) so the protocol stream on stdout stays clean (SEC-08).
- HTTP: server.streamable_http_app(streamable_http_path="/mcp", host="127.0.0.1",
  transport_security=LOCALHOST_TRANSPORT_SECURITY). An explicit TransportSecuritySettings localhost allowlist is
  passed (SEC-03, D-11) rather than relying on the SDK's inherited localhost default — a foreign Host is refused
  with 421 and a foreign Origin with 403 (CVE-2025-66416). The app is used directly and wrapped only by the
  pure-ASGI BearerGate; it is NOT re-mounted inside a custom Starlette, which would drop that middleware
  (Pitfall 8 / T-01-11).
"""

import socket
import sys
from collections.abc import Callable
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp
from uvicorn.config import STARTUP_FAILURE

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.bootstrap.cli import HTTP_HOST, MCP_PATH
from exegol_mcp.bootstrap.security import BearerGate

# SEC-03 / D-11: the explicit Host/Origin allowlist, mirroring the SDK's own localhost default so the port
# (2187 or operator-chosen) is covered by the `:*` wildcard. Passed explicitly (not inherited) so the
# DNS-rebinding protection is testable and survives wiring changes (RESEARCH Pitfall B). Not operator-extensible
# this phase (D-11 discretion): the localhost-only binding needs nothing more.
LOCALHOST_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
)

# uvicorn's records are left to the root ExeLogHandler on the standard error stream (R2): its startup and access lines
# print in the wrapper look ("[*] ...") and can never reach stdout (SEC-08). The three loggers keep no handler of
# their own and propagate. D-30: the startup and access lines are -v detail, so uvicorn sits at WARNING by default
# (a bind error still prints) and at INFO from -v; it never goes to DEBUG at any verbosity (RESEARCH A6). The access
# line carries the request line only (no headers, no bodies).
def uvicorn_log_config(verbose: int) -> dict[str, Any]:
    """The logging configuration handed to uvicorn for a -v count (D-30).

    Args:
        verbose: The -v count; 0 keeps uvicorn at WARNING, anything above at INFO.

    Returns:
        A dictConfig dict with uvicorn's three loggers propagating to the root handler, and no handler of their own.
    """
    level = "INFO" if verbose >= 1 else "WARNING"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {
            "uvicorn": {"level": level, "propagate": True},
            "uvicorn.error": {"level": level, "propagate": True},
            "uvicorn.access": {"level": level, "propagate": True},
        },
    }


# The INFO-level configuration the test harness (tests/http_harness.py) serves with.
UVICORN_LOG_CONFIG = uvicorn_log_config(1)


def run_stdio(server: MCPServer[ExegolPort]) -> None:
    """Serve over stdio. Blocks until the client disconnects."""
    server.run("stdio")


def build_http_app(
    server: MCPServer[ExegolPort], auth: str, secret: str | None, *, allow_no_auth: bool = False
) -> ASGIApp:
    """Return the ASGI app to serve over HTTP: the SDK streamable-HTTP app, bearer-gated when --auth bearer.

    The SDK app carries an explicit localhost TransportSecuritySettings (SEC-03, D-11) so a foreign Host/Origin is
    refused by the SDK middleware before BearerGate is even reached.

    Args:
        server: The MCP server (serves both eras).
        auth: "bearer" or "none".
        secret: The bearer secret, loaded once at startup; required when auth is "bearer".
        allow_no_auth: The deliberate no-auth opt-in (SEC-04, D-13). When auth is "none" the unauthenticated app is
            returned ONLY if this is true; otherwise the build fails closed.

    Raises:
        ValueError: When auth is "bearer" but no secret was provided.
        RuntimeError: When auth is "none" but allow_no_auth is false (fail closed, SEC-04/D-13).
    """
    app: ASGIApp = server.streamable_http_app(
        streamable_http_path=MCP_PATH, host=HTTP_HOST, transport_security=LOCALHOST_TRANSPORT_SECURITY
    )
    if auth == "bearer":
        if not secret:
            raise ValueError("Bearer auth is enabled but no secret was loaded")
        return BearerGate(app, secret)
    # auth == "none": fail closed. Unauthenticated HTTP is only reachable through the explicit config opt-in.
    if not allow_no_auth:
        raise RuntimeError(
            "Refusing to start unauthenticated HTTP: set 'security.allow_no_auth: true' in config.yml to allow it "
            "(trusted environments only). There is no CLI flag for no-auth (SEC-04, D-12/D-13)."
        )
    return app


class _AnnouncingServer(uvicorn.Server):
    """uvicorn's own server, which calls `on_bound` once the port is bound and the application has started.

    uvicorn sets `started` only after the socket is bound and its application lifespan is up, and it exits from
    `startup()` on a bind error, so the callback never runs for a port clash (GR-IN-01).
    """

    def __init__(self, config: uvicorn.Config, on_bound: Callable[[], None] | None) -> None:
        super().__init__(config)
        self._on_bound = on_bound

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started and self._on_bound is not None:
            self._on_bound()


def run_http(
    server: MCPServer[ExegolPort],
    port: int,
    auth: str,
    secret: str | None,
    *,
    allow_no_auth: bool = False,
    verbose: int = 0,
    on_bound: Callable[[], None] | None = None,
) -> None:
    """Serve over streamable HTTP on 127.0.0.1:<port>/mcp. Blocks until interrupted.

    When auth is "none" the server starts only if allow_no_auth is true (SEC-04/D-13); otherwise build_http_app
    fails closed before anything is served. The -v count decides how much of uvicorn's own output prints (D-30).

    The no-auth warning is the caller's to print, once, with the other start lines (it names the config key that
    allowed the start). `on_bound` runs once the port is actually bound, so the "serving on" line and the first-run
    client config are never printed for a start that did not bind (GR-IN-01). A start that never binds exits with
    uvicorn's own status after its bind error, as uvicorn.run does.

    Args:
        server: The MCP server.
        port: The port to bind on 127.0.0.1.
        auth: "bearer" or "none".
        secret: The bearer secret; required when auth is "bearer".
        allow_no_auth: The deliberate no-auth opt-in (SEC-04, D-13).
        verbose: The -v count.
        on_bound: Called once, after the port is bound; not called when the server fails to start.
    """
    app = build_http_app(server, auth, secret, allow_no_auth=allow_no_auth)
    config = uvicorn.Config(app, host=HTTP_HOST, port=port, log_config=uvicorn_log_config(verbose))
    http_server = _AnnouncingServer(config, on_bound)
    try:
        http_server.run()
    except KeyboardInterrupt:
        pass
    if not http_server.started:
        sys.exit(STARTUP_FAILURE)
