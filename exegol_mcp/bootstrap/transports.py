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

import logging

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp

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

# uvicorn installs its own logging handlers when it starts; keep them on the standard error stream so an access
# log line can never reach stdout (which would be the protocol stream in stdio mode, and is diagnostics in HTTP).
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(levelname)s %(name)s: %(message)s"}},
    "handlers": {"stderr": {"class": "logging.StreamHandler", "stream": "ext://sys.stderr", "formatter": "default"}},
    "loggers": {
        "uvicorn": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
    },
}


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


def run_http(
    server: MCPServer[ExegolPort], port: int, auth: str, secret: str | None, *, allow_no_auth: bool = False
) -> None:
    """Serve over streamable HTTP on 127.0.0.1:<port>/mcp. Blocks until interrupted.

    When auth is "none" the server starts only if allow_no_auth is true (SEC-04/D-13); otherwise build_http_app
    fails closed before anything is served.
    """
    app = build_http_app(server, auth, secret, allow_no_auth=allow_no_auth)
    if auth == "none":
        logging.getLogger(__name__).warning(
            "Authentication is DISABLED (mode: none). Use only in trusted environments."
        )
    uvicorn.run(app, host=HTTP_HOST, port=port, log_config=UVICORN_LOG_CONFIG)
