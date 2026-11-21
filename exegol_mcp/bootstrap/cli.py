import argparse
import sys

# The stable launch interface (COMP-01/COMP-02): HTTP always binds localhost, on this port and path by default.
HTTP_HOST = "127.0.0.1"
MCP_PATH = "/mcp"
DEFAULT_PORT = 2187

# D-12/SEC-04: `--auth none` is removed from the CLI. The only route to an unauthenticated server is a deliberate
# `security.allow_no_auth: true` edit in config.yml. argparse's own choices error is terse, so we intercept the
# removed value before parsing and exit with a message that points at the config key and the zero-setup bearer flow.
_NO_AUTH_GUIDANCE = (
    "error: --auth no longer accepts 'none'. HTTP is bearer-authenticated by default with a zero-setup, "
    "auto-generated secret. To run unauthenticated (trusted environments only), set 'security.allow_no_auth: true' "
    "in config.yml instead; there is no CLI flag for it."
)


def _reject_auth_none(argv: list[str]) -> None:
    """Exit non-zero with guidance if `--auth none` / `--auth=none` appears in argv (D-12)."""
    for i, tok in enumerate(argv):
        if tok == "--auth=none" or (tok == "--auth" and i + 1 < len(argv) and argv[i + 1] == "none"):
            print(f"exegol-mcp: {_NO_AUTH_GUIDANCE}", file=sys.stderr)
            raise SystemExit(2)


def parse_args(argv: list[str] | None = None):
    _reject_auth_none(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="exegol-mcp", description="Exegol MCP server")
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbosity level (-v for verbose, -vv for advanced, -vvv for debug)"
    )
    parser.add_argument(
        "--auth",
        choices=["bearer"],
        default=None,
        help=(
            "Authentication mode for the server, when in http mode "
            "(default: bearer; no-auth is config-only: security.allow_no_auth in config.yml)"
        ),
    )
    parser.add_argument(
        "-pc", "--print-config",
        action="store_true",
        help="Print the MCP configuration and exit",
    )
    parser.add_argument(
        "-p", "--port",
        action="store",
        default=None,
        type=int,
        help="Choose the port of the HTTP server (default: 2187)",
    )
    parser.add_argument(
        "-t", "--type",
        choices=["http", "stdio"],
        default=None,
        help="Choose the type of server to run (default: http)",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Accept the Exegol EULA non-interactively, for pipelines and stdio launches without a terminal",
    )
    parser.add_argument(
        "--no-update-check",
        action="store_true",
        help="Do not check PyPI for a newer exegol-mcp release (offline or air-gapped use)",
    )
    return parser.parse_args(argv)
