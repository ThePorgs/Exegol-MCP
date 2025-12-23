import argparse


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="exegol-mcp", description="Exegol MCP server")
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbosity level (-v for verbose, -vv for advanced, -vvv for debug)"
    )
    parser.add_argument(
        "--auth",
        choices=["none", "bearer"],
        default="bearer",
        help="Authentication mode for the server, when in http mode (default: bearer)",
    )
    parser.add_argument(
        "-pc", "--print-config",
        action="store_true",
        help="Print the MCP configuration and exit",
    )
    parser.add_argument(
        "-p", "--port",
        action="store",
        default=2187,
        type=int,
        help="Choose the port of the HTTP server (default: 2187)",
    )
    parser.add_argument(
        "-t", "--type",
        choices=["http", "stdio"],
        default="http",
        help="Choose the type of server to run (default: http)",
    )
    return parser.parse_args(argv)
