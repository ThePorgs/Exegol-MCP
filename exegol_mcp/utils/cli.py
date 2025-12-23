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
        choices=["bearer", "none"],
        default="bearer",
        help="Authentication mode for the HTTP server (default: bearer)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the mcp configuration template and exit",
    )
    parser.add_argument(
        "-p","--port",
        action="store",
        default=8000,
        type=int,
        help="Choose the port of the HTTP server (default: 8000)",
    )
    parser.add_argument(
        "--type",
        choices=["http", "stdio"],
        default="http",
        help="# TODO",
    )
    return parser.parse_args(argv)
