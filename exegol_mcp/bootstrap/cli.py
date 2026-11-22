"""The command line: the stable launch interface plus a help that reads like the Exegol wrapper's (D-33).

The parser mirrors the wrapper's ExegolArgParse: help and messages print through rich consoles with markup, the layout
is argparse's own (RawTextHelpFormatter), and the description, group titles and examples are authored here. Markup
comes only from this module's own strings, never from the command line. The module stays SDK-free.
"""

import argparse
import sys
from typing import IO, TYPE_CHECKING, Any, cast

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from exegol_mcp.bootstrap.logging_setup import stderr_console

if TYPE_CHECKING:
    from typing import NoReturn

    from _typeshed import SupportsWrite

# The stable launch interface (COMP-01/COMP-02): HTTP always binds localhost, on this port and path by default.
HTTP_HOST = "127.0.0.1"
MCP_PATH = "/mcp"
DEFAULT_PORT = 2187

# pyproject's Documentation URL; shared by the help and the `version` report.
DOCS_URL = "https://docs.exegol.com/mcp/"

# D-12/SEC-04: `--auth none` is removed from the CLI. The only route to an unauthenticated server is a deliberate
# `security.allow_no_auth: true` edit in config.yml. argparse's own choices error is terse, so we intercept the
# removed value before parsing and exit with a message that points at the config key and the zero-setup bearer flow.
_NO_AUTH_GUIDANCE = (
    "error: --auth no longer accepts 'none'. HTTP is bearer-authenticated by default with a zero-setup, "
    "auto-generated secret. To run unauthenticated (trusted environments only), set 'security.allow_no_auth: true' "
    "in config.yml instead; there is no CLI flag for it."
)

_DESCRIPTION = (
    "Exegol MCP gives AI agents the Exegol toolbox through the Model Context Protocol, over streamable HTTP or "
    "stdio.\n\n"
    f"[bold magenta]Exegol MCP documentation:[/bold magenta] [underline magenta]{DOCS_URL}[/underline magenta]"
)

# Example description -> command, in the wrapper's markup: the flags and the action are bright_blue inside the italic
# command, as in the wrapper's own usage examples.
_EXAMPLES = (
    ("Start the server over HTTP on 127.0.0.1:2187", "exegol-mcp"),
    ("Print the MCP client config, bearer token included", "exegol-mcp [bright_blue]--print-config[/bright_blue]"),
    (
        "Print the stdio entry for an MCP client",
        "exegol-mcp [bright_blue]--type stdio --print-config[/bright_blue]",
    ),
    (
        "Show the versions and debug information for a bug report",
        "exegol-mcp [bright_blue]version -vvv[/bright_blue]",
    ),
)


def _is_auth_option(token: str) -> bool:
    """Whether a command-line token names the hidden --auth option, in full or as an abbreviation argparse accepts.

    argparse resolves any unambiguous prefix of a long option, so `--au`, `--aut` and `--auth` all reach --auth
    (`--a` is ambiguous with --accept-eula and is an argparse error of its own). The `=value` part is not examined.
    """
    name = token.split("=", 1)[0]
    return len(name) >= len("--au") and "--auth".startswith(name)


def _reject_auth_none(argv: list[str]) -> None:
    """Exit non-zero with guidance if `--auth none` / `--auth=none` (or an abbreviation of --auth) is in argv (D-12)."""
    for i, tok in enumerate(argv):
        attached = "=" in tok and _is_auth_option(tok) and tok.split("=", 1)[1] == "none"
        separate = "=" not in tok and _is_auth_option(tok) and i + 1 < len(argv) and argv[i + 1] == "none"
        if attached or separate:
            # Assembled as a Text so nothing in the guidance is read as markup; the plain words are unchanged.
            guidance = Text.assemble(("exegol-mcp: error:", "bold red"), _NO_AUTH_GUIDANCE.removeprefix("error:"))
            stderr_console().print(guidance)
            raise SystemExit(2)


def _group_title(name: str) -> str:
    """Return the wrapper's group title markup, e.g. "Server options"."""
    return f"[bright_blue]{name}[/bright_blue] [blue]options[/blue]"


def _examples_epilog() -> str:
    """Build the wrapper's "Examples:" section: one line per example, the commands aligned in a column."""
    # The padding is computed on the text without its markup tags, as the wrapper's richLen does.
    widest = max(Text.from_markup(description).cell_len for description, _ in _EXAMPLES)
    lines = ["[blue]Examples:[/blue]"]
    for description, command in _EXAMPLES:
        padding = " " * (widest - Text.from_markup(description).cell_len + 2)
        lines.append(f"  {description}:{padding}[i]{command}[/i]")
    return "\n".join(lines) + "\n"


def _parser_options() -> dict[str, Any]:
    """Return the extra ArgumentParser keywords: argparse colours help on its own from Python 3.14, rich decides (D-04)."""
    return {"color": False} if sys.version_info >= (3, 14) else {}


class ExegolMcpArgParse(argparse.ArgumentParser):
    """An argparse parser that prints through rich consoles with markup, like the wrapper's ExegolArgParse."""

    def _print_message(self, message: str, file: "SupportsWrite[str] | None" = None) -> None:
        """Print argparse's help, usage or message through rich, which alone decides colour (D-04)."""
        if not message:
            return
        stream = file if file is not None else sys.stderr
        console = stderr_console() if stream is sys.stderr else Console(file=cast(IO[str], stream), soft_wrap=True)
        # argparse already laid the text out and wrapped it, so rich must not wrap it again; markup comes only from
        # this module's own strings, never from the command line.
        console.print(message, end="", markup=True, highlight=False, emoji=False)

    def error(self, message: str) -> "NoReturn":
        """Print the usage and the error through the console, then exit 2; the wording stays argparse's.

        The message carries command-line text (a bad choice, a bad number), which must print literally and never
        raise, so it is escaped before it meets the markup renderer.
        """
        self.print_usage(sys.stderr)
        self.exit(2, f"[bold red]{self.prog}: error:[/bold red] {escape(message)}\n")


def build_parser() -> ExegolMcpArgParse:
    """Build the command-line parser: the wrapper-style help with its groups and examples (D-33)."""
    parser = ExegolMcpArgParse(
        prog="exegol-mcp",
        description=_DESCRIPTION,
        epilog=_examples_epilog(),
        formatter_class=argparse.RawTextHelpFormatter,
        **_parser_options(),
    )
    # argparse's two default groups carry the positional and the -h/-v options; retitled as the wrapper does.
    parser._positionals.title = "[blue]Action[/blue]"
    parser._optionals.title = _group_title("Output")
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbosity level (-v for verbose, -vv for advanced, -vvv for debug)"
    )
    server = parser.add_argument_group(_group_title("Server"))
    server.add_argument(
        "-t", "--type",
        choices=["http", "stdio"],
        default=None,
        help="Choose the type of server to run (default: http)",
    )
    server.add_argument(
        "-p", "--port",
        action="store",
        default=None,
        type=int,
        help="Choose the port of the HTTP server (default: 2187)",
    )
    # D-32: bearer is the only choice on the command line, so the option is not offered in the help. Existing v1 client
    # entries still pass `--auth bearer` and must keep working, and `--auth none` keeps its refusal guidance (D-12).
    server.add_argument(
        "--auth",
        choices=["bearer"],
        default=None,
        help=argparse.SUPPRESS,
    )
    client = parser.add_argument_group(_group_title("Client config"))
    client.add_argument(
        "-pc", "--print-config",
        action="store_true",
        help="Print the MCP configuration and exit",
    )
    startup = parser.add_argument_group(_group_title("Startup"))
    startup.add_argument(
        "--accept-eula",
        action="store_true",
        help="Accept the Exegol EULA non-interactively, for pipelines and stdio launches without a terminal",
    )
    startup.add_argument(
        "--no-update-check",
        action="store_true",
        help="Do not check PyPI for a newer exegol-mcp release (offline or air-gapped use)",
    )
    # `version` is the only spelling, like the wrapper's `exegol version` (D-25): there is no --version flag, which also
    # keeps --v, --ve and --ver unambiguous abbreviations of --verbose. An optional positional keeps every existing
    # command line parsing as before and combines with -v in any position. argparse's own version action prints one
    # line and exits by itself, and subparsers would drop a -v count given before the action (RESEARCH Pattern 3).
    parser.add_argument(
        "action",
        nargs="?",
        choices=["version"],
        default=None,
        help="version: print the versions (and, with -vvv, debug information for bug reports), then exit",
    )
    return parser


def parse_args(argv: list[str] | None = None):
    _reject_auth_none(sys.argv[1:] if argv is None else argv)
    return build_parser().parse_args(argv)
