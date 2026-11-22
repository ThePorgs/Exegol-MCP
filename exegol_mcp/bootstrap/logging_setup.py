"""Process-wide logging on the standard error stream only, in the Exegol wrapper's ExeLog style (SEC-08, D-01).

In stdio mode the protocol owns the process's output stream, so every diagnostic, whether it comes from
exegol-mcp, the MCP SDK or the Exegol SDK, must be written to the standard error stream. Every record renders as one
line, "<prefix> <text>", with the wrapper's prefixes ("[*]" info, "[V]" verbose, "[A]" advanced, "[D]" debug, "[!]"
warning, "[-]" error, "[+]" success) through one rich Console(stderr=True, soft_wrap=True) that the SDK's own console
shares. Rich markup is parsed only for the SDK's logger, as the wrapper does; every other record's text is printed
literally, so operator text (paths, container names, config values) can never restyle a line or raise (D-01, D-20).
Two calls make this true:

- route_logging_to_stderr() installs the single stderr handler and the levels. It imports no Exegol SDK module, so
  the entry point calls it first, before any SDK module is imported. Exegol's import-time logging.basicConfig then
  finds a root handler and never binds one to stdout, and a D-15 self-check refusal raised before the SDK is
  imported is one stderr line (WR-13). The handler is rich-based rather than plain (D-21): it needs only rich, a
  declared dependency, so the pre-SDK handler and the post-SDK one render alike.
- configure_stderr_logging() imports the SDK glue, points the SDK console at the shared stderr console, applies the
  SDK verbosity and re-applies the same handler and levels. The entry point calls it after the self-check, before
  any transport starts serving.

The same setup is used in HTTP mode, where it is simply the conventional place for diagnostics.

One call is for the `version` action alone: cap_third_party_logs() keeps libraries other than exegol-mcp and the SDK to
warnings and errors while the report prints (D-27). A server start never calls it.
"""

import logging
import re

from rich.console import Console, ConsoleRenderable
from rich.errors import MarkupError
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import Traceback

_PACKAGE_LOGGER = "exegol_mcp"
_SSE_STARLETTE_LOGGER = "sse_starlette"

# ExeLog's own level numbers (exegol/utils/ExeLog.py: SUCCESS, VERBOSE, ADVANCED). Duplicated here because bootstrap
# code cannot import the SDK before the D-15 self-check ran; tests/test_exelog_handler.py pins them against the
# installed SDK, and the self-check names a renamed constant.
SUCCESS = 25
VERBOSE = 15
ADVANCED = 13

# The only logger the SDK logs through (exegol/utils/ExeLog.py: logger = logging.getLogger("main")). Its records are
# the only ones whose text may carry rich markup.
_SDK_LOGGER_NAME = "main"

# (threshold, prefix, style): the first row whose threshold the level reaches wins; below DEBUG still reads "[D]".
# The prefixes and styles are the ones ExeLog's own methods decorate their messages with.
_PREFIXES: tuple[tuple[int, str, str], ...] = (
    (logging.CRITICAL, "[!]", "bold red"),
    (logging.ERROR, "[-]", "bold red"),
    (logging.WARNING, "[!]", "bold orange3"),
    (SUCCESS, "[+]", "bold green"),
    (logging.INFO, "[*]", "bold blue"),
    (VERBOSE, "[V]", "bold blue"),
    (ADVANCED, "[A]", "bold yellow3"),
    (logging.DEBUG, "[D]", "bold yellow3"),
)

# The decoration ExeLog's methods put in front of every message: "[bold blue][*][/bold blue] text".
_EXELOG_PREFIX = re.compile(r"^\[(bold (?:blue|yellow3|orange3|red|green))\](\[[*VAD!+x-]\])\[/\1\] (.*)\Z", re.DOTALL)

# One shared console for the handler and the SDK. Rebuilt by every route_logging_to_stderr() call so that it re-reads
# the stream and the colour environment.
_console: Console | None = None


def level_prefix(levelno: int) -> Text:
    """Return the wrapper's styled prefix for a logging level, e.g. "[*]" for INFO.

    Args:
        levelno: A logging level number.

    Returns:
        The prefix as a styled Text, never parsed as markup.
    """
    for threshold, prefix, style in _PREFIXES:
        if levelno >= threshold:
            return Text(prefix, style=style)
    return Text(_PREFIXES[-1][1], style=_PREFIXES[-1][2])


def prefixed_line(levelno: int, *parts: str | Text, style: str | None = None) -> Text:
    """Build one wrapper-style line, the level prefix then the parts, without parsing any markup.

    Args:
        levelno: The logging level whose prefix opens the line.
        *parts: The text after the prefix; a str is taken literally, a Text keeps its own styling.
        style: Overrides the prefix style, for the wrapper's lines that use another colour.

    Returns:
        The assembled line, ready for Console.print().
    """
    prefix = level_prefix(levelno)
    if style is not None:
        prefix.style = style
    line = Text.assemble(prefix, " ")
    for part in parts:
        line.append_text(Text(part) if isinstance(part, str) else part)
    return line


class ExeLogHandler(RichHandler):
    """A RichHandler that prints each record as one wrapper-style line on a shared console.

    Both overrides exist because rich's stock handler is built for a log table: render_message parses the text as
    markup outside the handler's error guard (a stray closing tag in a path or container line would escape into
    the caller), and render pads every line to the console width and hard-wraps it inside a Table.
    """

    def __init__(self, console: Console) -> None:
        # No time, level or path columns: the prefix carries the level. Tracebacks stay plain text in the line.
        super().__init__(
            console=console,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        """Render a record as prefix and text, parsing markup only for the SDK's logger.

        Never raises: only the SDK's own text is parsed, and a MarkupError there falls back to the literal text.
        Every other logger's text, including exegol-mcp's own loggers that become ExeLog instances once the SDK is
        imported, prints exactly as typed (D-01).

        Args:
            record: The log record.
            message: The formatted message, still carrying ExeLog's decoration when an ExeLog logger made it.

        Returns:
            The styled line.
        """
        match = _EXELOG_PREFIX.match(message)
        if match is None:
            prefix = level_prefix(record.levelno)
            body = Text(message)
        else:
            prefix = Text(match.group(2), style=match.group(1))
            rest = match.group(3)
            if record.name == _SDK_LOGGER_NAME:
                try:
                    body = Text.from_markup(rest)
                except MarkupError:
                    body = Text(rest)
            else:
                body = Text(rest)
        if self.highlighter:
            body = self.highlighter(body)
        return Text.assemble(prefix, " ", body)

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Return the line as it is: no padding to the console width, no hard wrapping, one physical line."""
        return message_renderable


def make_stderr_console() -> Console:
    """Build the console every diagnostic goes through: standard error, never soft-wrapped into columns.

    Console(stderr=True) resolves sys.stderr on every write, so a replaced stream (capsys, a redirect) is followed.
    rich decides colour from the stream and honours NO_COLOR, FORCE_COLOR and TTY_COMPATIBLE (D-04).
    """
    return Console(stderr=True, soft_wrap=True)


def stderr_console() -> Console:
    """Return the process's shared stderr console, creating it on first use."""
    global _console
    if _console is None:
        _console = make_stderr_console()
    return _console


# D-02: the wrapper's ExeLog.setVerbosity mapping for exegol-mcp's own loggers (the SDK logger follows the same one).
# 3 and above is the single full-debug level.
_PACKAGE_LEVELS = {0: logging.INFO, 1: VERBOSE, 2: ADVANCED}
# D-16: third-party libraries (mcp, uvicorn, httpx, anyio) stay quiet until -vvv, which is DEBUG for everything.
_ROOT_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.INFO}


def _package_level(verbose: int) -> int:
    return _PACKAGE_LEVELS.get(max(verbose, 0), logging.DEBUG)


def _root_level(verbose: int) -> int:
    return _ROOT_LEVELS.get(max(verbose, 0), logging.DEBUG)


def route_logging_to_stderr(verbose: int = 0) -> None:
    """Route every diagnostic of the process to the standard error stream, without importing the Exegol SDK (SEC-08).

    Replaces the root logger's handlers with a single ExeLogHandler on a fresh shared stderr console and sets levels
    from the -v count: exegol-mcp's own loggers follow the wrapper's mapping (INFO, VERBOSE, ADVANCED, DEBUG for
    0/1/2/3+, D-02) and the root logger, which covers third-party libraries, is at WARNING/INFO/INFO/DEBUG (D-16).
    sse_starlette is capped at the root level but never below INFO, because at DEBUG it logs every streamed tool
    result body.

    The entry point calls it first, before any Exegol SDK module is imported. Exegol's import-time
    logging.basicConfig then finds a root handler and does nothing, so it never binds a handler to stdout, and a
    D-15 self-check refusal raised before the SDK is imported reaches the operator as one stderr line (WR-13).
    Calling it again only re-applies the handler and the levels.

    Args:
        verbose: The -v/--verbose count from the command line.
    """
    global _console
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    _console = make_stderr_console()
    root.addHandler(ExeLogHandler(_console))
    root.setLevel(_root_level(verbose))
    logging.getLogger(_PACKAGE_LOGGER).setLevel(_package_level(verbose))
    # sse_starlette logs every outgoing SSE chunk, i.e. every tool result body, at DEBUG: it only duplicates tool
    # output into stderr (G-03-1). The max matters: root's handlers do not re-check the root logger's level, so a
    # plain INFO while root is at WARNING would make -v0 louder than it is. The parent covers sse_starlette.sse.
    logging.getLogger(_SSE_STARLETTE_LOGGER).setLevel(max(_root_level(verbose), logging.INFO))


class _ReportRecordsFilter(logging.Filter):
    """Lets a record through when exegol-mcp or the Exegol SDK made it, or when it is a warning or worse (D-27)."""

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if name == _PACKAGE_LOGGER or name.startswith(_PACKAGE_LOGGER + ".") or name == _SDK_LOGGER_NAME:
            return True
        return record.levelno >= logging.WARNING


def cap_third_party_logs() -> None:
    """Keep every library except exegol-mcp and the Exegol SDK to warnings and errors, for the `version` action (D-27).

    While `version` runs, libraries such as keyring, docker-py, urllib3, requests, httpx and asyncio print only warnings
    and errors, at every -v count, so the -vvv report never carries registry host names or other third-party detail
    and is safe to paste into a public bug report. D-16 keeps third-party DEBUG for a server start: the server never
    calls this, and the next route_logging_to_stderr() call drops the cap because it builds a new handler.

    Two layers, because a third-party logger may carry a level of its own: the root logger goes to WARNING (a logger
    without a level then makes no record below it), and every root handler gets a filter that drops records of other
    loggers below WARNING (a logger set to DEBUG explicitly still reaches the handler otherwise). The calls are
    idempotent.
    """
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in root.handlers:
        if not any(isinstance(existing, _ReportRecordsFilter) for existing in handler.filters):
            handler.addFilter(_ReportRecordsFilter())


def configure_stderr_logging(verbose: int = 0) -> None:
    """Route every diagnostic of the process, the Exegol SDK's console included, to the standard error stream (SEC-08).

    Installs the handler and levels with route_logging_to_stderr(), then imports the SDK glue, points the SDK's
    console at the same shared stderr console and applies the -v count to the SDK logger through the SDK's own
    mapping. The entry point calls it after route_logging_to_stderr() and the D-15 self-check, before any transport
    starts serving; calling it again only re-applies the levels and the handler.

    Args:
        verbose: The -v/--verbose count from the command line.
    """
    route_logging_to_stderr(verbose)
    # Imported here: the adapter is the only package allowed to import exegol. Importing it runs the SDK's
    # module-level imports, so the entry point runs the D-15 self-check first (WR-13).
    from exegol_mcp.adapter import sdk_logging

    sdk_logging.route_console(stderr_console())
    sdk_logging.set_verbosity(verbose)
    # D-05: the earliest point the SDK is importable in main() (after the self-check), so no exegol-mcp code path can
    # reach ExeLog's exiting critical, which would kill the server or the `version` command.
    sdk_logging.set_critical_raise()
