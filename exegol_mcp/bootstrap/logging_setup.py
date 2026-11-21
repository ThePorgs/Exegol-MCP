"""Process-wide logging on the standard error stream only (SEC-08).

In stdio mode the protocol owns the process's output stream, so every diagnostic, whether it comes from
exegol-mcp, the MCP SDK or the Exegol SDK, must be written to the standard error stream. Two calls make this true:

- route_logging_to_stderr() installs the single stderr handler and the levels. It imports no Exegol SDK module, so
  the entry point calls it first, before any SDK module is imported. Exegol's import-time logging.basicConfig then
  finds a root handler and never binds one to stdout, and a D-15 self-check refusal raised before the SDK is
  imported is one stderr line (WR-13).
- configure_stderr_logging() imports the SDK, points its console at the standard error stream, applies its
  verbosity, and re-applies the same handler and levels. The entry point calls it after the self-check, before
  any transport starts serving.

The same setup is used in HTTP mode, where it is simply the conventional place for diagnostics.
"""

import logging
import sys
from typing import TextIO

try:
    from rich.errors import MarkupError
    from rich.text import Text
except ImportError:  # pragma: no cover - rich ships with the Exegol SDK
    Text = None  # type: ignore[assignment,misc]
    MarkupError = ValueError  # type: ignore[assignment,misc]

_PACKAGE_LOGGER = "exegol_mcp"
_SSE_STARLETTE_LOGGER = "sse_starlette"
_FORMAT = "%(levelname)s %(name)s: %(message)s"

# The Exegol SDK console is redirected once per process; later calls only re-apply verbosity.
_sdk_logger_name: str | None = None


class StderrHandler(logging.StreamHandler):
    """A StreamHandler that always writes to the current sys.stderr.

    The stream is looked up on every write instead of being bound at construction, so the handler follows any
    later replacement of sys.stderr and can never be pointed at another stream.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        logging.Handler.__init__(self, level)

    @property  # type: ignore[override]
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        # setStream() and StreamHandler internals may assign a stream: the standard error stream always wins.
        pass


class _SdkMarkupFormatter(logging.Formatter):
    """Plain-text formatter that strips the rich markup the Exegol SDK embeds in its log messages."""

    def __init__(self, fmt: str, sdk_logger_name: str | None) -> None:
        super().__init__(fmt)
        self._sdk_logger_name = sdk_logger_name

    def format(self, record: logging.LogRecord) -> str:
        if self._sdk_logger_name is not None and record.name == self._sdk_logger_name and Text is not None:
            message = record.getMessage()
            try:
                message = Text.from_markup(message).plain
            except MarkupError:
                pass
            # Format a copy: other handlers (pytest's capture, for one) still receive the original record.
            record = logging.makeLogRecord({**record.__dict__, "msg": message, "args": None})
        return super().format(record)


def _root_level(verbose: int) -> int:
    if verbose <= 0:
        return logging.WARNING
    if verbose == 1:
        return logging.INFO
    return logging.DEBUG


def route_logging_to_stderr(verbose: int = 0) -> None:
    """Route every diagnostic of the process to the standard error stream, without importing the Exegol SDK (SEC-08).

    Replaces the root logger's handlers with a single StderrHandler and sets levels from the -v count: the root
    logger at WARNING/INFO/DEBUG for 0/1/2+ and exegol-mcp's own loggers at INFO (DEBUG with -v). sse_starlette is
    capped at the root level but never below INFO, because at DEBUG it logs every streamed tool result body.

    The entry point calls it first, before any Exegol SDK module is imported. Exegol's import-time
    logging.basicConfig then finds a root handler and does nothing, so it never binds a handler to stdout, and a
    D-15 self-check refusal raised before the SDK is imported reaches the operator as one stderr line (WR-13).
    Calling it again only re-applies the handler and the levels.

    Args:
        verbose: The -v/--verbose count from the command line.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = StderrHandler()
    handler.setFormatter(_SdkMarkupFormatter(_FORMAT, _sdk_logger_name))
    root.addHandler(handler)
    root.setLevel(_root_level(verbose))
    logging.getLogger(_PACKAGE_LOGGER).setLevel(logging.DEBUG if verbose > 0 else logging.INFO)
    # sse_starlette logs every outgoing SSE chunk, i.e. every tool result body, at DEBUG: it only duplicates tool
    # output into stderr (G-03-1). The max matters: root's handlers do not re-check the root logger's level, so a
    # plain INFO while root is at WARNING would make -v0 louder than it is. The parent covers sse_starlette.sse.
    logging.getLogger(_SSE_STARLETTE_LOGGER).setLevel(max(_root_level(verbose), logging.INFO))


def configure_stderr_logging(verbose: int = 0) -> None:
    """Route every diagnostic of the process, the Exegol SDK's console included, to the standard error stream (SEC-08).

    Imports the Exegol SDK, points its console at the standard error stream (first call only), applies the -v count
    to the SDK logger through the SDK's own mapping, then installs the handler and levels with
    route_logging_to_stderr(), whose formatter now knows the SDK logger and strips its rich markup. The entry point
    calls it after route_logging_to_stderr() and the D-15 self-check, before any transport starts serving; calling
    it again only re-applies the levels and the handler.

    Args:
        verbose: The -v/--verbose count from the command line.
    """
    global _sdk_logger_name
    # Imported here: the adapter is the only package allowed to import exegol. Importing it runs the SDK's
    # module-level imports, so the entry point runs the D-15 self-check first (WR-13).
    from exegol_mcp.adapter import sdk_headless

    if _sdk_logger_name is None:
        _sdk_logger_name = sdk_headless.route_sdk_output_to_stderr()
    sdk_headless.set_sdk_verbosity(verbose)
    route_logging_to_stderr(verbose)
