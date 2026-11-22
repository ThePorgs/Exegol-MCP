"""The light SDK logging hooks: console routing and verbosity of the Exegol SDK's ExeLog (SEC-08, D-01, D-02).

Called by bootstrap.logging_setup (through configure_stderr_logging), by sdk_headless.patch_exegol_sdk and by the
`version` command. It imports nothing from the SDK but exegol.utils.ExeLog, the light subset D-19 allows, so the
`version` path can use it without loading the heavy SDK modules. It never builds a console of its own: the process's
one shared stderr console is handed in, so the SDK and exegol-mcp's handler render alike.
"""

import logging

from exegol.utils.ExeLog import ExeLog, logger
from rich.console import Console


def route_console(console: Console) -> None:
    """Make the SDK print and log through the given console.

    The SDK's root RichHandler captured the original stdout console at import time, so every root handler that still
    holds the previous SDK console is repointed too. Idempotent.

    Args:
        console: The process's shared stderr console.
    """
    previous = ExeLog.console
    ExeLog.console = console
    for handler in logging.getLogger().handlers:
        if getattr(handler, "console", None) is previous:
            handler.console = console  # type: ignore[attr-defined]


def set_verbosity(verbose: int) -> None:
    """Apply the operator's -v count to the SDK's logger, with the SDK's own level mapping (D-02)."""
    ExeLog.setVerbosity(verbose)


def sdk_logger_name() -> str:
    """Return the name of the SDK's logger, the only one whose records carry rich markup."""
    return logger.name


def set_critical_raise() -> None:
    """Make the SDK logger's critical raise RuntimeError instead of exiting the process (D-05).

    The wrapper's critical exits the process; inside the server or the `version` command that must raise instead.
    The method is per logger instance, so it is set on the SDK logger the SDK and the mirrors log through.
    """
    logger.setCriticalMethod("raise")
