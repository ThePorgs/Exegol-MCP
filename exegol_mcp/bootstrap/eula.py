"""Headless-safe Exegol EULA resolution (D-05/D-06).

The EULA state is the one the Exegol wrapper uses: LocalDatastore.is_eula_accepted() / update_eula() in the
datastore shared with the wrapper, so accepting in either counts for both. It is resolved once, at startup, in
this order:

1. --accept-eula: the operator accepts for this run. main() also hands the flag to the SDK
   (sdk_headless.set_accept_eula, which sets OptionKey.ACCEPT_EULA). Nothing is written to the datastore.
2. A prior acceptance in the shared datastore (by the wrapper or an earlier exegol-mcp run on a terminal).
3. A terminal: a human is asked on the standard error stream, and an acceptance is recorded with
   LocalDatastore.update_eula(True). Never in stdio mode, where stdin and stdout carry the protocol.
4. Anything else fails startup with the D-05 message.

D-06: no MCP tool, chat phrase or other agent-reachable path can accept or record the EULA. Only the three
operator paths above can, and this module is the only caller of update_eula in exegol-mcp.
"""

import sys
from argparse import Namespace
from collections.abc import Callable
from typing import Literal

from rich.console import Console
from rich.prompt import Confirm

from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.port import HostProbe

EULA_URL = "https://docs.exegol.com/legal/eula"

EULA_REQUIRED_MESSAGE = (
    "The Exegol EULA has not been accepted on this machine, and exegol-mcp cannot ask for it here (no terminal, "
    "or a stdio launch). Run `exegol-mcp` or `exegol` once in a terminal to read and accept it, or pass "
    f"--accept-eula to accept it for this run. EULA: {EULA_URL}"
)

EULA_DECLINED_MESSAGE = (
    f"The Exegol EULA was not accepted, so exegol-mcp cannot start. Read it at {EULA_URL}, then run exegol-mcp "
    "again in a terminal and accept it, or pass --accept-eula."
)

EulaOutcome = Literal["flag", "accepted", "prompted"]


class EulaNotAcceptedError(RuntimeError):
    """The Exegol EULA is not accepted and could not be accepted by an operator path (D-05)"""


def can_prompt(args: Namespace) -> bool:
    """True when a human can answer the EULA prompt: a terminal on stdin, and not the stdio protocol path."""
    if args.type == "stdio":
        # In stdio mode stdin and stdout are the MCP protocol stream: a prompt would read and corrupt it.
        return False
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def ask_on_stderr() -> bool:
    """Ask a human to accept the EULA, on a stderr console. Blocking: run it through run_blocking().

    This is not the SDK's ExegolRich prompt, which raises once patch_exegol_sdk() has run. The answer is read
    from sys.stdin directly, so nothing (not even the prompt text) is written to the standard output.
    """
    console = Console(stderr=True)
    console.print(f"[bold]Exegol EULA[/bold]: exegol-mcp drives Exegol, whose EULA must be accepted first: {EULA_URL}")
    return bool(
        Confirm.ask(
            "I confirm that I've read and accepted the EULA",
            console=console,
            default=False,
            stream=sys.stdin,
        )
    )


async def resolve_eula(
    probe: HostProbe,
    args: Namespace,
    prompt: Callable[[], bool] = ask_on_stderr,
) -> EulaOutcome:
    """Make sure the Exegol EULA is accepted before the server starts (D-05).

    Args:
        probe: Reads and records the EULA state in the datastore shared with the wrapper.
        args: The parsed command line (accept_eula, type).
        prompt: Asks a human on a terminal; returns True on acceptance. Replaced in tests.

    Returns:
        How the EULA is accepted: "flag" (--accept-eula), "accepted" (already in the datastore) or "prompted".

    Raises:
        EulaNotAcceptedError: With the D-05 guidance when no operator path accepted the EULA.
    """
    if args.accept_eula:
        # OptionKey.ACCEPT_EULA is set by main() through sdk_headless.set_accept_eula(args.accept_eula).
        return "flag"
    if await run_blocking(probe.is_eula_accepted):
        return "accepted"
    if not can_prompt(args):
        raise EulaNotAcceptedError(EULA_REQUIRED_MESSAGE)
    if not await run_blocking(prompt):
        raise EulaNotAcceptedError(EULA_DECLINED_MESSAGE)
    # A human accepted at the terminal: record it where the wrapper reads it (LocalDatastore.update_eula).
    await run_blocking(probe.update_eula, True)
    return "prompted"
