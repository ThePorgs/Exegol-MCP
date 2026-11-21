"""Error translation at the MCP boundary: tool handlers, resource templates and the resources/list handler.

mcp 2.x treats any exception other than ToolError as a crash: the model only sees "Error executing tool <name>"
and the real message stays in the server log. Services follow the project convention of ValueError for bad
input or an unknown name and RuntimeError for a bad state (container not running, Docker unavailable). Those
are anticipated failures the agent must read to recover, so every tool handler runs its service call inside the
async context manager tool_errors(<lifespan context>), which re-raises them as ToolError with the original message.

On any other failure the boundary first asks the lifespan context to re-check Docker and the EULA (D-08), so a
host that changed after startup (daemon gone, EULA revoked) is reported in words instead of as a crash. Bad input
(ValueError) skips that re-check: it is the agent's to fix and needs no Docker round-trip. Anything that is
neither anticipated nor explained by the re-check is still a crash.

The Exegol SDK guard errors (an unset SDK parameter per D-09, a refused prompt, a broken patch point per D-15) are
anticipated failures too and reach the agent in words.

The same ladder serves resources (D-04, RES-03); only the raised types differ. resource_errors() raises
ResourceNotFoundError for bad input (JSON-RPC -32602, which the spec asks for a nonexistent resource) and
ResourceError for every other anticipated failure (-32603 with the message). protocol_errors() raises MCPError with
those two codes directly, for handlers installed on the low-level server, where any other exception would reach the
client as a generic "Internal server error".
"""

import logging
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, runtime_checkable

from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError, ToolError
from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS

from exegol_mcp.adapter.errors import HeadlessPromptError, PatchPointError, UnknownParameterError

_log = logging.getLogger(__name__)

# Raised by the headless SDK setup when the SDK would guess, prompt, or run on a moved internal. They are mapped
# explicitly: UnknownParameterError is an AttributeError, so the RuntimeError branch alone would let it crash.
_SDK_GUARD_ERRORS = (UnknownParameterError, HeadlessPromptError, PatchPointError)


@runtime_checkable
class ReadinessDiagnoser(Protocol):
    """Explains a failed tool call by the host state that can change after startup (D-08)."""

    async def diagnose(self) -> str | None:
        """Re-check the host state that can change after startup (Docker, the EULA).

        Must never prompt and never change that state.

        Returns:
            A message the agent can act on when the host is no longer ready, or None when it still is.
        """
        ...


class ToolBoundary:
    """Context manager that turns anticipated and diagnosable failures into the boundary's error types.

    Built by tool_errors(), resource_errors() or protocol_errors() and used only with "async with". It has no sync
    form on purpose: the readiness re-check is awaited on exit, and a sync "with" would let a handler skip it (D-08).

    Args:
        diagnoser: Explains a failure by the host state that changed after startup, or None.
        invalid: Builds the error raised for bad input (a ValueError), from its message.
        failure: Builds the error raised for every other anticipated failure, from its message.
        passthrough: Exception types that are already mapped and leave the boundary unchanged.
    """

    def __init__(
        self,
        diagnoser: ReadinessDiagnoser | None,
        *,
        invalid: Callable[[str], Exception] = ToolError,
        failure: Callable[[str], Exception] = ToolError,
        passthrough: tuple[type[BaseException], ...] = (ToolError,),
    ) -> None:
        self._diagnoser = diagnoser
        self._invalid = invalid
        self._failure = failure
        self._passthrough = passthrough

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Cancellation and other BaseExceptions, an already-mapped error, and a clean exit all pass through untouched.
        if not isinstance(exc, Exception) or isinstance(exc, self._passthrough):
            return
        # 1. Bad input: the agent must read it, and no Docker round-trip can explain it.
        if isinstance(exc, ValueError):
            raise self._invalid(str(exc)) from exc
        # 2. Any other failure: a host that changed after startup is the root cause worth reporting (D-08).
        diagnosis = await self._diagnose()
        if diagnosis is not None:
            raise self._failure(diagnosis) from exc
        # 3. An Exegol SDK guard tripped: an unset parameter (D-09), a refused prompt, a broken patch point (D-15).
        if isinstance(exc, _SDK_GUARD_ERRORS):
            raise self._failure(str(exc)) from exc
        # 4. A bad state the service named itself.
        if isinstance(exc, RuntimeError):
            raise self._failure(str(exc)) from exc
        # 5. Anything else is a crash and propagates unchanged.

    async def _diagnose(self) -> str | None:
        if self._diagnoser is None:
            return None
        try:
            return await self._diagnoser.diagnose()
        except Exception:  # noqa: BLE001  # a failing diagnosis must never hide the original error
            _log.debug("Tool-time readiness re-check failed", exc_info=True)
            return None


def tool_errors(context: object | None = None) -> ToolBoundary:
    """Build the tool boundary for one handler call.

    Args:
        context: The lifespan context (ctx.request_context.lifespan_context). When it is a ReadinessDiagnoser,
            failures other than bad input are first explained by its diagnose() (D-08).

    Returns:
        A ToolBoundary for the handler to enter with "async with". On exit it raises ToolError (the cause is kept
        with from e) for a ValueError, for any failure the diagnoser explains, for an Exegol SDK guard error, and
        for a RuntimeError.
    """
    return ToolBoundary(_diagnoser_of(context))


def resource_errors(context: object | None = None) -> ToolBoundary:
    """Build the boundary for one resource template read.

    Args:
        context: The lifespan context (ctx.request_context.lifespan_context), used as in tool_errors().

    Returns:
        A ToolBoundary that raises ResourceNotFoundError (JSON-RPC -32602) for a ValueError, and ResourceError
        (-32603 with the message) for a diagnosed failure, an Exegol SDK guard error or a RuntimeError. A
        ResourceError raised inside passes through unchanged; the cause is kept with from e.
    """
    return ToolBoundary(
        _diagnoser_of(context),
        invalid=ResourceNotFoundError,
        failure=ResourceError,
        passthrough=(ResourceError,),
    )


def _invalid_params(message: str) -> MCPError:
    return MCPError(code=INVALID_PARAMS, message=message)


def _internal_error(message: str) -> MCPError:
    return MCPError(code=INTERNAL_ERROR, message=message)


def protocol_errors(context: object | None = None) -> ToolBoundary:
    """Build the boundary for one request handler installed on the low-level server (resources/list).

    Args:
        context: The lifespan context (ServerRequestContext.lifespan_context), used as in tool_errors().

    Returns:
        A ToolBoundary that raises MCPError with code INVALID_PARAMS (-32602) for a ValueError (a bad cursor), and
        with code INTERNAL_ERROR (-32603) carrying the message for a diagnosed failure, an Exegol SDK guard error or
        a RuntimeError. An MCPError raised inside passes through unchanged; the cause is kept with from e.
    """
    return ToolBoundary(
        _diagnoser_of(context),
        invalid=_invalid_params,
        failure=_internal_error,
        passthrough=(MCPError,),
    )


def _diagnoser_of(context: object | None) -> ReadinessDiagnoser | None:
    return context if isinstance(context, ReadinessDiagnoser) else None
