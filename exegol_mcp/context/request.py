"""Per-request state carried in a ContextVar.

v1 kept the download progress target in a class attribute shared by every call, so two concurrent downloads
reported progress to whichever request had set it last. Here each request binds its own RequestState: a
ContextVar value is local to the task that set it (and to the worker threads anyio starts from that task), so
concurrent requests never see each other's state.
"""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# Async progress sink, same shape as ctx.report_progress: (progress, total, message).
ProgressCallback = Callable[[float, float | None, str | None], Awaitable[None]]


@dataclass(frozen=True)
class RequestState:
    """State that belongs to one MCP request"""

    request_id: str | None = None
    progress: ProgressCallback | None = None


request_ctx: ContextVar[RequestState | None] = ContextVar("exegol_mcp_request", default=None)


@contextmanager
def bind_request(state: RequestState) -> Iterator[RequestState]:
    """Make state the current request state for the enclosed block, then restore the previous value.

    Args:
        state: The state of the request being handled.

    Yields:
        The bound state.
    """
    token = request_ctx.set(state)
    try:
        yield state
    finally:
        request_ctx.reset(token)
