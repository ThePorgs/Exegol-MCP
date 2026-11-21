"""The single blocking-offload seam (EXEC-01).

Every synchronous Exegol SDK or Docker call goes through run_blocking() so it runs in a worker thread and the
event loop stays free to answer other requests (tools/list, concurrent tool calls). In v1 a long command ran
on the loop and froze the whole server until it returned.

Image listing and download stay awaited on the loop. Container start, stop, create post-setup and removal are replaced
by mirrors (sdk_headless patch steps 10-14) that keep their status display on the loop and run each docker call through
run_blocking: a whole SDK coroutine never runs inside a worker thread, because it takes a loop-bound console lock.
Only the raw synchronous calls are offloaded.
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import anyio
import anyio.to_thread

T = TypeVar("T")

# Default bound on concurrent blocking calls (D-26). A slow call holds one token, so at most this many worker
# threads are busy with Exegol/Docker work at once.
_default_limiter = anyio.CapacityLimiter(8)

# Image pulls get their own, smaller limiter (D-25) so long downloads cannot use up the default tokens and
# starve command execution.
pull_limiter = anyio.CapacityLimiter(2)

# Every reader loop of the tagged job primitive (adapter/exec.py pump_job) draws a token from this dedicated
# limiter instead of the default pool, so background and foreground job readers can never starve the default
# tokens that listings and state reads need (RESEARCH Open Question 2, resolved). Sized to the built-in
# max_concurrent_jobs default (16) plus the default pool's token count (8); the lifespan resizes it at startup
# via size_jobs_limiter() from the resolved config (plan 02-05).
jobs_limiter = anyio.CapacityLimiter(16 + 8)


def size_jobs_limiter(max_concurrent_jobs: int) -> None:
    """Resize the jobs limiter to max_concurrent_jobs plus the default pool's token count (plan 02-05).

    A capped background/foreground job reader can then never hold a token a foreground exec's reader needs: the
    cap bounds job readers, and the extra default-pool-sized headroom keeps listings and state reads flowing.
    """
    jobs_limiter.total_tokens = max_concurrent_jobs + _default_limiter.total_tokens


async def run_blocking(fn: Callable[..., T], *args: Any, limiter: anyio.CapacityLimiter | None = None) -> T:
    """Run a synchronous callable in a worker thread without blocking the event loop.

    Args:
        fn: The blocking callable (a docker-py call, a synchronous SDK helper).
        *args: Positional arguments passed to fn.
        limiter: The capacity limiter to draw a token from. Defaults to the shared default limiter; pass
            pull_limiter for image pulls.

    Returns:
        Whatever fn returns.

    Raises:
        Any exception fn raises, re-raised in the calling task.
    """
    return await anyio.to_thread.run_sync(
        functools.partial(fn, *args),
        limiter=limiter or _default_limiter,
        # On cancellation, return control to the request right away even if the thread is still running.
        abandon_on_cancel=True,
    )
