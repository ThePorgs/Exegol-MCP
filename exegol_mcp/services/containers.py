"""Container services (SDK-free: they only talk to an ExegolPort)."""

import anyio

from exegol_mcp.adapter.port import ContainerState, ExegolPort
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.container import ContainerInfo

# D-18: container start/stop get their own internal timeout, separate from the per-call exec timeout keys.
# None resolves to this built-in default so a direct service call keeps working; the tool passes the
# operator-configured execution.container_op_timeout_s.
_DEFAULT_CONTAINER_OP_TIMEOUT_S = ExecutionLimits().container_op_timeout_s

# D-15: the error a stopped container gives a call that did not ask for auto_start. It keeps the Phase-1 "start it
# first" wording and names the auto_start alternative, so the agent can pick either path.
NOT_RUNNING_MESSAGE = (
    "Container '{name}' is not running. Please start it first using the 'start_container' tool, "
    "or pass auto_start=true to start it automatically."
)


async def list_containers(
    adapter: ExegolPort, *, run_id: str | None = None, managed_only: bool = False
) -> list[ContainerInfo]:
    """List the Exegol containers known to the adapter, in the adapter's order, with this_run set (LIFE-04, D-10).

    Args:
        adapter: The Exegol port.
        run_id: This server process's run id. A managed container whose run-id label equals it exactly (string
            equality, case included) is this run's; None marks nothing as this run's.
        managed_only: Keep only the containers exegol-mcp created (its managed label is exactly "true"),
            including those from earlier or crashed runs (LIFE-06).

    Returns:
        One entry per container; an empty list when none matches.
    """
    infos = await adapter.list_containers()
    if managed_only:
        infos = [info for info in infos if info.managed]
    return [
        info.model_copy(update={"this_run": info.managed and run_id is not None and info.run_id == run_id})
        for info in infos
    ]


async def _require_state(adapter: ExegolPort, name: str) -> ContainerState:
    """Return the live state of the named container.

    Raises:
        ValueError: If no container has that name.
    """
    state = await adapter.get_container_state(name)
    if state is None:
        raise ValueError(f"Container '{name}' not found. Call list_containers to see the existing containers.")
    return state


async def start_container(adapter: ExegolPort, name: str, *, timeout_s: float | None = None) -> bool:
    """Start the named container unless it already runs.

    The whole operation is bounded by an internal timeout (D-18), separate from the exec timeout keys. On
    expiry the agent gets a clear error naming the container instead of a hung call; the container may still
    be starting in the background.

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        timeout_s: Internal start timeout in seconds; None uses the built-in container_op_timeout_s default.

    Returns:
        The running state after the call: True once the container runs. An already running container is left
        untouched and True is returned.

    Raises:
        ValueError: If no container has that name.
        RuntimeError: If the start does not finish within timeout_s.
    """
    if timeout_s is None:
        timeout_s = _DEFAULT_CONTAINER_OP_TIMEOUT_S
    try:
        # Bound the whole operation (lookup, no-op check, state change) on its own internal deadline (D-18).
        with anyio.fail_after(timeout_s):
            # 1. Check that the container exists
            state = await _require_state(adapter, name)
            # 2. Already running: nothing to do
            if state.running:
                return True
            # 3. Start it and report the resulting state
            return await adapter.start_container(name)
    except TimeoutError:
        raise RuntimeError(
            f"Container '{name}' did not finish starting within {timeout_s:g} s. It may still be starting: "
            "call list_containers to check its status before retrying."
        ) from None


async def ensure_running(adapter: ExegolPort, name: str, *, auto_start: bool, timeout_s: float | None = None) -> None:
    """Make sure the named container runs before something is executed in it (LIFE-07, D-15).

    Shared by the exec, remote-exec and file services. A running container is left untouched. A stopped one is
    started only when the caller asked for it with auto_start, through start_container, so the start keeps its
    own container-op bound (D-18) and finishes before any exec timeout begins. Only the named container is ever
    started.

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        auto_start: Start the container when it is stopped instead of raising the not-running error.
        timeout_s: Internal start timeout in seconds; None uses the built-in container_op_timeout_s default.

    Raises:
        ValueError: If no container has that name (no start is attempted).
        RuntimeError: If the container is stopped and auto_start is false, or the start does not finish within
            timeout_s.
    """
    # 1. The container must exist: an unknown name is the agent's to fix and nothing is started.
    state = await _require_state(adapter, name)
    # 2. Already running: nothing to do.
    if state.running:
        return
    # 3. Stopped and not asked to start it: the clear error naming both options.
    if not auto_start:
        raise RuntimeError(NOT_RUNNING_MESSAGE.format(name=name))
    # 4. Start it under the existing bounded start (D-18 deadline and its "did not finish starting" message).
    await start_container(adapter, name, timeout_s=timeout_s)


async def stop_container(adapter: ExegolPort, name: str, *, timeout_s: float | None = None) -> bool:
    """Stop the named container unless it is already stopped.

    The whole operation is bounded by an internal timeout (D-18), separate from the exec timeout keys. On
    expiry the agent gets a clear error naming the container instead of a hung call; the container may still
    be stopping in the background.

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        timeout_s: Internal stop timeout in seconds; None uses the built-in container_op_timeout_s default.

    Returns:
        The stopped state after the call: True once the container is stopped. An already stopped container is
        left untouched and True is returned.

    Raises:
        ValueError: If no container has that name.
        RuntimeError: If the stop does not finish within timeout_s.
    """
    if timeout_s is None:
        timeout_s = _DEFAULT_CONTAINER_OP_TIMEOUT_S
    try:
        # Bound the whole operation (lookup, no-op check, state change) on its own internal deadline (D-18).
        with anyio.fail_after(timeout_s):
            # 1. Check that the container exists
            state = await _require_state(adapter, name)
            # 2. Already stopped: nothing to do
            if not state.running:
                return True
            # 3. Stop it. The port returns the running state, so a stopped container comes back as False.
            return not await adapter.stop_container(name)
    except TimeoutError:
        raise RuntimeError(
            f"Container '{name}' did not finish stopping within {timeout_s:g} s. It may still be stopping: "
            "call list_containers to check its status before retrying."
        ) from None
