"""Headless setup of the Exegol SDK (PROTO-05, D-05, D-09).

The exegol library is written for an interactive terminal. It reads its options from argparse through the
ParametersManager singleton, logs and prompts on a rich console bound to stdout, and exits the process on a
critical error. None of that is acceptable inside the MCP server: argparse would parse the server's own argv,
stdout carries the stdio protocol, a prompt would wait for a human who is not there, and exit(1) would kill
every session. patch_exegol_sdk() adapts the library in-process before anything else uses it.
"""

import asyncio
import functools
import logging
import math
import threading
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

import anyio
import docker.errors
import docker.models.containers
import exegol.model.ExegolContainer as _exegol_container_module
import exegol.profile.ProfileAskUser as _profile_ask_user_module
import exegol.utils.FsUtils as _exegol_fs_utils_module
from exegol.config.EnvInfo import EnvInfo
from exegol.console.cli.ParametersManager import ParametersManager
from exegol.console.ExegolPrompt import ExegolRich
from exegol.console.TUI import ExegolTUI
from exegol.utils.ContainerLogStream import ContainerLogStream

# The SDK imports DockerUtils locally inside ExegolContainer.remove to dodge its own circular import; sdk_headless sits
# outside that cycle. The removal mirror calls this module-level name, which tests replace.
from exegol.utils.DockerUtils import DockerUtils
from exegol.utils.ExeLog import logger
from exegol.utils.MetaSingleton import MetaSingleton

from exegol_mcp.adapter import sdk_logging
from exegol_mcp.adapter.download_progress import download_docker_layer
from exegol_mcp.adapter.errors import (
    ContainerNameConflictError,
    HeadlessPromptError,  # re-exported for existing callers
    UnknownParameterError,  # re-exported for existing callers
)
from exegol_mcp.adapter.exec import _docker_handle
from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.selfcheck import _broken, assert_patch_points
from exegol_mcp.bootstrap.logging_setup import stderr_console

_log = logging.getLogger(__name__)

# Every CLI parameter (argparse dest) exegol 6.x reads from ParametersManager that exegol-mcp leaves unset:
# one per option registered in OptionResolver, plus the positional selectors the managers read directly.
# None means "not given on the command line", so OptionResolver falls through to the profile, the user's
# config.yml and the built-in default, exactly as a wrapper run without that flag would.
# Any name NOT listed here (or set in _HEADLESS_VALUES) raises UnknownParameterError when read (D-09).
_UNSET_CLI_DESTS: tuple[str, ...] = (
    "api_key",
    "build_log",
    "build_path",
    "capabilities",
    "comment",
    "daemon",
    "desktop",
    "desktop_config",
    "devices",
    "envs",
    "exec",
    "exegol_resources",
    "gui",
    "hostname",
    "hosts_file",
    "image_tag",
    "imagetag",
    "info_all",
    "info_config",
    "info_sources",
    "license_id",
    "log",
    "log_compress",
    "log_method",
    "my_resources",
    "network",
    "no_backup",
    "ports",
    "privileged",
    "revoke",
    "select_all",
    "selector",
    "sentinel",
    "sentinel_profile",
    "sentinel_refresh",
    "sentinel_strategy",
    "share_timezone",
    "shell",
    "shell_type",
    "tmp",
    "update_fs_perms",
    "update_profile",
    "update_resources",
    "update_sentinel",
    "update_wrapper",
    "verbosity",
    "volumes",
    "vpn",
    "vpn_auth",
    "workspace_path",
    # Options with no CLI flag at all (profile/config.yml settings) are registered with the empty dest "", and
    # OptionResolver still asks the CLI tier for it: answer "not given", as argparse's ParametersManager does.
    "",
    # Positional selectors (not resolved through OptionResolver)
    "build_profile",
    "containertag",
    "mount_current_dir",
    "multicontainertag",
    "multiimagetag",
    "profile",
)

# Parameters exegol-mcp sets explicitly so the SDK runs without a human.
_HEADLESS_VALUES: dict[str, Any] = {
    "quiet": False,
    "arch": EnvInfo.arch,
    # Online by default; the SDK itself switches this to True when the registry is unreachable.
    "offline_mode": False,
    # Never accepted implicitly (D-05/D-06): only the operator's --accept-eula sets it, via set_accept_eula().
    "accept_eula": False,
    # Skip the SDK's "are you sure?" confirmations: the MCP tool call is the explicit request.
    "force_mode": True,
}

_parameters: "ParametersMocked | None" = None


class ParametersMocked:
    """Stand-in for exegol's ParametersManager that never parses argv.

    Every parameter exegol 6.x reads is set explicitly. Reading any other name raises UnknownParameterError
    instead of returning None (D-09): a renamed or new option in a later 6.x release must fail loudly rather
    than silently fall back to a default that could drive a prompt or the wrong behaviour.
    """

    def __init__(self) -> None:
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        for dest in _UNSET_CLI_DESTS:
            setattr(self, dest, None)
        for dest, value in _HEADLESS_VALUES.items():
            setattr(self, dest, value)

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails, so set parameters never reach this.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)  # keep Python's special-method probing (copy, pickle, rich) working
        _log.debug("Exegol SDK read the unset CLI parameter %r", name)
        raise UnknownParameterError(
            f"The Exegol SDK read the CLI parameter '{name}', which exegol-mcp does not set. The installed exegol "
            "release may have renamed or added an option; exegol-mcp refuses to guess its value."
        )

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter, as if it had been given on the command line."""
        setattr(self, name, value)

    def reset_all_parameters(self) -> None:
        """Drop every parameter set at runtime and restore the headless defaults."""
        self.__dict__.clear()
        self._apply_defaults()


# Margin (seconds) kept between the end of the SDK's startup-status wait and the enclosing service deadline, so
# the real adapter can still read the container's running state before the deadline fires (D-18). Floored at 0.
_DEADLINE_MARGIN_S = 2.0


# True only inside removal_offload, in the removing task (CR-01): a startup wait Exegol's removal fallback runs then
# ends at once instead of waiting for READY on the loop.
_removal_in_progress: ContextVar[bool] = ContextVar("exegol_mcp_removal_in_progress", default=False)


def _deadline_stop_time() -> datetime | None:
    """Wall-clock time at which the SDK startup wait must stop, or None when no anyio deadline encloses it.

    The startup mirror (patch step 11) builds the log stream on the loop, where this function reads the enclosing
    anyio deadline (a worker thread has no anyio task to ask), and runs each poll in a worker thread. The stop time
    still ends the READY wait shortly before that deadline, like the SDK's own Ctrl+C "skip startup status updates"
    path (the container keeps starting in the background), so the post-start setup runs before the deadline fires; it
    also bounds a poll thread a fired deadline abandoned. Inside removal_offload the stop time is "now": a removal
    never waits for a restart to finish.
    """
    if _removal_in_progress.get():
        return datetime.now()
    try:
        deadline = anyio.current_effective_deadline()
    except Exception:  # noqa: BLE001 - no running loop / async library raises its own type; means "no deadline"
        return None  # not inside an anyio task (e.g. a non-async SDK user): behave exactly like the SDK
    if deadline == math.inf:
        return None
    remaining = deadline - anyio.current_time() - _DEADLINE_MARGIN_S
    return datetime.now() + timedelta(seconds=max(0.0, remaining))


class _DeadlineContainerLogStream(ContainerLogStream):
    """ContainerLogStream that ends its READY wait before the enclosing anyio deadline (D-18).

    The SDK class switches its own 2 s timeout off as soon as container logs flow, so a startup sequence that
    never logs READY would wait forever. This subclass keeps the SDK's empty-poll timeout check enabled and
    makes its timeout date the earlier of the SDK's own date and a stop time derived from the enclosing service
    deadline. With no deadline it is indistinguishable from the SDK class, and the SDK's 2 s legacy-image
    detection is never lengthened.

    The base class stores its timeout flag and date under two name-mangled attributes. Overriding them as
    properties lets the base __init__/__next__ keep writing the SDK's own values (into shadow attributes), while
    reads enforce the stop time. The property names start with a single underscore, so they are NOT mangled
    again in this subclass body and match the base's mangled names exactly.
    """

    def __init__(self, container: Any, start_date: datetime | None = None, timeout: int = 5) -> None:
        self.__stop_time = _deadline_stop_time()
        # Shadow the SDK's own values; the property setters below fill them when the base __init__ assigns them.
        self._sdk_enable_timeout = False
        self._sdk_timeout_date = datetime.now()
        super().__init__(container, start_date=start_date, timeout=timeout)

    @property
    def _ContainerLogStream__enable_timeout(self) -> bool:
        # Keep the empty-poll timeout check enabled even after the SDK disabled it once logs flowed, so the stop
        # time below can end the wait. With no deadline, report the SDK's own value (identical behaviour).
        if self.__stop_time is not None:
            return True
        return self._sdk_enable_timeout

    @_ContainerLogStream__enable_timeout.setter
    def _ContainerLogStream__enable_timeout(self, value: bool) -> None:
        self._sdk_enable_timeout = value

    @property
    def _ContainerLogStream__timeout_date(self) -> datetime:
        if self.__stop_time is None:
            return self._sdk_timeout_date  # no deadline: identical to the SDK class
        if self._sdk_enable_timeout:
            # While the SDK still has its own timeout on (no logs yet), honour whichever fires first, so its 2 s
            # legacy-image detection is never lengthened.
            return min(self._sdk_timeout_date, self.__stop_time)
        # Logs flowed and the SDK disabled its own timeout: enforce only the deadline-derived stop time.
        return self.__stop_time

    @_ContainerLogStream__timeout_date.setter
    def _ContainerLogStream__timeout_date(self, value: datetime) -> None:
        self._sdk_timeout_date = value

    def stop_now(self) -> None:
        """End the stream's wait at its next poll, even when no service deadline encloses it.

        Called from the loop by the startup mirror's `finally`, so a poll thread abandoned by a cancelled caller ends
        within one poll instead of reading docker logs forever. A single attribute store, safe while a worker thread
        polls; the _ContainerLogStream__enable_timeout property already reports True once the stop time is set.
        """
        self.__stop_time = datetime.now()

    def __next__(self) -> str:
        # Stop immediately once the stop time has passed, covering the case where a partial line (no newline)
        # was buffered just before the container went silent.
        if self.__stop_time is not None and datetime.now() >= self.__stop_time:
            raise StopIteration
        return super().__next__()


def _refuse_prompt(kind: str) -> Callable[..., Awaitable[NoReturn]]:
    async def refuse(message: str, *args: Any, **kwargs: Any) -> NoReturn:
        raise HeadlessPromptError(
            f"The Exegol SDK tried to {kind} during a headless MCP call and was refused: {message!r}. "
            "Run the same operation once with the exegol wrapper in a terminal to settle it."
        )

    return refuse


# A prompt answerer: (kind, message) -> the answer, or None to refuse. kind is "confirm" or "acknowledge".
PromptPolicy = Callable[[str, str], bool | None]

# The answerer bound for the current task only (answer_prompts). None, the default, refuses every prompt (D-09).
_prompt_policy: ContextVar[PromptPolicy | None] = ContextVar("exegol_mcp_prompt_policy", default=None)

_PROMPT_DESCRIPTIONS = {"confirm": "ask for a confirmation", "acknowledge": "wait for an acknowledgement"}


@contextmanager
def answer_prompts(policy: PromptPolicy) -> Iterator[None]:
    """Let the policy answer the SDK's Confirm / Acknowledge prompts for the enclosed block, in this task only.

    The binding is a ContextVar, so a concurrent task never sees another task's policy, and it is reset in `finally`.
    Prompts are never re-enabled globally: outside a bound policy, and for any prompt the policy does not answer,
    the SDK still gets HeadlessPromptError (D-09). Ask has no policy: it is always refused.
    """
    token = _prompt_policy.set(policy)
    try:
        yield
    finally:
        _prompt_policy.reset(token)


def removal_policy(keep_workspace: bool, keep_logs: bool) -> PromptPolicy:
    """Answer exactly the prompts ExegolContainer.remove() asks, from the call's keep options (D-12, Pitfall 8).

    The Sentinel history prompt is answered "remove" only when keep_logs is False; the non-empty and unreadable
    workspace prompts are answered "delete" only when keep_workspace is False. Every other prompt (and every
    Acknowledge) gets None, so it is refused. A drift test pins the prompt texts to Exegol's source.
    """

    def answer(kind: str, message: str) -> bool | None:
        if kind != "confirm":
            return None
        if "Sentinel logging history" in message:
            return not keep_logs
        lowered = message.lower()
        if "workspace" in lowered and "delete" in lowered:
            return not keep_workspace
        return None

    return answer


def _offloaded_exec(container: Any) -> Callable[..., Awaitable[int]]:
    """Exegol's ExegolContainer.exec for one container, with its blocking docker exec in a worker thread (CR-01).

    Both quiet modes are mirrored. The synchronous one (as_daemon=False, quiet=True) is what Exegol's removal and
    post-start setup use: the in-container `rm -rf /workspace`, the Sentinel shred and the xauth cookie. The daemon one
    (as_daemon=True, quiet=True) is what Exegol's password update uses during the post-create setup (patch step 14);
    it detaches, returns 0 right after exec_run as Exegol does, and never logs the command, which carries the container
    password. Each mirrors Exegol's code with exactly Exegol's exec_run arguments, logging and return value, but runs
    the running-state read and exec_run through run_blocking, so the loop stays free and the deadline can interrupt the
    wait. A non-quiet call is Exegol's own method. A drift test pins the mirrored source.
    """

    async def exec_(
        command: Any, as_daemon: bool = True, quiet: bool = False, is_tmp: bool = False, show_output: bool = True
    ) -> int:
        if not quiet:
            result: int = await type(container).exec(
                container, command, as_daemon=as_daemon, quiet=quiet, is_tmp=is_tmp, show_output=show_output
            )
            return result
        if not await run_blocking(container.isRunning):
            # On the loop like Exegol's exec (console lock); its READY wait is skipped inside removal_offload.
            await container.start()
        exec_payload, str_cmd = _exegol_container_module.ExegolContainer.formatShellCommand(command, quiet)
        stream = await run_blocking(
            functools.partial(
                _docker_handle(container).exec_run,
                exec_payload,
                environment={"CMD": str_cmd, "DISABLE_AUTO_UPDATE": "true"},
                detach=as_daemon,
                # Exegol's stream=not as_daemon and not quiet is False in both quiet modes.
                stream=False,
            )
        )
        if as_daemon:
            # Exegol returns right after a detached exec_run, before any output handling or logging.
            return 0
        if type(stream.output) is bytes:
            if stream.exit_code is not None and stream.exit_code != 0:
                logger.debug(f"An error occurred while executing the command: [error {stream.exit_code}] {command}")
            if len(stream.output) > 0 and show_output:
                # errors="replace": an undecodable byte must not fail a removal (Exegol decodes strictly).
                text = stream.output.decode("utf-8", errors="replace")
                if stream.exit_code is None or stream.exit_code == 0:
                    logger.raw(text)
                else:
                    logger.error(text)
        return 0 if stream.exit_code is None else int(stream.exit_code)

    return exec_


def _offloaded_stop(container: Any) -> Callable[..., Awaitable[None]]:
    """Exegol's ExegolContainer.stop for one container, with the docker stop in a worker thread (CR-01)."""

    async def stop(timeout: int = 10) -> None:
        if await run_blocking(container.isRunning):
            logger.info(f"Stopping container {container.name}")
            # docker-py's Container.stop takes keyword arguments only.
            await run_blocking(functools.partial(_docker_handle(container).stop, timeout=timeout))

    return stop


# The ExegolContainer methods removal_offload shadows on the one object being removed.
_OFFLOADED_METHODS = ("exec", "stop")


@contextmanager
def removal_offload(container: Any) -> Iterator[None]:
    """Keep Exegol's removal of this one container off the event loop, for the enclosed block only (CR-01).

    Exegol's remove() runs its volume step while the container still runs, then stops it. When the host cannot
    delete root-owned workspace or Sentinel files, the volume step falls back to a synchronous in-container delete,
    restarting the container first if it is stopped. Both waits would run on the loop. For this removal and this
    ExegolContainer object only, this context:

    - shadows the instance attributes exec and stop with _offloaded_exec / _offloaded_stop (an instance attribute
      wins over the class method Exegol calls through self), so the in-container delete and the stop run their
      docker call in a worker thread;
    - binds _removal_in_progress in this task, so a fallback restart's READY wait ends at once instead of polling
      until READY (the container keeps booting in the background);
    - through the same binding, makes patch step 9 run Exegol's host-side Sentinel shred (FsUtils.secure_remove) in a
      worker thread, so the removal deadline can interrupt it (WR-09).

    Scope rule: start_container and stop_container now use the class-level mirrors of patch steps 10-12, which keep
    the status display; this context still shadows exec and stop on the one container being removed (its stop shadow
    skips the status display during a removal). That stop shadow predates the class-level stop mirror (patch step 10)
    and is kept on purpose: a removal needs no "Waiting to stop" status, and its tests pin it. Exegol's remove itself is
    the patch step 13 mirror, whose docker tail is one worker hop. The shadows are deleted (or a previous instance
    attribute restored) and the ContextVar reset in `finally`.
    """
    token = _removal_in_progress.set(True)
    attributes = vars(container)
    previous = {name: attributes[name] for name in _OFFLOADED_METHODS if name in attributes}
    container.exec = _offloaded_exec(container)
    container.stop = _offloaded_stop(container)
    try:
        yield
    finally:
        for name in _OFFLOADED_METHODS:
            if name in previous:
                setattr(container, name, previous[name])
            else:
                attributes.pop(name, None)
        _removal_in_progress.reset(token)


# docker-py's own ContainerCollection.create, kept at import so patch step 8 always wraps the original (WR-03).
# Read with getattr: if a docker-py release renamed it, the import still finishes and patch_exegol_sdk()'s
# assert_patch_points() names the missing point instead of an AttributeError traceback (WR-12, D-15).
_DOCKER_CONTAINERS_CREATE: Callable[..., Any] | None
_DOCKER_CONTAINERS_CREATE = getattr(getattr(docker.models.containers, "ContainerCollection", None), "create", None)

# The signature of Exegol's FsUtils.secure_remove coroutine function.
_SecureRemove = Callable[[Path, int], Coroutine[Any, Any, None]]

# Exegol's own host-side shred (FsUtils.secure_remove), kept at import so patch step 9 always wraps it (WR-09).
# Read with getattr for the same reason: a renamed SDK attribute is named by the self-check (WR-12, D-15).
_EXEGOL_SECURE_REMOVE: _SecureRemove | None = getattr(_exegol_fs_utils_module, "secure_remove", None)


def _exegol_secure_remove() -> _SecureRemove:
    """The import-time FsUtils.secure_remove, read at call time; a named PatchPointError if it was missing (WR-12).

    patch_exegol_sdk() already refuses startup in that case, so this is a backstop that -O cannot strip.
    """
    original = _EXEGOL_SECURE_REMOVE
    if original is None:
        raise _broken("exegol.utils.FsUtils.secure_remove", "is missing")
    return original


# Exegol's own ExegolContainer.stop, kept at import. Patch step 10 replaces the class attribute, so the drift tests
# read Exegol's source from here, and a renamed SDK method reaches assert_patch_points() as a named point instead of
# an AttributeError at import (WR-12, D-15).
_SDK_STOP: Callable[..., Any] | None
_SDK_STOP = getattr(getattr(_exegol_container_module, "ExegolContainer", None), "stop", None)

# Exegol's own ExegolContainer.__start_container and .start, kept at import for the same reasons as _SDK_STOP: patch
# steps 11-12 replace the class attributes, so the drift tests read Exegol's source from here.
_SDK_START_CONTAINER: Callable[..., Any] | None
_SDK_START_CONTAINER = getattr(
    getattr(_exegol_container_module, "ExegolContainer", None), "_ExegolContainer__start_container", None
)
_SDK_START: Callable[..., Any] | None
_SDK_START = getattr(getattr(_exegol_container_module, "ExegolContainer", None), "start", None)

# Exegol's own ExegolContainer.remove, kept at import for the same reasons: patch step 13 replaces the class attribute.
_SDK_REMOVE: Callable[..., Any] | None
_SDK_REMOVE = getattr(getattr(_exegol_container_module, "ExegolContainer", None), "remove", None)

# Exegol's own ExegolContainer.postCreateSetup, kept at import for the same reasons: patch step 14 replaces the class
# attribute.
_SDK_POST_CREATE_SETUP: Callable[..., Any] | None
_SDK_POST_CREATE_SETUP = getattr(getattr(_exegol_container_module, "ExegolContainer", None), "postCreateSetup", None)

# The ExegolContainer class attributes patch steps 10-14 replace (INT-01). The test restore fixtures loop over this
# tuple, so each later step only appends its method name here.
_PATCHED_CONTAINER_METHODS: tuple[str, ...] = (
    "stop",
    "_ExegolContainer__start_container",
    "start",
    "remove",
    "postCreateSetup",
)

# The default next(stream, _END) returns when a log stream ends: a StopIteration cannot cross from a worker thread into
# a coroutine (the future it would resolve raises RuntimeError instead).
_END: Any = object()


def _secure_remove_to_completion(file_path: Path, passes: int) -> None:
    """Run Exegol's secure_remove to completion on a private asyncio loop. Called in a worker thread (WR-09).

    Exegol's coroutine is asyncio-specific (its non-Linux path uses asyncio.gather) and its Linux path never awaits,
    so it is run whole with asyncio.run: the worker thread has no running loop. The module global is read at call
    time, so tests can substitute it.
    """
    asyncio.run(_exegol_secure_remove()(file_path, passes))


async def _secure_remove_offloaded(file_path: Path, passes: int = 3) -> None:
    """Exegol's secure_remove, in a worker thread while an exegol-mcp removal runs (WR-09).

    Exegol's host-side Sentinel shred overwrites every file several times with an fsync per pass and never yields, so
    on the loop it stalls every session and ignores the removal deadline. Scope rule, as for removal_offload: only
    while _removal_in_progress is bound (this removal, in the removing task and its worker threads) does the shred run
    through run_blocking, so the deadline can interrupt the wait (the thread is abandoned, as for the in-container
    delete). Anywhere else Exegol's original is awaited on the calling loop, unchanged. Exceptions propagate
    unchanged, so Exegol's own PermissionError fallback and its OSError / NotImplementedError handling still apply.
    """
    if not _removal_in_progress.get():
        await _exegol_secure_remove()(file_path, passes)
        return
    await run_blocking(_secure_remove_to_completion, file_path, passes)


# --- INT-01: container start, stop, create post-setup and removal off the event loop ------------------------------
#
# Exegol declares these methods `async def`, but they call docker-py (and sleep) inline, so every other request stalls
# for the length of the docker call and the anyio deadline around them (container_op_timeout_s) can never fire. The
# rule for each mirror below: the status display and its loop-bound console lock stay on the event loop, every
# docker-py call runs in a worker thread through run_blocking, and the mirror keeps Exegol's arguments, log lines,
# order and return value. A whole SDK coroutine is never run in a worker thread (it would bind the console lock to the
# wrong loop). Each mirror is installed at class level by one numbered step of patch_exegol_sdk(); a drift test pins
# the Exegol lines it copies.


async def _stop_offloaded(self: Any, timeout: int = 10) -> None:
    """Exegol's ExegolContainer.stop, with its docker calls in a worker thread (patch step 10, INT-01).

    Same order, log line, status text and docker timeout as Exegol's: read the running state, log, show the
    "Waiting to stop" status, stop. Only the running-state read and the docker stop run through run_blocking; the
    status display and its console lock stay on the loop. The enclosing anyio deadline is delivered at the await, and
    the abandoned worker thread finishes on its own (run_blocking never kills a thread), so container_op_timeout_s
    bounds the call.

    Args:
        self: The ExegolContainer being stopped.
        timeout: Seconds docker waits for the container to stop, as Exegol's own argument.
    """
    # 1. The running-state read reloads the container from the daemon: a docker call, off the loop.
    if await run_blocking(self.isRunning):
        logger.info(f"Stopping container {self.name}")
        # 2. The status display keeps Exegol's console lock on the loop; only the docker stop leaves it. docker-py's
        # Container.stop takes keyword arguments only. The model module's name is read at call time (tests patch it).
        async with _exegol_container_module.ExegolStatus(f"Waiting to stop ({timeout}s timeout)", spinner_style="blue"):
            await run_blocking(functools.partial(_docker_handle(self).stop, timeout=timeout))


@contextmanager
def _exec_shadow(container: Any) -> Iterator[None]:
    """Shadow exec on one container with the offloaded quiet mode, for the enclosed block only (INT-01).

    The post-start setup's xauth exec (quiet synchronous mode) and the post-create password update's exec (quiet daemon
    mode) are run inline by Exegol. The instance attribute wins over the class method Exegol calls through self. A previous instance attribute (the shadow
    removal_offload binds around a removal) is restored, otherwise the shadow is removed again, in `finally`.
    """
    attributes = vars(container)
    had_previous = "exec" in attributes
    previous = attributes.get("exec")
    container.exec = _offloaded_exec(container)
    try:
        yield
    finally:
        if had_previous:
            container.exec = previous
        else:
            attributes.pop("exec", None)


async def _start_container_offloaded(self: Any) -> None:
    """Exegol's ExegolContainer.__start_container, with its docker calls in a worker thread (patch step 11, INT-01).

    Same status text, docker error handling, log dispatch and closing log line as Exegol's. Only the docker start and
    each startup log poll (a read plus Exegol's time.sleep(1) between polls) run through run_blocking; the status
    display and its loop-bound console lock, and the log dispatch itself, stay on the loop. The SDK's
    KeyboardInterrupt handler is not copied: there is no terminal user. A caller cancelled mid-wait (the service
    deadline) is handled by the `finally` below, which ends the abandoned poll thread.

    Args:
        self: The ExegolContainer being started.
    """
    handle = _docker_handle(self)
    module = _exegol_container_module
    async with module.ExegolStatus(f"Waiting to start {self.name}", spinner_style="blue") as progress:
        start_date = datetime.now()
        # 1. The docker start returns once the container is created and running: a docker call, off the loop.
        try:
            await run_blocking(handle.start)
        except module.APIError as e:
            logger.debug(e)
            logger.critical(
                f"Docker raised a critical error when starting the container [green]{self.name}[/green], "
                f"error message is: {e.explanation}"
            )
        except module.ReadTimeout:
            logger.critical("Received a timeout error from docker... Unable to start the container, retry later.")
            raise
        if self.config.legacy_entrypoint:  # TODO in Exegol: improve startup compatibility check
            return
        # 2. The startup log stream is built on the loop: its constructor reads the enclosing anyio deadline, which a
        # worker thread cannot. Exegol times it out after 2 s when the image logs nothing (no status support).
        stream = module.ContainerLogStream(handle, start_date=start_date, timeout=2)
        try:
            # 3. Each poll is a docker log read, plus a 1 s sleep while the container is silent: off the loop.
            while True:
                line = await run_blocking(next, stream, _END)
                # Once the last log "READY" is received, the startup sequence is over and the execution can continue.
                if line is _END or line == "READY":
                    break
                elif line.startswith("[INFO]"):
                    logger.info(line.replace("[INFO]", ""))
                elif line.startswith("[VERBOSE]"):
                    logger.verbose(line.replace("[VERBOSE]", ""))
                elif line.startswith("[ADVANCED]"):
                    logger.advanced(line.replace("[ADVANCED]", ""))
                elif line.startswith("[DEBUG]"):
                    logger.debug(line.replace("[DEBUG]", ""))
                elif line.startswith("[WARNING]"):
                    logger.warning(line.replace("[WARNING]", ""))
                elif line.startswith("[ERROR]"):
                    logger.error(line.replace("[ERROR]", ""))
                elif line.startswith("[SUCCESS]"):
                    logger.success(line.replace("[SUCCESS]", ""))
                elif line.startswith("[PROGRESS]"):
                    line = line.replace("[PROGRESS]", "")
                    logger.verbose(line)
                    progress.update(status=f"[blue][Startup][/blue] {line}")
                else:
                    logger.debug(line)
            logger.verbose(f"Container started in {(datetime.now() - start_date).seconds} seconds")
        finally:
            # 4. Whatever ended the wait, a poll thread a cancelled caller abandoned must not keep polling docker.
            stop_now = getattr(stream, "stop_now", None)
            if stop_now is not None:
                stop_now()


async def _start_offloaded(self: Any) -> None:
    """Exegol's ExegolContainer.start, with its blocking steps off the loop (patch step 12, INT-01).

    The same steps in the same order as Exegol's: read the running state, run the pre-start host checks, log, start
    (the step 11 mirror), then the post-start setup. The running-state read and the pre-start checks (volume and VPN
    path checks that may log a critical, which raises RuntimeError here) run in worker threads. Exegol's own
    post-start setup runs on the loop with exec shadowed by the offloaded quiet mode. Accepted residual: for GUI
    containers its host xhost / xauth subprocesses stay on the loop, normally milliseconds.

    Args:
        self: The ExegolContainer being started.
    """
    if not await run_blocking(self.isRunning):
        await run_blocking(self._ExegolContainer__preStartChecks)
        logger.info(f"Starting container {self.name}")
        await self._ExegolContainer__start_container()
        with _exec_shadow(self):
            await self._ExegolContainer__postStartSetup()


# Exegol's removal tail mutates process-wide SDK state that is not thread-safe: DataCache() and DockerUtils() are
# MetaSingleton instances (no lock; a concurrent first construction raises "already being spawned") and
# remove_container_cache is an unlocked read-modify-write. On the loop the tails ran one after another; as worker
# threads (and an abandoned tail still running after its deadline) they would overlap, so they serialize here.
_REMOVE_TAIL_LOCK = threading.Lock()


def _remove_tail(container: Any, container_only: bool, backup_history: list[str] | None) -> None:
    """The docker tail of Exegol's ExegolContainer.remove, synchronous, run in ONE worker thread (patch step 13).

    The container removal, the cache entry, the backup removals and the network removals are one unit: a removal
    deadline that fires between them must never leave the container removed but its backups or networks behind
    (RESEARCH Pitfall 4). They therefore share a single run_blocking hop, and an abandoned thread finishes the whole
    unit. It removes exactly this container's own docker handle, the backup ids the caller passed and the networks
    Exegol marks shouldBeRemoved(), as Exegol's own tail does, with the SDK's log lines, handlers and order.

    The whole tail holds _REMOVE_TAIL_LOCK, so concurrent removals (and an abandoned tail still running) never
    construct or mutate the SDK's DataCache / DockerUtils singletons at the same time. The lock is taken in the worker
    thread, never on the loop.

    Args:
        container: The ExegolContainer being removed.
        container_only: As Exegol's: True removes only the container, not its backups or networks.
        backup_history: The backup container ids removed with it, as Exegol's.
    """
    with _REMOVE_TAIL_LOCK:
        try:
            _docker_handle(container).remove()
            logger.success(f"Container {container.name} successfully removed.")
        except _exegol_container_module.ReadTimeout:
            logger.critical(
                "Received a timeout error from docker while removing the container. Check container status later."
            )
            raise
        except _exegol_container_module.NotFound:
            logger.error(
                f"The container {container.name} has already been removed (probably created as a temporary container)."
            )
        # The model module's names are read at call time (tests patch them); DataCache would write the real
        # ~/.exegol cache.
        _exegol_container_module.DataCache().remove_container_cache(container.name)
        if not container_only:
            if backup_history is not None and len(backup_history) > 0:
                for c_id in backup_history:
                    try:
                        DockerUtils().removeContainerById(c_id)
                    except _exegol_container_module.ObjectNotFound:
                        continue
            for net in container.config.getNetworks():
                if net.shouldBeRemoved():
                    logger.debug(f"Network {net.getNetworkName()} will be removed.")
                    DockerUtils().removeNetwork(net.getNetworkName())


async def _remove_offloaded(self: Any, container_only: bool = False, backup_history: list[str] | None = None) -> None:
    """Exegol's ExegolContainer.remove, with its docker tail in one worker thread (patch step 13, INT-01).

    The same order as Exegol's: the volume step (its own `__removeVolume`, awaited unchanged, so patch step 9, the
    removal prompt policy and removal_offload still apply to it), then `self.stop(timeout=2)` (removal_offload's
    per-instance stop shadow during an exegol-mcp removal, else the step 10 mirror), then the "Removing container" log
    on the loop and the whole docker tail as ONE run_blocking call (_remove_tail). The tail is deliberately not
    shielded: a hung docker remove (up to the docker-py client timeout) would then hold the shielded caller, which is
    the bug being fixed. A removal deadline that fires during the tail is therefore delivered and the caller reports
    REMOVAL_TIMEOUT_MESSAGE, while the abandoned worker finishes the whole tail moments later (no half-removed
    container, backups or networks); the message already tells the agent to call list_containers before retrying.

    Accepted residual (T-04-47, T-05.1-25): `__removeVolume` is Exegol's and is not mirrored, so its host-side
    workspace and Sentinel directory steps, and the two `self.isRunning()` reads of its PermissionError fallbacks (a
    synchronous docker reload Exegol calls inside its coroutine), still run on the loop.

    Args:
        self: The ExegolContainer being removed.
        container_only: As Exegol's: True removes only the container, not its volume, backups or networks.
        backup_history: The backup container ids removed with it, as Exegol's.
    """
    if not container_only:
        await self._ExegolContainer__removeVolume()
    # Stop after workspace removal to support in-container backup removal (Exegol's own order).
    await self.stop(timeout=2)
    have_backup = backup_history is not None and len(backup_history) > 0
    backup_text = f" and {len(backup_history)} backup containers" if have_backup and backup_history is not None else ""
    logger.info(f"Removing container {self.name}{backup_text}")
    await run_blocking(_remove_tail, self, container_only, backup_history)


def _put_entrypoint_archive(handle: Any) -> None:
    """Build Exegol's entrypoint tar and upload it to the container root, synchronous, in ONE worker thread.

    The tar build reads Exegol's own ConstantConfig files only (never OptionResolver or the CLI tier, which create
    restores while an abandoned thread may still run), so it shares the upload's worker hop.

    Args:
        handle: The docker-py Container the entrypoint is uploaded to.
    """
    handle.put_archive("/", _exegol_container_module.ImageScriptSync.getImageSyncTarData(include_entrypoint=True))


async def _post_create_setup_offloaded(self: Any, is_temporary: bool = False) -> None:
    """Exegol's ExegolContainer.postCreateSetup, with its docker calls in a worker thread (patch step 14, INT-01).

    The same statements in the same sequence as Exegol's. For a container that is not temporary: the entrypoint upload
    (tar build and put_archive, one worker hop), the start when docker still reports "created" (the step 11 mirror),
    and the password update (Exegol's own `__updatePasswd`, whose exec runs the offloaded quiet daemon mode through
    _exec_shadow), where only an APIError saying the container "is not running" is reported, as a critical (a
    RuntimeError once patched); any other APIError is swallowed, as in Exegol. Then, as the next plain statement and in
    no try block, the post-start setup, so it also runs for a temporary container and is skipped when the password
    update raises anything but an APIError. Deliberately no try/finally around the sequence: that would run the
    post-start setup after a failed password update, which Exegol never does. The cached status read is no I/O.

    Args:
        self: The ExegolContainer just created.
        is_temporary: As Exegol's: True skips the upload, the start and the password update.
    """
    if not is_temporary:
        await run_blocking(_put_entrypoint_archive, _docker_handle(self))
        if _docker_handle(self).status.lower() == "created":
            await self._ExegolContainer__start_container()
        try:
            with _exec_shadow(self):
                await self._ExegolContainer__updatePasswd()
        except _exegol_container_module.APIError as e:
            if "is not running" in e.explanation:
                logger.critical("An unexpected error occurred. Exegol cannot start the container after its creation...")
    with _exec_shadow(self):
        await self._ExegolContainer__postStartSetup()


# True only inside create_conflict_guard(), in the creating task and the worker threads it starts (anyio runs them
# in a copy of the task's context), so another task's docker-py create is never affected.
_create_conflict_guard: ContextVar[bool] = ContextVar("exegol_mcp_create_conflict_guard", default=False)

# The HTTP status Docker answers a create whose container name is already in use.
_HTTP_CONFLICT = 409


@contextmanager
def create_conflict_guard() -> Iterator[None]:
    """Turn a Docker 409 name conflict on docker-py's create into ContainerNameConflictError, for this block only.

    Bound around exegol-mcp's own DockerUtils.createContainer call (WR-03, Pitfall 2). Exegol's createContainer
    rolls back on any APIError: it removes every container whose name matches exactly, then the network. On a 409
    that container belongs to another process, so the guarded create raises an error Exegol does not catch and the
    rollback never runs. Any other APIError, and every create outside this block, behaves exactly as docker-py's.
    """
    token = _create_conflict_guard.set(True)
    try:
        yield
    finally:
        _create_conflict_guard.reset(token)


def _guarded_containers_create(self: Any, image: Any, command: Any = None, **kwargs: Any) -> Any:
    """docker-py's ContainerCollection.create, with a 409 turned into ContainerNameConflictError under the guard."""
    original = _DOCKER_CONTAINERS_CREATE
    if original is None:
        # patch_exegol_sdk() already refused startup in that case (WR-12); a backstop that -O cannot strip.
        raise _broken("docker.models.containers.ContainerCollection.create", "is missing")
    try:
        return original(self, image, command, **kwargs)
    except docker.errors.APIError as e:
        if _create_conflict_guard.get() and e.status_code == _HTTP_CONFLICT:
            raise ContainerNameConflictError(
                "Docker refused the create: the container name is already in use"
            ) from None
        raise


def _answer_or_refuse(kind: str) -> Callable[..., Awaitable[bool | None]]:
    """A Confirm / Acknowledge replacement that answers from the bound policy, else refuses like _refuse_prompt."""
    refuse = _refuse_prompt(_PROMPT_DESCRIPTIONS[kind])

    async def answer_or_refuse(message: str, *args: Any, **kwargs: Any) -> bool | None:
        policy = _prompt_policy.get()
        answer = policy(kind, message) if policy is not None else None
        if answer is None:
            await refuse(message, *args, **kwargs)
        # An acknowledgement has no value: once the policy acknowledged it, return as Exegol's Acknowledge does.
        return answer if kind == "confirm" else None

    return answer_or_refuse


def _stdin_cannot_answer() -> bool:
    """Replacement for the profile ask helper's stdinCanAnswer: the server never answers from stdin."""
    return False


def _route_console_to_stderr() -> None:
    """Send everything the SDK prints or logs to stderr, through the process's shared stderr console.

    Reuses bootstrap.logging_setup's console rather than building a separate one: a second console would diverge
    from the log handler (no soft wrap, its own colour decision). Console(stderr=True) resolves sys.stderr on every
    write, so it follows any later stream redirection.
    """
    sdk_logging.route_console(stderr_console())


def patch_exegol_sdk() -> None:
    """Make the Exegol SDK safe to use headlessly inside the MCP server.

    Call once at startup, before any other SDK use. Calling it again restores the headless defaults.

    Raises:
        PatchPointError: If an SDK internal this patch relies on is missing (D-15).
    """
    global _parameters
    assert_patch_points()
    # 1. Diagnostics on stderr, never on stdout (the stdio protocol stream).
    _route_console_to_stderr()
    # 2. A deep SDK critical raises RuntimeError into the tool call instead of exiting the process.
    logger.setCriticalMethod("raise")
    # 3. Explicit, non-interactive parameters instead of argparse; unknown names raise (D-09).
    _parameters = ParametersMocked()
    MetaSingleton.mock(ParametersManager, _parameters)
    # 4. Any prompt the SDK still reaches raises instead of waiting on stdin (D-09). Confirm and Acknowledge consult
    # the task's bound prompt policy first (answer_prompts, only around a removal, D-12); with none bound, or for a
    # prompt the policy does not know, they refuse exactly as Ask always does.
    ExegolRich.Ask = staticmethod(_refuse_prompt("ask a question"))
    ExegolRich.Confirm = staticmethod(_answer_or_refuse("confirm"))
    ExegolRich.Acknowledge = staticmethod(_answer_or_refuse("acknowledge"))
    # 5. Image pull progress goes to the calling request's own progress sink, not a rich progress bar (D-25).
    ExegolTUI.downloadDockerLayer = staticmethod(download_docker_layer)
    # 6. Bound the startup-status READY wait at the enclosing service deadline (D-18). The startup mirror (step 11)
    # polls in a worker thread, so the loop stays free; this bound keeps the READY wait inside the service deadline,
    # ending it mirrors the SDK's Ctrl+C skip path (the container keeps starting in the background) and it ends an
    # abandoned poll thread. Only the name the model module imported is replaced (the startup reads it);
    # exegol.utils.ContainerLogStream is left untouched for other users.
    _exegol_container_module.ContainerLogStream = _DeadlineContainerLogStream
    # 7. A profile or Exegol config with a hostname/comment "ask user" flag must never prompt. The helper asks only
    # when stdin is a TTY, which it is when the HTTP server was launched from a terminal; reporting "cannot answer"
    # makes it keep the value that would otherwise apply (Pitfall 3). Only the name the helper module imported is
    # replaced; exegol.console.ExegolPrompt.stdinCanAnswer is left untouched for other users.
    _profile_ask_user_module.stdinCanAnswer = _stdin_cannot_answer
    # 8. A Docker 409 name conflict during exegol-mcp's own create must never reach Exegol's createContainer rollback,
    # which would remove another process's same-name container (WR-03, Pitfall 2). docker-py's create is replaced by
    # a wrapper that raises ContainerNameConflictError instead, only inside create_conflict_guard(); outside
    # exegol-mcp's create, behaviour is identical. Idempotent: the wrapper always delegates to the import-time original.
    docker.models.containers.ContainerCollection.create = _guarded_containers_create
    # 9. Exegol's host-side Sentinel shred (FsUtils.secure_remove) runs synchronously. While an exegol-mcp removal runs
    # (removal_offload), it goes to a worker thread so the loop stays free and the removal deadline applies (WR-09).
    # Only the name the model module imported is replaced (__removeVolume reads it at call time);
    # exegol.utils.FsUtils.secure_remove is left untouched. Idempotent: the wrapper delegates to the import-time
    # original.
    _exegol_container_module.secure_remove = _secure_remove_offloaded
    # 10. Exegol's stop runs docker-py's stop inline inside an async def, so the loop stalled for up to the 10 s stop
    # timeout and container_op_timeout_s could not fire (INT-01). The class attribute is replaced by a mirror that
    # moves only the docker calls to a worker thread; stop_container and the SDK's own callers (remove) get it with no
    # call-site change. Idempotent: the same function is assigned each time.
    _exegol_container_module.ExegolContainer.stop = _stop_offloaded  # type: ignore[method-assign]
    # 11. Exegol's __start_container runs docker-py's start and polls the startup logs with time.sleep(1) inline
    # inside an async def, so every other request stalled until READY and container_op_timeout_s could not fire
    # (INT-01). Replacing the mangled name covers every SDK caller of the startup (start, postCreateSetup, removal's
    # fallback restart).
    # Idempotent: the same function is assigned each time.
    _exegol_container_module.ExegolContainer._ExegolContainer__start_container = (  # type: ignore[attr-defined]
        _start_container_offloaded
    )
    # 12. Exegol's start reads the running state and runs the pre-start host checks inline before awaiting the step 11
    # startup; the mirror moves those to worker threads and runs the post-start setup with exec offloaded.
    _exegol_container_module.ExegolContainer.start = _start_offloaded  # type: ignore[method-assign]
    # 13. Exegol's remove ends with the docker container removal, the cache entry, the backup removals and the network
    # removals, all synchronous inside an async def, so remove_container and cleanup_containers stalled every other
    # request while Docker worked (INT-01). The mirror keeps Exegol's order (volume step, stop, container, cache,
    # backups, networks) and runs the whole docker tail as one worker hop. Idempotent: the same function is assigned
    # each time.
    _exegol_container_module.ExegolContainer.remove = _remove_offloaded  # type: ignore[method-assign]
    # 14. Exegol's postCreateSetup uploads the entrypoint with docker-py's put_archive, starts the container and runs the
    # password update's exec, all synchronous inside an async def, so create_container stalled every other request while
    # Docker worked and start_timeout_s could not fire (INT-01). The mirror keeps Exegol's statement sequence and runs
    # each docker call in a worker thread (the start through the step 11 mirror, the password exec through the quiet
    # daemon mode of _offloaded_exec). Idempotent: the same function is assigned each time.
    _exegol_container_module.ExegolContainer.postCreateSetup = _post_create_setup_offloaded  # type: ignore[method-assign]


def set_accept_eula(accepted: bool) -> None:
    """Record the operator's --accept-eula flag so OptionKey.ACCEPT_EULA resolves to it (D-05).

    Raises:
        RuntimeError: If patch_exegol_sdk() has not run yet.
    """
    if _parameters is None:
        raise RuntimeError("patch_exegol_sdk() must run before set_accept_eula()")
    _parameters.set_parameter("accept_eula", accepted)


@contextmanager
def cli_tier(values: Mapping[str, Any]) -> Iterator[None]:
    """Set CLI parameters for the duration of the block, as if typed on `exegol start`, then restore them.

    Exactly the previous value of each key set here is restored in `finally`, also when the block raises. It never
    calls reset_all_parameters(), which would revoke the operator's --accept-eula. The mocked parameters are a
    process singleton, so callers must serialize their blocks (lifecycle_sdk's create lock).

    Raises:
        RuntimeError: If patch_exegol_sdk() has not run yet.
        ValueError: If a key is not a CLI dest the headless mock declares (nothing is set then).
    """
    parameters = _parameters
    if parameters is None:
        raise RuntimeError("patch_exegol_sdk() must run before cli_tier()")
    unknown = sorted(key for key in values if not key or key not in _UNSET_CLI_DESTS)
    if unknown:
        raise ValueError(f"cli_tier() refuses CLI parameters the headless setup does not declare: {unknown}")
    previous = {key: getattr(parameters, key) for key in values}
    try:
        for key, value in values.items():
            parameters.set_parameter(key, value)
        yield
    finally:
        for key, value in previous.items():
            parameters.set_parameter(key, value)
