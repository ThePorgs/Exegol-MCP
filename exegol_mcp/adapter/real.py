"""RealExegolAdapter: the ExegolPort implementation over the exegol 6.x SDK.

Threading rule (RESEARCH Pitfall 9):
- Synchronous SDK and docker work (constructing DockerUtils, which connects to the daemon; container lookup and
  state reads; the container listing; cache resets; the demuxed exec) runs in a worker thread through
  run_blocking(), so a slow daemon never stalls other requests.
- ExegolContainer.start, stop, postCreateSetup and remove are replaced at startup by sdk_headless mirrors (patch
  steps 10-14). They keep the SDK's status display and its loop-bound console lock on the event loop and run each
  docker-py call (the start, the startup log polls, the stop, put_archive, the removal's docker tail) in a worker
  thread through run_blocking(), so a slow daemon never stalls other requests and container_op_timeout_s fires on
  time. The abandoned worker thread finishes on its own, as for exec. A whole SDK coroutine never runs in a worker
  thread, because the console lock is loop-bound and a thread would bind it to the wrong loop.
- DockerUtils.listImages/getOfficialImageFromList/downloadImage/downloadVersionTag, which show an ExegolStatus or
  ExegolProgress, are still awaited on the event loop (INT-02, out of scope for the container mirrors).

Residuals of the container mirrors (accepted): for GUI containers the post-start setup's host xhost / xauth
subprocesses run on the loop (normally milliseconds); Exegol's removal volume step keeps its host-side workspace and
Sentinel steps and the two running-state reads of its PermissionError fallbacks on the loop (documented under
T-04-47). A start or stop holds the SDK's console lock for its whole docker wait, so a concurrent status user (a stop
of another container, list_images, a removal's Sentinel step) queues behind it within its own deadline, and a
list_images waiting there holds _sdk_cache_lock, so a list_containers issued behind it waits too. tools/list and calls
that take neither lock are unaffected. Abandoned worker threads hold default-limiter tokens until docker returns
(the docker-py client timeout is 300 s).

Cache rule (CR-02): the DockerUtils singleton keeps one process-wide container and image cache, which
listContainers resets and fills in several steps and clearCache nulls from any thread. Every clearCache +
listContainers / listImages / getOfficialImageFromList sequence therefore runs under _sdk_cache_lock, taken on the
event loop, inside a cancellation-shielded scope. Exec, start, stop and container lookups (getContainer does not
use the cache) never take the lock, so a long command and a listing never wait on each other.

exegol exceptions are translated here and never leave the adapter: an unknown container or image raises
ValueError, a container in the wrong state raises RuntimeError. The SDK's critical errors already surface as
RuntimeError once patch_exegol_sdk() has run. An image that is already installed and up to date is not pulled.
A download exegol reports as failed raises RuntimeError.
"""

import os
from collections.abc import Coroutine
from typing import Any, Literal, TypeVar

import anyio
from exegol.exceptions.ExegolExceptions import ObjectNotFound
from exegol.utils.DockerUtils import DockerUtils

from exegol_mcp.adapter.exec import (
    exec_demuxed,
    kill_tagged_job,
    pump_tagged_job,
    read_tagged_job,
    start_tagged_job,
    sweep_tagged_jobs,
)
from exegol_mcp.adapter.files import list_container_directory, read_container_file, write_container_file
from exegol_mcp.adapter.lifecycle_sdk import (
    container_labels,
    create_managed_container,
    managed_identity,
    remove_managed_container,
)
from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.port import (
    IMAGE_NOT_DOWNLOADED_MESSAGE,
    IMAGE_UP_TO_DATE_MESSAGE,
    NOT_THIS_RUN_MESSAGE,
    UNMANAGED_REMOVAL_MESSAGE,
    ContainerState,
    CreateSpec,
    DirListing,
    FileRead,
    FileWrite,
    JobHandle,
    JobRead,
    RemovalOutcome,
)
from exegol_mcp.context.request import ProgressCallback, RequestState, bind_request, request_ctx
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.models.image import ImageInfo

T = TypeVar("T")

# The DockerUtils singleton and its container/image cache are process-wide, so this lock is process-wide too. It is
# taken on the event loop around every sequence that clears or fills that cache. anyio's lock binds to an event loop
# lazily, so one instance serves every event loop the process runs.
_sdk_cache_lock = anyio.Lock()


def _run_non_suspending(coro: Coroutine[Any, Any, T]) -> T:
    """Run an SDK coroutine that is async in signature only, to completion in the current (worker) thread.

    DockerUtils.listContainers is declared async but never awaits: it makes blocking docker calls. Driving it here
    keeps those calls off the event loop. If a later SDK release makes it really suspend, fail loudly instead of
    running it against the wrong loop.
    """
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value  # type: ignore[no-any-return]
    coro.close()
    raise RuntimeError(
        "An Exegol SDK call expected to complete synchronously suspended; this exegol release is not supported"
    )


def _fresh_docker() -> Any:
    """Return the DockerUtils singleton with its caches dropped.

    The singleton caches the container and image lists for the life of the process. That suits the one-shot
    wrapper, but a long-running server would otherwise never see containers or images changed after its first
    call. The first construction connects to the Docker daemon, so call this in a worker thread only. Callers must
    hold _sdk_cache_lock for the whole clear-and-refill sequence.
    """
    docker = DockerUtils()
    docker.clearCache()
    return docker


def _find_container(name: str) -> Any | None:
    try:
        return DockerUtils().getContainer(name)
    except ObjectNotFound:
        return None


def _container_state(name: str) -> ContainerState | None:
    container = _find_container(name)
    if container is None:
        return None
    # Managed status comes from the docker handle's labels, never from ContainerConfig (research Pattern 3).
    managed, run_id = managed_identity(container_labels(container))
    return ContainerState(running=container.isRunning(), managed=managed, run_id=run_id)


def _find_with_state(name: str) -> tuple[Any | None, bool]:
    container = _find_container(name)
    return container, container is not None and container.isRunning()


def _running_container_objects() -> list[Any]:
    """Return the raw ExegolContainer objects that are currently running (for the orphan sweep)."""
    containers = _run_non_suspending(_fresh_docker().listContainers())
    return [container for container in containers if container.isRunning()]


# Grace before SIGKILL when sweeping orphans at startup/shutdown. Orphans from a crashed server carry no state
# the server still tracks, so a short grace is enough to let a tool flush before it is force-killed (D-16).
_ORPHAN_SWEEP_GRACE_S = 5.0


def _features(config: Any) -> list[str]:
    features: list[str] = []
    if config.isGUIEnable():
        features.append("Console GUI")
    if config.isDesktopEnabled():
        features.append("Remote-Desktop")
    if config.isExegolResourcesEnable():
        features.append("Exegol-Resources")
    if config.isMyResourcesEnable():
        features.append("My-Resources")
    if config.isShellLoggingEnable():
        features.append("shell-logging")
    if config.isTimezoneShared():
        features.append("Shared-timezone")
    if config.isWrapperStartShared():
        # The wrapper's up-to-date spawn.sh is mounted into the container. v1 reported this as a second
        # "Exegol-Resources", duplicating the resources feature.
        features.append("Shared-wrapper-start")
    return features


def _container_info(container: Any) -> ContainerInfo:
    config = container.config
    network_driver, network_name = config.getNetwork()
    vpn_path = config.getVpnConfigPath()
    # Managed status comes from the docker handle's labels, never from config.getLabels (research Pattern 3).
    # this_run is the service's to set (04-04): the adapter does not know which server run is current.
    managed, run_id = managed_identity(container_labels(container))
    return ContainerInfo(
        name=container.name,
        creation_date=config.getCreationDate(),
        image_name=container.image.getName(),
        image_version=container.image.getImageVersion(),
        status=container.getRawStatus(),
        network_driver=network_driver,
        network_name=network_name,
        features=_features(config),
        devices=config.getDevices(),
        vpn=vpn_path.name if vpn_path else None,
        env=[e for e in config.getTextEnvs().split(os.linesep) if e],
        is_privileged=config.getPrivileged(),
        capabilities=["Docker default capabilities"] + config.getCapabilities(),
        comment=config.getComment(),
        managed=managed,
        run_id=run_id,
    )


def _list_containers() -> list[ContainerInfo]:
    containers = _run_non_suspending(_fresh_docker().listContainers())
    # Mapping reads each container's live status from the daemon, so it stays in this worker thread too.
    infos = [_container_info(container) for container in containers]
    return sorted(infos, key=lambda info: info.name)


def _image_info(image: Any) -> ImageInfo:
    return ImageInfo(
        name=image.getName(),
        version=image.getImageVersion(),
        latest_version=image.getLatestVersion(),
        is_installed=image.isInstall(),
        is_built_locally=image.isLocal(),
        is_up_to_date=image.isUpToDate(),
        build_date=image.getBuildDate(),
        license=image.getDisplayLicense(),
    )


class RealExegolAdapter:
    """ExegolPort over the exegol 6.x SDK. Requires patch_exegol_sdk() to have run first."""

    async def list_containers(self) -> list[ContainerInfo]:
        """Return every Exegol container on the host, sorted by name."""
        # run_blocking abandons its worker thread on cancellation. Without the shield, a cancelled call would release
        # the lock while its thread still resets and appends to the shared list, and the next caller would overlap it.
        async with _sdk_cache_lock:
            with anyio.CancelScope(shield=True):
                containers = await run_blocking(_list_containers)
        return containers

    async def get_container_state(self, name: str) -> ContainerState | None:
        """Return the live state of the named container, or None if no container has that name."""
        return await run_blocking(_container_state, name)

    async def _require_container(self, name: str) -> Any:
        container = await run_blocking(_find_container, name)
        if container is None:
            raise ValueError(f"Container '{name}' not found")
        return container

    async def start_container(self, name: str) -> bool:
        """Start the named container and return its running state afterwards."""
        container = await self._require_container(name)
        # 1. The patched SDK start: status display and console lock on the loop, the docker start and the startup
        # log polls in worker threads, so the service deadline is delivered.
        await container.start()
        # 2. Read the resulting state off the loop.
        return bool(await run_blocking(container.isRunning))

    async def stop_container(self, name: str) -> bool:
        """Stop the named container and return its running state afterwards (False once stopped)."""
        container = await self._require_container(name)
        # 1. The patched SDK stop: status display and console lock on the loop, the docker stop in a worker
        # thread, so the service deadline is delivered.
        await container.stop()
        # 2. Read the resulting state off the loop.
        return bool(await run_blocking(container.isRunning))

    async def list_images(self, scope: Literal["installed", "all"]) -> list[ImageInfo]:
        """Return the installed images, or every known image (installed and downloadable) for scope "all"."""
        async with _sdk_cache_lock:
            with anyio.CancelScope(shield=True):
                docker = await run_blocking(_fresh_docker)
                # listImages shows an ExegolStatus (console lock) and gathers SDK tasks: it stays on the loop.
                images = await docker.listImages(include_custom=True)
        # listImages returns a new filtered list, so filtering and mapping need not hold the lock.
        installed_only = scope == "installed"
        return [_image_info(image) for image in images if not installed_only or image.isInstall()]

    async def download_image(self, name: str, version: str, progress: ProgressCallback | None) -> bool:
        """Download or update an image, reporting layer progress to the callable. Return True once installed."""
        image_tag = name if version == "latest" else f"{name}-{version}"
        # 1. Resolve the image from the registry and local lists (on the loop: listImages, console lock), as one
        # locked cache section.
        async with _sdk_cache_lock:
            with anyio.CancelScope(shield=True):
                docker = await run_blocking(_fresh_docker)
                try:
                    image = await docker.getOfficialImageFromList(image_tag)
                except ObjectNotFound:
                    raise ValueError(f"Image '{image_tag}' not found") from None
                await run_blocking(docker.clearCache)

        # 2. Already installed and up to date: nothing to pull. exegol's downloadImage would skip the pull and
        # report False (updateCheck returns None), which reads as a failed download. Checking first also skips its
        # registry auth and license step for an image that is already here. Local and discontinued images count as
        # up to date, as in list_images. The report goes out after the locked section, never inside it.
        if image.isInstall() and image.isUpToDate():
            if progress is not None:
                await progress(100, 100, IMAGE_UP_TO_DATE_MESSAGE.format(image=image_tag))
            return True

        # 3. Pull, outside the cache lock so a long download never blocks listings. The SDK feeds the pull stream to
        # the progress hook patch_exegol_sdk() installed, which pulls each line off the loop and reports to the sink
        # bound here, for this request only.
        current = request_ctx.get()
        with bind_request(RequestState(request_id=current.request_id if current else None, progress=progress)):
            installed = await docker.downloadImage(image, install_mode=not image.isInstall())
        if not installed:
            # After the up-to-date check, a False from exegol means the pull was refused or failed (UNKNOWN_STATUS,
            # a registry HTTP 500 such as an unauthorized license, or a custom image). exegol already logged the
            # reason to stderr; the SDK text stays server-side and is never copied into this error (T-01-35).
            raise RuntimeError(IMAGE_NOT_DOWNLOADED_MESSAGE.format(image=image_tag))

        # 4. Keep the tags consistent, as the exegol wrapper does after an install.
        if image.isVersionSpecific():
            # A version-specific install also tags it as the local latest when no latest image is installed. The
            # lookup refills the cache, so it is a second locked section (sequential with step 1, never nested).
            async with _sdk_cache_lock:
                with anyio.CancelScope(shield=True):
                    try:
                        latest = await docker.getOfficialImageFromList(name)
                        needs_latest_tag = latest is None or not latest.isInstall()
                    except ObjectNotFound:
                        needs_latest_tag = True
            if needs_latest_tag:
                await run_blocking(docker.createLocalLastestImageTag, image)
        elif image.hasVersionTag():
            # A latest install also pulls its version-specific tag.
            result = await docker.downloadVersionTag(image)
            if isinstance(result, str):
                raise RuntimeError(f"Error while downloading the version tag of '{image_tag}': {result}")
        return True

    async def exec_command(self, name: str, command: str) -> ExecResult:
        """Run a shell command in the named, running container; return separate stdout, stderr and exit code."""
        # 1. Check that the container exists and is running, in one worker-thread hop.
        container, running = await run_blocking(_find_with_state, name)
        if container is None:
            raise ValueError(f"Container '{name}' not found")
        if not running:
            raise RuntimeError(f"Container '{name}' is not running")
        # 2. Run the demuxed exec off the loop (D-04).
        return await exec_demuxed(container, command)

    async def create_container(self, spec: CreateSpec) -> ContainerInfo:
        """Create, label and start a container through Exegol's own model, then return its info (LIFE-01)."""
        # The serialized, shielded SDK choreography lives in lifecycle_sdk; it takes the cache lock itself, after
        # its own create lock (never the reverse).
        container = await create_managed_container(spec, cache_lock=_sdk_cache_lock)
        return await run_blocking(_container_info, container)

    async def remove_container(
        self,
        name: str,
        keep_workspace: bool,
        keep_logs: bool,
        *,
        timeout_s: float,
        require_managed: bool = True,
        expected_run_id: str | None = None,
    ) -> RemovalOutcome:
        """Remove the named container through Exegol's own remove(), keeping what the call asks to keep (LIFE-03)."""
        container = await self._require_container(name)
        # The gate on the object that is removed (WR-02): the ExegolContainer is pinned to one Docker id, so these
        # labels are the ones of the container Exegol's remove() deletes, whatever an earlier lookup of the name saw.
        managed, run_id = managed_identity(container_labels(container))
        if require_managed and not managed:
            raise ValueError(UNMANAGED_REMOVAL_MESSAGE.format(name=name))
        if expected_run_id is not None and run_id != expected_run_id:
            raise ValueError(NOT_THIS_RUN_MESSAGE.format(name=name))
        # The backup lookup and the bounded, shielded, offloaded remove() live in lifecycle_sdk (CR-01).
        return await remove_managed_container(container, keep_workspace, keep_logs, timeout_s=timeout_s)

    async def _require_running(self, name: str) -> Any:
        """Return the running container object, or raise the same errors exec_command does."""
        container, running = await run_blocking(_find_with_state, name)
        if container is None:
            raise ValueError(f"Container '{name}' not found")
        if not running:
            raise RuntimeError(f"Container '{name}' is not running")
        return container

    async def read_file(self, name: str, path: str, max_bytes: int, timeout_s: float) -> FileRead:
        """Read a file in the named running container through the exec-based transfer (FILE-01)."""
        container = await self._require_running(name)
        return await read_container_file(container, path, max_bytes, timeout_s)

    async def write_file(
        self, name: str, path: str, data: bytes, overwrite: bool, mode: int | None, timeout_s: float
    ) -> FileWrite:
        """Write a file in the named running container through the staged archive and exec copy (FILE-02)."""
        container = await self._require_running(name)
        return await write_container_file(container, path, data, overwrite, mode, timeout_s)

    async def list_directory(
        self, name: str, path: str, depth: int, max_entries: int, timeout_s: float
    ) -> DirListing:
        """List a directory in the named running container with one bounded find exec (FILE-03)."""
        container = await self._require_running(name)
        return await list_container_directory(container, path, depth, max_entries, timeout_s)

    async def start_job(self, name: str, command: str, tag: str, cap_bytes: int) -> JobHandle:
        """Launch the tagged, process-grouped, streamed exec in the named running container (D-09/D-16)."""
        container = await self._require_running(name)
        return await start_tagged_job(container, name, command, tag, cap_bytes)

    async def pump_job(self, handle: JobHandle) -> None:
        """Drain the job into its buffers and record the real exit code when the stream ends."""
        await pump_tagged_job(handle)

    async def read_job(
        self, handle: JobHandle, stdout_offset: int, stderr_offset: int, max_bytes: int | None
    ) -> JobRead:
        """Serve a forward offset read of the job's two bounded buffers (D-08/D-22)."""
        return await read_tagged_job(handle, stdout_offset, stderr_offset, max_bytes)

    async def kill_job(self, name: str, tag: str, grace_s: float) -> None:
        """Kill the tagged process group in the named running container (SIGTERM, then SIGKILL after grace_s)."""
        container = await self._require_running(name)
        await kill_tagged_job(container, tag, grace_s)

    async def sweep_orphans(self) -> int:
        """Kill every tagged process across the host's running containers; return the total killed (D-11)."""
        # List running containers under the cache lock + shield, like the listing path, then sweep each.
        async with _sdk_cache_lock:
            with anyio.CancelScope(shield=True):
                containers = await run_blocking(_running_container_objects)
        total = 0
        for container in containers:
            total += await sweep_tagged_jobs(container, _ORPHAN_SWEEP_GRACE_S)
        return total
