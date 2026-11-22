"""Container creation through Exegol's own model objects, with no human step (LIFE-01, D-09).

Choreography (shaped on exegol's ExegolManager.__createContainer, without its prompts and recap): set the agent's
values on the mocked CLI tier as if typed on `exegol start`, resolve the installed image, resolve the hostname the
way the wrapper does, build the model with ExegolContainerTemplate.newContainer (ContainerConfig.configFromUser
reads every option through OptionResolver), stamp the managed labels, create the container with
DockerUtils.createContainer and start it with postCreateSetup.

Lock order: the create lock first, then real.py's DockerUtils cache lock (only around the image lookup), never the
reverse. The create lock serializes every create because OptionResolver and the mocked ParametersManager are
process singletons: two creates running at once would read each other's CLI values (Pitfall 1).

Name conflicts (Pitfall 2, WR-03): Exegol's createContainer rolls back on any Docker API error by removing every
container whose name matches exactly, so a conflict must never reach it. The lock only serializes this process, so
the name is checked at the start of the lock and again right before createContainer (_final_create_checks), and
createContainer runs under sdk_headless.create_conflict_guard(): a Docker 409 during exegol-mcp's create is raised
as ContainerNameConflictError, which skips Exegol's rollback, and is refused as NAME_IN_USE_MESSAGE. The same final
check re-resolves a host workspace, so one replaced by a symlink since the service validated it is refused (WR-04,
defense in depth). Accepted residual, T-04-59: between the final check and Docker's create, Exegol's model.prepare()
runs. It creates the workspace directory when missing, and when the operator's update_fs_perms option
(auto_update_workspace_fs) is on or Exegol just created the directory, it applies FsUtils.setGidPermission to it: a
recursive setgid chmod of its directories that follows a symlink, a host workspace included. A non-409 error at the
create still runs Exegol's rollback; no 409-path container removal remains.

Shield rationale: run_blocking abandons its worker thread on cancellation, so an unshielded cancel would release
the lock while createContainer still runs, and could leave the CLI tier installed for the next caller. The whole
section, the post-create start included, runs in a cancellation-shielded scope: a cancelled create finishes and
restores the tiers before the cancel is delivered (research A6). The start keeps its own deadline
(spec.start_timeout_s), which the shield does not hide from the SDK's deadline-aware startup wait.

Threading follows real.py: createContainer and the name lookup are synchronous docker calls and run in a worker
thread; getInstalledImage, ask_user_for_value and newContainer are SDK coroutines awaited on the loop. postCreateSetup
is the sdk_headless patch step 14 mirror: also awaited on the loop, with its entrypoint upload, start and password
update in worker threads and its status display on the loop.

Profiles (LIFE-02, D-01, D-05): with spec.profile, the profile is loaded inside the create lock through
adapter/profiles.load_allowed_profile (licence-gated, files re-read, in a worker thread) and installed with Exegol's
own apply_profile on the loop, before the image is resolved, exactly as `exegol start --profile` installs it. The
spec's safe-set values stay on the CLI tier, which outranks the profile tier: scalars replace the profile's value and
ports/envs append to it (Exegol's merge policies). The image is the call's, else the profile's image.tag. GUI and
desktop are left to the profile and the operator's Exegol config; only a no-profile create forces them off (D-25).
Both profile tiers are uninstalled in `finally`, so they never outlive the create that installed them (Pitfall 1).
Accepted residual: OptionResolver is process-global, so while a profile create holds the tiers, a concurrent
non-create reader (list_images, for one) can briefly see that profile's image.custom_images. No other container is
shaped by the profile: every create is serialized by the lock.

Prompts during a create (research A5): Exegol acknowledges some warnings with ExegolRich.Acknowledge, for example the
VPN DNS warning of a profile's OpenVPN file. The whole create section binds creation_policy(), which acknowledges
those and answers nothing else, so Confirm stays refused (D-09); outside a create every prompt is refused again.

Removal (LIFE-03, shaped on the wrapper's ExegolManager remove path, without its prompts): read the backup container
ids, the private workspace path and the Sentinel directory off the loop, then await
ExegolContainer.remove(backup_history=...). sdk_headless patch step 13 replaced remove with a mirror that keeps
Exegol's order and prompts and runs its docker tail (container, cache entry, backups, networks) as one worker-thread
hop. Its prompts are answered from the keep options by a ContextVar-scoped policy (sdk_headless answer_prompts /
removal_policy). container_only
is never passed: it would also skip the network and backup cleanup. Exegol's own order is kept (CR-01): the volume
step (Sentinel history, then the private workspace) runs while the container still runs, then Exegol stops it
itself. The server never pre-stops: that only forced Exegol's root-owned-file fallback to restart the container.
sdk_headless.removal_offload moves Exegol's in-container delete (`rm -rf /workspace` and the in-container Sentinel
shred) and its stop into worker threads, and ends a fallback restart's READY wait at once. Patch step 9 does the same
for Exegol's host-side Sentinel shred (FsUtils.secure_remove) while this removal runs (WR-09). The whole sequence is
cancellation-shielded so a started removal finishes (Pitfall 9), and anyio.move_on_after(timeout_s) INSIDE the
shield bounds Exegol's removal (execution.container_op_timeout_s): the deadline interrupts an offloaded wait and
gives any startup wait a stop time. The deadline is read from that scope's cancelled_caught, never from an exception
type, so an operating-system timeout is reported as a host error. The docker tail is one worker hop, so a deadline
that fires during it is delivered: the call then reports REMOVAL_TIMEOUT_MESSAGE while the abandoned worker finishes
the tail (no half-removed networks or backups), and list_containers shows the result. A removal that completed
without the deadline being delivered is still reported as done (WR-08). The host state is checked after that scope,
still shielded, bounded by _HOST_CHECK_TIMEOUT_S. Accepted residuals: T-04-46, a removal that runs past its deadline
after Exegol deleted the workspace leaves the container listed (the error names remove_container, and a second call
finishes it); T-04-47, the on-loop residual of a removal, exactly: (a) Exegol's host-side delete of the private
workspace (its rmtree, rmdir and directory listing, T-04-19) and the Sentinel directory steps of __removeVolume;
(b) the two running-state reads (`self.isRunning()`, a synchronous docker reload) in the
PermissionError fallbacks of __removeVolume, which Exegol calls synchronously inside its coroutine, so offloading them
would mean mirroring __removeVolume. That is at most two docker round trips, only when the host refuses the Sentinel
shred or the workspace delete, and a daemon hung at that instant holds the loop for up to the docker-py client
timeout (T-05.1-25). Every other docker call of a removal runs in a worker thread: the running-state reads of
removal_offload's exec and stop shadows, a fallback restart's start (patch step 11), the in-container deletes, the
stop, and the container, backup and network removals (patch step 13's single hop). The host Sentinel shred is not
part of the residual (patch step 9 offloads it). The outcome reports the host state after the
removal, checked off the loop, never the request (WR-06): Exegol deletes an empty private workspace and a Sentinel
directory with no history even when keeping them was asked (they hold no data, T-04-49), and keeps a WSL workspace
or one it lacks the rights to delete.
"""

import os
from pathlib import Path
from typing import Any, NamedTuple

import anyio
import anyio.lowlevel
import docker.errors
import requests.exceptions  # type: ignore[import-untyped]
from exegol.config.OptionResolver import OptionKey, OptionResolver
from exegol.exceptions.ExegolExceptions import ObjectNotFound
from exegol.model.ExegolContainerTemplate import ExegolContainerTemplate
from exegol.profile.ProfileApplication import apply_profile
from exegol.profile.ProfileAskUser import ask_user_for_value
from exegol.utils.DockerUtils import DockerUtils

from exegol_mcp.adapter import sdk_headless
from exegol_mcp.adapter.errors import ContainerNameConflictError
from exegol_mcp.adapter.exec import _DOCKER_HANDLE
from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.port import (
    IMAGE_NOT_INSTALLED_MESSAGE,
    IMAGE_REQUIRED_MESSAGE,
    MANAGED_LABEL,
    MANAGED_LABEL_VALUE,
    NAME_IN_USE_MESSAGE,
    REMOVAL_TIMEOUT_MESSAGE,
    RUN_ID_LABEL,
    CreateSpec,
    RemovalOutcome,
)
from exegol_mcp.adapter.profiles import load_allowed_profile

# Process-wide, like the singletons it protects. anyio's lock binds to an event loop lazily.
_create_lock = anyio.Lock()


def _cli_values(spec: CreateSpec) -> dict[str, Any]:
    """Map a spec to the CLI dests `exegol start` would set, leaving every unset field to Exegol's own fallbacks.

    The image tag is set only when the call names one, so a profile's image.tag can apply. Without a profile, GUI
    (the host X11/Wayland socket) and desktop are forced off: an agent never gets the host display on its own (D-25,
    D-04). With a profile both are left unset, so the profile and the operator's Exegol config decide, as
    `exegol start --profile` would. ports and envs are lists, as argparse would hand them over.
    """
    values: dict[str, Any] = {}
    if spec.image:
        values["imagetag"] = spec.image
    if spec.profile is None:
        values["gui"] = False
        values["desktop"] = False
    if spec.workspace is not None:
        values["workspace_path"] = spec.workspace
    if spec.network_mode is not None:
        values["network"] = spec.network_mode
    if spec.ports:
        values["ports"] = list(spec.ports)
    if spec.env:
        values["envs"] = list(spec.env)
    if spec.hostname is not None:
        values["hostname"] = spec.hostname
    if spec.comment is not None:
        values["comment"] = spec.comment
    return values


def container_labels(container: Any) -> dict[str, str]:
    """Return the full Docker label map of an ExegolContainer, read from its docker-py handle (research Pattern 3).

    Never read from ContainerConfig.getLabels: Exegol's label parser drops every key outside its own metadata and
    feature keys, the managed labels included. Labels are static for the container's lifetime, so they survive a
    server or daemon restart (LIFE-06). An empty dict when the handle (exec.py's private attribute) or its labels
    are missing, so a listing never fails on one odd container.
    """
    docker_container = getattr(container, _DOCKER_HANDLE, None)
    labels = getattr(docker_container, "labels", None) if docker_container is not None else None
    return dict(labels) if labels else {}


def managed_identity(labels: dict[str, str]) -> tuple[bool, str | None]:
    """Return (managed, run_id) from a label map: managed only for an exact MANAGED_LABEL_VALUE (D-10).

    Exact string equality only, never lowercased or stripped: "True", "1" or " true" are a hand-made label, not
    exegol-mcp's. run_id is reported only for a managed container.
    """
    managed = labels.get(MANAGED_LABEL) == MANAGED_LABEL_VALUE
    return managed, labels.get(RUN_ID_LABEL) if managed else None


def creation_policy() -> sdk_headless.PromptPolicy:
    """Answer the prompts a create may reach: acknowledge every Acknowledge, answer nothing else (research A5).

    An acknowledgement carries no decision (Exegol's VPN DNS warning, for one), so refusing it would only fail a
    headless create. Confirm gets None and stays refused (D-09); Ask never reaches a policy.
    """

    def answer(kind: str, message: str) -> bool | None:
        return True if kind == "acknowledge" else None

    return answer


def _name_in_use(name: str) -> bool:
    try:
        DockerUtils().getContainer(name)
    except ObjectNotFound:
        return False
    return True


def _final_create_checks(spec: CreateSpec) -> None:
    """Re-check, right before createContainer, what may have changed since the lock was taken (WR-03, WR-04).

    The name: another process (a second server, the exegol wrapper) can have created it meanwhile. The workspace:
    the service resolved it to a realpath on the allowlist, so it must still resolve to itself; a directory replaced
    by a symlink would bind wherever the link points.

    Raises:
        ValueError: The name is in use (NAME_IN_USE_MESSAGE), or the workspace no longer resolves to itself.
    """
    if _name_in_use(spec.name):
        raise ValueError(NAME_IN_USE_MESSAGE.format(name=spec.name))
    if spec.workspace is not None and os.path.realpath(spec.workspace) != spec.workspace:
        raise ValueError(
            f"Workspace '{spec.workspace}' no longer resolves to itself (it was replaced by a symlink since it was "
            "checked), so it is refused. Omit workspace to get a private Exegol workspace."
        )


async def create_managed_container(spec: CreateSpec, *, cache_lock: anyio.Lock) -> Any:
    """Create, label and start one Exegol container from the spec, and return the ExegolContainer.

    Args:
        spec: The validated create request.
        cache_lock: real.py's DockerUtils cache lock, taken around the installed-image lookup (passed in to avoid
            a circular import).

    Raises:
        ValueError: No image from the call or the profile (IMAGE_REQUIRED_MESSAGE), the name is already used
            (NAME_IN_USE_MESSAGE: at the start of the lock, in _final_create_checks right before the create, or by
            Docker's 409 during the create, which never reaches Exegol's rollback), the image is not installed
            (IMAGE_NOT_INSTALLED_MESSAGE), Exegol cannot use the profile (its reason, from load_allowed_profile), or
            the host workspace was replaced by a symlink since the service checked it.
        RuntimeError: The host's licence does not grant profiles (PROFILE_LICENCE_MESSAGE), the container was
            created but did not finish starting within spec.start_timeout_s, or the SDK reported a critical error
            (such as a profile host path Exegol refuses in apply_profile).
    """
    # Without a profile nothing else can name the image: refuse before taking the lock (D-07).
    if not spec.image and spec.profile is None:
        raise ValueError(IMAGE_REQUIRED_MESSAGE)
    with anyio.CancelScope(shield=True), sdk_headless.answer_prompts(creation_policy()):
        async with _create_lock:
            # 1. Check the name inside the lock: Exegol's createContainer rollback removes an exact-name match on any
            # Docker API error, so a conflict must never reach it (Pitfall 2). Checked again right before step g.
            if await run_blocking(_name_in_use, spec.name):
                raise ValueError(NAME_IN_USE_MESSAGE.format(name=spec.name))
            # 2. The agent's values on the CLI tier for the whole build, restored exactly afterwards.
            with sdk_headless.cli_tier(_cli_values(spec)):
                try:
                    # a. The profile, inside the lock and the tier window only (D-01, Pitfall 1): loaded fresh and
                    # licence-gated off the loop, then installed with Exegol's own apply_profile on the loop.
                    if spec.profile is not None:
                        profile = await run_blocking(load_allowed_profile, spec.profile)
                        apply_profile(profile)
                    # b. The image: the call's, else the profile's image.tag; never picked by the server (D-07).
                    image_tag = spec.image or OptionResolver().get(OptionKey.IMAGE_TAG)
                    if not image_tag:
                        raise ValueError(IMAGE_REQUIRED_MESSAGE)
                    # c. The installed image only: the adapter never pulls one (D-07).
                    async with cache_lock:
                        docker = await run_blocking(DockerUtils)
                        try:
                            image = await docker.getInstalledImage(image_tag)
                        except RuntimeError as e:
                            raise ValueError(IMAGE_NOT_INSTALLED_MESSAGE.format(image=image_tag)) from e
                    # d. The hostname as the wrapper resolves it; it never prompts after patch_exegol_sdk() step 7.
                    hostname = await ask_user_for_value(
                        OptionKey.HOSTNAME_ASK_USER,
                        OptionKey.HOSTNAME,
                        "Enter the container hostname",
                        fallback_default=f"exegol-{spec.name}",
                    )
                    # e. Exegol's own model: configFromUser reads every option through OptionResolver.
                    model = await ExegolContainerTemplate.newContainer(spec.name, image, hostname=hostname)
                    # f. The managed labels (D-09), before the container exists.
                    for key, value in spec.labels.items():
                        model.config.addLabel(key, value)
                    # g. The final checks (WR-03, WR-04), then the create (synchronous docker calls) off the loop. The
                    # guard reaches the worker thread through anyio's copied context: a Docker 409 skips Exegol's
                    # rollback, so another process's same-name container is never removed.
                    await run_blocking(_final_create_checks, spec)
                    try:
                        with sdk_headless.create_conflict_guard():
                            container = await run_blocking(DockerUtils().createContainer, model)
                    except ContainerNameConflictError:
                        raise ValueError(NAME_IN_USE_MESSAGE.format(name=spec.name)) from None
                    # h. The post-create setup (sdk_headless patch step 14): the entrypoint upload, the start and the
                    # password update run in worker threads, the status display (its console lock) on the loop; the
                    # whole step is bounded by spec.start_timeout_s.
                    try:
                        with anyio.fail_after(spec.start_timeout_s):
                            await container.postCreateSetup()
                    except TimeoutError:
                        raise RuntimeError(
                            f"Container '{spec.name}' was created but did not finish starting within "
                            f"{spec.start_timeout_s:g} s. It may still be starting: call list_containers to check "
                            "its status."
                        ) from None
                finally:
                    # Uninstall the profile tiers: the wrapper never tears them down, the server must (Pitfall 1).
                    OptionResolver().loadProfile(None)
                    OptionResolver().loadProfileUserConfig(None)
    return container


# Private volume path prefixes of a Docker volume defined from WSL: Exegol never deletes one (__removeVolume).
_WSL_VOLUME_PREFIXES = ("/wsl/", "\\wsl\\")


class _HostData(NamedTuple):
    """Where a container's workspace and Sentinel history live on the host, read before its removal (WR-06, IN-10)."""

    # Exegol's private volume path, "" when the container has none (a host-mounted or a disabled workspace).
    volume_path: str
    # The host directory mounted as the workspace when config.isWorkspaceCustom() is true, else None.
    host_workspace: str | None
    # The Sentinel audit history directory when Sentinel logging is enabled, else None.
    sentinel_dir: Path | None


def _host_data_paths(container: Any) -> _HostData:
    """Return where the container's workspace and Sentinel history live on the host (WR-06, IN-10).

    Read before Exegol's remove(), from the same ContainerConfig getters its __removeVolume reads. The private volume
    path is "" both for a host-mounted workspace and for one Exegol disabled (no /workspace mount): only
    isWorkspaceCustom() tells them apart. getHostWorkspacePath() is read only for a custom workspace, because for a
    disabled one it returns the literal "not found :(". A container whose workspace Exegol disabled therefore has
    neither path, and reports no workspace kept and host_workspace false.
    """
    config = container.config
    sentinel_dir = config.getSentinelPath() if config.isSentinelEnable() else None
    volume_path = config.getPrivateVolumePath()
    host_workspace = config.getHostWorkspacePath() if volume_path == "" and config.isWorkspaceCustom() else None
    return _HostData(volume_path=volume_path, host_workspace=host_workspace, sentinel_dir=sentinel_dir)


# The host check after a removal is bounded on its own, so a stalled host filesystem never holds a shielded caller.
_HOST_CHECK_TIMEOUT_S = 5.0


def _exists_or_kept(path: Path) -> bool:
    """Return whether path exists, counting a path the host cannot check as still there (WR-08).

    Python 3.10's Path.exists() raises PermissionError on EACCES. A removal must never report data as gone when the
    host could not say, so any OSError fails toward "still on the host".
    """
    try:
        return path.exists()
    except OSError:
        return True


def _host_data_remaining(data: _HostData) -> tuple[bool, bool]:
    """Return (workspace still on the host, Sentinel history still on the host) after a removal (WR-06, IN-10).

    A WSL workspace is never deleted by Exegol. Any other private workspace and the Sentinel directory are checked on
    the host: Exegol deletes an empty workspace and a history-less Sentinel directory even when keeping them was
    asked, and keeps one it lacks the rights to delete. A host-mounted workspace is never deleted, so it is kept
    while its directory exists. A container whose workspace Exegol disabled has none to keep. A path the host cannot
    check is reported as still there (WR-08).
    """
    if data.volume_path.startswith(_WSL_VOLUME_PREFIXES):
        workspace_remaining = True
    elif data.volume_path:
        workspace_remaining = _exists_or_kept(Path(data.volume_path))
    elif data.host_workspace is not None:
        workspace_remaining = _exists_or_kept(Path(data.host_workspace))
    else:
        workspace_remaining = False
    return workspace_remaining, data.sentinel_dir is not None and _exists_or_kept(data.sentinel_dir)


def _unknown_remaining(data: _HostData) -> tuple[bool, bool]:
    """The outcome when the host check did not answer within _HOST_CHECK_TIMEOUT_S: whatever could exist, does."""
    return data.volume_path != "" or data.host_workspace is not None, data.sentinel_dir is not None


async def remove_managed_container(
    container: Any, keep_workspace: bool, keep_logs: bool, *, timeout_s: float
) -> RemovalOutcome:
    """Remove one ExegolContainer through Exegol's own remove() and report what is still on the host (LIFE-03).

    The outcome reports the host state after the removal, not the request (WR-06, D-12): workspace_kept is True
    only when the workspace still exists (host-mounted, WSL, kept on request, or one Exegol could not delete), and
    logs_kept only when a Sentinel directory still exists. Exegol deletes an empty private workspace and a Sentinel
    directory with no history even when keeping them was asked; both hold no data and are reported as not kept. A
    host-mounted workspace (config.isWorkspaceCustom()) is reported as host_workspace and kept while it exists; a
    container whose workspace Exegol disabled reports no workspace kept and host_workspace false (IN-10).

    The deadline is read from its own scope (cancelled_caught), never from an exception type (WR-08). Exegol's
    docker tail runs as one worker hop (patch step 13), so a deadline that fires during it is delivered and reported
    as REMOVAL_TIMEOUT_MESSAGE while the abandoned worker finishes the tail; a removal that completed without the
    deadline being delivered is reported as done. The host check runs after that scope, still shielded, and
    is bounded by _HOST_CHECK_TIMEOUT_S; an unreadable path or a check that does not answer counts as still there.

    Args:
        container: The ExegolContainer to remove (the caller resolved it and enforces the managed-only policy).
        keep_workspace: Keep a private workspace on the host. A host-mounted workspace is never deleted.
        keep_logs: Keep the Sentinel audit history.
        timeout_s: The bound on Exegol's removal (execution.container_op_timeout_s). A cancelled caller waits for
            the removal to finish or reach it, plus at most _HOST_CHECK_TIMEOUT_S for the host check.

    Raises:
        ValueError: Docker no longer knows the container (docker.errors.NotFound): another process removed it while
            this removal ran. The message names list_containers.
        RuntimeError: The removal reached timeout_s before it finished (REMOVAL_TIMEOUT_MESSAGE), the SDK reported a
            critical error (such as a docker timeout while removing the container), any other docker or requests
            error (docker.errors.APIError, requests.exceptions.ReadTimeout), or a host OSError (an operating-system
            timeout included), whose detail the message carries. No raw docker or OS exception leaves this
            function (WR-01, WR-08).
    """
    with anyio.CancelScope(shield=True):
        # The deadline sits inside the shield, so the removal stays bounded and _deadline_stop_time() sees it. It is
        # bound before the try so its scope can be read whatever happens inside.
        deadline = anyio.move_on_after(timeout_s)
        try:
            with deadline:
                # 1. The backup containers Exegol keeps for this one, removed with it as the wrapper does.
                backups = [entry[0] for entry in await run_blocking(container.getExistingBackupContainers)]
                # 2. Where the workspace and the Sentinel history live, read before Exegol deletes anything (WR-06).
                # A host-mounted workspace is never deleted by Exegol (D-12); a disabled one has no path (IN-10).
                data = await run_blocking(_host_data_paths, container)
                # 3. Exegol's own removal (patch step 13) in its own order: workspace and Sentinel while the container
                # runs, then its stop, the container, backups and networks. Its docker tail runs as one worker hop;
                # its prompts are answered from the keep options and its in-container delete and stop run off the
                # loop through removal_offload, for this removal only.
                with (
                    sdk_headless.answer_prompts(sdk_headless.removal_policy(keep_workspace, keep_logs)),
                    sdk_headless.removal_offload(container),
                ):
                    await container.remove(backup_history=backups)
        except docker.errors.NotFound:
            # Checked before RequestException, which it subclasses (through APIError and requests' HTTPError).
            raise ValueError(
                f"Container '{container.name}' not found: it was removed while this removal ran. Call "
                "list_containers to see the existing containers."
            ) from None
        except requests.exceptions.RequestException as e:
            # docker's APIError and requests' timeouts from the docker calls Exegol and removal_offload make. Checked
            # before OSError, which it subclasses through IOError.
            detail = getattr(e, "explanation", None) or str(e) or type(e).__name__
            raise RuntimeError(f"Docker failed while removing container '{container.name}': {detail}") from None
        except OSError as e:
            # A host error, an operating-system timeout (ETIMEDOUT) included: never the server's deadline (WR-08).
            detail = str(e) or type(e).__name__
            raise RuntimeError(f"Removing container '{container.name}' failed on the host: {detail}") from None
        if deadline.cancelled_caught:
            # The deadline was delivered before Exegol's removal finished: a retry finishes it.
            raise RuntimeError(REMOVAL_TIMEOUT_MESSAGE.format(name=container.name, timeout=timeout_s)) from None
        # 4. Outside the deadline: the removal is done, so its outcome must never fail it. What is really left on the
        # host is checked off the loop (WR-06), bounded on its own, and fails toward "still on the host" (WR-08).
        with anyio.move_on_after(_HOST_CHECK_TIMEOUT_S) as host_check:
            workspace_remaining, logs_remaining = await run_blocking(_host_data_remaining, data)
        if host_check.cancelled_caught:
            workspace_remaining, logs_remaining = _unknown_remaining(data)
    # A cancelled caller gets its cancellation now that the removal is complete, not an outcome.
    await anyio.lowlevel.checkpoint_if_cancelled()
    return RemovalOutcome(
        workspace_kept=workspace_remaining,
        logs_kept=logs_remaining,
        host_workspace=data.host_workspace is not None,
    )
