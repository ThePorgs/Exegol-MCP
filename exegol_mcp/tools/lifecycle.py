"""MCP tool handlers for the container lifecycle. Thin wrappers: adapter from the lifespan, policy in the service."""

from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import Context

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.config import LifecycleSettings, ProfilesSettings
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.lifecycle import CleanupResult, RemovedContainer
from exegol_mcp.services import lifecycle as lifecycle_service
from exegol_mcp.tools.containers import _container_op_timeout
from exegol_mcp.tools.errors import tool_errors


def _lifecycle_settings(adapter: object) -> LifecycleSettings:
    config = getattr(adapter, "config", None)
    return config.lifecycle if config is not None else LifecycleSettings()


def _profiles_settings(adapter: object) -> ProfilesSettings:
    config = getattr(adapter, "config", None)
    return config.profiles if config is not None else ProfilesSettings()


async def create_container(
    image: str | None = None,
    profile: str | None = None,
    container_name: str | None = None,
    workspace: str | None = None,
    network_mode: Literal["host", "docker", "nat", "disabled"] | None = None,
    ports: list[str] | None = None,
    env: list[str] | None = None,
    hostname: str | None = None,
    comment: str | None = None,
    *,
    ctx: Context[ExegolPort, Any],
) -> ContainerInfo:
    """Create and start a new Exegol container from an installed image or an operator profile, with no prompt.

    IMPORTANT: The image must already be installed: pick one with list_images. The server never downloads an
    image here; install a missing one with download_image first. The container is created, started and its
    container info returned (managed=true, this_run=true). Names are lowercase ASCII (letters, digits, '_', '.',
    '-'); a leading 'exegol-' is dropped, and an omitted name is generated as mcp-<8 hex>. A name already in use is
    refused. The workspace is a private Exegol workspace unless you give a host directory the operator listed in
    the server's workspace allowlist (exactly that directory; '~' is the invoking user's home). Ports without an IP
    are published on 0.0.0.0 (every host interface); write 127.0.0.1:host_port:container_port to keep a port local.
    Each env entry must be KEY=VALUE. Re-using the name of a removed container whose private workspace was kept
    mounts those kept files again.

    PROFILES: profile names one of the Exegol container profiles the operator allowlisted for this server (exact,
    case-sensitive name); any other name is refused with the list of allowed ones. The operator may require a
    profile on every create. A profile can supply the image (then image may be omitted) and grants the
    capabilities only a profile can: privileged mode, devices, capabilities, extra mounts, VPN, desktop and the
    host display. The other arguments override the profile: image, workspace, network_mode, hostname and comment
    replace the profile's value, while ports and env are added to the profile's own. No argument can ever set
    privileged mode, devices, capabilities, mounts, VPN or desktop.

    RELATED TOOLS:
    - list_images: choose an installed image to create the container from
    - download_image: install an image that is not installed yet
    - list_containers: see the new container and which containers exegol-mcp created
    - exec_command: run commands in the new container
    - start_container: start the container again after it was stopped
    - remove_container: remove the container when you are done with it

    Args:
        image: Installed Exegol image name, as returned by list_images (for example "free"). Optional with a profile
            that names one.
        profile: An Exegol container profile the operator allowlisted; omit it to create without a profile.
        container_name: Name for the new container; omit it to get a generated one.
        workspace: Allowlisted host directory to mount as the workspace; omit it for a private workspace.
        network_mode: host, docker, nat or disabled; omit it for Exegol's default.
        ports: Ports to publish in Exegol's [ip:]host_port[-end][:container_port[-end]][:protocol] syntax.
        env: Environment variables as KEY=VALUE entries.
        hostname: Hostname of the container.
        comment: Free-text comment stored on the container.

    Returns:
        The new container's information, as list_containers reports it.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        settings = _lifecycle_settings(adapter)
        profiles = _profiles_settings(adapter)
        run_id = getattr(adapter, "run_id", None)
        if not run_id:
            raise RuntimeError("This server has no run id yet, so it cannot label a new container. Retry the call.")
        invoking_home = getattr(adapter, "invoking_home", None) or Path.home()
        return await lifecycle_service.create_container(
            adapter,
            image=image,
            name=container_name,
            run_id=run_id,
            settings=settings,
            invoking_home=invoking_home,
            profile=profile,
            profiles=profiles,
            workspace=workspace,
            network_mode=network_mode,
            ports=ports,
            env=env,
            hostname=hostname,
            comment=comment,
            start_timeout_s=_container_op_timeout(adapter),
        )


async def remove_container(
    container_name: str,
    keep_workspace: bool | None = None,
    keep_logs: bool | None = None,
    *,
    ctx: Context[ExegolPort, Any],
) -> RemovedContainer:
    """Stop and remove an Exegol container that exegol-mcp created, and report what is still on the host.

    IMPORTANT: Only containers created by exegol-mcp (managed=true in list_containers) can be removed, unless the
    operator allowed removing any Exegol container in the server config. Exegol deletes the workspace data the call
    does not keep, then stops and removes the container. The removal cannot be undone. It is bounded by the
    operator's execution.container_op_timeout_s: a removal that runs past it returns an error saying to call
    remove_container again to finish it. keep_workspace and keep_logs default to the operator's config, which keeps
    both unless the operator changed it. keep_logs covers only the Sentinel audit history: shell logs live under
    /workspace/logs and follow keep_workspace. A host-mounted workspace is never deleted. workspace_kept and
    logs_kept report what is still on the host afterwards: an empty workspace or a Sentinel directory without
    history is deleted even when keeping it was asked (nothing is lost), and is then reported as not kept. A
    private workspace still on the host is mounted again if a container with the same name is created later.

    RELATED TOOLS:
    - list_containers: find the container and check that it is managed
    - cleanup_containers: remove every container this server run (or any exegol-mcp run) created
    - create_container: create a new container
    - stop_container: stop a container without removing it

    Args:
        container_name: Name of the container to remove, as returned by list_containers.
        keep_workspace: Keep the private workspace on the host; omit it to use the operator default.
        keep_logs: Keep the Sentinel audit history; omit it to use the operator default.

    Returns:
        The removed container's name and whether its workspace and its Sentinel logs are still on the host.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await lifecycle_service.remove_container(
            adapter,
            container_name,
            keep_workspace=keep_workspace,
            keep_logs=keep_logs,
            settings=_lifecycle_settings(adapter),
            timeout_s=_container_op_timeout(adapter),
        )


async def cleanup_containers(
    scope: Literal["this_run", "all_managed"] = "this_run",
    keep_workspace: bool | None = None,
    keep_logs: bool | None = None,
    *,
    ctx: Context[ExegolPort, Any],
) -> CleanupResult:
    """Remove the Exegol containers exegol-mcp created: this server run's (the default) or every managed one.

    IMPORTANT: scope "this_run" (the default) removes only the containers this server process created. Scope
    "all_managed" also removes managed containers left by earlier or crashed runs, and by another exegol-mcp server
    that is still running. Containers not created by exegol-mcp are never touched, whatever the server config says.
    The server never cleans up on its own at shutdown, so a harness that wants a clean host calls this tool. Removal
    cannot be undone. keep_workspace and keep_logs behave as in remove_container: they default to the operator's
    config (which keeps both unless changed), keep_logs covers only the Sentinel audit history while shell logs
    follow keep_workspace, a host-mounted workspace is never deleted, and an empty workspace or a Sentinel directory
    without history is deleted even when keeping it was asked (nothing is lost). Each removal has the same bound as
    remove_container (the operator's execution.container_op_timeout_s). A container that fails to be removed, or
    whose removal runs past that bound, is listed under failed and the others are still removed. Nothing to remove
    gives empty lists, not an error.

    RELATED TOOLS:
    - list_containers: with managed_only=true, see which containers exegol-mcp created and which are this run's
    - remove_container: remove a single container
    - create_container: create a new container

    Args:
        scope: "this_run" for this server process's containers, "all_managed" for every exegol-mcp container.
        keep_workspace: Keep each private workspace on the host; omit it to use the operator default.
        keep_logs: Keep each Sentinel audit history; omit it to use the operator default.

    Returns:
        The scope, the removed container names and the failed removals, both sorted by name.
    """
    adapter = ctx.request_context.lifespan_context
    async with tool_errors(adapter):
        return await lifecycle_service.cleanup_containers(
            adapter,
            scope=scope,
            run_id=getattr(adapter, "run_id", None),
            keep_workspace=keep_workspace,
            keep_logs=keep_logs,
            settings=_lifecycle_settings(adapter),
            timeout_s=_container_op_timeout(adapter),
        )
