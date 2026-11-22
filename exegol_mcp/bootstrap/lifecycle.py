"""The server lifespan context (D-13): the one object tools and server_info both read.

build_server(config, adapter) yields its `adapter` argument as the lifespan context, which tools reach through
ctx.request_context.lifespan_context. server_info resolves its data source from that same context when it is a
ServerInfoSource (plan 03). So the context passed to build_server must be both: it forwards every ExegolPort call
to the real adapter, and it answers version_report() with the versions gathered at startup, a live Docker status
and the update notice once the background check has one. It is also the tools' ReadinessDiagnoser: when a tool call
fails, the tool boundary awaits diagnose(), which re-checks Docker and the EULA (D-08) so a host that changed after
startup reaches the agent in words.

The adapter is built once and shared (no per-request construction). The build_server lifespan sweeps tagged
in-container orphans at startup and shutdown and supervises background-job readers (02-05). The server removes no
container at shutdown or startup: cleanup is always an explicit remove_container / cleanup_containers call (D-14).
"""

from pathlib import Path
from typing import Literal

from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.port import (
    ContainerState,
    CreateSpec,
    DirListing,
    ExegolPort,
    FileRead,
    FileWrite,
    HostProbe,
    JobHandle,
    JobRead,
    RemovalOutcome,
)
from exegol_mcp.adapter.sudo import invoking_user_home
from exegol_mcp.bootstrap.readiness import ReadinessError, recheck
from exegol_mcp.bootstrap.versions import UpdateChecker, Versions, server_info_report
from exegol_mcp.context.request import ProgressCallback
from exegol_mcp.models.config import OperatorConfig
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.models.image import ImageInfo
from exegol_mcp.services.jobs import JobStore
from exegol_mcp.services.server_info import VersionReport


class ServerContext:
    """The lifespan context: an ExegolPort that delegates to the real adapter, a ServerInfoSource and the tools'
    ReadinessDiagnoser.

    Implements ServerInfoSource.version_report() so server_info reports real versions and Docker status instead of
    the package-metadata fallback's "not checked", and ReadinessDiagnoser.diagnose() so a failed tool call is
    explained by the Docker or EULA state that changed after startup (D-08).
    """

    def __init__(
        self,
        adapter: ExegolPort,
        versions: Versions,
        update_checker: UpdateChecker,
        probe: HostProbe,
        startup_docker_status: str,
        *,
        accept_eula: bool,
        config: OperatorConfig | None = None,
        invoking_home: Path | None = None,
    ) -> None:
        self._adapter = adapter
        self._versions = versions
        self._update_checker = update_checker
        self._probe = probe
        self._startup_docker_status = startup_docker_status
        self._accept_eula = accept_eula
        # The resolved operator config; tools read it from ctx.request_context.lifespan_context.config (CONF-03).
        self.config: OperatorConfig = config if config is not None else OperatorConfig()
        # The background-job store the lifespan builds and attaches at startup (02-05); the job tools read it
        # from ctx.request_context.lifespan_context.jobs. None until build_server's lifespan runs.
        self.jobs: JobStore | None = None
        # This server process's run id, stamped on every container it creates (D-09). The lifespan sets it (04-04).
        self.run_id: str | None = None
        # The invoking user's home (under sudo, not root's): create_container expands a workspace "~" against it
        # (D-08).
        self.invoking_home: Path = invoking_home if invoking_home is not None else invoking_user_home()

    # --- ServerInfoSource (D-13) ---
    async def version_report(self) -> VersionReport:
        """Report versions, a live Docker status and the current update notice.

        Re-pings Docker so a daemon that went down after startup shows up here (this is the tool an agent calls to
        diagnose a failure); falls back to the startup status if the ping cannot run.
        """
        try:
            docker_status = await run_blocking(self._probe.ping_docker)
        except Exception:  # noqa: BLE001  # server_info must never fail; report the daemon as down
            docker_status = f"unreachable (last seen: {self._startup_docker_status})"
        return server_info_report(self._versions, docker_status, self._update_checker.notice)

    # --- ReadinessDiagnoser (D-08) ---
    async def diagnose(self) -> str | None:
        """Re-check Docker and, unless --accept-eula was given, the EULA after a tool call failed.

        Never prompts and never writes the EULA datastore (recheck is read-only).

        Returns:
            The readiness message naming what is no longer ready, or None when the host is still ready.
        """
        try:
            await recheck(self._probe, self._accept_eula)
        except ReadinessError as e:
            return str(e)
        return None

    # --- ExegolPort (forwarded to the real adapter) ---
    async def list_containers(self) -> list[ContainerInfo]:
        return await self._adapter.list_containers()

    async def get_container_state(self, name: str) -> ContainerState | None:
        return await self._adapter.get_container_state(name)

    async def start_container(self, name: str) -> bool:
        return await self._adapter.start_container(name)

    async def stop_container(self, name: str) -> bool:
        return await self._adapter.stop_container(name)

    async def list_images(self, scope: Literal["installed", "all"]) -> list[ImageInfo]:
        return await self._adapter.list_images(scope)

    async def download_image(self, name: str, version: str, progress: ProgressCallback | None) -> bool:
        return await self._adapter.download_image(name, version, progress)

    async def exec_command(self, name: str, command: str) -> ExecResult:
        return await self._adapter.exec_command(name, command)

    async def create_container(self, spec: CreateSpec) -> ContainerInfo:
        return await self._adapter.create_container(spec)

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
        return await self._adapter.remove_container(
            name,
            keep_workspace,
            keep_logs,
            timeout_s=timeout_s,
            require_managed=require_managed,
            expected_run_id=expected_run_id,
        )

    async def start_job(self, name: str, command: str, tag: str, cap_bytes: int) -> JobHandle:
        return await self._adapter.start_job(name, command, tag, cap_bytes)

    async def pump_job(self, handle: JobHandle) -> None:
        await self._adapter.pump_job(handle)

    async def read_job(
        self, handle: JobHandle, stdout_offset: int, stderr_offset: int, max_bytes: int | None
    ) -> JobRead:
        return await self._adapter.read_job(handle, stdout_offset, stderr_offset, max_bytes)

    async def kill_job(self, name: str, tag: str, grace_s: float) -> None:
        await self._adapter.kill_job(name, tag, grace_s)

    async def sweep_orphans(self) -> int:
        return await self._adapter.sweep_orphans()

    async def read_file(self, name: str, path: str, max_bytes: int, timeout_s: float) -> FileRead:
        return await self._adapter.read_file(name, path, max_bytes, timeout_s)

    async def write_file(
        self, name: str, path: str, data: bytes, overwrite: bool, mode: int | None, timeout_s: float
    ) -> FileWrite:
        return await self._adapter.write_file(name, path, data, overwrite, mode, timeout_s)

    async def list_directory(
        self, name: str, path: str, depth: int, max_entries: int, timeout_s: float
    ) -> DirListing:
        return await self._adapter.list_directory(name, path, depth, max_entries, timeout_s)
