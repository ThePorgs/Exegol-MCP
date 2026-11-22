"""In-memory ExegolPort implementation for Docker-free tests."""

import posixpath
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import anyio

from exegol_mcp.adapter.offload import jobs_limiter, run_blocking
from exegol_mcp.adapter.port import (
    IMAGE_NOT_INSTALLED_MESSAGE,
    IMAGE_REQUIRED_MESSAGE,
    IMAGE_UP_TO_DATE_MESSAGE,
    MANAGED_LABEL,
    MANAGED_LABEL_VALUE,
    NAME_IN_USE_MESSAGE,
    NOT_THIS_RUN_MESSAGE,
    PROFILE_LICENCE_MESSAGE,
    RUN_ID_LABEL,
    UNMANAGED_REMOVAL_MESSAGE,
    ContainerState,
    CreateSpec,
    DirListing,
    DirRecord,
    FileRead,
    FileWrite,
    JobHandle,
    JobRead,
    RemovalOutcome,
)
from exegol_mcp.adapter.tail_buffer import TailBuffer, decode_output
from exegol_mcp.context.request import ProgressCallback
from exegol_mcp.models.config import OperatorConfig
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.models.image import ImageInfo

# A fixed, resolvable job id the lifespan seeds (opt-in via seed_contract_job) so the contract/era "call every
# tool" suites can call get_job/stop_job with a known id. Not a secret: a test fixture only.
SEEDED_JOB_ID = "contract-seed-job-0000"

# A managed container an earlier server run left behind, seeded opt-in (seed_managed_container) so the contract/era
# "call every tool" suites can call remove_container on a container the managed-only gate accepts (04-06).
SEEDED_MANAGED_CONTAINER = "mcp-seed"
SEEDED_MANAGED_RUN_ID = "earlier-run"


def _container(name: str, comment: str | None, managed: bool = False, run_id: str | None = None) -> ContainerInfo:
    """A canned container; managed/run_id seed one an earlier server run created (the fake records its labels)."""
    return ContainerInfo(
        name=name,
        creation_date="2026-10-01 12:00:00",
        image_name="nwodtuhs/exegol:free",
        image_version="6.0.0",
        status="Running",
        network_driver="host",
        network_name=None,
        features=["Exegol-Resources", "My-Resources", "shell-logging"],
        devices=[],
        vpn=None,
        env=["EXEGOL_ENV=1"],
        is_privileged=False,
        capabilities=["Docker default capabilities"],
        comment=comment,
        managed=managed,
        run_id=run_id if managed else None,
    )


def _managed_identity(labels: dict[str, str]) -> tuple[bool, str | None]:
    # The real adapter's exact-"true" rule (lifecycle_sdk.managed_identity), kept SDK-free here.
    managed = labels.get(MANAGED_LABEL) == MANAGED_LABEL_VALUE
    return managed, labels.get(RUN_ID_LABEL) if managed else None


def _image(name: str, installed: bool) -> ImageInfo:
    return ImageInfo(
        name=name,
        version="6.0.0" if installed else "",
        latest_version="6.0.0",
        is_installed=installed,
        is_built_locally=False,
        is_up_to_date=installed,
        build_date="2026-09-27" if installed else "",
        license="free" if name == "free" else "pro",
    )


# Two entries that differ only in trivial fields (name suffix and comment), so tests can check that
# list_containers keeps them distinct and returns them in a stable order.
DEFAULT_CONTAINERS: list[ContainerInfo] = [
    _container("exegol-lab", "first"),
    _container("exegol-lab2", "second"),
]

# One installed image and two that are only available for download.
DEFAULT_IMAGES: list[ImageInfo] = [
    _image("free", installed=True),
    _image("full", installed=False),
    _image("nightly", installed=False),
]


class FakeExegolAdapter:
    """ExegolPort implemented in memory over the whole port surface.

    Containers keep a running flag that start_container/stop_container flip; state_changes records each flip.
    Images are a canned list: list_images filters it by scope and download_image skips the pull for an
    installed, up-to-date image. downloads records every call; pulls records only the calls that installed or
    updated an image.

    exec_command blocks for up to exec_block_seconds in a synchronous wait, standing in for a synchronous
    docker exec. The test can end the wait early with exec_release. exec_started is set just before the
    blocking work begins and exec_started_at/exec_finished_at record when it ran.

    offload_exec=True (the default) sends that blocking work through run_blocking, the same way the real
    adapter offloads its docker calls. offload_exec=False runs it directly on the event loop: this is the v1
    bug, kept only so the responsiveness test can show that its measurement catches it.
    """

    def __init__(
        self,
        containers: list[ContainerInfo] | None = None,
        images: list[ImageInfo] | None = None,
        exec_block_seconds: float = 0.0,
        exec_result: ExecResult | None = None,
        offload_exec: bool = True,
        exec_router: Callable[[str], ExecResult | None] | None = None,
        config: OperatorConfig | None = None,
        seed_contract_job: bool = False,
        invoking_home: Path | None = None,
        seed_managed_container: bool = False,
        profiles: dict[str, str | None] | None = None,
        profile_licence: bool = True,
        files: dict[str, dict[str, bytes]] | None = None,
        dirs: set[tuple[str, str]] | None = None,
    ) -> None:
        # The resolved operator config later-plan tools read from the lifespan context; defaults keep every
        # build_server(None, FakeExegolAdapter()) test working with built-in defaults.
        self.config: OperatorConfig = config if config is not None else OperatorConfig()
        # The Exegol profiles this host's files define, each mapped to its image.tag (None: the file names no image),
        # and whether the cached licence session grants profiles (04-08). The allowlist is the service's job.
        self.profiles: dict[str, str | None] = dict(profiles or {})
        self.profile_licence = profile_licence
        self.containers: list[ContainerInfo] = (
            [c.model_copy() for c in DEFAULT_CONTAINERS] if containers is None else list(containers)
        )
        # Opt-in, like seed_contract_job, so the adapters other tests build keep exactly their containers.
        if seed_managed_container:
            self.containers.append(
                _container(SEEDED_MANAGED_CONTAINER, "seeded", managed=True, run_id=SEEDED_MANAGED_RUN_ID)
            )
        self.running: dict[str, bool] = {c.name: c.status.lower() == "running" for c in self.containers}
        self.images: list[ImageInfo] = [i.model_copy() for i in DEFAULT_IMAGES] if images is None else list(images)
        self.downloads: list[tuple[str, str]] = []
        # The downloads that installed or updated an image (an up-to-date image is skipped, so it is not recorded).
        self.pulls: list[tuple[str, str]] = []
        # Every start/stop the adapter performed, in order, so tests can prove a service call was a no-op.
        self.state_changes: list[tuple[str, str]] = []
        self.exec_block_seconds = exec_block_seconds
        self.exec_result = exec_result or ExecResult(stdout="ok\n", stderr="", exit_code=0)
        self.offload_exec = offload_exec
        # Optional canned responses keyed on the command string (for CSV/help/find catalog and remote tests).
        # When it returns None for a command, exec_command falls back to the blocking default below.
        self.exec_router = exec_router
        # Every command exec_command was asked to run, in order, so tests can assert what was built (e.g. the
        # shlex-quoted tool_name or the netexec invocation).
        self.exec_commands: list[str] = []
        self.exec_release = threading.Event()
        self.exec_started = anyio.Event()
        self.exec_started_at: float | None = None
        self.exec_finished_at: float | None = None
        # Simulated in-container process table for the job primitive: tag -> container name of the live process.
        # start_job records a tag, pump_job drops it when the command finishes, kill_job/sweep_orphans remove it.
        self.job_processes: dict[str, str] = {}
        self._job_commands: dict[str, str] = {}
        self._job_kill_events: dict[str, threading.Event] = {}
        self._killed_tags: set[str] = set()
        # The JobStore the build_server lifespan attaches at startup (02-05); None until then.
        self.jobs: object | None = None
        # How many times the lifespan asked for an orphan sweep (startup + shutdown), so a test can prove both.
        self.sweep_calls = 0
        # Jobs the lifespan should seed under a known id so the contract/era "call every tool" suites can
        # resolve get_job/stop_job. Opt-in (seed_contract_job) so adapters other tests build stay unpolluted.
        self.seed_jobs: list[tuple[str, str, str]] = (
            [(SEEDED_JOB_ID, "exegol-lab", "sleep 3600")] if seed_contract_job else []
        )
        # Docker labels per container name (create_container records the spec's labels here) and every spec a
        # create accepted, in order (04-03).
        self.labels: dict[str, dict[str, str]] = {
            c.name: {MANAGED_LABEL: MANAGED_LABEL_VALUE, **({RUN_ID_LABEL: c.run_id} if c.run_id else {})}
            for c in self.containers
            if c.managed
        }
        self.created_specs: list[CreateSpec] = []
        # Every removal the adapter performed, in order, as (name, keep_workspace, keep_logs) (04-05).
        self.removals: list[tuple[str, bool, bool]] = []
        # The timeout_s each removal was bounded by, in call order (CR-01: execution.container_op_timeout_s).
        self.removal_timeouts: list[float] = []
        # The run id the lifespan stamps on created containers (set in 04-04) and the invoking user's home a
        # workspace "~" expands against (D-08); tests pass a tmp path.
        self.run_id: str | None = None
        self.invoking_home: Path = invoking_home if invoking_home is not None else Path.home()
        # Per-container file contents keyed on normalized absolute paths (04-09). A path that is a prefix directory of
        # another key reads as a directory. The modes write_file applied are recorded per (container, path).
        self.files: dict[str, dict[str, bytes]] = (
            {"exegol-lab": {"/etc/hostname": b"exegol-lab\n"}}
            if files is None
            else {name: dict(entries) for name, entries in files.items()}
        )
        self.file_modes: dict[tuple[str, str], int] = {}
        # Explicit directories as (container, path), for empty ones; every file's parents are implied directories.
        self.dirs: set[tuple[str, str]] = set(dirs or ())

    async def list_containers(self) -> list[ContainerInfo]:
        return list(self.containers)

    async def get_container_state(self, name: str) -> ContainerState | None:
        if name not in self.running:
            return None
        managed, run_id = _managed_identity(self.labels.get(name, {}))
        return ContainerState(running=self.running[name], managed=managed, run_id=run_id)

    def _set_running(self, name: str, running: bool) -> bool:
        if name not in self.running:
            raise ValueError(f"Container '{name}' not found")
        self.state_changes.append(("start" if running else "stop", name))
        self.running[name] = running
        status = "Running" if running else "Stopped"
        self.containers = [c.model_copy(update={"status": status}) if c.name == name else c for c in self.containers]
        return running

    async def start_container(self, name: str) -> bool:
        return self._set_running(name, True)

    async def stop_container(self, name: str) -> bool:
        return self._set_running(name, False)

    async def list_images(self, scope: Literal["installed", "all"]) -> list[ImageInfo]:
        if scope == "installed":
            return [i for i in self.images if i.is_installed]
        return list(self.images)

    async def download_image(self, name: str, version: str, progress: ProgressCallback | None) -> bool:
        index = next((n for n, image in enumerate(self.images) if image.name == name), None)
        if index is None:
            raise ValueError(f"Image '{name}' not found")
        self.downloads.append((name, version))
        image = self.images[index]
        tag = name if version == "latest" else f"{name}-{version}"
        # Follow the port contract (Tasks 1-2): an installed, up-to-date image is not pulled.
        if image.is_installed and image.is_up_to_date:
            if progress is not None:
                await progress(100, 100, IMAGE_UP_TO_DATE_MESSAGE.format(image=tag))
            return True
        if progress is not None:
            await progress(0, 100, "Downloading layers: 0 / 2")
            await progress(50, 100, "Extracting layers: 1 / 2")
            await progress(100, 100, "Download complete")
        self.images[index] = image.model_copy(
            update={
                "is_installed": True,
                "is_up_to_date": True,
                "version": image.latest_version if version == "latest" else version,
            }
        )
        self.pulls.append((name, version))
        return True

    async def create_container(self, spec: CreateSpec) -> ContainerInfo:
        # The same errors, in the same order, as the real adapter (lifecycle_sdk.create_managed_container).
        if not spec.image and spec.profile is None:
            raise ValueError(IMAGE_REQUIRED_MESSAGE)
        if spec.name in self.running:
            raise ValueError(NAME_IN_USE_MESSAGE.format(name=spec.name))
        image_tag = spec.image
        if spec.profile is not None:
            # load_allowed_profile's errors: the licence gate first, then Exegol's "cannot be used".
            if not self.profile_licence:
                raise RuntimeError(PROFILE_LICENCE_MESSAGE)
            if spec.profile not in self.profiles:
                raise ValueError(f"Container profile '{spec.profile}' cannot be used: not found")
            image_tag = spec.image or self.profiles[spec.profile]
        if not image_tag:
            raise ValueError(IMAGE_REQUIRED_MESSAGE)
        if not any(image.name == image_tag and image.is_installed for image in self.images):
            raise ValueError(IMAGE_NOT_INSTALLED_MESSAGE.format(image=image_tag))
        managed, run_id = _managed_identity(spec.labels)
        info = _container(spec.name, spec.comment, managed=managed, run_id=run_id).model_copy(
            update={"image_name": image_tag, "env": list(spec.env)}
        )
        self.containers.append(info)
        self.running[spec.name] = True
        self.labels[spec.name] = dict(spec.labels)
        self.created_specs.append(spec)
        return info

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
        # The real adapter's errors and outcome; the fake has no host-mounted workspace to report.
        if name not in self.running:
            raise ValueError(f"Container '{name}' not found")
        # The real adapter's gate over the container's own labels, exact equality only, before anything changes.
        managed, run_id = _managed_identity(self.labels.get(name, {}))
        if require_managed and not managed:
            raise ValueError(UNMANAGED_REMOVAL_MESSAGE.format(name=name))
        if expected_run_id is not None and run_id != expected_run_id:
            raise ValueError(NOT_THIS_RUN_MESSAGE.format(name=name))
        self.removal_timeouts.append(timeout_s)
        self.containers = [c for c in self.containers if c.name != name]
        del self.running[name]
        self.labels.pop(name, None)
        self.removals.append((name, keep_workspace, keep_logs))
        return RemovalOutcome(workspace_kept=keep_workspace, logs_kept=keep_logs, host_workspace=False)

    # --- File transfer, in memory over the per-container files dict (04-09) -------------------------------------

    def _require_running(self, name: str) -> dict[str, bytes]:
        if name not in self.running:
            raise ValueError(f"Container '{name}' not found")
        if not self.running[name]:
            raise RuntimeError(f"Container '{name}' is not running")
        return self.files.setdefault(name, {})

    @staticmethod
    def _normalize(path: str) -> str:
        return posixpath.normpath(posixpath.join("/", path)).replace("//", "/")

    def _directories(self, name: str) -> set[str]:
        """Every directory of the container: the explicit ones and the parents of every file and directory."""
        explicit = {path for owner, path in self.dirs if owner == name}
        found: set[str] = {"/"} | explicit
        for leaf in [*self.files.get(name, {}), *explicit]:
            parent = posixpath.dirname(leaf)
            while parent not in found:
                found.add(parent)
                parent = posixpath.dirname(parent)
        return found

    def _is_dir(self, name: str, path: str) -> bool:
        return path in self._directories(name)

    async def read_file(self, name: str, path: str, max_bytes: int, timeout_s: float) -> FileRead:
        entries = self._require_running(name)
        resolved = self._normalize(path)
        if resolved in entries:
            data = entries[resolved]
            return FileRead(
                data=data[: max_bytes + 1], size=len(data), over_limit=len(data) > max_bytes, path=resolved
            )
        if self._is_dir(name, resolved):
            raise ValueError(f"'{path}' is a directory: use list_directory to see its entries")
        raise ValueError(f"No such file in the container: '{path}'")

    async def write_file(
        self, name: str, path: str, data: bytes, overwrite: bool, mode: int | None, timeout_s: float
    ) -> FileWrite:
        if mode is not None and not 0 <= mode <= 0o7777:
            raise ValueError(f"Invalid mode {mode!r}: give permission bits between 0 and 0o7777 (for example 0o755)")
        entries = self._require_running(name)
        resolved = self._normalize(path)
        if self._is_dir(name, resolved):
            raise ValueError(f"'{path}' is a directory: write_file writes files only")
        existed = resolved in entries
        if existed and not overwrite:
            raise ValueError(f"'{path}' already exists and overwrite is false: pass overwrite=true to replace it")
        parent = posixpath.dirname(resolved)
        while parent != "/":
            if parent in entries:
                raise ValueError(f"Cannot create the parent directory of '{path}' in the container: Not a directory")
            parent = posixpath.dirname(parent)
        entries[resolved] = bytes(data)
        if mode is not None:
            self.file_modes[(name, resolved)] = mode
        return FileWrite(path=resolved, size=len(data), created=not existed)

    async def list_directory(
        self, name: str, path: str, depth: int, max_entries: int, timeout_s: float
    ) -> DirListing:
        if depth < 1 or max_entries < 1:
            raise ValueError("depth and max_entries must both be at least 1")
        entries = self._require_running(name)
        resolved = self._normalize(path)
        if resolved in entries:
            raise ValueError(f"'{path}' is not a directory: use read_file to read a file")
        directories = self._directories(name)
        if resolved not in directories:
            raise ValueError(f"No such file or directory in the container: '{path}'")
        prefix = "/" if resolved == "/" else resolved + "/"
        records: list[DirRecord] = []
        for full in sorted({*entries, *directories} - {resolved}):
            relative = full[len(prefix) :] if full.startswith(prefix) else None
            if not relative or relative.count("/") + 1 > depth:
                continue
            if full in entries:
                mode = self.file_modes.get((name, full), 0o644)
                records.append(DirRecord(relative, "file", len(entries[full]), mode, 1700000000.0, None))
            else:
                records.append(DirRecord(relative, "dir", 4096, 0o755, 1700000000.0, None))
        return DirListing(path=resolved, entries=records[:max_entries], truncated=len(records) > max_entries)

    def _blocking_exec(self) -> ExecResult:
        self.exec_started_at = time.perf_counter()
        self.exec_release.wait(timeout=self.exec_block_seconds)
        self.exec_finished_at = time.perf_counter()
        return self.exec_result.model_copy()

    async def exec_command(self, name: str, command: str) -> ExecResult:
        self.exec_started.set()
        self.exec_commands.append(command)
        if self.exec_router is not None:
            routed = self.exec_router(command)
            if routed is not None:
                return routed.model_copy()
        if self.offload_exec:
            return await run_blocking(self._blocking_exec)
        return self._blocking_exec()

    # --- Job primitive, in memory over a simulated process table (Docker-free) ---------------------------------

    def _canned_result(self, command: str) -> ExecResult:
        if self.exec_router is not None:
            routed = self.exec_router(command)
            if routed is not None:
                return routed
        return self.exec_result

    def _wait_job(self, tag: str) -> bool:
        """Block like a slow command for up to exec_block_seconds; return True if the job was killed first."""
        event = self._job_kill_events.get(tag)
        deadline = time.perf_counter() + self.exec_block_seconds
        while time.perf_counter() < deadline:
            if event is not None and event.is_set():
                return True
            if self.exec_release.is_set():
                return False
            time.sleep(0.01)
        return event is not None and event.is_set()

    async def start_job(self, name: str, command: str, tag: str, cap_bytes: int) -> JobHandle:
        # Mirror exec_command's guards so the service's require_running and the adapter errors line up.
        if name not in self.running:
            raise ValueError(f"Container '{name}' not found")
        if not self.running[name]:
            raise RuntimeError(f"Container '{name}' is not running")
        self.exec_started.set()
        # Record the command so the existing exec/responsiveness assertions (exec_commands) keep holding.
        self.exec_commands.append(command)
        handle = JobHandle(name=name, tag=tag, stdout=TailBuffer(cap_bytes), stderr=TailBuffer(cap_bytes))
        self.job_processes[tag] = name
        self._job_commands[tag] = command
        self._job_kill_events[tag] = threading.Event()
        return handle

    async def pump_job(self, handle: JobHandle) -> None:
        result = self._canned_result(self._job_commands.get(handle.tag, ""))
        handle.stdout.append(result.stdout.encode("utf-8"))
        handle.stderr.append(result.stderr.encode("utf-8"))
        # Block like a slow command. offload_exec=True (the default) drains off the loop on the jobs limiter,
        # as the real pump does; offload_exec=False runs it on the loop so the responsiveness inversion test
        # can still show a loop-blocking exec stalls concurrent requests.
        if self.offload_exec:
            killed = await run_blocking(self._wait_job, handle.tag, limiter=jobs_limiter)
        else:
            killed = self._wait_job(handle.tag)
        if killed or handle.tag in self._killed_tags:
            handle.exit_code = None  # a killed pump finishes with no exit code (EXEC-06 mechanism)
        else:
            handle.exit_code = result.exit_code
        handle.running = False
        self.job_processes.pop(handle.tag, None)
        handle.done.set()

    async def read_job(
        self, handle: JobHandle, stdout_offset: int, stderr_offset: int, max_bytes: int | None
    ) -> JobRead:
        out_data, out_next, out_gap = handle.stdout.read(stdout_offset, limit=max_bytes)
        err_data, err_next, err_gap = handle.stderr.read(stderr_offset, limit=max_bytes)
        return JobRead(
            stdout=decode_output(out_data),
            stderr=decode_output(err_data),
            stdout_next_offset=out_next,
            stderr_next_offset=err_next,
            stdout_gap=out_gap,
            stderr_gap=err_gap,
            stdout_bytes_dropped=handle.stdout.dropped,
            stderr_bytes_dropped=handle.stderr.dropped,
            exit_code=handle.exit_code,
            running=handle.running,
        )

    async def kill_job(self, name: str, tag: str, grace_s: float) -> None:
        self._killed_tags.add(tag)
        event = self._job_kill_events.get(tag)
        if event is not None:
            event.set()
        self.job_processes.pop(tag, None)

    async def sweep_orphans(self) -> int:
        self.sweep_calls += 1
        count = len(self.job_processes)
        for tag in list(self.job_processes):
            self._killed_tags.add(tag)
            event = self._job_kill_events.get(tag)
            if event is not None:
                event.set()
        self.job_processes.clear()
        return count


# --- Canned calls for the full v2 tool surface (protocol-era, contract and transport tests) -------------------------

# The final v2 tool surface (PROTO-03/04, phase success criterion 5): exactly these names, nothing else.
V2_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_containers",
        "start_container",
        "stop_container",
        "list_images",
        "download_image",
        "exec_command",
        "exec_remote_command",
        "list_installed_tools",
        "get_tool_help",
        "list_offline_resources",
        "server_info",
        "get_job",
        "stop_job",
        "list_jobs",
        "create_container",
        "remove_container",
        "cleanup_containers",
        "read_file",
        "write_file",
        "list_directory",
    }
)

# Minimal arguments that are valid against a default FakeExegolAdapter built with catalog_router.
# get_job/stop_job use SEEDED_JOB_ID, which the lifespan resolves when the adapter is built with
# seed_contract_job=True; remove_container removes the managed container seed_managed_container=True adds (see the
# protocol-era "call every tool" suite).
MINIMAL_TOOL_ARGUMENTS: dict[str, dict[str, object]] = {
    "list_containers": {},
    "start_container": {"container_name": "exegol-lab"},
    "stop_container": {"container_name": "exegol-lab2"},
    "list_images": {},
    "download_image": {"image_name": "full"},
    "exec_command": {"container_name": "exegol-lab", "command": "id"},
    "exec_remote_command": {
        "container_name": "exegol-lab",
        "protocol": "ssh",
        "target_ip": "10.0.0.1",
        "username": "user",
        "password": "pass",
        "command": "id",
    },
    "list_installed_tools": {"container_name": "exegol-lab"},
    "get_tool_help": {"container_name": "exegol-lab", "tool_name": "nmap"},
    "list_offline_resources": {"container_name": "exegol-lab"},
    "server_info": {},
    "get_job": {"job_id": SEEDED_JOB_ID},
    "stop_job": {"job_id": SEEDED_JOB_ID},
    "list_jobs": {},
    "create_container": {"image": "free"},
    "remove_container": {"container_name": SEEDED_MANAGED_CONTAINER},
    "cleanup_containers": {},
    "read_file": {"container_name": "exegol-lab", "path": "/etc/hostname"},
    "write_file": {"container_name": "exegol-lab", "path": "/tmp/mcp-contract.txt", "content": "hello\n"},
    "list_directory": {"container_name": "exegol-lab", "path": "/etc"},
}

INSTALLED_TOOLS_CSV = "Tool,Version,Link,Description\nnmap,7.94,https://nmap.org,Network scanner\n"


def catalog_router(command: str) -> ExecResult | None:
    """exec_router answering the catalog commands; anything else gets the fake's default exec result."""
    if "installed_tools.csv" in command:
        return ExecResult(stdout=INSTALLED_TOOLS_CSV, stderr="", exit_code=0)
    if command.startswith("find /opt/resources/"):
        return ExecResult(stdout="/opt/resources/linux/a\n/opt/resources/linux/b\n", stderr="", exit_code=0)
    if command.startswith("nmap "):
        return ExecResult(stdout="Nmap usage\n", stderr="", exit_code=0)
    return None
