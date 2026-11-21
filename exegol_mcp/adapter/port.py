"""ExegolPort: the SDK-free boundary between services and the Exegol/Docker adapter."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import anyio

from exegol_mcp.adapter.tail_buffer import TailBuffer
from exegol_mcp.context.request import ProgressCallback
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.models.image import ImageInfo

# The environment variable carrying each job's unique tag on its in-container process group. kill_job and
# sweep_orphans match the full EXEGOL_MCP_JOB=<tag> string, never a prefix, so a kill never hits an unrelated
# process (T-02-02). The tag is minted with secrets.token_urlsafe and doubles as the job id in 02-05.
JOB_TAG_ENV = "EXEGOL_MCP_JOB"

# Raised when a background start is refused because max_concurrent_jobs is already reached (consumed by 02-05).
JOB_CAP_REACHED_MESSAGE = (
    "Cannot start a background job: the server is already running its maximum number of concurrent jobs "
    "({cap}). Wait for a running job to finish (check with list_jobs), or stop one with stop_job, then retry."
)

# The last progress report when download_image finds nothing to pull, formatted with the image tag.
IMAGE_UP_TO_DATE_MESSAGE = "Image '{image}' is already installed and up to date: nothing to download"

# The error raised when Exegol refuses or fails a pull, formatted with the image tag.
IMAGE_NOT_DOWNLOADED_MESSAGE = (
    "Image '{image}' was not downloaded: Exegol refused or failed the pull (a registry error, an image this "
    "host's Exegol license does not cover, or an image status Exegol cannot read right now). The reason is in "
    "the exegol-mcp server log on stderr. Check the image with list_images (scope \"all\") and try again later."
)

# Docker labels stamped on every container exegol-mcp creates (D-09), so it recognizes its own containers later,
# even after a crash or restart (LIFE-06). ONE-WAY: they sit on live containers and every later version reads them,
# so renaming them strands existing containers unless the old keys stay as aliases. They live in Exegol's own
# org.exegol.* family (research A3); Exegol's own label parser ignores them. Only the exact value "true" counts.
MANAGED_LABEL = "org.exegol.mcp.managed"
MANAGED_LABEL_VALUE = "true"
RUN_ID_LABEL = "org.exegol.mcp.run-id"

# create_container without an image (D-07): the server never picks one itself.
IMAGE_REQUIRED_MESSAGE = (
    "No image was given for the new container. Call list_images to choose an installed image (download_image "
    "installs a new one); the server never picks an image itself."
)

# create_container with an image that is not installed, formatted with the image tag (D-07).
IMAGE_NOT_INSTALLED_MESSAGE = (
    "Image '{image}' is not installed on this host. Call list_images to see the installed images, or "
    "download_image to install it first; create_container never downloads an image."
)

# create_container with a profile on a host whose cached Exegol licence session does not grant profiles (Pitfall 7).
# Lives here, SDK-free, so the fake adapter raises the same text adapter/profiles.py does.
PROFILE_LICENCE_MESSAGE = (
    "Exegol container profiles need a Pro, Team or Enterprise licence on this host, and the cached Exegol licence "
    "session does not grant one. If this host has such a licence, run the exegol wrapper once (for example "
    "`exegol info`) to refresh its cached licence session, then retry."
)

# create_container with a name an existing container already uses, formatted with the name (D-06).
NAME_IN_USE_MESSAGE = (
    "A container named '{name}' already exists. Choose another container_name, or omit it to get a generated one."
)


# D-11: a removal refused because the container lacks the exact managed label, formatted with the name. The refusal
# names the operator toggle, never a per-call flag (there is none). Lives here, SDK-free, because the adapter applies
# the gate to the container it is about to remove (WR-02) and the service pre-checks with the same text.
UNMANAGED_REMOVAL_MESSAGE = (
    "Container '{name}' was not created by exegol-mcp, so it is not removed. The operator can allow removing any "
    "Exegol container with lifecycle.allow_remove_unmanaged in config.yml."
)

# D-13 / WR-02: a this_run cleanup target whose own run id is no longer this run's (the name changed hands between the
# listing and the removal), formatted with the name.
NOT_THIS_RUN_MESSAGE = (
    "Container '{name}' is no longer one this server run created (another container now holds the name), so "
    "cleanup_containers(scope=\"this_run\") does not remove it. Call list_containers to check it."
)

# A removal that ran past execution.container_op_timeout_s (CR-01), formatted with the name and the limit. Exegol may
# have deleted the workspace already while the container is still listed: a second remove_container finishes it.
REMOVAL_TIMEOUT_MESSAGE = (
    "Container '{name}' was not fully removed within {timeout:g} s (execution.container_op_timeout_s). Exegol may "
    "still be deleting its data: call list_containers to check it, then call remove_container again to finish the "
    "removal."
)

@dataclass(frozen=True)
class ContainerState:
    """Live state of one container, with the managed labels read from Docker (D-09, D-10)"""

    running: bool
    # True only when MANAGED_LABEL is exactly MANAGED_LABEL_VALUE; run_id is RUN_ID_LABEL then, else None.
    managed: bool = False
    run_id: str | None = None


@dataclass(frozen=True)
class CreateSpec:
    """Everything the adapter needs to create one container (LIFE-01), already validated by the service.

    Only the core safe set of D-04 has a field: privileged mode, capabilities, devices, extra mounts, VPN and
    desktop cannot be expressed here. Without a profile the adapter forces GUI and desktop off (D-25); with one they
    are left to the profile and the operator's Exegol config, and the safe-set fields override it (D-05).
    """

    # Validated container name, without Exegol's "exegol-" prefix (Docker names it "exegol-<name>").
    name: str
    # Installed image tag; None means none was given: the profile's image.tag then, else IMAGE_REQUIRED_MESSAGE (D-07).
    image: str | None
    # Docker labels stamped at creation (MANAGED_LABEL / RUN_ID_LABEL, D-09).
    labels: dict[str, str] = field(default_factory=dict)
    # Absolute host path to mount as the workspace, or None for a private workspace.
    workspace: str | None = None
    network_mode: str | None = None
    ports: tuple[str, ...] = ()
    # KEY=VALUE entries only (an entry without "=" would copy a host value, research Pitfall 5).
    env: tuple[str, ...] = ()
    hostname: str | None = None
    comment: str | None = None
    # Bound on the post-create start (the container-op timeout, Phase 2 D-18).
    start_timeout_s: float = 50.0
    # An operator-allowlisted Exegol profile selection (the service checked profiles.allowed), or None (D-01).
    profile: str | None = None


@dataclass(frozen=True)
class RemovalOutcome:
    """What is still on the host after a removal, checked after it, never echoed from the request (LIFE-03, WR-06)"""

    # A path the host check cannot read, or a check that does not answer within its bound, counts as still on the host.
    #
    # True while the workspace still exists: a private one kept or not deletable, a WSL one, or a host-mounted one.
    # False once it is gone (an empty private workspace Exegol deletes unasked included), and false for a container
    # with no workspace.
    workspace_kept: bool
    # True when a Sentinel audit history directory still exists (shell logs live in the workspace and follow
    # workspace_kept). False without Sentinel logging, or once Exegol deleted it (a history-less one even when kept).
    logs_kept: bool
    # True for a host-mounted workspace (config.isWorkspaceCustom()), a host directory a removal never deletes.
    host_workspace: bool


@dataclass
class JobHandle:
    """A launched tagged job: its two bounded output buffers and live state (D-09, in memory only).

    Mutable: pump_job appends demuxed chunks into the buffers and, when the stream ends, records the real exit
    code (None while running), clears running and sets the done event. The adapter keeps the exec id and the
    stream generator in the private `stream` slot; the rest of the server never touches it.
    """

    name: str
    tag: str
    stdout: TailBuffer
    stderr: TailBuffer
    exit_code: int | None = None
    running: bool = True
    done: anyio.Event = field(default_factory=anyio.Event)
    # Adapter-private: the real adapter stores (exec_id, generator, api_client) here; the fake leaves it None.
    stream: Any = None


@dataclass(frozen=True)
class JobRead:
    """One incremental, offset-based read of a job's two streams (D-08/D-22).

    stdout/stderr are the new decoded text since the given offsets; the next offsets page forward; the gaps
    report bytes skipped because an offset was behind the retained window; the dropped counts are the total
    bytes dropped from the front of each stream for size (EXEC-03, not redaction).
    """

    stdout: str
    stderr: str
    stdout_next_offset: int
    stderr_next_offset: int
    stdout_gap: int
    stderr_gap: int
    stdout_bytes_dropped: int
    stderr_bytes_dropped: int
    exit_code: int | None
    running: bool


@dataclass(frozen=True)
class FileRead:
    """One bounded read of a container file (FILE-01). data is the raw bytes, never decoded."""

    # At most max_bytes + 1 bytes as the file held them, so a binary file round-trips byte-identically.
    data: bytes
    # The file size in bytes, or None when it is unknown: a pseudo file (size 0 to stat) larger than the limit.
    size: int | None
    # True when the file holds more than max_bytes bytes; data then stops after max_bytes + 1 bytes.
    over_limit: bool
    # The resolved in-container path (symlinks followed, "." and ".." removed).
    path: str


@dataclass(frozen=True)
class FileWrite:
    """The outcome of one verified container file write (FILE-02)."""

    # The resolved in-container path the bytes landed at (a symlink's target, not the link).
    path: str
    # The number of bytes written, checked against the file afterwards.
    size: int
    # True when the target did not exist before the write.
    created: bool


@dataclass(frozen=True)
class DirRecord:
    """One entry of a container directory listing (FILE-03, D-19)."""

    # The path relative to the listed directory ("a/b" at depth 2). Not valid UTF-8 -> U+FFFD replacements.
    path: str
    kind: Literal["file", "dir", "symlink", "other"]
    # Size in bytes as stat reports it (a symlink's own size, never its target's).
    size: int
    # Permission bits (for example 0o644).
    mode: int
    # Modification time, in seconds since the epoch.
    mtime: float
    # A symlink's target as stored in the link; None for anything else. The link is reported, never followed.
    target: str | None


@dataclass(frozen=True)
class DirListing:
    """A bounded listing of one container directory, in find order (the service sorts it)."""

    # The resolved in-container path of the listed directory.
    path: str
    entries: list[DirRecord]
    # True when the directory held more than max_entries entries within the depth: entries is the first ones.
    truncated: bool


@runtime_checkable
class ExegolPort(Protocol):
    """Async operations the services need from Exegol.

    The real adapter (the only module allowed to import exegol.* or docker) and the in-memory test fake
    both implement this protocol. Services and tools depend on it, never on a concrete adapter.
    Implementations do every synchronous Docker/SDK call through run_blocking() so the event loop stays free.
    """

    async def list_containers(self) -> list[ContainerInfo]:
        """Return every Exegol container on the host, in a stable order."""
        ...

    async def get_container_state(self, name: str) -> ContainerState | None:
        """Return the live state of the named container, or None if no container has that name."""
        ...

    async def start_container(self, name: str) -> bool:
        """Start the named container and return its running state afterwards (True once running)."""
        ...

    async def stop_container(self, name: str) -> bool:
        """Stop the named container and return its running state afterwards (False once stopped)."""
        ...

    async def list_images(self, scope: Literal["installed", "all"]) -> list[ImageInfo]:
        """Return the installed images, or every known image (installed and downloadable) for scope "all"."""
        ...

    async def download_image(self, name: str, version: str, progress: ProgressCallback | None) -> bool:
        """Download or update an image, reporting progress through the callable. Return True once the image is
        installed and up to date. An image that already is installed and up to date is not pulled: report
        IMAGE_UP_TO_DATE_MESSAGE as the last progress and return True.
        A pull Exegol refuses or fails raises RuntimeError (IMAGE_NOT_DOWNLOADED_MESSAGE); the result is never
        False."""
        ...

    async def exec_command(self, name: str, command: str) -> ExecResult:
        """Run a shell command in the named container and return its separate stdout, stderr and exit code."""
        ...

    async def create_container(self, spec: CreateSpec) -> ContainerInfo:
        """Create a container through Exegol's own model, with the spec values set on the CLI tier as if typed on
        `exegol start`, stamp every spec label, start it within spec.start_timeout_s and return its info.
        No prompt is ever asked. A name already in use raises ValueError(NAME_IN_USE_MESSAGE), a missing image
        ValueError(IMAGE_REQUIRED_MESSAGE), an uninstalled image ValueError(IMAGE_NOT_INSTALLED_MESSAGE), and a
        start that overruns its bound RuntimeError (the container then exists and may still be starting). It never
        pulls an image and never removes or alters another container. With spec.profile, the profile is loaded
        through Exegol and applied for this create only: a host without profile licensing raises
        RuntimeError(PROFILE_LICENCE_MESSAGE), a profile Exegol cannot use raises ValueError naming it and Exegol's
        reason, the image may come from the profile, and GUI and desktop are left to the profile. Without a
        profile, GUI and desktop are forced off."""
        ...

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
        """Remove the named container through Exegol's own remove(), with its backup containers and the networks it
        created, and return what is still on the host afterwards (checked, not the request: WR-06). keep_workspace
        keeps a private workspace on the host; keep_logs keeps the Sentinel audit history. A host-mounted workspace
        is never deleted: it is reported as host_workspace, and as kept while it exists. A container with no
        workspace reports none kept (IN-10). No prompt is ever asked.

        Exegol's removal is bounded by timeout_s: when the deadline is reached before it finishes,
        RuntimeError(REMOVAL_TIMEOUT_MESSAGE) is raised and a second call finishes the removal. The deadline is read
        from its own scope, so a removal that completed is never reported as timed out (WR-08). The host check after
        it is bounded by a few seconds, and a path it cannot read or a check that does not answer counts as still on
        the host. It never runs a restart wait or an in-container delete on the event loop (CR-01). A cancelled
        caller still waits for the removal to finish or reach its deadline, plus that bound, never longer. A host
        OSError (an operating-system timeout included) is a RuntimeError naming the container, never the deadline
        message. An unknown name raises ValueError.

        The adapter applies the gate to the resolved container's own labels right before removing it, so a name
        that changed hands after the caller's own lookup is still refused (WR-02, D-11, D-13). require_managed
        (the default, failing closed) refuses a container whose managed label is not exactly "true"
        (ValueError(UNMANAGED_REMOVAL_MESSAGE)). expected_run_id refuses a container whose run id is not exactly
        equal to it (ValueError(NOT_THIS_RUN_MESSAGE)). Nothing is removed then."""
        ...

    async def start_job(self, name: str, command: str, tag: str, cap_bytes: int) -> JobHandle:
        """Launch command in the named container under its own process group (setsid -w) carrying the unique env
        tag EXEGOL_MCP_JOB=<tag>, streamed through the low-level demuxed exec into two bounded TailBuffers of
        cap_bytes each. The wait flag keeps the exec attached until the command really ends, so its output and
        exit code are the command's own (without it, setsid forks and the exec returns 0 at once). Return the
        JobHandle at once (running, exit_code None); the caller drives pump_job to drain it. No file is written
        in the container (D-09). Unknown container -> ValueError, stopped -> RuntimeError."""
        ...

    async def pump_job(self, handle: JobHandle) -> None:
        """Drain the job's demuxed output into its buffers until the stream ends, then record the real exit
        code (None while running, never coerced to 0), clear running and set the completion event. The caller
        runs this inline (foreground), in a server-lifetime task group (background, 02-05), or waits on the
        handle's done event under a deadline (02-06)."""
        ...

    async def read_job(
        self, handle: JobHandle, stdout_offset: int, stderr_offset: int, max_bytes: int | None
    ) -> JobRead:
        """Return the new decoded text of each stream from the given offsets (at most max_bytes per stream),
        the next offsets, the skipped gaps, the per-stream dropped-byte counts, the running flag and exit
        code (D-08/D-22)."""
        ...

    async def kill_job(self, name: str, tag: str, grace_s: float) -> None:
        """Send SIGTERM to the tagged process group(s), then SIGKILL after grace_s (D-16, so tools can flush
        partial state). Afterwards no process with that exact tag remains (EXEC-06). Matches the full
        EXEGOL_MCP_JOB=<tag> string, never a prefix (T-02-02). The kill script runs as a plain /bin/sh argv, not
        through the SDK's zsh wrapper: zsh does not word-split unquoted parameters, so the group loops killed
        nothing (G-02-1). Under setsid -w a job spans two tagged groups; both are signalled."""
        ...

    async def sweep_orphans(self) -> int:
        """Kill every tagged process across the host's running containers and return how many orphaned jobs were
        found (the D-11 crash-recovery primitive; 02-05 wires it into the lifespan). The count is the number of
        distinct EXEGOL_MCP_JOB values, not the process-group count that setsid -w doubles. The sweep script runs
        as a plain /bin/sh argv (not the SDK zsh wrapper that broke word splitting, G-02-1)."""
        ...

    async def read_file(self, name: str, path: str, max_bytes: int, timeout_s: float) -> FileRead:
        """Read the file at path in the named running container, whatever the mount (rootfs, /proc, /sys, /dev,
        tmpfs, bind mounts). Reads at most max_bytes + 1 bytes, keeps them raw and sets over_limit when the file is
        larger than max_bytes. Every in-container command is bounded by timeout_s. Unknown container ->
        ValueError, stopped -> RuntimeError, missing file, directory, FIFO or socket -> ValueError, in-container
        time limit -> RuntimeError."""
        ...

    async def write_file(
        self, name: str, path: str, data: bytes, overwrite: bool, mode: int | None, timeout_s: float
    ) -> FileWrite:
        """Write data to path in the named running container with shell-redirection semantics: missing parent
        directories are created (0755), a symlink's target is updated, an existing file keeps its owner and mode
        unless mode (permission bits, 0 to 0o7777) is given, and the written size is checked. Unknown container ->
        ValueError, stopped -> RuntimeError, directory, FIFO or socket target -> ValueError, an existing file with
        overwrite False -> ValueError, in-container time limit -> RuntimeError, and a write that did not land
        (a size mismatch, a refused staging archive) -> RuntimeError. Never reports success for a lost write."""
        ...

    async def list_directory(
        self, name: str, path: str, depth: int, max_entries: int, timeout_s: float
    ) -> DirListing:
        """List the directory at path in the named running container down to depth levels (1: its direct entries),
        returning at most max_entries entries with truncated set when there were more. Symlinks are reported with
        their target and never followed (only a symlinked starting path is resolved). The command is bounded by
        timeout_s. Unknown container -> ValueError, stopped -> RuntimeError, missing path -> ValueError, not a
        directory -> ValueError, in-container time limit -> RuntimeError."""
        ...


@dataclass(frozen=True)
class ConfigStatus:
    """How the bundled Exegol library read the Exegol config file"""

    path: str
    # True when the library could not read the file (unreadable, not UTF-8, invalid YAML, a refused section)
    # and fell back to its defaults. exegol-mcp refuses to start in that case (D-07).
    refused: bool


@dataclass(frozen=True)
class ExegolMountDir:
    """A host directory Exegol bind-mounts into containers (read-write, or read-only), reads its mount settings from,
    exegol-mcp reads its own config from, or exegol-mcp and Exegol run code from (WR-10, CR-03)"""

    # The absolute realpath of the directory: what Docker finally binds.
    path: str
    # Where the directory comes from, readable by an operator and naming the option, for example
    # "Exegol's my-resources directory (customization.my_resources_path)".
    label: str
    # True when Exegol hands the directory's entries to containers one by one rather than the directory itself: the
    # private workspaces and Sentinel logs directories (their subdirectories, one per container) and the device
    # directories an allowlisted profile maps (system.devices), whose device nodes Docker maps into each container
    # (WR-14).
    children_mounted: bool
    # The absolute path Exegol hands Docker, os.path.abspath-normalized. "" means the same as path: a directory
    # reached without a symlink (hand-built fixtures).
    lexical: str = ""
    # Every host path Docker's bind resolution walks for the mount (adapter/mount_paths.mount_dir): the lexical
    # path, each looked-up entry with symlink targets followed, and the realpath of each prefix (CR-02). Empty means
    # path only.
    candidates: tuple[str, ...] = ()
    # True for a directory Exegol reads the settings that decide what containers mount from, rather than one it
    # mounts: its config directory and the container profile sources (IN-14 premise). An allowlist entry inside it,
    # containing it or equal to it is refused.
    holds_mount_settings: bool = False
    # True for exegol-mcp's own config directory and the one earlier versions used, which exegol-mcp moves its files
    # from once: exegol-mcp reads config.yml (the operator's restrictions) and the bearer secret from there. An
    # allowlist entry inside it, containing it or equal to it is refused, as for holds_mount_settings, with its own
    # reason.
    holds_server_config: bool = False
    # True for a directory exegol-mcp or Exegol runs code from on this host: the Python installation (sys.prefix,
    # sys.base_prefix, Exegol's install directory, exegol-mcp's package directory, CR-03) and every directory on its
    # import path at startup (sys.path, the working directory under python -m included, CR-04). An allowlist entry
    # inside it, containing it or equal to it is refused, as for holds_mount_settings, with its own reason.
    holds_host_code: bool = False
    # True for a path Exegol or an allowlisted profile binds read-only. No container writes through it, so an
    # allowlist entry inside it loads. Docker still walks its path at every create, so an entry containing any walked
    # path is refused (CR-03).
    contains_only: bool = False


@runtime_checkable
class HostProbe(Protocol):
    """Host state the startup readiness gate checks (D-05/D-07/D-08/D-15).

    Every method is synchronous and may block (Docker round trip, sqlite, file reads): callers run them through
    run_blocking(). The real implementation lives in adapter/host.py, the only side that imports exegol.
    """

    def assert_patch_points(self) -> None:
        """Raise naming the first Exegol SDK internal exegol-mcp relies on that is missing (D-15)."""
        ...

    def probe_docker(self) -> str:
        """Connect to the Docker daemon the way Exegol does and return a short status, or raise."""
        ...

    def ping_docker(self) -> str:
        """Ping the Docker daemon with a fresh, short-timeout client and return a short status, or raise."""
        ...

    def load_config(self) -> ConfigStatus:
        """Load the Exegol config with the bundled library and report whether it could be read."""
        ...

    def exegol_config_dir(self) -> str:
        """Return the Exegol config directory the bundled library uses."""
        ...

    def is_eula_accepted(self) -> bool:
        """Return whether the Exegol EULA is accepted in the datastore shared with the wrapper."""
        ...

    def update_eula(self, accepted: bool) -> None:
        """Record the EULA decision in the datastore shared with the wrapper."""
        ...

    def validate_profiles(self, names: Sequence[str]) -> list[str]:
        """Resolve the allowlisted container profiles through Exegol (D-02).

        Returns one problem string per allowlisted profile Exegol cannot resolve or validate, or a single licence
        problem when the host's cached Exegol licence session does not grant profiles. An empty list means ready.
        """
        ...

    def exegol_mount_dirs(self, profiles: Sequence[str]) -> list[ExegolMountDir]:
        """Return the host directories Exegol mounts into containers, reads its mount settings from or runs code from.

        These are Exegol's own (my-resources, exegol-resources, the private workspaces and Sentinel logs directories,
        as its config resolves them) and those the allowlisted profiles declare (WR-10), the directories Exegol reads
        its mount settings from, exegol-mcp's own config directory and the one earlier versions used, Exegol's
        read-only binds and the directories exegol-mcp and Exegol run code from,
        every directory on its import path included (CR-03, CR-04). Each entry carries the host paths Docker's bind
        resolution walks for it (candidates, CR-02). Raises when they cannot be read or walked.
        """
        ...
