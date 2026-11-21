"""Startup readiness gate (D-05/D-07/D-08/D-15, PROTO-05).

exegol-mcp refuses to start, with a distinct named message, rather than fail or hang inside a tool call when:

1. an Exegol SDK internal it patches or calls is missing (D-15 self-check),
2. Docker is unreachable,
3. the Exegol config cannot be read by the bundled Exegol library (D-07; a wrapper/library version difference is
   only information, reported by the banner and server_info, never a failure),
4. a container profile allowlisted in config.yml (profiles.allowed) cannot be resolved or validated by Exegol, or
   the host's cached Exegol licence session does not grant profiles (D-02, CONF-01); an empty allowlist skips it,
5. a workspace allowlist entry (lifecycle.workspace_allowlist in config.yml) is nested with a directory Exegol mounts
   into containers, maps devices from, reads its mount settings from, or runs code from, or with exegol-mcp's own
   config directory (WR-10, WR-11, CR-02, CR-03, CR-04, WR-14, WR-17, IN-25, WR-18, IN-28, D-24, T-05-50).
   bootstrap/config.py already refused dangerous entries and entries nested with each other when it loaded config.yml;
   this step adds the paths only Exegol's config, the allowlisted profiles, exegol-mcp's config store and this
   interpreter reveal, in six groups:
   - read-write mounts (Exegol's my-resources and exegol-resources directories, the X11 and Wayland GUI sockets, the
     read-write mounts of the allowlisted profiles): an entry inside one or containing one is refused;
   - directories whose entries Exegol hands to containers one by one: the private workspaces and Sentinel logs
     directories (Exegol's and the allowlisted profiles'), whose subdirectories it mounts, and the devices the
     allowlisted profiles map (system.devices), whose device nodes Docker maps. An entry inside, containing or equal
     to one is refused (WR-14);
   - read-only binds (spawn.sh, entrypoint.sh, the Sentinel hooks and logger from Exegol's install directory,
     /etc/timezone and /etc/localtime on Linux, the read-only mounts and VPN files of the allowlisted profiles): an
     entry containing one is refused, and an entry inside a read-only mount loads;
   - settings directories (Exegol's config directory, its container profile directory, the path: profile sources):
     an entry inside, containing or equal to one is refused (IN-14 premise, 04-21);
   - exegol-mcp's own config directories (~/.exegol-mcp, the invoking user's under sudo, where it reads config.yml
     and the bearer secret, and the platform directory earlier versions used, which it moves those files from once,
     listed even when missing): an entry inside, containing or equal to one is refused, so no container can rewrite
     the operator's restrictions or read the secret (T-05-50);
   - the Python installation exegol-mcp and Exegol run code from (sys.prefix, sys.base_prefix, the virtual environment
     the interpreter starts in (the directory holding the pyvenv.cfg site reads) even under -S, Exegol's install
     directory, exegol-mcp's package directory), its user site directory for this start's HOME and PYTHONUSERBASE even
     before it exists (WR-17) and even when a start disables it with -s, -S, -I or PYTHONNOUSERSITE, except in a virtual
     environment whose pyvenv.cfg (the one CPython reads, next to the interpreter or one level up) sets
     include-system-site-packages to anything but true, which never reads it unless a start's PYTHONEXECUTABLE or
     __PYVENV_LAUNCHER__ (macOS) names another executable; an unreadable pyvenv.cfg counts as true (IN-25, WR-18, IN-28,
     IN-31, IN-32), and every directory on its import path at startup (sys.path: PYTHONPATH and .pth entries, and the
     MCP client's working directory under a `python -m` or `python -c` launch typed by hand): an entry inside,
     containing or equal to one is refused (CR-03, CR-04). The launch --print-config prints keeps the working directory
     off the import path: the console script resolved through symlinks (WR-15), or without one `python -P -m exegol_mcp`
     on 3.11+ or a `python -c` launcher that drops it on 3.10 (WR-16).
   "Inside" compares the realpath. "Contains" covers every path Docker walks to mount the path, symlinks included,
   not only its realpath (CR-02, adapter/mount_paths.py). Letter case is folded on macOS and Windows (IN-08). If
   those paths cannot be read, startup stops (fail closed); an empty allowlist skips the step,
6. the Exegol EULA is not accepted and no operator path can accept it (D-05, see eula.py).

Every probe goes through run_blocking(), so the checks never block an event loop. At startup a failure raises
ReadinessError, which main() prints on the standard error stream before exiting with status 1. After startup,
recheck() runs the cheap checks again when a tool call fails, and its ReadinessError is a RuntimeError, so the
tool boundary hands its message to the agent (D-08). recheck() never re-checks profiles or the workspace allowlist.
A create re-reads its profile and re-checks that its workspace is still an exact allowlist entry that resolves to
itself, but it does not repeat step 5: an allowlisted profile edited after startup is checked at the next start
(accepted residual AR-04-13, IN-14). No allowlisted directory may overlap Exegol's config directory or a container
profile source, so that edit takes an operator action.

Step 5 does not check, and AR-04-13 accepts: the Wayland socket of another environment, such as the exegol wrapper's
(IN-17); Exegol's /dev/net/tun device; a profile's network.hosts_file, which Exegol reads on the host and does not
mount; links an operator places in a profile source; directories other than the user site added to Python's import path
after startup, code an import hook loads from outside it, a user site another start's HOME or PYTHONUSERBASE selects
(IN-29), a user site a start reaches through PYTHONEXECUTABLE or __PYVENV_LAUNCHER__ on macOS (IN-31), and another
Python environment, such as the exegol wrapper's own; the Python installation or an import-path directory lying inside a
directory Exegol mounts, which only the operator's own setup puts there (an installation or a PYTHONPATH or .pth entry
placed there, or a hand-written `python -m` or `python -c` launch started there), since every launch --print-config
prints keeps the working directory off the import path (T-04-96); the WSLg socket of a native Windows host; and other
directories the host runs or trusts (PATH directories, the directory of the interpreter a printed python launch names
when it lies outside the Python installation, ~/.ssh). Mounts exegol-mcp cannot see (the exegol wrapper's -V, a
hand-made docker run, a profile not on the allowlist, an Exegol config edited after startup) are not checked either.
"""

from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.port import ExegolMountDir, HostProbe
from exegol_mcp.bootstrap.eula import EulaNotAcceptedError, EulaOutcome, ask_on_stderr, resolve_eula
from exegol_mcp.services.host_paths import is_strictly_inside, path_parts

ReadinessCheck = Literal["patch-points", "docker", "config", "profiles", "workspace-allowlist", "eula"]

DOCKER_UNREACHABLE_MESSAGE = (
    "Docker is not reachable, so Exegol cannot run. Make sure Docker is running and that this user can use it, "
    "then check that the Exegol wrapper works with `exegol info`. Error: {error}"
)

CONFIG_UNREADABLE_MESSAGE = (
    "The Exegol config ({path}) cannot be read by the Exegol library bundled with exegol-mcp ({library}). "
    "exegol-mcp never migrates the Exegol config: upgrade the Exegol wrapper and run it once (for example "
    "`exegol info`), or fix the file, then start exegol-mcp again."
)

PROFILES_UNUSABLE_MESSAGE = "Container profiles listed in config.yml (profiles.allowed) cannot be used: "

WORKSPACE_ALLOWLIST_UNCHECKED_MESSAGE = (
    "The directories Exegol mounts into containers, or the directories exegol-mcp imports code from (sys.path, the "
    "working directory and the user site), could not be read, so lifecycle.workspace_allowlist in config.yml cannot "
    "be checked and exegol-mcp refuses to start. Error: {error}"
)

_MOUNT_PROBLEM_FIX = (
    "Allow a directory outside every directory Exegol or an allowlisted profile mounts, device directories included."
)

_SETTINGS_PROBLEM_FIX = "Allow a directory apart from Exegol's config directory and its container profile sources."

_SERVER_CONFIG_PROBLEM_FIX = (
    "Allow a directory apart from exegol-mcp's config directory (~/.exegol-mcp, and the one earlier exegol-mcp "
    "versions used) and from Exegol's config directory and its container profile sources."
)

_INSTALL_PROBLEM_FIX = (
    "Allow a directory apart from the Python installation exegol-mcp runs from: its environment, its base "
    "installation, the exegol and exegol-mcp packages, its user site directory and every directory on its import "
    "path (sys.path). Start exegol-mcp as --print-config prints it (its console script, or a python launch that keeps "
    "the working directory off that path), and keep the working directory out of PYTHONPATH, so the MCP client's "
    "working directory is not on that path."
)

_READ_ONLY_PROBLEM_FIX = (
    "Allow a directory that holds no path Exegol or an allowlisted profile binds into containers, read-only binds "
    "included."
)

EULA_REVOKED_MESSAGE = (
    "The Exegol EULA is no longer accepted on this machine. Run `exegol-mcp` or `exegol` in a terminal to accept "
    "it again, or restart exegol-mcp with --accept-eula."
)


class ReadinessError(RuntimeError):
    """exegol-mcp is not ready to serve; check names the failing step"""

    def __init__(self, check: ReadinessCheck, message: str) -> None:
        super().__init__(message)
        self.check = check


@dataclass(frozen=True)
class ReadinessReport:
    """What the readiness gate established, reused by the banner and server_info"""

    docker_status: str
    exegol_config_dir: str
    exegol_config_file: str
    eula: EulaOutcome


def _library_version() -> str:
    from importlib import metadata

    try:
        return f"exegol {metadata.version('exegol')}"
    except metadata.PackageNotFoundError:
        return "exegol, unknown version"


MountRelation = Literal["inside", "contains", "same"]


def _mount_relation(entry: Path, mount: ExegolMountDir, *, fold_case: bool | None) -> tuple[MountRelation, str] | None:
    """How an allowlist entry is nested with a directory Exegol mounts, and the witness host path; None when safe.

    "inside": the entry lies below the mount's realpath, so a container could swap the entry for a symlink. Witness:
    the realpath. "contains": a host path Docker's bind resolution walks for the mount (its lexical path, any of its
    candidates or its realpath, CR-02) lies below the entry, so a container mounting the entry could swap that path
    and redirect the mount. It is also "contains" when one of those paths is the entry itself while the mount's
    lexical path is not: the mount reaches the entry through a symlink, and a symlink chain through the entry's own
    children cannot be told apart. Witness: that path. "same": the entry is the mount's realpath, and Exegol either
    hands its entries to containers one by one (its subdirectories, one per container, or the device nodes of a
    profile's device directory, WR-14), so a container mounting the entry could plant the next one, reads its mount
    settings from it (holds_mount_settings, IN-14 premise), exegol-mcp reads its own config.yml and bearer secret
    from it (holds_server_config, T-05-50), or exegol-mcp and Exegol run code from it (holds_host_code, CR-03).
    Witness: the realpath.

    A read-only bind source (contains_only, CR-03) is never "inside" or "same": no container writes through it, so
    an entry inside it or equal to it loads; only "contains" applies. An entry equal to a directly
    mounted directory reached without a symlink is allowed, and whole components are compared, with letter case
    folded per fold_case (IN-08), so a sibling sharing a name prefix is never nested.
    """
    realpath = Path(mount.path)
    # Read-only stops writes through the bind, so an entry inside a contains_only source cannot be swapped by a
    # container that mounts the source; the paths Docker walks to reach it are still checked below.
    if not mount.contains_only and is_strictly_inside(entry, realpath, fold_case=fold_case):
        return "inside", mount.path
    lexical = mount.lexical or mount.path
    walked = (lexical, *mount.candidates, mount.path)
    for candidate in walked:
        if is_strictly_inside(Path(candidate), entry, fold_case=fold_case):
            return "contains", candidate
    entry_parts = path_parts(entry, fold_case=fold_case)
    if path_parts(Path(lexical), fold_case=fold_case) != entry_parts:
        for candidate in walked:
            if path_parts(Path(candidate), fold_case=fold_case) == entry_parts:
                return "contains", candidate
    same_dir = path_parts(realpath, fold_case=fold_case) == entry_parts
    if same_dir and (
        mount.children_mounted or mount.holds_mount_settings or mount.holds_server_config or mount.holds_host_code
    ):
        return "same", mount.path
    return None


def _mount_reason(relation: MountRelation, witness: str, mount: ExegolMountDir, lexical_witness: bool) -> str:
    """The reason half of a refusal for a directory Exegol mounts read-write or maps devices from, ending with its fix
    (WR-10, CR-02, WR-14)."""
    lexical = mount.lexical or mount.path
    if relation == "inside" and mount.children_mounted:
        reason = (
            f"lies inside {mount.label} ('{mount.path}'), whose subdirectories or device nodes Exegol hands to "
            "containers: a container could replace the entry with a symlink, or plant a device node in it, before "
            "the next create, so the entry is refused (D-24, WR-14)."
        )
    elif relation == "inside":
        reason = (
            f"lies inside {mount.label} ('{mount.path}'), which Exegol mounts read-write into containers: a "
            "container could replace the entry with a symlink before Docker mounts it, so the entry is refused (D-24)."
        )
    elif relation == "contains" and lexical_witness:
        reason = (
            f"contains {mount.label} ('{lexical}'): a container that mounts the entry could replace that directory "
            "with a symlink before Exegol mounts it, so the entry is refused (D-24)."
        )
    elif relation == "contains":
        reason = (
            f"holds '{witness}', which Docker walks through, following symlinks, to mount {mount.label} "
            f"('{lexical}'): a container that mounts the entry could replace it with a symlink and redirect that "
            "mount, so the entry is refused (D-24)."
        )
    else:
        reason = (
            f"is {mount.label}, whose subdirectories or device nodes Exegol hands to containers: a container that "
            "mounts the entry could plant a symlink or a device node there for the next create, so the entry is "
            "refused (D-24, WR-14)."
        )
    return f"{reason} {_MOUNT_PROBLEM_FIX}"


def _nesting_phrase(relation: MountRelation, witness: str, mount: ExegolMountDir, lexical_witness: bool) -> str:
    """How the entry is nested with a directory refused inside, containing or equal: the first half of the reason
    for the settings, exegol-mcp config and host-code directories."""
    lexical = mount.lexical or mount.path
    if relation == "inside":
        return f"lies inside {mount.label} ('{mount.path}')"
    if relation == "same":
        return f"is {mount.label} ('{mount.path}')"
    if lexical_witness:
        return f"contains {mount.label} ('{lexical}')"
    return f"holds '{witness}', which the path to {mount.label} ('{lexical}') goes through, following symlinks"


def _settings_reason(relation: MountRelation, witness: str, mount: ExegolMountDir, lexical_witness: bool) -> str:
    """The reason half of a refusal for a directory Exegol reads its mount settings from (IN-14 premise)."""
    phrase = _nesting_phrase(relation, witness, mount, lexical_witness)
    return (
        f"{phrase}: Exegol reads the settings that decide what containers mount from there, so an allowlisted "
        f"directory may not lie inside it, contain it or be it, and the entry is refused (D-24, IN-14). "
        f"{_SETTINGS_PROBLEM_FIX}"
    )


def _server_config_reason(relation: MountRelation, witness: str, mount: ExegolMountDir, lexical_witness: bool) -> str:
    """The reason half of a refusal for exegol-mcp's own config directory or the one earlier versions used, which it
    moves config.yml and the bearer secret from once (holds_server_config, T-05-50). Operator-facing: no planning
    IDs."""
    phrase = _nesting_phrase(relation, witness, mount, lexical_witness)
    return (
        f"{phrase}: exegol-mcp reads its config.yml (the operator's restrictions) and its bearer secret from there, so "
        "a container that mounts the entry could change those restrictions or read the secret; an allowlisted "
        f"directory may not lie inside it, contain it or be it, and the entry is refused. {_SERVER_CONFIG_PROBLEM_FIX}"
    )


def _host_code_reason(relation: MountRelation, witness: str, mount: ExegolMountDir, lexical_witness: bool) -> str:
    """The reason half of a refusal for a directory exegol-mcp and Exegol run code from: the Python installation, its
    user site directory and every directory on its import path (holds_host_code, CR-03, CR-04)."""
    phrase = _nesting_phrase(relation, witness, mount, lexical_witness)
    return (
        f"{phrase}: exegol-mcp and Exegol run code from there on this host, so a container that mounts the entry "
        f"could change what runs on the host; the entry is refused (D-24, CR-03). {_INSTALL_PROBLEM_FIX}"
    )


def _read_only_reason(relation: MountRelation, witness: str, mount: ExegolMountDir, lexical_witness: bool) -> str:
    """The reason half of a refusal for a path bound read-only: only "contains" (CR-03). A profile's device is refused
    like a children-mounted directory instead (WR-14)."""
    lexical = mount.lexical or mount.path
    if lexical_witness:
        reason = (
            f"contains {mount.label} ('{lexical}'): Docker resolves that path, following symlinks, each time it "
            "creates a container that binds it, so a container that mounts the entry could replace it with a symlink "
            "and expose any host path to the next container, read-only binds included; the entry is refused "
            "(D-24, CR-03)."
        )
    else:
        reason = (
            f"holds '{witness}', which Docker walks through, following symlinks, to bind {mount.label} "
            f"('{lexical}'): a container that mounts the entry could replace it with a symlink and expose any host "
            "path to the next container, read-only binds included; the entry is refused (D-24, CR-03)."
        )
    return f"{reason} {_READ_ONLY_PROBLEM_FIX}"


def workspace_allowlist_mount_problem(
    entries: Sequence[str], mounts: Sequence[ExegolMountDir], *, fold_case: bool | None = None
) -> str | None:
    """Return the first allowlist entry nested with a path Exegol mounts, reads or runs from, as a config.yml message.

    Every path Docker's bind resolution walks for a mount counts (its candidates, CR-02): an entry holding any of them
    is refused, and the message names that witness path. A directory Exegol reads its mount settings from (its config
    directory, a container profile source: holds_mount_settings) is refused when the entry lies inside it, contains
    it or is it, with its own wording and fix (IN-14 premise). So is exegol-mcp's own config directory, and the one
    earlier versions used, which it reads config.yml and the bearer secret from (holds_server_config, T-05-50), with
    its own wording and a fix naming both config directories. So is a directory exegol-mcp and Exegol run code from
    (the Python installation, its user site directory and every directory on its import path: holds_host_code,
    CR-03, CR-04), with its own wording and fix. A path bound read-only (contains_only, CR-03) is refused only when
    the entry contains a path Docker walks for it. A device an allowlisted profile maps is refused like a
    children-mounted directory: inside, containing or equal (WR-14).

    Args:
        entries: lifecycle.workspace_allowlist as bootstrap/config.py stored it (absolute realpath strings).
        mounts: The paths Exegol and the allowlisted profiles mount, read-write or read-only, those Exegol reads its
            mount settings from, exegol-mcp's own config directories and those exegol-mcp and Exegol run code from
            (HostProbe.exegol_mount_dirs).
        fold_case: Fold letter case; None follows services.host_paths.CASE_INSENSITIVE_HOST (IN-08).

    Returns:
        None when no entry is nested with one of them; otherwise the problem for the first one, in entry order then
        mount order, naming the entry's index, the entry, the directory (or the walked path) and where it comes from.
    """
    for index, raw in enumerate(entries):
        entry = Path(raw)
        for mount in mounts:
            found = _mount_relation(entry, mount, fold_case=fold_case)
            if found is None:
                continue
            relation, witness = found
            lexical_witness = path_parts(Path(witness), fold_case=fold_case) == path_parts(
                Path(mount.lexical or mount.path), fold_case=fold_case
            )
            prefix = f"Invalid config.yml: field 'lifecycle.workspace_allowlist.{index}': '{raw}' "
            if mount.holds_host_code:
                return prefix + _host_code_reason(relation, witness, mount, lexical_witness)
            if mount.holds_mount_settings:
                return prefix + _settings_reason(relation, witness, mount, lexical_witness)
            if mount.holds_server_config:
                return prefix + _server_config_reason(relation, witness, mount, lexical_witness)
            if mount.contains_only:
                return prefix + _read_only_reason(relation, witness, mount, lexical_witness)
            return prefix + _mount_reason(relation, witness, mount, lexical_witness)
    return None


async def check_readiness(
    probe: HostProbe,
    args: Namespace,
    prompt: Callable[[], bool] = ask_on_stderr,
    *,
    profiles: Sequence[str] = (),
    workspace_allowlist: Sequence[str] = (),
) -> ReadinessReport:
    """Run the startup readiness gate, in order, and stop at the first failure.

    Args:
        probe: Host probes over the Exegol SDK (RealHostProbe) or a fake in tests.
        args: The parsed command line (accept_eula and type drive the EULA step).
        prompt: The terminal EULA prompt, replaced in tests.
        profiles: The container profile allowlist from config.yml (profiles.allowed); empty skips the step.
        workspace_allowlist: lifecycle.workspace_allowlist from config.yml, as resolved realpaths; empty skips the
            Exegol-mount step.

    Returns:
        The Docker status and Exegol config location, and how the EULA was accepted.

    Raises:
        ReadinessError: Naming the failing check, with a message the operator can act on.
    """
    # 1. Every SDK internal exegol-mcp patches or calls is present (D-15).
    try:
        await run_blocking(probe.assert_patch_points)
    except Exception as e:  # any failure here is the patch point's own named error
        raise ReadinessError("patch-points", str(e)) from e

    # 2. Docker is reachable (v1's actionable `exegol info` message, with the cause).
    try:
        docker_status = await run_blocking(probe.probe_docker)
    except Exception as e:  # docker-py, requests and SDK criticals all mean "unreachable"
        raise ReadinessError("docker", DOCKER_UNREACHABLE_MESSAGE.format(error=e)) from e

    # 3. The bundled library can read the Exegol config; a version mismatch is not checked here (D-07/D-12).
    config_dir = await run_blocking(probe.exegol_config_dir)
    try:
        config = await run_blocking(probe.load_config)
    except Exception as e:  # an SDK critical while loading the config means unreadable
        message = CONFIG_UNREADABLE_MESSAGE.format(path=f"{config_dir}/config.yml", library=_library_version())
        raise ReadinessError("config", f"{message} Error: {e}") from e
    if config.refused:
        raise ReadinessError("config", CONFIG_UNREADABLE_MESSAGE.format(path=config.path, library=_library_version()))

    # 4. Every allowlisted container profile resolves through Exegol, licence-gated (D-02). The probe does the SDK
    # work, so bootstrap stays SDK-free; an empty allowlist adds no startup work.
    if profiles:
        problems = await run_blocking(probe.validate_profiles, list(profiles))
        if problems:
            raise ReadinessError("profiles", PROFILES_UNUSABLE_MESSAGE + "; ".join(problems))

    # 5. No allowlisted workspace is nested with a directory Exegol or an allowlisted profile mounts read-write
    # (WR-10), contains a path Exegol or an allowlisted profile binds read-only, or is nested with a directory
    # exegol-mcp and Exegol run code from: the Python installation and every directory on its import path, the user
    # site included even before it exists (CR-03, CR-04, WR-17) and even when this start disables it, unless the
    # pyvenv.cfg CPython reads (next to the interpreter or one level up) excludes system site packages (IN-25, WR-18).
    # Fail closed when those paths cannot be read; an empty allowlist adds no startup work.
    if workspace_allowlist:
        try:
            mounts = await run_blocking(probe.exegol_mount_dirs, list(profiles))
        except Exception as e:  # an SDK critical, a renamed profile field or an unreadable config: cannot check
            raise ReadinessError("workspace-allowlist", WORKSPACE_ALLOWLIST_UNCHECKED_MESSAGE.format(error=e)) from e
        problem = workspace_allowlist_mount_problem(workspace_allowlist, mounts)
        if problem is not None:
            raise ReadinessError("workspace-allowlist", problem)

    # 6. The EULA is accepted, or an operator accepts it now (D-05).
    try:
        eula = await resolve_eula(probe, args, prompt)
    except EulaNotAcceptedError as e:
        raise ReadinessError("eula", str(e)) from e

    return ReadinessReport(
        docker_status=docker_status,
        exegol_config_dir=config_dir,
        exegol_config_file=config.path,
        eula=eula,
    )


async def recheck(probe: HostProbe, accept_eula: bool) -> None:
    """Tool-time re-check of the state that can change after startup: Docker and the EULA (D-08).

    Never prompts. Called when a tool call fails, to turn an opaque failure into a clear reason.

    Raises:
        ReadinessError: A RuntimeError naming what is no longer ready.
    """
    try:
        await run_blocking(probe.ping_docker)
    except Exception as e:  # any failure to ping means the daemon went away
        raise ReadinessError("docker", DOCKER_UNREACHABLE_MESSAGE.format(error=e)) from e
    if not accept_eula and not await run_blocking(probe.is_eula_accepted):
        raise ReadinessError("eula", EULA_REVOKED_MESSAGE)
