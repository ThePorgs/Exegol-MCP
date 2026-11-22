"""RealHostProbe: the HostProbe the startup readiness gate runs against exegol 6.x.

Bootstrap code never imports exegol (D-19). The readiness gate and the EULA flow reach Docker, the Exegol config
and the EULA datastore shared with the wrapper only through this probe. Every method is synchronous and blocking:
the readiness gate calls each one through run_blocking().
"""

import os
import site
import sys
from collections.abc import Sequence
from pathlib import Path

import docker
from exegol.config.ConstantConfig import ConstantConfig
from exegol.config.EnvInfo import EnvInfo
from exegol.config.OptionResolver import OptionKey, OptionResolver
from exegol.config.UserConfig import UserConfig
from exegol.exceptions.ExegolExceptions import CancelOperation
from exegol.utils.DockerUtils import DockerUtils
from exegol.utils.GuiUtils import GuiUtils
from exegol.utils.LocalDatastore import LocalDatastore

from exegol_mcp.adapter import sudo
from exegol_mcp.adapter.mount_paths import mount_dir
from exegol_mcp.adapter.port import ConfigStatus, ExegolMountDir
from exegol_mcp.adapter.profiles import profile_mount_dirs, profile_problems
from exegol_mcp.adapter.selfcheck import _broken, assert_patch_points
from exegol_mcp.bootstrap import secret_store

# Bound on the tool-time Docker ping: a dead daemon must not stall the error the agent is waiting for.
_PING_TIMEOUT_SECONDS = 5

# Exegol's own host directories it bind-mounts read-write into containers (WR-10): the option each is resolved from, the
# operator-readable origin, and whether Exegol mounts the directory's subdirectories (one per container) rather than
# the directory itself. my-resources and exegol-resources are mounted whole (ContainerConfig.enableMyResources,
# enableExegolResources); a private workspace is <private_workspace_path>/<name> (prepareShare) and the Sentinel
# directory of a container sits under the Sentinel logs directory. The GUI sockets, the directories Exegol reads its
# mount settings from, the Python installation exegol-mcp and Exegol run code from with every directory on its import
# path (CR-03, CR-04) and its user site directory even before it exists (WR-17, IN-25), and Exegol's read-only bind
# sources are listed by _gui_socket_dirs, _exegol_settings_dirs, _python_install_dirs and _exegol_read_only_binds;
# exegol-mcp's own config directory and the ones earlier versions used by _exegol_mcp_config_dirs.
_EXEGOL_MOUNT_OPTIONS: tuple[tuple[OptionKey, str, bool], ...] = (
    (OptionKey.MY_RESOURCES_PATH, "Exegol's my-resources directory", False),
    (OptionKey.EXEGOL_RESOURCES_PATH, "Exegol's exegol-resources directory", False),
    (OptionKey.PRIVATE_WORKSPACE_PATH, "Exegol's private workspaces directory", True),
    (OptionKey.SENTINEL_LOGS_HOST_PATH, "Exegol's Sentinel logs directory", True),
)


# exegol-mcp's package directory as Python imported it (host.py lives in exegol_mcp/adapter/): the checkout's
# exegol_mcp/ for an editable install. A module attribute so tests can point it at a tmp tree.
_EXEGOL_MCP_PACKAGE_DIR: Path = Path(__file__).absolute().parent.parent


def _pyvenv_cfg_site_reads() -> tuple[str, str] | None:
    """The virtual environment this interpreter starts in and the pyvenv.cfg CPython's site.venv() reads there, as
    (site_prefix, cfg_path), or None (IN-28, IN-32).

    site.venv() takes the interpreter's executable: on macOS the __PYVENV_LAUNCHER__ value when that variable is set,
    otherwise sys.executable. It reads the first of <the executable's directory>/pyvenv.cfg and the pyvenv.cfg one
    level up that os.path.isfile accepts, and makes sys.prefix the directory one level up from the executable's
    (site_prefix) whichever candidate it read. Both candidates and the interpreter directory lie under site_prefix.
    None when the executable is empty or None, or when no candidate is a file (not a venv). Only reads.
    """
    executable: str | None
    if sys.platform == "darwin" and "__PYVENV_LAUNCHER__" in os.environ:
        executable = os.environ["__PYVENV_LAUNCHER__"]
    else:
        executable = sys.executable
    if not executable:
        return None
    exe_dir = os.path.dirname(os.path.abspath(executable))
    site_prefix = os.path.dirname(exe_dir)
    candidates = (os.path.join(exe_dir, "pyvenv.cfg"), os.path.join(site_prefix, "pyvenv.cfg"))
    found = [candidate for candidate in candidates if os.path.isfile(candidate)]
    if not found:
        return None
    return site_prefix, found[0]


def _user_site_read_by_a_normal_start() -> bool:
    """Whether a start of this interpreter without -s, -S, -I or PYTHONNOUSERSITE reads the user site, decided as
    CPython's site.venv() decides it (IN-25, WR-18, IN-28).

    site.venv() takes the interpreter's executable: on macOS the __PYVENV_LAUNCHER__ value when that variable is set,
    otherwise sys.executable. It reads the first of <the executable's directory>/pyvenv.cfg and the pyvenv.cfg one level
    up that os.path.isfile accepts. With no such file the interpreter is not a venv, and site reads the user site unless
    a flag disables it. Otherwise the file is parsed as site.venv() parses it: UTF-8, only lines containing '=', the key
    and the value stripped and lower-cased, the last include-system-site-packages line wins, and "true" when the key is
    absent. Any value but "true" disables the user site at every start unless a start's PYTHONEXECUTABLE or
    __PYVENV_LAUNCHER__ (macOS) names another executable (IN-31).

    sys.prefix plays no part: under -S, site.venv() never runs, and on 3.10-3.12 sys.prefix then stays the base
    installation's even in a venv (IN-28, IN-32). The executable and the file come from _pyvenv_cfg_site_reads. An
    empty or None executable, or a file that cannot be read or decoded (OSError, ValueError), returns True (fail
    closed). Only reads.
    """
    reads = _pyvenv_cfg_site_reads()
    if reads is None:
        return True
    system_site = "true"
    try:
        with open(reads[1], encoding="utf-8") as config:
            for line in config:
                if "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip().lower() == "include-system-site-packages":
                        system_site = value.strip().lower()
    except (OSError, ValueError):  # unreadable or not UTF-8 (UnicodeDecodeError): fail closed
        return True
    return system_site == "true"


def _python_install_dirs() -> list[ExegolMountDir]:
    """The Python installation exegol-mcp and Exegol run code from and its import path, for the startup allowlist
    check (CR-03, CR-04).

    An allowlisted directory that lies inside, contains or is one of these lets an agent's container change code that
    exegol-mcp, Exegol and the exegol wrapper of the same environment run on the host, as the operator or as root
    under sudo. Exegol's install directory also holds the files it binds read-only into containers
    (_exegol_read_only_binds).

    sys.path is where the host imports code from (CR-04). Under `python -m exegol_mcp` (and `python -c`), sys.path[0]
    is the working directory the MCP client started exegol-mcp in, often the project directory an operator
    allowlists; PYTHONPATH, .pth files (an editable checkout) and the user site add entries too. An allowlisted
    directory nested with one lets a container plant a module the host imports at the next start or the next lazy
    import.

    site adds the user site directory (site.getusersitepackages(), computed from this start's HOME and PYTHONUSERBASE)
    to the import path at startup only when it exists and the user site is enabled (WR-17). A container that creates it
    under an allowlisted entry gets a .pth file or usercustomize.py run at the next interpreter start, before main() and
    before any check. It is therefore listed, after the sys.path entries and even before it exists, whenever
    site.ENABLE_USER_SITE is true or a start of this interpreter without -s, -S, -I and PYTHONNOUSERSITE reads it
    (_user_site_read_by_a_normal_start: no pyvenv.cfg where site.venv() looks, one that includes system site packages,
    one that cannot be read, or an unknown executable). -s, -S, -I and PYTHONNOUSERSITE disable the user site for one
    start only, and a later start without them reads it (IN-25). A venv whose pyvenv.cfg excludes system site packages
    never reads it unless a start's PYTHONEXECUTABLE or __PYVENV_LAUNCHER__ (macOS) names another executable, so it is
    not listed there; that pyvenv.cfg is the one site.venv() reads, next to the interpreter or one level up (WR-18,
    IN-28, IN-31). mount_dir walks a missing path lexically.

    Listed, read at call time and in this order, each built by mount_dir with holds_host_code=True: sys.prefix,
    sys.base_prefix, the virtual environment this interpreter starts in (the directory holding the pyvenv.cfg site
    reads: listed when site.venv() finds a pyvenv.cfg, the dirname of the executable's directory, and dropped as a
    repeat of sys.prefix on a normal start; under -S on 3.10-3.12 it is what keeps both pyvenv.cfg candidates and the
    interpreter directory protected, IN-32), Exegol's install directory (ConstantConfig.src_root_path_obj:
    site-packages for a pip install, the clone for a git install), exegol-mcp's package directory, every sys.path
    entry in order, then the user site.
    A str, bytes or os.PathLike entry is decoded with os.fsdecode; '' is the working directory; any other object is
    skipped. An empty value is skipped, and so is one whose absolute path repeats an earlier root, so the earlier
    label wins (a user site already on sys.path keeps its import-path label).
    Not listed: directories other than the user site added to the import path after startup, code an import hook loads
    from outside it, a user site another start's HOME or PYTHONUSERBASE selects (IN-29), a user site a start reaches
    through PYTHONEXECUTABLE or __PYVENV_LAUNCHER__ on macOS (IN-31), and another Python environment (the exegol
    wrapper's own install when it is a different one). Not checked here: whether one of these directories itself lies
    inside a directory Exegol mounts, which only the operator's own setup puts there (an installation, a PYTHONPATH or
    .pth entry, or a hand-written `python -m` or `python -c` launch started there). Any error propagates, an unreadable
    working directory included (fail closed).
    """
    roots: list[tuple[str | os.PathLike[str], str]] = [
        (sys.prefix, "the Python environment exegol-mcp runs from (sys.prefix)"),
        (sys.base_prefix, "the Python installation that environment is based on (sys.base_prefix)"),
        (ConstantConfig.src_root_path_obj, "Exegol's install directory (ConstantConfig.src_root_path_obj)"),
        (_EXEGOL_MCP_PACKAGE_DIR, "exegol-mcp's package directory"),
    ]
    reads = _pyvenv_cfg_site_reads()
    if reads is not None:
        # IN-32: under -S on 3.10-3.12 sys.prefix is the base installation's, so this root is what keeps both pyvenv.cfg
        # candidates and the interpreter directory protected; on a normal start it repeats sys.prefix and is dropped.
        venv_label = (
            "the virtual environment this interpreter starts in (the directory holding the pyvenv.cfg site reads)"
        )
        roots.insert(2, (reads[0], venv_label))
    cwd = os.getcwd()
    import_path: list[object] = list(sys.path)
    for entry in import_path:
        # The import system reads only path entries; anything else on sys.path is ignored by it too.
        if not isinstance(entry, (str, bytes, os.PathLike)):
            continue
        text = os.fsdecode(entry)
        if text == "" or os.path.abspath(text) == cwd:
            label = (
                f"the working directory exegol-mcp was started in, which is on its import path (sys.path entry "
                f"'{text}': a python -m or -c launch, or an empty or explicit PYTHONPATH entry, puts it there)"
            )
        else:
            label = f"a directory on exegol-mcp's import path (sys.path entry '{text}')"
        # Python reads an empty entry as the working directory at each import; mount_dir makes a relative entry
        # absolute against the working directory, as the import system resolves it.
        roots.append((cwd if text == "" else text, label))
    if site.ENABLE_USER_SITE or _user_site_read_by_a_normal_start():
        user_site_label = (
            "the user site directory this Python adds to its import path at startup when it exists "
            "(site.getusersitepackages()), listed even before it exists and even when this start disables it"
        )
        roots.append((site.getusersitepackages(), user_site_label))
    dirs: list[ExegolMountDir] = []
    seen: set[str] = set()
    for value, label in roots:
        if not value:
            continue
        absolute = os.path.abspath(os.fspath(value))
        if absolute in seen:
            continue
        seen.add(absolute)
        dirs.append(mount_dir(value, label, False, holds_host_code=True))
    return dirs


def _exegol_read_only_binds() -> list[ExegolMountDir]:
    """The sources Exegol binds read-only into containers, for the startup allowlist check (CR-03).

    Read-only stops writes through the bind, not the swap of a path Docker walks to reach the source: a container
    that mounts an entry holding the source, or any path on the way to it, could replace it with a symlink before the
    next create, and must_exist follows the symlink, so the next container would bind any host file. Each source is
    built by mount_dir with contains_only=True: an entry containing a walked path is refused, one inside loads.

    Mirrors four sources: ContainerConfig.__init__ (spawn.sh, every new container), ExegolManager's temporary
    container path (entrypoint.sh), enableSentinel (the zsh and bash hooks and the logger) and enableSharedTimezone
    (/etc/timezone and /etc/localtime, only on a Linux shell host: not EnvInfo.is_windows_shell or is_mac_shell).
    They are listed whatever the Sentinel and timezone flags say, because the operator's config, a profile or the
    wrapper can enable them. A source that does not exist yet is still walked. Exegol's /dev/net/tun device for VPN,
    a fixed path under /dev, is not listed. Attributes are read directly, so a renamed one raises (fail closed).
    """
    binds: list[tuple[str | os.PathLike[str], str]] = [
        (
            ConstantConfig.spawn_context_path_obj,
            "Exegol's read-only bind of spawn.sh (ConstantConfig.spawn_context_path_obj, every new container)",
        ),
        (
            ConstantConfig.entrypoint_context_path_obj,
            (
                "Exegol's read-only bind of entrypoint.sh "
                "(ConstantConfig.entrypoint_context_path_obj, temporary containers)"
            ),
        ),
        (
            ConstantConfig.sentinel_zsh_context_path_obj,
            "Exegol's read-only bind of the Sentinel zsh hooks (ConstantConfig.sentinel_zsh_context_path_obj)",
        ),
        (
            ConstantConfig.sentinel_bash_context_path_obj,
            "Exegol's read-only bind of the Sentinel bash hooks (ConstantConfig.sentinel_bash_context_path_obj)",
        ),
        (
            ConstantConfig.sentinel_context_path_obj,
            "Exegol's read-only bind of the Sentinel logger (ConstantConfig.sentinel_context_path_obj)",
        ),
    ]
    if not (EnvInfo.is_windows_shell or EnvInfo.is_mac_shell):
        binds.append(("/etc/timezone", "Exegol's read-only bind of /etc/timezone (shared timezone)"))
        binds.append(("/etc/localtime", "Exegol's read-only bind of /etc/localtime (shared timezone)"))
    return [mount_dir(source, label, False, contains_only=True) for source, label in binds]


def _docker_status(info: dict) -> str:
    version = info.get("ServerVersion")
    return f"reachable (Docker {version})" if version else "reachable"


def _gui_socket_dirs() -> list[ExegolMountDir]:
    """The GUI sockets Exegol mounts read-write into a GUI container, for the startup allowlist check (WR-11).

    Mirrors ContainerConfig.enableGUI's two bind mounts, without its X11 availability check (isX11GuiAvailable is
    async and may start XQuartz on macOS). Both are listed whatever the GUI enable flags say, as this process's
    environment resolves them: a profile, the operator's Exegol config or the wrapper can enable the GUI. A socket
    path that does not exist yet is still listed: a container that mounts its parent could create it as a symlink
    before a GUI container binds it (addVolume's must_exist only checks that something is there).

    X11 is skipped when Exegol cannot share it yet (CancelOperation on Windows before Exegol's own GUI checks), when
    it is shared over the network (None, macOS), or for a WSLg UNC path (known only after those checks). Wayland is
    read only when GuiUtils.isWaylandGuiAvailable(), enableGUI's own condition, and skipped when it has no socket.
    Any other getter error propagates, so the startup check fails closed.
    """
    dirs: list[ExegolMountDir] = []
    try:
        x11 = GuiUtils.getX11SocketPath()
    except CancelOperation:
        x11 = None
    if x11 is not None and not str(x11).startswith("\\\\"):
        dirs.append(mount_dir(x11, "Exegol's X11 display socket directory (GUI sharing)", False))
    if GuiUtils.isWaylandGuiAvailable():
        wayland = GuiUtils.getWaylandSocketPath()
        if wayland is not None:
            dirs.append(mount_dir(wayland, "Exegol's Wayland display socket (GUI sharing)", False))
    return dirs


def _exegol_settings_dirs() -> list[ExegolMountDir]:
    """The directories Exegol reads the settings that decide what containers mount from (IN-14 premise, D-24, D-08).

    An allowlisted container profile is re-read at every create but checked only at startup, so no allowlisted
    workspace may reach a place Exegol reads profiles from: a container mounting it could rewrite the profile the
    next create applies. A subdirectory counts too, because Exegol scans a source with a recursive glob and the first
    file wins a duplicate name, so a planted file can shadow a profile. Exegol's config directory also holds
    config.yml, its datastore (EULA, licence session) and the Sentinel profiles.

    Listed, each built by mount_dir with holds_mount_settings=True: Exegol's config directory, its container profile
    directory (profile_component_path, where git: sources live) and every path: profile source, expanded with
    SudoInfo.expandUser and scanned in place as ContainerProfileManager.__enumerate_roots does. Every declared source
    counts, before Exegol's licence filter, so the set is a superset of what loads; a git: source adds nothing (it
    lives under the component directory) and a source spec that is not a mapping is skipped, as Exegol skips it.
    They are listed whether or not profiles.allowed is empty: the wrapper reads the same config and profiles. Any
    other error propagates (fail closed).
    """
    dirs = [
        mount_dir(
            ConstantConfig.exegol_config_path,
            "Exegol's config directory (config.yml, its datastore, Sentinel and container profiles)",
            False,
            holds_mount_settings=True,
        )
    ]
    component = OptionResolver().get(OptionKey.PROFILE_COMPONENT_PATH)
    if component:
        dirs.append(
            mount_dir(
                component,
                f"Exegol's container profile directory ({OptionKey.PROFILE_COMPONENT_PATH.value})",
                False,
                holds_mount_settings=True,
            )
        )
    sources = OptionResolver().get(OptionKey.PROFILE_SOURCES) or {}
    for key, spec in sources.items():
        if not isinstance(spec, dict) or not spec.get("path"):
            continue
        dirs.append(
            mount_dir(
                sudo.expand_user(spec["path"]),
                f"the container profile source '{key}' ({OptionKey.PROFILE_SOURCES.value})",
                False,
                holds_mount_settings=True,
            )
        )
    return dirs


def _exegol_mcp_config_dirs() -> list[ExegolMountDir]:
    """exegol-mcp's own config directory and the ones earlier versions used (T-05-50, G-05-1d, WR-09).

    exegol-mcp reads config.yml (the operator's restrictions: tool groups, allowlists, the no-auth opt-in) and the
    bearer secret from secret_store.user_config_dir() (~/.exegol-mcp, the invoking user's under sudo), and moves them
    there once from secret_store._platform_config_dir() when ~/.exegol-mcp lacks them. Under Linux sudo it also copies
    the bearer secret v1 left in root's own config directory when the invoking user has none
    (secret_store._sudo_legacy_secret_file_path). A container that mounts an allowlisted workspace reaching any of
    them could rewrite those restrictions or read the secret, or plant a config.yml or bearer.secret in an old
    directory for the next start to adopt.

    Each is built by mount_dir with holds_server_config=True and listed whether or not it exists: a missing path is
    walked lexically, never created. Only their paths are resolved, never their contents. Any error resolving them
    (UnresolvedSudoError under sudo when the invoking user is unknown, an unreadable component) propagates (fail
    closed). The paths come from secret_store so they always match where the files are read.
    """
    dirs = [
        mount_dir(
            secret_store.user_config_dir(),
            "exegol-mcp's config directory (config.yml, the bearer secret, the update cache)",
            False,
            holds_server_config=True,
        ),
        mount_dir(
            secret_store._platform_config_dir(),
            "the config directory earlier exegol-mcp versions used (its files are moved from there once)",
            False,
            holds_server_config=True,
        ),
    ]
    legacy_secret = secret_store._sudo_legacy_secret_file_path()
    if legacy_secret is not None:
        dirs.append(
            mount_dir(
                legacy_secret.parent,
                "the config directory exegol-mcp v1 used under sudo (root's; a sudo start copies its bearer secret "
                "from there)",
                False,
                holds_server_config=True,
            )
        )
    return dirs


class RealHostProbe:
    """HostProbe over the exegol 6.x SDK. Call every method through run_blocking()."""

    def assert_patch_points(self) -> None:
        """Raise PatchPointError naming the first missing SDK internal (D-15)."""
        assert_patch_points()

    def probe_docker(self) -> str:
        """Build the DockerUtils singleton, which connects to the daemon the way the wrapper does.

        With patch_exegol_sdk() applied, the SDK's own "unable to connect to docker" critical raises RuntimeError
        instead of exiting the process.
        """
        return _docker_status(DockerUtils().getDockerInfo())

    def ping_docker(self) -> str:
        """Ping the daemon with a fresh client: the DockerUtils singleton caches the startup daemon info."""
        client = docker.from_env(
            environment=EnvInfo.get_env_overlay(EnvInfo.docker_env_names),
            timeout=_PING_TIMEOUT_SECONDS,
        )
        try:
            client.ping()
            return _docker_status(client.version())
        finally:
            client.close()

    def load_config(self) -> ConfigStatus:
        """Load the Exegol config with the bundled library (a singleton: parsed once per process).

        exegol 6 never raises on an unreadable config.yml: it logs the problem, keeps its defaults and sets
        _config_refused. That flag is an SDK internal, so its absence is reported as a broken patch point (D-15)
        rather than read as "the config is fine".
        """
        config = UserConfig()
        refused = getattr(config, "_config_refused", None)
        if not isinstance(refused, bool):
            raise _broken("exegol.config.UserConfig.UserConfig._config_refused", "is missing")
        return ConfigStatus(path=str(config.getFilePath()), refused=refused)

    def exegol_config_dir(self) -> str:
        """The Exegol config directory, resolved by the library for the invoking user (sudo-aware)."""
        return str(ConstantConfig.exegol_config_path)

    def is_eula_accepted(self) -> bool:
        """Read the EULA flag from the sqlite datastore shared with the Exegol wrapper."""
        return bool(LocalDatastore().is_eula_accepted())

    def update_eula(self, accepted: bool) -> None:
        """Write the EULA flag to the datastore shared with the wrapper (LocalDatastore.update_eula).

        Only the bootstrap EULA flow calls this, after a human accepted at a terminal prompt (D-05/D-06).
        """
        LocalDatastore().update_eula(accepted)

    def validate_profiles(self, names: Sequence[str]) -> list[str]:
        """Resolve the allowlisted container profiles through Exegol's own profile manager, licence-gated (D-02).

        Reads only the cached licence session, never reloads it (adapter/profiles.py).
        """
        return profile_problems(names)

    def exegol_mount_dirs(self, profiles: Sequence[str]) -> list[ExegolMountDir]:
        """The host paths the startup allowlist check compares with every entry (WR-10, CR-03, CR-04).

        Exegol's own four directories come from its OptionResolver (no profile tier is loaded at startup, so the
        user's Exegol config decides). They are listed whatever the my-resources/exegol-resources enable flags say:
        a profile or the wrapper can still mount them. The GUI sockets follow (_gui_socket_dirs, WR-11), listed
        whatever the GUI flags say, then the directories Exegol reads its mount settings from (_exegol_settings_dirs,
        IN-14 premise), then exegol-mcp's own config directory and the one earlier versions used
        (_exegol_mcp_config_dirs, T-05-50), then the Python installation exegol-mcp and Exegol run code from, every
        directory on its import path and its user site directory even before it exists (_python_install_dirs, CR-03,
        CR-04, WR-17, IN-25, WR-18), then Exegol's read-only bind sources (_exegol_read_only_binds, CR-03), then the allowlisted
        profiles' mounts, read through Exegol's own profile models (adapter/profiles.py).
        Each entry is built by mount_paths.mount_dir and carries every host path Docker walks to mount it (CR-02): a
        path that cannot be walked raises, so the check fails closed. Call through run_blocking().
        """
        dirs: list[ExegolMountDir] = []
        for key, origin, children_mounted in _EXEGOL_MOUNT_OPTIONS:
            value = OptionResolver().get(key)
            if not value:
                continue
            dirs.append(mount_dir(value, f"{origin} ({key.value})", children_mounted))
        dirs.extend(_gui_socket_dirs())
        dirs.extend(_exegol_settings_dirs())
        dirs.extend(_exegol_mcp_config_dirs())
        dirs.extend(_python_install_dirs())
        dirs.extend(_exegol_read_only_binds())
        dirs.extend(profile_mount_dirs(list(profiles)))
        return dirs
