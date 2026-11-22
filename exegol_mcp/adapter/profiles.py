"""Container profiles through Exegol's own profile system (D-01, D-02, CONF-01, LIFE-02, Pitfall 7).

A profile is an Exegol 6 ContainerProfile file, resolved by Exegol's ContainerProfileManager exactly as
`exegol start --profile` resolves it. exegol-mcp keeps no profile format of its own: config.yml only names the
profiles agents may use (profiles.allowed).

Exegol restricts container profiles to Pro, Team and Enterprise licences, and that gate lives in ExegolManager,
which exegol-mcp bypasses. So this module checks SessionHandler().pro_feature_access() itself, at startup and on
every create-time load, and loads no profile without it. It reads only the session SessionHandler fast-loads from
the local datastore and never reloads it: a reload may contact the network and show interactive licence messages.

Exegol never raises on a missing or invalid profile: it logs an error and returns None. To name the reason, a
temporary handler on the Exegol logger collects the ERROR records the calling thread emits during the load and the
lookup, as plain text.

This module also reports the host paths the allowlisted profiles bind into containers (profile_mount_dirs), for the
startup workspace allowlist check (WR-10): their read-write volumes.mounts entries, their workspace path and their
Exegol path overrides, read from Exegol's own ContainerProfile models. Their ":ro" mounts, VPN config and auth file
are reported as contains_only entries (CR-03): read-only stops writes through the bind, not the swap of a path Docker
walks, so an entry containing one is refused while an entry inside one loads. Their system.devices host paths are
reported as children-mounted entries (WR-14): Docker maps the device nodes under them read-write, so an entry inside,
containing or equal to one is refused. A leading ~ expands through Exegol's own SudoInfo.expandUser (IN-12), and each
path carries the host paths Docker walks to bind it (CR-02).

This is the only module that imports exegol.profile.ContainerProfileManager or exegol.utils.SessionHandler.
Every function here is synchronous and blocking (file reads, sqlite): call it through run_blocking().
"""

import logging
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from exegol.profile.ContainerProfileManager import ContainerProfileManager
from exegol.utils.ExeLog import logger as _exegol_logger
from exegol.utils.SessionHandler import SessionHandler
from rich.errors import MarkupError
from rich.text import Text

from exegol_mcp.adapter import sudo
from exegol_mcp.adapter.mount_paths import mount_dir
from exegol_mcp.adapter.port import PROFILE_LICENCE_MESSAGE, ExegolMountDir

# ExeLog prefixes every error with a level marker ("[-] ", "[!] ", "[x] "); it carries no reason.
_LEVEL_MARKER = re.compile(r"^\[[-!x+*]\]\s*")

# A mirror of the volume syntax Exegol's ContainerConfig.addRawVolume parses a profile or CLI volume with, pinned by a
# drift test (tests/test_profiles.py): group 1 is the host path and group 7 the mode (None means read-write).
_EXEGOL_RAW_VOLUME = re.compile(r"^((\w:|\.|~)?([\\/][\w .,:\-|()&;]*)+):(([\\/][\w .,\-|()&;]*)+)(:(ro|rw))?$")


def _plain(message: str) -> str:
    """Exegol's rich markup turned to plain text, without its level marker."""
    try:
        text = Text.from_markup(message).plain
    except MarkupError:
        text = message
    return _LEVEL_MARKER.sub("", text.strip())


class _ErrorCollector(logging.Handler):
    """Collect the ERROR-and-above records one thread logs on the Exegol logger, as plain text."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.thread = threading.get_ident()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        # Another thread's SDK work (a concurrent tool call) is not this load's reason.
        if record.thread != self.thread:
            return
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # a record whose args do not match its format string
            message = str(record.msg)
        self.messages.append(_plain(message))


@contextmanager
def _collect_exegol_errors() -> Iterator[list[str]]:
    """Collect the errors the Exegol logger records in this thread for the enclosed block; detached in finally."""
    collector = _ErrorCollector()
    _exegol_logger.addHandler(collector)
    try:
        yield collector.messages
    finally:
        _exegol_logger.removeHandler(collector)


def _has_pro_access() -> bool:
    """Whether the cached Exegol licence session grants profiles (Pitfall 7).

    SessionHandler only fast-loads the session from the local datastore when it is built; this never calls
    reload_session(), which may contact the network and show interactive licence messages (Open Question 3).
    """
    return bool(SessionHandler().pro_feature_access())


def _mentions(message: str, selection: str) -> bool:
    """Whether a load-time error is about the file behind selection (`<source>.<name>` or a bare `<name>`)."""
    name = selection.partition(".")[2] if "." in selection else selection
    return re.search(rf"(?:^|[\\/\s'\"]){re.escape(name)}(?:\.ya?ml\b|')", message) is not None


def _reason(selection: str, load_errors: list[str], lookup_errors: list[str]) -> str:
    """What Exegol logged about selection: its file's load errors, then the lookup error, or 'not found'."""
    reasons = [message for message in load_errors if _mentions(message, selection)] + lookup_errors
    return "; ".join(reasons) or "not found"


def profile_problems(selections: Sequence[str]) -> list[str]:
    """Resolve every allowlisted profile through Exegol, for the startup readiness gate (D-02).

    Args:
        selections: The profile names from config.yml profiles.allowed, as `<source>.<name>` or a bare name.

    Returns:
        One "Profile '<name>': <reason>" entry per selection Exegol cannot resolve or validate, in the allowlist's
        order, or the licence problem alone. An empty list means ready; an empty allowlist does no work at all.
    """
    if not selections:
        return []
    if not _has_pro_access():
        return [PROFILE_LICENCE_MESSAGE]
    with _collect_exegol_errors() as load_errors:
        manager = ContainerProfileManager()
        manager.load_profiles()
    problems: list[str] = []
    for selection in selections:
        with _collect_exegol_errors() as lookup_errors:
            profile = manager.get_profile(selection)
        if profile is None:
            problems.append(f"Profile '{selection}': {_reason(selection, load_errors, lookup_errors)}")
    return problems


def _profile_mount_dir(
    raw: str | None, label: str, children_mounted: bool, *, contains_only: bool = False
) -> ExegolMountDir | None:
    """A profile-declared host path as Exegol mounts it, walked as Docker will resolve it (CR-02, IN-12).

    The path is expanded with Exegol's own SudoInfo.expandUser (sudo.expand_user), as addRawVolume, setWorkspaceShare,
    the profile path overrides and enableVPN do: ~ is the invoking user's home under sudo (D-08) and ~user expands
    too. It is then built by mount_paths.mount_dir, which records every host path Docker's bind resolution walks for
    it; contains_only marks a read-only bind (CR-03). None for an empty or missing value.
    """
    if not raw:
        return None
    return mount_dir(sudo.expand_user(raw), label, children_mounted, contains_only=contains_only)


def profile_mount_dirs(selections: Sequence[str]) -> list[ExegolMountDir]:
    """The host paths the allowlisted profiles bind into containers, for the startup allowlist check (WR-10, CR-03).

    Each profile is read through Exegol's own profile manager and models. Collected as read-write mounts: every
    volumes.mounts entry that Exegol's volume syntax accepts with mode rw or none, volumes.workspace_path,
    volumes.exegol_resources_path and customization.my_resources_path (mounted whole), and
    volumes.private_workspace_path and sentinel.sentinel_logs_host_path (their subdirectories are mounted). Collected
    as contains_only entries (CR-03), because read-only stops writes through the bind but not the swap of a path
    Docker walks: every ":ro" volumes.mounts entry, vpn.config and vpn.auth_file (bound read-only by enableVPN). An
    entry inside one of them loads; one containing a walked path is refused. Collected as children-mounted entries
    (WR-14): the host path of each system.devices entry, whose device nodes Docker maps read-write, so an entry
    inside, containing or equal to one is refused. A mount Exegol would refuse at create is skipped. Sections are
    read by attribute, so a renamed Exegol field raises rather than passing silently. A leading ~ expands as Exegol's
    SudoInfo.expandUser expands it (not for a device, which Exegol hands Docker as written), and each entry carries
    every host path Docker's bind resolution walks for it (_profile_mount_dir, CR-02, IN-12): a path that cannot be
    walked raises.

    Args:
        selections: The profile names from config.yml profiles.allowed.

    Returns:
        The directories, profile by profile; [] for no selections or without a pro licence (the readiness profiles
        step already refused that case). A selection Exegol cannot resolve contributes nothing: the profiles step
        already named it.
    """
    if not selections or not _has_pro_access():
        return []
    with _collect_exegol_errors():
        manager = ContainerProfileManager()
        manager.load_profiles()
    dirs: list[ExegolMountDir] = []

    def _add(raw: str | None, label: str, children_mounted: bool, *, contains_only: bool = False) -> None:
        mount = _profile_mount_dir(raw, label, children_mounted, contains_only=contains_only)
        if mount is not None:
            dirs.append(mount)

    for selection in selections:
        with _collect_exegol_errors():
            profile = manager.get_profile(selection)
        if profile is None:
            continue
        of = f"of profile '{selection}'"
        volumes = profile.volumes
        if volumes is not None:
            for raw in volumes.mounts or []:
                parsed = _EXEGOL_RAW_VOLUME.match(raw)
                if parsed is None:  # Exegol refuses it at create
                    continue
                read_only = parsed.group(7) == "ro"
                _add(parsed.group(1), f"the volumes.mounts entry '{raw}' {of}", False, contains_only=read_only)
            _add(volumes.workspace_path, f"the volumes.workspace_path {of}", False)
            _add(volumes.exegol_resources_path, f"the volumes.exegol_resources_path {of}", False)
            _add(volumes.private_workspace_path, f"the volumes.private_workspace_path {of}", True)
        customization = profile.customization
        if customization is not None:
            _add(customization.my_resources_path, f"the customization.my_resources_path {of}", False)
        sentinel = profile.sentinel
        if sentinel is not None:
            _add(sentinel.sentinel_logs_host_path, f"the sentinel.sentinel_logs_host_path {of}", True)
        # ContainerConfig.enableVPN binds both read-only after SudoInfo.expandUser.
        vpn = profile.vpn
        if vpn is not None:
            _add(vpn.config, f"the vpn.config {of}", False, contains_only=True)
            _add(vpn.auth_file, f"the vpn.auth_file {of}", False, contains_only=True)
        # Exegol maps a device read-write (ContainerConfig.__addDevice defaults readonly to False). For a directory
        # device, Docker maps every device node found under it at each create, and containers keep CAP_MKNOD, so an
        # entry inside, equal to or containing it could plant a node the next create maps: it is refused like a
        # children-mounted directory (WR-14). Exegol hands the path to Docker as written (addUserDevice), and Docker
        # resolves its symlinks.
        system = profile.system
        if system is not None:
            for raw in system.devices or []:
                host_path = raw.split(":", 1)[0]
                if host_path:
                    label = f"the system.devices entry '{raw}' {of}"
                    dirs.append(mount_dir(host_path, label, True))
    return dirs


def load_allowed_profile(selection: str) -> Any:
    """Load one profile through Exegol for a create, licence-gated and read fresh (LIFE-02, Pitfall 7).

    The profile files are re-read on every call and nothing is cached between calls, so a profile edited or broken
    after startup is seen as it is now. This never checks the allowlist (the caller does). ContainerProfileManager
    is a process-wide singleton whose load resets its profiles: callers serialize creates (the create lock).

    Args:
        selection: The profile name as `<source>.<name>` or a bare name.

    Returns:
        Exegol's ContainerProfile, ready for exegol.profile.ProfileApplication.apply_profile.

    Raises:
        RuntimeError: The cached Exegol licence session does not grant profiles (PROFILE_LICENCE_MESSAGE); no
            profile file is read.
        ValueError: Exegol cannot resolve or validate the profile, naming it and the reason Exegol logged.
    """
    if not _has_pro_access():
        raise RuntimeError(PROFILE_LICENCE_MESSAGE)
    with _collect_exegol_errors() as load_errors:
        manager = ContainerProfileManager()
        manager.load_profiles()
    with _collect_exegol_errors() as lookup_errors:
        profile = manager.get_profile(selection)
    if profile is None:
        raise ValueError(
            f"Container profile '{selection}' cannot be used: {_reason(selection, load_errors, lookup_errors)}"
        )
    return profile
