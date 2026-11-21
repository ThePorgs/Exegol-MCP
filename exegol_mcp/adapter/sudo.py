"""Invoking-user resolution under sudo (D-17, D-18), reusing exegol's own SudoInfo.

Under `sudo exegol-mcp` the process runs as root, but exegol-mcp's own files (the bearer secret) must belong to
the user who ran sudo, so sudo and non-sudo runs share them and existing HTTP client configs keep working.
exegol.config.SudoInfo already implements Exegol 6's rules for this (SUDO_UID and SUDO_HOME, passwd fallback,
resolved on Linux only) with the standard library alone. is_unresolved_sudo() reports sudo on a platform where
Exegol does not resolve the invoking user (macOS), where exegol-mcp refuses its own file store as Exegol refuses
EXEGOL_HOME. This module is the only importer of SudoInfo: bootstrap code gets plain values from here and never
imports exegol itself (D-19).

SudoInfo is imported inside each function, not at module level. Importing exegol_mcp.main therefore loads no exegol
module, and a renamed SudoInfo is named by the D-15 self-check main() runs first, as one stderr line (WR-13).
"""

import os
import sys
from pathlib import Path


def invoking_user_home() -> Path:
    """Home directory of the user running exegol-mcp: the invoking user's home under sudo."""
    from exegol.config.SudoInfo import SudoInfo

    return SudoInfo.getUserHome()


def expand_user(path: str) -> str:
    """Expand a leading ~ in path exactly as Exegol does for the profile and config paths it mounts (IN-12).

    This is Exegol's own SudoInfo.expandUser: under sudo, ~ and ~/ become the invoking user's home (never root's,
    D-08); otherwise Path.expanduser, so ~user also expands to that user's home. Nothing else is resolved.
    """
    from exegol.config.SudoInfo import SudoInfo

    return os.fspath(SudoInfo.expandUser(path))


def invoking_user_uid() -> int | None:
    """uid of the user running exegol-mcp (the invoking user's under sudo), or None on Windows."""
    from exegol.config.SudoInfo import SudoInfo

    return SudoInfo.getUserUid()


def invoking_user_gid() -> int | None:
    """Primary gid of the invoking user, or None on Windows or when the passwd database has no entry."""
    uid = invoking_user_uid()
    if uid is None or sys.platform == "win32":
        return None
    import pwd  # POSIX only

    try:
        return pwd.getpwuid(uid).pw_gid
    except KeyError:
        return None


def invoking_user_env(name: str) -> str | None:
    """An environment variable as the invoking user's shell had it under sudo (os.environ otherwise)."""
    from exegol.config.SudoInfo import SudoInfo

    return SudoInfo.getEnv(name)


def is_sudo_context() -> bool:
    """True when exegol-mcp runs as root through sudo on Linux (files it creates must be handed back)."""
    from exegol.config.SudoInfo import SudoInfo

    return bool(SudoInfo.isSudoContext())


def is_unresolved_sudo() -> bool:
    """True when exegol-mcp runs as root through sudo on a platform where Exegol does not resolve the invoking user.

    That is every POSIX platform but Linux (e.g. macOS). exegol-mcp's own files then cannot be attributed to that user.
    """
    from exegol.config.SudoInfo import SudoInfo

    return bool(SudoInfo.isUnresolvedSudo())
