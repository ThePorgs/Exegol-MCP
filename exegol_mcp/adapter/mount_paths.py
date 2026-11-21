"""The host paths Docker's bind resolution walks for a directory Exegol mounts (CR-02). SDK-free: stdlib only.

Exegol hands Docker the lexical bind source (ContainerConfig.addVolume passes Path(host_path).absolute(), '..' not
collapsed) and Docker resolves its symlinks when the container is created, not when exegol-mcp starts. The realpath
of a mount directory alone therefore hides the directory entries an attacker can swap: a container that mounts an
allowlisted directory read-write controls every entry inside it, so a mount path that goes through a symlink inside
that directory can be redirected between two creates. walked_paths resolves a path component by component, the way
the kernel resolves a bind source, and records every entry it looks up; mount_dir builds the ExegolMountDir the
startup workspace-allowlist check compares with every allowlist entry (bootstrap/readiness.py).
"""

import errno
import os
import stat
from collections import deque
from pathlib import Path

from exegol_mcp.adapter.port import ExegolMountDir

# Linux MAXSYMLINKS: the kernel gives up with ELOOP after following this many symlinks in one resolution.
MAX_SYMLINK_FOLLOWS = 40


def walked_paths(path: str) -> list[str]:
    """Return every directory entry the resolution of ``path`` looks up, in walk order, without duplicates.

    The walk starts at the path's anchor. "." is skipped and ".." moves to the parent of the directory reached so far.
    Any other component is looked up in that directory: the joined entry is recorded, then read with os.lstat. A
    symlink's target components are spliced in front of the remaining ones (an absolute target restarts at its
    anchor, a relative one continues from the directory reached so far). When a component does not exist, or is not
    a directory, the remaining components are recorded lexically under it, since a container could create them.

    Args:
        path: The host path to resolve (made absolute first, '..' kept).

    Returns:
        The looked-up entries, as strings.

    Raises:
        OSError: errno ELOOP after more than MAX_SYMLINK_FOLLOWS symlinks, or any error reading an entry other than
            a missing component (a permission error, for example): the caller fails closed.
    """
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    remaining: deque[str] = deque(absolute.parts[1:])
    walked: list[str] = []
    seen: set[str] = set()

    def _record(entry: Path) -> None:
        text = str(entry)
        if text not in seen:
            seen.add(text)
            walked.append(text)

    follows = 0
    while remaining:
        name = remaining.popleft()
        if name in ("", "."):
            continue
        if name == "..":
            current = current.parent
            continue
        entry = current / name
        _record(entry)
        try:
            info = os.lstat(entry)
        except (FileNotFoundError, NotADirectoryError):
            # Nothing exists there yet: a container that can write the parent could create the rest, so the rest
            # of the path is recorded as written.
            for rest in remaining:
                if rest in ("", "."):
                    continue
                entry = entry.parent if rest == ".." else entry / rest
                _record(entry)
            return walked
        if stat.S_ISLNK(info.st_mode):
            follows += 1
            if follows > MAX_SYMLINK_FOLLOWS:
                raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), path)
            target = Path(os.readlink(entry))
            if target.is_absolute():
                current = Path(target.anchor)
                remaining.extendleft(reversed(target.parts[1:]))
            else:
                remaining.extendleft(reversed(target.parts))
            continue
        current = entry
    return walked


def mount_dir(
    host_path: str | os.PathLike[str],
    label: str,
    children_mounted: bool,
    *,
    holds_mount_settings: bool = False,
    holds_server_config: bool = False,
    holds_host_code: bool = False,
    contains_only: bool = False,
) -> ExegolMountDir:
    """Build the ExegolMountDir for a host path Exegol mounts, reads its mount settings from or runs code from (CR-02).

    Every production ExegolMountDir is built here, so each one carries the paths Docker's bind resolution walks. That
    includes the sources of read-only binds, which may be files rather than directories (CR-03). Only
    host._gui_socket_dirs skips a Windows UNC path ("\\\\host\\share"). Any other caller passes one through, and on a
    native Windows host the walk then reads it over the network at startup, failing closed on an error (IN-16,
    accepted: native Windows is not a tested platform).

    Args:
        host_path: The path as Exegol hands it to Docker; the caller expands a leading ~ first.
        label: Where the directory comes from, readable by an operator.
        children_mounted: True when Exegol hands its entries to containers one by one rather than itself: its
            subdirectories (one per container), or the device nodes of a profile device directory (WR-14).
        holds_mount_settings: True for a directory Exegol reads the settings that decide what containers mount from
            (its config directory, a container profile source) rather than one it mounts.
        holds_server_config: True for exegol-mcp's own config directory or the one earlier versions used, which it
            reads config.yml and the bearer secret from: refused inside, containing or equal, like
            holds_mount_settings.
        holds_host_code: True for a directory exegol-mcp or Exegol runs code from on this host (the Python
            installation and every directory on its import path, CR-03, CR-04): refused inside, containing or
            equal, like holds_mount_settings.
        contains_only: True for a path Exegol or an allowlisted profile binds read-only: an entry containing any
            path Docker walks for it is refused, an entry inside it loads (CR-03).

    Returns:
        The entry: path is the realpath, lexical the absolute normalized path Exegol hands Docker, candidates the
        lexical path, then every entry the walk looks up, then the realpath of each prefix of the lexical path.

    Raises:
        OSError: From walked_paths (a symlink loop or an unreadable component): the startup check fails closed.
    """
    raw = os.fspath(Path(host_path).absolute())
    lexical = os.path.abspath(raw)
    candidates: list[str] = []

    def _add(path: str) -> None:
        if path not in candidates:
            candidates.append(path)

    _add(lexical)
    for walked in walked_paths(raw):
        _add(walked)
    if lexical != raw:
        for walked in walked_paths(lexical):
            _add(walked)
    lexical_path = Path(lexical)
    for prefix in (*reversed(lexical_path.parents), lexical_path):
        _add(os.path.realpath(prefix))
    return ExegolMountDir(
        path=os.path.realpath(lexical),
        label=label,
        children_mounted=children_mounted,
        lexical=lexical,
        candidates=tuple(candidates),
        holds_mount_settings=holds_mount_settings,
        holds_server_config=holds_server_config,
        holds_host_code=holds_host_code,
        contains_only=contains_only,
    )
