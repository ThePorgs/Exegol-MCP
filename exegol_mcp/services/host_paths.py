"""The host-path rule for directories an agent may mount (D-08, D-24). SDK-free: stdlib only.

expand_host_path mirrors Exegol's SudoInfo.expandUser: only a bare ``~`` or a ``~/`` prefix expands, against the
home the caller passes (the invoking user's home under sudo, never root's). dangerous_host_path_reason re-implements
Exegol's own profile-path refusal (ProfileUserConfigTier._resolve_host_path): the filesystem root, a home directory
and any ancestor of a home are never mountable, because Exegol mounts the path read-write and may apply a recursive
setgid chmod to it.

path_parts and is_strictly_inside are the nesting helpers every allowlist comparison uses. They compare whole path
components, so /ab is never inside /a and a path is never strictly inside itself. On macOS and Windows they fold
letter case (IN-08): those hosts treat names that differ only by case as the same directory, and os.path.realpath
keeps the caller's spelling, so a case-sensitive comparison would miss a nested pair or a home directory there.
"""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

HOST_PATH_MAX_LEN = 4096  # PATH_MAX on Linux; a longer value is malformed, not a path to resolve

# macOS (APFS default) and Windows compare names case-insensitively, and os.path.realpath does not canonicalize case
# there, so path comparisons fold case (IN-08). A case-sensitive APFS volume is treated as insensitive too: that only
# refuses more, never less. Read at call time, so tests can force either host rule.
CASE_INSENSITIVE_HOST: bool = sys.platform in ("darwin", "win32")


def path_parts(path: Path, *, fold_case: bool | None = None) -> tuple[str, ...]:
    """Return the components of ``path``, each case-folded when folding.

    Args:
        path: The path to split.
        fold_case: Fold letter case with str.casefold. None means CASE_INSENSITIVE_HOST, read at call time.
    """
    if fold_case is None:
        fold_case = CASE_INSENSITIVE_HOST
    if fold_case:
        return tuple(part.casefold() for part in path.parts)
    return path.parts


def is_strictly_inside(inner: Path, outer: Path, *, fold_case: bool | None = None) -> bool:
    """Return True when ``inner`` lies below ``outer``: outer's components are a proper prefix of inner's.

    Whole components are compared, so /ab is not inside /a, and a path is never strictly inside itself.

    Args:
        inner: The candidate nested path (absolute, already resolved).
        outer: The candidate enclosing path (absolute, already resolved).
        fold_case: As in path_parts; None follows CASE_INSENSITIVE_HOST.
    """
    inner_parts = path_parts(inner, fold_case=fold_case)
    outer_parts = path_parts(outer, fold_case=fold_case)
    return len(outer_parts) < len(inner_parts) and inner_parts[: len(outer_parts)] == outer_parts


def expand_host_path(raw: str, home: Path) -> Path:
    """Expand a leading ``~`` against ``home`` and resolve the result to an absolute realpath.

    Args:
        raw: The operator- or agent-supplied host path.
        home: The home a leading ``~`` expands to (the invoking user's home under sudo, D-08).

    Returns:
        The absolute, symlink-resolved path.

    Raises:
        ValueError: The path is empty, contains a NUL, is longer than HOST_PATH_MAX_LEN, uses the ``~user`` form, or
            is not absolute after expansion.
    """
    if not raw:
        raise ValueError("an empty path")
    if "\x00" in raw:
        raise ValueError("the path contains a NUL character")
    if len(raw) > HOST_PATH_MAX_LEN:
        raise ValueError(f"the path is longer than {HOST_PATH_MAX_LEN} characters")
    if raw == "~" or raw.startswith("~/"):
        text = str(home).rstrip("/") + raw[1:]
        text = text or "/"
    elif raw.startswith("~"):
        raise ValueError("the ~user form is not supported; use ~ or an absolute path")
    else:
        text = raw
    if not os.path.isabs(text):
        raise ValueError("the path must be absolute (or start with ~/)")
    return Path(os.path.realpath(text))


def dangerous_host_path_reason(
    path: Path,
    homes: Sequence[Path],
    *,
    fold_case: bool | None = None,
) -> str | None:
    """Return why ``path`` must never be mounted, or None when Exegol's profile-path rule allows it.

    Args:
        path: An absolute, already-resolved host path.
        homes: The home directories to protect (the invoking user's and the process's, which differ under sudo).
            Each is compared both as given and as its realpath.
        fold_case: Compare with letter case folded (IN-08). None follows CASE_INSENSITIVE_HOST, so a case-differing
            home, or an ancestor of one, is still refused on macOS and Windows.

    Returns:
        A short reason when ``path`` is the filesystem root, a home directory or an ancestor of one, else None.
    """
    if path.parent == path:
        return "the filesystem root is never mountable"
    parts = path_parts(path, fold_case=fold_case)
    for home in homes:
        for candidate in {home, Path(os.path.realpath(home))}:
            if parts == path_parts(candidate, fold_case=fold_case):
                return "a home directory is never mountable"
            if is_strictly_inside(candidate, path, fold_case=fold_case):
                return "an ancestor of a home directory is never mountable"
    return None
