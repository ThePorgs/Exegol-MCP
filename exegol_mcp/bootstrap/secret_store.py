"""Bearer secret storage for HTTP mode, ported from v1's SecretManager (COMP-01) with a sudo-aware location (D-18).

The OS keyring (service `exegol-mcp`, user `bearer`) on Windows/macOS when available holds the secret where v1 kept
it. Otherwise it is a 0600 file in exegol-mcp's config dir, `~/.exegol-mcp` on every OS next to the wrapper's
`~/.exegol` (G-05-1d), alongside config.yml and the update cache. The dir follows the invoking user the way Exegol 6
resolves it (through adapter/sudo.py): v1 resolved `~`, which is root's home under `sudo exegol-mcp`. Files created
there under sudo are handed back to that user, so sudo and non-sudo runs share one secret.

Earlier versions used the platform dir (`~/Library/Application Support/exegol-mcp`, `$XDG_CONFIG_HOME/exegol-mcp`
or `~/.config/exegol-mcp`, `%APPDATA%\\exegol-mcp`). The first read of config.yml, bearer.secret or
update-check.json that finds no file in `~/.exegol-mcp` moves the old one there, once: it is copied through the
hardened reader and a no-replace writer, then removed only when it is still the inode that was read. A file already
in `~/.exegol-mcp` always wins, so existing v1 secrets keep authenticating and nothing is ever overwritten. A file
the move cannot take safely is left in place: config.yml then stops startup (running on built-in defaults would drop
the operator's restrictions), a secret is minted afresh, and the update check fetches again.

Under sudo the store therefore acts as root inside a directory the invoking user controls (CR-01). On POSIX every
operation is relative to a directory descriptor opened without following a symbolic link, every file is opened
with O_NOFOLLOW, writes go to a freshly created (O_EXCL) temporary file that replaces the destination by name, so
an existing file is never truncated in place, and ownership is handed back with fchown only for the directories
and files the call itself created. When running as root, whether or not sudo's invoker is resolved, the config dir
and any file read from it must belong to the expected owner (the invoking user under Linux sudo, root otherwise),
and a file must have a single hard link; anything else is refused (UnsafeConfigPathError).

Under sudo on a platform where Exegol does not resolve the invoking user (every POSIX platform but Linux, e.g.
macOS), the file store and the update cache are refused outright (UnresolvedSudoError), as Exegol refuses
EXEGOL_HOME there: root never works in a config dir it cannot attribute to anyone, old or new.

There is no process-wide cache: the lifecycle loads the secret once and injects it where it is needed.

The at-rest story is honest (SEC-06, D-15): the 0600 file holds the raw `token_urlsafe` bytes and the OS keyring
holds the raw secret; there is no XOR "obfuscation" on the store path (it was security theater, never encryption —
the real controls are the file mode / OS keyring and the hardened descriptor-relative I/O). A file written by a
v1/Phase-1 server (single-byte XOR of the token) is transparently migrated on load: the read path decodes it and
rewrites the file raw in place (atomically), so existing HTTP client configs keep authenticating with no operator
action (D-16, COMP-01). The legacy XOR transform survives only as a clearly-named read-side decode helper used
solely for that migration, and is deletable in a later version.
"""

import contextlib
import errno
import logging
import os
import platform
import re
import secrets
import stat
import sys
from pathlib import Path

from exegol_mcp.adapter.sudo import (
    invoking_user_env,
    invoking_user_gid,
    invoking_user_home,
    invoking_user_uid,
    is_sudo_context,
    is_unresolved_sudo,
)

try:
    import keyring  # type: ignore
except Exception:  # noqa: BLE001  # pragma: no cover - optional dependency; any backend init error counts
    keyring = None  # type: ignore[assignment]  # fall back to file backend

APP_DIR_NAME = "exegol-mcp"
CONFIG_DIR_NAME = ".exegol-mcp"
SECRET_FILE_NAME = "bearer.secret"
KEYRING_SERVICE = "exegol-mcp"
KEYRING_USERNAME = "bearer"
# Files moved once from the old platform dir (G-05-1d): config.CONFIG_FILE_NAME, SECRET_FILE_NAME and
# versions.UPDATE_CACHE_FILE, named by value because both modules import this one.
ADOPTED_FILE_NAMES = ("config.yml", SECRET_FILE_NAME, "update-check.json")

_log = logging.getLogger(__name__)

# errno values an O_NOFOLLOW open gives for a symbolic link (ELOOP on Linux/macOS, EMLINK on FreeBSD), plus
# ENOTDIR when O_DIRECTORY meets something that is not a directory.
_REFUSED_ERRNOS = frozenset({errno.ELOOP, errno.EMLINK, errno.ENOTDIR})


class UnsafeConfigPathError(OSError):
    """A config-dir path is unsafe: a symbolic link, not a regular file or dir, foreign-owned or hard-linked."""


class UnresolvedSudoError(UnsafeConfigPathError):
    """exegol-mcp runs as root through sudo, but the sudo invoking user cannot be resolved on this platform."""


# --------------- Path helpers ---------------
def user_config_dir() -> Path:
    """exegol-mcp's config dir: ~/.exegol-mcp on every OS, like the wrapper's ~/.exegol (G-05-1d).

    The home is the invoking user's under sudo (D-18); XDG_CONFIG_HOME and APPDATA do not move it.
    """
    return invoking_user_home() / CONFIG_DIR_NAME


def _platform_config_dir() -> Path:
    """The config dir used before G-05-1d (the platform dir), read only as the source of the one-time move."""
    home = invoking_user_home()
    system = platform.system().lower()
    if system == "windows":
        base = invoking_user_env("APPDATA") or str(home / "AppData" / "Roaming")
    elif system == "darwin":  # macOS
        base = str(home / "Library" / "Application Support")
    else:  # Linux/Unix
        base = invoking_user_env("XDG_CONFIG_HOME") or str(home / ".config")
    return Path(base) / APP_DIR_NAME


def secret_file_path() -> Path:
    """Return the path where the bearer secret is stored on this OS."""
    return user_config_dir() / SECRET_FILE_NAME


def _legacy_secret_file_path() -> Path:
    """Where v1 put the secret: resolved from this process's own environment, i.e. root's home under sudo."""
    system = platform.system().lower()
    if system == "windows":
        base = os.getenv("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
    elif system == "darwin":  # macOS
        base = os.path.expanduser("~/Library/Application Support")
    else:  # Linux/Unix
        base = os.getenv("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / APP_DIR_NAME / SECRET_FILE_NAME


def _sudo_legacy_secret_file_path() -> Path | None:
    """Root's v1 secret that a Linux sudo start copies when the invoking user has none, or None outside sudo.

    The one place that decides when the copy can happen, so the startup allowlist check guards the same directory.
    """
    return _legacy_secret_file_path() if is_sudo_context() else None


def _check_plain_name(name: str) -> None:
    if not name or Path(name).name != name:
        raise ValueError(f"Not a plain file name: {name!r}")


# --------------- Ownership (D-18) ---------------
def _effective_uid() -> int:
    """This process's effective uid (root under sudo). A seam the tests patch to simulate a sudo run."""
    geteuid = getattr(os, "geteuid", None)  # POSIX only
    return geteuid() if geteuid is not None else -1


def _owner_uid() -> int:
    """uid that should own exegol-mcp's config files: the invoking user's under sudo, else this process's."""
    uid = invoking_user_uid()
    return uid if uid is not None else _effective_uid()


def _acting_for_other_user() -> bool:
    """True when the files belong to someone other than this process (root acting for the user under sudo)."""
    return os.name == "posix" and _owner_uid() != _effective_uid()


def _privileged() -> bool:
    """True when running as root (or acting for another user): config paths must then pass the ownership checks.

    Root never trusts a config path just because its owner equals its own uid. With no resolved invoker (Linux root
    without SUDO_UID: su -m, doas keepenv), the expected owner is root itself, so a user-owned config dir is refused.
    """
    return os.name == "posix" and (_effective_uid() == 0 or _acting_for_other_user())


def _legacy_owner_uid() -> int:
    """uid that owns root's own v1 legacy secret dir: a seam, as unprivileged tests cannot create root-owned files."""
    return _effective_uid()


def _hand_back(fd: int, path: Path) -> None:
    """fchown a directory or file this call just created to the invoking user when acting for them (D-18).

    Only ever called on a descriptor of something the call created, never on a path. Best-effort: a failure is
    logged and the file still works for this run (the ownership checks refuse it later if it stayed foreign).
    """
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    if not _acting_for_other_user():
        return
    uid = _owner_uid()
    gid = invoking_user_gid()
    try:
        os.fchown(fd, uid, gid if gid is not None else -1)
    except OSError as e:
        _log.warning(f"Could not hand '{path}' back to uid {uid}: {e}")


# --------------- No-follow, descriptor-relative file helpers (POSIX, CR-01) ---------------
def _open_dir_nofollow(name: str | Path, dir_fd: int | None = None) -> int:
    """Open a directory without following a symbolic link at its final component."""
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as e:
        if e.errno in _REFUSED_ERRNOS:
            raise UnsafeConfigPathError(e.errno, "symbolic link or not a directory", str(name)) from None
        raise


def _create_config_dir(directory: Path) -> int:
    """Create the missing components of directory (0700) and return a descriptor of the final one.

    The deepest existing ancestor is opened following symbolic links (a symlinked home is legitimate); every
    component below it is created by this call relative to its parent's descriptor, opened without following a
    link, and handed back to the invoking user through its own descriptor.
    """
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    missing: list[str] = []
    ancestor = directory
    while not os.path.lexists(ancestor):
        if ancestor.parent == ancestor:
            raise FileNotFoundError(errno.ENOENT, "no existing ancestor", str(directory))
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    owner = _owner_uid()
    parent_fd = os.open(ancestor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if _privileged() and os.fstat(parent_fd).st_uid != owner:
            raise UnsafeConfigPathError(errno.EPERM, "not owned by the invoking user", str(ancestor))
        current = ancestor
        for name in reversed(missing):
            current = current / name
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:  # created concurrently: treated as an existing entry
                created = False
            child_fd = _open_dir_nofollow(name, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
            if created:
                _hand_back(child_fd, current)
            elif _privileged() and os.fstat(child_fd).st_uid != owner:
                raise UnsafeConfigPathError(errno.EPERM, "not owned by the invoking user", str(current))
        return parent_fd
    except BaseException:
        os.close(parent_fd)
        raise


def _open_config_dir(
    directory: Path, create: bool, owner_uid: int | None = None, harden: bool = True
) -> int | None:
    """Open exegol-mcp's config dir as an O_DIRECTORY descriptor, never following a link at its final component.

    Args:
        directory: The config dir.
        create: Create it (and missing parents, 0700) when it does not exist.
        owner_uid: Who must own it; defaults to the invoking user. Root's own legacy dir passes the effective uid.
        harden: Tighten its mode to 0700. False for the old platform dir a move reads from: it is left as found.

    Returns:
        The descriptor (the caller closes it), or None when the dir does not exist and create is False.

    Raises:
        UnresolvedSudoError: Running under sudo on a platform where the invoking user is not resolved; nothing in
            the config dir is opened or created.
        UnsafeConfigPathError: The dir is a symbolic link, not a directory, or (when running as root or acting for
            another user) not owned by the expected owner.
    """
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    if is_unresolved_sudo():  # before any open, mkdir or fstat: root must not work in a dir it cannot attribute
        raise UnresolvedSudoError(
            errno.EPERM, "the sudo invoking user cannot be resolved on this platform", str(directory)
        )
    owner = _owner_uid() if owner_uid is None else owner_uid
    try:
        fd = _open_dir_nofollow(directory)
    except FileNotFoundError:
        if not create:
            return None
        fd = _create_config_dir(directory)
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):
            raise UnsafeConfigPathError(errno.ENOTDIR, "not a directory", str(directory))
        if _privileged() and st.st_uid != owner:
            raise UnsafeConfigPathError(errno.EPERM, "not owned by the invoking user", str(directory))
        if harden and st.st_uid == owner and stat.S_IMODE(st.st_mode) != 0o700:  # never chmod someone else's dir
            try:
                os.fchmod(fd, 0o700)
            except OSError as e:
                _log.debug(f"Could not chmod config dir '{directory}': {e}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_private_file(directory: Path, name: str, owner_uid: int) -> bytes | None:
    """Read a regular file in directory without following a symbolic link; None when it (or the dir) is missing.

    When running as root or acting for another user, the file must be owned by owner_uid and have exactly one hard
    link, checked before anything else touches it. Loose group/other permissions are then tightened to 0600
    through the descriptor.

    Raises:
        UnsafeConfigPathError: The dir or the file is refused.
    """
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    read = _read_private_file_and_stat(directory, name, owner_uid)
    return None if read is None else read[0]


def _read_private_file_and_stat(
    directory: Path, name: str, owner_uid: int, harden: bool = True
) -> tuple[bytes, os.stat_result] | None:
    """_read_private_file, also returning the fstat of the descriptor read, so a move knows which inode it copied.

    harden=False leaves the dir and file modes as found (the old platform dir a move reads from).
    """
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    dir_fd = _open_config_dir(directory, create=False, owner_uid=owner_uid, harden=harden)
    if dir_fd is None:
        return None
    try:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
        except FileNotFoundError:
            return None
        except OSError as e:
            if e.errno in _REFUSED_ERRNOS:
                raise UnsafeConfigPathError(e.errno, "symbolic link", str(directory / name)) from None
            raise
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise UnsafeConfigPathError(errno.EINVAL, "not a regular file", str(directory / name))
            if _privileged() and (st.st_uid != owner_uid or st.st_nlink != 1):
                raise UnsafeConfigPathError(
                    errno.EPERM, "not owned by the invoking user or has several hard links", str(directory / name)
                )
            if harden and st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                os.fchmod(fd, 0o600)
            chunks: list[bytes] = []
            while chunk := os.read(fd, 65536):
                chunks.append(chunk)
            return b"".join(chunks), st
        finally:
            os.close(fd)
    finally:
        os.close(dir_fd)


def _write_private_file(directory: Path, name: str, data: bytes) -> Path:
    """Write a 0600 file in directory through a new temporary file moved into place, never truncating in place.

    The existing entry, if any, must be a regular file (and the expected owner's when running as root); it is
    replaced by name, so a hard-linked inode keeps its content. Only the new file is handed back.

    Raises:
        UnsafeConfigPathError: The dir or the existing entry is refused.
        OSError: The write failed.
    """
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    dir_fd = _open_config_dir(directory, create=True)
    assert dir_fd is not None  # create=True never returns None
    path = directory / name
    try:
        try:
            existing: os.stat_result | None = os.lstat(name, dir_fd=dir_fd)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode):
                raise UnsafeConfigPathError(errno.EINVAL, "symbolic link or not a regular file", str(path))
            if _privileged() and existing.st_uid != _owner_uid():
                raise UnsafeConfigPathError(errno.EPERM, "not owned by the invoking user", str(path))
        temp = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
                _hand_back(fd, path)
            finally:
                os.close(fd)
            os.replace(temp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp, dir_fd=dir_fd)
            raise
    finally:
        os.close(dir_fd)
    return path


# --------------- One-time move from the old platform dir (G-05-1d) ---------------
def _create_private_file_no_replace(directory: Path, name: str, data: bytes) -> bool:
    """Create a 0600 file in directory (created 0700 if needed) without ever replacing an existing entry.

    The content goes to a new O_EXCL temporary file, handed back to the invoking user, then hard-linked to name, so
    the final name only ever appears complete and an entry that appeared meanwhile is never replaced. The link never
    follows a symbolic link at the temporary name (one the invoking user renamed over it, CR-01), and the published
    entry must then be a regular file with the inode that was written, or it is removed again. The temporary file
    stays open until that check, so its inode cannot be freed and handed to a file put at the temporary name (ext4
    reuses a freed inode number at once). The file and the dir are fsync'ed before True is returned, so a caller may
    then remove the only other copy (WR-08). The temporary name is always removed.

    Returns:
        True when the file was created, False when an entry with that name already exists (it is kept).

    Raises:
        UnsafeConfigPathError: The dir is refused, or the temporary name was replaced before the link; no file with
            that name is left.
        OSError: The write or the link failed; no file with that name was created.
    """
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    dir_fd = _open_config_dir(directory, create=True)
    assert dir_fd is not None  # create=True never returns None
    path = directory / name
    try:
        temp = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
        try:
            # The descriptor stays open until the published entry is checked: an open file keeps its inode, so a
            # symlink renamed over the temporary name can never receive the same inode number (ext4 reuses one at once).
            try:
                os.fchmod(fd, 0o600)  # whatever the umask
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
                _hand_back(fd, path)
                os.fsync(fd)  # the content is on disk before the name appears (WR-08)
                written = os.fstat(fd)
                try:
                    # linkat without AT_SYMLINK_FOLLOW: a symlink at the temporary name is linked itself, never its
                    # target.
                    os.link(temp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
                except FileExistsError:
                    return False
                try:
                    published = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                    if not stat.S_ISREG(published.st_mode) or (published.st_dev, published.st_ino) != (
                        written.st_dev,
                        written.st_ino,
                    ):
                        raise UnsafeConfigPathError(
                            errno.EPERM, "the temporary file was replaced before it was linked", str(path)
                        )
                    os.fsync(dir_fd)  # the name is on disk before a caller removes the only other copy (WR-08)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.unlink(name, dir_fd=dir_fd)
                    raise
                return True
            finally:
                os.close(fd)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _move_refused(old_path: Path, new_dir: Path, error: OSError) -> OSError:
    """The error for an old file the move did not take: same type and errno, naming both places."""
    reason = error.strerror or str(error)
    message = f"{reason}; it was not moved to {new_dir}: check it and move it there by hand"
    return type(error)(error.errno, message, str(old_path))


def _remove_moved_file(old_dir: Path, name: str, copied: os.stat_result) -> None:
    """Remove the old file by name only while it is still the inode that was copied, then the old dir if empty."""
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    old_path = old_dir / name
    try:
        dir_fd = _open_dir_nofollow(old_dir)
    except OSError as e:
        _log.warning(f"Could not remove '{old_path}' after moving it: {e}. It can be deleted by hand.")
        return
    try:
        current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (copied.st_dev, copied.st_ino):
            _log.warning(f"'{old_path}' changed while it was moved: it was left in place and can be deleted by hand.")
            return
        os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass  # already gone
    except OSError as e:
        _log.warning(f"Could not remove '{old_path}' after moving it: {e}. It can be deleted by hand.")
        return
    finally:
        os.close(dir_fd)
    with contextlib.suppress(OSError):  # only an empty dir goes; never recursive
        os.rmdir(old_dir)


def _read_moved_file(new_dir: Path, name: str) -> bytes | None:
    """The file in the new dir, read as read_user_config_file reads it: what a concurrent start moved there (CR-02)."""
    if os.name == "posix":
        return _read_private_file(new_dir, name, _owner_uid())
    try:
        return (new_dir / name).read_bytes()
    except FileNotFoundError:
        return None


def _move_posix(old_dir: Path, new_dir: Path, name: str) -> bytes | None:
    if sys.platform == "win32":  # POSIX-only helper: callers guard on os.name
        raise OSError(errno.ENOSYS, "POSIX-only config-dir helper")
    old_path = old_dir / name
    try:
        read = _read_private_file_and_stat(old_dir, name, _owner_uid(), harden=False)
        if read is not None:
            data, copied = read
            created = _create_private_file_no_replace(new_dir, name, data)
    except OSError as e:
        raise _move_refused(old_path, new_dir, e) from e
    if read is None:  # gone since the lexists check, moved by a concurrent start: what reached the new dir wins
        return _read_moved_file(new_dir, name)
    if not created:  # an entry appeared in the new dir meanwhile: it wins, the old file stays
        _log.info(f"'{new_dir / name}' appeared while '{old_path}' was moved: kept it, left the old file in place")
        return _read_moved_file(new_dir, name)
    _log.info(f"Moved '{old_path}' to '{new_dir / name}'")
    _remove_moved_file(old_dir, name, copied)
    return data


def _move_plain(old_dir: Path, new_dir: Path, name: str) -> bytes | None:
    """The Windows move: same contract with path operations (no descriptor-relative I/O there)."""
    old_path, new_path = old_dir / name, new_dir / name
    try:
        if old_dir.is_symlink() or old_path.is_symlink() or (old_path.exists() and not old_path.is_file()):
            raise UnsafeConfigPathError(errno.EINVAL, "symbolic link or not a regular file", str(old_path))
        try:
            data = old_path.read_bytes()
        except FileNotFoundError:  # gone since the lexists check, moved by a concurrent start: the new dir's wins
            return _read_moved_file(new_dir, name)
        new_dir.mkdir(parents=True, exist_ok=True)
        try:
            new_file = new_path.open("xb")
        except FileExistsError:
            return new_path.read_bytes()
        try:
            with new_file:
                new_file.write(data)
        except BaseException:
            with contextlib.suppress(OSError):
                new_path.unlink()
            raise
    except OSError as e:
        raise _move_refused(old_path, new_dir, e) from e
    _log.info(f"Moved '{old_path}' to '{new_path}'")
    try:
        old_path.unlink()
    except OSError as e:
        _log.warning(f"Could not remove '{old_path}' after moving it: {e}. It can be deleted by hand.")
        return data
    with contextlib.suppress(OSError):
        old_dir.rmdir()
    return data


def _move_from_platform_dir(name: str) -> bytes | None:
    """Move name from the old platform dir into ~/.exegol-mcp when only the old dir has it (G-05-1d).

    Called once the new dir was found to lack the file. Copy-then-remove: the old file is read through the hardened
    reader (no-follow dir and file, regular file, owner and single link when privileged), created in the new dir
    without replacing anything, then removed only while it is still the inode read. A file already in the new dir
    always wins. Nothing is read or created under sudo on a platform where the invoker is not resolved.

    A concurrent start may move the file first, at any point after the caller found the new dir lacking it (CR-02).
    The file is then read from the new dir, so it never reads as absent while it is only changing places.

    Returns:
        The file's content (now in the new dir), or None when neither dir has it.

    Raises:
        UnresolvedSudoError: Under sudo where the invoking user is not resolved; neither dir is touched.
        OSError: The old file exists but was not moved (refused or an I/O error); it is left where it is, and
            nothing partial stays in the new dir. The message names the old path and the new dir.
    """
    new_dir = user_config_dir()
    if os.name == "posix" and is_unresolved_sudo():  # before any stat: root must not work in an unattributed dir
        raise UnresolvedSudoError(
            errno.EPERM, "the sudo invoking user cannot be resolved on this platform", str(new_dir)
        )
    old_dir = _platform_config_dir()
    if os.path.realpath(old_dir) == os.path.realpath(new_dir):
        return None
    # lexists reads nothing: an old dir that is a symbolic link but holds no such file never blocks startup.
    if not os.path.lexists(old_dir / name) or os.path.lexists(new_dir / name):
        # Nothing to move, or a concurrent start just moved it (the new copy is linked before the old one goes).
        return _read_moved_file(new_dir, name)
    if os.name == "posix":
        return _move_posix(old_dir, new_dir, name)
    return _move_plain(old_dir, new_dir, name)


# --------------- Bearer secret file backend ---------------
_URLSAFE_TOKEN = re.compile(r"^[A-Za-z0-9_-]+$")


def _is_urlsafe_token(candidate: str) -> bool:
    """True for a non-empty string in the `secrets.token_urlsafe` charset (`[A-Za-z0-9_-]`)."""
    return bool(_URLSAFE_TOKEN.match(candidate))


def _legacy_xor_decode(data: bytes) -> bytes:
    """Decode a v1/Phase-1 single-byte-XOR secret file. Read-side migration only; never applied when storing.

    The XOR was never encryption and gave no protection; it survives here solely so an existing v1 secret can be
    recovered and rewritten raw (D-16). The transform is its own inverse, so this both decodes and (historically)
    encoded.
    """
    return bytes(c ^ 0x42 for c in data)


def _store_in_file(path: Path, secret: str) -> None:
    # Honest storage (SEC-06, D-15): the raw token bytes, never XOR'd. The file mode / OS keyring are the controls.
    if os.name == "posix":
        _write_private_file(path.parent, path.name, secret.encode("utf-8"))
    else:
        # On Windows the file resides in the user's AppData; OS ACLs apply per-user. Binary write of the raw bytes
        # with no `encoding=` (v1 passed encoding= to a binary open, a ValueError that stranded the secret and
        # minted a fresh one every run — Pitfall C).
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(secret.encode("utf-8"))


def _load_password_from_file(path: Path, owner_uid: int, migrate: bool = True) -> str | None:
    """Load a bearer secret from the file backend, migrating a legacy XOR file to raw on load (D-16).

    A file whose bytes are already a valid `token_urlsafe` token is used as-is. Otherwise the legacy XOR decode is
    tried, and only a decoded valid token is accepted; when `migrate` is True that token is rewritten raw in place
    through the existing atomic `_store_in_file` so later loads take the raw path and existing configs keep working.
    `migrate` is False when reading another location's secret to copy it elsewhere (legacy adoption), so that
    cross-read never rewrites or re-owns a file this run does not own.
    """
    try:
        if os.name == "posix":
            raw = _read_private_file(path.parent, path.name, owner_uid)
        else:
            raw = path.read_bytes() if path.is_file() else None
        if raw is None:
            return None
        # Raw format: the bytes are already a valid token. Use them as-is (no rewrite).
        candidate = raw.decode("utf-8", errors="replace").strip()
        if candidate and _is_urlsafe_token(candidate):
            return candidate
        # Legacy XOR format: decode and, only for a valid token, migrate the file raw in place.
        try:
            legacy = _legacy_xor_decode(raw).decode("utf-8").strip()
        except ValueError:
            legacy = ""
        if legacy and _is_urlsafe_token(legacy):
            if migrate:
                try:
                    _store_in_file(path, legacy)
                    _log.info(f"Migrated the bearer secret file '{path}' from the legacy format to raw bytes")
                except (OSError, ValueError) as e:
                    _log.warning(f"Could not rewrite the migrated bearer secret '{path}' raw: {e}")
            return legacy
    except (OSError, ValueError) as e:  # refused or unreadable file, or not UTF-8
        _log.warning(f"Failed to read bearer secret file '{path}': {e}")
    return None


def _report_store_failure(path: Path, error: Exception) -> None:
    if isinstance(error, UnresolvedSudoError):
        _log.error(
            f"Refused to store the bearer secret at '{path}': {error}. exegol-mcp supports sudo on Linux only: "
            "run exegol-mcp without sudo. The secret only lives for this run."
        )
    elif isinstance(error, UnsafeConfigPathError):
        _log.error(
            f"Refused to store the bearer secret at '{path}': {error}. It must be a regular file in a directory "
            "owned by the invoking user, not a symbolic link. The secret only lives for this run."
        )
    else:
        _log.error(f"Failed to store bearer secret at '{path}': {error}")


def _adopt_platform_secret(file_path: Path, had_entry: bool) -> str | None:
    """Take over the invoking user's secret from the old platform dir (G-05-1d), moving it into file_path's dir.

    A secret the move refuses stays where it is, with a warning naming it: the server then goes on as on a first
    run. Never used when an entry existed at file_path before it was read (had_entry), which always wins. An entry
    that appeared since was moved there by a concurrent start: it is read, never replaced by a new secret (CR-02).
    """
    if (os.name == "posix" and is_unresolved_sudo()) or had_entry:
        return None  # nothing is looked at under unresolved sudo; an existing entry wins
    try:
        moved = _move_from_platform_dir(file_path.name)
    except OSError as e:
        _log.warning(f"The previous bearer secret was not moved: {e}. A new secret is used.")
        return None
    if moved is None:
        return None
    return _load_password_from_file(file_path, _owner_uid())  # also rewrites a legacy XOR file raw


def _adopt_legacy_secret(file_path: Path) -> str | None:
    """Under sudo, take over a secret an earlier v1 sudo run stranded in root's config dir.

    Client configs written from that run keep authenticating, and later sudo and non-sudo runs share it.
    Only used when the invoking user has no secret of their own; root's copy is left in place.
    """
    legacy_path = _sudo_legacy_secret_file_path()
    if legacy_path is None or legacy_path == file_path:
        return None
    # Read root's own file to copy it to the invoking user's dir; never rewrite/re-own root's copy (migrate=False).
    secret = _load_password_from_file(legacy_path, _legacy_owner_uid(), migrate=False)
    if secret is None:
        return None
    try:
        _store_in_file(file_path, secret)
        _log.info(f"Moved the bearer secret from '{legacy_path}' to the invoking user's '{file_path}'")
    except (OSError, ValueError) as e:
        _report_store_failure(file_path, e)
    return secret


# --------------- Keyring helpers ---------------
def _is_keyring_available() -> bool:
    return keyring is not None


def _get_keyring_backend_name() -> str:
    if not _is_keyring_available():
        return ""
    try:
        kr = keyring.get_keyring()
        # Some backends provide a readable name attribute, else use class name
        return getattr(kr, "name", kr.__class__.__name__)
    except Exception:  # noqa: BLE001 - keyring backends raise their own error types
        return "unknown"


def _load_password_from_keyring() -> str | None:
    if not _is_keyring_available():
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception as e:  # noqa: BLE001  # pragma: no cover - system dependent
        _log.warning(f"Keyring get_password failed: {e}")
        return None


def _store_in_keyring(secret: str) -> bool:
    if not _is_keyring_available():
        return False
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, secret)
        return True
    except Exception as e:  # noqa: BLE001  # pragma: no cover - system dependent
        _log.warning(f"Keyring set_password failed: {e}")
        return False


# --------------- Public API ---------------
def write_user_config_file(name: str, data: bytes) -> Path:
    """Write a small private (0600) file in exegol-mcp's config dir, which is created 0700 if needed.

    On POSIX the file is written to a new temporary file and moved into place, never following a symbolic link.
    Under sudo only the directories and the file this call created are handed back to the invoking user (D-18),
    so a sudo run never leaves a root-owned config dir that a later non-sudo run cannot write to.

    Args:
        name: File name inside the config dir (no path separators).
        data: The file content.

    Returns:
        The path written.

    Raises:
        ValueError: If name is not a plain file name.
        OSError: If the directory or the file cannot be written, or is refused (UnsafeConfigPathError, or
            UnresolvedSudoError under sudo on a platform where the invoking user is not resolved).
    """
    _check_plain_name(name)
    directory = user_config_dir()
    if os.name == "posix":
        return _write_private_file(directory, name, data)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return path


def read_user_config_file(name: str) -> bytes | None:
    """Read a small private file from exegol-mcp's config dir; None when it (or the dir) does not exist.

    On POSIX the file is opened without following a symbolic link and must be a regular file; when running as root
    it must also belong to the expected owner and have a single hard link. Loose permissions are tightened to 0600.
    When the config dir lacks one of ADOPTED_FILE_NAMES and the old platform dir has it, it is moved first (G-05-1d).

    Args:
        name: File name inside the config dir (no path separators).

    Returns:
        The file content, or None.

    Raises:
        ValueError: If name is not a plain file name.
        OSError: If the file cannot be read, or is refused (UnsafeConfigPathError, or UnresolvedSudoError under
            sudo on a platform where the invoking user is not resolved), or an old platform-dir file exists but
            could not be moved (the message names it and the new dir).
    """
    _check_plain_name(name)
    directory = user_config_dir()
    data: bytes | None
    if os.name == "posix":
        data = _read_private_file(directory, name, _owner_uid())
    else:
        try:
            data = (directory / name).read_bytes()
        except FileNotFoundError:
            data = None
    if data is None and name in ADOPTED_FILE_NAMES:
        data = _move_from_platform_dir(name)
    return data


def load_bearer_secret(backend: str = "auto") -> tuple[str, bool, str | None]:
    """Load the bearer secret from secure storage (file or OS keyring), creating a new one on first run.

    Not cached: call it once at startup and pass the secret to whatever needs it.

    Args:
        backend: 'auto' (default), 'keyring', or 'file'.

    Returns:
        (secret, created, location), where location is:
            - the file path when using the file backend
            - 'keyring:<backend name>' when using keyring
            - None when a new secret could not be stored (it then only lives for this run)

    Raises:
        RuntimeError: If backend is 'keyring' and the keyring package is not installed.
    """
    system = platform.system().lower()
    use_keyring = False

    if backend == "keyring":
        if not _is_keyring_available():
            raise RuntimeError("Keyring backend requested but 'keyring' package is not installed")
        use_keyring = True
    elif backend == "file":
        use_keyring = False
    else:  # auto
        # Prefer keyring on Windows/macOS when available
        if system in ("windows", "darwin") and _is_keyring_available():
            use_keyring = True

    if use_keyring:
        # Try loading from keyring first
        secret = _load_password_from_keyring()
        if secret:
            return secret, False, f"keyring:{_get_keyring_backend_name()}"

        # Create new secret and store in keyring
        new_secret = secrets.token_urlsafe(32)
        if _store_in_keyring(new_secret):
            return new_secret, True, f"keyring:{_get_keyring_backend_name()}"
        _log.warning("Keyring not usable, falling back to file backend")
        # Fall back to file storage

    file_path = secret_file_path()

    # FILE BACKEND path (explicit file, or auto fallback)
    # Taken before the read, so an entry a concurrent start moves in afterwards is not mistaken for an unusable one
    # (CR-02). Nothing is looked at under unresolved sudo: the read below refuses before any stat.
    had_entry = not (os.name == "posix" and is_unresolved_sudo()) and os.path.lexists(file_path)
    existing = (
        _load_password_from_file(file_path, _owner_uid())
        or _adopt_platform_secret(file_path, had_entry)
        or _adopt_legacy_secret(file_path)
    )
    if existing:
        return existing, False, str(file_path)

    # Generate and store new
    secret = secrets.token_urlsafe(32)
    try:
        _store_in_file(file_path, secret)
    except (OSError, ValueError) as e:  # refused path (UnsafeConfigPathError) or a write failure
        _report_store_failure(file_path, e)
        # Still return the secret to allow running, but it only lives for this run
        return secret, True, None
    return secret, True, str(file_path)
