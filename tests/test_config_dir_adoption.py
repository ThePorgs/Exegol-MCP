"""The ~/.exegol-mcp config dir and the one-time move of the old platform-dir files (G-05-1d). Docker-free.

Every test works in tmp_path: the conftest guard points the move source at a temp dir, the `home` fixture points
the invoking user's home at another, and the end-to-end launch gets a temporary HOME and XDG_CONFIG_HOME. The
developer's real ~/.exegol-mcp and old platform dir are never read, moved or written.
"""

import os
import platform
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from exegol_mcp.bootstrap import config, secret_store, versions
from exegol_mcp.bootstrap.secret_store import (
    ADOPTED_FILE_NAMES,
    APP_DIR_NAME,
    CONFIG_DIR_NAME,
    SECRET_FILE_NAME,
    load_bearer_secret,
)
from tests.test_secret_store import FakeKeyring

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes and descriptors")

CONFIG_YML = b"tool_groups:\n  disabled: [images]\n"
CACHE_JSON = b'{"checked_at": 1.0, "latest_version": "9.9.9"}'


@pytest.fixture
def old_dir(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The old platform dir (here the Linux one, under the fake home), created 0700 and empty."""
    directory = home / ".config" / APP_DIR_NAME
    directory.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(secret_store, "_platform_config_dir", lambda: directory)
    return directory


@pytest.fixture
def new_dir(home: Path) -> Path:
    """~/.exegol-mcp under the fake home (not created)."""
    return home / CONFIG_DIR_NAME


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --------------- The suite guard and the constants ---------------
def test_the_guard_keeps_the_move_source_inside_tmp_path(tmp_path: Path) -> None:
    """T-05-33: under the autouse conftest guard the old platform dir is a temp path, never the real one."""
    source = secret_store._platform_config_dir()
    assert source.is_relative_to(tmp_path)
    assert not source.exists()


def test_adopted_file_names_match_the_files_their_modules_read() -> None:
    """The moved names are named by value in secret_store; they must stay the names config and versions read."""
    assert ADOPTED_FILE_NAMES == ("config.yml", "bearer.secret", "update-check.json")
    assert ADOPTED_FILE_NAMES == (config.CONFIG_FILE_NAME, SECRET_FILE_NAME, versions.UPDATE_CACHE_FILE)


# --------------- The move, file by file ---------------
@posix_only
def test_config_yml_is_moved_and_its_restrictions_apply(old_dir: Path, new_dir: Path) -> None:
    """An old-dir config.yml is moved on first read: same bytes, 0600 in a 0700 dir, old file and empty dir gone."""
    (old_dir / "config.yml").write_bytes(CONFIG_YML)

    cfg = config.load_operator_config()

    assert cfg.tool_groups.disabled == ["images"]
    assert (new_dir / "config.yml").read_bytes() == CONFIG_YML
    assert _mode(new_dir / "config.yml") == 0o600
    assert _mode(new_dir) == 0o700
    assert not old_dir.exists()  # the file went, then the emptied dir
    assert sorted(entry.name for entry in new_dir.iterdir()) == ["config.yml"]  # no temp file left
    assert config.load_operator_config().tool_groups.disabled == ["images"]  # later reads take the new file


@posix_only
def test_the_update_cache_is_moved_through_read_user_config_file(old_dir: Path, new_dir: Path) -> None:
    """update-check.json takes the same move through the public reader."""
    (old_dir / versions.UPDATE_CACHE_FILE).write_bytes(CACHE_JSON)

    assert secret_store.read_user_config_file(versions.UPDATE_CACHE_FILE) == CACHE_JSON

    assert (new_dir / versions.UPDATE_CACHE_FILE).read_bytes() == CACHE_JSON
    assert _mode(new_dir / versions.UPDATE_CACHE_FILE) == 0o600
    assert not (old_dir / versions.UPDATE_CACHE_FILE).exists()


@posix_only
def test_the_bearer_secret_is_moved_and_keeps_authenticating(old_dir: Path, new_dir: Path) -> None:
    """A v1/early-v2 file secret is taken over: same secret, not created, located in ~/.exegol-mcp."""
    (old_dir / SECRET_FILE_NAME).write_bytes(b"old-platform-secret_123")

    secret, created, location = load_bearer_secret(backend="file")

    assert (secret, created, location) == ("old-platform-secret_123", False, str(new_dir / SECRET_FILE_NAME))
    assert (new_dir / SECRET_FILE_NAME).read_bytes() == b"old-platform-secret_123"
    assert _mode(new_dir / SECRET_FILE_NAME) == 0o600
    assert not old_dir.exists()
    assert load_bearer_secret(backend="file") == (secret, False, location)


@posix_only
def test_a_legacy_xor_secret_is_moved_then_rewritten_raw(old_dir: Path, new_dir: Path) -> None:
    """The legacy XOR format is moved as is, then migrated raw in the new location (D-16)."""
    token = b"legacy-token_ABC123-xyz"
    (old_dir / SECRET_FILE_NAME).write_bytes(bytes(c ^ 0x42 for c in token))

    secret, created, _ = load_bearer_secret(backend="file")

    assert (secret, created) == (token.decode(), False)
    assert (new_dir / SECRET_FILE_NAME).read_bytes() == token
    assert not (old_dir / SECRET_FILE_NAME).exists()


@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
def test_a_read_with_nothing_to_move_creates_nothing(old_dir: Path, new_dir: Path, name: str) -> None:
    """Neither dir holds the file: the read gives None and no ~/.exegol-mcp is created by a read."""
    assert secret_store.read_user_config_file(name) is None
    assert not new_dir.exists()
    assert old_dir.is_dir()


def test_load_operator_config_with_nothing_to_move_gives_the_defaults(old_dir: Path, new_dir: Path) -> None:
    """No config.yml anywhere: the built-in defaults, nothing created."""
    assert config.load_operator_config().tool_groups.disabled == []
    assert not new_dir.exists()


# --------------- End to end through the real entry point ---------------
def _old_platform_dir(home: Path, xdg: Path) -> Path:
    """The old dir the child resolves, picked with platform.system() exactly as _platform_config_dir does."""
    system = platform.system().lower()
    if system == "darwin":
        return home / "Library" / "Application Support" / APP_DIR_NAME
    return xdg / APP_DIR_NAME


@pytest.mark.skipif(sys.platform == "win32", reason="the child's old dir is resolved the POSIX way")
def test_the_entry_point_moves_an_old_config_yml_and_fails_closed_on_it(tmp_path: Path) -> None:
    """A real `python -m exegol_mcp --type stdio` reads the old-dir config.yml through ~/.exegol-mcp.

    The file is invalid, so main() exits 1 naming config.yml, which proves the moved file is the one validated; it
    now sits in HOME/.exegol-mcp and no longer in the old dir.
    """
    home, xdg = tmp_path / "home", tmp_path / "home" / "cfg"
    old = _old_platform_dir(home, xdg)
    old.mkdir(parents=True)
    (old / "config.yml").write_bytes(b"bogus_key: 1\n")
    env = {name: value for name, value in os.environ.items() if name not in ("PYTHONSAFEPATH", "PYTHONPATH")}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(xdg))

    launched = subprocess.run(
        [sys.executable, "-m", "exegol_mcp", "--type", "stdio"],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert launched.returncode == 1, launched.stderr
    assert launched.stdout == ""
    assert "config.yml" in launched.stderr
    assert (home / CONFIG_DIR_NAME / "config.yml").read_bytes() == b"bogus_key: 1\n"
    assert not (old / "config.yml").exists()


# =============== Refusal, concurrency and sudo matrix (T-05-29..T-05-32) ===============
CONTENT = {"config.yml": CONFIG_YML, SECRET_FILE_NAME: b"old-platform-secret_123", "update-check.json": CACHE_JSON}
Setup = Callable[[Path, str, Path, pytest.MonkeyPatch], tuple[Path, list[Path]]]


def _entry(path: Path) -> tuple[int, int, int, bytes | str | None]:
    """Mode, inode, link count and content (link target for a symlink) of an entry, never following it."""
    st = os.lstat(path)
    content: bytes | str | None = None
    if stat.S_ISLNK(st.st_mode):
        content = os.readlink(path)
    elif stat.S_ISREG(st.st_mode):
        content = path.read_bytes()
    return st.st_mode, st.st_ino, st.st_nlink, content


def _tree(root: Path) -> dict[str, tuple[int, int, int, bytes | str | None]]:
    """Every entry under root (not following links), for "nothing was opened, created or changed" checks."""
    entries = {".": _entry(root)}
    for parent, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(parent) / name
            entries[str(path.relative_to(root))] = _entry(path)
    return entries


def _checker(fetched: list[float]) -> versions.UpdateChecker:
    """An update checker whose cache, if read, would be fresh: a fetch proves the cache was not used."""

    def pypi(timeout: float) -> str | None:
        fetched.append(timeout)
        return "2.0.0"

    return versions.UpdateChecker(enabled=True, current="2.0.0", fetch=pypi, now=lambda: 2.0)


def _assert_falls_back(name: str, old_path: Path, new_dir: Path) -> None:
    """Each file's own consequence of a refused move: config.yml fails closed, the secret and the cache start over."""
    if name == "config.yml":
        with pytest.raises(config.ConfigError) as refused:
            config.load_operator_config()
        assert str(old_path) in str(refused.value) and CONFIG_DIR_NAME in str(refused.value)
    elif name == SECRET_FILE_NAME:
        secret, created, location = load_bearer_secret(backend="file")
        assert (created, location) == (True, str(new_dir / SECRET_FILE_NAME))
        assert secret != CONTENT[SECRET_FILE_NAME].decode()
        assert (new_dir / SECRET_FILE_NAME).read_bytes() == secret.encode()
    else:
        fetched: list[float] = []
        _checker(fetched).check()
        assert len(fetched) == 1


def _symlinked_file(old_dir: Path, name: str, tmp_path: Path, mp: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    victim = tmp_path / "victim"
    victim.write_bytes(CONTENT[name])
    victim.chmod(0o640)
    (old_dir / name).symlink_to(victim)
    return old_dir / name, [old_dir / name, victim]


def _symlinked_dir(old_dir: Path, name: str, tmp_path: Path, mp: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    real = tmp_path / "real-old-dir"
    real.mkdir()
    real.chmod(0o755)
    (real / name).write_bytes(CONTENT[name])
    link = old_dir.parent / "linked-old-dir"
    link.symlink_to(real, target_is_directory=True)
    mp.setattr(secret_store, "_platform_config_dir", lambda: link)
    return link / name, [link, real, real / name]


def _directory(old_dir: Path, name: str, tmp_path: Path, mp: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    (old_dir / name).mkdir()
    return old_dir / name, [old_dir / name]


def _fifo(old_dir: Path, name: str, tmp_path: Path, mp: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    os.mkfifo(old_dir / name)  # opened O_NONBLOCK, so the read never waits for a writer
    return old_dir / name, [old_dir / name]


def _hard_linked(old_dir: Path, name: str, tmp_path: Path, mp: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    victim = tmp_path / "victim"
    victim.write_bytes(CONTENT[name])
    victim.chmod(0o644)
    os.link(victim, old_dir / name)
    return old_dir / name, [old_dir / name, victim]


def _foreign_owned(old_dir: Path, name: str, tmp_path: Path, mp: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    (old_dir / name).write_bytes(CONTENT[name])
    (old_dir / name).chmod(0o644)
    foreign_inode = (old_dir / name).stat().st_ino
    real_fstat = os.fstat

    def fstat(fd: int) -> os.stat_result:  # the file reads as someone else's, the one check the reader makes
        st = real_fstat(fd)
        if st.st_ino != foreign_inode:
            return st
        fields = list(st[:10])
        fields[stat.ST_UID] = os.geteuid() + 1
        return os.stat_result(fields)

    mp.setattr(os, "fstat", fstat)
    return old_dir / name, [old_dir / name]


REFUSALS: dict[str, tuple[Setup, bool]] = {  # case -> (setup, needs root acting for the invoker)
    "symlinked-file": (_symlinked_file, False),
    "symlinked-dir": (_symlinked_dir, False),
    "directory": (_directory, False),
    "fifo": (_fifo, False),
    "hard-linked-as-root": (_hard_linked, True),
    "foreign-owned-as-root": (_foreign_owned, True),
}


@posix_only
@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_an_unsafe_old_entry_is_never_moved(
    request: pytest.FixtureRequest,
    old_dir: Path,
    new_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    name: str,
) -> None:
    """T-05-29/T-05-31: a symlink, non-regular, hard-linked or foreign entry stays put, nothing lands in the new dir,
    the error names both places, and each file falls back its own way (config.yml stops startup)."""
    setup, as_root = REFUSALS[case]
    if as_root:
        request.getfixturevalue("sudo_as_root")
    old_path, watched = setup(old_dir, name, tmp_path, monkeypatch)
    before = [_entry(path) for path in watched]

    with pytest.raises(OSError) as refused:
        secret_store.read_user_config_file(name)

    assert str(old_path) in str(refused.value) and CONFIG_DIR_NAME in str(refused.value)
    assert "by hand" in str(refused.value)
    assert not new_dir.exists() or list(new_dir.iterdir()) == []  # no file, no temp file
    assert [_entry(path) for path in watched] == before
    _assert_falls_back(name, old_path, new_dir)
    assert [_entry(path) for path in watched] == before


@posix_only
@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
def test_a_file_already_in_the_new_dir_always_wins(old_dir: Path, new_dir: Path, name: str) -> None:
    """T-05-30: with both files present the new one is read and the old one keeps its bytes, mode and inode."""
    (old_dir / name).write_bytes(CONTENT[name])
    new_dir.mkdir(mode=0o700)
    (new_dir / name).write_bytes(b"already-here_456")
    (new_dir / name).chmod(0o600)
    before = _entry(old_dir / name)

    assert secret_store.read_user_config_file(name) == b"already-here_456"
    if name == SECRET_FILE_NAME:
        assert load_bearer_secret(backend="file") == ("already-here_456", False, str(new_dir / name))

    assert _entry(old_dir / name) == before
    assert (new_dir / name).read_bytes() == b"already-here_456"


@posix_only
def test_a_symlinked_old_dir_without_the_files_never_blocks_startup(
    old_dir: Path, new_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to move is not a refusal: every read gives None and config.yml falls back to the defaults."""
    real = tmp_path / "real-old-dir"
    real.mkdir()
    (real / "unrelated.txt").write_bytes(b"x")
    link = old_dir.parent / "linked-old-dir"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(secret_store, "_platform_config_dir", lambda: link)

    for name in ADOPTED_FILE_NAMES:
        assert secret_store.read_user_config_file(name) is None
    assert config.load_operator_config().tool_groups.disabled == []

    assert not new_dir.exists()
    assert link.is_symlink() and sorted(entry.name for entry in real.iterdir()) == ["unrelated.txt"]


@posix_only
@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
def test_an_entry_appearing_in_the_new_dir_during_the_move_is_kept(
    old_dir: Path, new_dir: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """T-05-30: the no-replace link loses to a file created between the read and the link; that file is kept
    byte-identical, the temporary file is removed and the old file is left untouched."""
    (old_dir / name).write_bytes(CONTENT[name])
    before = _entry(old_dir / name)
    real_link = os.link

    def racing_link(src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int, follow_symlinks: bool = True) -> None:
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dst_dir_fd)
        os.write(fd, b"concurrent-writer_789")
        os.close(fd)
        # now raises FileExistsError
        real_link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "link", racing_link)

    assert secret_store.read_user_config_file(name) == b"concurrent-writer_789"

    assert (new_dir / name).read_bytes() == b"concurrent-writer_789"
    assert sorted(entry.name for entry in new_dir.iterdir()) == [name]  # the temp file is gone
    assert _entry(old_dir / name) == before


@posix_only
@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
@pytest.mark.parametrize("link_follows", [False, True], ids=["link-honours-no-follow", "link-follows-anyway"])
def test_a_symlink_swapped_over_the_temporary_file_is_never_published(
    sudo_as_root: Path,
    old_dir: Path,
    new_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_follows: bool,
    name: str,
) -> None:
    """CR-01: the invoking user renames a symlink over the temporary name before root links it to the final name.

    The link never follows it, and the published entry is checked to be the inode that was written: either way no
    hard link to the symlink's target is left in the new dir, the move is refused naming both places, and the old
    file and the target are untouched. link-follows-anyway stands for a platform that ignores the no-follow flag.
    """
    (old_dir / name).write_bytes(CONTENT[name])
    victim = tmp_path / "victim"  # stands for a root-owned file such as /etc/shadow
    victim.write_bytes(b"root-only")
    before = [_entry(old_dir / name), _entry(victim)]
    real_link = os.link

    def swapping_link(src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int, follow_symlinks: bool = True) -> None:
        os.unlink(src, dir_fd=src_dir_fd)
        os.symlink(victim, src, dir_fd=src_dir_fd)
        follow = True if link_follows else follow_symlinks
        real_link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, follow_symlinks=follow)

    monkeypatch.setattr(os, "link", swapping_link)

    with pytest.raises(OSError) as refused:
        secret_store.read_user_config_file(name)

    assert str(old_dir / name) in str(refused.value) and CONFIG_DIR_NAME in str(refused.value)
    assert list(new_dir.iterdir()) == []  # no published entry, no temporary name
    assert [_entry(old_dir / name), _entry(victim)] == before


@posix_only
@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
def test_the_new_copy_and_its_dir_are_flushed_before_the_old_file_is_removed(
    old_dir: Path, new_dir: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """WR-08: the old file is the only copy until the new one is durable, so the new file and the new dir are
    fsync'ed before the old file is unlinked."""
    (old_dir / name).write_bytes(CONTENT[name])
    old_dir_inode = old_dir.stat().st_ino
    events: list[tuple[str, int]] = []
    real_fsync, real_unlink = os.fsync, os.unlink

    def fsync(fd: int) -> None:
        events.append(("fsync", os.fstat(fd).st_ino))
        real_fsync(fd)

    def unlink(path: str, *, dir_fd: int | None = None) -> None:
        if dir_fd is not None and path == name and os.fstat(dir_fd).st_ino == old_dir_inode:
            events.append(("unlink-old", 0))
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "unlink", unlink)

    assert secret_store.read_user_config_file(name) == CONTENT[name]

    removed = events.index(("unlink-old", 0))
    flushed = {inode for event, inode in events[:removed] if event == "fsync"}
    assert {(new_dir / name).stat().st_ino, new_dir.stat().st_ino} <= flushed


def _moved_by_another_start(old_dir: Path, new_dir: Path, name: str) -> None:
    """What a concurrent start's completed move leaves: the file in the new dir, the old file and its dir gone."""
    new_dir.mkdir(mode=0o700, exist_ok=True)
    (new_dir / name).write_bytes(CONTENT[name])
    (new_dir / name).chmod(0o600)
    (old_dir / name).unlink()
    old_dir.rmdir()


def _assert_reads_the_moved_file(name: str, new_dir: Path) -> None:
    """Each file's reader takes what the other start moved: config.yml keeps its restrictions, the secret is not
    minted again (the file keeps its bytes), the update cache is the moved one."""
    if name == "config.yml":
        assert config.load_operator_config().tool_groups.disabled == ["images"]
    elif name == SECRET_FILE_NAME:
        secret = CONTENT[SECRET_FILE_NAME].decode()
        assert load_bearer_secret(backend="file") == (secret, False, str(new_dir / SECRET_FILE_NAME))
    else:
        assert secret_store.read_user_config_file(name) == CONTENT[name]
    assert (new_dir / name).read_bytes() == CONTENT[name]


@posix_only
@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
@pytest.mark.parametrize("window", ["after-the-new-dir-read", "after-the-old-file-check"])
def test_a_file_another_start_moves_meanwhile_is_read_from_the_new_dir(
    old_dir: Path, new_dir: Path, monkeypatch: pytest.MonkeyPatch, window: str, name: str
) -> None:
    """CR-02: two starts at once. This one finds no file in the new dir, then the other one completes its move, either
    before this one checks the old dir or between that check and its read of the old file. The file never reads as
    absent: config.yml does not fall back to the defaults and no second secret is minted."""
    (old_dir / name).write_bytes(CONTENT[name])
    real_read = secret_store._read_private_file
    real_read_and_stat = secret_store._read_private_file_and_stat
    moved: list[bool] = []

    def read_then_other_start_moves(directory: Path, entry: str, owner_uid: int) -> bytes | None:
        data = real_read(directory, entry, owner_uid)
        if window == "after-the-new-dir-read" and directory == new_dir and not moved:
            moved.append(True)
            _moved_by_another_start(old_dir, new_dir, entry)
        return data

    def other_start_moves_then_read(directory: Path, entry: str, owner_uid: int, harden: bool = True) -> Any:
        if window == "after-the-old-file-check" and directory == old_dir and not moved:
            moved.append(True)
            _moved_by_another_start(old_dir, new_dir, entry)
        return real_read_and_stat(directory, entry, owner_uid, harden)

    monkeypatch.setattr(secret_store, "_read_private_file", read_then_other_start_moves)
    monkeypatch.setattr(secret_store, "_read_private_file_and_stat", other_start_moves_then_read)

    _assert_reads_the_moved_file(name, new_dir)

    assert moved == [True]
    assert not old_dir.exists()


@posix_only
@pytest.mark.parametrize("name", ADOPTED_FILE_NAMES)
def test_an_old_file_replaced_after_it_was_read_is_not_removed(
    old_dir: Path, new_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, name: str
) -> None:
    """Removal is by name only for the inode copied: a file swapped in meanwhile stays, and so does its dir."""
    (old_dir / name).write_bytes(CONTENT[name])
    real_read = secret_store._read_private_file_and_stat

    def read_then_replace(directory: Path, entry: str, owner_uid: int, harden: bool = True) -> Any:
        read = real_read(directory, entry, owner_uid, harden)
        if directory == old_dir:  # created while the original still exists, so it gets another inode
            (old_dir / ".swap").write_bytes(b"swapped-in")
            os.replace(old_dir / ".swap", old_dir / entry)
        return read

    monkeypatch.setattr(secret_store, "_read_private_file_and_stat", read_then_replace)

    with caplog.at_level("WARNING"):
        assert secret_store.read_user_config_file(name) == CONTENT[name]

    assert (new_dir / name).read_bytes() == CONTENT[name]
    assert (old_dir / name).read_bytes() == b"swapped-in"
    assert str(old_dir / name) in caplog.text and "left in place" in caplog.text


@posix_only
def test_the_old_dir_is_kept_while_it_holds_other_files(old_dir: Path, new_dir: Path) -> None:
    """Only an empty old dir is removed, never recursively."""
    (old_dir / "config.yml").write_bytes(CONFIG_YML)
    (old_dir / "notes.txt").write_bytes(b"mine")

    assert secret_store.read_user_config_file("config.yml") == CONFIG_YML

    assert sorted(entry.name for entry in old_dir.iterdir()) == ["notes.txt"]
    assert (new_dir / "config.yml").read_bytes() == CONFIG_YML


# --------------- Sudo, root and the keyring ---------------
@posix_only
def test_under_linux_sudo_the_new_dir_and_every_moved_file_are_handed_back(
    sudo_as_root: Path, old_dir: Path, new_dir: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    """T-05-32: root creates ~user/.exegol-mcp and the moved files, then hands each back to the invoker."""
    for name in ADOPTED_FILE_NAMES:
        (old_dir / name).write_bytes(CONTENT[name])

    for name in ADOPTED_FILE_NAMES:
        assert secret_store.read_user_config_file(name) == CONTENT[name]

    moved = {(new_dir / name).stat().st_ino for name in ADOPTED_FILE_NAMES}
    assert {inode for inode, _, _ in fchown_calls} == moved | {new_dir.stat().st_ino}
    assert {(uid, gid) for _, uid, gid in fchown_calls} == {(os.geteuid(), os.getegid())}
    assert all(_mode(new_dir / name) == 0o600 for name in ADOPTED_FILE_NAMES)
    assert not old_dir.exists()


@posix_only
def test_under_linux_sudo_the_invokers_old_secret_wins_over_roots_v1_secret(
    sudo_as_root: Path, old_dir: Path, new_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invoker's old platform dir is tried before root's v1 secret, which is then left alone."""
    monkeypatch.setattr(secret_store, "_legacy_owner_uid", os.geteuid)  # the test cannot create root-owned files
    legacy = tmp_path / "root" / ".config" / APP_DIR_NAME / SECRET_FILE_NAME
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"root-v1-secret")
    (old_dir / SECRET_FILE_NAME).write_bytes(b"invoker-old-secret")
    before = _entry(legacy)

    secret, created, location = load_bearer_secret(backend="file")

    assert (secret, created, location) == ("invoker-old-secret", False, str(new_dir / SECRET_FILE_NAME))
    assert _entry(legacy) == before
    assert not (old_dir / SECRET_FILE_NAME).exists()


@posix_only
def test_root_without_an_invoker_never_moves_from_a_user_owned_old_dir(
    root_without_invoker: Path, old_dir: Path, new_dir: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    """Root with no resolved invoker expects root-owned dirs: the user's old dir is refused and nothing is created."""
    for name in ADOPTED_FILE_NAMES:
        (old_dir / name).write_bytes(CONTENT[name])
    before = _tree(old_dir)

    with pytest.raises(config.ConfigError) as refused:
        config.load_operator_config()
    _, _, location = load_bearer_secret(backend="file")

    assert str(old_dir / "config.yml") in str(refused.value)
    assert location is None
    assert _tree(old_dir) == before
    assert not new_dir.exists()
    assert fchown_calls == []


@posix_only
def test_under_unresolved_sudo_neither_dir_is_opened_created_or_changed(
    macos_sudo: Path, old_dir: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    """macOS sudo: every read raises UnresolvedSudoError and the whole home, old dir included, stays as it was."""
    for name in ADOPTED_FILE_NAMES:
        (old_dir / name).write_bytes(CONTENT[name])
        (old_dir / name).chmod(0o644)
    before = _tree(macos_sudo)

    for name in ADOPTED_FILE_NAMES:
        with pytest.raises(secret_store.UnresolvedSudoError):
            secret_store.read_user_config_file(name)
        with pytest.raises(secret_store.UnresolvedSudoError):  # the engine refuses on its own too
            secret_store._move_from_platform_dir(name)
    _, _, location = load_bearer_secret(backend="file")
    with pytest.raises(config.ConfigError):
        config.load_operator_config()

    assert location is None
    assert _tree(macos_sudo) == before
    assert fchown_calls == []


def test_a_keyring_secret_leaves_the_old_secret_file_alone(
    home: Path, old_dir: Path, new_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No keyring entry is read differently and no old file is moved when the keyring holds the secret."""
    fake = FakeKeyring()
    fake.store[(secret_store.KEYRING_SERVICE, secret_store.KEYRING_USERNAME)] = "keyring-secret"
    monkeypatch.setattr(secret_store, "keyring", fake)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    (old_dir / SECRET_FILE_NAME).write_bytes(CONTENT[SECRET_FILE_NAME])
    before = _entry(old_dir / SECRET_FILE_NAME)

    assert load_bearer_secret() == ("keyring-secret", False, "keyring:FakeKeyring")

    assert _entry(old_dir / SECRET_FILE_NAME) == before
    assert not new_dir.exists()


# --------------- The Windows branch (path operations, exercised on POSIX CI) ---------------
def test_the_windows_branch_moves_a_regular_old_file(
    old_dir: Path, new_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.name != "posix": exclusive-create copy, then the old file and its emptied dir are removed."""
    (old_dir / "config.yml").write_bytes(CONFIG_YML)

    # The engine itself, and os.name only around the call: Path("...") cannot be built on POSIX under "nt".
    with monkeypatch.context() as m:
        m.setattr(os, "name", "nt")
        moved = secret_store._move_from_platform_dir("config.yml")

    assert moved == CONFIG_YML

    assert (new_dir / "config.yml").read_bytes() == CONFIG_YML
    assert not old_dir.exists()


def test_the_windows_branch_reads_a_file_another_start_moved_meanwhile(
    old_dir: Path, new_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR-02 on the Windows branch: an old file gone since the existence check is not "not a regular file"; the
    file the other start put in the new dir is read."""
    (old_dir / "config.yml").write_bytes(CONFIG_YML)
    _moved_by_another_start(old_dir, new_dir, "config.yml")

    with monkeypatch.context() as m:
        m.setattr(os, "name", "nt")
        moved = secret_store._move_plain(old_dir, new_dir, "config.yml")

    assert moved == CONFIG_YML
    assert (new_dir / "config.yml").read_bytes() == CONFIG_YML


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privileges on Windows")
def test_the_windows_branch_refuses_a_symlinked_old_file(
    old_dir: Path, new_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Windows branch never follows a symbolic link either; the error names the old file and the new dir."""
    victim = tmp_path / "victim"
    victim.write_bytes(CONFIG_YML)
    (old_dir / "config.yml").symlink_to(victim)

    with pytest.raises(OSError) as refused, monkeypatch.context() as m:
        m.setattr(os, "name", "nt")
        secret_store._move_from_platform_dir("config.yml")

    assert str(old_dir / "config.yml") in str(refused.value) and CONFIG_DIR_NAME in str(refused.value)
    assert not new_dir.exists()
    assert (old_dir / "config.yml").is_symlink() and victim.read_bytes() == CONFIG_YML
