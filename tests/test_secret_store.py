"""Bearer secret store: v1 location/format/backends kept, sudo-aware dir and hand-back (D-18). Docker-free.

Every test redirects the config dir into tmp_path and replaces the keyring, so the operator's real bearer secret
and OS keychain are never read or written. The live sudo path (sudo-rs host, D-17/A4) is a manual host check:
`sudo exegol-mcp --print-config` must put the secret under the invoking user's home, owned by that user.
"""

import errno
import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from exegol_mcp.adapter import sudo
from exegol_mcp.bootstrap import secret_store, versions
from exegol_mcp.bootstrap.secret_store import (
    APP_DIR_NAME,
    CONFIG_DIR_NAME,
    KEYRING_SERVICE,
    KEYRING_USERNAME,
    SECRET_FILE_NAME,
    load_bearer_secret,
    secret_file_path,
    user_config_dir,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes and ownership")


class FakeKeyring:
    """In-memory stand-in for the keyring module."""

    name = "FakeKeyring"

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_keyring(self) -> "FakeKeyring":
        return self

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password


def _snapshot(path: Path) -> tuple[bytes, int, int]:
    """Content, mode and inode of a file, read without following anything but the given path."""
    st = path.stat()
    return path.read_bytes(), stat.S_IMODE(st.st_mode), st.st_ino


def test_v1_constants_are_kept() -> None:
    assert (APP_DIR_NAME, SECRET_FILE_NAME, KEYRING_SERVICE, KEYRING_USERNAME) == (
        "exegol-mcp",
        "bearer.secret",
        "exegol-mcp",
        "bearer",
    )


def test_file_round_trip(home: Path) -> None:
    secret, created, location = load_bearer_secret(backend="file")
    path = secret_file_path()
    assert created is True
    assert location == str(path)
    assert path.is_relative_to(home)
    assert path.is_file()

    again, created_again, location_again = load_bearer_secret(backend="file")
    assert (again, created_again, location_again) == (secret, False, str(path))


@posix_only
def test_file_and_dir_modes_are_private(home: Path) -> None:
    load_bearer_secret(backend="file")
    path = secret_file_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@posix_only
def test_loose_file_permissions_are_hardened_on_read(home: Path) -> None:
    secret, _, _ = load_bearer_secret(backend="file")
    path = secret_file_path()
    path.chmod(0o644)
    assert load_bearer_secret(backend="file")[0] == secret
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_on_disk_format_is_the_raw_token(home: Path) -> None:
    # SEC-06/D-15: the 0600 file holds the raw token bytes, never XOR'd.
    secret, _, _ = load_bearer_secret(backend="file")
    raw = secret_file_path().read_bytes()
    assert raw == secret.encode("utf-8")
    assert load_bearer_secret(backend="file")[0] == secret


def test_a_legacy_xor_secret_is_migrated_to_raw_on_load(home: Path) -> None:
    # D-16/COMP-01: an existing v1 XOR file decodes to the original token, keeps authenticating, and is rewritten
    # raw in place so later loads take the raw path with no operator action.
    token = "legacy-token_ABC123-xyz"  # valid token_urlsafe charset
    path = secret_file_path()
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_bytes(bytes(c ^ 0x42 for c in token.encode("utf-8")))

    loaded, created, location = load_bearer_secret(backend="file")

    assert (loaded, created, location) == (token, False, str(path))
    assert path.read_bytes() == token.encode("utf-8")  # rewritten raw in place
    assert load_bearer_secret(backend="file")[0] == token


def test_windows_branch_writes_raw_bytes_without_valueerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pitfall C: the v1 Windows write passed encoding= to a binary open (ValueError). Exercise the Windows branch
    # on POSIX CI by patching os.name and assert it writes the raw bytes and reads them back.
    monkeypatch.setattr(secret_store.os, "name", "nt")
    path = tmp_path / "winconf" / SECRET_FILE_NAME
    token = "windows-token_XYZ789-abc"

    secret_store._store_in_file(path, token)  # must not raise ValueError

    assert path.read_bytes() == token.encode("utf-8")
    assert secret_store._load_password_from_file(path, 0) == token


def test_no_process_cache_each_call_reads_storage(home: Path) -> None:
    first, _, _ = load_bearer_secret(backend="file")
    secret_file_path().unlink()
    second, created, _ = load_bearer_secret(backend="file")
    assert created is True
    assert second != first
    assert not any("bearer_secret" in name and not callable(value) for name, value in vars(secret_store).items())


def test_auto_without_keyring_uses_the_file(home: Path) -> None:
    secret, created, location = load_bearer_secret()
    assert created is True
    assert location == str(secret_file_path())
    assert load_bearer_secret()[0] == secret


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_auto_prefers_keyring_on_macos_and_windows(home: Path, monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    fake = FakeKeyring()
    monkeypatch.setattr(secret_store, "keyring", fake)
    monkeypatch.setattr(platform, "system", lambda: system)

    secret, created, location = load_bearer_secret()
    assert (created, location) == (True, "keyring:FakeKeyring")
    assert fake.store == {(KEYRING_SERVICE, KEYRING_USERNAME): secret}
    assert load_bearer_secret() == (secret, False, "keyring:FakeKeyring")
    assert not secret_file_path().exists()


def test_keyring_backend_requires_the_package(home: Path) -> None:
    with pytest.raises(RuntimeError):
        load_bearer_secret(backend="keyring")


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_config_dir_is_dot_exegol_mcp_in_the_invoking_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    """G-05-1d: ~/.exegol-mcp on every OS, whatever XDG_CONFIG_HOME and APPDATA say."""
    invoker = tmp_path / "invoker"
    monkeypatch.setattr(secret_store, "invoking_user_home", lambda: invoker)
    monkeypatch.setattr(secret_store, "invoking_user_env", lambda name: str(tmp_path / "custom-config"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "custom-config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "custom-config"))
    monkeypatch.setattr(platform, "system", lambda: system)

    assert CONFIG_DIR_NAME == ".exegol-mcp"
    assert user_config_dir() == invoker / ".exegol-mcp"
    assert secret_file_path() == invoker / ".exegol-mcp" / SECRET_FILE_NAME


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Linux", Path(".config") / APP_DIR_NAME),
        ("Darwin", Path("Library") / "Application Support" / APP_DIR_NAME),
        ("Windows", Path("AppData") / "Roaming" / APP_DIR_NAME),
    ],
)
def test_platform_config_dir_is_the_pre_g_05_1d_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_platform_config_dir: Any,
    system: str,
    expected: Path,
) -> None:
    """The move source is exactly the dir earlier versions used, under the invoking user's home."""
    invoker = tmp_path / "invoker"
    monkeypatch.setattr(secret_store, "invoking_user_home", lambda: invoker)
    monkeypatch.setattr(secret_store, "invoking_user_env", lambda name: None)
    monkeypatch.setattr(platform, "system", lambda: system)

    assert real_platform_config_dir() == invoker / expected


@pytest.mark.parametrize(("system", "variable"), [("Linux", "XDG_CONFIG_HOME"), ("Windows", "APPDATA")])
def test_platform_config_dir_honours_the_invoking_user_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_platform_config_dir: Any, system: str, variable: str
) -> None:
    """The old dir follows the invoker's XDG_CONFIG_HOME or APPDATA, as user_config_dir() did before G-05-1d."""
    custom = tmp_path / "custom-config"
    monkeypatch.setattr(secret_store, "invoking_user_home", lambda: tmp_path / "invoker")
    monkeypatch.setattr(secret_store, "invoking_user_env", lambda name: str(custom) if name == variable else None)
    monkeypatch.setattr(platform, "system", lambda: system)

    assert real_platform_config_dir() == custom / APP_DIR_NAME
    assert user_config_dir() == tmp_path / "invoker" / CONFIG_DIR_NAME


@posix_only
def test_created_paths_are_handed_back_to_the_invoking_user_under_sudo(
    sudo_as_root: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    _, created, location = load_bearer_secret(backend="file")

    config = sudo_as_root / CONFIG_DIR_NAME
    secret_path = config / SECRET_FILE_NAME
    assert created is True
    assert location == str(secret_path)
    # ~/.exegol-mcp did not exist: it was created by this run, so it is handed back too, not left root-owned.
    created_inodes = {config.stat().st_ino, secret_path.stat().st_ino}
    assert {inode for inode, _, _ in fchown_calls} == created_inodes
    assert sudo_as_root.stat().st_ino not in created_inodes  # the home existed: never chowned
    assert {(uid, gid) for _, uid, gid in fchown_calls} == {(os.geteuid(), os.getegid())}
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.stat().st_mode) == 0o700


@posix_only
def test_no_chown_when_running_as_the_invoking_user(home: Path, fchown_calls: list[tuple[int, int, int]]) -> None:
    load_bearer_secret(backend="file")
    assert fchown_calls == []


@posix_only
def test_chown_failure_is_best_effort(sudo_as_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any) -> None:
        raise PermissionError("not root")

    monkeypatch.setattr(os, "fchown", refuse)

    secret, created, location = load_bearer_secret(backend="file")
    assert created is True
    assert location == str(secret_file_path())
    assert load_bearer_secret(backend="file")[0] == secret


@posix_only
def test_sudo_run_adopts_a_secret_v1_stranded_in_root_home(
    sudo_as_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fchown_calls: list[tuple[int, int, int]]
) -> None:
    monkeypatch.setattr(secret_store, "_legacy_owner_uid", os.geteuid)  # the test cannot create root-owned files
    legacy = tmp_path / "root" / ".config" / APP_DIR_NAME / SECRET_FILE_NAME
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(bytes(c ^ 0x42 for c in b"v1-sudo-secret"))

    secret, created, location = load_bearer_secret(backend="file")

    assert (secret, created) == ("v1-sudo-secret", False)
    assert location == str(sudo_as_root / CONFIG_DIR_NAME / SECRET_FILE_NAME)
    assert load_bearer_secret(backend="file")[0] == "v1-sudo-secret"  # now read from the invoking user's dir
    assert legacy.is_file()  # root's copy is left in place
    assert Path(location).stat().st_ino in {inode for inode, _, _ in fchown_calls}
    assert legacy.stat().st_ino not in {inode for inode, _, _ in fchown_calls}


@posix_only
def test_a_symlinked_bearer_secret_is_never_read_replaced_or_chowned(
    sudo_as_root: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]], caplog: pytest.LogCaptureFixture
) -> None:
    victim = tmp_path / "victim"
    victim.write_bytes(bytes(c ^ 0x42 for c in b"root-only-content"))
    victim.chmod(0o640)
    before = _snapshot(victim)
    secret_path = sudo_as_root / CONFIG_DIR_NAME / SECRET_FILE_NAME
    secret_path.parent.mkdir(parents=True, mode=0o700)
    secret_path.symlink_to(victim)

    with caplog.at_level("ERROR"):
        secret, created, location = load_bearer_secret(backend="file")

    assert secret != "root-only-content"
    assert (created, location) == (True, None)
    assert _snapshot(victim) == before  # not read into the secret, not truncated, not chmodded
    assert secret_path.is_symlink() and os.readlink(secret_path) == str(victim)
    assert victim.stat().st_ino not in {inode for inode, _, _ in fchown_calls}
    assert str(secret_path) in caplog.text and "symbolic link" in caplog.text
    assert [entry.name for entry in secret_path.parent.iterdir()] == [SECRET_FILE_NAME]  # no temp file left


@posix_only
def test_a_symlinked_config_dir_is_refused(
    sudo_as_root: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    (sudo_as_root / CONFIG_DIR_NAME).symlink_to(target, target_is_directory=True)

    _, created, location = load_bearer_secret(backend="file")

    assert (created, location) == (True, None)
    assert list(target.iterdir()) == []
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert target.stat().st_ino not in {inode for inode, _, _ in fchown_calls}


@posix_only
def test_a_hard_linked_bearer_secret_is_not_read_when_acting_for_another_user(
    sudo_as_root: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    victim = tmp_path / "victim"
    victim.write_bytes(bytes(c ^ 0x42 for c in b"root-only-content"))
    victim.chmod(0o600)
    secret_path = sudo_as_root / CONFIG_DIR_NAME / SECRET_FILE_NAME
    secret_path.parent.mkdir(parents=True, mode=0o700)
    os.link(victim, secret_path)
    assert victim.stat().st_nlink == 2
    before = _snapshot(victim)

    secret, created, _ = load_bearer_secret(backend="file")

    assert secret != "root-only-content"
    assert created is True
    assert _snapshot(victim) == before  # the entry was replaced by name; the shared inode was never written
    assert victim.stat().st_ino not in {inode for inode, _, _ in fchown_calls}


@posix_only
def test_a_config_dir_owned_by_another_user_is_refused(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fchown_calls: list[tuple[int, int, int]]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "root"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(secret_store, "_effective_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "is_sudo_context", lambda: True)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: os.geteuid() + 4242)
    monkeypatch.setattr(secret_store, "invoking_user_gid", lambda: 4343)
    config_dir = home / CONFIG_DIR_NAME  # owned by the test's uid, not the invoking user's
    config_dir.mkdir(parents=True, mode=0o755)
    config_dir.chmod(0o755)

    _, created, location = load_bearer_secret(backend="file")

    assert (created, location) == (True, None)
    assert list(config_dir.iterdir()) == []
    assert stat.S_IMODE(config_dir.stat().st_mode) == 0o755  # a foreign-owned dir is never chmodded
    assert fchown_calls == []


@posix_only
def test_a_symlinked_update_cache_is_never_read_replaced_or_chowned(
    sudo_as_root: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    now = 1_000_000.0
    victim = tmp_path / "victim.json"
    victim.write_text(json.dumps({"checked_at": now, "latest_version": "9.9.9"}))  # a fresh, valid cache payload
    victim.chmod(0o640)
    before = _snapshot(victim)
    cache = sudo_as_root / CONFIG_DIR_NAME / versions.UPDATE_CACHE_FILE
    cache.parent.mkdir(parents=True, mode=0o700)
    cache.symlink_to(victim)
    fetched: list[float] = []

    def pypi(timeout: float) -> str | None:
        fetched.append(timeout)
        return "2.0.0"

    notice = versions.UpdateChecker(enabled=True, current="2.0.0", fetch=pypi, now=lambda: now).check()

    assert len(fetched) == 1  # the victim was not read as the cache
    assert notice is None
    assert _snapshot(victim) == before
    assert cache.is_symlink() and os.readlink(cache) == str(victim)
    assert victim.stat().st_ino not in {inode for inode, _, _ in fchown_calls}


@posix_only
def test_write_user_config_file_refuses_a_symlinked_destination(
    sudo_as_root: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    victim = tmp_path / "victim"
    victim.write_bytes(b"root:x:0:0:root:/root:/bin/sh\n")
    before = _snapshot(victim)
    cache = sudo_as_root / CONFIG_DIR_NAME / versions.UPDATE_CACHE_FILE
    cache.parent.mkdir(parents=True, mode=0o700)
    cache.symlink_to(victim)

    with pytest.raises(OSError):
        secret_store.write_user_config_file(versions.UPDATE_CACHE_FILE, b'{"latest_version": "x"}')
    with pytest.raises(OSError):
        secret_store.read_user_config_file(versions.UPDATE_CACHE_FILE)

    assert _snapshot(victim) == before
    assert cache.is_symlink()
    assert victim.stat().st_ino not in {inode for inode, _, _ in fchown_calls}
    assert sorted(entry.name for entry in cache.parent.iterdir()) == [versions.UPDATE_CACHE_FILE]


@posix_only
def test_only_paths_created_by_the_call_are_handed_back(
    sudo_as_root: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    config = sudo_as_root  # the home: pre-existing, as on every host, never chowned nor chmodded
    config.chmod(0o755)

    first = secret_store.write_user_config_file("probe.json", b"{}")

    directory = config / CONFIG_DIR_NAME
    assert first == directory / "probe.json"
    assert [inode for inode, _, _ in fchown_calls] == [directory.stat().st_ino, first.stat().st_ino]
    assert config.stat().st_ino not in {inode for inode, _, _ in fchown_calls}
    assert stat.S_IMODE(config.stat().st_mode) == 0o755

    fchown_calls.clear()
    second = secret_store.write_user_config_file("probe.json", b'{"again": true}')

    assert [inode for inode, _, _ in fchown_calls] == [second.stat().st_ino]  # only the new file, not the dir
    assert secret_store.read_user_config_file("probe.json") == b'{"again": true}'


def test_no_adoption_outside_sudo(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_home = tmp_path / "root"
    monkeypatch.setenv("HOME", str(root_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    legacy = root_home / CONFIG_DIR_NAME / SECRET_FILE_NAME
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(bytes(c ^ 0x42 for c in b"someone-else"))

    secret, created, _ = load_bearer_secret(backend="file")
    assert created is True
    assert secret != "someone-else"


# --------------- Unresolved sudo (macOS): the file store and the update cache are refused ---------------
def _recording_fetch(fetched: list[float]) -> Any:
    def pypi(timeout: float) -> str | None:
        fetched.append(timeout)
        return "2.0.0"

    return pypi


@posix_only
@pytest.mark.parametrize(
    ("system", "euid", "sudo_uid", "expected"),
    [
        ("darwin", 0, "501", True),
        ("darwin", 0, "0", True),
        ("darwin", 0, None, False),
        ("darwin", 501, "501", False),
        ("linux", 0, "501", False),
    ],
)
def test_is_unresolved_sudo_follows_exegol_sudo_info(
    monkeypatch: pytest.MonkeyPatch, system: str, euid: int, sudo_uid: str | None, expected: bool
) -> None:
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(os, "geteuid", lambda: euid)
    if sudo_uid is None:
        monkeypatch.delenv("SUDO_UID", raising=False)
    else:
        monkeypatch.setenv("SUDO_UID", sudo_uid)

    assert sudo.is_unresolved_sudo() is expected


@posix_only
def test_unresolved_sudo_never_reads_or_chmods_a_hard_linked_bearer_secret(
    macos_sudo: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]], caplog: pytest.LogCaptureFixture
) -> None:
    config_dir = macos_sudo / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, mode=0o700)
    victim = tmp_path / "victim"
    victim.write_bytes(bytes(c ^ 0x42 for c in b"root-only-content"))
    victim.chmod(0o644)
    os.link(victim, config_dir / SECRET_FILE_NAME)
    assert victim.stat().st_nlink == 2
    before = _snapshot(victim)

    with caplog.at_level("ERROR"):
        secret, created, location = load_bearer_secret(backend="file")

    assert secret != "root-only-content"
    assert (created, location) == (True, None)
    assert _snapshot(victim) == before  # not read into the secret, not chmodded from 0o644, not replaced
    assert (config_dir / SECRET_FILE_NAME).stat().st_ino == victim.stat().st_ino
    assert sorted(entry.name for entry in config_dir.iterdir()) == [SECRET_FILE_NAME]
    assert fchown_calls == []
    assert "cannot be resolved" in caplog.text and "without sudo" in caplog.text


@posix_only
def test_unresolved_sudo_never_reads_or_chmods_a_hard_linked_update_cache(
    macos_sudo: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    now = 1_000_000.0
    config_dir = macos_sudo / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, mode=0o700)
    victim = tmp_path / "victim.json"
    victim.write_text(json.dumps({"checked_at": now, "latest_version": "9.9.9"}))  # a fresh, valid cache payload
    victim.chmod(0o644)
    os.link(victim, config_dir / versions.UPDATE_CACHE_FILE)
    before = _snapshot(victim)
    fetched: list[float] = []

    checker = versions.UpdateChecker(enabled=True, current="2.0.0", fetch=_recording_fetch(fetched), now=lambda: now)
    notice = checker.check()

    assert notice is None
    assert len(fetched) == 1  # the victim was not read as the cache
    assert _snapshot(victim) == before
    assert (config_dir / versions.UPDATE_CACHE_FILE).stat().st_ino == victim.stat().st_ino
    assert sorted(entry.name for entry in config_dir.iterdir()) == [versions.UPDATE_CACHE_FILE]
    assert fchown_calls == []


@posix_only
def test_unresolved_sudo_creates_no_config_dir_in_the_user_home(
    macos_sudo: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    _, _, location = load_bearer_secret(backend="file")
    fetched: list[float] = []
    versions.UpdateChecker(enabled=True, current="2.0.0", fetch=_recording_fetch(fetched), now=lambda: 1.0).check()

    assert location is None
    assert len(fetched) == 1
    assert not (macos_sudo / CONFIG_DIR_NAME).exists()
    assert fchown_calls == []


@posix_only
def test_unresolved_sudo_keeps_the_keyring_backend(macos_sudo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeKeyring()
    monkeypatch.setattr(secret_store, "keyring", fake)

    secret, created, location = load_bearer_secret()

    assert (created, location) == (True, "keyring:FakeKeyring")
    assert fake.store == {(KEYRING_SERVICE, KEYRING_USERNAME): secret}
    assert not (macos_sudo / CONFIG_DIR_NAME).exists()


# --------------- Running as root: ownership and single-link checks whatever the invoker ---------------
@posix_only
def test_running_as_root_is_privileged_whatever_the_invoker(monkeypatch: pytest.MonkeyPatch) -> None:
    for effective, invoker, expected in [(0, 0, True), (0, 1000, True), (1000, 1000, False)]:
        monkeypatch.setattr(secret_store, "_effective_uid", lambda effective=effective: effective)
        monkeypatch.setattr(secret_store, "invoking_user_uid", lambda invoker=invoker: invoker)
        assert secret_store._privileged() is expected, (effective, invoker)


@posix_only
def test_running_as_root_with_sudo_uid_zero_has_no_invoker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "0")

    assert sudo.invoking_user_uid() == 0
    assert sudo.is_unresolved_sudo() is False
    assert secret_store._privileged() is True


@posix_only
def test_running_as_root_without_an_invoker_never_reads_or_chmods_a_hard_linked_bearer_secret(
    root_without_invoker: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    config_dir = root_without_invoker / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, mode=0o700)
    victim = tmp_path / "victim"
    victim.write_bytes(bytes(c ^ 0x42 for c in b"root-only-content"))
    victim.chmod(0o644)
    os.link(victim, config_dir / SECRET_FILE_NAME)
    assert victim.stat().st_nlink == 2
    before = _snapshot(victim)

    secret, created, location = load_bearer_secret(backend="file")

    assert secret != "root-only-content"
    assert (created, location) == (True, None)
    assert _snapshot(victim) == before  # not read into the secret, not chmodded from 0o644, not replaced
    assert (config_dir / SECRET_FILE_NAME).stat().st_ino == victim.stat().st_ino
    assert sorted(entry.name for entry in config_dir.iterdir()) == [SECRET_FILE_NAME]
    assert fchown_calls == []


@posix_only
def test_running_as_root_without_an_invoker_never_reads_or_chmods_a_hard_linked_update_cache(
    root_without_invoker: Path, tmp_path: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    now = 1_000_000.0
    config_dir = root_without_invoker / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, mode=0o700)
    victim = tmp_path / "victim.json"
    victim.write_text(json.dumps({"checked_at": now, "latest_version": "9.9.9"}))  # a fresh, valid cache payload
    victim.chmod(0o644)
    os.link(victim, config_dir / versions.UPDATE_CACHE_FILE)
    before = _snapshot(victim)
    fetched: list[float] = []

    checker = versions.UpdateChecker(enabled=True, current="2.0.0", fetch=_recording_fetch(fetched), now=lambda: now)
    notice = checker.check()

    assert notice is None
    assert len(fetched) == 1  # the victim was not read as the cache
    assert _snapshot(victim) == before
    assert (config_dir / versions.UPDATE_CACHE_FILE).stat().st_ino == victim.stat().st_ino
    assert sorted(entry.name for entry in config_dir.iterdir()) == [versions.UPDATE_CACHE_FILE]
    assert fchown_calls == []


@posix_only
def test_running_as_root_without_an_invoker_creates_nothing_in_a_user_owned_tree(
    root_without_invoker: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    _, _, location = load_bearer_secret(backend="file")
    fetched: list[float] = []
    versions.UpdateChecker(enabled=True, current="2.0.0", fetch=_recording_fetch(fetched), now=lambda: 1.0).check()

    assert location is None
    assert len(fetched) == 1
    assert not (root_without_invoker / CONFIG_DIR_NAME).exists()
    assert fchown_calls == []


@posix_only
def test_running_as_root_refuses_a_hard_linked_file_in_its_own_config_dir(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fchown_calls: list[tuple[int, int, int]]
) -> None:
    # Root acting in its own config dir: the real uid is both the effective and the invoking uid.
    monkeypatch.setattr(secret_store, "_privileged", lambda: True)
    config_dir = user_config_dir()
    config_dir.mkdir(parents=True, mode=0o700)
    cache_victim = tmp_path / "victim.json"
    cache_victim.write_text(json.dumps({"checked_at": 1.0, "latest_version": "9.9.9"}))
    cache_victim.chmod(0o644)
    os.link(cache_victim, config_dir / versions.UPDATE_CACHE_FILE)
    secret_victim = tmp_path / "victim"
    secret_victim.write_bytes(bytes(c ^ 0x42 for c in b"root-only-content"))
    secret_victim.chmod(0o644)
    os.link(secret_victim, config_dir / SECRET_FILE_NAME)
    cache_before, secret_before = _snapshot(cache_victim), _snapshot(secret_victim)

    with pytest.raises(secret_store.UnsafeConfigPathError):
        secret_store.read_user_config_file(versions.UPDATE_CACHE_FILE)
    secret, _, _ = load_bearer_secret(backend="file")

    assert secret != "root-only-content"
    assert _snapshot(cache_victim) == cache_before
    assert _snapshot(secret_victim) == secret_before  # replaced by name at most, the shared inode never written
    assert fchown_calls == []


@posix_only
def test_running_as_root_still_reads_and_hardens_its_own_single_link_file(
    home: Path, monkeypatch: pytest.MonkeyPatch, fchown_calls: list[tuple[int, int, int]]
) -> None:
    monkeypatch.setattr(secret_store, "_privileged", lambda: True)

    secret, created, location = load_bearer_secret(backend="file")
    path = secret_file_path()
    assert (created, location) == (True, str(path))
    path.chmod(0o644)

    assert load_bearer_secret(backend="file") == (secret, False, str(path))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert fchown_calls == []


# --------------- Root-only refusal branches, each pinned on its own (D-12) ---------------
@pytest.fixture
def root_for_foreign_owner(home: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """Root acting for an invoking user who is not the test's uid, so every path the test creates reads as foreign.

    Returns the expected owner's uid.
    """
    foreign = os.geteuid() + 1
    monkeypatch.setattr(secret_store, "_effective_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: foreign)
    monkeypatch.setattr(secret_store, "invoking_user_gid", lambda: None)
    return foreign


@posix_only
def test_privileged_create_refuses_a_foreign_owned_ancestor(
    root_for_foreign_owner: int, home: Path, fchown_calls: list[tuple[int, int, int]]
) -> None:
    # The config dir does not exist; its deepest existing ancestor (the home) belongs to the test's uid.
    with pytest.raises(secret_store.UnsafeConfigPathError) as refused:
        secret_store.write_user_config_file("probe.json", b"{}")

    assert refused.value.errno == errno.EPERM
    assert refused.value.filename == str(home)
    assert list(home.iterdir()) == []  # nothing was created under the foreign-owned ancestor
    assert fchown_calls == []


@posix_only
def test_privileged_open_refuses_a_foreign_owned_config_dir(
    root_for_foreign_owner: int, fchown_calls: list[tuple[int, int, int]]
) -> None:
    config_dir = user_config_dir()
    config_dir.mkdir(parents=True, mode=0o755)
    config_dir.chmod(0o755)
    (config_dir / "probe.json").write_bytes(b"{}")

    for call in (
        lambda: secret_store.read_user_config_file("probe.json"),
        lambda: secret_store.write_user_config_file("probe.json", b'{"new": true}'),
    ):
        with pytest.raises(secret_store.UnsafeConfigPathError) as refused:
            call()
        assert refused.value.errno == errno.EPERM
        assert refused.value.filename == str(config_dir)

    assert stat.S_IMODE(config_dir.stat().st_mode) == 0o755  # a foreign-owned dir is never chmodded
    assert (config_dir / "probe.json").read_bytes() == b"{}"
    assert sorted(entry.name for entry in config_dir.iterdir()) == ["probe.json"]
    assert fchown_calls == []


@posix_only
def test_privileged_read_refuses_a_file_with_two_hard_links(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fchown_calls: list[tuple[int, int, int]]
) -> None:
    # Root acting for the test's uid: dir and file pass the owner check, only the link count is wrong.
    monkeypatch.setattr(secret_store, "_effective_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: os.geteuid())
    config_dir = user_config_dir()
    config_dir.mkdir(parents=True, mode=0o700)
    victim = tmp_path / "victim.json"
    victim.write_bytes(b'{"secret": "root-only"}')
    victim.chmod(0o644)
    os.link(victim, config_dir / "probe.json")
    before = _snapshot(victim)

    with pytest.raises(secret_store.UnsafeConfigPathError) as refused:
        secret_store.read_user_config_file("probe.json")

    assert refused.value.errno == errno.EPERM
    assert refused.value.filename == str(config_dir / "probe.json")
    assert _snapshot(victim) == before  # not chmodded from 0o644
    assert fchown_calls == []


@posix_only
def test_privileged_write_refuses_a_foreign_owned_existing_file(
    home: Path, monkeypatch: pytest.MonkeyPatch, fchown_calls: list[tuple[int, int, int]]
) -> None:
    # The dir passes the owner check (it is the test's uid's, the expected owner); the existing target is made to
    # read as someone else's through lstat, the one call the store makes on it before writing.
    monkeypatch.setattr(secret_store, "_effective_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: os.geteuid())
    config_dir = user_config_dir()
    config_dir.mkdir(parents=True, mode=0o700)
    target = config_dir / "probe.json"
    target.write_bytes(b'{"kept": true}')
    target.chmod(0o600)
    before = _snapshot(target)
    real_lstat = os.lstat

    def lstat(path: Any, *, dir_fd: int | None = None) -> os.stat_result:
        st = real_lstat(path, dir_fd=dir_fd)
        if dir_fd is None or path != "probe.json":
            return st
        fields = list(st[:10])
        fields[stat.ST_UID] = os.geteuid() + 1
        return os.stat_result(fields)

    monkeypatch.setattr(os, "lstat", lstat)

    with pytest.raises(secret_store.UnsafeConfigPathError) as refused:
        secret_store.write_user_config_file("probe.json", b'{"overwritten": true}')

    assert refused.value.errno == errno.EPERM
    assert refused.value.filename == str(target)
    assert _snapshot(target) == before  # the file keeps its bytes, mode and inode
    assert sorted(entry.name for entry in config_dir.iterdir()) == ["probe.json"]  # no temp file was created
    assert fchown_calls == []


@posix_only
@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_read_refuses_a_non_regular_file(home: Path, kind: str) -> None:
    config_dir = user_config_dir()
    config_dir.mkdir(parents=True, mode=0o700)
    entry = config_dir / "probe.json"
    if kind == "directory":
        entry.mkdir()
    else:
        os.mkfifo(entry)  # opened O_NONBLOCK, so the read never waits for a writer

    with pytest.raises(secret_store.UnsafeConfigPathError) as refused:
        secret_store.read_user_config_file("probe.json")

    assert refused.value.errno == errno.EINVAL
    assert refused.value.filename == str(entry)


@posix_only
def test_failed_write_leaves_no_temp_file(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = secret_store.write_user_config_file("probe.json", b"first")
    config_dir = first.parent

    def failing_replace(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.EIO, "simulated replace failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError) as failed:
        secret_store.write_user_config_file("probe.json", b"second")

    assert failed.value.errno == errno.EIO
    assert sorted(entry.name for entry in config_dir.iterdir()) == ["probe.json"]  # no .probe.json.*.tmp left
    assert first.read_bytes() == b"first"


_POSIX_ONLY_HELPERS: list[tuple[str, Any]] = [
    ("_hand_back", lambda tmp: secret_store._hand_back(-1, tmp)),
    ("_open_dir_nofollow", lambda tmp: secret_store._open_dir_nofollow(tmp)),
    ("_create_config_dir", lambda tmp: secret_store._create_config_dir(tmp / "new")),
    ("_open_config_dir", lambda tmp: secret_store._open_config_dir(tmp, create=True)),
    ("_read_private_file", lambda tmp: secret_store._read_private_file(tmp, "probe.json", 0)),
    ("_write_private_file", lambda tmp: secret_store._write_private_file(tmp, "probe.json", b"{}")),
    ("_read_private_file_and_stat", lambda tmp: secret_store._read_private_file_and_stat(tmp, "probe.json", 0)),
    (
        "_create_private_file_no_replace",
        lambda tmp: secret_store._create_private_file_no_replace(tmp, "probe.json", b"{}"),
    ),
    ("_remove_moved_file", lambda tmp: secret_store._remove_moved_file(tmp, "probe.json", os.stat_result((0,) * 10))),
    ("_move_posix", lambda tmp: secret_store._move_posix(tmp, tmp / "new", "probe.json")),
]


@pytest.mark.parametrize(("helper", "call"), _POSIX_ONLY_HELPERS, ids=[name for name, _ in _POSIX_ONLY_HELPERS])
def test_posix_helpers_refuse_on_win32(home: Path, tmp_path: Path, helper: str, call: Any) -> None:
    touched: list[str] = []

    def forbidden(name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"{helper} touched the filesystem through os.{name} on win32")

        return record

    with pytest.MonkeyPatch.context() as m:
        m.setattr(sys, "platform", "win32")
        for name in ("open", "mkdir", "fstat", "lstat", "stat", "fchown", "fchmod", "replace", "link", "unlink", "rmdir"):
            m.setattr(os, name, forbidden(name), raising=False)
        m.setattr(os.path, "lexists", forbidden("path.lexists"))
        m.setattr(secret_store, "_acting_for_other_user", lambda: True)  # _hand_back would otherwise return early
        with pytest.raises(OSError) as refused:
            call(tmp_path)

    assert refused.value.errno == errno.ENOSYS
    assert refused.traceback[-1].name == helper  # raised by the helper's own guard, not by a callee's
    assert touched == []
