"""Suite-wide isolation of the config-dir move source (G-05-1d), plus the config-dir fixtures shared by the
secret-store and config-dir move tests.

exegol-mcp moves config.yml, bearer.secret and update-check.json from the platform dir earlier versions used
(~/Library/Application Support/exegol-mcp, ~/.config/exegol-mcp, %APPDATA%\\exegol-mcp) into ~/.exegol-mcp the
first time one is read and the new dir lacks it. A test that redirects only user_config_dir() would otherwise move
the developer's real files out of that old dir. The autouse guard below points the move source at a directory
inside the test's tmp_path for every test; tests that need an old file point it at their own directory.
"""

import os
import platform
from collections.abc import Callable
from pathlib import Path

import pytest

from exegol_mcp.bootstrap import secret_store

_REAL_PLATFORM_CONFIG_DIR = secret_store._platform_config_dir


@pytest.fixture(autouse=True)
def isolated_platform_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The old platform dir, as the move engine sees it: a path in tmp_path that does not exist."""
    old_dir = tmp_path / "old-platform-config-dir"
    monkeypatch.setattr(secret_store, "_platform_config_dir", lambda: old_dir)
    return old_dir


@pytest.fixture
def real_platform_config_dir() -> Callable[[], Path]:
    """The original _platform_config_dir, for the tests of its path logic (they patch what it reads)."""
    return _REAL_PLATFORM_CONFIG_DIR


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake invoking-user home, no sudo, no keyring, and no host XDG/APPDATA override."""
    user_home = tmp_path / "home"
    user_home.mkdir()
    monkeypatch.setattr(secret_store, "invoking_user_home", lambda: user_home)
    monkeypatch.setattr(secret_store, "invoking_user_env", lambda name: None)
    monkeypatch.setattr(secret_store, "is_sudo_context", lambda: False)
    monkeypatch.setattr(secret_store, "is_unresolved_sudo", lambda: False)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: os.geteuid() if os.name == "posix" else None)
    monkeypatch.setattr(secret_store, "invoking_user_gid", lambda: None)
    monkeypatch.setattr(secret_store, "keyring", None)
    return user_home


@pytest.fixture
def fchown_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int]]:
    """Record every os.fchown as (inode, uid, gid) without changing ownership (the tests do not run as root)."""
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: calls.append((os.fstat(fd).st_ino, uid, gid)))
    return calls


@pytest.fixture
def sudo_as_root(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Simulated `sudo exegol-mcp` on Linux: effective uid 0 acting for the invoking user (this test's real uid).

    Files really belong to the test's uid, which is the invoking user's, so the ownership checks pass for paths
    the "user" made and every hand-back is an fchown the fchown_calls fixture can observe. Root's own home (the
    v1 legacy location) is an empty temp dir, never the operator's real one.
    """
    root_home = tmp_path / "root"
    monkeypatch.setenv("HOME", str(root_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(secret_store, "_effective_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "is_sudo_context", lambda: True)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: os.geteuid())
    monkeypatch.setattr(secret_store, "invoking_user_gid", lambda: os.getegid())
    return home


@pytest.fixture
def macos_sudo(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Simulated `sudo exegol-mcp` on macOS, where Exegol does not resolve the invoking user.

    SudoInfo.getUserUid() falls back to the effective uid 0 and the home is the user's $HOME, which sudo keeps, so
    the config dir is home / CONFIG_DIR_NAME, in a tree the user controls.
    """
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(secret_store, "_effective_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "invoking_user_gid", lambda: 0)
    monkeypatch.setattr(secret_store, "is_sudo_context", lambda: False)
    monkeypatch.setattr(secret_store, "is_unresolved_sudo", lambda: True)
    monkeypatch.setenv("HOME", str(tmp_path / "root"))
    return home


@pytest.fixture
def root_without_invoker(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Linux root with no SUDO_UID (su -m, doas keepenv) whose HOME points into the user's tree.

    No invoker is resolved, so the expected owner is root itself; the config dir the user's tree yields
    (home / CONFIG_DIR_NAME) belongs to the test's uid, i.e. to someone other than root.
    """
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(secret_store, "_effective_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "invoking_user_uid", lambda: 0)
    monkeypatch.setattr(secret_store, "invoking_user_gid", lambda: 0)
    monkeypatch.setattr(secret_store, "is_sudo_context", lambda: False)
    monkeypatch.setattr(secret_store, "is_unresolved_sudo", lambda: False)
    monkeypatch.setenv("HOME", str(tmp_path / "root"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home
