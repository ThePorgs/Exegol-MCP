"""The read-only Exegol probes behind `version -vvv` (D-09, D-19, D-23). Docker-free: no test reads the real ~/.exegol.

The datastore is a scratch sqlite file; Docker is a dead socket or a fake client.
"""

import inspect
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from exegol.config.ConstantConfig import ConstantConfig
from exegol.utils.LocalDatastore import LocalDatastore

from exegol_mcp.adapter import diagnostics
from exegol_mcp.adapter.diagnostics import DockerFacts, SdkFacts

DEAD_DOCKER = "unix:///nonexistent/docker.sock"


def _make_datastore(path: Path, eula: str | None) -> None:
    """A closed WAL datastore with the SDK's kv table and, when eula is given, the EULA row."""
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS kv (key SMALLINT NOT NULL UNIQUE, value TEXT NOT NULL)")
        if eula is not None:
            connection.execute("INSERT INTO kv VALUES (?, ?)", (diagnostics._EULA_KEY, eula))
        connection.commit()
    finally:
        connection.close()  # the last connection checkpoints and removes the -wal and -shm files


@contextmanager
def _read_only_directory(directory: Path) -> Iterator[None]:
    original = directory.stat().st_mode
    directory.chmod(0o500)
    try:
        yield
    finally:
        directory.chmod(stat.S_IMODE(original))  # so the tmp cleanup works


def _listing(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def test_the_eula_read_leaves_no_file_in_a_read_only_directory(tmp_path: Path) -> None:
    directory = tmp_path / "exegol"
    directory.mkdir()
    db_path = directory / ".datastore"
    _make_datastore(db_path, "1")
    before = _listing(directory)

    with _read_only_directory(directory):
        state = diagnostics.read_eula_state(db_path)

    assert state == "accepted"
    assert _listing(directory) == before == [".datastore"]  # no -wal, no -shm


def test_the_eula_states(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    refused = tmp_path / "refused"
    empty = tmp_path / "empty"
    corrupt = tmp_path / "corrupt"
    _make_datastore(accepted, "1")
    _make_datastore(refused, "0")
    _make_datastore(empty, None)
    corrupt.write_bytes(os.urandom(4096))

    assert diagnostics.read_eula_state(accepted) == "accepted"
    assert diagnostics.read_eula_state(refused) == "not accepted"
    assert diagnostics.read_eula_state(empty) == "not accepted"
    assert diagnostics.read_eula_state(tmp_path / "missing") == "not accepted (no datastore)"
    assert diagnostics.read_eula_state(corrupt) == "unknown (DatabaseError)"
    assert not (tmp_path / "missing").exists()  # a missing datastore is not created by looking


def test_the_eula_read_sees_an_uncheckpointed_row_of_an_open_writer(tmp_path: Path) -> None:
    # D-23: with a writer holding the database open, committed rows may live only in the write-ahead log.
    db_path = tmp_path / ".datastore"
    writer = sqlite3.connect(db_path, isolation_level=None)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE kv (key SMALLINT NOT NULL UNIQUE, value TEXT NOT NULL)")
        writer.execute("INSERT INTO kv VALUES (0, '1')")
        writer_files = _listing(tmp_path)
        assert {".datastore-wal", ".datastore-shm"} <= set(writer_files)

        assert diagnostics.read_eula_state(db_path) == "accepted"
        assert _listing(tmp_path) == writer_files  # the read created nothing

        writer.execute("UPDATE kv SET value = '0' WHERE key = 0")
        assert diagnostics.read_eula_state(db_path) == "not accepted"
        assert _listing(tmp_path) == writer_files
    finally:
        writer.close()


def test_a_lone_wal_file_is_read_immutable_and_creates_nothing(tmp_path: Path) -> None:
    # D-23: a lone -wal is a crashed writer's leftover; a non-immutable open would create the -shm.
    db_path = tmp_path / ".datastore"
    _make_datastore(db_path, "1")
    (tmp_path / ".datastore-wal").write_bytes(b"")
    before = _listing(tmp_path)

    assert diagnostics.read_eula_state(db_path) == "accepted"
    assert _listing(tmp_path) == before == [".datastore", ".datastore-wal"]


def test_docker_down_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", DEAD_DOCKER)

    facts = diagnostics.probe_docker()

    assert facts.status.startswith("unreachable (")
    assert (facts.engine, facts.desktop) == ("unknown", "unknown")


class _FakeDocker:
    def __init__(self, info: dict[str, str]) -> None:
        self._info = info
        self.closed = False

    def info(self) -> dict[str, str]:
        return self._info

    def close(self) -> None:
        self.closed = True


def test_a_reachable_docker_is_described_like_the_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    fake = _FakeDocker({"ServerVersion": "29.8.2", "OperatingSystem": "Docker Desktop", "OSType": "linux"})

    def from_env(**kwargs: object) -> _FakeDocker:
        seen.update(kwargs)
        return fake

    monkeypatch.setattr(diagnostics.docker, "from_env", from_env)

    facts = diagnostics.probe_docker()

    assert facts == DockerFacts("reachable (Docker 29.8.2)", "Docker Desktop (linux)", "On")
    assert seen["timeout"] == 3
    assert fake.closed


def test_a_plain_engine_is_not_docker_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDocker({"OperatingSystem": "Ubuntu 24.04", "OSType": "linux"})
    monkeypatch.setattr(diagnostics.docker, "from_env", lambda **_kwargs: fake)

    assert diagnostics.probe_docker() == DockerFacts("reachable", "Ubuntu 24.04 (linux)", "Off")
    assert fake.closed


def test_a_failing_probe_prints_unknown_and_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ConstantConfig, "exegol_config_path", tmp_path)
    monkeypatch.setattr(diagnostics, "probe_docker", lambda: DockerFacts("reachable", "Linux (linux)", "Off"))

    def broken(_path: Path) -> str:
        raise ValueError("boom")

    monkeypatch.setattr(diagnostics, "read_eula_state", broken)

    facts = diagnostics.collect_sdk_facts()

    assert facts.eula == "unknown (ValueError)"
    assert facts.exegol_config_dir == str(tmp_path) or facts.exegol_config_dir.startswith(str(tmp_path))
    assert facts.exegol_config_file == f"{tmp_path / 'config.yml'} (absent)"
    assert facts.datastore_file == str(tmp_path / ".datastore")
    assert facts.host_os.strip() and facts.arch.strip()
    assert facts.docker == DockerFacts("reachable", "Linux (linux)", "Off")
    assert diagnostics.best_effort(lambda: "fine") == "fine"
    assert diagnostics.best_effort(lambda: 1 / 0) == "unknown (ZeroDivisionError)"


def test_the_exegol_config_file_line_states_present_or_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ConstantConfig, "exegol_config_path", tmp_path)
    monkeypatch.setattr(diagnostics, "probe_docker", lambda: DockerFacts("reachable", "Linux (linux)", "Off"))
    config = tmp_path / "config.yml"

    assert diagnostics.collect_sdk_facts().exegol_config_file == f"{config} (absent)"

    config.write_text("")
    assert diagnostics.collect_sdk_facts().exegol_config_file == f"{config} (present)"

    config.unlink()
    config.mkdir()  # only a regular file counts as the config file
    assert diagnostics.collect_sdk_facts().exegol_config_file == f"{config} (absent)"


def test_a_forced_exegol_home_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ConstantConfig, "exegol_config_path", tmp_path)
    monkeypatch.setattr(ConstantConfig, "exegol_config_forced", True)
    monkeypatch.setattr(diagnostics, "probe_docker", lambda: DockerFacts("reachable", "Linux (linux)", "Off"))

    facts = diagnostics.collect_sdk_facts()

    assert facts.exegol_config_dir == f"{tmp_path} (from EXEGOL_HOME)"
    assert isinstance(facts, SdkFacts)


def test_the_datastore_path_and_eula_key_match_the_sdk() -> None:
    # The datastore module is deliberately not imported by diagnostics (building it creates and migrates the file),
    # so what it derives is pinned here against the installed SDK.
    assert LocalDatastore.getDbPath() == ConstantConfig.exegol_config_path / diagnostics._DATASTORE_FILE_NAME
    assert LocalDatastore.Key.EULA.value == diagnostics._EULA_KEY
    source = inspect.getsource(LocalDatastore)
    assert "CREATE TABLE IF NOT EXISTS kv (key SMALLINT NOT NULL UNIQUE, value TEXT NOT NULL)" in source
    assert 'self.get(self.Key.EULA) == "1"' in source
