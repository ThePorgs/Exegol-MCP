"""Read-only Exegol facts for `exegol-mcp version -vvv` (D-09), modelled on the wrapper's print_debug_banner.

The wrapper prints these as debug lines (ExegolManager.py:413-442); here they come back as strings for the `version`
report. Only ConstantConfig and EnvInfo are imported from the SDK, plus docker-py (D-19): nothing here builds an SDK
singleton, so no file or directory is created in the Exegol config directory. Every probe returns a string and never
raises, and the probes run one after another in the calling thread because the SDK's singleton metaclass is not
thread-safe (RESEARCH Pitfall 12).
"""

import contextlib
import platform
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import docker
from exegol.config.ConstantConfig import ConstantConfig
from exegol.config.EnvInfo import EnvInfo

# A short bound: `version` must stay quick on a slow daemon. host.py's tool-time ping uses 5.
_DOCKER_PROBE_TIMEOUT_SECONDS = 3

# The datastore file name and the EULA key, derived from LocalDatastore.py:16-19 instead of importing that module:
# building it creates and migrates the file (D-19). The drift tests in tests/test_version_diagnostics.py pin both.
_DATASTORE_FILE_NAME = ".datastore"
_EULA_KEY = 0

_UNKNOWN = "unknown"


@dataclass(frozen=True)
class DockerFacts:
    """What the Docker daemon answered, as report strings."""

    status: str
    engine: str
    desktop: str


@dataclass(frozen=True)
class SdkFacts:
    """The Exegol-side facts of the `version -vvv` section, as report strings."""

    exegol_config_dir: str
    exegol_config_file: str
    datastore_file: str
    eula: str
    host_os: str
    arch: str
    docker: DockerFacts


def best_effort(probe: Callable[[], str]) -> str:
    """Run a probe and return its string, or "unknown (<ErrorClass>)" when it raises (D-09).

    Args:
        probe: A function that returns the report string.

    Returns:
        The probe's string; a failing probe is shown, never raised.
    """
    try:
        return probe()
    except Exception as error:  # noqa: BLE001  # D-09: a probe is best effort, any failure is a line in the report
        return f"{_UNKNOWN} ({type(error).__name__})"


def _file_state(path: Path) -> str:
    """The path with whether a regular file is there: "<path> (present)" or "<path> (absent)" (D-28)."""
    return f"{path} ({'present' if path.is_file() else 'absent'})"


def read_eula_state(db_path: Path) -> str:
    """Read the EULA flag of the Exegol datastore without creating or changing anything (D-19, D-23).

    Args:
        db_path: The datastore file.

    Returns:
        "accepted", "not accepted", "not accepted (no datastore)" or "unknown (<ErrorClass>)".
    """
    if not db_path.is_file():
        return "not accepted (no datastore)"
    # A plain mode=ro open of a WAL database with no open writer creates the -wal and -shm files (root-owned under
    # sudo) and fails in a read-only directory, so immutable is the default (RESEARCH Pitfall 4). When a writer has
    # the database open its -wal and -shm pair exists: committed rows may live only in the log, which an immutable
    # open would not see, and mode=ro reuses the pair and creates nothing (D-23). A lone -wal is a crashed writer's
    # leftover: a non-immutable open would create the -shm, so it stays on the immutable path.
    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    query = "mode=ro" if wal.exists() and shm.exists() else "mode=ro&immutable=1"
    try:
        with contextlib.closing(sqlite3.connect(f"{db_path.as_uri()}?{query}", uri=True)) as connection:
            row = connection.execute("SELECT value FROM kv WHERE key = ?", (_EULA_KEY,)).fetchone()
    except sqlite3.Error as error:
        return f"{_UNKNOWN} ({type(error).__name__})"
    return "accepted" if row is not None and row[0] == "1" else "not accepted"


def probe_docker(timeout: float = _DOCKER_PROBE_TIMEOUT_SECONDS) -> DockerFacts:
    """Ask the daemon who it is, with a fresh bounded client (D-09).

    Construction itself contacts the daemon (docker-py reads the API version), so the client is built and used inside
    one guard. Docker Desktop is derived from the engine's operating system like EnvInfo.initData does, without
    changing EnvInfo.

    Args:
        timeout: The client timeout in seconds.

    Returns:
        The status, the engine ("<OperatingSystem> (<OSType>)") and "On"/"Off" for Docker Desktop; the status
        "unreachable (<ErrorClass>)" and "unknown" for the others when the daemon does not answer.
    """
    try:
        client = docker.from_env(environment=EnvInfo.get_env_overlay(EnvInfo.docker_env_names), timeout=timeout)
        try:
            info = client.info()
        finally:
            client.close()
    except Exception as error:  # noqa: BLE001  # D-09: a dead, slow or misconfigured daemon is a line, never a failure
        return DockerFacts(f"unreachable ({type(error).__name__})", _UNKNOWN, _UNKNOWN)
    version = info.get("ServerVersion")
    operating_system = str(info.get("OperatingSystem", ""))
    return DockerFacts(
        status=f"reachable (Docker {version})" if version else "reachable",
        engine=f"{operating_system or _UNKNOWN} ({info.get('OSType') or _UNKNOWN})",
        desktop="On" if operating_system.casefold() == "docker desktop" else "Off",
    )


def collect_sdk_facts() -> SdkFacts:
    """Collect the Exegol-side facts one after another, each one best effort.

    Returns:
        The facts; a probe that failed holds "unknown (<ErrorClass>)".
    """
    def config_path() -> Path:
        return Path(str(ConstantConfig.exegol_config_path))

    def config_dir() -> str:
        origin = " (from EXEGOL_HOME)" if ConstantConfig.exegol_config_forced else ""
        return f"{config_path()}{origin}"

    def host_os() -> str:
        return f"{EnvInfo.current_platform} {platform.release()}"

    def arch() -> str:
        raw = f" (raw: {EnvInfo.raw_arch})" if EnvInfo.raw_arch != EnvInfo.arch else ""
        return f"{EnvInfo.arch}{raw}"

    try:
        docker_facts = probe_docker()
    except Exception as error:  # noqa: BLE001  # D-09: covers an answer probe_docker could not read
        docker_facts = DockerFacts(f"{_UNKNOWN} ({type(error).__name__})", _UNKNOWN, _UNKNOWN)
    return SdkFacts(
        exegol_config_dir=best_effort(config_dir),
        # config.yml is the wrapper's config file name (its DataFileUtils resolution), a file in the config directory.
        exegol_config_file=best_effort(lambda: _file_state(config_path() / "config.yml")),
        datastore_file=best_effort(lambda: str(config_path() / _DATASTORE_FILE_NAME)),
        eula=best_effort(lambda: read_eula_state(config_path() / _DATASTORE_FILE_NAME)),
        host_os=best_effort(host_os),
        arch=best_effort(arch),
        docker=docker_facts,
    )
