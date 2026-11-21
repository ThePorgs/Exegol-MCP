"""Startup banner (D-10), server_info data (D-13) and the background update check (D-11).

The banner looks like `exegol info`: exegol-mcp version, bundled Exegol library version, host wrapper version when
it can be detected, Docker status and config dirs. It goes to the console (stdout) in HTTP mode and to the
standard error stream only in stdio mode, where stdout carries the protocol (SEC-08).

A wrapper/library version difference is information, never a warning or a failure (D-07/D-12): the banner and
server_info show it as a compatibility state.

The update check asks PyPI for the latest exegol-mcp release in a daemon thread, with a short timeout and a cache
in exegol-mcp's config dir. It never blocks or fails startup, swallows every error (offline, air-gapped, proxy),
and does nothing at all with --no-update-check.
"""

import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
import sysconfig
import threading
import time
import urllib.request
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from exegol_mcp import __version__
from exegol_mcp.bootstrap import secret_store
from exegol_mcp.bootstrap.cli import HTTP_HOST, MCP_PATH
from exegol_mcp.services.server_info import VersionReport, compatibility_state

PYPI_JSON_URL = "https://pypi.org/pypi/exegol-mcp/json"
UPDATE_CHECK_TIMEOUT_SECONDS = 3.0
UPDATE_CACHE_FILE = "update-check.json"
UPDATE_CACHE_TTL_SECONDS = 24 * 60 * 60
# The PyPI JSON for exegol-mcp is a few kB; refuse to buffer anything absurd.
_MAX_PYPI_RESPONSE_BYTES = 2 * 1024 * 1024

_log = logging.getLogger(__name__)

# The release number, plus a PEP 440 alpha/beta/rc tag, which ranks below its own final release (2.0.0b1 < 2.0.0).
_RELEASE = re.compile(r"^v?(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?")
_VERSION_LINE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


# --------------- Versions ---------------
@dataclass(frozen=True)
class Versions:
    """What the banner and server_info report about this installation"""

    exegol_mcp: str
    exegol_library: str | None
    exegol_wrapper: str | None
    docker_status: str
    config_dirs: list[str] = field(default_factory=list)

    @property
    def compatibility(self) -> str:
        """'match', 'mismatch' (information only) or 'unknown', as server_info reports it."""
        return compatibility_state(self.exegol_library, self.exegol_wrapper)


def library_version() -> str | None:
    """Version of the Exegol library bundled with exegol-mcp."""
    try:
        return importlib.metadata.version("exegol")
    except importlib.metadata.PackageNotFoundError:
        return None


def _dist_info_version(environment: Path) -> str | None:
    """Read the exegol version from an environment's site-packages metadata, without importing anything."""
    for metadata_file in [
        *environment.glob("lib/python*/site-packages/exegol-*.dist-info/METADATA"),
        *environment.glob("Lib/site-packages/exegol-*.dist-info/METADATA"),
    ]:
        name = version = None
        for line in metadata_file.read_text(encoding="utf-8", errors="replace").splitlines()[:20]:
            if line.startswith("Name:"):
                name = line.partition(":")[2].strip().lower()
            elif line.startswith("Version:"):
                version = line.partition(":")[2].strip()
        if name == "exegol" and version:
            return version
    return None


def detect_wrapper_version(which: Callable[[str], str | None] = shutil.which) -> str | None:
    """Version of the Exegol wrapper on the PATH, when it can be read from files (best effort, never runs it).

    Covers the pip/pipx/uv installs (the dist-info next to the wrapper's interpreter or script) and a git clone
    (exegol.py next to exegol/config/ConstantConfig.py). Returns None when the wrapper is absent or unreadable.
    """
    try:
        found = which("exegol")
        if not found:
            return None
        script = Path(found).resolve()
        clone_constants = script.parent / "exegol" / "config" / "ConstantConfig.py"
        if clone_constants.is_file():
            match = _VERSION_LINE.search(clone_constants.read_text(encoding="utf-8", errors="replace"))
            return match.group(1) if match else None
        environments = [script.parent.parent]
        with script.open("rb") as handle:
            first_line = handle.readline(1024)
        if first_line.startswith(b"#!"):
            interpreter = first_line[2:].decode("utf-8", errors="replace").strip().split(" ")[0]
            if interpreter and Path(interpreter).is_absolute():
                environments.insert(0, Path(interpreter).parent.parent)
        for environment in dict.fromkeys(environments):
            version = _dist_info_version(environment)
            if version:
                return version
    except (OSError, ValueError) as e:
        _log.debug(f"Could not detect the Exegol wrapper version: {e}")
    return None


def gather_versions(docker_status: str, exegol_config_dir: str) -> Versions:
    """Collect the banner and server_info facts (file reads only: no network, no Docker call).

    Args:
        docker_status: The status the readiness gate established.
        exegol_config_dir: The Exegol config directory the bundled library uses.
    """
    return Versions(
        exegol_mcp=__version__,
        exegol_library=library_version(),
        exegol_wrapper=detect_wrapper_version(),
        docker_status=docker_status,
        config_dirs=[exegol_config_dir, str(secret_store.user_config_dir())],
    )


def server_info_report(versions: Versions, docker_status: str, update_notice: str | None) -> VersionReport:
    """The data the server_info tool reports (D-13): the banner facts, a live Docker status and the update notice.

    The compatibility state is derived from these versions by the server_info service, as information only.
    """
    return VersionReport(
        exegol_mcp_version=versions.exegol_mcp,
        exegol_library_version=versions.exegol_library,
        exegol_wrapper_version=versions.exegol_wrapper,
        docker_status=docker_status,
        config_dirs=list(versions.config_dirs),
        update_notice=update_notice,
    )


# --------------- Banner ---------------
def banner_console(args: Namespace) -> Console:
    """The console the banner goes to: stdout in HTTP mode, the standard error stream only in stdio mode."""
    if args.type == "stdio":
        # stdout is the MCP protocol stream in stdio mode (SEC-08).
        return Console(stderr=True)
    return Console()


def _endpoint(args: Namespace) -> str:
    if args.type == "stdio":
        return "stdio"
    return f"http://{HTTP_HOST}:{args.port}{MCP_PATH} ({args.auth} auth)"


def _wrapper_line(versions: Versions) -> str:
    if versions.exegol_wrapper is None:
        return "not detected"
    if versions.compatibility == "mismatch":
        return f"{versions.exegol_wrapper} (differs from the bundled library: information only)"
    return versions.exegol_wrapper


def print_banner(args: Namespace, versions: Versions, console: Console | None = None) -> None:
    """Print the startup banner (D-10) on the console for this transport (stderr only in stdio mode)."""
    console = console or banner_console(args)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    rows = [
        ("Exegol library", versions.exegol_library or "unknown"),
        ("Exegol wrapper", _wrapper_line(versions)),
        ("Docker", versions.docker_status),
        ("Config dirs", ", ".join(versions.config_dirs)),
        ("Endpoint", _endpoint(args)),
        ("Update check", "disabled (--no-update-check)" if args.no_update_check else "in the background"),
    ]
    for label, value in rows:
        # Text, not markup: paths and versions are printed as they are.
        table.add_row(label, Text(value))
    console.print(Text(f"exegol-mcp {versions.exegol_mcp}", style="bold"))
    console.print(table)


# --------------- MCP client config ---------------
# The Python 3.10 stdio fallback (WR-16). `python -c` puts '' (the working directory) on sys.path only after site,
# sitecustomize and usercustomize have run. sys is built in, so the first statement imports nothing from that
# directory. The filter drops every empty entry wherever it sits and is a no-op without one, which is safer than
# deleting sys.path[0]: that would drop a real entry whenever no '' is there. sys.exit(main()) is the console
# script's own contract, because main() returns the exit status. The text holds no double quote and no backslash,
# so the printed JSON shows it verbatim.
_DROP_WORKING_DIRECTORY = "import sys; sys.path[:] = [entry for entry in sys.path if entry]"
_CWD_FREE_LAUNCHER = _DROP_WORKING_DIRECTORY + "; from exegol_mcp.main import main; sys.exit(main())"


def _safe_path_flag_supported() -> bool:
    """Whether this interpreter accepts -P.

    Python 3.11 added -P (PYTHONSAFEPATH), which keeps the working directory off sys.path under -m. Python 3.10,
    which requires-python still allows, rejects it with "Unknown option: -P". This is a function so tests can force
    either branch.
    """
    return sys.version_info >= (3, 11)


def _console_script() -> str | None:
    """This environment's exegol-mcp console script, or None when it has none.

    Only two places count: sys.argv[0] when it is the exegol-mcp script running this process (the same
    installation), then this interpreter's own scripts directory (sysconfig.get_path("scripts")). A scripts directory
    of another scheme, such as ~/.local/bin, may hold another installation's script (pipx links there), so it is
    never searched, and neither is PATH.

    The script is returned resolved through symlinks (WR-15). CPython puts the resolved script's directory first on
    sys.path, so the printed command names the file whose directory the startup check lists at every start through
    it. A pipx or uv tool link in ~/.local/bin, a PATH directory no check lists, therefore prints the script inside
    the tool's environment. A link whose target is version-specific, such as a Homebrew Cellar or Nix store path or a
    hand-made 'current' link, is printed resolved too, so re-run --print-config after an upgrade that moves the
    script (IN-24).
    """
    names = ("exegol-mcp", "exegol-mcp.exe")
    candidates: list[str] = []
    if sys.argv and os.path.basename(sys.argv[0]) in names:
        candidates.append(sys.argv[0])
    candidates.append(os.path.join(sysconfig.get_path("scripts"), names[1] if os.name == "nt" else names[0]))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return None


def stdio_launch() -> tuple[str, list[str]]:
    """The command and arguments an MCP client runs to start this server over stdio (CR-04, WR-16).

    No launch this returns puts the working directory the MCP client spawns the server in (often the project
    directory) on sys.path, so a module a container plants there never runs on the host, before or after the
    startup check. The console script comes first, because sys.path[0] is then the script's own directory inside
    the environment. Without one, it is `python -P -m exegol_mcp` on Python 3.11+, else `python -c` with
    _CWD_FREE_LAUNCHER on 3.10, which drops the working directory before importing exegol_mcp.

    A `python -m exegol_mcp` typed by hand keeps working; only the startup check protects it, after its imports
    (T-04-96). An exegol-mcp importable only from that working directory, such as an uninstalled checkout the
    client starts in, is not reached by the fallback. This is by design (AR-04-18).
    """
    script = _console_script()
    if script is not None:
        return script, ["--type", "stdio"]
    if _safe_path_flag_supported():
        return sys.executable, ["-P", "-m", "exegol_mcp", "--type", "stdio"]
    return sys.executable, ["-c", _CWD_FREE_LAUNCHER, "--type", "stdio"]


def client_config_template(args: Namespace, secret: str | None, created: bool) -> str:
    """The MCP client config entry for this server.

    The stdio entry launches this environment's exegol-mcp console script, which keeps the MCP client's working
    directory off the server's import path (stdio_launch, CR-04); without one, it prints a python launch that also
    keeps that directory off the path (-P on 3.11+, the -c launcher on 3.10; stdio_launch, WR-16). The bearer secret
    is shown only with --print-config or right after it was created; otherwise it is redacted.
    """
    entry: dict[str, object]
    if args.type == "stdio":
        command, launch_args = stdio_launch()
        entry = {"type": "stdio", "command": command, "args": launch_args}
    else:
        entry = {"type": "http", "url": f"http://{HTTP_HOST}:{args.port}{MCP_PATH}"}
        if args.auth == "bearer":
            secret_hint = secret if args.print_config or created else "<redacted> (use --print-config to reveal)"
            entry["headers"] = {"Authorization": f"Bearer {secret_hint}"}
    return f'"exegol-mcp": {json.dumps(entry, indent=4)}'


def print_client_config(args: Namespace, secret: str | None, created: bool, console: Console | None = None) -> None:
    """Print the client config template block (at startup, or for --print-config, which then exits).

    --print-config is a CLI output path, so the block goes to stdout. At startup it follows the banner: stdout in
    HTTP mode, the standard error stream in stdio mode, where stdout is the protocol stream (SEC-08).
    """
    if console is None:
        console = Console() if args.print_config else banner_console(args)
    separator = "=" * 56
    # console.out: no markup, no highlighting and no wrapping, so the block can be copied as is.
    console.out("MCP client template:")
    console.out(separator)
    console.out(client_config_template(args, secret, created))
    console.out(separator)
    if args.type == "http" and args.auth == "bearer" and created:
        console.out("Use '--print-config' to display the secret again if needed later.")
    if args.type == "http" and args.auth == "none":
        # No-auth is reachable only through the 'security.allow_no_auth' config key now, not a CLI --auth none
        # (D-12/D-13); point the operator at the key they set.
        _log.warning(
            "Authentication is DISABLED (enabled by the 'security.allow_no_auth' config key). "
            "Use only in trusted environments."
        )


# --------------- Background update check ---------------
def fetch_latest_version(timeout: float = UPDATE_CHECK_TIMEOUT_SECONDS) -> str | None:
    """Ask PyPI for the latest exegol-mcp release (read-only JSON endpoint; nothing is ever installed)."""
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"exegol-mcp/{__version__}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # fixed https URL
        data = json.loads(response.read(_MAX_PYPI_RESPONSE_BYTES))
    version = data.get("info", {}).get("version")
    return version if isinstance(version, str) and _RELEASE.match(version) else None


def _release(version: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    match = _RELEASE.match(version.strip())
    if not match:
        return None
    release = tuple(int(part) for part in match.group(1).split("."))
    # A final release outranks every pre-release of the same number; pre-releases order as a < b < rc, then by N.
    stage = (1,) if match.group(2) is None else (0, ("a", "b", "rc").index(match.group(2)), int(match.group(3)))
    return release, stage


def is_newer(latest: str, current: str) -> bool:
    """True when latest is a strictly higher release than current; False when either cannot be parsed."""
    latest_release, current_release = _release(latest), _release(current)
    if latest_release is None or current_release is None:
        return False
    return latest_release > current_release


class UpdateChecker:
    """Best-effort "a newer exegol-mcp is on PyPI" check (D-11). check() and start() never raise."""

    def __init__(
        self,
        *,
        enabled: bool,
        current: str = __version__,
        fetch: Callable[[float], str | None] = fetch_latest_version,
        timeout: float = UPDATE_CHECK_TIMEOUT_SECONDS,
        ttl: float = UPDATE_CACHE_TTL_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.enabled = enabled
        self._current = current
        self._fetch = fetch
        self._timeout = timeout
        self._ttl = ttl
        self._now = now
        self._notice: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def notice(self) -> str | None:
        """The update notice once the check found a newer release, else None."""
        return self._notice

    def start(self) -> threading.Thread | None:
        """Run the check in a daemon thread and return at once (once per checker; no-op when disabled)."""
        with self._lock:
            if not self.enabled or self._thread is not None:
                return self._thread
            self._thread = threading.Thread(target=self.check, name="exegol-mcp-update-check", daemon=True)
            self._thread.start()
            return self._thread

    def check(self) -> str | None:
        """Run the check now, in the calling thread. Returns the notice, or None. Never raises."""
        if not self.enabled:
            return None
        try:
            latest = self._read_cache()
            if latest is None:
                latest = self._fetch(self._timeout)
                if latest:
                    self._write_cache(latest)
        except Exception as e:  # noqa: BLE001  # offline, DNS, TLS, proxy, bad JSON: never fail startup
            _log.debug(f"Update check skipped: {e}")
            return None
        if latest and is_newer(latest, self._current):
            self._notice = (
                f"exegol-mcp {latest} is available (this server runs {self._current}). "
                "Upgrade it with the tool you installed it with, for example `pipx upgrade exegol-mcp`."
            )
            _log.info(self._notice)
        return self._notice

    def _read_cache(self) -> str | None:
        # Same no-follow rules as the bearer secret (CR-01): a refused, missing or garbled cache means "fetch again".
        try:
            raw = secret_store.read_user_config_file(UPDATE_CACHE_FILE)
            if raw is None:
                return None
            data = json.loads(raw.decode("utf-8"))
            checked_at = float(data["checked_at"])
            latest = data["latest_version"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(latest, str) or not 0 <= self._now() - checked_at < self._ttl:
            return None
        return latest

    def _write_cache(self, latest: str) -> None:
        payload = json.dumps({"checked_at": self._now(), "latest_version": latest}).encode("utf-8")
        try:
            secret_store.write_user_config_file(UPDATE_CACHE_FILE, payload)
        except (OSError, ValueError) as e:
            _log.debug(f"Could not cache the update check result: {e}")
