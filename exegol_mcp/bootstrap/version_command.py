"""The read-only `version` action: what `exegol-mcp version` prints (D-07, D-08, D-25).

The report reads like the wrapper's `exegol version` (ExegolManager.print_version): the exegol-mcp version, the Exegol
library and wrapper versions, the docs link and, for a pre-release, an Alpha/Beta/release candidate notice. It goes to
stdout whatever --type says: no server runs here, so stdout is not a protocol stream and scripts and bug reports can
capture it (R1). Log records and any SDK console output stay on stderr.

The action only looks. It starts no server, runs no readiness gate, needs no EULA, and never touches what a server
start touches: the bearer secret, the config-dir adoption, mode changes, the update cache or Docker. Everything the
report reads comes from package metadata, file reads and the read-only helpers added for it. Adapter modules are
imported inside functions only, so importing this module loads no Exegol SDK module.
"""

import importlib.metadata
import json
import logging
import platform
import sys
import threading
import time
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from rich.console import Console
from rich.errors import MarkupError
from rich.text import Text

from exegol_mcp import __version__
from exegol_mcp.bootstrap import secret_store, versions
from exegol_mcp.bootstrap.cli import DOCS_URL
from exegol_mcp.bootstrap.config import CONFIG_FILE_NAME, ConfigError, parse_operator_config
from exegol_mcp.bootstrap.logging_setup import cap_third_party_logs, prefixed_line, stderr_console
from exegol_mcp.bootstrap.versions import (
    Versions,
    detect_wrapper_version,
    library_version,
    prerelease_stage,
    show_wrapper_line,
    version_headline,
    version_text,
    wrapper_text,
)
from exegol_mcp.models.config import OperatorConfig

_SDK_NOT_LOADED = "unknown (Exegol library not loaded)"
# The labels of the Exegol-side lines, printed as unknown when the SDK did not load (D-18).
_EXEGOL_LABELS = (
    "Exegol config directory",
    "Exegol config file",
    "Datastore file",
    "EULA",
    "Host OS",
    "Arch",
    "Docker",
    "Docker engine",
    "Docker desktop",
)


class _CriticalCapture(logging.Handler):
    """Keeps the last CRITICAL record, so the SDK's own message can be quoted in the report (D-22)."""

    def __init__(self) -> None:
        super().__init__(level=logging.CRITICAL)
        self.last: logging.LogRecord | None = None

    def emit(self, record: logging.LogRecord) -> None:
        self.last = record


def _plain_message(record: logging.LogRecord) -> str:
    """The SDK message of a record without its rich decoration and its "[!] " prefix."""
    message = record.getMessage()
    try:
        message = Text.from_markup(message).plain
    except MarkupError:
        pass
    return message.removeprefix("[!] ").strip()


def prerelease_notice(version: str) -> Text | None:
    """The wrapper's pre-release warning for a version, or None for a final release.

    Args:
        version: The exegol-mcp version.

    Returns:
        A "[!]" line in the wrapper's wording and colours (ExegolManager.py:405-410), or None.
    """
    stage = prerelease_stage(version)
    if stage == "a":
        return prefixed_line(
            logging.WARNING,
            "You are currently using an ",
            Text("Alpha", style="red"),
            " version of Exegol MCP, which may be unstable.",
        )
    if stage == "b":
        return prefixed_line(
            logging.WARNING,
            "You are currently using a ",
            Text("Beta", style="orange3"),
            " version of Exegol MCP, which may be unstable.",
        )
    if stage == "rc":
        return prefixed_line(
            logging.WARNING,
            "You are currently using a ",
            Text("release candidate", style="orange3"),
            " of Exegol MCP, which may still change.",
        )
    return None


def _load_sdk(args: Namespace) -> str | None:
    """Load the light part of the Exegol SDK once, guarded, and route its console and critical method.

    The SDK validates EXEGOL_HOME in a class body at import and exits on a bad value before a raise mode can be
    set, and its own critical line also reaches stderr through the root handler (D-18, D-22). A temporary CRITICAL
    handler on the root logger keeps that message for the report. This is the only SDK import of the base report, so
    a failure is reported once and every SDK-dependent part is skipped.

    Args:
        args: The parsed command line (the -v count sets the SDK's verbosity).

    Returns:
        None on success, else one worded sentence for the "[!]" line; never a bare exception class.
    """
    capture = _CriticalCapture()
    root = logging.getLogger()
    root.addHandler(capture)
    try:
        from exegol_mcp.adapter import sdk_logging  # imports exegol.utils.ExeLog only: the light subset (D-19)

        sdk_logging.route_console(stderr_console())
        sdk_logging.set_critical_raise()
        sdk_logging.set_verbosity(args.verbose)
        return None
    except KeyboardInterrupt:
        raise
    except SystemExit:
        problem = "The Exegol library stopped while loading (its error is printed above, e.g. an invalid EXEGOL_HOME)"
        if capture.last is not None:
            problem += f": {_plain_message(capture.last)}"
        return problem
    except BaseException as error:  # noqa: BLE001  # whatever the light import raises is a line, never a traceback
        detail = str(error).strip().splitlines()[0] if str(error).strip() else ""
        return f"The Exegol library could not be loaded ({type(error).__name__}{', ' + detail if detail else ''})"
    finally:
        root.removeHandler(capture)


@dataclass(frozen=True)
class _ConfigState:
    """What reading config.yml found: loaded, pending (read from the old dir), absent, invalid or unreadable."""

    config: OperatorConfig | None
    kind: str
    message: str = ""

    @property
    def problem(self) -> str | None:
        """The problem in words for the report, or None when the file loaded or is absent."""
        if self.kind == "invalid":
            return self.message
        if self.kind == "unreadable":
            return f"{CONFIG_FILE_NAME} could not be read: {self.message}"
        return None


def _operator_config_state() -> _ConfigState:
    """Read and validate config.yml without changing anything."""
    try:
        raw = secret_store.peek_user_config_file(CONFIG_FILE_NAME)
        if raw is None:
            # A config.yml still in the old platform dir is what the next server start adopts and loads, so it is read
            # from there, read-only, and its update_check opt-out holds (D-10).
            waiting = secret_store.peek_pending_adoption(CONFIG_FILE_NAME)
            if waiting is not None:
                return _ConfigState(parse_operator_config(waiting[0]), "pending", str(waiting[1]))
        return _ConfigState(parse_operator_config(raw), "absent" if raw is None else "loaded")
    except ConfigError as error:
        return _ConfigState(None, "invalid", str(error))
    except Exception as error:  # noqa: BLE001  # D-09: an OSError or a moved SDK name is a line, never a traceback
        return _ConfigState(None, "unreadable", f"{type(error).__name__}: {error}")


@dataclass(frozen=True)
class _ReportState:
    """What the base report learned that the -vvv section needs: whether the SDK loaded and how config.yml read."""

    sdk_loaded: bool
    config: _ConfigState | None = None


def _print_state(console: Console, args: Namespace) -> _ReportState:
    """Print the parts of the report that need the SDK: config.yml problems and an already-cached update notice.

    Returns:
        Whether the light SDK load succeeded and, when it did, how config.yml read.
    """
    problem = _load_sdk(args)
    if problem is not None:
        console.print(
            prefixed_line(
                logging.WARNING, Text(problem.rstrip(".")), ". Exegol MCP's config and update state are not shown."
            )
        )
        return _ReportState(sdk_loaded=False)
    state = _operator_config_state()
    report = _ReportState(sdk_loaded=True, config=state)
    if state.problem is not None:
        console.print(prefixed_line(logging.WARNING, Text(state.problem)))
    if args.verbose >= 3 or args.no_update_check or state.config is None or state.config.update_check is False:
        return report  # at -vvv the debug section reads the cache itself, through the same bounded check
    # Only a cached notice: no network call without -vvv (D-10), and nothing is written.
    checker = versions.UpdateChecker(enabled=True, read_only=True, fetch=versions.fetch_latest_version)
    update_notice = checker.cached_notice()
    if update_notice is not None:
        console.print(prefixed_line(logging.INFO, Text(update_notice)))
    return report


def _debug_line(label: str, value: str | Text) -> Text:
    """One "[D] Label: value" line of the -vvv section; the value is literal text, never markup."""
    return prefixed_line(logging.DEBUG, f"{label}: ", value)


def _switch(value: str) -> Text:
    """The wrapper's On/Off words: green for On, orange3 for Off, plain for anything else (no emoji)."""
    return Text(value, style={"On": "green", "Off": "orange3"}.get(value, ""))


def _print_exegol_facts(console: Console, sdk_loaded: bool) -> None:
    """Print the Exegol-side lines of the -vvv section (D-09, D-18)."""
    if not sdk_loaded:
        for label in _EXEGOL_LABELS:
            console.print(_debug_line(label, _SDK_NOT_LOADED))
        return
    try:
        # Inside the function: only a loaded light SDK, only at -vvv (D-19). diagnostics imports docker and more of the
        # SDK at module level, so a name that moved there surfaces here, as a line (D-09, D-18).
        from exegol_mcp.adapter import diagnostics

        facts = diagnostics.collect_sdk_facts()
    except Exception as error:  # noqa: BLE001  # D-09: whatever the import or the collection raises is "unknown"
        for label in _EXEGOL_LABELS:
            console.print(_debug_line(label, f"unknown ({type(error).__name__})"))
        return
    console.print(_debug_line("Exegol config directory", facts.exegol_config_dir))
    console.print(_debug_line("Exegol config file", facts.exegol_config_file))
    console.print(_debug_line("Datastore file", facts.datastore_file))
    console.print(_debug_line("EULA", facts.eula))
    console.print(_debug_line("Host OS", facts.host_os))
    console.print(_debug_line("Arch", facts.arch))
    console.print(_debug_line("Docker", facts.docker.status))
    console.print(_debug_line("Docker engine", facts.docker.engine))
    console.print(_debug_line("Docker desktop", _switch(facts.docker.desktop)))


def install_method(package_dir: Path, direct_url_json: str | None) -> str:
    """How exegol-mcp is installed, by the wrapper's rules (ConstantConfig's pip, pipx, uv and git flags).

    Args:
        package_dir: The directory of the imported exegol_mcp package.
        direct_url_json: The text of the distribution's PEP 610 direct_url.json, or None.

    Returns:
        "editable (<source path>)", "pipx", "uv tool", "pip", "source checkout" or "unknown".
    """
    editable = _editable_source(direct_url_json)
    if editable is not None:
        return f"editable ({editable})"
    where = package_dir.as_posix()
    if "/pipx/venvs/" in where:
        return "pipx"
    if "/uv/tools/" in where:
        return "uv tool"
    if package_dir.parent.name == "site-packages":
        return "pip"
    if (package_dir.parent / ".git").exists():
        return "source checkout"
    return "unknown"


def _editable_source(direct_url_json: str | None) -> str | None:
    """The source directory of an editable install from PEP 610 direct_url.json; malformed counts as absent."""
    if not direct_url_json:
        return None
    try:
        data = json.loads(direct_url_json)
        if data["dir_info"]["editable"] is not True:
            return None
        return unquote(urlparse(str(data["url"])).path) or None
    except (ValueError, KeyError, TypeError):
        return None


def _best_effort(probe: Callable[[], str]) -> str:
    """The probe's string, or "unknown (<ErrorClass>)": a failing probe is a line, never a traceback (D-09)."""
    try:
        return probe()
    except Exception as error:  # noqa: BLE001  # D-09: a probe is best effort, any failure is shown
        return f"unknown ({type(error).__name__})"


def _best_effort_text(probe: Callable[[], Text]) -> Text:
    """Like _best_effort for a styled value: a failing probe is the plain "unknown (<ErrorClass>)" (D-09)."""
    try:
        return probe()
    except Exception as error:  # noqa: BLE001  # D-09: a probe is best effort, any failure is shown
        return Text(f"unknown ({type(error).__name__})")


def _config_file_line(state: _ConfigState) -> str:
    path = secret_store.user_config_dir() / CONFIG_FILE_NAME
    if state.kind == "absent":
        return f"{path} (absent, built-in defaults)"
    if state.kind == "loaded":
        return f"{path} (loaded)"
    if state.kind == "pending":
        return f"{path} (absent; loaded from {state.message}, where it waits to be moved)"
    return f"{path} ({state.kind}: {state.message})"


def _print_exegol_mcp_state(console: Console, sdk_loaded: bool) -> None:
    """Print the exegol-mcp side lines that need the SDK's sudo logic: config dir, pending move, config.yml, storage."""
    if not sdk_loaded:
        for label in ("Exegol MCP config directory", "Config file", "Bearer secret storage"):
            console.print(_debug_line(label, _SDK_NOT_LOADED))
        return
    console.print(_debug_line("Exegol MCP config directory", _best_effort(lambda: str(secret_store.user_config_dir()))))
    pending = _best_effort(lambda: ", ".join(str(path) for path in secret_store.pending_adoptions()))
    if pending:
        console.print(_debug_line("Waiting to move into the config directory", pending))
    console.print(_debug_line("Config file", _best_effort(lambda: _config_file_line(_operator_config_state()))))
    console.print(_debug_line("Bearer secret storage", _best_effort(secret_store.secret_storage_location)))


def _installation() -> str:
    import exegol_mcp

    try:
        direct_url = importlib.metadata.distribution("exegol-mcp").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    return install_method(Path(exegol_mcp.__file__).resolve().parent, direct_url)


def _update_checker(args: Namespace, state: _ReportState) -> tuple[versions.UpdateChecker | None, str | None]:
    """Decide whether the -vvv update check may run (D-10, R3).

    Returns:
        (checker, None) when it may run, else (None, the reason as the report's value).
    """
    if args.no_update_check:
        return None, "disabled (--no-update-check)"
    if not state.sdk_loaded:
        return None, "not run (Exegol library not loaded)"
    if state.config is None or state.config.config is None:
        # An unreadable or invalid file cannot confirm the opt-out, so nothing is fetched.
        return None, "not run (config.yml did not load)"
    if state.config.config.update_check is False:
        return None, "disabled (update_check: false in config.yml)"
    # Read at call time, so a test can replace them. read_only: the debug report never writes the cache.
    checker = versions.UpdateChecker(
        enabled=True,
        read_only=True,
        current=__version__,
        fetch=versions.fetch_latest_version,
        timeout=versions.UPDATE_CHECK_TIMEOUT_SECONDS,
    )
    return checker, None


def _update_check_line(checker: versions.UpdateChecker, thread: threading.Thread | None, timeout: float) -> Text:
    """The value of the "Update check" line once the bounded wait is over (D-28).

    PyPI answers with the latest stable release, so a pre-release running ahead of it reads "no newer release", never
    "up to date". The PyPI version is in the version colour like every other version value.
    """
    if thread is not None and thread.is_alive():
        return Text(f"no answer within {timeout:g} s")
    if checker.error is not None:
        return Text(f"failed ({checker.error})")
    latest = checker.latest
    if latest is None:
        return Text("no answer")
    if versions.is_newer(latest, __version__):
        return Text.assemble("update available: ", version_text(latest))
    return Text.assemble("no newer release on PyPI (latest stable: ", version_text(latest), ")")


def _print_debug_section(console: Console, args: Namespace, state: _ReportState) -> None:
    """Print the -vvv "[D] Label: value" section, modelled on the wrapper's print_debug_banner (D-09, D-10)."""
    checker, skipped = _update_checker(args, state)
    # The check starts before any probe and is waited on for at most its own timeout, counted from its start: urllib's
    # timeout does not bound DNS, so the wait is a join on the daemon thread, never on the network call (D-10).
    thread = checker.start() if checker is not None else None
    started = time.monotonic()
    timeout = versions.UPDATE_CHECK_TIMEOUT_SECONDS

    _print_exegol_mcp_state(console, state.sdk_loaded)
    _print_exegol_facts(console, state.sdk_loaded)
    console.print(_debug_line("Installation", _best_effort(_installation)))
    console.print(
        _debug_line(
            "Python",
            _best_effort_text(
                lambda: Text.assemble(version_text(platform.python_version()), f" ({sys.executable})")
            ),
        )
    )
    console.print(_debug_line("MCP SDK", _best_effort_text(lambda: version_text(importlib.metadata.version("mcp")))))

    if checker is None or thread is None:
        console.print(_debug_line("Update check", skipped or "not run"))
        return
    thread.join(max(0.0, timeout - (time.monotonic() - started)))
    console.print(_debug_line("Update check", _update_check_line(checker, thread, timeout)))
    if checker.notice is not None and not thread.is_alive():
        console.print(prefixed_line(logging.INFO, Text(checker.notice)))


def run_version(args: Namespace) -> int:
    """Print the version report on stdout and return the exit status (always 0).

    Args:
        args: The parsed command line.

    Returns:
        0: a problem with an optional part of the report is a line in it, never a failure (D-08, D-18).
    """
    # D-27: libraries other than exegol-mcp and the SDK print only warnings while the report runs, before the first
    # probe or SDK import, so a pasted -vvv report carries no registry host name or request line.
    cap_third_party_logs()
    console = Console(soft_wrap=True)
    # __version__ is read here, as a module global, so a test can monkeypatch it.
    info = Versions(
        exegol_mcp=__version__,
        exegol_library=library_version(),
        exegol_wrapper=detect_wrapper_version(),
        docker_status="not checked",
    )
    console.print(version_headline(info.exegol_mcp))
    console.print(prefixed_line(logging.INFO, "Exegol library: ", version_text(info.exegol_library)))
    # D-26: the bundled library is what runs, so the wrapper line shows only when it differs or is missing (always at -vvv).
    if show_wrapper_line(info, verbose=args.verbose):
        console.print(prefixed_line(logging.INFO, "Exegol wrapper: ", wrapper_text(info)))
    console.print(
        prefixed_line(
            logging.INFO,
            "More about Exegol MCP at: ",
            Text(DOCS_URL, style="underline magenta"),
            style="bold magenta",
        )
    )
    state = _print_state(console, args)
    if args.verbose >= 3:
        _print_debug_section(console, args, state)
    notice = prerelease_notice(__version__)
    if notice is not None:
        console.print()
        console.print(notice)
    return 0
