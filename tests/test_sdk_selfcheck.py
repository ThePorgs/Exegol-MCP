"""Headless Exegol SDK setup (D-09) and the D-15 patch-point self-check, without Docker.

These tests import the exegol SDK but never construct DockerUtils, so they need no Docker daemon. Every SDK
global the headless patch changes is restored after each test by the sdk_globals fixture.
"""

import ast
import logging
import os
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path

import exegol.model.ExegolContainer as _container_module
import exegol.profile.ProfileAskUser as _ask_module
import exegol.utils.FsUtils as _fs_utils_module
import pytest
from docker.models import containers as docker_containers_module
from docker.models.containers import ContainerCollection
from exegol.config import OptionResolver as option_resolver_module
from exegol.config.DataCache import DataCache
from exegol.config.OptionResolver import OptionKey, OptionResolver
from exegol.config.SudoInfo import SudoInfo
from exegol.console.cli.ParametersManager import ParametersManager
from exegol.console.ExegolPrompt import ExegolRich
from exegol.console.TUI import ExegolTUI
from exegol.model.ContainerConfig import ContainerConfig
from exegol.model.ExegolContainer import ExegolContainer
from exegol.model.ExegolContainerTemplate import ExegolContainerTemplate
from exegol.model.ExegolNetwork import ExegolNetwork
from exegol.utils import ExeLog as exelog_module
from exegol.utils.DockerUtils import DockerUtils
from exegol.utils.ExeLog import ExeLog, logger
from exegol.utils.imgsync.ImageScriptSync import ImageScriptSync
from exegol.utils.LocalDatastore import LocalDatastore
from exegol.utils.MetaSingleton import MetaSingleton

from exegol_mcp.adapter import sdk_headless, selfcheck
from exegol_mcp.bootstrap import logging_setup


@pytest.fixture
def sdk_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register an undo for every SDK global patch_exegol_sdk() replaces."""
    monkeypatch.setattr(MetaSingleton, "_MetaSingleton__instances", dict(MetaSingleton._MetaSingleton__instances))
    monkeypatch.setattr(ExeLog, "console", ExeLog.console)
    monkeypatch.setattr(logging_setup, "_console", logging_setup._console)
    monkeypatch.setattr(logger, "_ExeLog__critical_method", logger._ExeLog__critical_method)
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "console"):
            monkeypatch.setattr(handler, "console", handler.console)
    for name in ("Ask", "Confirm", "Acknowledge"):
        monkeypatch.setattr(ExegolRich, name, ExegolRich.__dict__[name])
    monkeypatch.setattr(ExegolTUI, "downloadDockerLayer", ExegolTUI.__dict__["downloadDockerLayer"])
    monkeypatch.setattr(_ask_module, "stdinCanAnswer", _ask_module.stdinCanAnswer)
    monkeypatch.setattr(ContainerCollection, "create", ContainerCollection.__dict__["create"])
    monkeypatch.setattr(_container_module, "secure_remove", _container_module.secure_remove)
    for name in sdk_headless._PATCHED_CONTAINER_METHODS:
        monkeypatch.setattr(ExegolContainer, name, ExegolContainer.__dict__[name])


# --- D-15 self-check -------------------------------------------------------------------------------------------


def test_assert_patch_points_passes_on_the_installed_sdk() -> None:
    selfcheck.assert_patch_points()


def test_the_docker_py_create_patch_step_8_guards_is_self_checked() -> None:
    assert ("docker.models.containers", "ContainerCollection.create") in selfcheck._CALLABLE_POINTS


@pytest.mark.usefixtures("sdk_globals")
def test_the_host_shred_patch_step_9_is_idempotent_and_self_checked() -> None:
    # Both names are pinned: the one __removeVolume calls, which step 9 replaces, and Exegol's original (WR-09).
    assert ("exegol.model.ExegolContainer", "secure_remove") in selfcheck._CALLABLE_POINTS
    assert ("exegol.utils.FsUtils", "secure_remove") in selfcheck._CALLABLE_POINTS
    exegol_original = _fs_utils_module.secure_remove
    assert sdk_headless._EXEGOL_SECURE_REMOVE is exegol_original

    sdk_headless.patch_exegol_sdk()
    sdk_headless.patch_exegol_sdk()

    # Patched twice, the model module's name is the wrapper, which still delegates to Exegol's own function.
    assert _container_module.secure_remove is sdk_headless._secure_remove_offloaded
    assert sdk_headless._EXEGOL_SECURE_REMOVE is exegol_original
    assert _fs_utils_module.secure_remove is exegol_original
    assert exegol_original.__module__ == "exegol.utils.FsUtils"


@pytest.mark.parametrize(
    ("owner", "attribute", "point"),
    [
        (MetaSingleton, "mock", "exegol.utils.MetaSingleton.MetaSingleton.mock"),
        (ExegolTUI, "downloadDockerLayer", "exegol.console.TUI.ExegolTUI.downloadDockerLayer"),
        (LocalDatastore, "is_eula_accepted", "exegol.utils.LocalDatastore.LocalDatastore.is_eula_accepted"),
        (LocalDatastore, "update_eula", "exegol.utils.LocalDatastore.LocalDatastore.update_eula"),
        (ExeLog, "setCriticalMethod", "exegol.utils.ExeLog.ExeLog.setCriticalMethod"),
        (SudoInfo, "getUserHome", "exegol.config.SudoInfo.SudoInfo.getUserHome"),
        (SudoInfo, "getUserUid", "exegol.config.SudoInfo.SudoInfo.getUserUid"),
        (SudoInfo, "isUnresolvedSudo", "exegol.config.SudoInfo.SudoInfo.isUnresolvedSudo"),
        (ExegolRich, "Confirm", "exegol.console.ExegolPrompt.ExegolRich.Confirm"),
        (ExegolContainer, "formatShellCommand", "exegol.model.ExegolContainer.ExegolContainer.formatShellCommand"),
        (OptionResolver, "get", "exegol.config.OptionResolver.OptionResolver.get"),
        # Lifecycle (lifecycle_sdk.py, 04-03)
        (
            ExegolContainerTemplate,
            "newContainer",
            "exegol.model.ExegolContainerTemplate.ExegolContainerTemplate.newContainer",
        ),
        (ContainerConfig, "addLabel", "exegol.model.ContainerConfig.ContainerConfig.addLabel"),
        (DockerUtils, "createContainer", "exegol.utils.DockerUtils.DockerUtils.createContainer"),
        (ExegolContainer, "postCreateSetup", "exegol.model.ExegolContainer.ExegolContainer.postCreateSetup"),
        (_ask_module, "stdinCanAnswer", "exegol.profile.ProfileAskUser.stdinCanAnswer"),
        (OptionResolver, "loadProfile", "exegol.config.OptionResolver.OptionResolver.loadProfile"),
        # Container start, stop, create post-setup and removal off the loop (INT-01)
        (_container_module, "ExegolStatus", "exegol.model.ExegolContainer.ExegolStatus"),
        (_container_module, "APIError", "exegol.model.ExegolContainer.APIError"),
        (_container_module, "ReadTimeout", "exegol.model.ExegolContainer.ReadTimeout"),
        (
            ExegolContainer,
            "_ExegolContainer__start_container",
            "exegol.model.ExegolContainer.ExegolContainer._ExegolContainer__start_container",
        ),
        (
            ExegolContainer,
            "_ExegolContainer__preStartChecks",
            "exegol.model.ExegolContainer.ExegolContainer._ExegolContainer__preStartChecks",
        ),
        (
            ExegolContainer,
            "_ExegolContainer__postStartSetup",
            "exegol.model.ExegolContainer.ExegolContainer._ExegolContainer__postStartSetup",
        ),
        (ExeLog, "verbose", "exegol.utils.ExeLog.ExeLog.verbose"),
        (ExeLog, "advanced", "exegol.utils.ExeLog.ExeLog.advanced"),
        (ExeLog, "success", "exegol.utils.ExeLog.ExeLog.success"),
        (ExeLog, "VERBOSE", "exegol.utils.ExeLog.ExeLog.VERBOSE"),
        # The names the removal mirror's single-hop docker tail reads (patch step 13)
        (_container_module, "DataCache", "exegol.model.ExegolContainer.DataCache"),
        (_container_module, "NotFound", "exegol.model.ExegolContainer.NotFound"),
        (_container_module, "ObjectNotFound", "exegol.model.ExegolContainer.ObjectNotFound"),
        (DockerUtils, "removeContainerById", "exegol.utils.DockerUtils.DockerUtils.removeContainerById"),
        (DockerUtils, "removeNetwork", "exegol.utils.DockerUtils.DockerUtils.removeNetwork"),
        (ExegolNetwork, "shouldBeRemoved", "exegol.model.ExegolNetwork.ExegolNetwork.shouldBeRemoved"),
        (ExegolNetwork, "getNetworkName", "exegol.model.ExegolNetwork.ExegolNetwork.getNetworkName"),
        (ContainerConfig, "getNetworks", "exegol.model.ContainerConfig.ContainerConfig.getNetworks"),
        (DataCache, "remove_container_cache", "exegol.config.DataCache.DataCache.remove_container_cache"),
        # The names the post-create mirror reads (patch step 14)
        (
            ExegolContainer,
            "_ExegolContainer__updatePasswd",
            "exegol.model.ExegolContainer.ExegolContainer._ExegolContainer__updatePasswd",
        ),
        (_container_module, "ImageScriptSync", "exegol.model.ExegolContainer.ImageScriptSync"),
        (
            ImageScriptSync,
            "getImageSyncTarData",
            "exegol.utils.imgsync.ImageScriptSync.ImageScriptSync.getImageSyncTarData",
        ),
    ],
)
def test_a_missing_patch_point_is_named(
    monkeypatch: pytest.MonkeyPatch, owner: type, attribute: str, point: str
) -> None:
    monkeypatch.delattr(owner, attribute)

    with pytest.raises(selfcheck.PatchPointError, match=re.escape(point)):
        selfcheck.assert_patch_points()


def test_a_missing_docker_py_container_collection_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deleted at module level: ContainerCollection.create is also found on docker-py's base Collection, so deleting
    # only the subclass attribute would not make the point disappear.
    monkeypatch.delattr(docker_containers_module, "ContainerCollection")

    with pytest.raises(
        selfcheck.PatchPointError, match=re.escape("docker.models.containers.ContainerCollection.create")
    ):
        selfcheck.assert_patch_points()


# A fresh interpreter, so the deleted attribute never leaks into the test session's process-wide SDK state. The
# modules sdk_headless's import chain needs are imported first, then the attribute is deleted, then sdk_headless is
# imported for the first time, which is when its import-time originals are read (WR-12).
_IMPORT_TIME_ORIGINAL_SCRIPT = """
import sys
import docker.models.containers
import exegol.model.ExegolContainer
import exegol.utils.FsUtils
import exegol_mcp.adapter.exec
from exegol_mcp.adapter.errors import PatchPointError
if sys.argv[1] == "fs_utils":
    del exegol.utils.FsUtils.secure_remove
else:
    del docker.models.containers.ContainerCollection
import exegol_mcp.adapter.sdk_headless as sdk_headless
try:
    sdk_headless.patch_exegol_sdk()
except PatchPointError as e:
    print(e)
    sys.exit(3)
"""


@pytest.mark.parametrize(
    ("case", "point"),
    [
        ("fs_utils", "exegol.utils.FsUtils.secure_remove"),
        ("docker_create", "docker.models.containers.ContainerCollection.create"),
    ],
)
def test_a_missing_import_time_original_is_named_by_the_self_check(case: str, point: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_TIME_ORIGINAL_SCRIPT, case],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 3, result.stderr
    assert point in result.stdout
    assert "AttributeError" not in result.stderr


async def test_a_call_through_a_missing_import_time_original_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    # Backstop behind the startup refusal: a call through a missing original is the named PatchPointError, never a
    # TypeError on None, and not an assert that -O would strip (WR-12).
    monkeypatch.setattr(sdk_headless, "_EXEGOL_SECURE_REMOVE", None)
    monkeypatch.setattr(sdk_headless, "_DOCKER_CONTAINERS_CREATE", None)

    with pytest.raises(selfcheck.PatchPointError, match=re.escape("exegol.utils.FsUtils.secure_remove")):
        sdk_headless._secure_remove_to_completion(Path("unused"), 1)
    with pytest.raises(selfcheck.PatchPointError, match=re.escape("exegol.utils.FsUtils.secure_remove")):
        await sdk_headless._secure_remove_offloaded(Path("unused"), 1)
    with pytest.raises(
        selfcheck.PatchPointError, match=re.escape("docker.models.containers.ContainerCollection.create")
    ):
        sdk_headless._guarded_containers_create(None, "unused-image")


# --- WR-13: the self-check runs before the first SDK import statement --------------------------------------------

# A fresh interpreter deletes one SDK name before the entry point is imported, so every exegol import statement of
# exegol_mcp that names it runs against the incomplete SDK. The real main() must then refuse with one named stderr
# line, never an ImportError traceback (D-15, WR-13).
_SDK_IMPORT_SCRIPT = """
import importlib
import sys
delattr(importlib.import_module(sys.argv[1]), sys.argv[2])
from exegol_mcp.main import main
sys.exit(main(["--type", "stdio", "--accept-eula", "--no-update-check"]))
"""


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    # No host config.yml, bearer secret or Exegol config is ever read by these subprocesses.
    return {**os.environ, "HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "cfg")}


def _run_python(tmp_path: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=_isolated_env(tmp_path),
    )


@pytest.mark.parametrize(
    ("module", "name", "expected"),
    [
        # The verifier's repro: a module-level from-import of sdk_headless.
        ("exegol.console.TUI", "ExegolTUI", "exegol.console.TUI.ExegolTUI"),
        # From-imported by sdk_headless and host.py, and by the SDK's own modules.
        ("exegol.config.EnvInfo", "EnvInfo", "EnvInfo"),
        # Imported by adapter/sudo.py, which the entry point loads.
        ("exegol.config.SudoInfo", "SudoInfo", "SudoInfo"),
    ],
)
def test_a_missing_sdk_import_is_one_named_stderr_line(tmp_path: Path, module: str, name: str, expected: str) -> None:
    result = _run_python(tmp_path, _SDK_IMPORT_SCRIPT, module, name)

    assert result.returncode == 1, result.stderr
    assert result.stdout == ""
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, result.stderr
    assert expected in lines[0]
    assert "install a supported exegol 6.x release" in lines[0]
    assert lines[0].startswith("[-] ")
    assert "Traceback" not in result.stderr


def test_the_entry_point_defers_every_sdk_import(tmp_path: Path) -> None:
    script = (
        "import sys, exegol_mcp.main\n"
        "print(sorted(m for m in sys.modules if m == 'exegol' or m.startswith('exegol.')))\n"
    )

    result = _run_python(tmp_path, script)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


_STDERR_ROUTING_SCRIPT = """
import logging
from exegol_mcp.bootstrap.logging_setup import ExeLogHandler, route_logging_to_stderr
route_logging_to_stderr(0)
import exegol.utils.ExeLog
handlers = logging.getLogger().handlers
assert len(handlers) == 1 and type(handlers[0]) is ExeLogHandler, handlers
exegol.utils.ExeLog.logger.warning("probe-line")
route_logging_to_stderr(0)  # idempotent: still exactly one root handler (LIFE-03)
handlers = logging.getLogger().handlers
assert len(handlers) == 1 and type(handlers[0]) is ExeLogHandler, handlers
"""


def test_stderr_routing_precedes_the_first_sdk_import(tmp_path: Path) -> None:
    # Exegol's import-time logging.basicConfig binds a RichHandler to stdout only when the root logger has no
    # handler yet; routed first, it does nothing (SEC-08).
    result = _run_python(tmp_path, _STDERR_ROUTING_SCRIPT)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert "probe-line" in result.stderr


def test_a_failed_sdk_import_is_named_with_its_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes importlib refuse the import ("import of ... halted"); the reason must reach the
    # operator, so a missing name inside the SDK's own import statements is named, on one line.
    monkeypatch.setitem(sys.modules, "exegol.console.cli.ParametersManager", None)

    with pytest.raises(selfcheck.PatchPointError) as raised:
        selfcheck.assert_patch_points()

    message = str(raised.value)
    assert "exegol.console.cli.ParametersManager" in message
    assert "cannot be imported (" in message
    assert "halted" in message
    assert "\n" not in message


def _is_sdk_module(name: str) -> bool:
    # exegol_mcp is not the SDK: "exegol_mcp" neither equals "exegol" nor starts with "exegol.".
    return name == "exegol" or name.startswith("exegol.")


def test_every_sdk_import_and_class_attribute_is_self_checked() -> None:
    import exegol_mcp

    points = (
        *selfcheck._CALLABLE_POINTS,
        *selfcheck._ATTRIBUTE_POINTS,
        *selfcheck._VERSION_ONLY_ATTRIBUTE_POINTS,
        ("exegol.config.OptionResolver", "OptionKey"),
    )
    option_members = {member for member, _ in selfcheck._OPTION_KEYS}
    # The root logger is a standard logging.Logger instance; a named logger would be an ExeLog once exegol is
    # imported (it sets the logger class), so its ExeLog-only methods would count as standard.
    standard_logger_attributes = set(dir(logging.getLogger()))
    assert "raw" not in standard_logger_attributes

    def covered(module: str, path: str) -> bool:
        return any(m == module and (p == path or p.startswith(f"{path}.")) for m, p in points)

    package_root = Path(exegol_mcp.__file__).parent
    violations: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        where = source.relative_to(package_root).as_posix()
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        # Local name -> (SDK module, from-imported name, or None for a module alias)
        bound: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and _is_sdk_module(node.module):
                for alias in node.names:
                    if not covered(node.module, alias.name):
                        violations.append(f"{where}:{node.lineno} from {node.module} import {alias.name}")
                    bound[alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_sdk_module(alias.name):
                        continue
                    if not any(m == alias.name for m, _ in points):
                        violations.append(f"{where}:{node.lineno} import {alias.name}")
                    if alias.asname:
                        bound[alias.asname] = (alias.name, None)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound):
                continue
            module, name = bound[node.value.id]
            attr = node.attr
            if name is None:
                ok = covered(module, attr)
            elif name == "OptionKey":
                ok = attr in option_members
            elif name == "logger":
                ok = attr in standard_logger_attributes or covered("exegol.utils.ExeLog", f"ExeLog.{attr}")
            else:
                ok = covered(module, f"{name}.{attr}")
            if not ok:
                violations.append(f"{where}:{node.lineno} {node.value.id}.{attr}")

    assert violations == [], "\n".join(violations)


def test_a_missing_module_level_point_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(exelog_module, "ExeLog")

    with pytest.raises(selfcheck.PatchPointError, match=re.escape("exegol.utils.ExeLog.ExeLog")):
        selfcheck.assert_patch_points()


def test_version_only_names_are_not_startup_patch_points() -> None:
    # Review WR-02: a name only `version -vvv` reads must never be able to stop the server from starting.
    startup = {*selfcheck._CALLABLE_POINTS, *selfcheck._ATTRIBUTE_POINTS}
    assert selfcheck._VERSION_ONLY_ATTRIBUTE_POINTS
    assert not startup & set(selfcheck._VERSION_ONLY_ATTRIBUTE_POINTS)


@pytest.mark.parametrize(("module_name", "path"), selfcheck._VERSION_ONLY_ATTRIBUTE_POINTS)
def test_a_renamed_version_only_name_does_not_stop_the_server_from_starting(
    monkeypatch: pytest.MonkeyPatch, module_name: str, path: str
) -> None:
    owner_path, attribute = path.rsplit(".", 1)
    owner = selfcheck._resolve(module_name, owner_path)
    monkeypatch.delattr(owner, attribute)

    selfcheck.assert_patch_points()  # does not raise


@pytest.mark.parametrize(("module_name", "path"), selfcheck._VERSION_ONLY_ATTRIBUTE_POINTS)
def test_every_version_only_name_exists_in_the_installed_sdk(module_name: str, path: str) -> None:
    # The drift alarm for the names `version -vvv` reads: a test failure here, not a refusal to start.
    selfcheck._resolve(module_name, path)


def test_a_renamed_option_key_member_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 6.x release that dropped the ACCEPT_EULA member (kept the other two).
    renamed = Enum("OptionKey", {"FORCE_MODE": "force_mode", "OFFLINE_MODE": "offline_mode"})
    monkeypatch.setattr(option_resolver_module, "OptionKey", renamed)

    with pytest.raises(selfcheck.PatchPointError, match=re.escape("OptionKey.ACCEPT_EULA")):
        selfcheck.assert_patch_points()


def test_a_changed_option_key_value_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    # The member still exists but now maps to a different option name, so forcing "force_mode" would miss it.
    changed = Enum("OptionKey", {"ACCEPT_EULA": "accept_eula", "FORCE_MODE": "force", "OFFLINE_MODE": "offline_mode"})
    monkeypatch.setattr(option_resolver_module, "OptionKey", changed)

    with pytest.raises(selfcheck.PatchPointError, match=re.escape("OptionKey.FORCE_MODE")):
        selfcheck.assert_patch_points()


@pytest.mark.parametrize(
    ("member", "option_name"),
    [("IMAGE_TAG", "imagetag"), ("GUI", "gui"), ("DESKTOP", "desktop"), ("HOSTNAME_ASK_USER", None)],
)
def test_a_create_option_key_that_moved_is_named(
    monkeypatch: pytest.MonkeyPatch, member: str, option_name: str | None
) -> None:
    members = {key.name: key.value for key in OptionKey if key.name != member}
    if option_name is not None:
        members[member] = f"{option_name}_renamed"  # still present, but mapped to another option
    monkeypatch.setattr(option_resolver_module, "OptionKey", Enum("OptionKey", members))

    with pytest.raises(selfcheck.PatchPointError, match=re.escape(f"OptionKey.{member}")):
        selfcheck.assert_patch_points()


# --- D-09 headless patch ---------------------------------------------------------------------------------------


@pytest.mark.usefixtures("sdk_globals")
def test_an_option_that_was_never_set_raises_naming_it() -> None:
    sdk_headless.patch_exegol_sdk()

    with pytest.raises(AttributeError, match="renamed_in_a_future_exegol"):
        _ = ParametersManager().renamed_in_a_future_exegol


@pytest.mark.usefixtures("sdk_globals")
def test_the_mock_never_returns_none_for_an_unknown_option() -> None:
    sdk_headless.patch_exegol_sdk()

    assert getattr(ParametersManager(), "unknown_option", "sentinel-default") == "sentinel-default"
    with pytest.raises(sdk_headless.UnknownParameterError, match="unknown_option"):
        _ = ParametersManager().unknown_option


@pytest.mark.usefixtures("sdk_globals")
def test_headless_options_resolve_non_interactive() -> None:
    sdk_headless.patch_exegol_sdk()
    resolver = OptionResolver()

    # The CLI tier answers for these, so get() never falls through to the user's config.yml.
    assert resolver.get(OptionKey.FORCE_MODE) is True
    assert resolver.get(OptionKey.OFFLINE_MODE) is False
    # The EULA is never accepted implicitly: only the operator's --accept-eula (or a prior acceptance) does it.
    assert resolver.get(OptionKey.ACCEPT_EULA) is False


@pytest.mark.usefixtures("sdk_globals")
def test_accept_eula_can_be_forced_by_the_operator_flag() -> None:
    sdk_headless.patch_exegol_sdk()

    sdk_headless.set_accept_eula(True)

    assert OptionResolver().get(OptionKey.ACCEPT_EULA) is True


@pytest.mark.usefixtures("sdk_globals")
def test_the_cli_tier_of_every_registered_option_resolves() -> None:
    # OptionResolver asks the CLI tier first for every option, including the ones with no CLI flag (empty dest).
    sdk_headless.patch_exegol_sdk()
    resolver = OptionResolver()

    for key in OptionKey:
        resolver.cliValue(key)  # raises UnknownParameterError if the mock does not answer for its dest


@pytest.mark.usefixtures("sdk_globals")
def test_the_sdk_can_switch_offline_mode_at_runtime() -> None:
    # WebRegistryUtils sets ParametersManager().offline_mode = True when the registry is unreachable.
    sdk_headless.patch_exegol_sdk()

    ParametersManager().offline_mode = True

    assert OptionResolver().get(OptionKey.OFFLINE_MODE) is True


_LOG_SCRIPT = """
import sys
from exegol.utils.ExeLog import ExeLog, logger
if sys.argv[1] == "patched":
    from exegol_mcp.adapter.sdk_headless import patch_exegol_sdk
    patch_exegol_sdk()
logger.warning("headless-marker")
ExeLog.console.print("console-marker")
"""


def _run_log_script(mode: str) -> subprocess.CompletedProcess[str]:
    # A fresh interpreter: under pytest, exegol's import-time logging.basicConfig is a no-op (pytest's capture
    # handler is already on the root logger), so the SDK's stdout RichHandler only exists in a real process.
    return subprocess.run(
        [sys.executable, "-c", _LOG_SCRIPT, mode], capture_output=True, text=True, timeout=60, check=True
    )


def test_exegol_logging_goes_to_stderr() -> None:
    result = _run_log_script("patched")

    assert result.stdout == ""
    assert "headless-marker" in result.stderr
    assert "console-marker" in result.stderr


def test_unpatched_exegol_logging_is_detected_on_stdout() -> None:
    # Control for the test above: without the patch the SDK logs on stdout, and the check sees it.
    result = _run_log_script("unpatched")

    assert "headless-marker" in result.stdout


@pytest.mark.usefixtures("sdk_globals")
def test_a_critical_sdk_error_raises_instead_of_exiting() -> None:
    sdk_headless.patch_exegol_sdk()

    with pytest.raises(RuntimeError, match="docker is busy"):
        logger.critical("docker is busy")


@pytest.mark.usefixtures("sdk_globals")
def test_the_patch_keeps_the_shared_stderr_console(monkeypatch: pytest.MonkeyPatch) -> None:
    # patch step 1 reuses the process's one stderr console: a second console would diverge from the log handler.
    root = logging.getLogger()
    package = logging.getLogger("exegol_mcp")
    sse = logging.getLogger("sse_starlette")
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "level", root.level)
    monkeypatch.setattr(package, "level", package.level)
    monkeypatch.setattr(sse, "level", sse.level)
    monkeypatch.setattr(logger, "level", logger.level)

    logging_setup.configure_stderr_logging(0)
    sdk_headless.patch_exegol_sdk()

    handler = root.handlers[0]
    assert handler.console is ExeLog.console is logging_setup.stderr_console()  # type: ignore[attr-defined]


@pytest.mark.usefixtures("sdk_globals")
async def test_an_sdk_prompt_raises_instead_of_waiting_for_input() -> None:
    sdk_headless.patch_exegol_sdk()

    with pytest.raises(sdk_headless.HeadlessPromptError, match="remove your Sentinel logging history"):
        await ExegolRich.Confirm("Do you want to remove your Sentinel logging history?", default=False)
    with pytest.raises(sdk_headless.HeadlessPromptError, match="Enter the path"):
        await ExegolRich.Ask("Enter the path of your workspace")
    with pytest.raises(sdk_headless.HeadlessPromptError, match="License has expired"):
        await ExegolRich.Acknowledge("License has expired.")


@pytest.mark.usefixtures("sdk_globals")
def test_patching_twice_starts_from_fresh_parameters() -> None:
    sdk_headless.patch_exegol_sdk()
    sdk_headless.set_accept_eula(True)

    sdk_headless.patch_exegol_sdk()

    assert OptionResolver().get(OptionKey.ACCEPT_EULA) is False


@pytest.mark.usefixtures("sdk_globals")
def test_image_pulls_report_through_the_per_request_progress_hook() -> None:
    from exegol_mcp.adapter import download_progress

    sdk_headless.patch_exegol_sdk()

    assert ExegolTUI.downloadDockerLayer is download_progress.download_docker_layer
