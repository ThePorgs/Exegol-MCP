"""D-15 self-check of the Exegol SDK internals exegol-mcp patches or relies on.

exegol-mcp accepts any exegol 6.x release (exegol>=6.0.2,<7), but it patches several SDK internals to run
headless and the adapter calls SDK methods by name. A 6.x release that moved or renamed one of them would
otherwise fail deep inside a tool call, or worse, silently skip a patch. assert_patch_points() checks every
one of them up front and names the first that is missing, so the server refuses to start with a clear reason.

Importing this module imports no exegol module: assert_patch_points() imports them. main() therefore runs the
check before any SDK import statement of exegol-mcp, so a missing name in one is named on one stderr line rather
than raised as an ImportError traceback (WR-13). A module the check cannot import is named with its ImportError
text, so a missing name inside the SDK's own import statements is named too.
"""

import importlib
from importlib import metadata
from types import ModuleType
from typing import Any

# Re-exported for existing callers: selfcheck.PatchPointError.
from exegol_mcp.adapter.errors import PatchPointError

# Callables exegol-mcp patches (the headless setup) or calls (the adapter), as (module, attribute path).
_CALLABLE_POINTS: tuple[tuple[str, str], ...] = (
    # Headless setup (sdk_headless.py)
    ("exegol.utils.MetaSingleton", "MetaSingleton.mock"),
    ("exegol.console.cli.ParametersManager", "ParametersManager"),
    ("exegol.console.TUI", "ExegolTUI.downloadDockerLayer"),
    # The startup-status READY wait class, patched into the model module with a deadline-aware subclass (D-18);
    # the base lives in exegol.utils. If a 6.x release moves either, the server refuses to start.
    ("exegol.model.ExegolContainer", "ContainerLogStream"),
    ("exegol.utils.ContainerLogStream", "ContainerLogStream"),
    ("exegol.console.ExegolPrompt", "ExegolRich.Ask"),
    ("exegol.console.ExegolPrompt", "ExegolRich.Confirm"),
    ("exegol.console.ExegolPrompt", "ExegolRich.Acknowledge"),
    ("exegol.utils.ExeLog", "ExeLog.setCriticalMethod"),
    # sdk_logging.set_verbosity at startup, and the SDK's raw output the removal exec mirrors (logger.raw)
    ("exegol.utils.ExeLog", "ExeLog.setVerbosity"),
    ("exegol.utils.ExeLog", "ExeLog.raw"),
    ("exegol.config.OptionResolver", "OptionResolver.get"),
    # EULA state shared with the wrapper (D-05)
    ("exegol.utils.LocalDatastore", "LocalDatastore.is_eula_accepted"),
    ("exegol.utils.LocalDatastore", "LocalDatastore.update_eula"),
    # Startup readiness probes (host.py, D-07/D-08)
    ("exegol.config.UserConfig", "UserConfig"),
    ("exegol.config.UserConfig", "UserConfig.getFilePath"),
    ("exegol.utils.DockerUtils", "DockerUtils.getDockerInfo"),
    ("exegol.config.EnvInfo", "EnvInfo.get_env_overlay"),
    # sudo-aware invoking-user resolution (sudo.py, D-17/D-18)
    ("exegol.config.SudoInfo", "SudoInfo.getUserHome"),
    ("exegol.config.SudoInfo", "SudoInfo.getUserUid"),
    ("exegol.config.SudoInfo", "SudoInfo.isUnresolvedSudo"),
    # sudo.invoking_user_env and sudo.is_sudo_context, read by the secret store at startup
    ("exegol.config.SudoInfo", "SudoInfo.getEnv"),
    ("exegol.config.SudoInfo", "SudoInfo.isSudoContext"),
    # The ~ expansion the startup check compares profile mounts with, as Exegol mounts them (sudo.expand_user, IN-12)
    ("exegol.config.SudoInfo", "SudoInfo.expandUser"),
    # The GUI socket mounts the startup workspace-allowlist check reads (host._gui_socket_dirs, WR-11)
    ("exegol.utils.GuiUtils", "GuiUtils.getX11SocketPath"),
    ("exegol.utils.GuiUtils", "GuiUtils.getWaylandSocketPath"),
    ("exegol.utils.GuiUtils", "GuiUtils.isWaylandGuiAvailable"),
    ("exegol.exceptions.ExegolExceptions", "CancelOperation"),
    # Container and image operations (real.py, exec.py)
    ("exegol.utils.DockerUtils", "DockerUtils.listContainers"),
    ("exegol.utils.DockerUtils", "DockerUtils.getContainer"),
    ("exegol.utils.DockerUtils", "DockerUtils.clearCache"),
    ("exegol.utils.DockerUtils", "DockerUtils.listImages"),
    ("exegol.utils.DockerUtils", "DockerUtils.getOfficialImageFromList"),
    ("exegol.utils.DockerUtils", "DockerUtils.downloadImage"),
    ("exegol.utils.DockerUtils", "DockerUtils.downloadVersionTag"),
    ("exegol.utils.DockerUtils", "DockerUtils.createLocalLastestImageTag"),
    ("exegol.model.ExegolContainer", "ExegolContainer.isRunning"),
    ("exegol.model.ExegolContainer", "ExegolContainer.start"),
    ("exegol.model.ExegolContainer", "ExegolContainer.stop"),
    ("exegol.model.ExegolContainer", "ExegolContainer.formatShellCommand"),
    ("exegol.model.ExegolContainer", "ExegolContainer.getRawStatus"),
    *(
        ("exegol.model.ContainerConfig", f"ContainerConfig.{getter}")
        for getter in (
            "isGUIEnable",
            "isDesktopEnabled",
            "isExegolResourcesEnable",
            "isMyResourcesEnable",
            "isShellLoggingEnable",
            "isTimezoneShared",
            "isWrapperStartShared",
            "getNetwork",
            "getVpnConfigPath",
            "getCreationDate",
            "getComment",
            "getTextEnvs",
            "getDevices",
            "getPrivileged",
            "getCapabilities",
        )
    ),
    *(
        ("exegol.model.ExegolImage", f"ExegolImage.{getter}")
        for getter in (
            "getName",
            "getImageVersion",
            "getLatestVersion",
            "isInstall",
            "isLocal",
            "isUpToDate",
            "getBuildDate",
            "getDisplayLicense",
            "isVersionSpecific",
            "hasVersionTag",
        )
    ),
    ("exegol.exceptions.ExegolExceptions", "ObjectNotFound"),
    # Lifecycle (lifecycle_sdk.py): the create choreography and its headless prompt guard (Pitfalls 1-3, 12)
    ("exegol.model.ExegolContainerTemplate", "ExegolContainerTemplate.newContainer"),
    ("exegol.model.ContainerConfig", "ContainerConfig.addLabel"),
    ("exegol.utils.DockerUtils", "DockerUtils.createContainer"),
    ("exegol.utils.DockerUtils", "DockerUtils.getInstalledImage"),
    ("exegol.model.ExegolContainer", "ExegolContainer.postCreateSetup"),
    ("exegol.profile.ProfileAskUser", "ask_user_for_value"),
    ("exegol.profile.ProfileAskUser", "stdinCanAnswer"),
    ("exegol.config.OptionResolver", "OptionResolver.loadProfile"),
    ("exegol.config.OptionResolver", "OptionResolver.loadProfileUserConfig"),
    # The docker-py create Exegol's createContainer calls, which patch step 8 guards so a 409 skips Exegol's rollback
    # (WR-03). docker-py is a dependency of exegol, so a release that moved it would break the guard silently.
    ("docker.models.containers", "ContainerCollection.create"),
    # Removal (lifecycle_sdk.remove_managed_container): Exegol's own remove(), its backups, the private workspace
    # path, and the volume step whose prompt texts the scoped removal policy answers (Pitfalls 8-9)
    ("exegol.model.ExegolContainer", "ExegolContainer.remove"),
    ("exegol.model.ExegolContainer", "ExegolContainer.getExistingBackupContainers"),
    ("exegol.model.ExegolContainer", "ExegolContainer._ExegolContainer__removeVolume"),
    ("exegol.model.ContainerConfig", "ContainerConfig.getPrivateVolumePath"),
    # The getters the removal outcome reads for a host-mounted or a disabled workspace (IN-10)
    ("exegol.model.ContainerConfig", "ContainerConfig.isWorkspaceCustom"),
    ("exegol.model.ContainerConfig", "ContainerConfig.getHostWorkspacePath"),
    # The Sentinel getters the removal reads to report what is still on the host afterwards (WR-06)
    ("exegol.model.ContainerConfig", "ContainerConfig.isSentinelEnable"),
    ("exegol.model.ContainerConfig", "ContainerConfig.getSentinelPath"),
    # The method sdk_headless.removal_offload mirrors for Exegol's in-container removal fallback (CR-01)
    ("exegol.model.ExegolContainer", "ExegolContainer.exec"),
    # The name __removeVolume calls for the host Sentinel shred, which patch step 9 replaces during a removal, and
    # Exegol's original it delegates to (WR-09)
    ("exegol.model.ExegolContainer", "secure_remove"),
    ("exegol.utils.FsUtils", "secure_remove"),
    # Container start, stop, create post-setup and removal off the loop (sdk_headless patch steps 10-14, INT-01): the
    # SDK methods the mirrors replace, the private steps they call and the names they read from the model module
    ("exegol.model.ExegolContainer", "ExegolStatus"),
    ("exegol.model.ExegolContainer", "APIError"),
    ("exegol.model.ExegolContainer", "ReadTimeout"),
    ("exegol.model.ExegolContainer", "ExegolContainer._ExegolContainer__start_container"),
    ("exegol.model.ExegolContainer", "ExegolContainer._ExegolContainer__preStartChecks"),
    ("exegol.model.ExegolContainer", "ExegolContainer._ExegolContainer__postStartSetup"),
    # The password update and the entrypoint tar the post-create mirror reads (patch step 14)
    ("exegol.model.ExegolContainer", "ExegolContainer._ExegolContainer__updatePasswd"),
    ("exegol.model.ExegolContainer", "ImageScriptSync"),
    ("exegol.utils.imgsync.ImageScriptSync", "ImageScriptSync.getImageSyncTarData"),
    # The startup log levels the start mirror dispatches (the SDK's own set, ExeLog.verbose et al.)
    ("exegol.utils.ExeLog", "ExeLog.verbose"),
    ("exegol.utils.ExeLog", "ExeLog.advanced"),
    ("exegol.utils.ExeLog", "ExeLog.success"),
    # The names the removal mirror's single-hop docker tail reads (patch step 13): the model module's own names, the
    # docker calls and the network and cache objects Exegol's remove uses
    ("exegol.model.ExegolContainer", "DataCache"),
    ("exegol.model.ExegolContainer", "NotFound"),
    ("exegol.model.ExegolContainer", "ObjectNotFound"),
    ("exegol.utils.DockerUtils", "DockerUtils.removeContainerById"),
    ("exegol.utils.DockerUtils", "DockerUtils.removeNetwork"),
    ("exegol.model.ExegolNetwork", "ExegolNetwork.shouldBeRemoved"),
    ("exegol.model.ExegolNetwork", "ExegolNetwork.getNetworkName"),
    ("exegol.model.ContainerConfig", "ContainerConfig.getNetworks"),
    ("exegol.config.DataCache", "DataCache.remove_container_cache"),
    # Profiles (profiles.py, lifecycle_sdk.py): Exegol's own profile manager, the profile application the create
    # path calls, and the cached-session licence check (D-01, Pitfall 7). The Exegol logger the profile error
    # collector attaches to is the `logger` attribute point below.
    ("exegol.profile.ContainerProfileManager", "ContainerProfileManager.load_profiles"),
    ("exegol.profile.ContainerProfileManager", "ContainerProfileManager.get_profile"),
    ("exegol.profile.ProfileApplication", "apply_profile"),
    ("exegol.utils.SessionHandler", "SessionHandler.pro_feature_access"),
)

# Non-callable attributes the headless setup reads or replaces.
_ATTRIBUTE_POINTS: tuple[tuple[str, str], ...] = (
    ("exegol.utils.ExeLog", "ExeLog.console"),
    ("exegol.utils.ExeLog", "logger"),
    # The level numbers bootstrap.logging_setup duplicates because it cannot import the SDK: a renamed constant is a
    # named refusal here, a renumbered one is caught by the drift test in tests/test_exelog_handler.py.
    ("exegol.utils.ExeLog", "ExeLog.SUCCESS"),
    ("exegol.utils.ExeLog", "ExeLog.VERBOSE"),
    ("exegol.utils.ExeLog", "ExeLog.ADVANCED"),
    ("exegol.config.EnvInfo", "EnvInfo.arch"),
    ("exegol.config.EnvInfo", "EnvInfo.docker_env_names"),
    ("exegol.config.ConstantConfig", "ConstantConfig.exegol_config_path"),
    # Exegol's install directory, its read-only bind sources and the timezone-bind condition, read by
    # host._python_install_dirs and host._exegol_read_only_binds for the startup allowlist check (CR-03)
    ("exegol.config.ConstantConfig", "ConstantConfig.src_root_path_obj"),
    ("exegol.config.ConstantConfig", "ConstantConfig.entrypoint_context_path_obj"),
    ("exegol.config.ConstantConfig", "ConstantConfig.spawn_context_path_obj"),
    ("exegol.config.ConstantConfig", "ConstantConfig.sentinel_zsh_context_path_obj"),
    ("exegol.config.ConstantConfig", "ConstantConfig.sentinel_bash_context_path_obj"),
    ("exegol.config.ConstantConfig", "ConstantConfig.sentinel_context_path_obj"),
    ("exegol.config.EnvInfo", "EnvInfo.is_windows_shell"),
    ("exegol.config.EnvInfo", "EnvInfo.is_mac_shell"),
)

# Names only the `version -vvv` debug section reads (adapter/diagnostics.py, D-09). The server never uses them, so a
# release that renames one must not stop it from starting: assert_patch_points() does not check this list, and the
# `version` path tolerates a missing name through its per-probe best-effort wrapper ("unknown (AttributeError)").
# The list exists so the AST coverage test still sees every SDK name exegol-mcp reads, and a drift test can say so.
_VERSION_ONLY_ATTRIBUTE_POINTS: tuple[tuple[str, str], ...] = (
    ("exegol.config.ConstantConfig", "ConstantConfig.exegol_config_forced"),
    ("exegol.config.EnvInfo", "EnvInfo.raw_arch"),
    ("exegol.config.EnvInfo", "EnvInfo.current_platform"),
)

# OptionKey members exegol-mcp forces or sets, with the option name each must still map to.
_OPTION_KEYS: tuple[tuple[str, str], ...] = (
    ("ACCEPT_EULA", "accept_eula"),
    ("FORCE_MODE", "force_mode"),
    ("OFFLINE_MODE", "offline_mode"),
    # The create CLI tier (lifecycle_sdk._cli_values) and the hostname/comment ask flags (Pitfall 3)
    ("IMAGE_TAG", "imagetag"),
    ("WORKSPACE_PATH", "workspace_path"),
    ("NETWORK", "network"),
    ("PORTS", "ports"),
    ("ENVS", "envs"),
    ("HOSTNAME", "hostname"),
    ("COMMENT", "comment"),
    ("GUI", "gui"),
    ("DESKTOP", "desktop"),
    ("HOSTNAME_ASK_USER", "network.hostname_ask_user"),
    ("COMMENT_ASK_USER", "metadata.comment_ask_user"),
    # The Exegol directories the startup workspace-allowlist check reads (RealHostProbe.exegol_mount_dirs, WR-10)
    ("MY_RESOURCES_PATH", "customization.my_resources_path"),
    ("EXEGOL_RESOURCES_PATH", "volumes.exegol_resources_path"),
    ("PRIVATE_WORKSPACE_PATH", "volumes.private_workspace_path"),
    ("SENTINEL_LOGS_HOST_PATH", "sentinel.sentinel_logs_host_path"),
    # Where Exegol reads container profiles from, reported as settings directories by the same check
    # (host._exegol_settings_dirs, IN-14 premise)
    ("PROFILE_COMPONENT_PATH", "profile_component_path"),
    ("PROFILE_SOURCES", "profile_sources"),
)

_MISSING = object()


def _installed_exegol() -> str:
    try:
        return metadata.version("exegol")
    except metadata.PackageNotFoundError:
        return "unknown"


def _broken(point: str, detail: str) -> PatchPointError:
    return PatchPointError(
        f"Exegol SDK patch point '{point}' {detail} in the installed exegol {_installed_exegol()}. "
        "This exegol release is not compatible with this exegol-mcp version: install a supported exegol 6.x "
        "release or upgrade exegol-mcp."
    )


def _import(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        # The ImportError text names what is missing when the SDK's own import statements fail; whitespace is
        # collapsed so the refusal stays one line.
        detail = " ".join(str(error).split())
        raise _broken(module_name, f"cannot be imported ({detail})") from None


def _resolve(module_name: str, path: str) -> Any:
    """Walk path from the module, raising PatchPointError naming the full point at the first missing step."""
    obj: Any = _import(module_name)
    for name in path.split("."):
        obj = getattr(obj, name, _MISSING)
        if obj is _MISSING:
            raise _broken(f"{module_name}.{path}", "is missing")
    return obj


def assert_patch_points() -> None:
    """Check that every Exegol SDK internal exegol-mcp patches or relies on is present.

    The check only imports the SDK and inspects attributes: it never connects to Docker.

    Raises:
        PatchPointError: Naming the first patch point that is missing, not callable, or no longer maps to the
            expected option name.
    """
    for module_name, path in _CALLABLE_POINTS:
        if not callable(_resolve(module_name, path)):
            raise _broken(f"{module_name}.{path}", "is no longer callable")

    for module_name, path in _ATTRIBUTE_POINTS:
        _resolve(module_name, path)

    option_key = _resolve("exegol.config.OptionResolver", "OptionKey")
    for member, option_name in _OPTION_KEYS:
        point = f"exegol.config.OptionResolver.OptionKey.{member}"
        value = getattr(option_key, member, _MISSING)
        if value is _MISSING:
            raise _broken(point, "is missing")
        if getattr(value, "value", value) != option_name:
            raise _broken(point, f"no longer maps to the '{option_name}' option")
