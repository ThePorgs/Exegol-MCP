"""Container creation and removal policy (LIFE-01, LIFE-02, LIFE-03, D-01..D-12, D-24, D-25). SDK-free: only an
ExegolPort.

Every agent-supplied create argument is validated here before the port sees it, so the adapter only ever receives
the D-04 core safe set: a validated name, an installed image tag, an allowlisted host workspace or none, a network
mode other than `attached`, Exegol-syntax ports, KEY=VALUE env entries, a hostname and a comment. Privileged mode,
capabilities, devices, extra mounts, VPN, desktop and the host display cannot be expressed (CreateSpec has no field
for them). Every failure is a ValueError the agent reads verbatim. Nothing here logs an argument.

Profiles (D-01, D-03, D-05): a profile must be an exact, case-sensitive member of the operator's profiles.allowed,
and profiles.require_profile makes one mandatory; both are checked before any adapter call. On top of a profile the
agent may override only the same safe set, through the same validation: the profile alone grants privileged mode,
devices, capabilities, mounts, VPN or desktop. The image may then come from the profile (the adapter resolves it).

Removal is managed-only: a container without the exact managed label is refused unless the operator set
lifecycle.allow_remove_unmanaged (D-11), and an omitted keep option falls back to the operator default, which keeps
everything (D-12). Cleanup selects managed containers only, whatever that toggle says (D-13). The port applies the
same gate again to the container it is about to remove (require_managed, and expected_run_id for a this_run cleanup),
so a name that changed hands after the service's lookup is still refused (WR-02).
"""

import re
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

from exegol_mcp.adapter.port import (
    IMAGE_REQUIRED_MESSAGE,
    MANAGED_LABEL,
    MANAGED_LABEL_VALUE,
    NAME_IN_USE_MESSAGE,
    RUN_ID_LABEL,
    UNMANAGED_REMOVAL_MESSAGE,
    CreateSpec,
    ExegolPort,
)
from exegol_mcp.models.config import ExecutionLimits, LifecycleSettings, ProfilesSettings
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.models.lifecycle import CleanupFailure, CleanupResult, CleanupScope, RemovedContainer
from exegol_mcp.services.containers import _require_state
from exegol_mcp.services.host_paths import dangerous_host_path_reason, expand_host_path

# D-06 / Pitfalls 2 and 14: lowercase ASCII only, so a name never changes case on macOS/Windows hosts and never
# collides with a differently-cased one. Checked after the leading "exegol-" prefixes are stripped.
CONTAINER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")
# Pitfall 5: a KEY= prefix is mandatory. A bare name would make Exegol copy the server's own environment value.
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Exegol's documented port syntax (ContainerConfig.addRawPort):
# [<host_ipv4>:]<host_port>[-<end_host_port>][:<container_port>[-<end_container_port>]][:<protocol>]. Stricter than
# Exegol's own regex, whose bare optional separators accept "8080:", "80::tcp" or "1-2-3" and then misparse them
# (IN-03): here every separator needs its part, and a container end needs a container start. ASCII so \d never
# accepts a non-ASCII digit int() would still parse.
PORT_RE = re.compile(
    r"^(?:(?P<ip>\d{1,3}(?:\.\d{1,3}){3}):)?(?P<host>\d+)(?:-(?P<host_end>\d+))?"
    r"(?::(?P<cont>\d+)(?:-(?P<cont_end>\d+))?)?(?::(?P<proto>udp|tcp|sctp))?$",
    re.ASCII,
)
# RFC 1123 labels (1-63 characters, no leading or trailing hyphen) joined by dots, 253 characters at most.
_HOSTNAME_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
HOSTNAME_RE = re.compile(rf"^(?=.{{1,253}}$){_HOSTNAME_LABEL}(?:\.{_HOSTNAME_LABEL})*$")
COMMENT_MAX_LEN = 512
MAX_LIST_ITEMS = 64
# Exegol's network modes minus `attached`, which would join an arbitrary Docker network by name (profiles only).
NETWORK_MODES = ("host", "docker", "nat", "disabled")

_EXEGOL_PREFIX = "exegol-"
_MAX_PORT = 65535
_DEFAULT_CONTAINER_OP_TIMEOUT_S = ExecutionLimits().container_op_timeout_s

_NAME_RULE_MESSAGE = (
    "Invalid container_name: after any leading 'exegol-' is removed it must be 1 to 63 characters of lowercase "
    "ASCII letters, digits, '_', '.' or '-', starting with a letter or a digit. Omit container_name to get a "
    "generated one."
)
# D-01: an agent's profile against an empty allowlist.
NO_PROFILES_OFFERED_MESSAGE = (
    "No container profiles are offered on this server: the operator lists them under profiles.allowed in config.yml"
)
_WORKSPACE_DISABLED_MESSAGE = (
    "Host-path workspaces are disabled on this server: omit workspace to get a private Exegol workspace. The "
    "operator can allow specific host directories by listing them under lifecycle.workspace_allowlist in config.yml."
)


def generate_name() -> str:
    """Return a fresh container name for a create without one (D-06): `mcp-` and 8 random hex characters."""
    return "mcp-" + secrets.token_hex(4)


def _validated_name(name: str | None) -> str:
    # 1. None -> generated; otherwise strip the leading "exegol-" prefixes like ExegolContainerTemplate does.
    if name is None:
        return generate_name()
    while name.startswith(_EXEGOL_PREFIX):
        name = name[len(_EXEGOL_PREFIX) :]
    if not CONTAINER_NAME_RE.fullmatch(name):
        raise ValueError(_NAME_RULE_MESSAGE)
    return name


def _validated_workspace(workspace: str, settings: LifecycleSettings, invoking_home: Path) -> str:
    # D-24: only an exact allowlist entry, never a subdirectory or a parent of one, and never a dangerous path.
    if not settings.workspace_allowlist:
        raise ValueError(_WORKSPACE_DISABLED_MESSAGE)
    try:
        path = expand_host_path(workspace, invoking_home)
    except ValueError as e:
        raise ValueError(f"Invalid workspace: {e}. Omit workspace to get a private Exegol workspace.") from None
    allowed = [Path(entry) for entry in settings.workspace_allowlist]
    if path not in allowed:
        raise ValueError(
            f"Workspace '{path}' is not an allowed host directory. This server allows exactly these directories "
            f"(not their subdirectories): {', '.join(str(entry) for entry in allowed)}. Omit workspace to get a "
            "private Exegol workspace."
        )
    # Defense in depth: the loader already refuses a dangerous entry, but a config built another way may not.
    reason = dangerous_host_path_reason(path, (invoking_home, Path.home()))
    if reason is not None:
        raise ValueError(f"Workspace '{path}' is refused: {reason}. Omit workspace to get a private Exegol workspace.")
    return str(path)


def _check_profile(profile: str | None, profiles: ProfilesSettings) -> None:
    # D-01: exact string membership only, never lowercased or stripped. D-03: require_profile makes one mandatory.
    allowed = ", ".join(profiles.allowed)
    if profile is None:
        if not profiles.require_profile:
            return
        if not profiles.allowed:
            raise ValueError(
                f"This server requires a container profile, but none is offered. {NO_PROFILES_OFFERED_MESSAGE}."
            )
        raise ValueError(f"This server requires a container profile: pass profile as one of: {allowed}.")
    if not profiles.allowed:
        raise ValueError(f"{NO_PROFILES_OFFERED_MESSAGE}.")
    if profile not in profiles.allowed:
        raise ValueError(
            f"Container profile '{profile}' is not offered on this server. Allowed profiles (exact names): {allowed}."
        )


def _check_count(field_name: str, entries: Sequence[str]) -> None:
    if len(entries) > MAX_LIST_ITEMS:
        raise ValueError(f"{field_name} accepts at most {MAX_LIST_ITEMS} entries, got {len(entries)}.")


def _validated_port(entry: str) -> str:
    match = PORT_RE.fullmatch(entry)
    error = ValueError(
        f"Invalid port '{entry}': use Exegol's [ip:]host_port[-end][:container_port[-end]][:protocol] syntax "
        f"(protocol tcp, udp or sctp, ports 0 to {_MAX_PORT}, a range start not above its end), for example "
        "8080:80 or 127.0.0.1:2222:22:tcp."
    )
    if match is None:
        raise error
    ip = match.group("ip")
    if ip is not None and any(int(octet) > 255 for octet in ip.split(".")):
        raise error
    host_start = int(match.group("host"))
    host_end = int(match.group("host_end")) if match.group("host_end") else host_start
    container_start = int(match.group("cont")) if match.group("cont") else host_start
    container_end = int(match.group("cont_end")) if match.group("cont_end") else container_start
    if max(host_start, host_end, container_start, container_end) > _MAX_PORT:
        raise error
    if host_end < host_start or container_end < container_start:
        raise error
    return entry


def _validated_env(entry: str) -> str:
    if not ENV_RE.match(entry) or "\x00" in entry:
        raise ValueError(
            "Invalid env entry: each entry must be KEY=VALUE, with KEY made of ASCII letters, digits and '_' and not "
            "starting with a digit, and no NUL character. A bare KEY is refused: it would copy the server's own "
            "environment value into the container."
        )
    return entry


async def create_container(
    adapter: ExegolPort,
    *,
    image: str | None,
    name: str | None,
    run_id: str,
    settings: LifecycleSettings,
    invoking_home: Path,
    profile: str | None = None,
    profiles: ProfilesSettings | None = None,
    workspace: str | None = None,
    network_mode: str | None = None,
    ports: Sequence[str] | None = None,
    env: Sequence[str] | None = None,
    hostname: str | None = None,
    comment: str | None = None,
    start_timeout_s: float | None = None,
) -> ContainerInfo:
    """Validate the profile and the D-04 safe set, then create, label and start one container through the port.

    Args:
        adapter: The Exegol port.
        image: Installed image tag; required without a profile (D-07), never picked or pulled here. With a profile,
            None lets the adapter use the profile's image.tag (IMAGE_REQUIRED_MESSAGE when it names none).
        name: Container name, or None for a generated `mcp-<8 hex>` one (D-06).
        run_id: This server process's run id, stamped in RUN_ID_LABEL next to MANAGED_LABEL (D-09).
        settings: The operator lifecycle settings (the D-24 workspace allowlist).
        invoking_home: The home a workspace `~` expands to (the invoking user's under sudo, D-08).
        profile: An Exegol profile selection, exactly as listed in profiles.allowed, or None (D-01).
        profiles: The operator's profile allowlist and require_profile toggle; None means none offered (D-01, D-03).
        workspace: An allowlisted host directory, or None for a private Exegol workspace.
        network_mode: One of NETWORK_MODES, or None for Exegol's default.
        ports: Exegol port specs, at most MAX_LIST_ITEMS.
        env: KEY=VALUE entries, at most MAX_LIST_ITEMS.
        hostname: RFC 1123 hostname.
        comment: Free-text comment, at most COMMENT_MAX_LEN characters.
        start_timeout_s: Bound on the post-create start; None uses the built-in container_op_timeout_s default.

    Returns:
        The new container's info with this_run set.

    Raises:
        ValueError: The profile is not allowlisted or is required and missing, any argument breaks its rule, the
            image is missing or not installed, the name is in use, or Exegol cannot use the profile.
        RuntimeError: The start did not finish within start_timeout_s (the container then exists), or the host's
            licence does not grant profiles.
    """
    # 1. Profile: allowlisted exactly, and required when the operator says so (D-01, D-03), before any adapter call
    _check_profile(profile, profiles if profiles is not None else ProfilesSettings())
    # 2. Name: generated or validated (D-06)
    name = _validated_name(name)
    # 3. Image: required without a profile; the server never picks one (D-07). A profile may supply it.
    if not image and profile is None:
        raise ValueError(IMAGE_REQUIRED_MESSAGE)
    # 4. Workspace: private unless an exact, non-dangerous allowlist entry (D-24, D-08)
    host_workspace = None if workspace is None else _validated_workspace(workspace, settings, invoking_home)
    # 5. Network mode: never `attached` (D-04)
    if network_mode is not None and network_mode not in NETWORK_MODES:
        raise ValueError(f"Invalid network_mode '{network_mode}': use one of {', '.join(NETWORK_MODES)}.")
    # 6. Ports and env: bounded lists, Exegol's port syntax, KEY=VALUE only (Pitfall 5)
    _check_count("ports", ports or ())
    _check_count("env", env or ())
    port_specs = tuple(_validated_port(entry) for entry in ports or ())
    env_entries = tuple(_validated_env(entry) for entry in env or ())
    # 7. Hostname and comment
    if hostname is not None and not HOSTNAME_RE.fullmatch(hostname):
        raise ValueError(
            "Invalid hostname: use RFC 1123 labels (ASCII letters, digits and '-', 1 to 63 characters each, not "
            "starting or ending with '-') joined by dots, 253 characters at most."
        )
    if comment is not None and (len(comment) > COMMENT_MAX_LEN or "\x00" in comment):
        raise ValueError(f"Invalid comment: at most {COMMENT_MAX_LEN} characters and no NUL character.")
    # 8. The name must be free; the adapter re-checks inside its create lock (Pitfall 2)
    if await adapter.get_container_state(name) is not None:
        raise ValueError(NAME_IN_USE_MESSAGE.format(name=name))
    # 9. Build the spec with this run's labels (D-09)
    spec = CreateSpec(
        name=name,
        image=image or None,
        labels={MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: run_id},
        workspace=host_workspace,
        network_mode=network_mode,
        ports=port_specs,
        env=env_entries,
        hostname=hostname,
        comment=comment,
        start_timeout_s=_DEFAULT_CONTAINER_OP_TIMEOUT_S if start_timeout_s is None else start_timeout_s,
        profile=profile,
    )
    # 10. Create, then mark it as this run's
    return (await adapter.create_container(spec)).model_copy(update={"this_run": True})


def _resolved_keep(
    keep_workspace: bool | None, keep_logs: bool | None, settings: LifecycleSettings
) -> tuple[bool, bool]:
    # D-12: an omitted argument falls back to the operator default; the built-in defaults keep both.
    return (
        settings.keep_workspace if keep_workspace is None else keep_workspace,
        settings.keep_logs if keep_logs is None else keep_logs,
    )


def _op_timeout(timeout_s: float | None) -> float:
    # CR-01: every removal is bounded; an ad-hoc caller without the operator's value gets the built-in default.
    return _DEFAULT_CONTAINER_OP_TIMEOUT_S if timeout_s is None else timeout_s


async def remove_container(
    adapter: ExegolPort,
    name: str,
    *,
    keep_workspace: bool | None,
    keep_logs: bool | None,
    settings: LifecycleSettings,
    timeout_s: float | None = None,
) -> RemovedContainer:
    """Remove one container exegol-mcp created, keeping what the call or the operator default asks for (LIFE-03).

    Args:
        adapter: The Exegol port.
        name: The container to remove.
        keep_workspace: Keep a private workspace on the host; None uses lifecycle.keep_workspace.
        keep_logs: Keep the Sentinel audit history; None uses lifecycle.keep_logs.
        settings: The operator lifecycle settings (allow_remove_unmanaged and the keep defaults).
        timeout_s: The bound on the removal (execution.container_op_timeout_s); None uses the built-in default.

    Returns:
        The removed container's name and what is still on the host afterwards (WR-06). A host-mounted workspace is
        never deleted and is reported as kept while its directory exists; a container with no workspace reports none
        kept (IN-10).

    Raises:
        ValueError: No container has that name (naming list_containers), or it lacks the managed label and the
            operator did not set allow_remove_unmanaged (D-11).
        RuntimeError: Docker or the host failed the removal, or it reached timeout_s before it finished
            (REMOVAL_TIMEOUT_MESSAGE, CR-01). A removal that finished after the deadline is reported done (WR-08).
    """
    # 1. The container must exist
    state = await _require_state(adapter, name)
    # 2. Managed-only unless the operator allowed otherwise (D-11): a fast pre-check, before any removal work
    require_managed = not settings.allow_remove_unmanaged
    if require_managed and not state.managed:
        raise ValueError(UNMANAGED_REMOVAL_MESSAGE.format(name=name))
    # 3. Resolve the keep options against the operator defaults (D-12)
    workspace, logs = _resolved_keep(keep_workspace, keep_logs, settings)
    # 4. Remove through the port, which applies the same gate to the container it removes (WR-02), and report what
    # is still on the host
    outcome = await adapter.remove_container(
        name,
        keep_workspace=workspace,
        keep_logs=logs,
        timeout_s=_op_timeout(timeout_s),
        require_managed=require_managed,
    )
    return RemovedContainer(name=name, workspace_kept=outcome.workspace_kept, logs_kept=outcome.logs_kept)


async def cleanup_containers(
    adapter: ExegolPort,
    *,
    scope: str,
    run_id: str | None,
    keep_workspace: bool | None,
    keep_logs: bool | None,
    settings: LifecycleSettings,
    timeout_s: float | None = None,
) -> CleanupResult:
    """Remove this run's managed containers, or every managed one, and report each outcome (LIFE-04, D-13).

    Args:
        adapter: The Exegol port.
        scope: "this_run" (managed containers whose run id equals run_id exactly) or "all_managed" (every managed
            container, including earlier, crashed or concurrent runs' ones, D-10).
        run_id: This server process's run id; None selects nothing for "this_run".
        keep_workspace: Keep each private workspace; None uses lifecycle.keep_workspace.
        keep_logs: Keep each Sentinel audit history; None uses lifecycle.keep_logs.
        settings: The operator lifecycle settings (the keep defaults; allow_remove_unmanaged is ignored here).
        timeout_s: The bound on each removal (execution.container_op_timeout_s); None uses the built-in default. A
            removal that reaches it before finishing raises RuntimeError(REMOVAL_TIMEOUT_MESSAGE) and is listed under
            failed; one that finished after it is listed under removed (WR-08).

    Returns:
        The scope, the removed names and the per-container failures, both in name order. A failure of any kind
        (an Exception) is listed under failed with its message, or its type name when the message is empty, and the
        remaining containers are still removed (WR-01). Cancellation is not swallowed: it propagates and no result
        is returned. An empty result when nothing matched.

    Raises:
        ValueError: The scope is not one of the two values.
    """
    # 1. Only the two scopes exist
    scopes = get_args(CleanupScope)
    if scope not in scopes:
        raise ValueError(f"Invalid scope '{scope}': use one of {', '.join(scopes)}.")
    # 2. Read the live containers
    containers = await adapter.list_containers()
    # 3. Managed only, never an unlabeled container, even with allow_remove_unmanaged (D-13)
    targets = sorted(
        info.name
        for info in containers
        if info.managed and (scope == "all_managed" or (run_id is not None and info.run_id == run_id))
    )
    workspace, logs = _resolved_keep(keep_workspace, keep_logs, settings)
    op_timeout = _op_timeout(timeout_s)
    # The port re-checks each target's own labels right before removing it (WR-02): always managed whatever the
    # toggle (D-13), and this run's exact id for "this_run", so a name that changed hands since the listing is refused.
    expected_run_id = run_id if scope == "this_run" else None
    removed: list[str] = []
    failed: list[CleanupFailure] = []
    # 4. One container's failure, of any kind, is reported and never stops the others (WR-01). Cancellation is a
    # BaseException and still propagates.
    for name in targets:
        try:
            await adapter.remove_container(
                name,
                keep_workspace=workspace,
                keep_logs=logs,
                timeout_s=op_timeout,
                require_managed=True,
                expected_run_id=expected_run_id,
            )
        except Exception as e:  # noqa: BLE001 - one container's failure must never stop the others (LIFE-04)
            failed.append(CleanupFailure(name=name, error=str(e) or type(e).__name__))
        else:
            removed.append(name)
    # 5. Both lists follow the sorted target order
    return CleanupResult(scope=cast(CleanupScope, scope), removed=removed, failed=failed)
