"""Load and validate the operator config file (config.yml) at startup (CONF-03/CONF-04, D-01..D-06).

The file is read once, before build_server, from the fixed sudo-aware config dir via secret_store (D-02) and parsed
with yaml.safe_load only (T-02-06: a crafted config can never construct arbitrary Python objects). A missing or
empty file means "all built-in defaults" (D-03, RESEARCH Pitfall 5); only a present-but-invalid file fails startup,
with a single ConfigError naming the offending field (CONF-04), mirroring the ReadinessError path in main().

After validation, every lifecycle.workspace_allowlist entry is resolved (a leading ``~`` is the invoking user's home
under sudo, then realpath) and checked with Exegol's profile-path rule; `/`, a home or an ancestor of one, a relative
path or a malformed entry stops startup naming the entry (D-24, D-08). An entry that resolves inside another entry
stops startup too, naming both (WR-04). Nesting and the home/root rule compare whole path components, with letter case
folded on macOS and Windows, where names differing only by case are one directory (IN-08). Entries are stored as
absolute paths, so later checks compare resolved paths exactly and never widen to a subdirectory or parent.

resolve() applies the CLI > file > built-in default precedence for the mirrored launch keys (D-05/D-06); every other
section comes only from the file or the defaults and is carried through unchanged.
"""

from argparse import Namespace
from pathlib import Path

import yaml
from pydantic import ValidationError

from exegol_mcp.adapter import sudo
from exegol_mcp.bootstrap import secret_store
from exegol_mcp.bootstrap.cli import DEFAULT_PORT
from exegol_mcp.models.config import OperatorConfig
from exegol_mcp.services.host_paths import dangerous_host_path_reason, expand_host_path, is_strictly_inside

CONFIG_FILE_NAME = "config.yml"


class ConfigError(Exception):
    """config.yml is present but invalid; the message names the offending field (CONF-04)."""


def _invoking_home() -> Path:
    """The home a leading ``~`` expands to: the invoking user's under sudo (D-08). A seam tests monkeypatch."""
    return sudo.invoking_user_home()


def _resolve_workspace_allowlist(cfg: OperatorConfig) -> OperatorConfig:
    """Resolve every workspace allowlist entry to an absolute realpath and refuse dangerous or nested ones (D-24).

    An entry nested inside another is refused because a container that mounts the outer directory could swap the inner
    one for a symlink between the create-time check and Docker's bind (WR-04). Equal entries are not nested. Both
    the nesting test and the home/root rule compare whole path components, with letter case folded on macOS and
    Windows (services.host_paths.CASE_INSENSITIVE_HOST), so a case-differing spelling cannot slip past them (IN-08).

    Raises:
        ConfigError: An entry is malformed (empty, NUL, too long, relative, ``~user``), is `/`, a home directory or
            an ancestor of one, or resolves inside another entry. The message names
            ``lifecycle.workspace_allowlist.<index>`` (and the outer entry's index for a nested one).
    """
    entries = cfg.lifecycle.workspace_allowlist
    if not entries:
        return cfg
    home = _invoking_home()
    homes = (home, Path.home())  # under sudo these differ: the invoker's home and root's (Exegol's rule)
    resolved: list[str] = []
    for index, raw in enumerate(entries):
        try:
            path = expand_host_path(raw, home)
        except ValueError as e:
            reason: str | None = str(e)
        else:
            reason = dangerous_host_path_reason(path, homes)
        if reason is not None:
            raise ConfigError(f"Invalid {CONFIG_FILE_NAME}: field 'lifecycle.workspace_allowlist.{index}': {reason}")
        resolved.append(str(path))
    # Compare resolved paths, so an entry reaching inside another through a symlinked parent is caught too. Case is
    # folded on a case-insensitive host, where realpath keeps the operator's spelling (IN-08).
    for i, inner in enumerate(resolved):
        for j, outer in enumerate(resolved):
            if i != j and is_strictly_inside(Path(inner), Path(outer)):
                raise ConfigError(
                    f"Invalid {CONFIG_FILE_NAME}: field 'lifecycle.workspace_allowlist.{i}': '{inner}' lies inside "
                    f"lifecycle.workspace_allowlist.{j} ('{outer}'). Nested entries are refused: a container that "
                    "mounts the outer directory could replace the inner one with a symlink before Docker mounts it "
                    "(D-24)."
                )
    lifecycle = cfg.lifecycle.model_copy(update={"workspace_allowlist": resolved})
    return cfg.model_copy(update={"lifecycle": lifecycle})


def load_operator_config() -> OperatorConfig:
    """Read config.yml and validate it into an OperatorConfig.

    A missing or empty file yields OperatorConfig() built-in defaults (D-03). The file is never written.

    Returns:
        The parsed, validated config, or the all-defaults config when no usable file is present.

    Raises:
        ConfigError: The file exists but cannot be read, is not valid YAML, or fails validation. The message names
            the first offending field's dotted path (CONF-04).
    """
    try:
        raw = secret_store.read_user_config_file(CONFIG_FILE_NAME)
    except OSError as e:
        raise ConfigError(f"{CONFIG_FILE_NAME} could not be read: {e}") from e

    if not raw:
        return _resolve_workspace_allowlist(OperatorConfig())  # absent or empty -> all defaults (D-03)

    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise ConfigError(f"{CONFIG_FILE_NAME} is not valid YAML: {e}") from e

    try:
        cfg = OperatorConfig.model_validate(data or {})  # None/empty mapping -> all defaults
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(part) for part in first["loc"]) or "(root)"
        raise ConfigError(f"Invalid {CONFIG_FILE_NAME}: field '{loc}': {first['msg']}") from e
    return _resolve_workspace_allowlist(cfg)


def resolve(args: Namespace, file_cfg: OperatorConfig) -> OperatorConfig:
    """Resolve the mirrored launch keys with CLI > file > built-in default precedence (D-05/D-06).

    A value given on the CLI wins; otherwise the file value is used when set; otherwise the built-in launch default
    (http / 2187 / bearer / update-check on). Every other section (execution, security, tool_groups, profiles,
    lifecycle, files) is carried through from the file unchanged — no CLI flag sets them.

    Args:
        args: The parsed command line. The mirrored flags default to None so "set on the CLI" is distinguishable
            from "left to the file".
        file_cfg: The config loaded from config.yml (or the all-defaults config).

    Returns:
        A new OperatorConfig with transport/port/auth/update_check resolved and every other file section kept.
    """
    transport = args.type if args.type is not None else (file_cfg.transport or "http")
    port = args.port if args.port is not None else (file_cfg.port or DEFAULT_PORT)
    auth = args.auth if args.auth is not None else (file_cfg.auth or "bearer")

    # D-13 / SEC-04: the no-auth path fails closed. "none" can only come from the file now (the CLI no longer
    # accepts it); it survives ONLY when security.allow_no_auth is explicitly true, otherwise it is coerced to
    # bearer. A CLI --auth bearer wins regardless, because it already resolved auth to "bearer" above (CLI > config,
    # the safe direction). Any unknown/absent value therefore resolves to bearer, never silently to an open server.
    if auth == "none" and not file_cfg.security.allow_no_auth:
        auth = "bearer"

    if args.no_update_check:
        update_check = False  # the store_true flag can only force-disable
    elif file_cfg.update_check is not None:
        update_check = file_cfg.update_check
    else:
        update_check = True

    return file_cfg.model_copy(
        update={"transport": transport, "port": port, "auth": auth, "update_check": update_check}
    )
