"""The server's operator config model (config.yml).

OperatorConfig is parsed from config.yml (D-01: YAML + pydantic) and is pure override on top of the built-in
defaults defined here (D-03: the file is optional and never written). Every section sets
``model_config = ConfigDict(extra="forbid")`` so an unknown or misspelled key is a named startup error (CONF-04,
D-05) instead of a silently-ignored typo. A bare section key (``files:`` with no value) means that section's defaults.

No section is inert. Each is read by a feature:
- ``execution``: the exec/job bounds and listing page size (CONF-03, Phase 2).
- ``security``: the deliberate no-auth opt-in (SEC-04, Phase 3).
- ``tool_groups``: build_server registers only the tools of enabled groups (CONF-02, D-22, D-23, G-05-1a).
- ``profiles``: the Exegol profile allowlist and ``require_profile`` that create_container enforces (CONF-01,
  D-01..D-03).
- ``lifecycle``: remove/cleanup defaults and the host-path workspace allowlist create_container checks (D-11, D-12,
  D-24); the allowlist entries are resolved to absolute realpaths and checked at load in bootstrap/config.py.
- ``files``: the read/write size limit and the list_directory caps (D-17, D-19).
The mirrored launch keys (D-06) let a deployment pin transport/port/auth/update-check in the file, with the CLI flag
winning when both are set (resolved in bootstrap/config.py).
"""

import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The four operator-facing tool groups the maintainer chose at UAT (G-05-1a), superseding D-21's seven. Defined here,
# not in exegol_mcp.tools, so the config model never imports the tool layer (tools import models). server_info
# belongs to no group and is never gateable.
ToolGroup = Literal["containers", "images", "execution", "files"]
TOOL_GROUPS: tuple[str, ...] = get_args(ToolGroup)

# D-01: an Exegol profile selection is `sourcekey.name` or a bare name. No whitespace and no path separator, so an
# entry can never be read as a path.
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PROFILE_NAME_MAX_LEN = 128


class ExecutionLimits(BaseModel):
    """Per-server execution bounds: timeouts, output/buffer caps, job limits and listing page size (CONF-03)."""

    model_config = ConfigDict(extra="forbid")  # CONF-04: an unknown execution key names itself at startup

    timeout_default_s: float = Field(
        default=50,
        gt=0,
        description="Default foreground exec/remote-exec timeout in seconds when the caller gives none (D-14)",
    )
    timeout_max_s: float = Field(
        default=600,
        gt=0,
        description="Maximum foreground exec/remote-exec timeout in seconds; a larger request is clamped (D-14)",
    )
    output_cap_bytes: int = Field(
        default=1_048_576,
        gt=0,
        description="Per-stream byte cap on captured stdout/stderr; the tail is kept and a marker reports the drop",
    )
    job_buffer_cap_bytes: int = Field(
        default=1_048_576,
        gt=0,
        description="Per-stream byte cap on a live background job's tail buffer",
    )
    max_concurrent_jobs: int = Field(
        default=16,
        gt=0,
        description="Maximum number of concurrent background jobs per server (D-10)",
    )
    finished_job_ttl_s: float = Field(
        default=3600,
        gt=0,
        description="How long a finished job and its output are retained before being swept, in seconds (D-10)",
    )
    max_job_runtime_s: float = Field(
        default=86_400,
        gt=0,
        description="Maximum wall-clock runtime of a background job before it is auto-killed, in seconds (D-15)",
    )
    kill_grace_s: float = Field(
        default=5,
        ge=0,
        description="Grace period between SIGTERM and SIGKILL when stopping an in-container process, in seconds (D-16)",
    )
    listing_page_size: int = Field(
        default=200,
        gt=0,
        description="Default page size for paginated listings (installed tools, offline resources) (D-23)",
    )
    container_op_timeout_s: float = Field(
        default=50,
        gt=0,
        description="Internal timeout for container start/stop, separate from the exec timeout keys (D-18)",
    )


class SecuritySettings(BaseModel):
    """Security-policy opt-ins (SEC-04, D-13). The only deliberate route to an unauthenticated HTTP server."""

    model_config = ConfigDict(extra="forbid")  # CONF-04: an unknown security key names itself at startup

    allow_no_auth: bool = Field(
        default=False,
        description=(
            "Allow starting HTTP without authentication (D-13). Defaults False so the server fails closed to "
            "bearer: unauthenticated HTTP starts ONLY when this is explicitly set true in config.yml, and a CLI "
            "--auth bearer still overrides it. There is no CLI flag for no-auth (SEC-04, D-12)."
        ),
    )


class ToolGroupsSettings(BaseModel):
    """Operator tool-group gating (CONF-02, D-22, D-23): a deny-list of groups whose tools are not registered."""

    model_config = ConfigDict(extra="forbid")  # CONF-04: an unknown tool_groups key names itself at startup

    disabled: list[ToolGroup] = Field(
        default_factory=list,
        description=(
            "Tool groups to disable (D-23): containers, images, execution, files. A listed group's tools are "
            "absent from tools/list and a call to one fails as an unknown tool. server_info is "
            "never gated. All groups are on by default (D-22). An unknown group name stops startup. Gating hides "
            "tools; it is not a security boundary: exec_command can still move files when `files` is disabled."
        ),
    )


class ProfilesSettings(BaseModel):
    """The operator's Exegol profile allowlist (CONF-01, D-01..D-03). exegol-mcp keeps no profile format of its own."""

    model_config = ConfigDict(extra="forbid")  # CONF-04: an unknown profiles key names itself at startup

    allowed: list[str] = Field(
        default_factory=list,
        description=(
            "Exegol profile selections (`sourcekey.name` or a bare name) agents may pass to create_container (D-01). "
            "Profiles are Exegol's own ContainerProfile files; empty means no profile is offered."
        ),
    )
    require_profile: bool = Field(
        default=False,
        description="When true, create_container rejects any call without an allowlisted profile (D-03)",
    )

    @field_validator("allowed")
    @classmethod
    def _valid_unique_profile_names(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        for name in value:
            if not name:
                raise ValueError("a profile name must not be empty")
            if len(name) > _PROFILE_NAME_MAX_LEN:
                raise ValueError(f"profile name {name[:32]!r}... is longer than {_PROFILE_NAME_MAX_LEN} characters")
            if not _PROFILE_NAME_RE.match(name):
                raise ValueError(
                    f"profile name {name!r} must start with a letter or digit and use only letters, digits, '_', "
                    "'.' and '-' (no whitespace or path separator)"
                )
            if name in seen:
                raise ValueError(f"profile name {name!r} is listed twice")
            seen.add(name)
        return value


class LifecycleSettings(BaseModel):
    """Container removal defaults and the host-path workspace allowlist (D-11, D-12, D-24)."""

    model_config = ConfigDict(extra="forbid")  # CONF-04: an unknown lifecycle key names itself at startup

    allow_remove_unmanaged: bool = Field(
        default=False,
        description=(
            "Let remove_container remove Exegol containers this server did not create (D-11). Off by default, so "
            "hand-made containers are protected; cleanup_containers never removes them either way (D-13)."
        ),
    )
    keep_workspace: bool = Field(
        default=True,
        description=(
            "Default for remove/cleanup keep_workspace when a call omits it; a host mount is never deleted (D-12)"
        ),
    )
    keep_logs: bool = Field(
        default=True,
        description="Default for remove/cleanup keep_logs when a call omits it (D-12)",
    )
    workspace_allowlist: list[str] = Field(
        default_factory=list,
        description=(
            "Host directories an agent may mount as its workspace (D-24). Each entry allows exactly that directory, "
            "never its subdirectories or parents. A leading ~ is the invoking user's home under sudo; entries are "
            "resolved to absolute realpaths at startup. `/`, a home directory or an ancestor of one, and a relative "
            "path stop startup. An entry inside another entry stops startup. On macOS and Windows these checks ignore "
            "letter case, so an entry inside a case-differing spelling of another entry counts as nested, and a "
            "case-differing home is still refused. An entry stops startup when it is inside a directory Exegol mounts "
            "read-write into containers, or contains one, or contains any path Docker walks through, symlinks "
            "included, to mount one. Those directories are Exegol's my-resources and exegol-resources directories, the "
            "X11 and Wayland display sockets it shares with GUI containers, and the read-write mounts of the "
            "allowlisted profiles. An entry inside, containing or equal to the private workspaces or Sentinel logs "
            "directory, or to a device the allowlisted profiles map (system.devices), also stops startup, because "
            "Exegol hands their subdirectories or device nodes to containers one by one; Docker maps every device node "
            "found under a device directory. An entry that contains a file or directory Exegol binds read-only into "
            "containers, or any path Docker walks to bind it, also stops startup. Those are spawn.sh, entrypoint.sh, "
            "and the Sentinel hooks and logger from Exegol's install directory; /etc/timezone and /etc/localtime on "
            "Linux; and the read-only mounts and VPN files of the allowlisted profiles. An entry inside a read-only "
            "mount loads. An entry inside, containing or equal to Exegol's config directory, its container profile "
            "directory or a path: profile source also stops startup, because Exegol reads the settings that decide "
            "what containers mount from there. An entry inside, containing or equal to the Python installation "
            "exegol-mcp runs from also stops startup, because exegol-mcp and Exegol run code from there: its "
            "environment (sys.prefix, for example a project's .venv), the virtual environment the interpreter starts "
            "in (the directory holding the pyvenv.cfg site reads), even under -S, the base installation "
            "(sys.base_prefix, /usr for a system Python), Exegol's install directory, exegol-mcp's package directory, "
            "its user site directory for this start's HOME and PYTHONUSERBASE, even before it exists and even when a "
            "start disables it with -s, -S, -I or PYTHONNOUSERSITE (not in a virtual environment whose pyvenv.cfg, the "
            "one CPython reads next to the interpreter or one level up, sets include-system-site-packages to anything "
            "but true, which never reads it unless a start's PYTHONEXECUTABLE or __PYVENV_LAUNCHER__ (macOS) names "
            "another executable; an unreadable pyvenv.cfg counts as true), and every directory on its import path when "
            "it starts (sys.path: PYTHONPATH and .pth entries, and the MCP client's working directory under a `python "
            "-m exegol_mcp` or `python -c` launch typed by hand). The launch --print-config prints keeps that "
            "directory off the path: the exegol-mcp console script, resolved through symlinks, or without one `python "
            "-P -m exegol_mcp` on Python 3.11+ or a `python -c` launcher that drops it on Python 3.10. These "
            "directories are read when exegol-mcp starts: restart it after editing an allowlisted profile. Not "
            "checked: the WSLg display socket of a native Windows host; the Wayland socket of another environment (the "
            "exegol wrapper's); Exegol's /dev/net/tun device; a profile's network.hosts_file, which Exegol reads on "
            "the host; links placed in a profile source that point into an allowlisted directory; directories other "
            "than the user site added to Python's import path after startup, and code an import hook loads from "
            "outside it; a user site another start's HOME or PYTHONUSERBASE selects; a user site a start reaches "
            "through PYTHONEXECUTABLE or __PYVENV_LAUNCHER__ on macOS; another Python environment, such as the exegol "
            "wrapper's own; the Python installation or an import-path directory lying inside a directory Exegol "
            "mounts, which only the operator's own setup puts there (an installation or a PYTHONPATH or .pth entry "
            "placed there, or a hand-written `python -m` or `python -c` launch started there); any other directory "
            "whose contents this host runs or trusts (PATH directories, the directory of the interpreter a printed "
            "python launch names when it lies outside the Python installation, such as Homebrew's opt link, and "
            "~/.ssh); and directories other containers mount (the exegol wrapper's -V, a docker run), which exegol-mcp "
            "cannot see. Keep allowlisted directories apart from them. Empty means agents get only a private workspace."
        ),
    )


class FilesSettings(BaseModel):
    """File-transfer limits for read_file, write_file and list_directory (D-17, D-19)."""

    model_config = ConfigDict(extra="forbid")  # CONF-04: an unknown files key names itself at startup

    max_bytes: int = Field(
        default=10_485_760,
        gt=0,
        description="Largest file read_file or write_file transfers, in raw bytes before base64 (D-17, FILE-04)",
    )
    max_list_depth: int = Field(
        default=5,
        gt=0,
        description="Largest depth list_directory accepts (D-19)",
    )
    max_list_entries: int = Field(
        default=10_000,
        gt=0,
        description="Most entries list_directory collects before paging (D-19)",
    )


class OperatorConfig(BaseModel):
    """The whole operator config: every section above plus the mirrored launch keys."""

    model_config = ConfigDict(extra="forbid")  # CONF-04: an unknown top-level section names itself at startup

    execution: ExecutionLimits = Field(
        default_factory=ExecutionLimits,
        description="Execution bounds (timeouts, output caps, job limits, page size)",
    )

    security: SecuritySettings = Field(
        default_factory=SecuritySettings,
        description="Security policy opt-ins (the deliberate no-auth gate, SEC-04/D-13)",
    )

    # D-06: every launch flag is mirrored as an optional config key; None means "left to the CLI flag or the
    # built-in default", resolved with CLI > file > default precedence in bootstrap/config.py.
    transport: Literal["http", "stdio"] | None = Field(
        default=None,
        description="Transport to launch when --type is not given on the CLI",
    )
    port: int | None = Field(
        default=None,
        gt=0,
        description="HTTP port to bind when --port is not given on the CLI",
    )
    auth: Literal["none", "bearer"] | None = Field(
        default=None,
        description="HTTP auth mode when --auth is not given on the CLI",
    )
    update_check: bool | None = Field(
        default=None,
        description="Whether to check PyPI for a newer release when --no-update-check is not given on the CLI",
    )

    profiles: ProfilesSettings = Field(
        default_factory=ProfilesSettings,
        description="The Exegol profile allowlist and require_profile (CONF-01, D-01/D-03)",
    )
    lifecycle: LifecycleSettings = Field(
        default_factory=LifecycleSettings,
        description="Remove/cleanup defaults and the host-path workspace allowlist (D-11, D-12, D-24)",
    )
    files: FilesSettings = Field(
        default_factory=FilesSettings,
        description="File-transfer size limit and directory-listing caps (D-17, D-19)",
    )
    tool_groups: ToolGroupsSettings = Field(
        default_factory=ToolGroupsSettings,
        description="Tool-group gating: groups listed in tool_groups.disabled are not registered (CONF-02, D-23)",
    )

    @field_validator("profiles", "lifecycle", "files", "tool_groups", mode="before")
    @classmethod
    def _bare_section_means_defaults(cls, value: Any) -> Any:
        """A bare section key (``tool_groups:`` with no value) parses to None in YAML; treat it as the defaults."""
        return {} if value is None else value
