"""Console-script entry point: wire the bootstrap layers into a running server (COMP-01/COMP-02).

main() is the only thing here; everything it calls lives in exegol_mcp.bootstrap and exegol_mcp.adapter. It keeps
the stable launch interface: the `exegol-mcp` console script, -t/--type, -p/--port, --auth, -pc/--print-config,
-v, and the defaults (HTTP on 127.0.0.1:2187/mcp). The read-only `version` action prints the versions and returns
before anything below starts.

Order matters (RESEARCH Pitfall 2, plan 04 notes):
1. parse the command line;
2. route every diagnostic to the standard error stream before any Exegol SDK module is imported (SEC-08), in the
   Exegol wrapper's "[*] text" style, at the wrapper's verbosity levels (-v verbose, -vv advanced, -vvv debug);
2b. `version`: print the read-only report and return (no self-check, config load, secret or gate);
3. run the D-15 self-check, then route the SDK's console; a missing or renamed SDK name stops startup here with its
   named message on one stderr line (D-15, WR-12, WR-13);
4. load the operator config (-v names the file in play);
5. patch the SDK headless and apply --accept-eula;
6. run the startup readiness gate (patch points/Docker/config/profiles/workspace allowlist/EULA), which may prompt
   for the EULA on a terminal;
7. only then gather versions and start a transport. The serving line (the facts from -v) and the client config on the
   first HTTP run only (a --print-config hint on later HTTP starts, nothing in stdio) print once HTTP has bound its
   port (GR-IN-01), or right before stdio runs.

Importing this module imports no exegol module (adapter/sudo.py and adapter/selfcheck.py import the SDK inside
their functions), so no SDK import statement runs before step 3.
"""

import asyncio
import logging
import os
import sys
from argparse import Namespace

from exegol_mcp.adapter import selfcheck  # loads no exegol module: assert_patch_points() imports them
from exegol_mcp.adapter.errors import PatchPointError
from exegol_mcp.app import build_server
from exegol_mcp.bootstrap import secret_store, version_command, versions
from exegol_mcp.bootstrap.cli import parse_args
from exegol_mcp.bootstrap.config import CONFIG_FILE_NAME, ConfigError, load_operator_config, resolve
from exegol_mcp.bootstrap.lifecycle import ServerContext
from exegol_mcp.bootstrap.logging_setup import VERBOSE, configure_stderr_logging, route_logging_to_stderr
from exegol_mcp.bootstrap.readiness import ReadinessError, ReadinessReport, check_readiness
from exegol_mcp.bootstrap.transports import run_http, run_stdio

_log = logging.getLogger(__name__)


def _load_secret(args: Namespace) -> tuple[str | None, bool]:
    """Load the bearer secret once for HTTP bearer mode; (None, False) otherwise."""
    if args.type == "http" and args.auth == "bearer":
        secret, created, location = secret_store.load_bearer_secret()
        if location:
            # An operator fact -v showed before (D-17): where the secret lives, never its value (SEC-02).
            _log.log(VERBOSE, f"Bearer secret stored at '{location}'")
        return secret, created
    return None, False


def main(argv: list[str] | None = None) -> int:
    """Start the Exegol MCP server. Returns the process exit status."""
    args = parse_args(argv)

    # 2. Diagnostics to stderr before any Exegol SDK module is imported, so Exegol's import-time logging setup finds
    # a root handler and never binds one to stdout (SEC-08).
    route_logging_to_stderr(args.verbose)

    # 2b. `version` is read-only and runs before the self-check, the config load, the SDK patch, the secret and the
    # readiness gate, so it works with Docker down, a broken config.yml or no EULA (D-08). Its report goes to stdout
    # because no server runs, so stdout is not a protocol stream (R1).
    if args.action == "version":
        return version_command.run_version(args)

    # 3. The self-check runs before configure_stderr_logging imports sdk_headless and its module-level SDK imports,
    # so a missing SDK name is one named stderr line and exit status 1 (D-15, WR-13). Only PatchPointError is
    # caught; any other exception keeps its traceback (T-04-84).
    try:
        selfcheck.assert_patch_points()
        configure_stderr_logging(args.verbose)
    except PatchPointError as e:
        _log.error(str(e))
        return 1

    # 4. Load the operator config right away, before the SDK patch or the readiness gate: a present-but-invalid
    # config.yml must fail fast with one clear stderr line naming the field (CONF-04), never a traceback, and stdio
    # stdout stays clean. A missing/empty file yields built-in defaults (D-03).
    try:
        file_cfg = load_operator_config()
    except ConfigError as e:
        _log.error(str(e))
        return 1
    # -v names the config file in play (D-17): a stat only, the content is never logged.
    config_path = secret_store.user_config_dir() / CONFIG_FILE_NAME
    if os.path.lexists(config_path):
        _log.log(VERBOSE, f"Loaded the operator config from '{config_path}'")
    else:
        _log.log(VERBOSE, f"No operator config at '{config_path}': built-in defaults")

    # Resolve the mirrored launch keys with CLI > file > built-in default precedence (D-05/D-06), then write the
    # concrete transport/port/auth back onto the Namespace so the readiness gate, banner, client config and
    # transport all read resolved values (the mirrored flags default to None so the file can supply them).
    config = resolve(args, file_cfg)
    args.type = config.transport
    args.port = config.port
    args.auth = config.auth

    # 5. The patch runs the D-15 self-check again. A missing or renamed SDK internal is one clear stderr line and
    # exit status 1, like every other startup refusal; any other exception keeps its traceback (WR-12).
    try:
        from exegol_mcp.adapter import sdk_headless  # imports exegol; only the adapter may

        sdk_headless.patch_exegol_sdk()
        sdk_headless.set_accept_eula(args.accept_eula)
    except PatchPointError as e:
        _log.error(str(e))
        return 1

    from exegol_mcp.adapter.host import RealHostProbe
    from exegol_mcp.adapter.real import RealExegolAdapter

    probe = RealHostProbe()

    # 6. Readiness gate (offloaded; prompts for the EULA only on a terminal, never in stdio). A failure is a clear
    # message on stderr and a non-zero exit, not a traceback.
    try:
        report: ReadinessReport = asyncio.run(
            check_readiness(
                probe,
                args,
                profiles=config.profiles.allowed,
                workspace_allowlist=config.lifecycle.workspace_allowlist,
            )
        )
    except ReadinessError as e:
        _log.error(str(e))
        return 1

    # 7. Versions + banner. --print-config prints the client template (after the EULA prompt) and exits.
    collected = versions.gather_versions(report.docker_status, report.exegol_config_dir)
    secret, created = _load_secret(args)

    if args.print_config:
        versions.print_client_config(args, secret, created)
        return 0

    update_checker = versions.UpdateChecker(enabled=bool(config.update_check))
    context = ServerContext(
        RealExegolAdapter(),
        collected,
        update_checker,
        probe,
        report.docker_status,
        accept_eula=args.accept_eula,
        config=config,
    )
    server = build_server(config, context)

    announced = False

    def announce_start() -> None:
        """Print what a start says (the serving line, the facts, the client config) and start the update check."""
        nonlocal announced
        announced = True
        versions.print_banner(args, collected, update_check=bool(config.update_check))
        versions.print_startup_client_config(args, secret, created)
        update_checker.start()

    if args.type == "http":
        # GR-IN-01: the serving line and the first-run token are printed once the port is bound, never before: a
        # start that cannot bind prints the bind error and neither of them. D-30 hides uvicorn's own "running on"
        # line at the default level, so this line is the one success statement a start makes.
        try:
            run_http(
                server,
                port=args.port,
                auth=args.auth,
                secret=secret,
                allow_no_auth=config.security.allow_no_auth,
                verbose=args.verbose,
                on_bound=announce_start,
            )
        finally:
            if created and not announced:
                # The secret was created by this start but its first-run block never printed (a later start is not
                # "created" any more): point at --print-config, which prints the entry with the token.
                versions.print_unshown_secret_hint(args)
    elif args.type == "stdio":
        # Nothing to bind: stdio prints at once (nothing at the default level, SEC-08), before the server runs.
        announce_start()
        run_stdio(server)
    else:  # pragma: no cover - argparse restricts --type to the two choices
        raise NotImplementedError(f"Unknown server type: {args.type}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m exegol_mcp`
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
