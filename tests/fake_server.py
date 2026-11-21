"""Docker-free launcher for the real exegol-mcp entry point, shared by the subprocess tests (D-24, SEC-08, COMP-01/02).

`python -m tests.fake_server <exegol-mcp arguments>` runs exactly `exegol_mcp.main.main()` (argument parsing,
stderr logging, the headless SDK patch, the readiness gate, the banner, the transport), with only the pieces that
reach the host swapped for test doubles:

- RealHostProbe -> _LauncherProbe: Docker reachable, config readable, EULA accepted. No Docker call, no datastore.
- RealExegolAdapter -> FakeExegolAdapter: the in-memory port, with an optional blocking exec.
- secret_store.load_bearer_secret -> the secret from EXEGOL_MCP_TEST_SECRET. Without it, bearer mode refuses to
  start rather than reaching the real keyring or secret file.
- versions.fetch_latest_version -> None, so no PyPI request is ever made (the harness also passes
  --no-update-check).

The selection lives here, in the test tree, rather than as an environment switch in the package: a shipped server
never has a way to swap its adapter for a fake.

Environment knobs (all optional):
- EXEGOL_MCP_TEST_EXEC_BLOCK: seconds the fake exec blocks (default 0).
- EXEGOL_MCP_TEST_OFFLOAD: "0" runs the blocking exec on the event loop (the v1 bug), for the inversion checks.
- EXEGOL_MCP_TEST_SECRET: the bearer secret for an HTTP launch.
- EXEGOL_MCP_TEST_NOISE: "1" makes the probe and the adapter write marker lines through the Exegol SDK logger, the
  SDK console, an mcp.* logger and (during a tool call) a bare print(), so the stdout-purity test can check where
  each one lands.

The test process passes a temporary HOME/XDG_CONFIG_HOME (see server_env), so even the SDK's own config dir
resolution stays out of the user's real ~/.exegol and exegol-mcp config dir.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mcp import StdioServerParameters

from exegol_mcp.adapter.port import ConfigStatus
from exegol_mcp.models.container import ContainerInfo
from tests.fakes import FakeExegolAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
# Marker prefix every deliberate diagnostic line carries, so tests can find them on stderr.
NOISE_MARKER = "EXEGOL-MCP-TEST-NOISE"


def emit_noise(where: str, *, include_print: bool = False) -> None:
    """Write one marker line through every diagnostic channel that could reach stdout if it were misrouted."""
    # Imported lazily: by the time this runs, main() has patched the SDK and routed logging to stderr.
    from exegol.utils.ExeLog import ExeLog, logger

    logger.warning(f"{NOISE_MARKER} exegol-logger {where}")
    logger.info(f"{NOISE_MARKER} exegol-logger-info {where}")
    ExeLog.console.print(f"{NOISE_MARKER} exegol-console {where}")
    logging.getLogger("mcp.server").warning(f"{NOISE_MARKER} mcp-logger {where}")
    logging.getLogger("exegol_mcp").warning(f"{NOISE_MARKER} package-logger {where}")
    if include_print:
        # A stray print() from a dependency during a session: the SDK diverts fd 1 while serving stdio.
        print(f"{NOISE_MARKER} bare-print {where}", flush=True)


def _noisy() -> bool:
    return os.environ.get("EXEGOL_MCP_TEST_NOISE") == "1"


class _LauncherProbe:
    """HostProbe for a healthy host: Docker reachable, config readable, EULA accepted."""

    def assert_patch_points(self) -> None:
        # The real SDK self-check: it only inspects the installed exegol package, it never touches Docker.
        from exegol_mcp.adapter.selfcheck import assert_patch_points

        assert_patch_points()

    def probe_docker(self) -> str:
        if _noisy():
            emit_noise("readiness")
        return "reachable (fake Docker)"

    def ping_docker(self) -> str:
        return "reachable (fake Docker)"

    def load_config(self) -> ConfigStatus:
        return ConfigStatus(path=str(Path.home() / ".exegol" / "config.yml"), refused=False)

    def exegol_config_dir(self) -> str:
        return str(Path.home() / ".exegol")

    def validate_profiles(self, names) -> list[str]:
        return []

    def is_eula_accepted(self) -> bool:
        return True

    def update_eula(self, accepted: bool) -> None:  # pragma: no cover - never reached: the EULA is accepted
        raise AssertionError("the test launcher must never write the EULA datastore")


class _LauncherAdapter(FakeExegolAdapter):
    """The fake port, optionally noisy on list_containers so SDK-style output happens mid-session."""

    async def list_containers(self) -> list[ContainerInfo]:
        if _noisy():
            emit_noise("tool-call", include_print=True)
        return await super().list_containers()


def _refuse_real_secret_store(backend: str = "auto") -> tuple[str, bool, str | None]:
    raise RuntimeError(
        "tests.fake_server: bearer mode needs EXEGOL_MCP_TEST_SECRET; the real secret store is off limits"
    )


def install_doubles() -> FakeExegolAdapter:
    """Swap the host-facing classes main() uses for the test doubles. Returns the adapter the server will use."""
    import exegol_mcp.adapter.host as host_module
    import exegol_mcp.adapter.real as real_module
    from exegol_mcp.bootstrap import secret_store, versions

    adapter = _LauncherAdapter(
        exec_block_seconds=float(os.environ.get("EXEGOL_MCP_TEST_EXEC_BLOCK", "0")),
        offload_exec=os.environ.get("EXEGOL_MCP_TEST_OFFLOAD", "1") != "0",
    )
    host_module.RealHostProbe = _LauncherProbe  # type: ignore[misc,assignment]
    real_module.RealExegolAdapter = lambda: adapter  # type: ignore[misc,assignment]

    secret = os.environ.get("EXEGOL_MCP_TEST_SECRET")
    if secret:
        secret_store.load_bearer_secret = lambda backend="auto": (secret, False, None)  # type: ignore[assignment]
    else:
        secret_store.load_bearer_secret = _refuse_real_secret_store  # type: ignore[assignment]
    versions.fetch_latest_version = lambda timeout=0.0: None  # type: ignore[assignment]
    return adapter


def server_env(home: Path, **knobs: str) -> dict[str, str]:
    """Environment for a launcher subprocess: an isolated HOME and config dir, plus the given EXEGOL_MCP_TEST_* knobs.

    Args:
        home: A temporary directory used as HOME (and the XDG config base) for the subprocess.
        knobs: Extra variables, e.g. EXEGOL_MCP_TEST_EXEC_BLOCK="2".
    """
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
        # A narrow terminal width keeps rich from probing the (absent) terminal.
        "COLUMNS": "200",
    }
    for name in ("SYSTEMROOT", "TMPDIR", "LANG", "LC_ALL"):
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(knobs)
    return env


def launcher_command(*server_args: str) -> list[str]:
    """argv that starts the real entry point through this launcher (always with --no-update-check)."""
    return [sys.executable, "-m", "tests.fake_server", *server_args, "--no-update-check"]


def stdio_args() -> tuple[str, ...]:
    """The exegol-mcp arguments of a stdio launch, as an MCP client config would pass them."""
    return ("--type", "stdio", "--accept-eula")


def stdio_server_parameters(home: Path, **knobs: str) -> StdioServerParameters:
    """SDK stdio-client parameters launching the real entry point over stdio through this launcher."""
    command = launcher_command(*stdio_args())
    return StdioServerParameters(command=command[0], args=command[1:], env=server_env(home, **knobs), cwd=REPO_ROOT)


if __name__ == "__main__":
    install_doubles()
    from exegol_mcp.main import main

    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        pass
