"""Opt-in live create -> use -> files -> remove suite against a real Exegol image (LIFE-01/03/06/07, FILE-01..04).

The whole module is skipped unless EXEGOL_MCP_LIVE_IMAGE names an installed Exegol image (its Exegol name, e.g.
`full`). When set, every test drives the REAL stdio entry point (`python -m exegol_mcp --type stdio`) with the
user's own HOME, config.yml and Docker, through the SDK's own MCP `Client` — never tests.fake_server.

One ordered scenario creates a container with a generated name and a private workspace, recognises it as managed
from a fresh server process (a new run id), verifies research assumption A1 (GNU realpath -m, stat -c, timeout,
head -c/-z and find -printf on the real image), moves files through every path the file tools promise (binary,
tmpfs, /proc, a symlink, a listing, the size limit), stops it and wakes it with auto_start, then removes it without
keeping its workspace or logs. The containers the user made by hand are snapshotted before and compared after, and
the suite never stops, removes or changes one: the hand-made-removal refusal check only runs when the operator's
config keeps lifecycle.allow_remove_unmanaged false, and the cleanup call uses the this_run scope only.

Just before the removal, the scenario writes a file into a new /workspace subdirectory and then times tools/list
from the same session for as long as remove_container runs (LIVE_REMOVAL_MAX_LATENCY_S). This is the CR-01 Linux
check: on a Linux host where exegol-mcp runs as a non-root docker-group user, those directories are root-owned, the
host cannot delete them, and Exegol falls back to deleting them from inside the container. On Docker Desktop, which
maps file ownership, the host delete succeeds and the step passes trivially.

A second test creates and removes a container from the Exegol profile named by EXEGOL_MCP_LIVE_PROFILE (it must be
in the operator's profiles.allowed, and a Pro licence may be needed); it also needs EXEGOL_MCP_LIVE_IMAGE set.

These scenarios cannot run under the Docker-free gate; they are the maintainer's backstop. Run them with:

    EXEGOL_MCP_LIVE_IMAGE=full uv run pytest tests/test_live_lifecycle.py -q
    EXEGOL_MCP_LIVE_IMAGE=full EXEGOL_MCP_LIVE_PROFILE=<allowlisted profile> uv run pytest tests/test_live_lifecycle.py -q
"""

import base64
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, StdioServerParameters

from exegol_mcp.bootstrap.config import load_operator_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# The slowest tools/list allowed while a real removal runs (CR-01). Generous for real docker API round trips; a
# removal that ran Exegol's in-container delete on the event loop would hold tools/list for the whole delete.
LIVE_REMOVAL_MAX_LATENCY_S = 5.0

_IMAGE = os.environ.get("EXEGOL_MCP_LIVE_IMAGE")
_PROFILE = os.environ.get("EXEGOL_MCP_LIVE_PROFILE")

pytestmark = pytest.mark.skipif(
    not _IMAGE,
    reason="opt-in live suite: set EXEGOL_MCP_LIVE_IMAGE=<installed Exegol image name, e.g. full> to run",
)

# A1: every GNU coreutils/findutils feature the file tools rely on, in one /bin/sh script that fails on the first
# missing one (set -e). Each probe prints a known line so a silently different implementation is caught too.
_A1_PROBE = (
    "set -e\n"
    'for tool in realpath stat timeout head find; do command -v "$tool"; done\n'
    "realpath -m /nonexistent/x\n"
    "find /etc -maxdepth 0 -printf '%y\\n'\n"
    "stat -c 'stat:%F' /etc\n"
    "printf 'z1\\0z2\\0' | head -z -n 1 | head -c 2; echo\n"
    "timeout 5 true\n"
    "echo a1-ok\n"
)


def _server_parameters() -> StdioServerParameters:
    """SDK stdio-client parameters for the REAL entry point (the user's own HOME, config, adapter and Docker)."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "exegol_mcp", "--type", "stdio", "--no-update-check"],
        env=dict(os.environ),
        cwd=str(REPO_ROOT),
    )


@asynccontextmanager
async def _session() -> AsyncIterator[Client]:
    """Open a fresh MCP client session against a freshly launched real server (a new run id) over stdio."""
    async with Client(_server_parameters()) as client:
        yield client


def _text(result: Any) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


async def _call(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool, assert it did not error, and return its structured content."""
    result = await client.call_tool(name, arguments)
    assert not result.is_error, (name, result.content)
    assert result.structured_content is not None, (name, result.content)
    return result.structured_content


async def _call_error(client: Client, name: str, arguments: dict[str, Any]) -> str:
    """Call a tool that must fail, and return its error text."""
    result = await client.call_tool(name, arguments)
    assert result.is_error, (name, result.structured_content)
    return _text(result)


async def _containers(client: Client, managed_only: bool = False) -> list[dict[str, Any]]:
    payload = await _call(client, "list_containers", {"managed_only": managed_only})
    containers = payload["result"]
    assert isinstance(containers, list), payload
    return containers


async def _unmanaged_snapshot(client: Client) -> dict[str, tuple[str, str | None]]:
    """Name -> (status, comment) of every container exegol-mcp did not create: the user's hand-made ones."""
    return {c["name"]: (c["status"], c["comment"]) for c in await _containers(client) if not c["managed"]}


async def _exec(client: Client, container: str, command: str, **extra: Any) -> dict[str, Any]:
    payload = await _call(client, "exec_command", {"container_name": container, "command": command, **extra})
    assert payload["timed_out"] is False, payload
    assert payload["exit_code"] == 0, payload
    return payload


def _docker_exists(name: str) -> bool:
    """Read-only: True if Docker still knows the exact container exegol-<name>."""
    out = subprocess.run(
        ["docker", "container", "inspect", f"exegol-{name}"], capture_output=True, text=True, check=False
    )
    return out.returncode == 0


def _docker_force_remove(name: str) -> None:
    """Last resort for the exact container exegol-<name> this suite created, never any other."""
    subprocess.run(["docker", "rm", "-f", f"exegol-{name}"], capture_output=True, check=False)


async def _ensure_removed(name: str) -> None:
    """Remove the container this suite created, and only that one, whatever state a failure left it in.

    The tool removal comes first (no workspace or logs kept). `docker rm -f` is the last resort, and only for the
    exact name the suite created.
    """
    if not _docker_exists(name):
        return
    try:
        async with _session() as client:
            await client.call_tool(
                "remove_container", {"container_name": name, "keep_workspace": False, "keep_logs": False}
            )
    finally:
        if _docker_exists(name):
            _docker_force_remove(name)


async def _read(client: Client, container: str, path: str, encoding: str = "auto") -> dict[str, Any]:
    return await _call(client, "read_file", {"container_name": container, "path": path, "encoding": encoding})


async def _write(client: Client, container: str, path: str, content: str, encoding: str = "utf-8") -> dict[str, Any]:
    return await _call(
        client,
        "write_file",
        {"container_name": container, "path": path, "content": content, "encoding": encoding},
    )


async def _list_all(client: Client, container: str, path: str) -> list[dict[str, Any]]:
    """Every entry of a one-level listing, following next_cursor through every page."""
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        arguments: dict[str, Any] = {"container_name": container, "path": path}
        if cursor is not None:
            arguments["cursor"] = cursor
        page = await _call(client, "list_directory", arguments)
        assert isinstance(page["items"], list), page
        items.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            return items


async def _files_scenario(client: Client, name: str, max_bytes: int) -> None:
    """Step 5: every file path the tools promise, on the real container."""
    # Binary bytes round-trip exactly through base64.
    data = bytes(range(256))
    written = await _write(client, name, "/tmp/mcp-live.bin", base64.b64encode(data).decode(), encoding="base64")
    assert written["size"] == 256, written
    read_back = await _read(client, name, "/tmp/mcp-live.bin", encoding="base64")
    assert read_back["encoding"] == "base64", read_back
    assert base64.b64decode(read_back["content"]) == data

    # A tmpfs mount (docker get_archive returns 404 there; the exec transfer does not).
    await _write(client, name, "/dev/shm/mcp-live.txt", "tmpfs round trip\n")
    shm = await _read(client, name, "/dev/shm/mcp-live.txt")
    assert (shm["encoding"], shm["content"]) == ("utf-8", "tmpfs round trip\n"), shm

    # A /proc file (stat reports size 0; the read still returns its content).
    proc = await _read(client, name, "/proc/version")
    assert proc["encoding"] == "utf-8", proc
    assert proc["content"].strip(), proc

    # A write through a symlink lands at the target.
    await _exec(client, name, "rm -f /tmp/mcp-target /tmp/mcp-link && ln -s /tmp/mcp-target /tmp/mcp-link")
    through = await _write(client, name, "/tmp/mcp-link", "written through the link\n")
    assert through["path"] == "/tmp/mcp-target", through
    target = await _read(client, name, "/tmp/mcp-target")
    assert target["content"] == "written through the link\n", target

    # A directory listing, sorted by relative path across every page.
    entries = await _list_all(client, name, "/etc")
    assert "passwd" in {entry["name"] for entry in entries}
    paths = [entry["path"] for entry in entries]
    assert paths == sorted(paths)

    # One byte over the operator's files.max_bytes is refused, whatever the config sets.
    error = await _call_error(
        client,
        "write_file",
        {"container_name": name, "path": "/tmp/mcp-live-too-big.txt", "content": "a" * (max_bytes + 1)},
    )
    assert "files.max_bytes" in error, error


async def test_full_lifecycle_create_use_files_auto_start_remove() -> None:
    assert _IMAGE
    config = load_operator_config()
    created: str | None = None
    try:
        # 1-2. Snapshot the user's hand-made containers, then create one with a generated name and private workspace.
        async with _session() as first:
            images = await _call(first, "list_images", {})
            installed = {image["name"] for image in images["result"] if image["is_installed"]}
            assert _IMAGE in installed, f"EXEGOL_MCP_LIVE_IMAGE={_IMAGE!r} is not installed (installed: {installed})"
            before = await _unmanaged_snapshot(first)

            info = await _call(first, "create_container", {"image": _IMAGE})
            created = info["name"]
            assert created.startswith("mcp-"), info
            assert info["managed"] is True, info
            assert info["this_run"] is True, info
            assert info["status"].lower() == "running", info
            first_run_id = info["run_id"]
            assert first_run_id, info

        async with _session() as second:
            # 3. A fresh server process (a new run id) still recognises it as managed, from the first run (LIFE-06).
            managed = {c["name"]: c for c in await _containers(second, managed_only=True)}
            assert created in managed, sorted(managed)
            assert managed[created]["managed"] is True
            assert managed[created]["this_run"] is False
            assert managed[created]["run_id"] == first_run_id

            # 4. A1: the GNU tools the file transfer relies on are on the real image.
            probe = await _exec(second, created, _A1_PROBE)
            lines = probe["stdout"].splitlines()
            assert "/nonexistent/x" in lines, probe
            assert "d" in lines, probe
            assert "stat:directory" in lines, probe
            assert "z1" in lines, probe
            assert lines[-1] == "a1-ok", probe

            # 5. Files (FILE-01..04).
            await _files_scenario(second, created, config.files.max_bytes)

            # 6. Stopped, then woken by auto_start on the next exec (LIFE-07).
            await _call(second, "stop_container", {"container_name": created})
            stopped = {c["name"]: c for c in await _containers(second, managed_only=True)}
            assert stopped[created]["status"].lower() != "running", stopped[created]
            await _exec(second, created, "true", auto_start=True)

            # 7. A hand-made container is refused, and left exactly as it was (D-11).
            if not config.lifecycle.allow_remove_unmanaged and before:
                victim = min(before)
                error = await _call_error(second, "remove_container", {"container_name": victim})
                assert "not created by exegol-mcp" in error, error
                assert (await _unmanaged_snapshot(second)).get(victim) == before[victim]
            else:
                print(
                    "skipped the hand-made-removal refusal check: "
                    + (
                        "lifecycle.allow_remove_unmanaged is true in config.yml"
                        if config.lifecycle.allow_remove_unmanaged
                        else "this host has no hand-made Exegol container"
                    )
                )

            # 8a. The CR-01 Linux-ownership path: the container creates these directories as root, so a non-root
            # docker-group host user cannot delete them and Exegol deletes them from inside the container.
            await _call(
                second,
                "write_file",
                {
                    "container_name": created,
                    "path": "/workspace/mcp-live-owned/nested/data.txt",
                    "content": "owned by the container's root\n",
                    "auto_start": True,
                },
            )

            # 8. Removed without keeping its workspace or logs, and gone from the listing (LIFE-03), while tools/list
            # from the same session keeps answering for the whole removal (CR-01).
            removal: dict[str, Any] = {}
            latencies: list[float] = []

            async def remove() -> None:
                removal["result"] = await _call(
                    second,
                    "remove_container",
                    {"container_name": created, "keep_workspace": False, "keep_logs": False},
                )

            async with anyio.create_task_group() as tg:
                tg.start_soon(remove)
                while "result" not in removal:
                    started = time.perf_counter()
                    await second.list_tools()
                    latencies.append(time.perf_counter() - started)
                    await anyio.sleep(0.1)
            assert latencies and max(latencies) < LIVE_REMOVAL_MAX_LATENCY_S, latencies
            removed = removal["result"]
            assert removed == {"name": created, "workspace_kept": False, "logs_kept": False}, removed
            assert created not in {c["name"] for c in await _containers(second)}

            # 9. Nothing is left for an explicit cleanup of this run either (D-14).
            cleanup = await _call(second, "cleanup_containers", {})
            assert cleanup["removed"] == [], cleanup
            assert cleanup["failed"] == [], cleanup

            # 10. Every hand-made container is listed unchanged.
            assert await _unmanaged_snapshot(second) == before
    finally:
        if created is not None:
            await _ensure_removed(created)
    assert created is not None and not _docker_exists(created)


@pytest.mark.skipif(
    not _PROFILE,
    reason="opt-in profile check: set EXEGOL_MCP_LIVE_PROFILE=<a profile in the operator's profiles.allowed> to run",
)
async def test_create_from_an_allowlisted_profile_then_remove() -> None:
    assert _PROFILE
    created: str | None = None
    try:
        async with _session() as client:
            info = await _call(client, "create_container", {"profile": _PROFILE})
            created = info["name"]
            assert info["managed"] is True, info
            assert info["this_run"] is True, info
            assert info["status"].lower() == "running", info

            removed = await _call(
                client, "remove_container", {"container_name": created, "keep_workspace": False, "keep_logs": False}
            )
            assert removed["name"] == created, removed
            assert created not in {c["name"] for c in await _containers(client)}
    finally:
        if created is not None:
            await _ensure_removed(created)
    assert created is not None and not _docker_exists(created)
