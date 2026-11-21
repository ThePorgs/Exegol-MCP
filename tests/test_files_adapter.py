"""Exec-based container file transfer (adapter/files.py): stub-handle tests and Linux-only real-shell tests.

Every in-container file command is a fixed `/bin/sh -c <script> sh <path> ...` argv: the path is a positional
parameter, never formatted into the script (T-04-29). Reads are a probe exec plus a bounded `head -c` exec whose raw
stdout bytes are kept as they are (FILE-01). Writes stage through put_archive into a fresh /.exegol-mcp-<16 hex>/
directory, then one exec copies the bytes with shell-redirection semantics and checks the size (FILE-02, T-04-31).

The contract tests run on every host against a docker-py call-shape double. The Linux-only tests run the real
scripts under the host's /bin/sh against a tmp directory standing in for the container; they need GNU coreutils and
findutils, so they are skipped elsewhere (macOS included). Plan 04-12's opt-in live suite covers a real container.
"""

import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from docker.errors import APIError
from exegol.exceptions.ExegolExceptions import ObjectNotFound

from exegol_mcp.adapter import files, real
from exegol_mcp.adapter.port import DirListing, DirRecord, FileRead, FileWrite
from exegol_mcp.adapter.real import RealExegolAdapter

_GNU_TOOLS = ("realpath", "stat", "timeout", "head", "find", "wc", "cat", "mkdir", "chmod", "rm")
_NEEDS_LINUX = not (sys.platform.startswith("linux") and all(shutil.which(tool) for tool in _GNU_TOOLS))
_LINUX_ONLY = pytest.mark.skipif(_NEEDS_LINUX, reason="needs Linux with GNU coreutils/findutils under /bin/sh")

_STAGING_RE = re.compile(r"^\.exegol-mcp-[0-9a-f]{16}$")

# The exact exec_run keyword arguments: demuxed, non-TTY, and no environment.
_EXEC_KWARGS = {"detach": False, "stream": False, "demux": True, "tty": False}

Reply = tuple[int | None, tuple[bytes | None, bytes | None]] | BaseException


# --- docker-py call-shape double -------------------------------------------------------------------------------


class _Docker:
    """Records exec_run argv/kwargs and put_archive calls, answering exec_run from a queue of replies."""

    def __init__(self, replies: list[Reply], put_error: BaseException | None = None) -> None:
        self._replies = list(replies)
        # docker-py 7.1 put_archive raises on any failure before it returns, so a refusal is an exception here too.
        self.put_error = put_error
        self.exec_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.puts: list[tuple[str, bytes]] = []

    def exec_run(self, cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        self.exec_calls.append((list(cmd), kwargs))
        reply = self._replies.pop(0) if self._replies else (0, (None, None))
        if isinstance(reply, BaseException):
            raise reply
        exit_code, output = reply
        return SimpleNamespace(exit_code=exit_code, output=output)

    def put_archive(self, path: str, data: bytes) -> bool:
        self.puts.append((path, data))
        if self.put_error is not None:
            raise self.put_error
        return True

    def scripts(self) -> list[str]:
        return [argv[2] for argv, _ in self.exec_calls]


def _container(docker: _Docker, running: bool = True, name: str = "lab") -> SimpleNamespace:
    # ExegolContainer keeps the docker-py container in a private, name-mangled attribute.
    return SimpleNamespace(_ExegolContainer__container=docker, name=name, isRunning=lambda: running)


def _ok(stdout: bytes = b"", stderr: bytes | None = None, code: int | None = 0) -> Reply:
    # docker-py reports a stream that wrote nothing as None.
    return code, (stdout or None, stderr)


def _nul(*fields: str) -> bytes:
    return b"".join(field.encode() + b"\0" for field in fields)


def _probe(path: str, kind: str, size: int) -> Reply:
    return _ok(_nul(path, kind, str(size)))


def _prep(path: str, status: str, existed: str = "0") -> Reply:
    return _ok(_nul(path, status, existed))


# --- read_container_file --------------------------------------------------------------------------------------


async def test_the_probe_gets_the_path_as_a_positional_argument_with_no_environment() -> None:
    docker = _Docker([_probe("/etc/x", "file", 3), _ok(b"abc")])

    await files.read_container_file(_container(docker), "/etc/$(touch x)", 10, 30.0)

    argv, kwargs = docker.exec_calls[0]
    assert argv[:5] == ["/bin/sh", "-c", files._PROBE_SCRIPT, "sh", "/etc/$(touch x)"]
    assert "/etc/$(touch x)" not in argv[2]
    assert kwargs == _EXEC_KWARGS


async def test_a_read_returns_the_raw_stdout_bytes_unchanged() -> None:
    docker = _Docker([_probe("/etc/bin", "file", 3), _ok(b"\x00\xff\xfe")])

    result = await files.read_container_file(_container(docker), "/etc/../etc/bin", 10, 30.0)

    assert result == FileRead(data=b"\x00\xff\xfe", size=3, over_limit=False, path="/etc/bin")
    argv, kwargs = docker.exec_calls[1]
    assert argv[:3] == ["/bin/sh", "-c", files._READ_SCRIPT]
    assert "11" in argv[4:]
    assert kwargs == _EXEC_KWARGS


async def test_a_file_larger_than_the_limit_is_over_limit_with_the_probe_size() -> None:
    docker = _Docker([_probe("/big", "file", 100), _ok(b"12345")])

    result = await files.read_container_file(_container(docker), "/big", 4, 30.0)

    assert result.over_limit is True
    assert result.size == 100
    assert result.data == b"12345"
    assert "5" in docker.exec_calls[1][0][4:]


async def test_a_pseudo_file_with_a_zero_probe_size_is_still_read_up_to_the_limit() -> None:
    docker = _Docker([_probe("/proc/cpuinfo", "file", 0), _ok(b"cpu\n")])

    result = await files.read_container_file(_container(docker), "/proc/cpuinfo", 10, 30.0)

    assert "11" in docker.exec_calls[1][0][4:]
    assert result == FileRead(data=b"cpu\n", size=4, over_limit=False, path="/proc/cpuinfo")


async def test_an_over_limit_pseudo_file_has_an_unknown_size() -> None:
    docker = _Docker([_probe("/proc/kallsyms", "file", 0), _ok(b"x" * 11)])

    result = await files.read_container_file(_container(docker), "/proc/kallsyms", 10, 30.0)

    assert (result.over_limit, result.size) == (True, None)


@pytest.mark.parametrize(
    ("kind", "match"),
    [("missing", "No such file"), ("directory", "list_directory"), ("fifo", "FIFO"), ("socket", "socket")],
)
async def test_a_target_that_is_not_a_readable_file_is_refused_before_any_read(kind: str, match: str) -> None:
    docker = _Docker([_probe("/x", kind, 0)])

    with pytest.raises(ValueError, match=match):
        await files.read_container_file(_container(docker), "/x", 10, 30.0)
    assert len(docker.exec_calls) == 1


async def test_a_read_that_hits_the_in_container_time_limit_is_a_runtime_error() -> None:
    docker = _Docker([_probe("/dev/slow", "file", 0), _ok(code=137)])

    with pytest.raises(RuntimeError, match="time limit"):
        await files.read_container_file(_container(docker), "/dev/slow", 10, 30.0)


async def test_a_probe_that_hits_the_time_limit_is_a_runtime_error() -> None:
    docker = _Docker([_ok(code=137)])

    with pytest.raises(RuntimeError, match="time limit"):
        await files.read_container_file(_container(docker), "/x", 10, 30.0)


# --- write_container_file -------------------------------------------------------------------------------------


@pytest.mark.parametrize(("overwrite", "flag"), [(True, "1"), (False, "0")])
async def test_the_write_prep_gets_the_path_and_the_overwrite_flag_as_positional_arguments(
    overwrite: bool, flag: str
) -> None:
    docker = _Docker([_prep("/root/a", "ok"), _ok()])

    await files.write_container_file(_container(docker), "/root/$(id)", b"hi", overwrite, None, 30.0)

    argv, kwargs = docker.exec_calls[0]
    assert argv[:3] == ["/bin/sh", "-c", files._WRITE_PREP_SCRIPT]
    assert argv[4:6] == ["/root/$(id)", flag]
    assert "/root/$(id)" not in argv[2]
    assert kwargs == _EXEC_KWARGS


@pytest.mark.parametrize(
    ("status", "overwrite", "match"),
    [
        ("directory", True, "directory"),
        ("exists", False, "overwrite"),
        ("fifo", True, "FIFO"),
        ("socket", True, "socket"),
    ],
)
async def test_a_refused_write_target_stages_nothing(status: str, overwrite: bool, match: str) -> None:
    docker = _Docker([_prep("/root/a", status, "1")])

    with pytest.raises(ValueError, match=match):
        await files.write_container_file(_container(docker), "/root/a", b"hi", overwrite, None, 30.0)
    assert docker.puts == []
    assert len(docker.exec_calls) == 1


async def test_a_write_stages_a_private_tar_on_the_rootfs_then_copies_with_a_size_check() -> None:
    docker = _Docker([_prep("/root/a", "ok"), _ok()])

    await files.write_container_file(_container(docker), "/root/a", b"\x00\xffhi", True, None, 30.0)

    [(dest, archive)] = docker.puts
    assert dest == "/"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
        members = {member.name: member for member in tar.getmembers()}
        [dir_name] = [name for name in members if "/" not in name]
        assert _STAGING_RE.match(dir_name)
        directory, data = members[dir_name], members[f"{dir_name}/data"]
        assert directory.isdir() and directory.mode == 0o700 and directory.uid == 0
        assert data.isfile() and data.mode == 0o600 and (data.uid, data.gid) == (0, 0) and data.mtime > 0
        extracted = tar.extractfile(data)
        assert extracted is not None and extracted.read() == b"\x00\xffhi"
    argv, kwargs = docker.exec_calls[1]
    assert argv[:3] == ["/bin/sh", "-c", files._COPY_SCRIPT]
    assert argv[4:8] == ["/root/a", f"/{dir_name}", "", "4"]
    assert kwargs == _EXEC_KWARGS
    # The copy completed, so its own trap removed the staging directory: no separate cleanup exec.
    assert len(docker.exec_calls) == 2


@pytest.mark.parametrize(("mode", "text"), [(0o755, "0755"), (0o600, "0600"), (None, "")])
async def test_the_mode_reaches_the_copy_as_a_four_digit_octal_string(mode: int | None, text: str) -> None:
    docker = _Docker([_prep("/root/a", "ok"), _ok()])

    await files.write_container_file(_container(docker), "/root/a", b"hi", True, mode, 30.0)

    assert docker.exec_calls[1][0][6] == text


@pytest.mark.parametrize(("existed", "created"), [("0", True), ("1", False)])
async def test_the_write_result_carries_the_resolved_path_the_size_and_whether_it_created_the_file(
    existed: str, created: bool
) -> None:
    docker = _Docker([_prep("/root/real", "ok", existed), _ok()])

    result = await files.write_container_file(_container(docker), "/root/link", b"hello", True, None, 30.0)

    assert result == FileWrite(path="/root/real", size=5, created=created)
    assert docker.exec_calls[1][0][4] == "/root/real"


async def test_a_size_mismatch_after_the_copy_says_the_write_did_not_land() -> None:
    docker = _Docker([_prep("/dev/shm/a", "ok"), _ok(code=7)])

    with pytest.raises(RuntimeError, match="did not land"):
        await files.write_container_file(_container(docker), "/dev/shm/a", b"hi", True, None, 30.0)


async def test_a_copy_that_hits_the_time_limit_is_a_runtime_error() -> None:
    docker = _Docker([_prep("/root/a", "ok"), _ok(code=137)])

    with pytest.raises(RuntimeError, match="time limit"):
        await files.write_container_file(_container(docker), "/root/a", b"hi", True, None, 30.0)


async def test_a_refused_put_archive_fails_the_write_and_removes_the_staging_directory() -> None:
    refusal = APIError("500 Server Error", explanation="read-only file system")
    docker = _Docker([_prep("/root/a", "ok")], put_error=refusal)

    with pytest.raises(RuntimeError, match="Docker refused the staging archive") as raised:
        await files.write_container_file(_container(docker), "/root/a", b"hi", True, None, 30.0)

    assert "read-only file system" in str(raised.value)
    assert docker.scripts() == [files._WRITE_PREP_SCRIPT, files._CLEANUP_SCRIPT]
    [(_, archive)] = docker.puts
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
        dir_name = next(name for name in tar.getnames() if "/" not in name)
    assert docker.exec_calls[-1][0][4] == f"/{dir_name}"


async def test_a_put_archive_transport_error_propagates_after_the_cleanup() -> None:
    docker = _Docker([_prep("/root/a", "ok")], put_error=ConnectionError("docker went away"))

    with pytest.raises(ConnectionError):
        await files.write_container_file(_container(docker), "/root/a", b"hi", True, None, 30.0)

    assert docker.scripts() == [files._WRITE_PREP_SCRIPT, files._CLEANUP_SCRIPT]


async def test_a_copy_exec_that_raises_still_runs_one_cleanup_exec() -> None:
    docker = _Docker([_prep("/root/a", "ok"), ConnectionError("docker went away"), _ok()])

    with pytest.raises(ConnectionError):
        await files.write_container_file(_container(docker), "/root/a", b"hi", True, None, 30.0)

    assert docker.scripts() == [files._WRITE_PREP_SCRIPT, files._COPY_SCRIPT, files._CLEANUP_SCRIPT]
    copy_dir = docker.exec_calls[1][0][5]
    assert docker.exec_calls[2][0][4:] == [copy_dir]


async def test_two_writes_stage_into_two_different_directories() -> None:
    docker = _Docker([_prep("/root/a", "ok"), _ok(), _prep("/root/a", "ok"), _ok()])
    container = _container(docker)

    await files.write_container_file(container, "/root/a", b"one", True, None, 30.0)
    await files.write_container_file(container, "/root/a", b"two", True, None, 30.0)

    first, second = docker.exec_calls[1][0][5], docker.exec_calls[3][0][5]
    assert first != second
    assert all(_STAGING_RE.match(path.lstrip("/")) for path in (first, second))


class _SlowCopyDocker(_Docker):
    """Answers every prep with "ok" and holds each copy exec for a moment, recording how many overlap."""

    def __init__(self) -> None:
        super().__init__([])
        self.active = self.peak = 0
        self._guard = threading.Lock()

    def exec_run(self, cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        self.exec_calls.append((list(cmd), kwargs))
        if cmd[2] == files._WRITE_PREP_SCRIPT:
            return SimpleNamespace(exit_code=0, output=(_nul("/root/c.txt", "ok", "1"), None))
        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self._guard:
            self.active -= 1
        return SimpleNamespace(exit_code=0, output=(None, None))


async def test_concurrent_writes_to_one_target_both_land_and_never_copy_at_the_same_time() -> None:
    docker = _SlowCopyDocker()
    container = _container(docker)
    results: list[FileWrite] = []

    async def write(payload: bytes) -> None:
        results.append(await files.write_container_file(container, "/root/c.txt", payload, True, None, 30.0))

    async with anyio.create_task_group() as tg:
        tg.start_soon(write, b"A" * 10)
        tg.start_soon(write, b"B" * 10)

    assert [r.size for r in results] == [10, 10]
    assert docker.peak == 1
    staging = {argv[5] for argv, _ in docker.exec_calls if argv[2] == files._COPY_SCRIPT}
    assert len(staging) == 2
    assert files._TARGET_LOCKS == {}


@pytest.mark.parametrize("mode", [-1, 0o10000])
async def test_a_mode_outside_the_permission_bits_is_refused_before_any_exec(mode: int) -> None:
    docker = _Docker([])

    with pytest.raises(ValueError, match="mode"):
        await files.write_container_file(_container(docker), "/root/a", b"hi", True, mode, 30.0)
    assert docker.exec_calls == [] and docker.puts == []


def test_the_staging_tar_holds_a_private_directory_and_the_payload() -> None:
    with tarfile.open(fileobj=io.BytesIO(files.staging_tar("ab12", b"\x00\xffhi")), mode="r") as tar:
        members = {member.name: member for member in tar.getmembers()}
        directory, data = members[".exegol-mcp-ab12"], members[".exegol-mcp-ab12/data"]
        assert directory.isdir() and directory.mode == 0o700
        assert data.mode == 0o600 and data.uid == 0 and data.mtime > 0
        extracted = tar.extractfile(data)
        assert extracted is not None and extracted.read() == b"\x00\xffhi"


def test_every_file_script_sets_umask_022_and_bounds_its_command_in_container() -> None:
    for script in (files._PROBE_SCRIPT, files._READ_SCRIPT, files._WRITE_PREP_SCRIPT, files._COPY_SCRIPT):
        assert script.lstrip().startswith("umask 022;"), script
        assert 'timeout -s KILL "$' in script, script


# --- list_container_directory ----------------------------------------------------------------------------------


def _record(kind: str, size: str, mode: str, mtime: str, target: str, name: bytes | str) -> bytes:
    raw_name = name if isinstance(name, bytes) else name.encode()
    return _nul(kind, size, mode, mtime, target) + raw_name + b"\0"


_HOSTS = _record("f", "12", "644", "1700000000.5000000000", "", "hosts")
_SSL = _record("d", "4096", "755", "1700000001.0000000000", "", "ssl")


async def test_the_listing_runs_one_bounded_find_with_the_path_as_a_positional_argument() -> None:
    docker = _Docker([_ok(b"/etc\0")])

    await files.list_container_directory(_container(docker), "/etc/$(id)", 2, 10, 30.0)

    [(argv, kwargs)] = docker.exec_calls
    assert argv == ["/bin/sh", "-c", files._LIST_SCRIPT, "sh", "/etc/$(id)", "2", str(6 * 11), "30"]
    assert "/etc/$(id)" not in argv[2]
    assert kwargs == _EXEC_KWARGS


def test_the_list_script_sets_umask_022_bounds_find_and_reports_symlinks_without_following_them() -> None:
    script = files._LIST_SCRIPT
    assert script.lstrip().startswith("umask 022;")
    assert 'timeout -s KILL "$4" find -H' in script
    assert "-printf '%y\\0%s\\0%m\\0%T@\\0%l\\0%P\\0'" in script
    assert 'head -z -n "$3"' in script
    assert " -L " not in script and "-follow" not in script


async def test_the_listing_parses_nul_separated_records() -> None:
    docker = _Docker([_ok(b"/etc\0" + _HOSTS + _SSL)])

    listing = await files.list_container_directory(_container(docker), "/etc", 1, 10, 30.0)

    assert listing == DirListing(
        path="/etc",
        entries=[
            DirRecord(path="hosts", kind="file", size=12, mode=0o644, mtime=1700000000.5, target=None),
            DirRecord(path="ssl", kind="dir", size=4096, mode=0o755, mtime=1700000001.0, target=None),
        ],
        truncated=False,
    )


@pytest.mark.parametrize(
    ("letter", "kind"), [("f", "file"), ("d", "dir"), ("l", "symlink"), ("p", "other"), ("s", "other"), ("c", "other")]
)
async def test_the_kind_letters_map_to_file_dir_symlink_or_other(letter: str, kind: str) -> None:
    docker = _Docker([_ok(b"/x\0" + _record(letter, "0", "644", "1.0", "", "e"))])

    listing = await files.list_container_directory(_container(docker), "/x", 1, 10, 30.0)

    assert [entry.kind for entry in listing.entries] == [kind]


async def test_a_symlink_entry_carries_its_target() -> None:
    docker = _Docker([_ok(b"/x\0" + _record("l", "11", "777", "1.0", "/etc/hosts", "link"))])

    [entry] = (await files.list_container_directory(_container(docker), "/x", 1, 10, 30.0)).entries

    assert (entry.kind, entry.target, entry.mode) == ("symlink", "/etc/hosts", 0o777)


async def test_more_records_than_max_entries_keeps_the_first_ones_and_sets_truncated() -> None:
    extra = _record("f", "1", "600", "2.0", "", "zzz")
    docker = _Docker([_ok(b"/etc\0" + _HOSTS + _SSL + extra)])

    listing = await files.list_container_directory(_container(docker), "/etc", 1, 2, 30.0)

    assert [entry.path for entry in listing.entries] == ["hosts", "ssl"]
    assert listing.truncated is True


async def test_exactly_max_entries_records_is_not_truncated() -> None:
    docker = _Docker([_ok(b"/etc\0" + _HOSTS + _SSL)])

    listing = await files.list_container_directory(_container(docker), "/etc", 1, 2, 30.0)

    assert (len(listing.entries), listing.truncated) == (2, False)


async def test_a_trailing_partial_record_is_ignored() -> None:
    docker = _Docker([_ok(b"/etc\0" + _HOSTS + b"f\x0012\x00644\x00")])

    listing = await files.list_container_directory(_container(docker), "/etc", 1, 10, 30.0)

    assert [entry.path for entry in listing.entries] == ["hosts"]


async def test_an_empty_directory_lists_no_entries() -> None:
    docker = _Docker([_ok(b"/tmp/empty\0")])

    assert await files.list_container_directory(_container(docker), "/tmp/empty", 1, 10, 30.0) == DirListing(
        path="/tmp/empty", entries=[], truncated=False
    )


async def test_a_name_that_is_not_utf8_decodes_with_replacement_characters() -> None:
    docker = _Docker([_ok(b"/x\0" + _record("f", "1", "644", "1.0", "", b"bad\xff"))])

    [entry] = (await files.list_container_directory(_container(docker), "/x", 1, 10, 30.0)).entries

    assert entry.path == "bad\ufffd"


@pytest.mark.parametrize(
    ("code", "error", "match"),
    [(3, ValueError, "No such file or directory"), (4, ValueError, "read_file"), (137, RuntimeError, "time limit")],
)
async def test_listing_exit_codes_map_to_clear_errors(code: int, error: type[Exception], match: str) -> None:
    docker = _Docker([_ok(code=code)])

    with pytest.raises(error, match=match):
        await files.list_container_directory(_container(docker), "/x", 1, 10, 30.0)


# --- RealExegolAdapter.read_file / write_file ----------------------------------------------------------------


class _FakeDockerUtils:
    def __init__(self, containers: dict[str, SimpleNamespace]) -> None:
        self.containers = containers

    def getContainer(self, tag: str) -> SimpleNamespace:
        if tag not in self.containers:
            raise ObjectNotFound
        return self.containers[tag]


@pytest.fixture
def stub_docker(monkeypatch: pytest.MonkeyPatch) -> _Docker:
    docker = _Docker([_probe("/etc/hostname", "file", 4), _ok(b"lab\n")])
    utils = _FakeDockerUtils({"lab": _container(docker), "lab2": _container(_Docker([]), running=False)})
    monkeypatch.setattr(real, "DockerUtils", lambda: utils)
    return docker


async def test_the_real_adapter_reads_through_the_exec_transfer(stub_docker: _Docker) -> None:
    result = await RealExegolAdapter().read_file("lab", "/etc/hostname", 100, 30.0)

    assert result == FileRead(data=b"lab\n", size=4, over_limit=False, path="/etc/hostname")
    assert stub_docker.scripts() == [files._PROBE_SCRIPT, files._READ_SCRIPT]


async def test_the_real_adapter_writes_through_the_staged_transfer(stub_docker: _Docker) -> None:
    docker = _Docker([_prep("/tmp/a", "ok"), _ok()])
    real.DockerUtils().containers["lab"] = _container(docker)

    result = await RealExegolAdapter().write_file("lab", "/tmp/a", b"x", True, None, 30.0)

    assert result == FileWrite(path="/tmp/a", size=1, created=True)
    assert len(docker.puts) == 1


@pytest.mark.parametrize("operation", ["read", "write"])
async def test_the_real_adapter_file_calls_refuse_unknown_and_stopped_containers(
    stub_docker: _Docker, operation: str
) -> None:
    adapter = RealExegolAdapter()

    async def call(name: str) -> object:
        if operation == "read":
            return await adapter.read_file(name, "/etc/hostname", 100, 30.0)
        return await adapter.write_file(name, "/tmp/a", b"x", True, None, 30.0)

    with pytest.raises(ValueError, match="Container 'ghost' not found"):
        await call("ghost")
    with pytest.raises(RuntimeError, match="Container 'lab2' is not running"):
        await call("lab2")


async def test_the_real_adapter_lists_through_the_find_exec(stub_docker: _Docker) -> None:
    docker = _Docker([_ok(b"/etc\0" + _HOSTS)])
    real.DockerUtils().containers["lab"] = _container(docker)

    listing = await RealExegolAdapter().list_directory("lab", "/etc", 1, 10, 30.0)

    assert [entry.path for entry in listing.entries] == ["hosts"]
    assert docker.scripts() == [files._LIST_SCRIPT]


async def test_the_real_adapter_listing_refuses_unknown_and_stopped_containers(stub_docker: _Docker) -> None:
    with pytest.raises(ValueError, match="Container 'ghost' not found"):
        await RealExegolAdapter().list_directory("ghost", "/etc", 1, 10, 30.0)
    with pytest.raises(RuntimeError, match="Container 'lab2' is not running"):
        await RealExegolAdapter().list_directory("lab2", "/etc", 1, 10, 30.0)


# --- Linux only: the real scripts under the host /bin/sh --------------------------------------------------------


def _sh(script: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    # No inherited environment beyond PATH, like the container exec (which passes none of its own).
    return subprocess.run(
        ["/bin/sh", "-c", script, "sh", *args],
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        timeout=30,
        check=False,
    )


def _real(path: Path) -> str:
    return os.path.realpath(path)


@_LINUX_ONLY
def test_linux_the_probe_reports_the_resolved_path_the_kind_and_the_size(tmp_path: Path) -> None:
    target = tmp_path / "f.bin"
    target.write_bytes(b"abc")
    (tmp_path / "link").symlink_to(target)
    os.mkfifo(tmp_path / "pipe")

    assert _sh(files._PROBE_SCRIPT, str(tmp_path / "link"), "5").stdout == _nul(_real(target), "file", "3")
    assert _sh(files._PROBE_SCRIPT, str(tmp_path), "5").stdout.split(b"\0")[1] == b"directory"
    assert _sh(files._PROBE_SCRIPT, str(tmp_path / "pipe"), "5").stdout.split(b"\0")[1] == b"fifo"
    missing = _sh(files._PROBE_SCRIPT, str(tmp_path / "nope" / "x"), "5")
    assert missing.returncode == 0
    assert missing.stdout == _nul(_real(tmp_path) + "/nope/x", "missing", "0")


@_LINUX_ONLY
def test_linux_the_read_script_returns_the_exact_bytes_of_a_binary_file(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 3
    target = tmp_path / "blob"
    target.write_bytes(payload)

    whole = _sh(files._READ_SCRIPT, str(target), str(len(payload) + 1), "5")
    capped = _sh(files._READ_SCRIPT, str(target), "11", "5")

    assert (whole.returncode, whole.stdout) == (0, payload)
    assert capped.stdout == payload[:11]


@_LINUX_ONLY
def test_linux_the_write_prep_creates_parents_0755_and_reports_refusals(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"

    result = _sh(files._WRITE_PREP_SCRIPT, str(target), "1", "5")

    assert result.stdout == _nul(_real(tmp_path) + "/a/b/c.txt", "ok", "0")
    assert (tmp_path / "a").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "a" / "b").stat().st_mode & 0o777 == 0o755
    assert not target.exists()
    target.write_bytes(b"old")
    assert _sh(files._WRITE_PREP_SCRIPT, str(target), "0", "5").stdout.split(b"\0")[1] == b"exists"
    assert _sh(files._WRITE_PREP_SCRIPT, str(target), "1", "5").stdout.split(b"\0")[1:3] == [b"ok", b"1"]
    assert _sh(files._WRITE_PREP_SCRIPT, str(tmp_path), "1", "5").stdout.split(b"\0")[1] == b"directory"


def _stage(root: Path, payload: bytes) -> Path:
    stage = root / ".exegol-mcp-0123456789abcdef"
    stage.mkdir(mode=0o700)
    (stage / "data").write_bytes(payload)
    return stage


@_LINUX_ONLY
def test_linux_the_copy_writes_through_a_symlink_and_removes_the_staging_directory(tmp_path: Path) -> None:
    real_file = tmp_path / "real.txt"
    real_file.write_bytes(b"old content")
    link = tmp_path / "link"
    link.symlink_to(real_file)
    stage = _stage(tmp_path, b"\x00new\xff")

    result = _sh(files._COPY_SCRIPT, str(link), str(stage), "", "5", "5")

    assert result.returncode == 0, result.stderr
    assert link.is_symlink()
    assert real_file.read_bytes() == b"\x00new\xff"
    assert not stage.exists()


@_LINUX_ONLY
def test_linux_the_copy_creates_0644_files_and_applies_a_mode(tmp_path: Path) -> None:
    plain, script = tmp_path / "plain", tmp_path / "run.sh"

    _sh(files._COPY_SCRIPT, str(plain), str(_stage(tmp_path, b"x")), "", "1", "5")
    _sh(files._COPY_SCRIPT, str(script), str(_stage(tmp_path, b"#!/bin/sh\n")), "0755", "10", "5")

    assert plain.stat().st_mode & 0o777 == 0o644
    assert script.stat().st_mode & 0o777 == 0o755


@_LINUX_ONLY
def test_linux_a_failed_copy_still_removes_the_staging_directory(tmp_path: Path) -> None:
    stage = _stage(tmp_path, b"abc")

    mismatch = _sh(files._COPY_SCRIPT, str(tmp_path / "out"), str(stage), "", "999", "5")

    assert mismatch.returncode == 7
    assert not stage.exists()
    stage = _stage(tmp_path, b"abc")
    unwritable = _sh(files._COPY_SCRIPT, str(tmp_path / "no" / "dir" / "out"), str(stage), "", "3", "5")
    assert unwritable.returncode != 0
    assert not stage.exists()


def _list_records(stdout: bytes) -> tuple[bytes, list[list[bytes]]]:
    head, *fields = stdout.split(b"\0")[:-1]
    return head, [fields[i : i + 6] for i in range(0, len(fields) - len(fields) % 6, 6)]


@_LINUX_ONLY
def test_linux_the_list_script_reports_a_symlink_without_listing_what_it_points_to(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c.txt").write_bytes(b"abc")
    (tmp_path / "empty").mkdir()
    (tmp_path / "ln").symlink_to(tmp_path / "a")
    # Set explicitly so the expected mode does not depend on the process umask (G-04-2: Debian's 0002 gives 775).
    (tmp_path / "a").chmod(0o755)

    result = _sh(files._LIST_SCRIPT, str(tmp_path), "2", "600", "5")

    assert result.returncode == 0, result.stderr
    head, records = _list_records(result.stdout)
    assert head == _real(tmp_path).encode()
    by_path = {record[5].decode(): record for record in records}
    assert set(by_path) == {"a", "a/b", "empty", "ln"}
    assert by_path["ln"][0] == b"l" and by_path["ln"][4] == str(tmp_path / "a").encode()
    assert by_path["a"][0] == b"d" and by_path["a"][2] == b"755"


@_LINUX_ONLY
def test_linux_the_list_script_caps_records_and_exits_3_or_4_for_a_missing_path_or_a_file(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"f{index}").write_bytes(b"x")

    capped = _sh(files._LIST_SCRIPT, str(tmp_path), "1", str(6 * 3), "5")

    assert len(_list_records(capped.stdout)[1]) == 3
    assert _sh(files._LIST_SCRIPT, str(tmp_path / "missing"), "1", "60", "5").returncode == 3
    assert _sh(files._LIST_SCRIPT, str(tmp_path / "f0"), "1", "60", "5").returncode == 4
