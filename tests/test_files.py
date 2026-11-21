"""read_file, write_file and list_directory over the in-memory MCP Client (FILE-01..04, LIFE-07), Docker-free.

The fake adapter keeps per-container files as raw bytes (plan 04-09). These tests drive the tools end to end: the
path rule (absolute, no NUL, at most 4096 characters) refuses bad paths before any adapter call, the one raw-byte
size limit (files.max_bytes) is exact at its boundary and never truncates, valid UTF-8 comes back as text and
anything else as base64, and auto_start starts a stopped container first (D-15, D-16, D-17, D-20). list_directory
returns sorted, rich entries paged through the shared opaque cursor, bounded by the depth and entry caps (D-19).
"""

import base64
import logging
from typing import Any

import pytest
from mcp import Client

from exegol_mcp.adapter.port import ContainerState, DirListing, DirRecord, FileRead, FileWrite
from exegol_mcp.app import build_server
from exegol_mcp.models.config import OperatorConfig
from exegol_mcp.services import files as files_service
from tests.fakes import FakeExegolAdapter


class _RecordingFake(FakeExegolAdapter):
    """Records every port call the file tools can make, so a test can prove a refusal came first."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[str] = []
        self.read_limits: list[int] = []
        self.write_calls: list[tuple[str, bytes, bool, int | None]] = []
        self.list_calls: list[tuple[str, int, int]] = []

    async def get_container_state(self, name: str) -> ContainerState | None:
        self.calls.append("get_container_state")
        return await super().get_container_state(name)

    async def read_file(self, name: str, path: str, max_bytes: int, timeout_s: float) -> FileRead:
        self.calls.append("read_file")
        self.read_limits.append(max_bytes)
        return await super().read_file(name, path, max_bytes, timeout_s)

    async def write_file(
        self, name: str, path: str, data: bytes, overwrite: bool, mode: int | None, timeout_s: float
    ) -> FileWrite:
        self.calls.append("write_file")
        self.write_calls.append((path, data, overwrite, mode))
        return await super().write_file(name, path, data, overwrite, mode, timeout_s)

    async def list_directory(
        self, name: str, path: str, depth: int, max_entries: int, timeout_s: float
    ) -> DirListing:
        self.calls.append("list_directory")
        self.list_calls.append((path, depth, max_entries))
        return await super().list_directory(name, path, depth, max_entries, timeout_s)


class _PseudoFileFake(_RecordingFake):
    """A pseudo file (stat size 0) longer than the limit: the port reports over_limit with an unknown size."""

    async def read_file(self, name: str, path: str, max_bytes: int, timeout_s: float) -> FileRead:
        self.calls.append("read_file")
        return FileRead(data=b"x" * (max_bytes + 1), size=None, over_limit=True, path=path)


def _fake(files: dict[str, bytes] | None = None, max_bytes: int | None = None, **kwargs: Any) -> _RecordingFake:
    config = OperatorConfig.model_validate({"files": {"max_bytes": max_bytes}}) if max_bytes is not None else None
    seeded = {"exegol-lab": {"/etc/hostname": b"exegol-lab\n", **(files or {})}}
    return _RecordingFake(files=seeded, config=config, **kwargs)


async def _read(fake: FakeExegolAdapter, **arguments: Any) -> Any:
    async with Client(build_server(None, fake)) as client:
        return await client.call_tool("read_file", {"container_name": "exegol-lab", **arguments})


# --- read_file: text, binary, forced base64 (FILE-01, D-16) ---------------------------------------------------------


async def test_read_file_returns_a_utf8_file_as_text() -> None:
    result = await _read(_fake(), path="/etc/hostname")
    assert not result.is_error, result.content
    assert result.structured_content == {
        "path": "/etc/hostname",
        "encoding": "utf-8",
        "content": "exegol-lab\n",
        "size": 11,
    }


async def test_read_file_returns_non_ascii_utf8_as_text_with_its_raw_byte_size() -> None:
    result = await _read(_fake({"/tmp/h.txt": "héllo\n".encode()}), path="/tmp/h.txt")
    assert not result.is_error, result.content
    content = result.structured_content
    assert content["encoding"] == "utf-8"
    assert content["content"] == "héllo\n"
    assert content["size"] == 7


async def test_read_file_returns_binary_as_base64_that_decodes_to_the_exact_bytes() -> None:
    raw = b"\x00\xff\xfe\n"
    result = await _read(_fake({"/tmp/b.bin": raw}), path="/tmp/b.bin")
    assert not result.is_error, result.content
    content = result.structured_content
    assert content["encoding"] == "base64"
    assert base64.b64decode(content["content"], validate=True) == raw
    assert content["size"] == 4


async def test_read_file_with_encoding_base64_forces_base64_on_a_text_file() -> None:
    result = await _read(_fake(), path="/etc/hostname", encoding="base64")
    assert not result.is_error, result.content
    content = result.structured_content
    assert content["encoding"] == "base64"
    assert content["content"] == base64.b64encode(b"exegol-lab\n").decode("ascii")
    assert content["size"] == 11


async def test_read_file_returns_an_encoded_surrogate_as_base64_never_replacement_characters() -> None:
    raw = b"\xed\xa0\x80"
    result = await _read(_fake({"/tmp/s.bin": raw}), path="/tmp/s.bin")
    assert not result.is_error, result.content
    assert result.structured_content["encoding"] == "base64"
    assert base64.b64decode(result.structured_content["content"]) == raw


async def test_read_file_returns_a_truncated_multibyte_character_as_base64() -> None:
    raw = b"\xe2\x82"
    result = await _read(_fake({"/tmp/t.bin": raw}), path="/tmp/t.bin")
    assert not result.is_error, result.content
    assert result.structured_content["encoding"] == "base64"
    assert base64.b64decode(result.structured_content["content"]) == raw


async def test_read_file_of_an_empty_file_is_empty_utf8_text() -> None:
    result = await _read(_fake({"/tmp/empty": b""}), path="/tmp/empty")
    assert not result.is_error, result.content
    assert result.structured_content == {"path": "/tmp/empty", "encoding": "utf-8", "content": "", "size": 0}


async def test_read_file_returns_credentials_in_a_file_unredacted() -> None:
    raw = b"admin:Passw0rd!\n"
    result = await _read(_fake({"/root/creds.txt": raw}), path="/root/creds.txt")
    assert not result.is_error, result.content
    assert result.structured_content["content"] == "admin:Passw0rd!\n"


async def test_read_file_of_a_missing_file_or_a_directory_is_a_clear_error() -> None:
    missing = await _read(_fake(), path="/etc/nope")
    assert missing.is_error
    assert "/etc/nope" in str(missing.content)
    directory = await _read(_fake(), path="/etc")
    assert directory.is_error
    assert "directory" in str(directory.content)


# --- read_file: the size limit (FILE-04, D-17) ----------------------------------------------------------------------


async def test_read_file_passes_files_max_bytes_to_the_port() -> None:
    fake = _fake(max_bytes=8, files={"/tmp/eight": b"12345678"})
    result = await _read(fake, path="/tmp/eight")
    assert not result.is_error, result.content
    assert fake.read_limits == [8]


async def test_read_file_of_exactly_max_bytes_is_accepted() -> None:
    result = await _read(_fake(max_bytes=8, files={"/tmp/eight": b"12345678"}), path="/tmp/eight")
    assert not result.is_error, result.content
    assert result.structured_content["content"] == "12345678"
    assert result.structured_content["size"] == 8


async def test_read_file_of_max_bytes_plus_one_is_refused_with_the_exact_size_and_limit() -> None:
    result = await _read(_fake(max_bytes=8, files={"/tmp/nine": b"123456789"}), path="/tmp/nine")
    assert result.is_error
    text = str(result.content)
    assert "9 bytes" in text
    assert "8-byte limit" in text
    assert "files.max_bytes" in text
    assert "exec_command" in text
    assert "workspace" in text
    # Never cut to fit: no part of the file reaches the agent.
    assert "12345678" not in text


async def test_read_file_states_the_exact_size_without_rounding() -> None:
    limit = 10_485_760
    fake = _fake(max_bytes=limit, files={"/tmp/big": b"a" * (limit + 1)})
    result = await _read(fake, path="/tmp/big")
    assert result.is_error
    text = str(result.content)
    assert "10485761 bytes" in text
    assert "10485760-byte limit" in text


async def test_read_file_of_a_pseudo_file_of_unknown_size_says_larger_than_the_limit() -> None:
    config = OperatorConfig.model_validate({"files": {"max_bytes": 8}})
    result = await _read(_PseudoFileFake(config=config), path="/proc/cpuinfo")
    assert result.is_error
    text = str(result.content)
    assert "larger than" in text
    assert "8-byte limit" in text
    assert "files.max_bytes" in text


# --- read_file: the path rule (D-20) --------------------------------------------------------------------------------


async def test_read_file_refuses_a_relative_path_before_any_adapter_call() -> None:
    fake = _fake()
    result = await _read(fake, path="etc/hostname")
    assert result.is_error
    assert "absolute" in str(result.content)
    assert fake.calls == []


async def test_read_file_refuses_a_path_with_a_nul_before_any_adapter_call() -> None:
    fake = _fake()
    result = await _read(fake, path="/etc/host\x00name")
    assert result.is_error
    assert "NUL" in str(result.content)
    assert fake.calls == []


async def test_read_file_refuses_a_path_over_4096_characters_before_any_adapter_call() -> None:
    fake = _fake()
    result = await _read(fake, path="/" + "a" * 4096)
    assert result.is_error
    assert "4096" in str(result.content)
    assert fake.calls == []


async def test_read_file_accepts_a_path_of_exactly_4096_characters() -> None:
    path = "/" + "a" * 4095
    result = await _read(_fake({path: b"x"}), path=path)
    assert not result.is_error, result.content


# --- read_file: auto_start (LIFE-07, D-15) --------------------------------------------------------------------------


async def test_read_file_on_a_stopped_container_without_auto_start_names_auto_start() -> None:
    fake = _fake()
    fake.running["exegol-lab"] = False
    result = await _read(fake, path="/etc/hostname")
    assert result.is_error
    text = str(result.content)
    assert "start_container" in text
    assert "auto_start" in text
    assert fake.state_changes == []
    assert "read_file" not in fake.calls


async def test_read_file_with_auto_start_starts_a_stopped_container_then_reads() -> None:
    fake = _fake()
    fake.running["exegol-lab"] = False
    result = await _read(fake, path="/etc/hostname", auto_start=True)
    assert not result.is_error, result.content
    assert result.structured_content["content"] == "exegol-lab\n"
    assert fake.state_changes == [("start", "exegol-lab")]


async def test_read_file_on_an_unknown_container_is_not_found() -> None:
    fake = _fake()
    async with Client(build_server(None, fake)) as client:
        result = await client.call_tool("read_file", {"container_name": "nope", "path": "/etc/hostname"})
    assert result.is_error
    assert "not found" in str(result.content)
    assert "read_file" not in fake.calls


# --- logging: contents never reach the server logs (Phase 3 D-08) ---------------------------------------------------


async def test_read_file_never_logs_the_content(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    secret = "s3cr3t-read-marker"
    result = await _read(_fake({"/root/secret": secret.encode()}), path="/root/secret")
    assert not result.is_error, result.content
    assert secret not in caplog.text


# --- write_file: round trips (FILE-02, D-18) ------------------------------------------------------------------------


async def _write(fake: FakeExegolAdapter, **arguments: Any) -> Any:
    async with Client(build_server(None, fake)) as client:
        return await client.call_tool("write_file", {"container_name": "exegol-lab", **arguments})


async def test_write_file_then_read_file_round_trips_utf8_text() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/a.txt", content="héllo\n")
    assert not written.is_error, written.content
    assert written.structured_content == {"path": "/tmp/a.txt", "size": 7, "created": True}
    result = await _read(fake, path="/tmp/a.txt")
    assert result.structured_content == {"path": "/tmp/a.txt", "encoding": "utf-8", "content": "héllo\n", "size": 7}


async def test_write_file_with_base64_round_trips_binary_bytes() -> None:
    raw = b"\x00\xff\xfe\n"
    fake = _fake()
    written = await _write(fake, path="/tmp/b.bin", content=base64.b64encode(raw).decode("ascii"), encoding="base64")
    assert not written.is_error, written.content
    assert written.structured_content["size"] == 4
    assert fake.files["exegol-lab"]["/tmp/b.bin"] == raw
    result = await _read(fake, path="/tmp/b.bin")
    assert result.structured_content["encoding"] == "base64"
    assert base64.b64decode(result.structured_content["content"]) == raw


async def test_write_file_overwrites_an_existing_file_by_default() -> None:
    fake = _fake()
    written = await _write(fake, path="/etc/hostname", content="other\n")
    assert not written.is_error, written.content
    assert written.structured_content["created"] is False
    assert fake.files["exegol-lab"]["/etc/hostname"] == b"other\n"
    assert fake.write_calls[-1][2] is True


async def test_write_file_creates_missing_parent_directories() -> None:
    fake = _fake()
    written = await _write(fake, path="/opt/new/dir/tool.sh", content="#!/bin/sh\n")
    assert not written.is_error, written.content
    assert fake.files["exegol-lab"]["/opt/new/dir/tool.sh"] == b"#!/bin/sh\n"


async def test_write_file_of_an_empty_content_writes_an_empty_file() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/empty", content="")
    assert not written.is_error, written.content
    assert written.structured_content["size"] == 0
    assert fake.files["exegol-lab"]["/tmp/empty"] == b""


# --- write_file: the size limit and the length prechecks (FILE-04, D-17) --------------------------------------------


async def test_write_file_of_exactly_max_bytes_is_accepted() -> None:
    fake = _fake(max_bytes=8)
    written = await _write(fake, path="/tmp/eight", content="12345678")
    assert not written.is_error, written.content
    assert written.structured_content["size"] == 8


async def test_write_file_of_max_bytes_plus_one_raw_bytes_is_refused_with_the_exact_size_and_limit() -> None:
    # 8 characters, 9 bytes once encoded: past the character precheck, caught on the raw-byte size.
    fake = _fake(max_bytes=8)
    written = await _write(fake, path="/tmp/nine", content="héllo\nab")
    assert written.is_error
    text = str(written.content)
    assert "9 bytes" in text
    assert "8-byte limit" in text
    assert "files.max_bytes" in text
    assert fake.calls == []


async def test_write_file_of_nine_decoded_base64_bytes_is_refused_with_the_exact_size() -> None:
    fake = _fake(max_bytes=8)
    content = base64.b64encode(b"123456789").decode("ascii")
    assert len(content) == 12  # within 4 * ceil(8 / 3): only the decoded size is over
    written = await _write(fake, path="/tmp/nine", content=content, encoding="base64")
    assert written.is_error
    text = str(written.content)
    assert "9 bytes" in text
    assert "8-byte limit" in text
    assert fake.calls == []


async def test_write_file_refuses_a_too_long_base64_string_before_decoding_it() -> None:
    fake = _fake(max_bytes=8)
    # 4 * ceil(8 / 3) + 4 characters that are not even base64: the length check fires before any decode.
    written = await _write(fake, path="/tmp/x", content="!" * 16, encoding="base64")
    assert written.is_error
    text = str(written.content)
    assert "8-byte limit" in text
    assert "files.max_bytes" in text
    assert "not valid base64" not in text
    assert fake.calls == []


async def test_write_file_refuses_a_too_long_utf8_string_before_encoding_it() -> None:
    fake = _fake(max_bytes=8)
    written = await _write(fake, path="/tmp/x", content="123456789")
    assert written.is_error
    text = str(written.content)
    assert "9 characters" in text
    assert "8-byte limit" in text
    assert "files.max_bytes" in text
    assert fake.calls == []


async def test_write_file_refuses_invalid_base64() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/x", content="not base64!", encoding="base64")
    assert written.is_error
    assert "base64" in str(written.content)
    assert fake.calls == []


# --- write_file: mode, overwrite, directories, paths (D-18, D-20) ---------------------------------------------------


@pytest.mark.parametrize("mode", ["0755", "755"])
async def test_write_file_passes_an_octal_mode_string_to_the_port(mode: str) -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/run.sh", content="#!/bin/sh\nid\n", mode=mode)
    assert not written.is_error, written.content
    assert fake.write_calls[-1][3] == 0o755
    assert fake.file_modes[("exegol-lab", "/tmp/run.sh")] == 0o755


async def test_write_file_without_a_mode_passes_none() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/plain", content="x")
    assert not written.is_error, written.content
    assert fake.write_calls[-1][3] is None


@pytest.mark.parametrize("mode", ["999", "rwx", "07777x", "", "0o755", "75"])
async def test_write_file_refuses_a_mode_that_is_not_an_octal_string(mode: str) -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/run.sh", content="x", mode=mode)
    assert written.is_error
    assert "mode" in str(written.content)
    assert fake.calls == []


async def test_write_file_with_overwrite_false_refuses_an_existing_file() -> None:
    fake = _fake()
    written = await _write(fake, path="/etc/hostname", content="other\n", overwrite=False)
    assert written.is_error
    assert "overwrite" in str(written.content)
    assert fake.files["exegol-lab"]["/etc/hostname"] == b"exegol-lab\n"


async def test_write_file_with_overwrite_false_creates_a_new_file() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/new", content="x", overwrite=False)
    assert not written.is_error, written.content
    assert written.structured_content["created"] is True


async def test_write_file_to_a_directory_is_a_clear_error() -> None:
    written = await _write(_fake(), path="/etc", content="x")
    assert written.is_error
    assert "directory" in str(written.content)


async def test_write_file_refuses_a_relative_path_before_any_adapter_call() -> None:
    fake = _fake()
    written = await _write(fake, path="tmp/x", content="x")
    assert written.is_error
    assert "absolute" in str(written.content)
    assert fake.calls == []


async def test_write_file_refuses_a_path_with_a_nul_before_any_adapter_call() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/a\x00b", content="x")
    assert written.is_error
    assert fake.calls == []


# --- write_file: auto_start and the container (LIFE-07, D-15) -------------------------------------------------------


async def test_write_file_on_a_stopped_container_without_auto_start_names_auto_start() -> None:
    fake = _fake()
    fake.running["exegol-lab"] = False
    written = await _write(fake, path="/tmp/a.txt", content="x")
    assert written.is_error
    assert "auto_start" in str(written.content)
    assert fake.state_changes == []
    assert "write_file" not in fake.calls


async def test_write_file_with_auto_start_starts_a_stopped_container_then_writes() -> None:
    fake = _fake()
    fake.running["exegol-lab"] = False
    written = await _write(fake, path="/tmp/a.txt", content="x", auto_start=True)
    assert not written.is_error, written.content
    assert fake.state_changes == [("start", "exegol-lab")]
    assert fake.files["exegol-lab"]["/tmp/a.txt"] == b"x"


async def test_write_file_on_an_unknown_container_is_not_found() -> None:
    fake = _fake()
    async with Client(build_server(None, fake)) as client:
        written = await client.call_tool("write_file", {"container_name": "nope", "path": "/tmp/a", "content": "x"})
    assert written.is_error
    assert "not found" in str(written.content)
    assert "write_file" not in fake.calls


async def test_write_file_never_logs_the_content(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    secret = "s3cr3t-write-marker"
    written = await _write(_fake(), path="/tmp/secret", content=secret)
    assert not written.is_error, written.content
    assert secret not in caplog.text


# --- list_directory: sorted, rich, paged entries (FILE-03, D-19) ----------------------------------------------------

# The fake stamps every entry with this mtime (1700000000.0 seconds since the epoch).
_FAKE_MTIME = "2023-11-14T22:13:20+00:00"


def _list_fake(
    files: dict[str, bytes] | None = None,
    *,
    dirs: set[str] | None = None,
    settings: dict[str, Any] | None = None,
    fake_class: type[_RecordingFake] = _RecordingFake,
    **kwargs: Any,
) -> _RecordingFake:
    config = OperatorConfig.model_validate(settings) if settings is not None else None
    seeded = {"exegol-lab": {"/etc/hostname": b"exegol-lab\n", **(files or {})}}
    explicit = {("exegol-lab", path) for path in (dirs or set())}
    return fake_class(files=seeded, dirs=explicit, config=config, **kwargs)


async def _list(fake: FakeExegolAdapter, **arguments: Any) -> Any:
    async with Client(build_server(None, fake)) as client:
        return await client.call_tool("list_directory", {"container_name": "exegol-lab", **arguments})


async def test_list_directory_lists_one_level_sorted_with_rich_entries() -> None:
    fake = _list_fake({"/etc/passwd": b"root:x:0:0\n", "/etc/ssl/certs/ca.pem": b"pem"})
    result = await _list(fake, path="/etc")
    assert not result.is_error, result.content
    assert result.structured_content == {
        "path": "/etc",
        "items": [
            {
                "name": "hostname",
                "path": "hostname",
                "type": "file",
                "size": 11,
                "mode": "0644",
                "mtime": _FAKE_MTIME,
                "symlink_target": None,
            },
            {
                "name": "passwd",
                "path": "passwd",
                "type": "file",
                "size": 11,
                "mode": "0644",
                "mtime": _FAKE_MTIME,
                "symlink_target": None,
            },
            {
                "name": "ssl",
                "path": "ssl",
                "type": "dir",
                "size": 4096,
                "mode": "0755",
                "mtime": _FAKE_MTIME,
                "symlink_target": None,
            },
        ],
        "next_cursor": None,
        "truncated": False,
    }


async def test_list_directory_reports_the_resolved_path_from_the_port() -> None:
    result = await _list(_list_fake(), path="/etc/")
    assert not result.is_error, result.content
    assert result.structured_content["path"] == "/etc"


async def test_list_directory_walks_every_entry_exactly_once_through_next_cursor() -> None:
    names = ["e", "c", "a", "d", "b"]
    fake = _list_fake(
        {f"/data/{name}": name.encode() for name in names}, settings={"execution": {"listing_page_size": 2}}
    )
    seen: list[str] = []
    pages = 0
    cursor: str | None = None
    async with Client(build_server(None, fake)) as client:
        while True:
            arguments: dict[str, Any] = {"container_name": "exegol-lab", "path": "/data"}
            if cursor is not None:
                arguments["cursor"] = cursor
            result = await client.call_tool("list_directory", arguments)
            assert not result.is_error, result.content
            page = result.structured_content
            assert len(page["items"]) <= 2
            seen.extend(item["path"] for item in page["items"])
            pages += 1
            cursor = page["next_cursor"]
            if cursor is None:
                break
    assert seen == ["a", "b", "c", "d", "e"]
    assert pages == 3


async def test_list_directory_refuses_a_forged_cursor() -> None:
    result = await _list(_list_fake(), path="/etc", cursor="not-a-cursor!")
    assert result.is_error
    assert "cursor" in str(result.content)


async def test_list_directory_refuses_an_out_of_range_cursor() -> None:
    forged = base64.urlsafe_b64encode(b"999").decode("ascii")
    result = await _list(_list_fake(), path="/etc", cursor=forged)
    assert result.is_error
    assert "out of range" in str(result.content)


async def test_list_directory_and_the_catalog_share_one_cursor_module() -> None:
    from exegol_mcp.services import catalog, paging

    assert catalog._page is paging.page
    assert catalog._decode_cursor is paging.decode_cursor
    assert catalog._encode_cursor is paging.encode_cursor
    assert paging.decode_cursor(paging.encode_cursor(7)) == 7


# --- list_directory: depth bounds (D-19, FILE-04) -------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0, -1, 6])
async def test_list_directory_refuses_a_depth_outside_1_to_max_list_depth_before_any_adapter_call(depth: int) -> None:
    fake = _list_fake()
    result = await _list(fake, path="/etc", depth=depth)
    assert result.is_error
    text = str(result.content)
    assert "files.max_list_depth" in text
    assert "1" in text and "5" in text
    assert fake.calls == []


async def test_list_directory_refuses_one_past_the_operators_max_list_depth() -> None:
    fake = _list_fake(settings={"files": {"max_list_depth": 2}})
    result = await _list(fake, path="/etc", depth=3)
    assert result.is_error
    assert "files.max_list_depth" in str(result.content)
    assert fake.calls == []


async def test_list_directory_accepts_a_depth_equal_to_max_list_depth() -> None:
    fake = _list_fake({"/etc/ssl/certs/ca.pem": b"pem"}, settings={"files": {"max_list_depth": 2}})
    result = await _list(fake, path="/etc", depth=2)
    assert not result.is_error, result.content
    assert [item["path"] for item in result.structured_content["items"]] == ["hostname", "ssl", "ssl/certs"]
    assert fake.list_calls == [("/etc", 2, 10_000)]


async def test_list_directory_service_refuses_a_bool_depth() -> None:
    fake = _list_fake()
    with pytest.raises(ValueError, match="files.max_list_depth"):
        await files_service.list_directory(
            fake,
            "exegol-lab",
            "/etc",
            depth=True,
            cursor=None,
            page_size=200,
            max_depth=5,
            max_entries=10_000,
            op_timeout_s=5.0,
            auto_start=False,
        )
    assert fake.calls == []


# --- list_directory: nesting, empty directories, truncation, symlinks (FILE-03 edges) -------------------------------


async def test_list_directory_keeps_same_named_nested_entries_distinct_through_their_paths() -> None:
    fake = _list_fake({"/srv/x": b"top", "/srv/ssl/x": b"nested"})
    result = await _list(fake, path="/srv", depth=2)
    assert not result.is_error, result.content
    items = result.structured_content["items"]
    assert [(item["name"], item["path"], item["type"]) for item in items] == [
        ("ssl", "ssl", "dir"),
        ("x", "ssl/x", "file"),
        ("x", "x", "file"),
    ]


async def test_list_directory_of_an_empty_directory_is_an_empty_page() -> None:
    result = await _list(_list_fake(dirs={"/empty"}), path="/empty")
    assert not result.is_error, result.content
    assert result.structured_content == {"path": "/empty", "items": [], "next_cursor": None, "truncated": False}


async def test_list_directory_passes_max_list_entries_to_the_port_and_reports_truncation() -> None:
    fake = _list_fake(
        {"/data/a": b"a", "/data/b": b"b", "/data/c": b"c"}, settings={"files": {"max_list_entries": 2}}
    )
    result = await _list(fake, path="/data")
    assert not result.is_error, result.content
    page = result.structured_content
    assert page["truncated"] is True
    assert [item["path"] for item in page["items"]] == ["a", "b"]
    assert fake.list_calls == [("/data", 1, 2)]


class _CannedListingFake(_RecordingFake):
    """The port returns a fixed listing, in find order, holding a symlink to a directory and a FIFO."""

    async def list_directory(
        self, name: str, path: str, depth: int, max_entries: int, timeout_s: float
    ) -> DirListing:
        self.calls.append("list_directory")
        self.list_calls.append((path, depth, max_entries))
        return DirListing(
            path="/opt/real",
            entries=[
                DirRecord("tools", "symlink", 9, 0o777, 0.0, "/usr/share/tools"),
                DirRecord("fifo", "other", 0, 0o600, 0.0, None),
                DirRecord("bin", "dir", 4096, 0o755, 0.0, None),
            ],
            truncated=True,
        )


async def test_list_directory_reports_a_symlink_with_its_target_without_following_it() -> None:
    fake = _list_fake(fake_class=_CannedListingFake)
    result = await _list(fake, path="/opt/link")
    assert not result.is_error, result.content
    page = result.structured_content
    assert page["path"] == "/opt/real"
    assert page["truncated"] is True
    assert page["items"] == [
        {
            "name": "bin",
            "path": "bin",
            "type": "dir",
            "size": 4096,
            "mode": "0755",
            "mtime": "1970-01-01T00:00:00+00:00",
            "symlink_target": None,
        },
        {
            "name": "fifo",
            "path": "fifo",
            "type": "other",
            "size": 0,
            "mode": "0600",
            "mtime": "1970-01-01T00:00:00+00:00",
            "symlink_target": None,
        },
        {
            "name": "tools",
            "path": "tools",
            "type": "symlink",
            "size": 9,
            "mode": "0777",
            "mtime": "1970-01-01T00:00:00+00:00",
            "symlink_target": "/usr/share/tools",
        },
    ]


# --- list_directory: missing and file paths, path rule (FILE-04) ----------------------------------------------------


async def test_list_directory_of_a_missing_path_names_the_path() -> None:
    result = await _list(_list_fake(), path="/nope/here")
    assert result.is_error
    assert "/nope/here" in str(result.content)


async def test_list_directory_of_a_file_points_to_read_file() -> None:
    result = await _list(_list_fake(), path="/etc/hostname")
    assert result.is_error
    assert "read_file" in str(result.content)


async def test_list_directory_refuses_a_relative_path_before_any_adapter_call() -> None:
    fake = _list_fake()
    result = await _list(fake, path="etc")
    assert result.is_error
    assert "absolute" in str(result.content)
    assert fake.calls == []


# --- list_directory: auto_start (LIFE-07, D-15) ---------------------------------------------------------------------


async def test_list_directory_on_a_stopped_container_without_auto_start_names_auto_start() -> None:
    fake = _list_fake()
    fake.running["exegol-lab"] = False
    result = await _list(fake, path="/etc")
    assert result.is_error
    assert "auto_start" in str(result.content)
    assert fake.state_changes == []
    assert "list_directory" not in fake.calls


async def test_list_directory_with_auto_start_starts_a_stopped_container_then_lists() -> None:
    fake = _list_fake()
    fake.running["exegol-lab"] = False
    result = await _list(fake, path="/etc", auto_start=True)
    assert not result.is_error, result.content
    assert [item["name"] for item in result.structured_content["items"]] == ["hostname"]
    assert fake.state_changes == [("start", "exegol-lab")]


# --- hostile metadata and odd payloads never break the files tools (WR-05, IN-02, IN-05) ----------------------------

_MTIME_MAX = "9999-12-31T23:59:59+00:00"
_MTIME_MIN = "0001-01-01T00:00:00+00:00"


class _PlantedMtimeFake(_RecordingFake):
    """The port returns entries whose mtimes a container process planted: far future, far past, inf and NaN."""

    async def list_directory(
        self, name: str, path: str, depth: int, max_entries: int, timeout_s: float
    ) -> DirListing:
        self.calls.append("list_directory")
        return DirListing(
            path=path,
            entries=[
                DirRecord("a-future", "file", 1, 0o644, 1e12, None),
                DirRecord("b-past", "file", 1, 0o644, -1e13, None),
                DirRecord("c-inf", "file", 1, 0o644, float("inf"), None),
                DirRecord("d-nan", "file", 1, 0o644, float("nan"), None),
                DirRecord("e-normal", "file", 1, 0o644, 1700000000.0, None),
            ],
            truncated=False,
        )


async def test_one_out_of_range_mtime_never_fails_the_listing() -> None:
    result = await _list(_list_fake(fake_class=_PlantedMtimeFake), path="/tmp")
    assert not result.is_error, result.content
    mtimes = {item["name"]: item["mtime"] for item in result.structured_content["items"]}
    assert mtimes == {
        "a-future": _MTIME_MAX,
        "b-past": _MTIME_MIN,
        "c-inf": _MTIME_MAX,
        "d-nan": _MTIME_MIN,
        "e-normal": _FAKE_MTIME,
    }


def test_the_mtime_limits_map_to_themselves_and_one_second_past_them_clamps() -> None:
    mtime_text = files_service._mtime_text
    assert mtime_text(253402300799) == _MTIME_MAX
    assert mtime_text(253402300800) == _MTIME_MAX
    assert mtime_text(-62135596800) == _MTIME_MIN
    assert mtime_text(-62135596801) == _MTIME_MIN
    assert mtime_text(0.0) == "1970-01-01T00:00:00+00:00"


async def test_non_ascii_base64_gets_the_agent_facing_base64_error() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/x", content="é===", encoding="base64")
    assert written.is_error
    text = str(written.content)
    assert "not valid base64" in text
    assert "ASCII characters" not in text
    assert "write_file" not in fake.calls


async def test_a_lone_surrogate_in_text_content_suggests_base64() -> None:
    fake = _fake()
    written = await _write(fake, path="/tmp/x", content="a\ud800b", encoding="utf-8")
    assert written.is_error
    text = str(written.content)
    assert "base64" in text
    assert "surrogates not allowed" not in text
    assert "write_file" not in fake.calls


@pytest.mark.parametrize(
    "path",
    [
        "/dev/stdout",
        "/dev/stderr",
        "/dev/stdin",
        "/dev/fd/1",
        "/proc/self/fd/1",
        "/proc/thread-self/x",
        "//proc/self/environ",
        "/proc/./self/x",
    ],
)
async def test_a_per_process_write_target_is_refused_before_any_write(path: str) -> None:
    fake = _fake()
    written = await _write(fake, path=path, content="x")
    assert written.is_error
    text = str(written.content)
    assert path in text
    assert "exec_command" in text
    assert fake.calls == []


@pytest.mark.parametrize("path", ["/proc/1/fd/1", "/dev/null"])
async def test_a_real_pid_or_device_write_target_reaches_the_adapter(path: str) -> None:
    fake = _fake()
    await _write(fake, path=path, content="x")
    assert "write_file" in fake.calls
