"""Contract tests for ExegolPort, the in-memory fake, the ImageInfo field names and per-request state."""

import inspect
import secrets

import anyio
import anyio.lowlevel
import pytest

from exegol_mcp.adapter.port import (
    MANAGED_LABEL,
    MANAGED_LABEL_VALUE,
    RUN_ID_LABEL,
    ContainerState,
    CreateSpec,
    ExegolPort,
    JobHandle,
)
from exegol_mcp.bootstrap.lifecycle import ServerContext
from exegol_mcp.context.request import RequestState, bind_request, request_ctx
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.models.image import ImageInfo
from tests.fakes import FakeExegolAdapter

# The five job primitives Task 02-02 adds to the port (the phase tracer).
JOB_METHODS = {"start_job", "pump_job", "read_job", "kill_job", "sweep_orphans"}

# The container file transfer methods plan 04-09 adds (FILE-01..03, port side).
FILE_METHODS = {"read_file", "write_file", "list_directory"}

PORT_METHODS = {
    "list_containers",
    "get_container_state",
    "start_container",
    "stop_container",
    "list_images",
    "download_image",
    "exec_command",
    "create_container",
    "remove_container",
} | JOB_METHODS | FILE_METHODS


def _port_method_names() -> set[str]:
    return {name for name, value in vars(ExegolPort).items() if not name.startswith("_") and callable(value)}


def test_port_declares_the_full_phase_surface() -> None:
    assert _port_method_names() == PORT_METHODS
    assert all(inspect.iscoroutinefunction(getattr(ExegolPort, name)) for name in PORT_METHODS)


def test_fake_satisfies_the_port_with_coroutine_methods() -> None:
    fake = FakeExegolAdapter()
    assert isinstance(fake, ExegolPort)
    for name in _port_method_names():
        assert inspect.iscoroutinefunction(getattr(fake, name)), name


def test_object_missing_a_port_method_is_not_a_port() -> None:
    class Partial:
        async def list_containers(self) -> list[object]:
            return []

    assert not isinstance(Partial(), ExegolPort)


async def test_container_state_follows_start_and_stop() -> None:
    fake = FakeExegolAdapter()
    assert await fake.get_container_state("missing") is None
    assert await fake.get_container_state("exegol-lab") == ContainerState(running=True)
    assert await fake.stop_container("exegol-lab") is False
    assert await fake.get_container_state("exegol-lab") == ContainerState(running=False)
    assert await fake.start_container("exegol-lab") is True
    assert await fake.get_container_state("exegol-lab") == ContainerState(running=True)


async def test_installed_images_are_a_subset_of_all_images() -> None:
    fake = FakeExegolAdapter()
    installed = await fake.list_images("installed")
    every = await fake.list_images("all")
    assert installed, "the default fake has at least one installed image"
    assert all(image.is_installed for image in installed)
    assert {i.name for i in installed} < {i.name for i in every}


async def test_empty_fake_lists_no_images_for_either_scope() -> None:
    fake = FakeExegolAdapter(images=[])
    assert await fake.list_images("installed") == []
    assert await fake.list_images("all") == []


async def test_download_image_reports_progress_then_installs() -> None:
    fake = FakeExegolAdapter()
    reports: list[tuple[float, float | None, str | None]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        reports.append((progress, total, message))

    assert "full" not in {i.name for i in await fake.list_images("installed")}
    assert await fake.download_image("full", "latest", sink) is True
    assert len(reports) >= 2
    assert reports[-1][0] == reports[-1][1] == 100
    assert "full" in {i.name for i in await fake.list_images("installed")}


def test_image_info_uses_the_corrected_latest_version_field() -> None:
    assert "latest_version" in ImageInfo.model_fields


# --- Job primitives: tagged start, pump-then-read, kill-by-tag and sweep, all Docker-free over the fake ----------


async def test_start_job_leaves_a_tagged_process() -> None:
    fake = FakeExegolAdapter()
    tag = secrets.token_urlsafe(16)
    handle = await fake.start_job("exegol-lab", "id", tag, cap_bytes=1024)
    assert isinstance(handle, JobHandle)
    assert handle.running is True
    assert handle.exit_code is None
    assert fake.job_processes.get(tag) == "exegol-lab"
    # The command is still recorded on exec_commands, so the existing exec/responsiveness assertions hold.
    assert fake.exec_commands == ["id"]


async def test_start_job_unknown_container_raises_value_error() -> None:
    fake = FakeExegolAdapter()
    with pytest.raises(ValueError, match="not found"):
        await fake.start_job("ghost", "id", secrets.token_urlsafe(16), 1024)


async def test_start_job_stopped_container_raises_runtime_error() -> None:
    fake = FakeExegolAdapter()
    await fake.stop_container("exegol-lab")
    with pytest.raises(RuntimeError):
        await fake.start_job("exegol-lab", "id", secrets.token_urlsafe(16), 1024)


async def test_pump_then_read_returns_the_canned_output_with_offsets_and_exit_code() -> None:
    fake = FakeExegolAdapter(exec_result=ExecResult(stdout="hello\n", stderr="warn\n", exit_code=0))
    tag = secrets.token_urlsafe(16)
    handle = await fake.start_job("exegol-lab", "echo hello", tag, cap_bytes=1024)
    await fake.pump_job(handle)
    assert handle.running is False
    assert handle.exit_code == 0
    assert tag not in fake.job_processes  # the process left the table when it finished
    read = await fake.read_job(handle, 0, 0, None)
    assert read.stdout == "hello\n"
    assert read.stderr == "warn\n"
    assert read.exit_code == 0
    assert read.running is False
    assert read.stdout_next_offset == len(b"hello\n")
    assert read.stderr_next_offset == len(b"warn\n")
    assert read.stdout_bytes_dropped == 0


async def test_kill_during_a_blocked_pump_removes_the_tag_and_ends_with_no_exit_code() -> None:
    fake = FakeExegolAdapter(exec_block_seconds=5.0, exec_result=ExecResult(stdout="partial\n", stderr="", exit_code=0))
    tag = secrets.token_urlsafe(16)
    handle = await fake.start_job("exegol-lab", "sleep 100", tag, cap_bytes=1024)
    async with anyio.create_task_group() as tg:
        tg.start_soon(fake.pump_job, handle)
        await anyio.sleep(0.05)  # let the pump enter its blocking wait off the loop
        assert tag in fake.job_processes
        await fake.kill_job("exegol-lab", tag, grace_s=0.0)
    assert handle.running is False
    assert handle.exit_code is None  # EXEC-06: a killed pump finishes with a null exit code, never 0
    assert tag not in fake.job_processes


async def test_sweep_orphans_empties_the_table_and_returns_the_count() -> None:
    fake = FakeExegolAdapter()
    t1, t2 = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
    await fake.start_job("exegol-lab", "a", t1, 1024)
    await fake.start_job("exegol-lab2", "b", t2, 1024)
    assert len(fake.job_processes) == 2
    killed = await fake.sweep_orphans()
    assert killed == 2
    assert fake.job_processes == {}


def test_server_context_exposes_the_job_primitives_as_coroutines() -> None:
    for name in JOB_METHODS:
        assert inspect.iscoroutinefunction(getattr(ServerContext, name)), name


async def test_concurrent_requests_each_see_only_their_own_state() -> None:
    assert request_ctx.get() is None
    seen: dict[str, list[RequestState | None]] = {}

    async def handle(request_id: str) -> None:
        state = RequestState(request_id=request_id)
        with bind_request(state):
            observed = [request_ctx.get()]
            for _ in range(5):
                await anyio.lowlevel.checkpoint()  # yield like anyio.sleep(0) so the other request runs between reads
                observed.append(request_ctx.get())
        observed.append(request_ctx.get())
        seen[request_id] = observed

    async with anyio.create_task_group() as tg:
        tg.start_soon(handle, "request-a")
        tg.start_soon(handle, "request-b")

    for request_id, observed in seen.items():
        *inside, after = observed
        assert all(state is not None and state.request_id == request_id for state in inside), observed
        assert after is None
    assert set(seen) == {"request-a", "request-b"}
    assert request_ctx.get() is None


# --- create_container on the fake (04-03): the same errors as the real adapter ------------------------------------

_LABELS = {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "abc123def456"}


async def test_fake_create_appends_a_running_container_and_records_the_spec() -> None:
    fake = FakeExegolAdapter()
    spec = CreateSpec(name="mcp-1a2b3c4d", image="free", labels=dict(_LABELS), comment="agent box")

    info = await fake.create_container(spec)

    assert info.name == "mcp-1a2b3c4d"
    assert info.comment == "agent box"
    assert info in await fake.list_containers()
    state = await fake.get_container_state("mcp-1a2b3c4d")
    assert state is not None and state.running is True
    assert fake.created_specs == [spec]
    assert fake.labels["mcp-1a2b3c4d"] == _LABELS


async def test_fake_create_refuses_a_name_already_in_use() -> None:
    fake = FakeExegolAdapter()
    await fake.create_container(CreateSpec(name="mcp-1", image="free"))

    with pytest.raises(ValueError, match="already exists"):
        await fake.create_container(CreateSpec(name="mcp-1", image="free"))
    with pytest.raises(ValueError, match="already exists"):
        await fake.create_container(CreateSpec(name="exegol-lab", image="free"))


@pytest.mark.parametrize("image", ["full", "ghost"])
async def test_fake_create_refuses_an_image_that_is_not_installed(image: str) -> None:
    fake = FakeExegolAdapter()

    with pytest.raises(ValueError, match="not installed"):
        await fake.create_container(CreateSpec(name="mcp-1", image=image))
    assert fake.created_specs == []


async def test_fake_create_refuses_a_missing_image() -> None:
    with pytest.raises(ValueError, match="list_images"):
        await FakeExegolAdapter().create_container(CreateSpec(name="mcp-1", image=None))



# --- managed/run_id on the fake (04-03 Task 2): the same exact-"true" rule as the real adapter ---------------------


async def test_fake_create_reports_and_lists_the_managed_labels() -> None:
    fake = FakeExegolAdapter()

    info = await fake.create_container(CreateSpec(name="mcp-1", image="free", labels=dict(_LABELS)))

    assert (info.managed, info.run_id, info.this_run) == (True, "abc123def456", False)
    [listed] = [c for c in await fake.list_containers() if c.name == "mcp-1"]
    assert (listed.managed, listed.run_id) == (True, "abc123def456")
    assert await fake.get_container_state("mcp-1") == ContainerState(running=True, managed=True, run_id="abc123def456")


async def test_fake_create_without_an_exact_true_label_is_unmanaged() -> None:
    fake = FakeExegolAdapter()

    info = await fake.create_container(
        CreateSpec(name="mcp-1", image="free", labels={MANAGED_LABEL: "True", RUN_ID_LABEL: "abc"})
    )

    assert (info.managed, info.run_id) == (False, None)
    assert await fake.get_container_state("mcp-1") == ContainerState(running=True)


async def test_a_fake_seeded_with_a_managed_container_from_an_earlier_run_recognizes_it() -> None:
    from tests.fakes import _container

    fake = FakeExegolAdapter(containers=[_container("exegol-old", None, managed=True, run_id="earlier-run")])

    assert fake.labels["exegol-old"] == {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "earlier-run"}
    [old] = await fake.list_containers()
    assert (old.managed, old.run_id) == (True, "earlier-run")
    assert await fake.get_container_state("exegol-old") == ContainerState(
        running=True, managed=True, run_id="earlier-run"
    )


# --- remove_container on the fake (04-05 Task 1): mirrors the real adapter's outcome and errors --------------------


async def test_fake_remove_drops_the_container_and_records_the_call() -> None:
    from exegol_mcp.adapter.port import RemovalOutcome

    fake = FakeExegolAdapter()
    await fake.create_container(CreateSpec(name="mcp-1", image="free", labels=dict(_LABELS)))

    outcome = await fake.remove_container("mcp-1", keep_workspace=False, keep_logs=True, timeout_s=30.0)

    assert outcome == RemovalOutcome(workspace_kept=False, logs_kept=True, host_workspace=False)
    assert fake.removals == [("mcp-1", False, True)]
    assert "mcp-1" not in fake.running
    assert "mcp-1" not in fake.labels
    assert all(c.name != "mcp-1" for c in await fake.list_containers())
    assert await fake.get_container_state("mcp-1") is None


async def test_fake_remove_keeps_what_the_call_asked_to_keep() -> None:
    from exegol_mcp.adapter.port import RemovalOutcome

    fake = FakeExegolAdapter()

    outcome = await fake.remove_container(
        "exegol-lab", keep_workspace=True, keep_logs=False, timeout_s=30.0, require_managed=False
    )

    assert outcome == RemovalOutcome(workspace_kept=True, logs_kept=False, host_workspace=False)


async def test_fake_remove_of_an_unknown_or_already_removed_name_raises() -> None:
    fake = FakeExegolAdapter()

    with pytest.raises(ValueError, match="not found"):
        await fake.remove_container("ghost", keep_workspace=True, keep_logs=True, timeout_s=30.0)
    await fake.remove_container(
        "exegol-lab", keep_workspace=True, keep_logs=True, timeout_s=30.0, require_managed=False
    )
    with pytest.raises(ValueError, match="not found"):
        await fake.remove_container(
            "exegol-lab", keep_workspace=True, keep_logs=True, timeout_s=30.0, require_managed=False
        )
    assert fake.removals == [("exegol-lab", True, True)]


async def test_fake_remove_applies_the_managed_gate_to_its_own_labels_by_default() -> None:
    """WR-02: the fake refuses like the real adapter, before mutating anything; the default fails closed."""
    from exegol_mcp.services.lifecycle import UNMANAGED_REMOVAL_MESSAGE

    fake = FakeExegolAdapter()
    fake.labels["exegol-lab2"] = {MANAGED_LABEL: "True"}

    for name in ("exegol-lab", "exegol-lab2"):
        with pytest.raises(ValueError) as caught:
            await fake.remove_container(name, keep_workspace=True, keep_logs=True, timeout_s=30.0)
        assert str(caught.value) == UNMANAGED_REMOVAL_MESSAGE.format(name=name)
    assert fake.removals == []
    assert {c.name for c in await fake.list_containers()} == {"exegol-lab", "exegol-lab2"}


async def test_fake_remove_checks_the_exact_run_id_when_one_is_expected() -> None:
    fake = FakeExegolAdapter()
    await fake.create_container(CreateSpec(name="mcp-1", image="free", labels=dict(_LABELS)))
    run_id = _LABELS[RUN_ID_LABEL]

    with pytest.raises(ValueError, match="is no longer one this server run created"):
        await fake.remove_container(
            "mcp-1", keep_workspace=True, keep_logs=True, timeout_s=30.0, expected_run_id=run_id.upper() + "x"
        )
    assert fake.removals == []
    await fake.remove_container("mcp-1", keep_workspace=True, keep_logs=True, timeout_s=30.0, expected_run_id=run_id)
    assert fake.removals == [("mcp-1", True, True)]


# --- read_file / write_file on the fake (04-09 Task 1): the real adapter's values and errors -----------------------


async def test_fake_reads_the_default_file_byte_for_byte() -> None:
    from exegol_mcp.adapter.port import FileRead

    result = await FakeExegolAdapter().read_file("exegol-lab", "/etc/../etc/hostname", 100, 30.0)

    assert result == FileRead(data=b"exegol-lab\n", size=11, over_limit=False, path="/etc/hostname")


async def test_fake_read_over_the_limit_keeps_at_most_one_extra_byte() -> None:
    fake = FakeExegolAdapter(files={"exegol-lab": {"/big": b"0123456789"}})

    result = await fake.read_file("exegol-lab", "/big", 4, 30.0)

    assert (result.data, result.size, result.over_limit) == (b"01234", 10, True)


async def test_fake_read_refuses_a_missing_file_and_a_directory() -> None:
    fake = FakeExegolAdapter()

    with pytest.raises(ValueError, match="No such file"):
        await fake.read_file("exegol-lab", "/etc/missing", 100, 30.0)
    with pytest.raises(ValueError, match="list_directory"):
        await fake.read_file("exegol-lab", "/etc", 100, 30.0)


async def test_fake_write_then_read_round_trips_and_reports_created() -> None:
    from exegol_mcp.adapter.port import FileWrite

    fake = FakeExegolAdapter()

    first = await fake.write_file("exegol-lab", "/tmp/new/a.bin", b"\x00\xff", True, 0o755, 30.0)
    second = await fake.write_file("exegol-lab", "/tmp/new/a.bin", b"\x01", True, None, 30.0)

    assert first == FileWrite(path="/tmp/new/a.bin", size=2, created=True)
    assert second == FileWrite(path="/tmp/new/a.bin", size=1, created=False)
    assert (await fake.read_file("exegol-lab", "/tmp/new/a.bin", 100, 30.0)).data == b"\x01"


async def test_fake_write_refuses_a_directory_and_an_existing_file_without_overwrite() -> None:
    fake = FakeExegolAdapter()

    with pytest.raises(ValueError, match="directory"):
        await fake.write_file("exegol-lab", "/etc", b"x", True, None, 30.0)
    with pytest.raises(ValueError, match="overwrite"):
        await fake.write_file("exegol-lab", "/etc/hostname", b"x", False, None, 30.0)
    assert (await fake.read_file("exegol-lab", "/etc/hostname", 100, 30.0)).data == b"exegol-lab\n"


@pytest.mark.parametrize("operation", ["read", "write"])
async def test_fake_file_calls_refuse_unknown_and_stopped_containers(operation: str) -> None:
    fake = FakeExegolAdapter()
    await fake.stop_container("exegol-lab2")

    async def call(name: str) -> object:
        if operation == "read":
            return await fake.read_file(name, "/etc/hostname", 100, 30.0)
        return await fake.write_file(name, "/tmp/a", b"x", True, None, 30.0)

    with pytest.raises(ValueError, match="not found"):
        await call("ghost")
    with pytest.raises(RuntimeError, match="not running"):
        await call("exegol-lab2")


def test_server_context_exposes_the_file_methods_as_coroutines() -> None:
    for name in FILE_METHODS:
        assert inspect.iscoroutinefunction(getattr(ServerContext, name)), name


# --- list_directory on the fake (04-09 Task 2): derived from the files dict, the real adapter's errors -----------


async def test_fake_lists_files_and_implied_directories_up_to_depth() -> None:
    fake = FakeExegolAdapter(files={"exegol-lab": {"/etc/hostname": b"lab\n", "/etc/ssl/certs/ca.pem": b"pem"}})

    one = await fake.list_directory("exegol-lab", "/etc", 1, 100, 30.0)
    two = await fake.list_directory("exegol-lab", "/etc", 2, 100, 30.0)

    assert one.path == "/etc" and one.truncated is False
    assert [(e.path, e.kind) for e in one.entries] == [("hostname", "file"), ("ssl", "dir")]
    assert [(e.path, e.kind) for e in two.entries] == [("hostname", "file"), ("ssl", "dir"), ("ssl/certs", "dir")]
    assert one.entries[0].size == 4 and one.entries[0].target is None


async def test_fake_lists_an_explicit_empty_directory_with_no_entries() -> None:
    fake = FakeExegolAdapter(dirs={("exegol-lab", "/tmp/empty")})

    listing = await fake.list_directory("exegol-lab", "/tmp/empty", 1, 100, 30.0)
    parent = await fake.list_directory("exegol-lab", "/tmp", 1, 100, 30.0)

    assert listing.entries == [] and listing.truncated is False
    assert [(e.path, e.kind) for e in parent.entries] == [("empty", "dir")]


async def test_fake_listing_caps_entries_and_sets_truncated() -> None:
    fake = FakeExegolAdapter(files={"exegol-lab": {f"/d/f{i}": b"x" for i in range(5)}})

    listing = await fake.list_directory("exegol-lab", "/d", 1, 3, 30.0)

    assert len(listing.entries) == 3 and listing.truncated is True


async def test_fake_listing_refuses_a_missing_path_a_file_and_unknown_or_stopped_containers() -> None:
    fake = FakeExegolAdapter()
    await fake.stop_container("exegol-lab2")

    with pytest.raises(ValueError, match="No such file or directory"):
        await fake.list_directory("exegol-lab", "/nope", 1, 100, 30.0)
    with pytest.raises(ValueError, match="read_file"):
        await fake.list_directory("exegol-lab", "/etc/hostname", 1, 100, 30.0)
    with pytest.raises(ValueError, match="not found"):
        await fake.list_directory("ghost", "/etc", 1, 100, 30.0)
    with pytest.raises(RuntimeError, match="not running"):
        await fake.list_directory("exegol-lab2", "/etc", 1, 100, 30.0)
