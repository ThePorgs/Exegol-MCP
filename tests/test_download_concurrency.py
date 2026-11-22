"""EXEC-09: concurrent image downloads keep their progress streams isolated, Docker-free.

Phase-1 groundwork binds each download's own RequestState in a ContextVar (bind_request), and report_progress
reports to request_ctx.get().progress — a ContextVar value is local to the task that set it, so two concurrent
downloads never report to each other's client. v1 kept the progress target in a shared class attribute, so the
last request to set it won. These tests prove the isolation by running two downloads lockstep (checkpoints between
every report so the two tasks interleave) and tagging each report with the bound request's id: a leaked ContextVar
would deliver a message to the wrong sink and fail the assertion. This is a test only — no production change.

Modelled on tests/test_port_contract.py::test_concurrent_requests_each_see_only_their_own_state.
"""

import anyio
import anyio.lowlevel
from mcp import Client

from exegol_mcp.adapter.download_progress import report_progress
from exegol_mcp.app import build_server
from exegol_mcp.context.request import ProgressCallback, request_ctx
from exegol_mcp.services import images as images_service
from tests.fakes import FakeExegolAdapter

# The staged progress milestones each download reports, in order; the last is the shared "Download complete".
_STAGES = ((0, "Downloading layers: 0 / 2"), (50, "Extracting layers: 1 / 2"), (100, "Download complete"))


class _InterleavedDownloadAdapter(FakeExegolAdapter):
    """download_image reports staged progress through the production request-scoped sink (report_progress reads
    request_ctx), yielding between every report so two concurrent downloads run interleaved. Each message is
    tagged with the bound request's id, so a leaked ContextVar would send a report to the wrong sink."""

    async def download_image(self, name: str, version: str, progress: ProgressCallback | None) -> bool:
        state = request_ctx.get()
        request_id = state.request_id if state is not None else "<unbound>"
        for pct, label in _STAGES:
            await anyio.lowlevel.checkpoint()  # yield like anyio.sleep(0) so the other download runs between reports
            await report_progress(pct, 100, f"{request_id}:{label}")
        return True


def _recording_sink() -> tuple[ProgressCallback, list[tuple[float, str | None]]]:
    seen: list[tuple[float, str | None]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, message))

    return sink, seen


async def test_two_concurrent_downloads_each_receive_only_their_own_progress() -> None:
    adapter = _InterleavedDownloadAdapter()
    sink_a, seen_a = _recording_sink()
    sink_b, seen_b = _recording_sink()

    async def download(request_id: str, sink: ProgressCallback) -> None:
        await images_service.download_image(adapter, "full", "latest", progress=sink, request_id=request_id)

    async with anyio.create_task_group() as tg:
        tg.start_soon(download, "req-a", sink_a)
        tg.start_soon(download, "req-b", sink_b)

    # Each sink saw every stage of its OWN download and nothing from the other (no ContextVar leak).
    assert [msg for _, msg in seen_a] == [f"req-a:{label}" for _, label in _STAGES]
    assert [msg for _, msg in seen_b] == [f"req-b:{label}" for _, label in _STAGES]
    # Both downloads reached the final 'Download complete'.
    assert seen_a[-1] == (100, "req-a:Download complete")
    assert seen_b[-1] == (100, "req-b:Download complete")


async def test_an_interleaved_download_never_leaks_a_message_into_the_other_stream() -> None:
    # A stronger edge: with reports interleaved, neither sink ever holds a message tagged with the other's id.
    adapter = _InterleavedDownloadAdapter()
    sink_a, seen_a = _recording_sink()
    sink_b, seen_b = _recording_sink()

    async def download(request_id: str, sink: ProgressCallback) -> None:
        await images_service.download_image(adapter, "full", "latest", progress=sink, request_id=request_id)

    async with anyio.create_task_group() as tg:
        tg.start_soon(download, "alpha", sink_a)
        tg.start_soon(download, "beta", sink_b)

    assert all(msg is not None and msg.startswith("alpha:") for _, msg in seen_a), seen_a
    assert all(msg is not None and msg.startswith("beta:") for _, msg in seen_b), seen_b
    # The request state is cleared once both downloads return (bind_request resets it in a finally).
    assert request_ctx.get() is None


# --- D-11 integration: the lifespan sweeps tagged orphans at start and shutdown (the phase gate) ----------


async def test_the_lifespan_sweeps_tagged_orphans_at_start_and_shutdown() -> None:
    # A crashed previous run left a tagged process in the (fake) in-container process table. Entering and exiting
    # the server lifespan must sweep it at BOTH ends (D-11), so a crash/restart leaves no tagged processes.
    adapter = FakeExegolAdapter()
    adapter.job_processes["orphan-from-a-crash"] = "exegol-lab"
    server = build_server(None, adapter)
    async with Client(server) as client:
        # The startup sweep already ran before the server served its first request: the orphan is gone.
        assert adapter.sweep_calls >= 1
        assert adapter.job_processes == {}
        assert "orphan-from-a-crash" not in adapter.job_processes
        await client.list_tools()  # the server is serving normally after the startup sweep
    # The shutdown sweep ran too, so the sweep happened at both ends of the lifespan.
    assert adapter.sweep_calls == 2
