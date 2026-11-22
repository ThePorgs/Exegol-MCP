"""CR-02 regression: concurrent tool calls must not interleave inside the Exegol SDK's DockerUtils cache.

The EXEC-01 offload lets tool calls overlap. exegol 6.0.2's DockerUtils is a process-wide singleton whose cache the
adapter clears and refills on every listing: clearCache() (DockerUtils.py lines 81-85) nulls it from any thread, and
listContainers() (lines 145-167) assigns a new empty list to the instance, makes a blocking API call, appends into
whatever list the instance points to by then, and returns the instance attribute. Two overlapping listings therefore
returned duplicated containers (or None) before the fix.

_RacyDocker reproduces that algorithm without Docker and counts how many callers are inside its cache section at once,
so these tests prove the adapter's _sdk_cache_lock serializes every cache-touching sequence, and (inversion) that the
same calls do overlap once the lock is replaced by a no-op.
"""

import contextlib
import threading
import time
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from exegol.exceptions.ExegolExceptions import ObjectNotFound

from exegol_mcp.adapter import real
from exegol_mcp.adapter.real import RealExegolAdapter
from tests.test_real_adapter import _Container, _image

CONTAINER_NAMES = ["exegol-a", "exegol-b"]


class _RacyDocker:
    """A DockerUtils fake that mirrors the exegol 6.0.2 cache algorithm and records overlap in its cache section."""

    def __init__(self) -> None:
        self.containers_cache: list[_Container] | None = None
        self.images_cache: list[SimpleNamespace] | None = None
        self._counter = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.clears_during_listing = 0
        self.pulled: list[str] = []
        # When set, a cache fill blocks on this event instead of sleeping (cancellation test).
        self.gate: threading.Event | None = None

    def _enter(self) -> None:
        with self._counter:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _leave(self) -> None:
        with self._counter:
            self.active -= 1

    def clearCache(self) -> None:
        with self._counter:
            if self.active > 0:
                self.clears_during_listing += 1
        self.containers_cache = None
        self.images_cache = None

    async def listContainers(self) -> list[_Container] | None:
        # Never awaits, like the SDK's, so the adapter drives it to completion in its worker thread.
        self._enter()
        try:
            if self.containers_cache is None:
                self.containers_cache = []
                if self.gate is not None:
                    self.gate.wait()
                else:
                    time.sleep(0.2)  # noqa: ASYNC251 - mirrors the SDK's blocking docker API call
                for name in CONTAINER_NAMES:
                    self.containers_cache.append(_Container(name))
        finally:
            self._leave()
        return self.containers_cache

    async def listImages(self, include_custom: bool = False) -> list[SimpleNamespace]:
        # Mirrors DockerUtils.listImages (lines 513-550): suspend on the loop while filling, assign, read back, assert.
        self._enter()
        try:
            if self.images_cache is None:
                await anyio.sleep(0.2)
                self.images_cache = [_image("free", installed=True), _image("full", installed=False)]
            images = self.images_cache
            if images is None:
                raise AssertionError("image cache cleared between assign and read")
        finally:
            self._leave()
        return list(images)

    async def getOfficialImageFromList(self, tag: str) -> SimpleNamespace:
        for image in await self.listImages():
            if image.getName() == tag:
                return image
        raise ObjectNotFound

    async def downloadImage(self, image: SimpleNamespace, install_mode: bool = False) -> bool:
        # The pull does not touch the cache: it runs outside the section.
        self.pulled.append(image.getName())
        await anyio.sleep(0.2)
        return True


@pytest.fixture
def racy_docker(monkeypatch: pytest.MonkeyPatch) -> _RacyDocker:
    fake = _RacyDocker()
    monkeypatch.setattr(real, "DockerUtils", lambda: fake)
    return fake


async def _overlapping_listings(count: int) -> list[Any]:
    """Run count list_containers calls at once; return each call's names, or the exception it raised."""
    results: list[Any] = [None] * count

    async def one(index: int) -> None:
        try:
            results[index] = [c.name for c in await RealExegolAdapter().list_containers()]
        except (AttributeError, TypeError) as error:
            # Without the lock a concurrent clearCache can null the cache mid-fill (append to None) or before the
            # return (iterating None). The inversion test records that crash instead of failing on it.
            results[index] = error

    async with anyio.create_task_group() as tg:
        for index in range(count):
            tg.start_soon(one, index)
    return results


async def test_overlapping_list_containers_each_list_every_container_once(racy_docker: _RacyDocker) -> None:
    results = await _overlapping_listings(3)

    assert results == [CONTAINER_NAMES, CONTAINER_NAMES, CONTAINER_NAMES]
    assert racy_docker.max_active == 1


async def test_without_the_lock_overlapping_listings_overlap_in_the_cache(
    racy_docker: _RacyDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(real, "_sdk_cache_lock", contextlib.nullcontext())

    await _overlapping_listings(3)

    assert racy_docker.max_active > 1


async def test_mixed_listing_and_download_calls_never_overlap_in_the_cache(racy_docker: _RacyDocker) -> None:
    adapter = RealExegolAdapter()
    results: dict[str, Any] = {}

    async def containers() -> None:
        results["containers"] = [c.name for c in await adapter.list_containers()]

    async def installed() -> None:
        results["installed"] = [i.name for i in await adapter.list_images("installed")]

    async def all_images() -> None:
        results["all"] = [i.name for i in await adapter.list_images("all")]

    async def download() -> None:
        # "full" is not installed, so this really pulls and overlaps the listings (unlike the up-to-date "free",
        # which the Task 1 short-circuit would skip, leaving no real pull to overlap the cache section).
        results["download"] = await adapter.download_image("full", "latest", None)

    async with anyio.create_task_group() as tg:
        for call in (containers, installed, all_images, download):
            tg.start_soon(call)

    assert results == {
        "containers": CONTAINER_NAMES,
        "installed": ["free"],
        "all": ["free", "full"],
        "download": True,
    }
    assert racy_docker.pulled == ["full"]
    assert racy_docker.max_active == 1
    assert racy_docker.clears_during_listing == 0


async def test_a_cancelled_listing_keeps_the_cache_locked_until_its_thread_finishes(racy_docker: _RacyDocker) -> None:
    gate = threading.Event()
    racy_docker.gate = gate
    adapter = RealExegolAdapter()
    first_scope = anyio.CancelScope()
    second: dict[str, list[str]] = {}

    async def first_call() -> None:
        with first_scope:
            await adapter.list_containers()

    async def second_call() -> None:
        second["names"] = [c.name for c in await adapter.list_containers()]

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            try:
                tg.start_soon(first_call)
                while racy_docker.active != 1:
                    await anyio.sleep(0.01)
                # The first call's worker thread is now blocked inside the cache section.
                first_scope.cancel()
                tg.start_soon(second_call)
                await anyio.sleep(0.1)
                assert racy_docker.max_active == 1
            finally:
                # Always release the blocked worker thread, or a failure here would hang the shielded task forever.
                gate.set()

    assert second["names"] == CONTAINER_NAMES
    assert racy_docker.max_active == 1
