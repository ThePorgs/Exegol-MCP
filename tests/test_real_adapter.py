"""RealExegolAdapter behaviour against a fake DockerUtils, without Docker.

The fakes stand in for the exegol SDK objects the adapter touches (DockerUtils, ExegolContainer, ContainerConfig,
ExegolImage), so these tests check the adapter's own logic: mapping, scope filtering, not-found translation, the
offload-vs-loop split and per-request progress binding.
"""

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from exegol.exceptions.ExegolExceptions import ObjectNotFound
from mcp import Client

from exegol_mcp.adapter import real
from exegol_mcp.adapter.port import (
    IMAGE_NOT_DOWNLOADED_MESSAGE,
    IMAGE_UP_TO_DATE_MESSAGE,
    ContainerState,
    ExegolPort,
)
from exegol_mcp.adapter.real import RealExegolAdapter
from exegol_mcp.app import build_server
from exegol_mcp.context.request import request_ctx
from exegol_mcp.models.exec import ExecResult


class _Config:
    def __init__(self, all_features: bool) -> None:
        self.on = all_features

    def isGUIEnable(self) -> bool:
        return self.on

    isDesktopEnabled = isExegolResourcesEnable = isMyResourcesEnable = isGUIEnable
    isShellLoggingEnable = isTimezoneShared = isWrapperStartShared = isGUIEnable

    def getNetwork(self) -> tuple[str, None]:
        return "host", None

    def getVpnConfigPath(self) -> None:
        return None

    def getCreationDate(self) -> str:
        return "2026-10-01 12:00:00"

    def getComment(self) -> None:
        return None

    def getTextEnvs(self) -> str:
        return "A=1\nB=2\n"

    def getDevices(self) -> list[str]:
        return []

    def getPrivileged(self) -> bool:
        return False

    def getCapabilities(self) -> list[str]:
        return ["NET_ADMIN"]


class _Container:
    def __init__(self, name: str, running: bool = True, all_features: bool = False) -> None:
        self.name = name
        self.running = running
        self.config = _Config(all_features)
        self.image = SimpleNamespace(getName=lambda: "free", getImageVersion=lambda: "6.0.0")
        self.lifecycle_threads: list[int] = []

    def getRawStatus(self) -> str:
        return "running" if self.running else "exited"

    def isRunning(self) -> bool:
        return self.running

    async def start(self) -> None:
        self.lifecycle_threads.append(threading.get_ident())
        self.running = True

    async def stop(self) -> None:
        self.lifecycle_threads.append(threading.get_ident())
        self.running = False


def _image(name: str, installed: bool, up_to_date: bool | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        getName=lambda: name,
        getImageVersion=lambda: "6.0.0" if installed else "",
        getLatestVersion=lambda: "6.0.0",
        isInstall=lambda: installed,
        isLocal=lambda: False,
        isUpToDate=lambda: installed if up_to_date is None else up_to_date,
        getBuildDate=lambda: "",
        getDisplayLicense=lambda: "free",
        isVersionSpecific=lambda: False,
        hasVersionTag=lambda: False,
    )


class _FakeDocker:
    def __init__(self, containers: list[_Container], images: list[SimpleNamespace]) -> None:
        self.containers = containers
        self.images = images
        self.cache_clears = 0
        self.progress_seen: list[Any] = []
        self.download_calls: list[tuple[str, bool]] = []
        self.download_result = True

    def clearCache(self) -> None:
        self.cache_clears += 1

    async def listContainers(self) -> list[_Container]:
        return list(self.containers)

    def getContainer(self, tag: str) -> _Container:
        for container in self.containers:
            if container.name == tag:
                return container
        raise ObjectNotFound

    async def listImages(self, include_custom: bool = False) -> list[SimpleNamespace]:
        return list(self.images)

    async def getOfficialImageFromList(self, tag: str) -> SimpleNamespace:
        for image in self.images:
            if image.getName() == tag:
                return image
        raise ObjectNotFound

    async def downloadImage(self, image: SimpleNamespace, install_mode: bool = False) -> bool:
        state = request_ctx.get()
        self.progress_seen.append(state.progress if state else None)
        self.download_calls.append((image.getName(), install_mode))
        # Mirror DockerUtils.downloadImage (exegol 6.0.2): for an installed, up-to-date image updateCheck()
        # returns None, so nothing is pulled and the SDK hands back False.
        if image.isInstall() and image.isUpToDate():
            return False
        return self.download_result


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> _FakeDocker:
    fake = _FakeDocker(
        containers=[_Container("lab2", running=False), _Container("lab", all_features=True)],
        images=[_image("free", installed=False), _image("full", installed=False)],
    )
    monkeypatch.setattr(real, "DockerUtils", lambda: fake)
    return fake


def test_the_real_adapter_implements_the_port() -> None:
    assert isinstance(RealExegolAdapter(), ExegolPort)


async def test_containers_are_mapped_sorted_and_read_fresh(docker: _FakeDocker) -> None:
    containers = await RealExegolAdapter().list_containers()

    assert [c.name for c in containers] == ["lab", "lab2"]
    assert docker.cache_clears == 1
    lab = containers[0]
    assert lab.env == ["A=1", "B=2"]
    assert lab.capabilities == ["Docker default capabilities", "NET_ADMIN"]
    assert lab.status == "running"


async def test_each_container_feature_is_listed_once(docker: _FakeDocker) -> None:
    [lab, _] = await RealExegolAdapter().list_containers()

    assert len(lab.features) == len(set(lab.features))
    assert lab.features.count("Exegol-Resources") == 1
    assert "Shared-wrapper-start" in lab.features


async def test_container_state_is_none_for_an_unknown_name(docker: _FakeDocker) -> None:
    adapter = RealExegolAdapter()

    assert await adapter.get_container_state("ghost") is None
    assert await adapter.get_container_state("lab2") == ContainerState(running=False)


async def test_start_and_stop_return_the_new_state(docker: _FakeDocker) -> None:
    adapter = RealExegolAdapter()

    assert await adapter.start_container("lab2") is True
    assert await adapter.stop_container("lab2") is False


async def test_starting_an_unknown_container_raises_value_error(docker: _FakeDocker) -> None:
    with pytest.raises(ValueError, match="Container 'ghost' not found"):
        await RealExegolAdapter().start_container("ghost")


async def test_installed_scope_is_empty_when_nothing_is_installed(docker: _FakeDocker) -> None:
    adapter = RealExegolAdapter()

    assert await adapter.list_images("installed") == []
    assert [i.name for i in await adapter.list_images("all")] == ["free", "full"]


async def test_images_carry_the_latest_version_field(docker: _FakeDocker) -> None:
    [free, _] = await RealExegolAdapter().list_images("all")

    assert free.latest_version == "6.0.0"


async def test_download_binds_the_callers_progress_sink(docker: _FakeDocker) -> None:
    async def sink(progress: float, total: float | None, message: str | None) -> None:
        return None

    assert await RealExegolAdapter().download_image("free", "latest", sink) is True
    assert docker.progress_seen == [sink]
    assert request_ctx.get() is None


async def test_an_already_up_to_date_image_is_not_pulled_and_says_so(docker: _FakeDocker) -> None:
    docker.images = [_image("free", installed=True)]
    seen: list[tuple[float, float | None, str | None, bool]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message, real._sdk_cache_lock.locked()))

    assert await RealExegolAdapter().download_image("free", "latest", sink) is True
    # The up-to-date message is the single (and therefore last) report, sent outside the cache lock.
    assert seen == [(100, 100, IMAGE_UP_TO_DATE_MESSAGE.format(image="free"), False)]
    assert docker.download_calls == []

    # Again with no progress sink: still True, still no pull, and no request state leaks.
    assert await RealExegolAdapter().download_image("free", "latest", None) is True
    assert docker.download_calls == []
    assert request_ctx.get() is None


async def test_download_image_tool_already_up_to_date_returns_true_over_the_real_adapter(docker: _FakeDocker) -> None:
    docker.images = [_image("free", installed=True)]
    reports: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        reports.append((progress, total, message))

    async with Client(build_server(None, RealExegolAdapter())) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        result = await client.call_tool("download_image", {"image_name": "free"}, progress_callback=on_progress)

    assert not result.is_error
    assert result.structured_content == {"result": True}
    assert docker.download_calls == []
    assert reports[-1][2] == IMAGE_UP_TO_DATE_MESSAGE.format(image="free")
    description = tools["download_image"].description or ""
    assert "already installed and up to date is not downloaded again" in description
    assert "including when it already was" in description


async def test_an_outdated_installed_image_is_pulled_in_update_mode(docker: _FakeDocker) -> None:
    docker.images = [_image("free", installed=True, up_to_date=False)]

    assert await RealExegolAdapter().download_image("free", "latest", None) is True
    # install_mode is False: the image is installed, so this is an update, and the pull did happen.
    assert docker.download_calls == [("free", False)]


async def test_a_refused_pull_raises_was_not_downloaded(docker: _FakeDocker) -> None:
    docker.images = [_image("full", installed=False)]
    docker.download_result = False

    with pytest.raises(RuntimeError, match="Image 'full' was not downloaded"):
        await RealExegolAdapter().download_image("full", "latest", None)
    assert docker.download_calls == [("full", True)]


async def test_download_image_tool_was_not_downloaded_error_reaches_the_client(docker: _FakeDocker) -> None:
    docker.images = [_image("full", installed=False)]
    docker.download_result = False

    async with Client(build_server(None, RealExegolAdapter())) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        result = await client.call_tool("download_image", {"image_name": "full"})

    assert result.is_error
    text = " ".join(getattr(block, "text", "") for block in result.content)
    assert "Image 'full' was not downloaded" in text
    assert IMAGE_NOT_DOWNLOADED_MESSAGE.format(image="full") in text
    assert result.structured_content != {"result": False}
    assert "is an error, never false" in (tools["download_image"].description or "")


async def test_downloading_an_unknown_image_raises_value_error(docker: _FakeDocker) -> None:
    with pytest.raises(ValueError, match="Image 'ghost-1.0.0' not found"):
        await RealExegolAdapter().download_image("ghost", "1.0.0", None)


async def test_exec_in_a_stopped_container_raises(docker: _FakeDocker) -> None:
    with pytest.raises(RuntimeError, match="Container 'lab2' is not running"):
        await RealExegolAdapter().exec_command("lab2", "id")


async def test_exec_in_an_unknown_container_raises_value_error(docker: _FakeDocker) -> None:
    with pytest.raises(ValueError, match="Container 'ghost' not found"):
        await RealExegolAdapter().exec_command("ghost", "id")


async def test_exec_delegates_to_the_demuxed_path(docker: _FakeDocker, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, str]] = []

    async def fake_exec(container: Any, command: str) -> ExecResult:
        calls.append((container, command))
        return ExecResult(stdout="uid=0\n", stderr="", exit_code=0)

    monkeypatch.setattr(real, "exec_demuxed", fake_exec)

    result = await RealExegolAdapter().exec_command("lab", "id")

    assert result.stdout == "uid=0\n"
    assert calls == [(docker.containers[1], "id")]
