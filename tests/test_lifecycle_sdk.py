"""The real create choreography (LIFE-01) and the label read-back (LIFE-06), without Docker.

The stubs stand in for the exegol SDK names lifecycle_sdk imports (DockerUtils, ExegolContainerTemplate,
ask_user_for_value, OptionResolver), so these tests check the adapter's own logic: the CLI tier it sets and
restores, the labels it stamps, the lock and the shield around the create, the name re-check and the error
translation. The live create path runs in plan 04-12's opt-in suite.
"""

import errno
import inspect
import logging
import os
import shutil
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import anyio
import anyio.lowlevel
import pytest
import requests
from docker.errors import APIError as DockerAPIError
from docker.models.containers import ContainerCollection
from exegol.config.OptionResolver import OptionKey
from exegol.exceptions.ExegolExceptions import ObjectNotFound
from exegol.utils.DockerUtils import DockerUtils
from mcp import Client

from exegol_mcp.adapter import errors as adapter_errors
from exegol_mcp.adapter import lifecycle_sdk, sdk_headless
from exegol_mcp.adapter.errors import HeadlessPromptError
from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.adapter.port import (
    IMAGE_REQUIRED_MESSAGE,
    MANAGED_LABEL,
    MANAGED_LABEL_VALUE,
    NAME_IN_USE_MESSAGE,
    RUN_ID_LABEL,
    CreateSpec,
)
from exegol_mcp.adapter.real import RealExegolAdapter
from exegol_mcp.app import build_server
from exegol_mcp.models.config import LifecycleSettings
from exegol_mcp.models.container import ContainerInfo
from exegol_mcp.services import lifecycle as lifecycle_service
from tests.fakes import FakeExegolAdapter
from tests.test_responsiveness import BLOCK_SECONDS, MAX_CONCURRENT_LATENCY

# The CLI dests a create may set; each must read its previous value again once the create is over.
_TIER_DESTS = ("imagetag", "workspace_path", "network", "ports", "envs", "hostname", "comment", "gui", "desktop")


@pytest.fixture
def sdk_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore every SDK global patch_exegol_sdk() mutates, including the stdinCanAnswer patch point."""
    import exegol.model.ExegolContainer as container_module
    import exegol.profile.ProfileAskUser as ask_module
    from exegol.console.ExegolPrompt import ExegolRich
    from exegol.console.TUI import ExegolTUI
    from exegol.utils.ExeLog import ExeLog, logger
    from exegol.utils.MetaSingleton import MetaSingleton

    monkeypatch.setattr(MetaSingleton, "_MetaSingleton__instances", dict(MetaSingleton._MetaSingleton__instances))
    monkeypatch.setattr(ExeLog, "console", ExeLog.console)
    monkeypatch.setattr(logger, "_ExeLog__critical_method", logger._ExeLog__critical_method)
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "console"):
            monkeypatch.setattr(handler, "console", handler.console)
    for name in ("Ask", "Confirm", "Acknowledge"):
        monkeypatch.setattr(ExegolRich, name, ExegolRich.__dict__[name])
    monkeypatch.setattr(ExegolTUI, "downloadDockerLayer", ExegolTUI.__dict__["downloadDockerLayer"])
    monkeypatch.setattr(container_module, "ContainerLogStream", container_module.ContainerLogStream)
    monkeypatch.setattr(container_module, "secure_remove", container_module.secure_remove)
    monkeypatch.setattr(ask_module, "stdinCanAnswer", ask_module.stdinCanAnswer)
    monkeypatch.setattr(ContainerCollection, "create", ContainerCollection.__dict__["create"])
    for name in sdk_headless._PATCHED_CONTAINER_METHODS:
        monkeypatch.setattr(container_module.ExegolContainer, name, container_module.ExegolContainer.__dict__[name])
    monkeypatch.setattr(sdk_headless, "_parameters", sdk_headless._parameters)


@pytest.fixture
def patched(sdk_restore: None) -> Any:
    """The headless SDK with the operator's --accept-eula given, as the server runs it."""
    sdk_headless.patch_exegol_sdk()
    sdk_headless.set_accept_eula(True)
    return sdk_headless._parameters


# --- SDK stubs ---------------------------------------------------------------------------------------------------


class _Config:
    """The ContainerConfig getters _container_info reads, plus addLabel recording into the shared event log."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.labels: dict[str, str] = {}

    def addLabel(self, key: str, value: str) -> None:
        self.events.append(("label", key, value))
        self.labels[key] = value

    def isGUIEnable(self) -> bool:
        return False

    isDesktopEnabled = isExegolResourcesEnable = isMyResourcesEnable = isGUIEnable
    isShellLoggingEnable = isTimezoneShared = isWrapperStartShared = isGUIEnable

    def getNetwork(self) -> tuple[str, None]:
        return "host", None

    def getVpnConfigPath(self) -> None:
        return None

    def getCreationDate(self) -> str:
        return "2026-11-04 12:00:00"

    def getComment(self) -> None:
        return None

    def getTextEnvs(self) -> str:
        return ""

    def getDevices(self) -> list[str]:
        return []

    def getPrivileged(self) -> bool:
        return False

    def getCapabilities(self) -> list[str]:
        return []


class _Model:
    def __init__(self, name: str, image: Any, hostname: str | None, events: list[Any]) -> None:
        self.name = name
        self.image = image
        self.hostname = hostname
        self.config = _Config(events)


class _Created:
    """The ExegolContainer createContainer hands back: a config, an image and a docker-py handle with labels."""

    def __init__(self, model: _Model, docker: "_StubDocker") -> None:
        self.name = model.name
        self.config = model.config
        self.image = SimpleNamespace(getName=lambda: model.image.getName(), getImageVersion=lambda: "6.0.0")
        self._ExegolContainer__container = SimpleNamespace(labels=dict(model.config.labels))
        self._docker = docker
        self.post_create_completed = False

    def getRawStatus(self) -> str:
        return "running"

    def isRunning(self) -> bool:
        return True

    async def postCreateSetup(self) -> None:
        self._docker.events.append(("post_create", self.name))
        self._docker.post_create_started.set()
        await anyio.sleep(self._docker.post_create_seconds)
        self.post_create_completed = True


class _StubDocker:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.existing: set[str] = set()
        self.installed: set[str] = {"free", "full"}
        self.create_threads: list[int] = []
        self.created: list[_Created] = []
        self.post_create_seconds = 0.0
        self.post_create_started = anyio.Event()
        # Lookups of a name answered "free" before it reads as taken: a same-name create landing in between.
        self.free_lookups: dict[str, int] = {}
        # Run in createContainer's worker thread before it creates (a raise stands in for docker refusing it).
        self.on_create: Callable[[], None] | None = None

    def getContainer(self, tag: str) -> Any:
        self.events.append(("lookup", tag))
        if tag in self.free_lookups:
            self.free_lookups[tag] -= 1
            if self.free_lookups[tag] < 0:
                return SimpleNamespace(name=tag)
            raise ObjectNotFound
        if tag in self.existing:
            return SimpleNamespace(name=tag)
        raise ObjectNotFound

    async def getInstalledImage(self, tag: str) -> Any:
        self.events.append(("image", tag))
        if tag not in self.installed:
            # What the SDK's logger.critical does once patch_exegol_sdk() switched it to "raise".
            raise RuntimeError(f"The desired image is not installed or do not exist ({tag}). Exiting.")
        return SimpleNamespace(getName=lambda: tag)

    def createContainer(self, model: _Model) -> _Created:
        self.create_threads.append(threading.get_ident())
        if self.on_create is not None:
            self.on_create()
        self.events.append(("create", model.name))
        created = _Created(model, self)
        self.created.append(created)
        return created


class _StubTemplate:
    """ExegolContainerTemplate.newContainer, recording the CLI tier it ran under."""

    def __init__(self, docker: _StubDocker) -> None:
        self.docker = docker
        self.seen: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str | None]] = []
        self.raise_error: Exception | None = None
        self.suspend_seconds = 0.0
        # Awaited while the model is built, standing in for SDK work that reaches a prompt (research A5).
        self.during: Callable[[], Awaitable[None]] | None = None

    async def newContainer(self, name: str, image: Any, config: Any = None, hostname: str | None = None) -> _Model:
        self.calls.append((name, hostname))
        parameters = sdk_headless._parameters
        before = {dest: getattr(parameters, dest) for dest in _TIER_DESTS}
        # Give a concurrent create the chance to overwrite the tier if the lock did not serialize them.
        await anyio.sleep(self.suspend_seconds)
        after = {dest: getattr(parameters, dest) for dest in _TIER_DESTS}
        if self.during is not None:
            await self.during()
        # The profile tier the stubbed resolver holds while this model is built (None: no profile installed).
        profile_tier = _StubResolver.tiers["profile"]
        self.seen.append({"name": name, "before": before, "after": after, "profile_tier": profile_tier})
        if self.raise_error is not None:
            raise self.raise_error
        return _Model(name, image, hostname, self.docker.events)


class _StubResolver:
    """OptionResolver's two profile tiers and get(): the profile tier answers get() (the CLI tier is the spec's)."""

    calls: ClassVar[list[tuple[str, Any]]] = []
    tiers: ClassVar[dict[str, Any]] = {"profile": None, "user": None}

    def loadProfile(self, profile: Any) -> None:
        self.calls.append(("loadProfile", profile))
        self.tiers["profile"] = profile

    def loadProfileUserConfig(self, profile_user_config: Any) -> None:
        self.calls.append(("loadProfileUserConfig", profile_user_config))
        self.tiers["user"] = profile_user_config

    def get(self, key: Any) -> Any:
        self.calls.append(("get", key))
        profile = self.tiers["profile"]
        return None if profile is None else profile.get(key)


class _StubProfiles:
    """adapter/profiles.load_allowed_profile and Exegol's apply_profile, at the names lifecycle_sdk imports."""

    def __init__(self, docker: _StubDocker) -> None:
        self.docker = docker
        self.available: dict[str, Any] = {"web": SimpleNamespace(name="web", image_tag="free")}
        self.load_error: Exception | None = None
        self.apply_error: Exception | None = None
        self.loads: list[tuple[str, int]] = []
        self.applied: list[tuple[Any, bool, int]] = []

    def load(self, selection: str) -> Any:
        self.loads.append((selection, threading.get_ident()))
        self.docker.events.append(("load_profile", selection))
        if self.load_error is not None:
            raise self.load_error
        return self.available[selection]

    def apply(self, profile: Any) -> None:
        self.applied.append((profile, lifecycle_sdk._create_lock.locked(), threading.get_ident()))
        self.docker.events.append(("apply_profile", profile.name))
        if self.apply_error is not None:
            # What Exegol's logger.critical raises for a refused profile host path once the SDK is patched.
            raise self.apply_error
        resolver = lifecycle_sdk.OptionResolver()
        resolver.loadProfileUserConfig({"profile": profile.name})
        resolver.loadProfile({OptionKey.IMAGE_TAG: profile.image_tag, "profile": profile.name})


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch, patched: Any) -> _StubDocker:
    stub = _StubDocker()
    monkeypatch.setattr(lifecycle_sdk, "DockerUtils", lambda: stub)
    return stub


@pytest.fixture
def template(monkeypatch: pytest.MonkeyPatch, docker: _StubDocker) -> _StubTemplate:
    stub = _StubTemplate(docker)
    monkeypatch.setattr(lifecycle_sdk, "ExegolContainerTemplate", stub)
    return stub


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch, docker: _StubDocker) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    async def ask_user_for_value(
        ask_key: Any, value_key: Any, prompt: str, fallback_default: str | None = None
    ) -> str | None:
        calls.append((ask_key, value_key, prompt, fallback_default))
        return fallback_default

    monkeypatch.setattr(lifecycle_sdk, "ask_user_for_value", ask_user_for_value)
    return calls


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(_StubResolver, "calls", calls)
    monkeypatch.setattr(_StubResolver, "tiers", {"profile": None, "user": None})
    monkeypatch.setattr(lifecycle_sdk, "OptionResolver", _StubResolver)
    return calls


@pytest.fixture
def profiles(monkeypatch: pytest.MonkeyPatch, docker: _StubDocker, resolver: list[tuple[str, Any]]) -> _StubProfiles:
    stub = _StubProfiles(docker)
    # raising=False: before 04-08 lifecycle_sdk imported neither name, so the tests fail on their assertions.
    monkeypatch.setattr(lifecycle_sdk, "load_allowed_profile", stub.load, raising=False)
    monkeypatch.setattr(lifecycle_sdk, "apply_profile", stub.apply, raising=False)
    return stub


def _spec(name: str = "mcp-1a2b3c4d", image: str | None = "free", **overrides: Any) -> CreateSpec:
    labels = overrides.pop("labels", {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "abc123def456"})
    return CreateSpec(name=name, image=image, labels=labels, **overrides)


def _tier(parameters: Any) -> dict[str, Any]:
    return {dest: getattr(parameters, dest) for dest in _TIER_DESTS}


# --- The create choreography (Task 1) ----------------------------------------------------------------------------


@pytest.mark.usefixtures("asked", "resolver")
async def test_create_runs_new_container_under_the_spec_tier_and_restores_it(
    patched: Any, docker: _StubDocker, template: _StubTemplate
) -> None:
    patched.set_parameter("comment", "operator comment set earlier")
    previous = _tier(patched)
    spec = _spec(
        workspace="/srv/engagement",
        network_mode="bridge",
        ports=("8080:80",),
        env=("A=1", "B=2"),
        hostname="box",
        comment="agent box",
    )

    info = await RealExegolAdapter().create_container(spec)

    assert info.name == "mcp-1a2b3c4d"
    [seen] = template.seen
    assert seen["name"] == "mcp-1a2b3c4d"
    assert seen["before"] == {
        "imagetag": "free",
        "workspace_path": "/srv/engagement",
        "network": "bridge",
        "ports": ["8080:80"],
        "envs": ["A=1", "B=2"],
        "hostname": "box",
        "comment": "agent box",
        "gui": False,
        "desktop": False,
    }
    assert _tier(patched) == previous
    assert patched.accept_eula is True


@pytest.mark.usefixtures("asked", "resolver")
async def test_gui_and_desktop_are_forced_off_and_unset_fields_are_left_alone(
    patched: Any, docker: _StubDocker, template: _StubTemplate
) -> None:
    await RealExegolAdapter().create_container(_spec())

    [seen] = template.seen
    assert seen["before"] == {
        "imagetag": "free",
        "workspace_path": None,
        "network": None,
        "ports": None,
        "envs": None,
        "hostname": None,
        "comment": None,
        "gui": False,
        "desktop": False,
    }


@pytest.mark.usefixtures("resolver")
async def test_labels_are_stamped_before_create_which_runs_in_a_worker_thread(
    docker: _StubDocker, template: _StubTemplate, asked: list[tuple[Any, ...]]
) -> None:
    loop_thread = threading.get_ident()

    await RealExegolAdapter().create_container(_spec())

    labels = [event for event in docker.events if event[0] == "label"]
    assert labels == [("label", MANAGED_LABEL, "true"), ("label", RUN_ID_LABEL, "abc123def456")]
    create_at = docker.events.index(("create", "mcp-1a2b3c4d"))
    assert all(docker.events.index(label) < create_at for label in labels)
    assert docker.create_threads and docker.create_threads[0] != loop_thread
    assert docker.events[-1] == ("post_create", "mcp-1a2b3c4d")
    assert docker.created[0].post_create_completed is True
    # The hostname helper got Exegol's own naming rule as its fallback, and newContainer got its answer.
    [(_, _, _, fallback)] = asked
    assert fallback == "exegol-mcp-1a2b3c4d"
    assert template.calls == [("mcp-1a2b3c4d", "exegol-mcp-1a2b3c4d")]


@pytest.mark.usefixtures("asked", "resolver")
async def test_a_name_in_use_is_refused_inside_the_lock_before_any_create(
    docker: _StubDocker, template: _StubTemplate
) -> None:
    docker.existing.add("taken")

    with pytest.raises(ValueError) as error:
        await RealExegolAdapter().create_container(_spec(name="taken"))

    assert str(error.value) == NAME_IN_USE_MESSAGE.format(name="taken")
    assert template.calls == []
    assert not any(event[0] == "create" for event in docker.events)


@pytest.mark.usefixtures("asked", "resolver")
async def test_an_image_that_is_not_installed_is_refused_and_never_pulled(
    patched: Any, docker: _StubDocker, template: _StubTemplate
) -> None:
    previous = _tier(patched)

    with pytest.raises(ValueError, match="not installed") as error:
        await RealExegolAdapter().create_container(_spec(image="nightly"))

    assert "download_image" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert template.calls == []
    assert _tier(patched) == previous


@pytest.mark.usefixtures("asked", "resolver")
async def test_a_missing_image_is_refused_with_the_image_required_message(
    docker: _StubDocker, template: _StubTemplate
) -> None:
    with pytest.raises(ValueError) as error:
        await RealExegolAdapter().create_container(_spec(image=None))

    assert str(error.value) == IMAGE_REQUIRED_MESSAGE
    assert template.calls == []


@pytest.mark.usefixtures("asked", "resolver")
async def test_a_start_that_overruns_the_timeout_raises_a_clear_runtime_error(
    docker: _StubDocker, template: _StubTemplate
) -> None:
    docker.post_create_seconds = 5.0

    with pytest.raises(RuntimeError, match="did not finish starting") as error:
        await RealExegolAdapter().create_container(_spec(start_timeout_s=0.05))

    assert "list_containers" in str(error.value)


@pytest.mark.usefixtures("asked")
async def test_a_failing_new_container_still_restores_every_tier(
    patched: Any, docker: _StubDocker, template: _StubTemplate, resolver: list[tuple[str, Any]]
) -> None:
    previous = _tier(patched)
    template.raise_error = RuntimeError("configFromUser failed")

    with pytest.raises(RuntimeError, match="configFromUser failed"):
        await RealExegolAdapter().create_container(_spec(hostname="box", ports=("1:1",)))

    assert _tier(patched) == previous
    assert patched.accept_eula is True
    assert ("loadProfile", None) in resolver
    assert ("loadProfileUserConfig", None) in resolver


@pytest.mark.usefixtures("asked", "resolver")
async def test_two_concurrent_creates_each_see_only_their_own_tier(
    docker: _StubDocker, template: _StubTemplate
) -> None:
    template.suspend_seconds = 0.05
    adapter = RealExegolAdapter()

    async with anyio.create_task_group() as tg:
        tg.start_soon(adapter.create_container, _spec(name="one", image="free"))
        tg.start_soon(adapter.create_container, _spec(name="two", image="full"))

    by_name = {seen["name"]: seen for seen in template.seen}
    assert by_name["one"]["before"]["imagetag"] == by_name["one"]["after"]["imagetag"] == "free"
    assert by_name["two"]["before"]["imagetag"] == by_name["two"]["after"]["imagetag"] == "full"


@pytest.mark.usefixtures("asked", "resolver")
async def test_a_cancelled_create_still_finishes_its_shielded_section(
    patched: Any, docker: _StubDocker, template: _StubTemplate
) -> None:
    previous = _tier(patched)
    docker.post_create_seconds = 0.1

    async with anyio.create_task_group() as tg:
        tg.start_soon(RealExegolAdapter().create_container, _spec())
        await docker.post_create_started.wait()
        tg.cancel_scope.cancel()

    assert docker.created[0].post_create_completed is True
    assert _tier(patched) == previous


# --- Creating from an Exegol profile (04-08 Task 1, LIFE-02) ------------------------------------------------------

_VPN_DNS_PROMPT = "Your VPN configuration won't support dynamic DNS servers."
_PROFILE_EVENTS = {"lookup", "load_profile", "apply_profile", "image", "create"}


def _torn_down() -> bool:
    return _StubResolver.tiers == {"profile": None, "user": None}


@pytest.mark.usefixtures("asked")
async def test_a_profile_is_loaded_off_the_loop_and_applied_inside_the_lock_before_the_image(
    docker: _StubDocker, template: _StubTemplate, profiles: _StubProfiles
) -> None:
    loop_thread = threading.get_ident()

    await RealExegolAdapter().create_container(_spec(image=None, profile="web"))

    [(selection, load_thread)] = profiles.loads
    assert selection == "web"
    assert load_thread != loop_thread
    [(applied, lock_held, apply_thread)] = profiles.applied
    assert applied is profiles.available["web"]
    assert lock_held is True
    assert apply_thread == loop_thread
    assert [event for event in docker.events if event[0] in _PROFILE_EVENTS] == [
        ("lookup", "mcp-1a2b3c4d"),
        ("load_profile", "web"),
        ("apply_profile", "web"),
        ("image", "free"),
        # The final name check right before createContainer (WR-03).
        ("lookup", "mcp-1a2b3c4d"),
        ("create", "mcp-1a2b3c4d"),
    ]
    assert _torn_down()


@pytest.mark.usefixtures("asked", "profiles")
async def test_with_a_profile_gui_desktop_and_a_missing_image_are_left_off_the_cli_tier(
    patched: Any, docker: _StubDocker, template: _StubTemplate
) -> None:
    previous = _tier(patched)

    await RealExegolAdapter().create_container(_spec(image=None, profile="web"))

    [seen] = template.seen
    assert seen["before"] == dict.fromkeys(_TIER_DESTS)
    assert seen["profile_tier"][OptionKey.IMAGE_TAG] == "free"
    assert _tier(patched) == previous


@pytest.mark.usefixtures("asked", "profiles")
async def test_an_image_given_on_the_call_replaces_the_profiles_image(
    docker: _StubDocker, template: _StubTemplate
) -> None:
    await RealExegolAdapter().create_container(_spec(image="full", profile="web"))

    assert [event for event in docker.events if event[0] == "image"] == [("image", "full")]
    [seen] = template.seen
    assert seen["before"]["imagetag"] == "full"
    assert seen["before"]["gui"] is None and seen["before"]["desktop"] is None


@pytest.mark.usefixtures("asked")
async def test_a_profile_without_an_image_and_no_image_argument_is_refused_naming_list_images(
    docker: _StubDocker, template: _StubTemplate, profiles: _StubProfiles
) -> None:
    profiles.available["web"] = SimpleNamespace(name="web", image_tag=None)

    with pytest.raises(ValueError) as error:
        await RealExegolAdapter().create_container(_spec(image=None, profile="web"))

    assert str(error.value) == IMAGE_REQUIRED_MESSAGE
    assert "list_images" in str(error.value)
    assert template.calls == []
    assert not any(event[0] in {"image", "create"} for event in docker.events)
    assert _torn_down()


@pytest.mark.usefixtures("asked")
@pytest.mark.parametrize(
    "error",
    [
        ValueError("Container profile 'web' cannot be used: not found"),
        RuntimeError("Exegol container profiles need a Pro, Team or Enterprise licence on this host"),
    ],
    ids=["cannot-be-used", "licence"],
)
async def test_a_loader_error_propagates_unchanged_and_the_tiers_are_still_torn_down(
    patched: Any,
    docker: _StubDocker,
    template: _StubTemplate,
    profiles: _StubProfiles,
    resolver: list[tuple[str, Any]],
    error: Exception,
) -> None:
    previous = _tier(patched)
    profiles.load_error = error

    with pytest.raises(type(error)) as raised:
        await RealExegolAdapter().create_container(_spec(image=None, profile="web"))

    assert raised.value is error
    assert profiles.applied == []
    assert template.calls == []
    assert ("loadProfile", None) in resolver
    assert ("loadProfileUserConfig", None) in resolver
    assert _tier(patched) == previous


@pytest.mark.usefixtures("asked")
async def test_a_refused_profile_host_path_fails_the_create_and_the_tiers_are_torn_down(
    patched: Any, docker: _StubDocker, template: _StubTemplate, profiles: _StubProfiles
) -> None:
    previous = _tier(patched)
    profiles.apply_error = RuntimeError("The profile volume path '/' is refused")

    with pytest.raises(RuntimeError, match="is refused"):
        await RealExegolAdapter().create_container(_spec(image=None, profile="web"))

    assert template.calls == []
    assert _torn_down()
    assert _tier(patched) == previous


@pytest.mark.usefixtures("asked")
async def test_a_no_profile_create_right_after_a_profile_create_sees_no_profile_tier(
    docker: _StubDocker, template: _StubTemplate, profiles: _StubProfiles
) -> None:
    adapter = RealExegolAdapter()

    await adapter.create_container(_spec(name="one", image=None, profile="web"))
    await adapter.create_container(_spec(name="two"))

    by_name = {seen["name"]: seen for seen in template.seen}
    assert by_name["one"]["profile_tier"] is not None
    assert by_name["two"]["profile_tier"] is None
    assert by_name["two"]["before"]["gui"] is False
    assert by_name["two"]["before"]["desktop"] is False
    assert [selection for selection, _ in profiles.loads] == ["web"]
    assert _torn_down()


@pytest.mark.usefixtures("asked")
async def test_two_creates_with_the_same_profile_give_two_independent_containers(
    docker: _StubDocker, template: _StubTemplate, profiles: _StubProfiles
) -> None:
    adapter = RealExegolAdapter()

    await adapter.create_container(_spec(name="one", image=None, profile="web"))
    await adapter.create_container(_spec(name="two", image=None, profile="web"))

    assert [created.name for created in docker.created] == ["one", "two"]
    assert [selection for selection, _ in profiles.loads] == ["web", "web"]
    assert len(profiles.applied) == 2
    assert _torn_down()


@pytest.mark.usefixtures("asked", "resolver")
async def test_an_acknowledge_during_a_create_is_answered_and_confirm_stays_refused(
    patched: Any, docker: _StubDocker, template: _StubTemplate
) -> None:
    from exegol.console.ExegolPrompt import ExegolRich

    outcomes: list[tuple[str, str]] = []

    async def during() -> None:
        for kind, prompt, message in (
            ("acknowledge", ExegolRich.Acknowledge, _VPN_DNS_PROMPT),
            ("confirm", ExegolRich.Confirm, "Do you want to continue?"),
        ):
            try:
                await prompt(message)
            except HeadlessPromptError:
                outcomes.append((kind, "refused"))
            else:
                outcomes.append((kind, "answered"))

    template.during = during

    await RealExegolAdapter().create_container(_spec())

    assert outcomes == [("acknowledge", "answered"), ("confirm", "refused")]
    # The policy is create-scoped: once the create is over, an Acknowledge is refused again (D-09).
    with pytest.raises(HeadlessPromptError):
        await ExegolRich.Acknowledge(_VPN_DNS_PROMPT)


def test_the_creation_policy_acknowledges_and_answers_nothing_else() -> None:
    policy = lifecycle_sdk.creation_policy()

    assert policy("acknowledge", _VPN_DNS_PROMPT) is True
    assert policy("confirm", "Do you want to continue?") is None
    assert policy("confirm", _VPN_DNS_PROMPT) is None


def test_exegol_still_acknowledges_the_vpn_dns_warning_during_a_create() -> None:
    import inspect

    from exegol.model import ContainerConfig

    assert f'ExegolRich.Acknowledge("{_VPN_DNS_PROMPT}")' in inspect.getsource(ContainerConfig)


def test_cli_values_with_a_profile_leave_gui_desktop_and_a_missing_image_unset() -> None:
    assert lifecycle_sdk._cli_values(CreateSpec(name="a", image=None, profile="web")) == {}
    assert lifecycle_sdk._cli_values(CreateSpec(name="a", image="free", profile="web", ports=("1:1",))) == {
        "imagetag": "free",
        "ports": ["1:1"],
    }


# --- The fake mirrors the profile create (04-08 Task 1) -----------------------------------------------------------


async def test_the_fake_creates_from_a_profile_with_its_mapped_image() -> None:
    fake = FakeExegolAdapter(profiles={"web": "free"})

    info = await fake.create_container(CreateSpec(name="mcp-1", image=None, profile="web"))

    assert info.image_name == "free"
    assert fake.created_specs[-1].profile == "web"
    assert fake.created_specs[-1].image is None


async def test_the_fake_prefers_the_image_given_on_the_call() -> None:
    fake = FakeExegolAdapter(profiles={"web": "nightly"})

    info = await fake.create_container(CreateSpec(name="mcp-1", image="free", profile="web"))

    assert info.image_name == "free"


async def test_the_fake_refuses_an_unknown_profile_as_exegol_cannot_use_it() -> None:
    fake = FakeExegolAdapter(profiles={"web": "free"})

    with pytest.raises(ValueError, match="cannot be used"):
        await fake.create_container(CreateSpec(name="mcp-1", image=None, profile="ops"))

    assert fake.created_specs == []


async def test_the_fake_without_a_pro_licence_refuses_every_profile() -> None:
    fake = FakeExegolAdapter(profiles={"web": "free"}, profile_licence=False)
    before = list(fake.containers)

    with pytest.raises(RuntimeError, match="Pro"):
        await fake.create_container(CreateSpec(name="mcp-1", image=None, profile="web"))

    assert fake.containers == before


async def test_the_fake_refuses_a_profile_without_an_image_naming_list_images() -> None:
    fake = FakeExegolAdapter(profiles={"web": None})

    with pytest.raises(ValueError) as error:
        await fake.create_container(CreateSpec(name="mcp-1", image=None, profile="web"))

    assert str(error.value) == IMAGE_REQUIRED_MESSAGE


# --- cli_tier and the stdinCanAnswer patch (Task 1) ---------------------------------------------------------------


def test_cli_tier_sets_values_and_restores_the_recorded_ones(patched: Any) -> None:
    patched.set_parameter("hostname", "earlier")

    with sdk_headless.cli_tier({"hostname": "box", "gui": False}):
        assert patched.hostname == "box"
        assert patched.gui is False

    assert patched.hostname == "earlier"
    assert patched.gui is None
    assert patched.accept_eula is True


def test_cli_tier_restores_on_an_exception(patched: Any) -> None:
    with pytest.raises(KeyError), sdk_headless.cli_tier({"imagetag": "free"}):
        raise KeyError("boom")

    assert patched.imagetag is None


def test_cli_tier_refuses_a_dest_it_does_not_know(patched: Any) -> None:
    with pytest.raises(ValueError, match="privileged_mode"), sdk_headless.cli_tier({"privileged_mode": True}):
        pass  # pragma: no cover - never entered


def test_cli_tier_requires_the_headless_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdk_headless, "_parameters", None)

    with pytest.raises(RuntimeError, match="patch_exegol_sdk"), sdk_headless.cli_tier({"imagetag": "free"}):
        pass  # pragma: no cover - never entered


@pytest.mark.usefixtures("patched")
def test_the_profile_ask_helper_can_never_prompt_after_the_patch() -> None:
    import exegol.profile.ProfileAskUser as ask_module

    assert ask_module.stdinCanAnswer() is False


def test_cli_values_maps_only_what_the_spec_sets() -> None:
    assert lifecycle_sdk._cli_values(CreateSpec(name="a", image="free")) == {
        "imagetag": "free",
        "gui": False,
        "desktop": False,
    }
    assert lifecycle_sdk._cli_values(
        CreateSpec(name="a", image="free", workspace="/w", network_mode="host", ports=("1:1",), env=("A=1",))
    ) == {
        "imagetag": "free",
        "gui": False,
        "desktop": False,
        "workspace_path": "/w",
        "network": "host",
        "ports": ["1:1"],
        "envs": ["A=1"],
    }


# --- Label read-back from the docker handle (Task 2, LIFE-06) ------------------------------------------------------


def _labeled(labels: dict[str, str] | None, *, handle: bool = True, name: str = "lab") -> Any:
    """An ExegolContainer stub whose docker-py handle carries the given labels; its config has no label getter."""
    attributes: dict[str, Any] = {
        "name": name,
        "config": _Config([]),
        "image": SimpleNamespace(getName=lambda: "free", getImageVersion=lambda: "6.0.0"),
        "getRawStatus": lambda: "running",
        "isRunning": lambda: True,
    }
    if handle:
        attributes["_ExegolContainer__container"] = SimpleNamespace(labels=labels)
    return SimpleNamespace(**attributes)


def test_the_new_fields_default_to_unmanaged() -> None:
    from exegol_mcp.adapter.port import ContainerState
    from exegol_mcp.models.container import ContainerInfo

    state = ContainerState(running=True)
    assert state.managed is False and state.run_id is None
    fields = ContainerInfo.model_fields
    assert fields["managed"].default is False
    assert fields["run_id"].default is None
    assert fields["this_run"].default is False
    assert all(fields[key].description for key in ("managed", "run_id", "this_run"))


def test_an_exact_true_managed_label_is_read_from_the_docker_handle() -> None:
    from exegol_mcp.adapter import real

    container = _labeled({MANAGED_LABEL: "true", RUN_ID_LABEL: "abc"})
    assert not hasattr(container.config, "getLabels")  # the config is never consulted for labels

    info = real._container_info(container)

    assert (info.managed, info.run_id, info.this_run) == (True, "abc", False)


@pytest.mark.parametrize(
    "labels",
    [
        {MANAGED_LABEL: "True", RUN_ID_LABEL: "abc"},
        {MANAGED_LABEL: "1", RUN_ID_LABEL: "abc"},
        {MANAGED_LABEL: "false", RUN_ID_LABEL: "abc"},
        {MANAGED_LABEL: " true", RUN_ID_LABEL: "abc"},
        {RUN_ID_LABEL: "abc"},
        {},
    ],
    ids=["wrong-case", "numeric", "false", "padded", "missing-managed", "no-labels"],
)
def test_anything_but_an_exact_true_is_unmanaged(labels: dict[str, str]) -> None:
    from exegol_mcp.adapter import real

    info = real._container_info(_labeled(labels))

    assert (info.managed, info.run_id) == (False, None)


def test_container_labels_is_empty_without_a_handle_or_labels() -> None:
    assert lifecycle_sdk.container_labels(_labeled(None, handle=False)) == {}
    assert lifecycle_sdk.container_labels(_labeled(None)) == {}
    assert lifecycle_sdk.container_labels(_labeled({"a": "b"})) == {"a": "b"}


def test_a_container_without_a_handle_still_maps_as_unmanaged() -> None:
    from exegol_mcp.adapter import real

    info = real._container_info(_labeled(None, handle=False))

    assert (info.managed, info.run_id) == (False, None)


async def test_container_state_carries_the_managed_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from exegol_mcp.adapter import real
    from exegol_mcp.adapter.port import ContainerState

    containers = {
        "lab": _labeled({MANAGED_LABEL: "true", RUN_ID_LABEL: "abc"}, name="lab"),
        "hand": _labeled({MANAGED_LABEL: "True"}, name="hand"),
    }

    def get_container(tag: str) -> Any:
        if tag in containers:
            return containers[tag]
        raise ObjectNotFound

    monkeypatch.setattr(real, "DockerUtils", lambda: SimpleNamespace(getContainer=get_container))
    adapter = RealExegolAdapter()

    assert await adapter.get_container_state("lab") == ContainerState(running=True, managed=True, run_id="abc")
    assert await adapter.get_container_state("hand") == ContainerState(running=True)
    assert await adapter.get_container_state("ghost") is None


@pytest.mark.usefixtures("asked", "resolver")
async def test_the_created_container_reads_back_as_managed_with_its_run_id(
    docker: _StubDocker, template: _StubTemplate
) -> None:
    info = await RealExegolAdapter().create_container(_spec())

    assert (info.managed, info.run_id, info.this_run) == (True, "abc123def456", False)


# --- The removal choreography (04-05 Task 1, LIFE-03) --------------------------------------------------------------


class _DockerHandle:
    """The docker-py Container handle: stop(**kwargs) only, as docker-py declares it, recording thread and grace."""

    def __init__(self) -> None:
        self.labels = {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "abc"}
        self.stops: list[tuple[int, dict[str, Any]]] = []

    def stop(self, **kwargs: Any) -> None:
        self.stops.append((threading.get_ident(), kwargs))


class _RemovableConfig:
    def __init__(self, volume_path: str) -> None:
        self.volume_path = volume_path
        # A host-mounted workspace (Exegol's __workspace_custom_path), None without one (IN-10).
        self.custom_workspace: str | None = None

    def getPrivateVolumePath(self) -> str:
        return self.volume_path

    def isWorkspaceCustom(self) -> bool:
        return bool(self.custom_workspace)

    def getHostWorkspacePath(self) -> str:
        # Exegol's own literal for a container with neither a custom nor a private workspace.
        return self.custom_workspace or "not found :("

    def isSentinelEnable(self) -> bool:
        return False

    def getSentinelPath(self) -> Path | None:
        return None


class _Removable:
    """An ExegolContainer whose remove() records its keyword arguments and can suspend, raise or ask a prompt."""

    def __init__(self, *, running: bool = True, volume_path: str = "/var/lib/exegol/workspaces/lab") -> None:
        self.name = "lab"
        self.running = running
        self.config = _RemovableConfig(volume_path)
        self._ExegolContainer__container = _DockerHandle()
        self.backups = [("bak-id-1", "lab-bak1"), ("bak-id-2", "lab-bak2")]
        self.remove_calls: list[dict[str, Any]] = []
        self.remove_started = anyio.Event()
        self.remove_seconds = 0.0
        self.remove_error: Exception | None = None
        # Seconds remove() blocks synchronously after its last await: the stub's stand-in for a removal whose last step
        # does not yield before the deadline is read (Exegol's tail now runs in one worker hop, patch step 13; WR-08).
        self.sync_block_seconds = 0.0
        self.removed = False
        # Prompts remove() asks through ExegolRich.Confirm, and the answers it got (Task 2).
        self.prompts: list[str] = []
        self.answers: list[Any] = []

    def isRunning(self) -> bool:
        return self.running

    def getExistingBackupContainers(self) -> list[tuple[str, str]]:
        return list(self.backups)

    async def remove(self, **kwargs: Any) -> None:
        from exegol.console.ExegolPrompt import ExegolRich

        self.remove_calls.append(kwargs)
        self.remove_started.set()
        for message in self.prompts:
            self.answers.append(await ExegolRich.Confirm(message, default=False))
        await anyio.sleep(self.remove_seconds)
        if self.remove_error is not None:
            raise self.remove_error
        if self.sync_block_seconds:
            # Synchronous on purpose: a removal that finishes without a checkpoint after its deadline has passed, which
            # the outcome logic must still report as done.
            time.sleep(self.sync_block_seconds)  # noqa: ASYNC251 - reproduces a tail with no checkpoint
        self.removed = True


@pytest.fixture
def removable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Removable:
    from exegol_mcp.adapter import real

    # A real private workspace directory: the stub remove() never deletes it, so it is still on the host after.
    workspace = tmp_path / "workspaces" / "lab"
    workspace.mkdir(parents=True)
    container = _Removable(volume_path=str(workspace))

    def get_container(tag: str) -> Any:
        if tag == container.name:
            return container
        raise ObjectNotFound

    monkeypatch.setattr(real, "DockerUtils", lambda: SimpleNamespace(getContainer=get_container))
    return container


@pytest.mark.parametrize("running", [True, False])
async def test_remove_never_stops_the_container_itself_and_removes_with_the_backups(
    removable: _Removable, running: bool
) -> None:
    """CR-01: no pre-stop. Exegol's own remove() runs its volume step while the container runs, then stops it."""
    from exegol_mcp.adapter.port import RemovalOutcome

    removable.running = running

    outcome = await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=30.0)

    # WR-06: host state, not the request. The stub remove() leaves the workspace in place, and without Sentinel
    # logging there is no audit history to keep.
    assert outcome == RemovalOutcome(workspace_kept=True, logs_kept=False, host_workspace=False)
    # The fake remove() is the only stop path: the adapter never stops the docker handle itself.
    assert removable._ExegolContainer__container.stops == []
    [kwargs] = removable.remove_calls
    assert kwargs["backup_history"] == ["bak-id-1", "bak-id-2"]
    # Never container_only: it would skip Exegol's network and backup cleanup.
    assert kwargs.get("container_only", False) is False
    assert removable.removed is True


async def test_a_host_mounted_workspace_is_always_reported_as_kept(removable: _Removable, tmp_path: Path) -> None:
    from exegol_mcp.adapter.port import RemovalOutcome

    host_dir = tmp_path / "engagement"
    host_dir.mkdir()
    removable.config.volume_path = ""
    removable.config.custom_workspace = str(host_dir)

    outcome = await RealExegolAdapter().remove_container("lab", keep_workspace=False, keep_logs=False, timeout_s=30.0)

    assert outcome == RemovalOutcome(workspace_kept=True, logs_kept=False, host_workspace=True)


async def test_a_host_mounted_workspace_that_is_gone_is_reported_not_kept(
    removable: _Removable, tmp_path: Path
) -> None:
    from exegol_mcp.adapter.port import RemovalOutcome

    removable.config.volume_path = ""
    removable.config.custom_workspace = str(tmp_path / "moved-away")

    outcome = await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=False, timeout_s=30.0)

    assert outcome == RemovalOutcome(workspace_kept=False, logs_kept=False, host_workspace=True)


async def test_a_container_without_a_workspace_reports_nothing_kept(removable: _Removable) -> None:
    """IN-10: Exegol disabled the workspace (no /workspace mount): no private path and no custom one."""
    from exegol_mcp.adapter.port import RemovalOutcome

    removable.config.volume_path = ""

    outcome = await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=30.0)

    assert outcome == RemovalOutcome(workspace_kept=False, logs_kept=False, host_workspace=False)


def test_exegols_own_getters_report_no_workspace_for_a_workspace_disabled_container(tmp_path: Path) -> None:
    """IN-10 over Exegol's real ContainerConfig getters, built without its docker-reading __init__."""
    from exegol.model.ContainerConfig import ContainerConfig

    config = object.__new__(ContainerConfig)
    config._ContainerConfig__workspace_custom_path = None  # type: ignore[attr-defined]
    config._ContainerConfig__workspace_dedicated_path = None  # type: ignore[attr-defined]
    config._ContainerConfig__sentinel_path = None  # type: ignore[attr-defined]
    container = SimpleNamespace(config=config)

    data = lifecycle_sdk._host_data_paths(container)

    assert (data.volume_path, data.host_workspace, data.sentinel_dir) == ("", None, None)
    assert lifecycle_sdk._host_data_remaining(data) == (False, False)

    # A host-mounted workspace: Exegol reads it through getHostWorkspacePath(), and it is kept while it exists.
    host_dir = tmp_path / "engagement"
    host_dir.mkdir()
    config._ContainerConfig__workspace_custom_path = str(host_dir)  # type: ignore[attr-defined]

    data = lifecycle_sdk._host_data_paths(container)

    assert (data.volume_path, data.host_workspace) == ("", str(host_dir))
    assert lifecycle_sdk._host_data_remaining(data)[0] is True


async def test_a_cancelled_removal_still_completes_then_the_cancellation_propagates(removable: _Removable) -> None:
    removable.remove_seconds = 0.1
    seen: dict[str, Any] = {}

    async def remove() -> None:
        with anyio.CancelScope() as scope:
            seen["scope"] = scope
            await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=30.0)
            seen["returned"] = True
        seen["cancelled"] = scope.cancelled_caught

    async with anyio.create_task_group() as tg:
        tg.start_soon(remove)
        await removable.remove_started.wait()
        seen["scope"].cancel()

    assert removable.removed is True
    assert "returned" not in seen
    assert seen["cancelled"] is True


async def test_a_docker_timeout_on_removal_propagates_as_runtime_error(removable: _Removable) -> None:
    # What logger.critical raises under the headless patch when docker times out removing the container.
    removable.remove_error = RuntimeError("Received a timeout error from docker while removing the container.")

    with pytest.raises(RuntimeError, match="timeout error from docker"):
        await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=30.0)


async def test_a_container_removed_concurrently_is_not_found_naming_list_containers(removable: _Removable) -> None:
    """WR-01: a docker NotFound mid-removal (another process removed it) becomes ValueError, never a raw error."""
    import docker.errors

    removable.remove_error = docker.errors.NotFound("gone")

    with pytest.raises(ValueError, match="not found") as caught:
        await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=30.0)
    assert "list_containers" in str(caught.value)
    assert "'lab'" in str(caught.value)


@pytest.mark.parametrize("kind", ["api_error", "read_timeout"])
async def test_a_docker_error_during_removal_is_a_runtime_error_naming_the_container(
    removable: _Removable, kind: str
) -> None:
    import requests.exceptions

    error: Exception
    if kind == "api_error":
        error = DockerAPIError("boom", explanation="conflict")
    else:
        error = requests.exceptions.ReadTimeout()
    removable.remove_error = error

    with pytest.raises(RuntimeError) as caught:
        await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=30.0)
    message = str(caught.value)
    assert "'lab'" in message
    assert not isinstance(caught.value, requests.exceptions.RequestException)
    if kind == "api_error":
        assert "conflict" in message
    else:
        assert "ReadTimeout" in message


@pytest.mark.parametrize("managed_value", ["True", "1", " true", None])
async def test_the_adapter_refuses_a_container_whose_own_label_is_not_exactly_managed(
    removable: _Removable, managed_value: str | None
) -> None:
    """WR-02: the gate reads the resolved container's own docker handle, right before Exegol's remove()."""
    labels = {RUN_ID_LABEL: "abc"}
    if managed_value is not None:
        labels[MANAGED_LABEL] = managed_value
    removable._ExegolContainer__container.labels = labels

    with pytest.raises(ValueError) as caught:
        await RealExegolAdapter().remove_container(
            "lab", keep_workspace=True, keep_logs=True, timeout_s=30.0, require_managed=True
        )

    assert str(caught.value) == lifecycle_service.UNMANAGED_REMOVAL_MESSAGE.format(name="lab")
    assert removable.remove_calls == []
    assert removable.removed is False


async def test_the_adapter_removes_an_unmanaged_container_when_the_caller_does_not_require_managed(
    removable: _Removable,
) -> None:
    removable._ExegolContainer__container.labels = {MANAGED_LABEL: "True"}

    await RealExegolAdapter().remove_container(
        "lab", keep_workspace=True, keep_logs=True, timeout_s=30.0, require_managed=False
    )

    assert removable.removed is True


@pytest.mark.parametrize("run_id", ["ABC", "abc ", "other", None])
async def test_the_adapter_refuses_a_run_id_that_is_not_exactly_the_expected_one(
    removable: _Removable, run_id: str | None
) -> None:
    """Edge LIFE-04 adjacency: exact string equality only, never case-folded or stripped."""
    labels = {MANAGED_LABEL: MANAGED_LABEL_VALUE}
    if run_id is not None:
        labels[RUN_ID_LABEL] = run_id
    removable._ExegolContainer__container.labels = labels

    with pytest.raises(ValueError) as caught:
        await RealExegolAdapter().remove_container(
            "lab", keep_workspace=True, keep_logs=True, timeout_s=30.0, expected_run_id="abc"
        )

    assert "is no longer one this server run created" in str(caught.value)
    assert "'lab'" in str(caught.value)
    assert removable.remove_calls == []


async def test_the_adapter_removes_a_container_carrying_the_expected_run_id(removable: _Removable) -> None:
    await RealExegolAdapter().remove_container(
        "lab", keep_workspace=True, keep_logs=True, timeout_s=30.0, expected_run_id="abc"
    )

    assert removable.removed is True


async def test_removing_an_unknown_container_raises_not_found(removable: _Removable) -> None:
    with pytest.raises(ValueError, match="not found"):
        await RealExegolAdapter().remove_container("ghost", keep_workspace=True, keep_logs=True, timeout_s=30.0)
    assert removable.remove_calls == []


# --- The scoped removal prompt policy (04-05 Task 2, D-12, Pitfall 8) ---------------------------------------------

_SENTINEL_PROMPT = "Do you want to remove your Sentinel logging history?"
_WORKSPACE_PROMPT = "Workspace [magenta]/x[/magenta] is not empty, do you want to delete it?"
_UNREADABLE_PROMPT = (
    "Insufficient permission to view workspace files [magenta]/x[/magenta], do you want to delete your container's "
    "workspace?"
)


@pytest.mark.usefixtures("patched")
async def test_with_no_policy_bound_every_prompt_is_still_refused() -> None:
    from exegol.console.ExegolPrompt import ExegolRich

    with pytest.raises(sdk_headless.HeadlessPromptError, match="Sentinel logging history"):
        await ExegolRich.Confirm(_SENTINEL_PROMPT, default=False)
    with pytest.raises(sdk_headless.HeadlessPromptError, match="License has expired"):
        await ExegolRich.Acknowledge("License has expired.")


@pytest.mark.usefixtures("patched")
async def test_a_bound_removal_policy_answers_only_the_removal_prompts() -> None:
    from exegol.console.ExegolPrompt import ExegolRich

    with sdk_headless.answer_prompts(sdk_headless.removal_policy(keep_workspace=False, keep_logs=True)):
        assert await ExegolRich.Confirm(_SENTINEL_PROMPT, default=False) is False
        assert await ExegolRich.Confirm(_WORKSPACE_PROMPT, default=False) is True
        assert await ExegolRich.Confirm(_UNREADABLE_PROMPT, default=False) is True
        with pytest.raises(sdk_headless.HeadlessPromptError, match="update the image"):
            await ExegolRich.Confirm("Do you want to update the image?", default=False)
        with pytest.raises(sdk_headless.HeadlessPromptError, match="License has expired"):
            await ExegolRich.Acknowledge("License has expired.")
        with pytest.raises(sdk_headless.HeadlessPromptError, match="Enter the path"):
            await ExegolRich.Ask("Enter the path of your workspace")

    with pytest.raises(sdk_headless.HeadlessPromptError, match="Sentinel logging history"):
        await ExegolRich.Confirm(_SENTINEL_PROMPT, default=False)


@pytest.mark.usefixtures("patched")
async def test_a_bound_policy_can_acknowledge_and_is_unbound_on_an_exception() -> None:
    from exegol.console.ExegolPrompt import ExegolRich

    def acknowledge_all(kind: str, message: str) -> bool | None:
        return True if kind == "acknowledge" else None

    with pytest.raises(KeyError), sdk_headless.answer_prompts(acknowledge_all):
        assert await ExegolRich.Acknowledge("Noted.") is None
        raise KeyError("boom")

    with pytest.raises(sdk_headless.HeadlessPromptError):
        await ExegolRich.Acknowledge("Noted.")


@pytest.mark.usefixtures("patched")
async def test_two_concurrent_tasks_each_see_only_their_own_policy() -> None:
    from exegol.console.ExegolPrompt import ExegolRich

    answers: dict[str, list[bool]] = {}

    async def answer(label: str, keep: bool) -> None:
        with sdk_headless.answer_prompts(sdk_headless.removal_policy(keep_workspace=keep, keep_logs=keep)):
            seen = [await ExegolRich.Confirm(_WORKSPACE_PROMPT, default=False)]
            await anyio.sleep(0.02)
            seen.append(await ExegolRich.Confirm(_SENTINEL_PROMPT, default=False))
        answers[label] = seen

    async with anyio.create_task_group() as tg:
        tg.start_soon(answer, "keep", True)
        tg.start_soon(answer, "delete", False)

    assert answers == {"keep": [False, False], "delete": [True, True]}


def test_the_removal_policy_matches_the_known_prompt_texts_only() -> None:
    policy = sdk_headless.removal_policy(keep_workspace=True, keep_logs=False)

    assert policy("confirm", _SENTINEL_PROMPT) is True
    assert policy("confirm", _WORKSPACE_PROMPT) is False
    assert policy("confirm", _UNREADABLE_PROMPT) is False
    assert policy("confirm", "Do you want to update the image?") is None
    assert policy("acknowledge", _SENTINEL_PROMPT) is None


def test_exegol_still_asks_the_removal_prompts_the_policy_matches() -> None:
    """Drift guard: an Exegol rewording must fail CI, not silently fall back to a refused removal (Pitfall 8)."""
    import inspect

    from exegol.model.ExegolContainer import ExegolContainer

    source = inspect.getsource(ExegolContainer._ExegolContainer__removeVolume)  # type: ignore[attr-defined]

    assert "Sentinel logging history" in source
    assert "do you want to delete" in source
    assert "workspace" in source


@pytest.mark.usefixtures("patched")
@pytest.mark.parametrize(
    ("keep_workspace", "keep_logs", "expected"),
    [(True, True, [False, False]), (True, False, [False, True]), (False, True, [True, False])],
)
async def test_remove_answers_exegols_prompts_from_the_keep_options(
    removable: _Removable, keep_workspace: bool, keep_logs: bool, expected: list[bool]
) -> None:
    removable.prompts = [_WORKSPACE_PROMPT, _SENTINEL_PROMPT]

    await RealExegolAdapter().remove_container("lab", keep_workspace=keep_workspace, keep_logs=keep_logs, timeout_s=30.0)

    assert removable.answers == expected
    assert removable.removed is True


def test_the_removal_patch_points_are_self_checked() -> None:
    from exegol_mcp.adapter import selfcheck

    for point in (
        ("exegol.model.ExegolContainer", "ExegolContainer.remove"),
        # The method removal_offload mirrors for Exegol's in-container fallback (CR-01).
        ("exegol.model.ExegolContainer", "ExegolContainer.exec"),
        ("exegol.model.ExegolContainer", "ExegolContainer.getExistingBackupContainers"),
        ("exegol.model.ExegolContainer", "ExegolContainer._ExegolContainer__removeVolume"),
        ("exegol.model.ContainerConfig", "ContainerConfig.getPrivateVolumePath"),
        # The getters the outcome reads for a host-mounted or a disabled workspace (IN-10).
        ("exegol.model.ContainerConfig", "ContainerConfig.isWorkspaceCustom"),
        ("exegol.model.ContainerConfig", "ContainerConfig.getHostWorkspacePath"),
    ):
        assert point in selfcheck._CALLABLE_POINTS, point
    selfcheck.assert_patch_points()


# --- CR-01: the removal is bounded and keeps the loop free (04-13) -------------------------------------------------
#
# These tests drive Exegol's REAL ExegolContainer.remove() and __removeVolume over a stub docker handle, with the
# host shutil.rmtree refusing the workspace the way root-owned files make it refuse on a Linux docker-group host.
# Exegol then deletes the workspace from inside the container, which is the path that froze the server.


class _RemovalHandle:
    """The docker-py Container behind a real ExegolContainer: every call recorded with its thread, in order.

    exec_run blocks on exec_release for up to exec_block_seconds, like a long in-container `rm -rf`. Both
    timestamps use time.perf_counter so a test can tell whether a concurrent request answered during the block.
    """

    def __init__(self, *, exec_block_seconds: float = 0.0, status: str = "running") -> None:
        self.name = "lab"
        self.id = "lab-id"
        self.labels = {MANAGED_LABEL: MANAGED_LABEL_VALUE, RUN_ID_LABEL: "abc"}
        self.attrs: dict[str, Any] = {"State": {"Status": status}}
        self.exec_block_seconds = exec_block_seconds
        self.exec_release = threading.Event()
        self.exec_entered = threading.Event()
        self.exec_done = threading.Event()
        # Every logs() read; an image whose startup never prints READY would be read again every second.
        self.log_reads = 0
        self.exec_started_at: float | None = None
        self.exec_finished_at: float | None = None
        self.execs: list[tuple[int, Any, dict[str, Any]]] = []
        self.stops: list[tuple[int, dict[str, Any]]] = []
        self.starts: list[int] = []
        self.removes: list[int] = []
        # The order of the state-changing docker calls: "exec", "stop", "start", "remove".
        self.calls: list[str] = []

    def reload(self) -> None:
        return None

    def exec_run(self, cmd: Any, **kwargs: Any) -> Any:
        self.exec_started_at = time.perf_counter()
        self.execs.append((threading.get_ident(), cmd, kwargs))
        self.exec_entered.set()
        self.exec_release.wait(self.exec_block_seconds)
        self.exec_finished_at = time.perf_counter()
        self.calls.append("exec")
        self.exec_done.set()
        return SimpleNamespace(exit_code=0, output=b"")

    def logs(self, stream: bool = False, follow: bool = False, since: Any = None, until: Any = None) -> Any:
        # One startup line that is never READY, on every read.
        self.log_reads += 1
        return iter([b"[INFO] booting\n"])

    def stop(self, **kwargs: Any) -> None:
        self.stops.append((threading.get_ident(), kwargs))
        self.calls.append("stop")
        self.attrs["State"]["Status"] = "exited"

    def start(self) -> None:
        self.starts.append(threading.get_ident())
        self.calls.append("start")
        self.attrs["State"]["Status"] = "running"

    def remove(self, **kwargs: Any) -> None:
        self.removes.append(threading.get_ident())
        self.calls.append("remove")


class _RemovalConfig:
    """The ContainerConfig getters Exegol's remove() and __removeVolume read."""

    legacy_entrypoint = False

    def __init__(self, workspace: Path, sentinel: Path | None = None, volume_path: str | None = None) -> None:
        self.workspace = workspace
        self.sentinel = sentinel
        # A private volume path that is not the local workspace, such as a WSL one ("/wsl/..."); None uses it.
        self.volume_path = volume_path

    def isSentinelEnable(self) -> bool:
        return self.sentinel is not None

    def getSentinelPath(self) -> Path | None:
        return self.sentinel

    def getPrivateVolumePath(self) -> str:
        return str(self.workspace) if self.volume_path is None else self.volume_path

    def getHostWorkspacePath(self) -> str:
        return str(self.workspace)

    def isWorkspaceCustom(self) -> bool:
        return False

    def getNetworks(self) -> list[Any]:
        return []

    def getBackupHistory(self) -> str | None:
        return None


def _removal_container(handle: _RemovalHandle, workspace: Path, *, sentinel: Path | None = None) -> Any:
    """A real ExegolContainer over the stub handle, built without its docker-reading __init__."""
    from exegol.model.ExegolContainer import ExegolContainer

    container = object.__new__(ExegolContainer)
    container._ExegolContainer__container = handle  # type: ignore[attr-defined]
    container.config = _RemovalConfig(workspace, sentinel)  # type: ignore[assignment]
    container.name = handle.name
    return container


class _RemovalScene:
    def __init__(self, handle: _RemovalHandle, container: Any, workspace: Path) -> None:
        self.handle = handle
        self.container = container
        self.workspace = workspace
        self.rmtree_calls: list[str] = []


@pytest.fixture
def scene(patched: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _RemovalScene:
    """Exegol's real removal over a non-empty private workspace the host rmtree refuses once (CR-01)."""
    import shutil

    import exegol.model.ExegolContainer as container_module

    from exegol_mcp.adapter import real

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.txt").write_text("owned by root in the container")
    handle = _RemovalHandle(exec_block_seconds=BLOCK_SECONDS)
    container = _removal_container(handle, workspace)
    current = _RemovalScene(handle, container, workspace)

    def rmtree(path: str, *args: Any, **kwargs: Any) -> None:
        # The first host delete is refused (root-owned files); the one after the in-container delete succeeds.
        current.rmtree_calls.append(path)
        if len(current.rmtree_calls) == 1:
            raise PermissionError(13, "Permission denied", path)
        shutil.rmtree(path, *args, **kwargs)

    def get_container(tag: str) -> Any:
        if tag == handle.name:
            return container
        raise ObjectNotFound

    # Only the names Exegol's model module imported are replaced, never the global shutil or DataCache.
    monkeypatch.setattr(container_module, "shutil", SimpleNamespace(rmtree=rmtree))
    monkeypatch.setattr(container_module, "DataCache", lambda: SimpleNamespace(remove_container_cache=lambda n: None))
    monkeypatch.setattr(real, "DockerUtils", lambda: SimpleNamespace(getContainer=get_container))
    return current


async def _wait_for_thread_event(event: threading.Event, timeout_s: float = 5.0) -> None:
    """Poll a threading.Event from the loop (an anyio.Event cannot be set from a worker thread)."""
    with anyio.fail_after(timeout_s):
        while not event.is_set():
            await anyio.sleep(0.01)


async def _remove_through_the_service(timeout_s: float = 30.0) -> Any:
    return await lifecycle_service.remove_container(
        RealExegolAdapter(),
        "lab",
        keep_workspace=False,
        keep_logs=False,
        settings=LifecycleSettings(),
        timeout_s=timeout_s,
    )


async def test_removal_fallback_runs_off_the_loop_while_tools_list_stays_fast(scene: _RemovalScene) -> None:
    handle = scene.handle
    loop_thread = threading.get_ident()
    result: dict[str, Any] = {}

    async def remove() -> None:
        result["removed"] = await _remove_through_the_service()

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(remove)
            await _wait_for_thread_event(handle.exec_entered)
            await anyio.sleep(0.05)  # let the in-container delete settle into its blocking wait
            started = time.perf_counter()
            await client.list_tools()
            latency = time.perf_counter() - started
            answered_during_the_delete = handle.exec_finished_at is None
            handle.exec_release.set()
    finally:
        handle.exec_release.set()

    assert latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, latency
    assert answered_during_the_delete
    # Exegol's in-container delete ran in a worker thread, with exactly Exegol's environment.
    [(exec_thread, _payload, exec_kwargs)] = handle.execs
    assert exec_thread != loop_thread
    assert exec_kwargs["environment"]["CMD"] == "rm -rf /workspace"
    # Exegol's own stop ran once, off the loop, with its 2 s grace, after the delete; no restart, then the remove.
    [(stop_thread, stop_kwargs)] = handle.stops
    assert stop_thread != loop_thread
    assert stop_kwargs == {"timeout": 2}
    assert handle.calls == ["exec", "stop", "remove"]
    assert handle.starts == []
    assert not scene.workspace.exists()
    assert result["removed"].name == "lab"


async def test_removal_fallback_on_the_loop_is_detected(
    scene: _RemovalScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inversion: without removal_offload, Exegol's own exec blocks the loop and tools/list waits for it."""
    from contextlib import contextmanager

    @contextmanager
    def no_offload(container: Any) -> Any:
        yield

    monkeypatch.setattr(sdk_headless, "removal_offload", no_offload)
    handle = scene.handle
    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(_remove_through_the_service)
            await _wait_for_thread_event(handle.exec_entered)
            await client.list_tools()
            assert handle.exec_started_at is not None
            since_exec_start = time.perf_counter() - handle.exec_started_at
            stalled_until_the_delete_finished = handle.exec_finished_at is not None
    finally:
        handle.exec_release.set()

    assert stalled_until_the_delete_finished
    assert since_exec_start >= MAX_CONCURRENT_LATENCY, since_exec_start


# --- CR-01: every fallback path is bounded, and the mirrored Exegol code is pinned (04-13 Task 2) -----------------


async def test_a_stopped_container_restarted_by_the_fallback_skips_the_ready_wait(scene: _RemovalScene) -> None:
    handle = scene.handle
    handle.attrs["State"]["Status"] = "exited"
    handle.exec_release.set()
    loop_thread = threading.get_ident()

    started = time.perf_counter()
    removed = await _remove_through_the_service(timeout_s=30.0)
    elapsed = time.perf_counter() - started

    # Exegol restarted the container for its in-container delete, but never waited for READY: without the skip the
    # SDK stream would read these never-READY logs every second until the deadline.
    assert handle.starts and len(handle.starts) == 1
    assert handle.log_reads == 0
    [(exec_thread, _payload, exec_kwargs)] = handle.execs
    assert exec_thread != loop_thread
    assert exec_kwargs["environment"]["CMD"] == "rm -rf /workspace"
    assert handle.calls == ["start", "exec", "stop", "remove"]
    assert elapsed < 3.0, elapsed
    assert not scene.workspace.exists()
    assert removed.name == "lab"


async def test_a_removal_past_its_deadline_raises_and_a_retry_finishes(scene: _RemovalScene) -> None:
    handle = scene.handle
    handle.exec_block_seconds = 10.0
    first: dict[str, Any] = {}

    async def remove() -> None:
        started = time.perf_counter()
        try:
            await _remove_through_the_service(timeout_s=1.0)
        except RuntimeError as e:
            first["error"] = str(e)
        first["elapsed"] = time.perf_counter() - started

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(remove)
            await _wait_for_thread_event(handle.exec_entered)
            started = time.perf_counter()
            await client.list_tools()
            latency = time.perf_counter() - started
        # The removal stopped at its deadline: the docker removal never ran and the container is still listed.
        assert handle.removes == []
        assert await RealExegolAdapter().get_container_state("lab") is not None
        assert "exec" not in vars(scene.container) and "stop" not in vars(scene.container)
    finally:
        handle.exec_release.set()
    # The abandoned in-container delete finishes in its worker thread once released; nothing outlives the test.
    await _wait_for_thread_event(handle.exec_done)

    assert "not fully removed within 1 s" in first["error"]
    assert "remove_container" in first["error"]
    assert first["elapsed"] < 1.0 + MAX_CONCURRENT_LATENCY, first["elapsed"]
    assert latency < MAX_CONCURRENT_LATENCY, latency

    # Edge LIFE-03 idempotency: a second call finishes the removal.
    removed = await _remove_through_the_service(timeout_s=30.0)
    assert removed.name == "lab"
    assert len(handle.removes) == 1
    assert not scene.workspace.exists()


async def test_a_cancelled_caller_waits_at_most_for_the_removal_deadline(scene: _RemovalScene) -> None:
    handle = scene.handle
    handle.exec_block_seconds = 10.0
    seen: dict[str, Any] = {}

    async def remove() -> None:
        with anyio.CancelScope() as scope:
            seen["scope"] = scope
            try:
                await _remove_through_the_service(timeout_s=1.0)
            except RuntimeError as e:
                seen["error"] = str(e)
        seen["returned_at"] = time.perf_counter()

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(remove)
            await _wait_for_thread_event(handle.exec_entered)
            cancelled_at = time.perf_counter()
            seen["scope"].cancel()
    finally:
        handle.exec_release.set()
    await _wait_for_thread_event(handle.exec_done)

    # Edge LIFE-03 concurrency: the shield holds the cancel until the removal deadline, never until the 10 s release.
    waited = seen["returned_at"] - cancelled_at
    assert waited < 1.0 + MAX_CONCURRENT_LATENCY, waited
    assert "remove_container" in seen["error"]
    assert handle.removes == []


async def test_the_sentinel_shred_fallback_runs_off_the_loop(
    scene: _RemovalScene, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol.model.ExegolContainer as container_module

    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    scene.container.config.sentinel = sentinel

    async def refused_secure_remove(path: Any) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(container_module, "secure_remove", refused_secure_remove)
    handle = scene.handle
    handle.exec_release.set()
    loop_thread = threading.get_ident()

    await _remove_through_the_service(timeout_s=30.0)

    # The Sentinel shred, then the workspace delete: both through the offloaded exec, in worker threads.
    shreds = [entry for entry in handle.execs if "shred" in entry[2]["environment"]["CMD"]]
    assert len(shreds) == 1
    assert all(thread != loop_thread for thread, _payload, _kwargs in handle.execs)
    assert handle.starts == []
    assert not sentinel.exists()


async def test_the_removal_deadline_is_visible_inside_the_shield(tmp_path: Path) -> None:
    from datetime import datetime

    container = _removal_container(_RemovalHandle(), tmp_path)

    assert sdk_headless._deadline_stop_time() is None
    with anyio.CancelScope(shield=True):
        assert sdk_headless._deadline_stop_time() is None
        with anyio.fail_after(5):
            assert sdk_headless._deadline_stop_time() is not None
            with sdk_headless.removal_offload(container):
                stop_time = sdk_headless._deadline_stop_time()
                assert stop_time is not None
                assert stop_time <= datetime.now()
            await anyio.lowlevel.checkpoint()  # a checkpoint inside the scopes, as a real removal has


def test_the_removal_offload_touches_only_its_container_and_unbinds(tmp_path: Path) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    removed = _removal_container(_RemovalHandle(), tmp_path)
    other = _removal_container(_RemovalHandle(), tmp_path)
    earlier_stop = object()
    removed.stop = earlier_stop

    with sdk_headless.removal_offload(removed):
        assert {"exec", "stop"} <= set(vars(removed))
        assert sdk_headless._removal_in_progress.get() is True
        # Another container object keeps Exegol's own methods.
        assert "exec" not in vars(other) and "stop" not in vars(other)
        assert other.exec.__func__ is ExegolContainer.exec
        assert other.stop.__func__ is ExegolContainer.stop

    assert "exec" not in vars(removed)
    assert removed.stop is earlier_stop  # a previous instance attribute is restored, not deleted
    assert sdk_headless._removal_in_progress.get() is False

    del removed.stop
    with pytest.raises(KeyError), sdk_headless.removal_offload(removed):
        raise KeyError("boom")
    assert "exec" not in vars(removed) and "stop" not in vars(removed)
    assert removed.exec.__func__ is ExegolContainer.exec
    assert sdk_headless._removal_in_progress.get() is False


def test_exegol_still_runs_the_volume_step_before_its_own_stop() -> None:
    """Drift guard: removal_offload relies on Exegol deleting the volume while the container still runs (CR-01)."""
    import inspect

    # Patch step 13 replaces the class attribute, so Exegol's own remove is read from the import-time original.
    assert sdk_headless._SDK_REMOVE is not None
    source = inspect.getsource(sdk_headless._SDK_REMOVE)

    assert "__removeVolume()" in source
    assert "self.stop(timeout=2)" in source
    assert source.index("__removeVolume()") < source.index("self.stop(timeout=2)")


def test_exegol_exec_and_stop_still_match_the_offloaded_mirror() -> None:
    """Drift guard: _offloaded_exec / _offloaded_stop mirror these lines of Exegol; a change must fail CI (CR-01)."""
    import inspect

    from exegol.model.ExegolContainer import ExegolContainer

    exec_source = inspect.getsource(ExegolContainer.exec)
    # Patch step 10 replaces the class attribute, so Exegol's own stop is read from the import-time original.
    assert sdk_headless._SDK_STOP is not None
    stop_source = inspect.getsource(sdk_headless._SDK_STOP)
    volume_source = inspect.getsource(ExegolContainer._ExegolContainer__removeVolume)  # type: ignore[attr-defined]

    assert 'environment={"CMD": str_cmd, "DISABLE_AUTO_UPDATE": "true"}' in exec_source
    assert "formatShellCommand(command, quiet)" in exec_source
    assert "stream=not as_daemon and not quiet" in exec_source
    assert "self.__container.stop(timeout=timeout)" in stop_source
    assert 'await self.exec("rm -rf /workspace", as_daemon=False, quiet=True)' in volume_source
    assert "await self.__start_container()" in volume_source


# --- INT-01: the removal's docker tail runs off the loop, in one worker hop (05.1-03, patch step 13) ---------------
#
# Exegol's remove() ends with the docker container removal, the cache entry, the backup removals and the network
# removals, all synchronous inside an async def. Patch step 13 mirrors it: the volume step and the stop are awaited as
# Exegol's, and the whole tail is ONE run_blocking hop, so a deadline never leaves a container half removed.


class _BlockedDockerRemoval:
    """The stub handle's remove() standing in for a docker removal blocked on the daemon (an instance attribute).

    Blocks until release is set or block_seconds pass, then records the removal on the handle as the default does.
    """

    def __init__(self, handle: _RemovalHandle, block_seconds: float = BLOCK_SECONDS) -> None:
        self.handle = handle
        self.block_seconds = block_seconds
        self.entered = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        handle.remove = self  # type: ignore[method-assign]

    def __call__(self, **kwargs: Any) -> None:
        self.started_at = time.perf_counter()
        self.entered.set()
        self.release.wait(self.block_seconds)
        self.finished_at = time.perf_counter()
        _RemovalHandle.remove(self.handle, **kwargs)
        self.done.set()


async def test_the_docker_removal_runs_off_the_loop_while_tools_list_stays_fast(scene: _RemovalScene) -> None:
    handle = scene.handle
    handle.exec_release.set()  # the volume step's in-container delete returns at once
    removal = _BlockedDockerRemoval(handle)
    loop_thread = threading.get_ident()
    result: dict[str, Any] = {}

    async def remove() -> None:
        result["removed"] = await _remove_through_the_service()

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(remove)
            await _wait_for_thread_event(removal.entered)
            await anyio.sleep(0.05)  # let the removal settle into its blocking wait
            started = time.perf_counter()
            await client.list_tools()
            latency = time.perf_counter() - started
            answered_during_the_removal = removal.finished_at is None
            removal.release.set()
    finally:
        removal.release.set()

    assert latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, latency
    assert answered_during_the_removal
    [remove_thread] = handle.removes
    assert remove_thread != loop_thread
    # Exegol's order: the volume step's in-container delete, Exegol's own stop, then the docker removal.
    assert handle.calls == ["exec", "stop", "remove"]
    assert result["removed"].name == "lab"


async def test_the_removal_tail_runs_in_one_worker_thread_in_exegols_order(
    scene: _RemovalScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    import exegol.model.ExegolContainer as container_module

    handle = scene.handle
    handle.exec_release.set()
    steps: list[tuple[int, str, str | None]] = []

    def step(kind: str, argument: str | None) -> None:
        steps.append((threading.get_ident(), kind, argument))

    def remove(**kwargs: Any) -> None:
        step("remove", None)
        _RemovalHandle.remove(handle, **kwargs)

    def remove_container_by_id(c_id: str) -> None:
        step("backup", c_id)
        if c_id == "bak-gone":
            raise ObjectNotFound

    handle.remove = remove  # type: ignore[method-assign]
    # (id, name) tuples, as Exegol returns them; remove_managed_container passes the ids.
    scene.container.getExistingBackupContainers = lambda: [("bak-1", "lab-bak-1"), ("bak-gone", "lab-bak-gone")]
    scene.container.config.getNetworks = lambda: [
        SimpleNamespace(getNetworkName=lambda: "lab-net", shouldBeRemoved=lambda: True),
        SimpleNamespace(getNetworkName=lambda: "shared-net", shouldBeRemoved=lambda: False),
    ]
    monkeypatch.setattr(
        sdk_headless,
        "DockerUtils",
        lambda: SimpleNamespace(
            removeContainerById=remove_container_by_id, removeNetwork=lambda name: step("network", name)
        ),
    )
    monkeypatch.setattr(
        container_module,
        "DataCache",
        lambda: SimpleNamespace(remove_container_cache=lambda name: step("cache", name)),
    )
    loop_thread = threading.get_ident()

    removed = await _remove_through_the_service(timeout_s=30.0)

    assert [(kind, argument) for _thread, kind, argument in steps] == [
        ("remove", None),
        ("cache", "lab"),
        ("backup", "bak-1"),
        ("backup", "bak-gone"),
        ("network", "lab-net"),
    ]
    # One worker hop: every step of the tail ran in the same thread, and it was not the loop's.
    tail_threads = {thread for thread, _kind, _argument in steps}
    assert len(tail_threads) == 1
    assert loop_thread not in tail_threads
    # The network Exegol marks as shared is never removed.
    assert ("network", "shared-net") not in [(kind, argument) for _thread, kind, argument in steps]
    assert handle.calls == ["exec", "stop", "remove"]
    assert removed.name == "lab"


async def test_a_deadline_inside_the_removal_tail_is_reported_and_the_worker_finishes_it(
    scene: _RemovalScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    import exegol.model.ExegolContainer as container_module

    handle = scene.handle
    handle.exec_release.set()
    removal = _BlockedDockerRemoval(handle, block_seconds=5.0)
    cache_removed = threading.Event()
    cache_steps: list[str] = []

    def remove_container_cache(name: str) -> None:
        cache_steps.append(name)
        cache_removed.set()

    monkeypatch.setattr(
        container_module, "DataCache", lambda: SimpleNamespace(remove_container_cache=remove_container_cache)
    )
    failure: dict[str, Any] = {}

    try:
        started = time.perf_counter()
        try:
            await _remove_through_the_service(timeout_s=1.0)
        except RuntimeError as e:
            failure["error"] = str(e)
        failure["elapsed"] = time.perf_counter() - started
        # The deadline fired while the docker removal was blocked: the tail is still unfinished.
        assert removal.entered.is_set()
        assert not removal.done.is_set()
        assert cache_steps == []
    finally:
        removal.release.set()
    # Released, the abandoned worker finishes the whole tail: the container is not left half removed.
    await _wait_for_thread_event(removal.done)
    await _wait_for_thread_event(cache_removed)

    assert "not fully removed within 1 s" in failure["error"]
    assert "remove_container" in failure["error"]
    assert failure["elapsed"] < 1.0 + MAX_CONCURRENT_LATENCY, failure["elapsed"]
    assert cache_steps == ["lab"]
    assert handle.calls == ["exec", "stop", "remove"]


async def test_the_sdk_removal_tail_on_the_loop_is_detected(
    scene: _RemovalScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inversion: with Exegol's own remove, the docker removal blocks the loop and tools/list waits for it."""
    import exegol.model.ExegolContainer as container_module

    assert sdk_headless._SDK_REMOVE is not None
    monkeypatch.setattr(container_module.ExegolContainer, "remove", sdk_headless._SDK_REMOVE)
    handle = scene.handle
    handle.exec_release.set()
    removal = _BlockedDockerRemoval(handle)
    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(_remove_through_the_service)
            await _wait_for_thread_event(removal.entered)
            await client.list_tools()
            assert removal.started_at is not None
            since_removal_start = time.perf_counter() - removal.started_at
            stalled_until_the_removal_finished = removal.finished_at is not None
    finally:
        removal.release.set()

    assert stalled_until_the_removal_finished
    assert since_removal_start >= MAX_CONCURRENT_LATENCY, since_removal_start


async def test_the_volume_fallbacks_running_checks_are_the_only_docker_calls_on_the_loop(
    scene: _RemovalScene, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin the accepted on-loop residual of a removal (T-04-47, T-05.1-25).

    Exegol's __removeVolume is awaited unchanged, so its two `self.isRunning()` reads (a synchronous docker reload)
    in the PermissionError fallbacks run on the loop: one before the in-container Sentinel shred, one before the
    in-container workspace delete. Every other docker call of the removal runs in a worker thread. A change that
    offloads those two reads, or adds another on-loop docker call, must update this test and T-04-47 together.
    """
    import exegol.model.ExegolContainer as container_module

    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    scene.container.config.sentinel = sentinel

    async def refused_secure_remove(path: Any) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(container_module, "secure_remove", refused_secure_remove)
    handle = scene.handle
    handle.exec_release.set()
    reloads: list[tuple[int, int]] = []

    def recording_reload() -> None:
        reloads.append((threading.get_ident(), len(handle.execs)))

    handle.reload = recording_reload  # type: ignore[method-assign]
    loop_thread = threading.get_ident()

    await _remove_through_the_service(timeout_s=30.0)

    # Exegol's isRunning() before the Sentinel shred (no exec yet), then before the workspace delete (one exec done).
    assert [entry for entry in reloads if entry[0] == loop_thread] == [(loop_thread, 0), (loop_thread, 1)]
    # Every other reload (the exec and stop shadows, the service's state reads) ran in a worker thread.
    assert [entry for entry in reloads if entry[0] != loop_thread]
    for threads in (
        [thread for thread, _payload, _kwargs in handle.execs],
        [thread for thread, _kwargs in handle.stops],
        handle.removes,
    ):
        assert threads
        assert all(thread != loop_thread for thread in threads)
    assert handle.starts == []
    assert handle.calls == ["exec", "exec", "stop", "remove"]


def test_patch_installs_the_remove_mirror_idempotently(patched: Any) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    sdk_headless.patch_exegol_sdk()
    sdk_headless.patch_exegol_sdk()

    assert ExegolContainer.remove is sdk_headless._remove_offloaded
    # Exegol's own method is kept at import, so the drift pins read the SDK's source, not the mirror's.
    assert sdk_headless._SDK_REMOVE is not None
    assert sdk_headless._SDK_REMOVE.__qualname__ == "ExegolContainer.remove"
    assert "remove" in sdk_headless._PATCHED_CONTAINER_METHODS


def test_the_sdk_remove_still_matches_the_mirrored_lines() -> None:
    """Drift guard: _remove_offloaded and _remove_tail copy these lines of Exegol's remove; a change must fail CI."""
    import inspect

    from exegol.model.ExegolContainer import ExegolContainer

    assert sdk_headless._SDK_REMOVE is not None
    source = inspect.getsource(sdk_headless._SDK_REMOVE)
    for line in (
        "self.__container.remove()",
        "DataCache().remove_container_cache(self.name)",
        "DockerUtils().removeContainerById(c_id)",
        "DockerUtils().removeNetwork(net.getNetworkName())",
        'logger.info(f"Removing container {self.name}{backup_text}")',
        'logger.success(f"Container {self.name} successfully removed.")',
        "except ReadTimeout:",
        "except NotFound:",
        "except ObjectNotFound:",
        "if net.shouldBeRemoved():",
        'logger.debug(f"Network {net.getNetworkName()} will be removed.")',
    ):
        assert line in source, line
    order = [
        "__removeVolume()",
        "self.stop(timeout=2)",
        "self.__container.remove()",
        "DataCache()",
        "removeContainerById",
        "removeNetwork",
    ]
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions)
    # The SDK side of the accepted on-loop residual: exactly two synchronous running-state reads in the volume step
    # (T-04-47, T-05.1-25). An SDK release that adds another must fail CI here.
    volume_source = inspect.getsource(ExegolContainer._ExegolContainer__removeVolume)  # type: ignore[attr-defined]
    assert volume_source.count("self.isRunning()") == 2


# --- INT-01: create_container's post-create setup runs off the loop (05.1-04, patch step 14) ----------------------
#
# Exegol's postCreateSetup uploads the entrypoint with docker-py's put_archive, starts the container and runs the
# password update's exec, all synchronous inside an async def. Patch step 14 mirrors it statement by statement. These
# tests drive Exegol's REAL (patched) postCreateSetup, __start_container, __updatePasswd and __postStartSetup over a
# stub docker handle, through create_managed_container, so the call-site code is the production one.

# The container password the stub config holds; it must never reach a log record.
_CREATED_PASSWORD = "s3cret-pw"


class _CreatedHandle(_RemovalHandle):
    """The docker-py Container of a freshly created container: "created" state, a blockable put_archive.

    put_archive records its thread and arguments, then blocks on its own Events like the removal stub's exec. logs()
    answers the startup script once (a non-ASCII UTF-8 [INFO] line, then READY) and nothing afterwards. exec_error,
    when set, is raised by exec_run, standing in for docker refusing the password update.
    """

    def __init__(self, *, put_archive_block_seconds: float = 0.0) -> None:
        super().__init__(status="created")
        self.put_archive_block_seconds = put_archive_block_seconds
        self.put_archive_entered = threading.Event()
        self.put_archive_release = threading.Event()
        self.put_archive_done = threading.Event()
        self.put_archive_started_at: float | None = None
        self.put_archive_finished_at: float | None = None
        self.archives: list[tuple[int, str, Any]] = []
        self.log_script: list[list[bytes]] = [[b"[INFO] d\xc3\xa9marrage\n", b"READY\n"]]
        self.exec_error: Exception | None = None

    @property
    def status(self) -> str:
        # docker-py's Container.status reads the cached attrs: no I/O.
        return str(self.attrs["State"]["Status"])

    def put_archive(self, path: str, data: Any) -> bool:
        self.put_archive_started_at = time.perf_counter()
        self.archives.append((threading.get_ident(), path, data))
        self.put_archive_entered.set()
        self.put_archive_release.wait(self.put_archive_block_seconds)
        self.put_archive_finished_at = time.perf_counter()
        self.calls.append("put_archive")
        self.put_archive_done.set()
        return True

    def exec_run(self, cmd: Any, **kwargs: Any) -> Any:
        if self.exec_error is not None:
            raise self.exec_error
        return super().exec_run(cmd, **kwargs)

    def logs(self, stream: bool = False, follow: bool = False, since: Any = None, until: Any = None) -> Any:
        self.log_reads += 1
        if self.log_script:
            return iter(self.log_script.pop(0))
        return iter(())


class _CreatedConfig(_RemovalConfig):
    """The ContainerConfig getters the post-create setup and the password update read."""

    def __init__(self, workspace: Path, passwd: str | None = _CREATED_PASSWORD) -> None:
        super().__init__(workspace)
        self.passwd = passwd

    def getPasswd(self) -> str | None:
        return self.passwd

    def getUsername(self) -> str:
        return "root"

    def isGUIEnable(self) -> bool:
        return False

    def getVpnConfigPath(self) -> None:
        return None

    def getVolumes(self) -> list[Any]:
        return []


class _ImageSync:
    """ImageScriptSync at the name Exegol's model module imported: records the thread and argument of each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def getImageSyncTarData(self, include_entrypoint: bool = False) -> bytes:
        self.calls.append((threading.get_ident(), include_entrypoint))
        return b"entrypoint-tar"


class _CreatedScene:
    def __init__(self, handle: _CreatedHandle, config: _CreatedConfig, container: Any, sync: _ImageSync) -> None:
        self.handle = handle
        self.config = config
        self.container = container
        self.sync = sync


@pytest.fixture
def real_created(
    patched: Any, docker: _StubDocker, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> _CreatedScene:
    """A real ExegolContainer over a "created" stub handle, which the `docker` stub's createContainer hands back."""
    import asyncio

    import exegol.console.ExegolStatus as status_module
    import exegol.model.ExegolContainer as container_module

    handle = _CreatedHandle()
    config = _CreatedConfig(tmp_path)
    container = _removal_container(handle, tmp_path)
    container.config = config  # type: ignore[assignment]
    # Both are set by the SDK's __init__ and read by its post-start setup (it runs once per object).
    container._ExegolContainer__post_start_applied = False  # type: ignore[attr-defined]
    container._ExegolContainer__xhost_applied = False  # type: ignore[attr-defined]
    docker.createContainer = lambda model: container  # type: ignore[method-assign,assignment,misc]
    sync = _ImageSync()
    # Only the names Exegol's model module imported are replaced: ImageScriptSync would read the real entrypoint, and a
    # console lock contended in an earlier test would stay bound to that test's loop.
    monkeypatch.setattr(container_module, "ImageScriptSync", sync)
    monkeypatch.setattr(status_module, "ConsoleLock", asyncio.Lock())
    return _CreatedScene(handle, config, container, sync)


async def _create_through_the_adapter_code(**overrides: Any) -> Any:
    return await lifecycle_sdk.create_managed_container(_spec(**overrides), cache_lock=anyio.Lock())


@pytest.mark.usefixtures("asked", "resolver", "template")
async def test_the_post_create_setup_runs_off_the_loop_while_tools_list_stays_fast(
    real_created: _CreatedScene, caplog: pytest.LogCaptureFixture
) -> None:
    from exegol.utils.ExeLog import logger as sdk_logger

    caplog.set_level(logging.INFO, logger=sdk_logger.name)
    handle = real_created.handle
    handle.put_archive_block_seconds = BLOCK_SECONDS
    loop_thread = threading.get_ident()
    result: dict[str, Any] = {}

    async def create() -> None:
        result["container"] = await _create_through_the_adapter_code()

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(create)
            await _wait_for_thread_event(handle.put_archive_entered)
            await anyio.sleep(0.05)  # let the upload settle into its blocking wait
            started = time.perf_counter()
            await client.list_tools()
            latency = time.perf_counter() - started
            answered_during_the_upload = handle.put_archive_finished_at is None
            handle.put_archive_release.set()
    finally:
        handle.put_archive_release.set()

    assert latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, latency
    assert answered_during_the_upload
    # Exegol's order: the entrypoint upload, the docker start, then the password update's exec.
    assert handle.calls == ["put_archive", "start", "exec"]
    [(upload_thread, upload_path, upload_data)] = handle.archives
    assert upload_thread != loop_thread
    assert (upload_path, upload_data) == ("/", b"entrypoint-tar")
    # The tar build shares the upload's worker hop and asks for the entrypoint, as Exegol does.
    assert real_created.sync.calls == [(upload_thread, True)]
    [start_thread] = handle.starts
    assert start_thread != loop_thread
    [(exec_thread, _payload, exec_kwargs)] = handle.execs
    assert exec_thread != loop_thread
    assert exec_kwargs["detach"] is True
    assert exec_kwargs["stream"] is False
    assert "chpasswd" in exec_kwargs["environment"]["CMD"]
    # A non-ASCII UTF-8 startup line reached the SDK's log dispatch as the SDK's ContainerLogStream decodes it.
    assert any("démarrage" in record.getMessage() for record in caplog.records)
    assert result["container"] is real_created.container
    assert real_created.container._ExegolContainer__post_start_applied is True


@pytest.mark.usefixtures("asked", "resolver", "template")
async def test_a_create_without_a_password_runs_no_password_exec(real_created: _CreatedScene) -> None:
    real_created.config.passwd = None

    container = await _create_through_the_adapter_code()

    assert container is real_created.container
    assert real_created.handle.calls == ["put_archive", "start"]
    assert real_created.handle.execs == []


@pytest.mark.usefixtures("asked", "resolver", "template")
async def test_a_blocked_post_create_setup_returns_the_create_timeout_on_time(real_created: _CreatedScene) -> None:
    handle = real_created.handle
    handle.put_archive_block_seconds = 5.0

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match=r"was created but did not finish starting within 0\.3 s") as error:
            await _create_through_the_adapter_code(start_timeout_s=0.3)
        elapsed = time.perf_counter() - started
        # The abandoned worker thread is still inside the upload: it is never killed, it finishes on its own.
        still_blocked = not handle.put_archive_done.is_set()
    finally:
        handle.put_archive_release.set()
    await _wait_for_thread_event(handle.put_archive_done)

    assert elapsed < 2.0, elapsed
    assert "list_containers" in str(error.value)
    assert still_blocked
    # The abandoned upload is the only docker call: the deadline fired before the start.
    assert handle.calls == ["put_archive"]


@pytest.mark.usefixtures("asked", "resolver", "template")
async def test_the_sdk_post_create_setup_on_the_loop_is_detected(
    real_created: _CreatedScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inversion: with Exegol's own postCreateSetup, the blocked upload stalls the loop and tools/list waits."""
    from exegol.model.ExegolContainer import ExegolContainer

    assert sdk_headless._SDK_POST_CREATE_SETUP is not None
    monkeypatch.setattr(ExegolContainer, "postCreateSetup", sdk_headless._SDK_POST_CREATE_SETUP)
    handle = real_created.handle
    handle.put_archive_block_seconds = BLOCK_SECONDS
    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(_create_through_the_adapter_code)
            await _wait_for_thread_event(handle.put_archive_entered)
            await client.list_tools()
            assert handle.put_archive_started_at is not None
            since_upload_start = time.perf_counter() - handle.put_archive_started_at
            stalled_until_the_upload_finished = handle.put_archive_finished_at is not None
    finally:
        handle.put_archive_release.set()

    assert stalled_until_the_upload_finished
    assert since_upload_start >= MAX_CONCURRENT_LATENCY, since_upload_start


@pytest.mark.usefixtures("asked", "resolver", "template")
async def test_a_not_running_api_error_from_the_password_update_raises_runtime_error(
    real_created: _CreatedScene,
) -> None:
    real_created.handle.exec_error = DockerAPIError("boom", explanation="Container lab is not running")

    with pytest.raises(RuntimeError, match="Exegol cannot start the container after its creation"):
        await _create_through_the_adapter_code()

    # The SDK's statements are sequential: the failed password update skipped the post-start setup.
    assert real_created.container._ExegolContainer__post_start_applied is False


@pytest.mark.usefixtures("asked", "resolver", "template")
async def test_another_api_error_from_the_password_update_is_swallowed_as_in_the_sdk(
    real_created: _CreatedScene,
) -> None:
    real_created.handle.exec_error = DockerAPIError("boom", explanation="something else broke")

    container = await _create_through_the_adapter_code()

    assert container is real_created.container
    assert container._ExegolContainer__post_start_applied is True


async def test_a_temporary_post_create_setup_runs_only_the_post_start_setup(
    real_created: _CreatedScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    post_start_runs: list[Any] = []

    async def post_start_setup(self: Any) -> None:
        post_start_runs.append(self)

    monkeypatch.setattr(ExegolContainer, "_ExegolContainer__postStartSetup", post_start_setup)

    await real_created.container.postCreateSetup(is_temporary=True)

    # No upload, no start and no password exec for a temporary container; the post-start setup still runs once.
    assert real_created.handle.calls == []
    assert real_created.handle.archives == []
    assert real_created.sync.calls == []
    assert post_start_runs == [real_created.container]


async def test_a_non_api_error_from_the_password_update_skips_the_post_start_setup(
    real_created: _CreatedScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    post_start_runs: list[Any] = []

    async def post_start_setup(self: Any) -> None:
        post_start_runs.append(self)

    monkeypatch.setattr(ExegolContainer, "_ExegolContainer__postStartSetup", post_start_setup)
    real_created.handle.exec_error = requests.exceptions.ReadTimeout("docker is busy")

    with pytest.raises(requests.exceptions.ReadTimeout):
        await real_created.container.postCreateSetup()

    assert real_created.handle.calls == ["put_archive", "start"]
    assert post_start_runs == []


@pytest.mark.usefixtures("asked", "resolver", "template")
async def test_the_password_never_reaches_a_log_record(
    real_created: _CreatedScene, caplog: pytest.LogCaptureFixture
) -> None:
    from exegol.utils.ExeLog import logger as sdk_logger

    caplog.set_level(logging.DEBUG, logger=sdk_logger.name)

    await _create_through_the_adapter_code()

    messages = [record.getMessage() for record in caplog.records]
    # Non-vacuous: the SDK's own password-update line (username only) was captured, so the capture saw the step.
    assert any("Updating the root password inside the container" in message for message in messages)
    # The password step did run (its exec carries the password, which only the docker call sees).
    assert "chpasswd" in real_created.handle.execs[0][2]["environment"]["CMD"]
    assert not any(_CREATED_PASSWORD in message for message in messages)
    assert _CREATED_PASSWORD not in caplog.text
    assert not any(_CREATED_PASSWORD in str(record.args) for record in caplog.records)


async def test_the_offloaded_exec_quiet_daemon_mode_detaches_and_returns_zero(
    patched: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    handle = _CreatedHandle()
    handle.attrs["State"]["Status"] = "running"
    container = _removal_container(handle, tmp_path)
    exec_ = sdk_headless._offloaded_exec(container)
    loop_thread = threading.get_ident()

    assert await exec_("true", quiet=True) == 0

    [(daemon_thread, _payload, daemon_kwargs)] = handle.execs
    assert daemon_thread != loop_thread
    assert daemon_kwargs["detach"] is True
    assert daemon_kwargs["stream"] is False
    assert daemon_kwargs["environment"]["CMD"] == "true"
    # The synchronous quiet mode keeps detach False.
    assert await exec_("true", as_daemon=False, quiet=True) == 0
    assert handle.execs[1][2]["detach"] is False
    assert handle.execs[1][2]["stream"] is False
    # A non-quiet call is still Exegol's own exec, whatever the mode.
    delegated: list[tuple[Any, bool, bool]] = []

    async def sdk_exec(self: Any, command: Any, as_daemon: bool = True, quiet: bool = False, **_: Any) -> int:
        delegated.append((command, as_daemon, quiet))
        return 7

    monkeypatch.setattr(ExegolContainer, "exec", sdk_exec)
    assert await exec_("true") == 7
    assert await exec_("true", as_daemon=False) == 7
    assert delegated == [("true", True, False), ("true", False, False)]
    assert len(handle.execs) == 2


def test_patch_installs_the_post_create_mirror_idempotently(patched: Any) -> None:
    from exegol.model.ExegolContainer import ExegolContainer

    sdk_headless.patch_exegol_sdk()
    sdk_headless.patch_exegol_sdk()

    assert ExegolContainer.postCreateSetup is sdk_headless._post_create_setup_offloaded
    # Exegol's own method is kept at import, so the drift pins read the SDK's source, not the mirror's.
    assert sdk_headless._SDK_POST_CREATE_SETUP is not None
    assert sdk_headless._SDK_POST_CREATE_SETUP.__qualname__ == "ExegolContainer.postCreateSetup"
    assert "postCreateSetup" in sdk_headless._PATCHED_CONTAINER_METHODS


def test_the_sdk_post_create_setup_still_matches_the_mirrored_lines() -> None:
    """Drift guard: _post_create_setup_offloaded copies this statement structure of Exegol's; a change must fail CI."""
    import inspect

    from exegol.model.ExegolContainer import ExegolContainer

    assert sdk_headless._SDK_POST_CREATE_SETUP is not None
    source = inspect.getsource(sdk_headless._SDK_POST_CREATE_SETUP)
    order = [
        "if not is_temporary:",
        'self.__container.put_archive("/", ImageScriptSync.getImageSyncTarData(include_entrypoint=True))',
        'if self.__container.status.lower() == "created":',
        "await self.__start_container()",
        "await self.__updatePasswd()",
        "except APIError as e:",
        'if "is not running" in e.explanation:',
        "Exegol cannot start the container after its creation...",
        "await self.__postStartSetup()",
    ]
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions), order
    # The post-start setup is a plain statement at the method body's indentation: outside `if not is_temporary:` and
    # in no try block, so it runs for a temporary container and is skipped when the password update raises.
    assert "\n        await self.__postStartSetup()\n" in source
    assert "finally" not in source
    assert source.count("try:") == 1
    password_source = inspect.getsource(ExegolContainer._ExegolContainer__updatePasswd)  # type: ignore[attr-defined]
    assert "if self.config.getPasswd() is not None:" in password_source
    assert 'chpasswd", quiet=True)' in password_source
    exec_source = inspect.getsource(ExegolContainer.exec)
    assert "detach=as_daemon" in exec_source
    assert "if as_daemon:" in exec_source


# --- WR-06: a removal reports what is still on the host, not what was asked (04-14 Task 1) -------------------------
#
# Exegol's real remove() and __removeVolume run over the stub handle. Exegol deletes an empty private workspace and a
# Sentinel directory with no history without asking, and keeps a workspace the host cannot delete or a WSL one, so the
# keep options alone cannot say what survived: the adapter checks the host after the removal.


class _HostScene:
    def __init__(self, handle: _RemovalHandle, container: Any, workspace: Path) -> None:
        self.handle = handle
        self.container = container
        self.workspace = workspace
        # True: every host rmtree is refused, as root-owned files the in-container delete cannot reach make it.
        self.refuse_rmtree = False
        self.rmtree_calls: list[str] = []


@pytest.fixture
def host_scene(patched: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _HostScene:
    """Exegol's real removal over an empty private workspace on a tmp dir, the host rmtree real unless refused."""
    import shutil

    import exegol.model.ExegolContainer as container_module

    from exegol_mcp.adapter import real

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    handle = _RemovalHandle()
    handle.exec_release.set()
    container = _removal_container(handle, workspace)
    current = _HostScene(handle, container, workspace)

    def rmtree(path: str, *args: Any, **kwargs: Any) -> None:
        current.rmtree_calls.append(path)
        if current.refuse_rmtree:
            raise PermissionError(13, "Permission denied", path)
        shutil.rmtree(path, *args, **kwargs)

    def get_container(tag: str) -> Any:
        if tag == handle.name:
            return container
        raise ObjectNotFound

    monkeypatch.setattr(container_module, "shutil", SimpleNamespace(rmtree=rmtree))
    monkeypatch.setattr(container_module, "DataCache", lambda: SimpleNamespace(remove_container_cache=lambda n: None))
    monkeypatch.setattr(real, "DockerUtils", lambda: SimpleNamespace(getContainer=get_container))
    return current


async def _remove_on_the_host(*, keep_workspace: bool, keep_logs: bool) -> Any:
    return await lifecycle_service.remove_container(
        RealExegolAdapter(),
        "lab",
        keep_workspace=keep_workspace,
        keep_logs=keep_logs,
        settings=LifecycleSettings(),
        timeout_s=30.0,
    )


async def test_an_empty_workspace_exegol_deletes_unprompted_is_reported_not_kept(host_scene: _HostScene) -> None:
    removed = await _remove_on_the_host(keep_workspace=True, keep_logs=True)

    # Exegol os.rmdir's an empty private workspace with no prompt, even when keeping it was asked (nothing is lost).
    assert not host_scene.workspace.exists()
    assert removed.workspace_kept is False
    assert host_scene.handle.calls == ["stop", "remove"]


async def test_a_kept_non_empty_workspace_is_reported_kept(host_scene: _HostScene) -> None:
    (host_scene.workspace / "notes.txt").write_text("loot")

    removed = await _remove_on_the_host(keep_workspace=True, keep_logs=True)

    assert (host_scene.workspace / "notes.txt").read_text() == "loot"
    assert removed.workspace_kept is True
    assert host_scene.rmtree_calls == []


async def test_a_workspace_the_host_cannot_delete_is_reported_kept(host_scene: _HostScene) -> None:
    (host_scene.workspace / "root-owned.txt").write_text("root")
    host_scene.refuse_rmtree = True

    removed = await _remove_on_the_host(keep_workspace=False, keep_logs=False)

    # Exegol tried the host, then the in-container delete, then the host again, and gave up with a warning.
    assert len(host_scene.rmtree_calls) == 2
    [(_thread, _payload, exec_kwargs)] = host_scene.handle.execs
    assert exec_kwargs["environment"]["CMD"] == "rm -rf /workspace"
    assert host_scene.workspace.exists()
    assert removed.workspace_kept is True


async def test_a_deleted_workspace_is_reported_not_kept(host_scene: _HostScene) -> None:
    (host_scene.workspace / "data.txt").write_text("gone soon")

    removed = await _remove_on_the_host(keep_workspace=False, keep_logs=False)

    assert host_scene.rmtree_calls == [str(host_scene.workspace)]
    assert not host_scene.workspace.exists()
    assert removed.workspace_kept is False
    assert removed.logs_kept is False


async def test_a_wsl_workspace_is_reported_kept(host_scene: _HostScene) -> None:
    from exegol_mcp.adapter.port import RemovalOutcome

    # Docker volumes defined from WSL do not expose the real path: Exegol never deletes them.
    host_scene.container.config.volume_path = "/wsl/lab"

    outcome = await RealExegolAdapter().remove_container("lab", keep_workspace=False, keep_logs=False, timeout_s=30.0)

    assert outcome == RemovalOutcome(workspace_kept=True, logs_kept=False, host_workspace=False)
    assert host_scene.rmtree_calls == []
    assert host_scene.handle.calls == ["stop", "remove"]


@pytest.mark.parametrize(
    ("sentinel_state", "logs_kept"),
    [("history", True), ("no_history", False), ("disabled", False)],
)
async def test_sentinel_history_is_reported_from_the_host(
    host_scene: _HostScene, tmp_path: Path, sentinel_state: str, logs_kept: bool
) -> None:
    from exegol.config.StaticContainerPath import StaticFileName

    sentinel = tmp_path / "sentinel"
    if sentinel_state != "disabled":
        sentinel.mkdir()
        host_scene.container.config.sentinel = sentinel
    if sentinel_state == "history":
        (sentinel / StaticFileName.SENTINEL_LIVE_LOG.value).write_text('{"cmd": "nxc smb 10.0.0.1"}\n')

    removed = await _remove_on_the_host(keep_workspace=False, keep_logs=True)

    # With history, keep_logs keeps it. Without history, Exegol securely removes the directory unprompted.
    assert removed.logs_kept is logs_kept
    assert sentinel.exists() is logs_kept


async def test_the_host_state_checks_run_off_the_loop(host_scene: _HostScene, monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.get_ident()
    threads: dict[str, int] = {}
    read_paths = lifecycle_sdk._host_data_paths
    check_remaining = lifecycle_sdk._host_data_remaining

    def recording_paths(container: Any) -> Any:
        threads["paths"] = threading.get_ident()
        return read_paths(container)

    def recording_remaining(data: Any) -> tuple[bool, bool]:
        threads["remaining"] = threading.get_ident()
        return check_remaining(data)

    monkeypatch.setattr(lifecycle_sdk, "_host_data_paths", recording_paths)
    monkeypatch.setattr(lifecycle_sdk, "_host_data_remaining", recording_remaining)

    await _remove_on_the_host(keep_workspace=True, keep_logs=True)

    assert set(threads) == {"paths", "remaining"}
    assert threads["paths"] != loop_thread
    assert threads["remaining"] != loop_thread


# --- WR-08: a removal's outcome is read after its deadline (04-17) ------------------------------------------------
#
# Exegol's docker tail (the container, backup and network removals) now runs in one worker hop (patch step 13); the
# stub's synchronous sleep stands in for a removal whose last step does not yield before the deadline is read. The
# deadline is read from its own scope, and the host state is checked after that scope: a removal that finished past
# its deadline is done, never reported as timed out.


async def test_a_removal_that_finished_past_its_deadline_is_reported_done(removable: _Removable) -> None:
    from exegol_mcp.adapter.port import RemovalOutcome

    removable.sync_block_seconds = 0.6

    outcome = await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=0.3)

    assert outcome == RemovalOutcome(workspace_kept=True, logs_kept=False, host_workspace=False)
    assert removable.removed is True


class _ListingAdapter(RealExegolAdapter):
    """The real adapter, listing one managed container of run "abc" without Docker."""

    async def list_containers(self) -> list[ContainerInfo]:
        return [
            ContainerInfo(
                name="lab",
                creation_date="2026-11-07",
                image_name="free",
                image_version="1.0",
                status="Running",
                network_driver=None,
                network_name=None,
                features=[],
                devices=[],
                vpn=None,
                env=[],
                is_privileged=False,
                capabilities=[],
                comment=None,
                managed=True,
                run_id="abc",
            )
        ]


async def test_cleanup_lists_a_removal_that_finished_past_its_deadline_under_removed(removable: _Removable) -> None:
    removable.sync_block_seconds = 0.6

    result = await lifecycle_service.cleanup_containers(
        _ListingAdapter(),
        scope="this_run",
        run_id="abc",
        keep_workspace=None,
        keep_logs=None,
        settings=LifecycleSettings(),
        timeout_s=0.3,
    )

    assert result.removed == ["lab"]
    assert result.failed == []
    assert removable.removed is True


async def test_an_os_timeout_error_during_a_removal_is_not_reported_as_the_deadline(removable: _Removable) -> None:
    error = OSError(errno.ETIMEDOUT, os.strerror(errno.ETIMEDOUT))
    assert isinstance(error, TimeoutError)
    removable.remove_error = error

    with pytest.raises(RuntimeError) as caught:
        await RealExegolAdapter().remove_container("lab", keep_workspace=True, keep_logs=True, timeout_s=30.0)

    message = str(caught.value)
    assert "lab" in message
    assert os.strerror(errno.ETIMEDOUT) in message
    assert "not fully removed" not in message


def _unreadable(monkeypatch: pytest.MonkeyPatch, *paths: Path) -> None:
    """Make Path.exists() raise PermissionError for these paths only, as an EACCES parent does on Python 3.10."""
    original = Path.exists
    refused = {str(p) for p in paths}

    def exists(self: Path, *args: Any, **kwargs: Any) -> bool:
        if str(self) in refused:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)


def test_host_data_remaining_reports_an_unreadable_path_as_still_on_the_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    sentinel = tmp_path / "sentinel"
    _unreadable(monkeypatch, workspace, sentinel)

    data = lifecycle_sdk._HostData(volume_path=str(workspace), host_workspace=None, sentinel_dir=sentinel)

    assert lifecycle_sdk._host_data_remaining(data) == (True, True)


async def test_an_unreadable_host_path_after_a_removal_is_reported_as_still_on_the_host(
    removable: _Removable, monkeypatch: pytest.MonkeyPatch
) -> None:
    _unreadable(monkeypatch, Path(removable.config.volume_path))

    outcome = await RealExegolAdapter().remove_container("lab", keep_workspace=False, keep_logs=False, timeout_s=30.0)

    assert outcome.workspace_kept is True
    assert removable.removed is True


async def test_a_stalled_host_check_is_bounded_and_reports_still_on_the_host(
    removable: _Removable, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def stalled(*args: Any) -> tuple[bool, bool]:
        # A stat on unreachable network storage: the check does not answer within its bound.
        release.wait(5.0)
        return False, False

    monkeypatch.setattr(lifecycle_sdk, "_HOST_CHECK_TIMEOUT_S", 0.2, raising=False)
    monkeypatch.setattr(lifecycle_sdk, "_host_data_remaining", stalled)
    try:
        started = time.perf_counter()
        outcome = await RealExegolAdapter().remove_container(
            "lab", keep_workspace=False, keep_logs=False, timeout_s=30.0
        )
        elapsed = time.perf_counter() - started
    finally:
        release.set()

    assert elapsed < 0.2 + MAX_CONCURRENT_LATENCY, elapsed
    # A private volume path is set and there is no Sentinel: the workspace may still exist, no history can.
    assert outcome.workspace_kept is True
    assert outcome.logs_kept is False
    assert removable.removed is True


# --- WR-03 / WR-04: a name conflict never reaches Exegol's create rollback (04-16) ---------------------------------


def _api_error(status: int) -> DockerAPIError:
    """A docker-py APIError as the daemon sends it, with the HTTP status on its response."""
    response = requests.Response()
    response.status_code = status
    explanation = "Conflict. The container name is already in use" if status == 409 else "server error"
    return DockerAPIError(explanation, response=response, explanation=explanation)


class _DockerApi:
    """The low-level docker-py API under ContainerCollection.create: create_container fails, listings are recorded."""

    _version = "1.44"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.creates: list[dict[str, Any]] = []
        self.listings: list[dict[str, Any]] = []

    def create_container(self, **kwargs: Any) -> dict[str, str]:
        self.creates.append(kwargs)
        raise self.error

    def containers(self, **kwargs: Any) -> list[dict[str, Any]]:
        # Exegol's exact-name rollback lists the name first: any call here means it went looking for one to remove.
        self.listings.append(kwargs)
        return []


class _DockerClient:
    """A docker-py client whose `containers` is docker-py's real ContainerCollection over the stub API."""

    def __init__(self, error: Exception) -> None:
        self.api = _DockerApi(error)

    @property
    def containers(self) -> ContainerCollection:
        return ContainerCollection(client=self)


class _CreateModel:
    """The ExegolContainerTemplate surface Exegol's real createContainer reads, recording prepare and rollback."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.name = "mcp-race"
        self.image = SimpleNamespace(getDockerRef=lambda: "nwodtuhs/exegol:free")
        self.config = SimpleNamespace(
            getVolumes=list,
            getEntrypointCommand=lambda: (["/.exegol/entrypoint.sh"], ["load_setups", "endless"]),
            hostname="exegol-mcp-race",
            getExtraHost=dict,
            getDevices=list,
            getEnvs=dict,
            getLabels=dict,
            getPorts=dict,
            getPrivileged=lambda: False,
            getCapabilities=list,
            getSysctls=dict,
            shm_size="64M",
            interactive=True,
            tty=True,
            getWorkingDir=lambda: "/workspace",
            isNetworkDisabled=lambda: False,
            getNetwork=lambda: (None, None),
        )

    def prepare(self) -> None:
        self.calls.append("prepare")

    def rollback(self) -> None:
        self.calls.append("rollback")

    def getContainerName(self) -> str:
        return "exegol-mcp-race"


def _exegol_docker_utils(client: _DockerClient) -> DockerUtils:
    """Exegol's real DockerUtils over the stub client, built without connecting to a daemon."""
    utils = object.__new__(DockerUtils)
    utils._DockerUtils__client = client  # type: ignore[attr-defined]
    return utils


@pytest.mark.usefixtures("patched")
async def test_a_409_inside_the_guard_is_a_name_conflict_error_and_nothing_else_changes() -> None:
    conflict = _DockerClient(_api_error(409))

    # Inside exegol-mcp's create, seen from the worker thread createContainer runs in.
    with sdk_headless.create_conflict_guard(), pytest.raises(adapter_errors.ContainerNameConflictError) as error:
        await run_blocking(conflict.containers.create, "img")
    assert not isinstance(error.value, DockerAPIError)
    assert not isinstance(error.value, RuntimeError)

    # Outside the guard docker-py raises its APIError unchanged.
    with pytest.raises(DockerAPIError) as outside:
        conflict.containers.create("img")
    assert outside.value.status_code == 409

    # Inside the guard, any other status passes through unchanged.
    server = _DockerClient(_api_error(500))
    with sdk_headless.create_conflict_guard(), pytest.raises(DockerAPIError) as passed:
        await run_blocking(server.containers.create, "img")
    assert passed.value.status_code == 500
    assert sdk_headless._create_conflict_guard.get() is False


@pytest.mark.usefixtures("patched")
def test_exegols_create_rollback_never_runs_on_a_409_under_the_guard() -> None:
    client = _DockerClient(_api_error(409))
    model = _CreateModel()

    with sdk_headless.create_conflict_guard(), pytest.raises(adapter_errors.ContainerNameConflictError):
        DockerUtils.createContainer(_exegol_docker_utils(client), model)

    assert [create["name"] for create in client.api.creates] == ["exegol-mcp-race"]
    # No model.rollback() and no container listing, so the other process's container cannot be removed.
    assert model.calls == ["prepare"]
    assert client.api.listings == []


@pytest.mark.usefixtures("patched")
def test_exegols_create_rollback_still_runs_on_a_non_409_error() -> None:
    client = _DockerClient(_api_error(500))
    model = _CreateModel()

    with pytest.raises(RuntimeError):
        DockerUtils.createContainer(_exegol_docker_utils(client), model)

    assert model.calls == ["prepare", "rollback"]
    assert [listing["filters"]["name"] for listing in client.api.listings] == ["exegol-mcp-race"]


@pytest.mark.usefixtures("asked")
async def test_a_name_conflict_at_create_is_refused_with_name_in_use_and_the_tiers_restored(
    patched: Any, docker: _StubDocker, template: _StubTemplate, profiles: _StubProfiles
) -> None:
    previous = _tier(patched)
    guard_seen: list[bool] = []

    def refused_by_docker() -> None:
        guard_seen.append(sdk_headless._create_conflict_guard.get())
        raise adapter_errors.ContainerNameConflictError("Docker refused the create: the name is already in use")

    docker.on_create = refused_by_docker

    with pytest.raises(ValueError) as error:
        await RealExegolAdapter().create_container(_spec(name="raced", image=None, profile="web"))

    assert str(error.value) == NAME_IN_USE_MESSAGE.format(name="raced")
    assert guard_seen == [True]
    assert sdk_headless._create_conflict_guard.get() is False
    assert not any(event[0] == "create" for event in docker.events)
    assert _tier(patched) == previous
    assert _torn_down()


@pytest.mark.usefixtures("asked", "resolver")
async def test_the_name_is_checked_again_right_before_create(docker: _StubDocker, template: _StubTemplate) -> None:
    # Free at the lock's first check, taken by another process by the final one.
    docker.free_lookups["late"] = 1

    with pytest.raises(ValueError) as error:
        await RealExegolAdapter().create_container(_spec(name="late"))

    assert str(error.value) == NAME_IN_USE_MESSAGE.format(name="late")
    # The final check ran after the model was built, and createContainer never ran.
    assert template.calls == [("late", "exegol-late")]
    assert [event for event in docker.events if event[0] == "lookup"] == [("lookup", "late"), ("lookup", "late")]
    assert docker.create_threads == []


@pytest.mark.usefixtures("asked", "resolver")
async def test_a_workspace_replaced_by_a_symlink_before_create_is_refused(
    docker: _StubDocker, template: _StubTemplate, tmp_path: Path
) -> None:
    workspace = os.path.realpath(tmp_path / "engagement")
    elsewhere = tmp_path / "elsewhere"
    os.mkdir(workspace)
    elsewhere.mkdir()

    async def swap_for_a_symlink() -> None:
        # Between the service's validation and Docker's bind: the allowlisted directory becomes a symlink.
        os.rmdir(workspace)
        os.symlink(elsewhere, workspace)

    template.during = swap_for_a_symlink

    with pytest.raises(ValueError) as error:
        await RealExegolAdapter().create_container(_spec(workspace=workspace))

    assert workspace in str(error.value)
    assert "symlink" in str(error.value)
    assert docker.create_threads == []


def test_docker_still_creates_containers_the_way_exegol_expects() -> None:
    # The 409 guard relies on Exegol calling docker-py's ContainerCollection.create and rolling back on APIError.
    source = inspect.getsource(DockerUtils.createContainer)
    for fragment in ("self.__client.containers.create", "except APIError", "model.rollback()", "c.remove()"):
        assert fragment in source, fragment


@pytest.mark.usefixtures("sdk_restore")
def test_the_patch_guards_docker_pys_create_and_never_wraps_itself() -> None:
    sdk_headless.patch_exegol_sdk()
    sdk_headless.patch_exegol_sdk()

    assert ContainerCollection.create is sdk_headless._guarded_containers_create
    original = sdk_headless._DOCKER_CONTAINERS_CREATE
    assert original is not sdk_headless._guarded_containers_create
    assert original.__module__ == "docker.models.containers"
    # Patched twice, a 409 outside the guard still reaches docker-py exactly once and raises its APIError.
    client = _DockerClient(_api_error(409))
    with pytest.raises(DockerAPIError):
        client.containers.create("img")
    assert len(client.api.creates) == 1


# --- WR-09: Exegol's host Sentinel shred runs off the loop (04-18) -------------------------------------------------
#
# Exegol's host-side Sentinel shred (FsUtils.secure_remove) is a coroutine that never yields: on Linux it walks the tree
# by descriptor and overwrites every file three times, with an fsync per pass. Patch step 9 runs it in a worker thread
# while an exegol-mcp removal runs. These tests drive Exegol's REAL remove() and __removeVolume over the stub handle,
# with a Sentinel directory that holds no history, so Exegol shreds it without a prompt.


class _SlowShred:
    """A secure_remove stand-in that blocks synchronously, like Exegol's Linux shred, then deletes the tree."""

    def __init__(self, block_seconds: float = BLOCK_SECONDS) -> None:
        self.block_seconds = block_seconds
        self.entered = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()
        self.thread: int | None = None
        self.started_at: float | None = None
        self.calls: list[tuple[Path, int]] = []

    async def __call__(self, file_path: Path, passes: int = 3) -> None:
        self.thread = threading.get_ident()
        self.calls.append((file_path, passes))
        self.started_at = time.perf_counter()
        self.entered.set()
        self.release.wait(self.block_seconds)  # no await, like Exegol's shred
        shutil.rmtree(file_path)
        self.done.set()


def _empty_sentinel(host_scene: _HostScene, tmp_path: Path) -> Path:
    """Give the scene's container a Sentinel directory with no history: Exegol shreds it without asking."""
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    host_scene.container.config.sentinel = sentinel
    return sentinel


async def _wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    with anyio.fail_after(timeout_s):
        while not predicate():
            await anyio.sleep(0.01)


async def test_a_slow_host_sentinel_shred_runs_off_the_loop_while_tools_list_stays_fast(
    host_scene: _HostScene, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = _empty_sentinel(host_scene, tmp_path)
    shred = _SlowShred()
    monkeypatch.setattr(sdk_headless, "_EXEGOL_SECURE_REMOVE", shred)
    loop_thread = threading.get_ident()
    result: dict[str, Any] = {}
    latency = float("inf")
    answered_during_the_shred = False

    async def remove() -> None:
        result["removed"] = await _remove_on_the_host(keep_workspace=False, keep_logs=False)

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(remove)
            await _wait_until(lambda: shred.entered.is_set() or "removed" in result)
            if shred.entered.is_set():
                await anyio.sleep(0.05)  # let the shred settle into its blocking wait
                started = time.perf_counter()
                await client.list_tools()
                latency = time.perf_counter() - started
                answered_during_the_shred = not shred.done.is_set()
                shred.release.set()
    finally:
        shred.release.set()

    # Exegol's host shred went through exegol-mcp's offload, which calls the import-time original.
    assert shred.entered.is_set(), "Exegol's host Sentinel shred never reached the offloaded secure_remove"
    await _wait_for_thread_event(shred.done)
    assert latency < MAX_CONCURRENT_LATENCY < BLOCK_SECONDS, latency
    assert answered_during_the_shred
    assert shred.thread != loop_thread
    assert shred.calls == [(sentinel, 3)]
    assert not sentinel.exists()
    assert result["removed"].logs_kept is False


async def test_a_slow_host_sentinel_shred_on_the_loop_is_detected(
    host_scene: _HostScene, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Inversion: with the wrapper bypassed, the slow shred is awaited on the loop and tools/list waits for it."""
    import exegol.model.ExegolContainer as container_module

    _empty_sentinel(host_scene, tmp_path)
    shred = _SlowShred()
    monkeypatch.setattr(container_module, "secure_remove", shred)

    async def remove() -> None:
        await _remove_on_the_host(keep_workspace=False, keep_logs=False)

    try:
        async with Client(build_server(None, FakeExegolAdapter())) as client, anyio.create_task_group() as tg:
            tg.start_soon(remove)
            await _wait_for_thread_event(shred.entered)
            await client.list_tools()
            assert shred.started_at is not None
            since_shred_start = time.perf_counter() - shred.started_at
            stalled_until_the_shred_finished = shred.done.is_set()
    finally:
        shred.release.set()

    assert stalled_until_the_shred_finished
    assert since_shred_start >= MAX_CONCURRENT_LATENCY, since_shred_start


# --- WR-09: the offload's scope, its deadline, and Exegol's unchanged shred (04-18 Task 2) -------------------------


async def test_a_host_shred_past_the_removal_deadline_raises_and_leaves_the_container_listed(
    host_scene: _HostScene, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _empty_sentinel(host_scene, tmp_path)
    shred = _SlowShred(block_seconds=10.0)
    monkeypatch.setattr(sdk_headless, "_EXEGOL_SECURE_REMOVE", shred)

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError) as raised:
            await lifecycle_service.remove_container(
                RealExegolAdapter(),
                "lab",
                keep_workspace=False,
                keep_logs=False,
                settings=LifecycleSettings(),
                timeout_s=1.0,
            )
        elapsed = time.perf_counter() - started
        # The removal stopped at its deadline during the shred: docker's remove never ran, the container is listed.
        assert shred.entered.is_set()
        assert host_scene.handle.removes == []
        assert await RealExegolAdapter().get_container_state("lab") is not None
    finally:
        shred.release.set()
    # The abandoned shred finishes in its worker thread once released; nothing outlives the test.
    await _wait_for_thread_event(shred.done)

    assert "not fully removed within 1 s" in str(raised.value)
    assert "remove_container" in str(raised.value)
    assert elapsed < 1.0 + MAX_CONCURRENT_LATENCY, elapsed


async def test_the_shred_offload_outside_a_removal_keeps_exegols_own_behaviour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[int, Path, int]] = []

    async def recorder(file_path: Path, passes: int = 3) -> None:
        calls.append((threading.get_ident(), file_path, passes))

    monkeypatch.setattr(sdk_headless, "_EXEGOL_SECURE_REMOVE", recorder)
    loop_thread = threading.get_ident()
    target = tmp_path / "sentinel"

    await sdk_headless._secure_remove_offloaded(target)
    with sdk_headless.removal_offload(_removal_container(_RemovalHandle(), tmp_path)):
        await sdk_headless._secure_remove_offloaded(target)
    await sdk_headless._secure_remove_offloaded(target)

    [outside, inside, after] = calls
    # Outside a removal: Exegol's own coroutine, awaited on the calling loop with its default 3 passes.
    assert outside == (loop_thread, target, 3)
    assert after == (loop_thread, target, 3)
    # Inside one: the same call, in a worker thread.
    assert inside[0] != loop_thread
    assert inside[1:] == (target, 3)


async def test_the_offloaded_shred_runs_exegols_real_secure_remove_in_a_worker_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import exegol.utils.FsUtils as fs_utils_module

    tree = tmp_path / "sentinel"
    (tree / "artifacts").mkdir(parents=True)
    (tree / "live.log").write_bytes(b"audit line\n" * 512)
    (tree / "artifacts" / "capture.bin").write_bytes(os.urandom(4096))
    threads: list[int] = []
    run_to_completion = sdk_headless._secure_remove_to_completion

    def recording(file_path: Path, passes: int) -> None:
        threads.append(threading.get_ident())
        run_to_completion(file_path, passes)

    monkeypatch.setattr(sdk_headless, "_secure_remove_to_completion", recording)
    loop_thread = threading.get_ident()

    # No substitute: Exegol's own descriptor-based (Linux) or portable shred is what runs.
    assert sdk_headless._EXEGOL_SECURE_REMOVE is fs_utils_module.secure_remove
    with sdk_headless.removal_offload(_removal_container(_RemovalHandle(), tmp_path)):
        await sdk_headless._secure_remove_offloaded(tree)

    assert not tree.exists()
    assert len(threads) == 1
    assert threads[0] != loop_thread


def test_exegols_volume_step_still_calls_the_module_secure_remove() -> None:
    """Drift guard: patch step 9 replaces the name __removeVolume reads at call time, imported from FsUtils (WR-09)."""
    import exegol.model.ExegolContainer as container_module
    import exegol.utils.FsUtils as fs_utils_module
    from exegol.model.ExegolContainer import ExegolContainer

    volume_source = inspect.getsource(ExegolContainer._ExegolContainer__removeVolume)  # type: ignore[attr-defined]
    module_source = inspect.getsource(container_module)

    assert "await secure_remove(sentinel_dir)" in volume_source
    assert "from exegol.utils.FsUtils import secure_remove" in module_source
    assert inspect.iscoroutinefunction(fs_utils_module.secure_remove)
    assert list(inspect.signature(fs_utils_module.secure_remove).parameters) == ["file_path", "passes"]


def test_two_removal_tails_never_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """WR-01: removal tails share the SDK's DataCache / DockerUtils singletons, so they serialize (Docker-free)."""
    import exegol.model.ExegolContainer as container_module

    events: list[str] = []
    first_in_remove = threading.Event()
    release_first = threading.Event()

    def make_container(name: str, blocking: bool) -> Any:
        def remove() -> None:
            events.append(f"{name}:remove")
            if blocking:
                first_in_remove.set()
                assert release_first.wait(5.0)

        handle = SimpleNamespace(remove=remove)
        return SimpleNamespace(name=name, config=SimpleNamespace(getNetworks=list), _handle=handle)

    monkeypatch.setattr(sdk_headless, "_docker_handle", lambda container: container._handle)
    monkeypatch.setattr(
        container_module,
        "DataCache",
        lambda: SimpleNamespace(remove_container_cache=lambda name: events.append(f"{name}:cache")),
    )
    first = threading.Thread(target=sdk_headless._remove_tail, args=(make_container("one", True), True, None))
    second = threading.Thread(target=sdk_headless._remove_tail, args=(make_container("two", False), True, None))
    first.start()
    try:
        assert first_in_remove.wait(5.0)
        second.start()
        # The first tail is blocked inside docker; the second must not have started any step.
        second.join(0.3)
        assert second.is_alive()
        assert events == ["one:remove"]
    finally:
        release_first.set()
        first.join(5.0)
        second.join(5.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert events == ["one:remove", "one:cache", "two:remove", "two:cache"]
