"""Container profiles through Exegol's own profile system (D-01, D-02, CONF-01, LIFE-02, Pitfall 7).

Docker-free and licence-free: ContainerProfileManager and SessionHandler are replaced, at the names
exegol_mcp.adapter.profiles imports, by stubs that log on the real Exegol logger the way Exegol does.
"""

import getpass
import os
import sys
from pathlib import Path

import pytest
from exegol.utils.ExeLog import logger as exegol_logger

# --- Stubs ----------------------------------------------------------------------------------------------------------


class _Session:
    """SessionHandler stub: a cached licence tier, and a reload that must never run."""

    pro = True

    def pro_feature_access(self) -> bool:
        return self.pro

    async def reload_session(self, *args, **kwargs) -> bool:  # pragma: no cover - the test fails if reached
        raise AssertionError("exegol-mcp must never reload the licence session")


class _Profile:
    """A resolved ContainerProfile stand-in."""

    def __init__(self, name: str) -> None:
        self.name = name


class _Manager:
    """ContainerProfileManager stub. Each load_profiles() call takes the next scripted state.

    A state is (profiles, load_errors, lookup_errors): the profiles found, the errors logged while loading the files,
    and per selection the error get_profile logs when it returns None.
    """

    def __init__(self, *states: tuple[dict, list[str], dict[str, str]]) -> None:
        self.states = list(states)
        self.loads = 0
        self.lookups: list[str] = []
        self._current: tuple[dict, list[str], dict[str, str]] = ({}, [], {})

    def __call__(self) -> "_Manager":  # the MetaSingleton: every construction returns the same manager
        return self

    def load_profiles(self) -> bool:
        index = min(self.loads, len(self.states) - 1)
        self.loads += 1
        self._current = self.states[index]
        for message in self._current[1]:
            exegol_logger.error(message)
        return not self._current[1]

    def get_profile(self, selection: str):
        self.lookups.append(selection)
        profiles, _, lookup_errors = self._current
        if selection in profiles:
            return profiles[selection]
        exegol_logger.error(lookup_errors.get(selection, f"Container profile '{selection}' not found."))
        return None


@pytest.fixture
def profiles_module():
    from exegol_mcp.adapter import profiles

    return profiles


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch, profiles_module) -> _Session:
    stub = _Session()
    monkeypatch.setattr(profiles_module, "SessionHandler", lambda: stub)
    return stub


def _install(monkeypatch: pytest.MonkeyPatch, profiles_module, manager: _Manager) -> _Manager:
    monkeypatch.setattr(profiles_module, "ContainerProfileManager", manager)
    return manager


WEB = _Profile("web")


# --- Task 1: startup validation (profile_problems) ------------------------------------------------------------------


def test_an_empty_allowlist_touches_neither_exegol_nor_the_licence(monkeypatch: pytest.MonkeyPatch) -> None:
    from exegol_mcp.adapter import profiles

    def _boom(*args, **kwargs):
        raise AssertionError("an empty allowlist must add no startup work")

    monkeypatch.setattr(profiles, "ContainerProfileManager", _boom)
    monkeypatch.setattr(profiles, "SessionHandler", _boom)

    assert profiles.profile_problems([]) == []


@pytest.mark.usefixtures("session")
def test_a_missing_profile_is_named_with_exegols_reason_markup_stripped(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    manager = _install(
        monkeypatch,
        profiles_module,
        _Manager(({"web": WEB}, [], {"ops": "Container profile [bold]'ops'[/bold] not found."})),
    )

    problems = profiles_module.profile_problems(["web", "ops"])

    assert problems == ["Profile 'ops': Container profile 'ops' not found."]
    assert manager.loads == 1
    assert manager.lookups == ["web", "ops"]


@pytest.mark.usefixtures("session")
def test_a_file_that_fails_validation_names_exegols_validation_reason(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    load_errors = [
        "/home/u/.exegol/profiles/local/ops.yml: 'network.mode': Input should be 'host' or 'docker'",
        "/home/u/.exegol/profiles/local/other.yml: 'vpn': Extra inputs are not permitted",
    ]
    _install(monkeypatch, profiles_module, _Manager(({"web": WEB}, load_errors, {})))

    problems = profiles_module.profile_problems(["ops", "web"])

    assert len(problems) == 1
    assert problems[0].startswith("Profile 'ops': ")
    assert "'network.mode': Input should be" in problems[0]
    assert "not found" in problems[0]
    # A broken sibling file is not blamed on this profile.
    assert "other.yml" not in problems[0]


@pytest.mark.usefixtures("session")
def test_every_allowlisted_profile_resolved_means_no_problem(monkeypatch: pytest.MonkeyPatch, profiles_module) -> None:
    _install(monkeypatch, profiles_module, _Manager(({"web": WEB, "local.ops": _Profile("ops")}, [], {})))

    assert profiles_module.profile_problems(["web", "local.ops"]) == []


def test_without_a_pro_licence_profiles_are_never_loaded(
    monkeypatch: pytest.MonkeyPatch, profiles_module, session: _Session
) -> None:
    session.pro = False
    manager = _install(monkeypatch, profiles_module, _Manager(({"web": WEB}, [], {})))

    problems = profiles_module.profile_problems(["web"])

    assert problems == [profiles_module.PROFILE_LICENCE_MESSAGE]
    assert "Pro" in profiles_module.PROFILE_LICENCE_MESSAGE
    assert "exegol" in profiles_module.PROFILE_LICENCE_MESSAGE
    assert manager.loads == 0


@pytest.mark.usefixtures("session")
def test_the_licence_comes_from_the_cached_session_without_a_reload(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    # _Session.reload_session raises: reaching it would fail this test.
    _install(monkeypatch, profiles_module, _Manager(({"web": WEB}, [], {})))

    assert profiles_module.profile_problems(["web"]) == []


@pytest.mark.usefixtures("session")
def test_the_collector_detaches_from_the_exegol_logger(monkeypatch: pytest.MonkeyPatch, profiles_module) -> None:
    handlers_before = list(exegol_logger.handlers)
    _install(monkeypatch, profiles_module, _Manager(({}, [], {})))
    profiles_module.profile_problems(["ops"])
    assert exegol_logger.handlers == handlers_before

    # An error logged between two calls is not attributed to the next one.
    exegol_logger.error("/home/u/.exegol/profiles/local/ops.yml: 'image': stray error from elsewhere")
    _install(monkeypatch, profiles_module, _Manager(({}, [], {"ops": "Container profile 'ops' not found."})))

    assert profiles_module.profile_problems(["ops"]) == ["Profile 'ops': Container profile 'ops' not found."]


def test_the_real_probe_delegates_profile_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from exegol_mcp.adapter import host

    seen: list[list[str]] = []

    def _problems(names):
        seen.append(list(names))
        return ["Profile 'ops': not found"]

    monkeypatch.setattr(host, "profile_problems", _problems)

    assert host.RealHostProbe().validate_profiles(["web", "ops"]) == ["Profile 'ops': not found"]
    assert seen == [["web", "ops"]]


# --- Task 2: create-time loading (load_allowed_profile) -------------------------------------------------------------


@pytest.mark.usefixtures("session")
def test_a_create_time_load_re_reads_the_profiles_on_every_call(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    manager = _install(monkeypatch, profiles_module, _Manager(({"web": WEB}, [], {})))

    assert profiles_module.load_allowed_profile("web") is WEB
    assert manager.loads == 1
    assert profiles_module.load_allowed_profile("web") is WEB
    assert manager.loads == 2


@pytest.mark.usefixtures("session")
def test_a_missing_profile_at_create_time_names_it_and_exegols_reason(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    _install(
        monkeypatch, profiles_module, _Manager(({}, [], {"web": "Container profile [bold]'web'[/bold] not found."}))
    )

    with pytest.raises(ValueError) as raised:
        profiles_module.load_allowed_profile("web")

    message = str(raised.value)
    assert "'web'" in message
    assert "cannot be used" in message
    assert "not found" in message
    assert "[bold]" not in message and "[-]" not in message


@pytest.mark.usefixtures("session")
def test_a_profile_that_breaks_after_startup_fails_with_its_validation_reason(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    broken = "/home/u/.exegol/profiles/local/web.yml: 'network.mode': Input should be 'host' or 'docker'"
    _install(monkeypatch, profiles_module, _Manager(({"web": WEB}, [], {}), ({}, [broken], {})))

    assert profiles_module.load_allowed_profile("web") is WEB
    with pytest.raises(ValueError) as raised:
        profiles_module.load_allowed_profile("web")

    message = str(raised.value)
    assert "Container profile 'web' cannot be used" in message
    assert "'network.mode': Input should be 'host' or 'docker'" in message


def test_without_a_pro_licence_a_create_never_loads_a_profile(
    monkeypatch: pytest.MonkeyPatch, profiles_module, session: _Session
) -> None:
    session.pro = False
    manager = _install(monkeypatch, profiles_module, _Manager(({"web": WEB}, [], {})))

    with pytest.raises(RuntimeError) as raised:
        profiles_module.load_allowed_profile("web")

    assert str(raised.value) == profiles_module.PROFILE_LICENCE_MESSAGE
    assert manager.loads == 0
    assert manager.lookups == []


@pytest.mark.usefixtures("session")
def test_a_create_time_load_never_reloads_the_licence_session(monkeypatch: pytest.MonkeyPatch, profiles_module) -> None:
    # _Session.reload_session raises: reaching it would fail this test.
    _install(monkeypatch, profiles_module, _Manager(({"web": WEB}, [], {})))

    assert profiles_module.load_allowed_profile("web") is WEB


def test_the_profile_patch_points_are_self_checked() -> None:
    from exegol_mcp.adapter import selfcheck

    for point in (
        ("exegol.profile.ContainerProfileManager", "ContainerProfileManager.load_profiles"),
        ("exegol.profile.ContainerProfileManager", "ContainerProfileManager.get_profile"),
        ("exegol.profile.ProfileApplication", "apply_profile"),
        ("exegol.utils.SessionHandler", "SessionHandler.pro_feature_access"),
    ):
        assert point in selfcheck._CALLABLE_POINTS, point
    assert ("exegol.utils.ExeLog", "logger") in selfcheck._ATTRIBUTE_POINTS
    selfcheck.assert_patch_points()


# --- The allowlisted profiles' read-write mounts, for the startup allowlist check (WR-10) ---------------------------


def _real_profile(data: dict):
    from exegol.profile.ContainerProfile import ContainerProfile

    return ContainerProfile.model_validate(data)


def _rw_profile(root: str):
    """A real profile declaring every mount field the startup check reads, under root: the read-write mounts and path
    overrides, a ':ro' mount, the VPN config and auth file, and a device."""
    return _real_profile(
        {
            "volumes": {
                "mounts": ["~/data:/data", f"{root}/ro:/ro:ro", f"{root}/rw:/rw:rw"],
                "workspace_path": f"{root}/ws",
                "private_workspace_path": f"{root}/pw",
                "exegol_resources_path": f"{root}/er",
            },
            "customization": {"my_resources_path": f"{root}/mr"},
            "sentinel": {"sentinel_logs_host_path": f"{root}/sl"},
            "vpn": {"config": f"{root}/vpn/client.ovpn", "auth_file": f"{root}/vpn/creds"},
            "system": {"devices": [f"{root}/dev/ttyX"]},
        }
    )


@pytest.mark.usefixtures("session")
def test_profile_mount_dirs_reads_the_rw_mounts_and_path_overrides_of_real_profiles(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path
) -> None:
    import os
    from pathlib import Path

    from exegol_mcp.adapter import sudo

    root = os.path.realpath(tmp_path)
    home = os.path.join(root, "home")
    monkeypatch.setattr(sudo, "invoking_user_home", lambda: Path(home))
    # Outside sudo, Exegol's SudoInfo.expandUser follows HOME (Path.expanduser, IN-12).
    monkeypatch.setenv("HOME", home)
    manager = _install(monkeypatch, profiles_module, _Manager(({"web": _rw_profile(root)}, [], {})))

    dirs = profiles_module.profile_mount_dirs(["web"])

    assert manager.loads == 1
    read_write = {
        os.path.join(home, "data"): False,
        os.path.join(root, "rw"): False,
        os.path.join(root, "ws"): False,
        os.path.join(root, "er"): False,
        os.path.join(root, "mr"): False,
        os.path.join(root, "pw"): True,
        os.path.join(root, "sl"): True,
        # A device is mapped read-write, its device nodes one by one: children-mounted, not read-only (WR-14).
        os.path.join(root, "dev", "ttyX"): True,
    }
    read_only = {
        os.path.join(root, "ro"): "volumes.mounts",
        os.path.join(root, "vpn", "client.ovpn"): "vpn.config",
        os.path.join(root, "vpn", "creds"): "vpn.auth_file",
    }
    assert len(dirs) == 11
    assert {d.path: d.children_mounted for d in dirs if not d.contains_only} == read_write
    contains_only = {d.path: d for d in dirs if d.contains_only}
    assert set(contains_only) == set(read_only)
    for path, field in read_only.items():
        assert contains_only[path].children_mounted is False
        assert field in contains_only[path].label, field
    labels = {d.path: d.label for d in dirs}
    for d in dirs:
        assert "'web'" in d.label
    assert "volumes.mounts" in labels[os.path.join(home, "data")]
    assert "~/data:/data" in labels[os.path.join(home, "data")]
    assert "volumes.workspace_path" in labels[os.path.join(root, "ws")]
    assert "volumes.exegol_resources_path" in labels[os.path.join(root, "er")]
    assert "customization.my_resources_path" in labels[os.path.join(root, "mr")]
    assert "volumes.private_workspace_path" in labels[os.path.join(root, "pw")]
    assert "sentinel.sentinel_logs_host_path" in labels[os.path.join(root, "sl")]
    assert "system.devices" in labels[os.path.join(root, "dev", "ttyX")]


@pytest.mark.usefixtures("session")
def test_profile_mount_dirs_of_a_profile_without_volumes_is_empty(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    _install(monkeypatch, profiles_module, _Manager(({"web": _real_profile({})}, [], {})))

    assert profiles_module.profile_mount_dirs(["web"]) == []


def test_profile_mount_dirs_without_a_pro_licence_is_empty(
    monkeypatch: pytest.MonkeyPatch, profiles_module, session: _Session
) -> None:
    manager = _install(monkeypatch, profiles_module, _Manager(({"web": _real_profile({})}, [], {})))

    assert profiles_module.profile_mount_dirs([]) == []
    session.pro = False
    assert profiles_module.profile_mount_dirs(["web"]) == []
    assert manager.loads == 0


class _NoGui:
    """GuiUtils stand-in for a host that shares no GUI socket."""

    @classmethod
    def getX11SocketPath(cls) -> None:
        return None

    @classmethod
    def isWaylandGuiAvailable(cls) -> bool:
        return False

    @classmethod
    def getWaylandSocketPath(cls):  # pragma: no cover - the test fails if reached
        raise AssertionError("getWaylandSocketPath must not be called without Wayland")


def test_the_real_probe_adds_the_allowlisted_profiles_mount_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from exegol_mcp.adapter import host
    from exegol_mcp.adapter.port import ExegolMountDir

    class _Resolver:
        def get(self, key):
            return tmp_path / str(key.value)

    marker = ExegolMountDir("/marker", "the volumes.mounts entry '/marker:/m' of profile 'web'", False)
    seen: list[list[str]] = []

    def _profile_dirs(selections):
        seen.append(list(selections))
        return [marker]

    monkeypatch.setattr(host, "OptionResolver", _Resolver)
    monkeypatch.setattr(host, "profile_mount_dirs", _profile_dirs)
    # Hermetic: no GUI socket, no Exegol settings directory, no Python install root and no read-only bind from this
    # host (this resolver returns a Path for every key, which PROFILE_SOURCES cannot be).
    monkeypatch.setattr(host, "GuiUtils", _NoGui)
    monkeypatch.setattr(host, "_exegol_settings_dirs", list)
    monkeypatch.setattr(host, "_exegol_mcp_config_dirs", list)
    monkeypatch.setattr(host, "_python_install_dirs", list)
    monkeypatch.setattr(host, "_exegol_read_only_binds", list)

    dirs = host.RealHostProbe().exegol_mount_dirs(["web"])

    assert len(dirs) == 5
    assert [d.children_mounted for d in dirs[:4]] == [False, False, True, True]
    assert dirs[4] is marker
    assert seen == [["web"]]


def test_exegols_raw_volume_syntax_still_matches_the_mirrored_pattern(profiles_module) -> None:
    import inspect

    from exegol.model.ContainerConfig import ContainerConfig

    assert profiles_module._EXEGOL_RAW_VOLUME.pattern in inspect.getsource(ContainerConfig.addRawVolume)


def test_the_profile_fields_the_mount_check_reads_still_exist() -> None:
    from exegol.profile.ContainerProfile import ContainerProfile, CustomizationSection, SentinelSection, VolumesSection

    assert {"volumes", "customization", "sentinel"} <= set(ContainerProfile.model_fields)
    assert {"mounts", "workspace_path", "private_workspace_path", "exegol_resources_path"} <= set(
        VolumesSection.model_fields
    )
    assert "my_resources_path" in CustomizationSection.model_fields
    assert "sentinel_logs_host_path" in SentinelSection.model_fields


def test_the_mount_path_option_keys_are_self_checked() -> None:
    from exegol_mcp.adapter import selfcheck

    for pair in (
        ("MY_RESOURCES_PATH", "customization.my_resources_path"),
        ("EXEGOL_RESOURCES_PATH", "volumes.exegol_resources_path"),
        ("PRIVATE_WORKSPACE_PATH", "volumes.private_workspace_path"),
        ("SENTINEL_LOGS_HOST_PATH", "sentinel.sentinel_logs_host_path"),
    ):
        assert pair in selfcheck._OPTION_KEYS, pair
    selfcheck.assert_patch_points()


# --- CR-02 and IN-12: profile mounts walked as Docker resolves them, ~ expanded as Exegol expands it ----------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
@pytest.mark.usefixtures("session")
async def test_a_symlinked_profile_mount_under_an_entry_is_refused_by_the_bind_walk(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path
) -> None:
    from exegol_mcp.bootstrap import readiness
    from tests.test_bootstrap import FakeHostProbe, _args, _never_prompt

    root = Path(os.path.realpath(tmp_path))
    entry = root / "entry"
    entry.mkdir()
    (root / "outside" / "share").mkdir(parents=True)
    os.symlink(root / "outside" / "share", entry / "share")
    profile = _real_profile({"volumes": {"mounts": [f"{entry}/share:/share"]}})
    _install(monkeypatch, profiles_module, _Manager(({"web": profile}, [], {})))

    dirs = profiles_module.profile_mount_dirs(["web"])
    problem = readiness.workspace_allowlist_mount_problem([str(entry)], dirs, fold_case=False)

    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert "volumes.mounts" in problem
    assert f"{entry}/share" in problem
    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=dirs),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            profiles=("web",),
            workspace_allowlist=[str(entry)],
        )
    assert raised.value.check == "workspace-allowlist"


def test_profile_paths_expand_user_like_exegol(monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path) -> None:
    from exegol.config.SudoInfo import SudoInfo

    user = getpass.getuser()
    named = profiles_module._profile_mount_dir(f"~{user}/share", "the volumes.workspace_path of profile 'web'", False)

    assert named is not None
    assert named.lexical == os.path.abspath(os.path.expanduser(f"~{user}/share"))
    assert "~" not in named.lexical

    home = Path(os.path.realpath(tmp_path)) / "home"
    monkeypatch.setenv("HOME", str(home))
    bare = profiles_module._profile_mount_dir("~/share", "the volumes.workspace_path of profile 'web'", False)

    assert bare is not None
    assert bare.lexical == os.path.abspath(os.fspath(SudoInfo.expandUser("~/share")))
    assert bare.lexical == str(home / "share")
    assert profiles_module._profile_mount_dir("", "label", False) is None
    assert profiles_module._profile_mount_dir(None, "label", False) is None


@pytest.mark.usefixtures("session")
def test_every_profile_mount_dir_carries_its_bind_walk(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path
) -> None:
    from tests.test_bootstrap import _assert_one_role

    root = os.path.realpath(tmp_path)
    monkeypatch.setenv("HOME", os.path.join(root, "home"))
    _install(monkeypatch, profiles_module, _Manager(({"web": _rw_profile(root)}, [], {})))

    dirs = profiles_module.profile_mount_dirs(["web"])

    assert len(dirs) == 11
    for mount in dirs:
        assert mount.lexical
        assert mount.candidates[0] == mount.lexical
        assert mount.path == os.path.realpath(mount.lexical)
        _assert_one_role(mount)


def test_the_expand_user_patch_point_is_self_checked() -> None:
    from exegol_mcp.adapter import selfcheck

    assert ("exegol.config.SudoInfo", "SudoInfo.expandUser") in selfcheck._CALLABLE_POINTS
    selfcheck.assert_patch_points()


# --- CR-03: read-only profile mounts and VPN files: an entry containing one is refused ------------------------------


@pytest.mark.usefixtures("session")
async def test_a_read_only_mount_of_a_profile_under_an_entry_stops_startup(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path
) -> None:
    from exegol_mcp.bootstrap import readiness
    from tests.test_bootstrap import FakeHostProbe, _args, _never_prompt

    root = Path(os.path.realpath(tmp_path))
    entry = root / "entry"
    (entry / "share").mkdir(parents=True)
    profile = _real_profile({"volumes": {"mounts": [f"{entry}/share:/share:ro"]}})
    _install(monkeypatch, profiles_module, _Manager(({"web": profile}, [], {})))

    dirs = profiles_module.profile_mount_dirs(["web"])
    problem = readiness.workspace_allowlist_mount_problem([str(entry)], dirs, fold_case=False)

    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert "volumes.mounts" in problem
    assert f"{entry}/share" in problem
    assert "read-only" in problem
    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=dirs),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            profiles=("web",),
            workspace_allowlist=[str(entry)],
        )
    assert raised.value.check == "workspace-allowlist"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
@pytest.mark.usefixtures("session")
def test_a_symlinked_read_only_mount_of_a_profile_under_an_entry_is_refused(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path
) -> None:
    from exegol_mcp.bootstrap import readiness

    root = Path(os.path.realpath(tmp_path))
    entry = root / "entry"
    entry.mkdir()
    (root / "outside" / "share").mkdir(parents=True)
    os.symlink(root / "outside" / "share", entry / "share")
    profile = _real_profile({"volumes": {"mounts": [f"{entry}/share:/share:ro"]}})
    _install(monkeypatch, profiles_module, _Manager(({"web": profile}, [], {})))

    problem = readiness.workspace_allowlist_mount_problem(
        [str(entry)], profiles_module.profile_mount_dirs(["web"]), fold_case=False
    )

    # The CR-03 swap on a read-only mount: Docker walks entry/share, which a container mounting the entry controls.
    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert f"{entry}/share" in problem
    assert problem.endswith(readiness._READ_ONLY_PROBLEM_FIX)


@pytest.mark.usefixtures("session")
def test_an_entry_inside_a_read_only_mount_of_a_profile_loads(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path
) -> None:
    from exegol_mcp.bootstrap import readiness

    root = os.path.realpath(tmp_path)
    entry = os.path.join(root, "wordlists", "custom")

    for mode, refused in (("ro", False), ("rw", True)):
        profile = _real_profile({"volumes": {"mounts": [f"{root}/wordlists:/w:{mode}"]}})
        _install(monkeypatch, profiles_module, _Manager(({"web": profile}, [], {})))
        dirs = profiles_module.profile_mount_dirs(["web"])

        problem = readiness.workspace_allowlist_mount_problem([entry], dirs, fold_case=False)

        assert len(dirs) == 1, mode
        assert (problem is not None) is refused, mode


@pytest.mark.parametrize(
    ("data", "field"),
    [
        ({"vpn": {"config": "{entry}/vpn/client.ovpn"}}, "vpn.config"),
        ({"vpn": {"auth_file": "{entry}/creds"}}, "vpn.auth_file"),
    ],
    ids=["vpn-config", "vpn-auth"],
)
@pytest.mark.usefixtures("session")
def test_a_profile_vpn_path_under_an_entry_is_refused_as_a_read_only_mount(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path, data: dict, field: str
) -> None:
    from exegol_mcp.bootstrap import readiness

    entry = os.path.join(os.path.realpath(tmp_path), "entry")
    section, values = next(iter(data.items()))
    filled = {
        key: [item.format(entry=entry) for item in value] if isinstance(value, list) else value.format(entry=entry)
        for key, value in values.items()
    }
    _install(monkeypatch, profiles_module, _Manager(({"web": _real_profile({section: filled})}, [], {})))

    dirs = profiles_module.profile_mount_dirs(["web"])
    problem = readiness.workspace_allowlist_mount_problem([entry], dirs, fold_case=False)

    assert [mount.contains_only for mount in dirs] == [True]
    assert problem is not None
    assert "lifecycle.workspace_allowlist.0" in problem
    assert field in problem
    assert problem.endswith(readiness._READ_ONLY_PROBLEM_FIX)


@pytest.mark.usefixtures("session")
def test_an_empty_vpn_section_or_device_adds_no_read_only_mount(
    monkeypatch: pytest.MonkeyPatch, profiles_module
) -> None:
    for data in ({"vpn": {}}, {"system": {}}, {"system": {"devices": []}}, {"system": {"devices": [""]}}):
        _install(monkeypatch, profiles_module, _Manager(({"web": _real_profile(data)}, [], {})))
        assert profiles_module.profile_mount_dirs(["web"]) == [], data


# --- WR-14: profile devices: an entry inside, equal to or containing a device directory is refused ------------------


@pytest.mark.parametrize("case", ["inside", "equal", "inside-mapping-syntax", "containing"])
@pytest.mark.usefixtures("session")
async def test_an_entry_nested_with_a_profile_device_directory_stops_startup(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path, case: str
) -> None:
    from exegol_mcp.bootstrap import readiness
    from tests.test_bootstrap import FakeHostProbe, _args, _never_prompt

    root = Path(os.path.realpath(tmp_path))
    devdir = root / "devs" / "usb"
    (devdir / "001").mkdir(parents=True)
    # Exegol's mapping syntax: the host path is the text before the first ':' (addUserDevice hands it all to Docker).
    raw = f"{devdir}:/dev/bus/usb:rwm" if case == "inside-mapping-syntax" else str(devdir)
    entry = {
        "inside": devdir / "001",
        "equal": devdir,
        "inside-mapping-syntax": devdir / "001",
        "containing": root / "devs",
    }[case]
    profile = _real_profile({"system": {"devices": [raw]}})
    _install(monkeypatch, profiles_module, _Manager(({"web": profile}, [], {})))

    dirs = profiles_module.profile_mount_dirs(["web"])

    # Docker maps every device node found under the directory at each create, read-write, and containers keep
    # CAP_MKNOD: a container mounting the entry could plant a node there for the next create (WR-14).
    with pytest.raises(readiness.ReadinessError) as raised:
        await readiness.check_readiness(
            FakeHostProbe(mount_dirs=dirs),
            _args("--type", "stdio"),
            prompt=_never_prompt,
            profiles=("web",),
            workspace_allowlist=[str(entry)],
        )
    message = str(raised.value)
    assert raised.value.check == "workspace-allowlist"
    assert "lifecycle.workspace_allowlist.0" in message
    assert "system.devices" in message
    assert message.endswith(readiness._MOUNT_PROBLEM_FIX), message
    if case != "containing":
        assert "device node" in message, message
    assert len(dirs) == 1
    assert dirs[0].children_mounted is True
    assert dirs[0].contains_only is False
    assert dirs[0].lexical == str(devdir)


@pytest.mark.usefixtures("session")
def test_a_profile_device_directory_is_reported_as_children_mounted(
    monkeypatch: pytest.MonkeyPatch, profiles_module, tmp_path
) -> None:
    import inspect

    import exegol.model.ContainerConfig as container_config

    from tests.test_bootstrap import _assert_one_role

    root = Path(os.path.realpath(tmp_path))
    devdir = root / "devs" / "usb"
    (devdir / "001").mkdir(parents=True)
    profile = _real_profile({"system": {"devices": [str(devdir), f"{root}/dev/ttyX", ""]}})
    _install(monkeypatch, profiles_module, _Manager(({"web": profile}, [], {})))

    dirs = profiles_module.profile_mount_dirs(["web"])

    assert [mount.lexical for mount in dirs] == [str(devdir), os.path.join(root, "dev", "ttyX")]
    for mount in dirs:
        assert mount.children_mounted is True
        assert mount.contains_only is False
        assert "system.devices" in mount.label
        _assert_one_role(mount)
    # Idempotency: profile_mount_dirs only reads.
    assert profiles_module.profile_mount_dirs(["web"]) == dirs
    # Drift guard: Exegol maps a profile device read-write by default and hands addUserDevice's string through as
    # written. A read-only device default would change what WR-14 guards against.
    add_device = inspect.getsource(container_config.ContainerConfig._ContainerConfig__addDevice)
    assert "readonly: bool = False" in add_device
    assert "perm += 'w'" in add_device
    assert "self.__addDevice(user_device_config)" in inspect.getsource(container_config)


def test_the_profile_read_only_mount_fields_still_exist() -> None:
    import inspect

    import exegol.model.ContainerConfig as container_config
    from exegol.config.OptionResolver import OptionKey
    from exegol.profile.ContainerProfile import ContainerProfile, SystemSection, VpnSection
    from exegol.profile.ProfileFieldMap import PROFILE_FIELD_MAP

    assert {"vpn", "system"} <= set(ContainerProfile.model_fields)
    assert {"config", "auth_file"} <= set(VpnSection.model_fields)
    assert "devices" in SystemSection.model_fields
    assert PROFILE_FIELD_MAP["vpn.config"] == OptionKey.VPN
    assert PROFILE_FIELD_MAP["vpn.auth_file"] == OptionKey.VPN_AUTH
    assert PROFILE_FIELD_MAP["system.devices"] == OptionKey.DEVICES
    assert "SudoInfo.expandUser(resolved_vpn)" in inspect.getsource(container_config.ContainerConfig.enableVPN)
    module_source = inspect.getsource(container_config)
    assert "SudoInfo.expandUser(input_vpn_auth)" in module_source
    assert "self.__addDevice(user_device_config)" in module_source
