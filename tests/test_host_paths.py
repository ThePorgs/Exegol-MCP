"""The SDK-free host-path rule (D-08, D-24): sudo-aware `~` expansion and Exegol's dangerous-path refusal.

Pure tests: nothing here touches the real home directory or the filesystem beyond os.path.realpath.
"""

import os
from pathlib import Path

import pytest

from exegol_mcp.services import host_paths
from exegol_mcp.services.host_paths import HOST_PATH_MAX_LEN, dangerous_host_path_reason, expand_host_path

HOME = Path("/home/u")


# --- expand_host_path -----------------------------------------------------------------------------------------------


def test_a_tilde_slash_prefix_expands_against_the_given_home() -> None:
    assert expand_host_path("~/eng", home=HOME) == Path(os.path.realpath("/home/u/eng"))


def test_a_bare_tilde_expands_to_the_realpath_of_home() -> None:
    assert expand_host_path("~", home=HOME) == Path(os.path.realpath("/home/u"))


def test_an_absolute_path_is_returned_as_its_realpath() -> None:
    assert expand_host_path("/opt/engagements/../engagements", home=HOME) == Path(os.path.realpath("/opt/engagements"))


def test_the_result_is_always_absolute() -> None:
    assert expand_host_path("~/eng", home=HOME).is_absolute()


@pytest.mark.parametrize(
    "raw",
    ["eng", "./eng", "~root/x", "~u", "", "/opt/a\x00b", "/" + "a" * HOST_PATH_MAX_LEN],
    ids=["relative", "dot-relative", "tilde-user", "tilde-name", "empty", "nul", "too-long"],
)
def test_malformed_paths_are_refused(raw: str) -> None:
    with pytest.raises(ValueError):
        expand_host_path(raw, home=HOME)


def test_a_path_at_the_length_limit_is_accepted() -> None:
    raw = "/" + "a" * (HOST_PATH_MAX_LEN - 1)
    assert len(raw) == HOST_PATH_MAX_LEN
    assert expand_host_path(raw, home=HOME).is_absolute()


# --- dangerous_host_path_reason -------------------------------------------------------------------------------------


def test_the_filesystem_root_is_dangerous() -> None:
    assert dangerous_host_path_reason(Path("/"), (HOME,)) is not None


def test_a_home_directory_is_dangerous() -> None:
    assert dangerous_host_path_reason(HOME, (HOME,)) is not None


def test_the_realpath_of_a_home_directory_is_dangerous() -> None:
    assert dangerous_host_path_reason(Path(os.path.realpath(HOME)), (HOME,)) is not None


def test_an_ancestor_of_a_home_is_dangerous() -> None:
    assert dangerous_host_path_reason(Path("/home"), (HOME,)) is not None


def test_every_given_home_is_checked() -> None:
    # Under sudo the invoking user's home and the process home (root's) both count, like Exegol's rule.
    assert dangerous_host_path_reason(Path("/root"), (HOME, Path("/root"))) is not None


def test_a_subdirectory_of_a_home_is_not_dangerous() -> None:
    assert dangerous_host_path_reason(Path("/home/u/eng"), (HOME,)) is None


def test_an_unrelated_directory_is_not_dangerous() -> None:
    assert dangerous_host_path_reason(Path("/opt/engagements"), (HOME,)) is None


def test_a_sibling_sharing_a_name_prefix_is_not_dangerous() -> None:
    # /home/user2 is neither /home/u nor an ancestor of it; a string-prefix check would get this wrong.
    assert dangerous_host_path_reason(Path("/home/u2"), (HOME,)) is None


# --- IN-08: whole-component nesting, with letter case folded on a case-insensitive host -----------------------------


def test_is_strictly_inside_compares_whole_components() -> None:
    assert host_paths.is_strictly_inside(Path("/a/b"), Path("/a"), fold_case=False)
    # A path is never strictly inside itself, and /ab only shares a name prefix with /a.
    assert not host_paths.is_strictly_inside(Path("/a"), Path("/a"), fold_case=False)
    assert not host_paths.is_strictly_inside(Path("/ab"), Path("/a"), fold_case=False)
    assert not host_paths.is_strictly_inside(Path("/a"), Path("/a/b"), fold_case=False)


def test_case_folding_finds_a_case_differing_nested_pair() -> None:
    assert host_paths.is_strictly_inside(Path("/u/work/sub"), Path("/u/Work"), fold_case=True)
    assert not host_paths.is_strictly_inside(Path("/u/work/sub"), Path("/u/Work"), fold_case=False)
    # Folding never makes a path strictly inside a case-differing spelling of itself.
    assert not host_paths.is_strictly_inside(Path("/u/work"), Path("/u/Work"), fold_case=True)


def test_the_case_fold_default_follows_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_paths, "CASE_INSENSITIVE_HOST", True)
    assert host_paths.is_strictly_inside(Path("/u/work/sub"), Path("/u/Work"))
    monkeypatch.setattr(host_paths, "CASE_INSENSITIVE_HOST", False)
    assert not host_paths.is_strictly_inside(Path("/u/work/sub"), Path("/u/Work"))


def test_a_case_differing_home_is_still_dangerous_on_a_case_insensitive_host() -> None:
    home = (Path("/Users/me"),)
    assert dangerous_host_path_reason(Path("/users/me"), home, fold_case=True) == "a home directory is never mountable"
    assert dangerous_host_path_reason(Path("/USERS"), home, fold_case=True) == (
        "an ancestor of a home directory is never mountable"
    )
    assert dangerous_host_path_reason(Path("/users/me"), home, fold_case=False) is None
    assert dangerous_host_path_reason(Path("/USERS"), home, fold_case=False) is None
