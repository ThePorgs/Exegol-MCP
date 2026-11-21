"""Docker-free lock on the CI gate structure (QUAL-03, D-09 to D-13).

The workflows themselves only run on GitHub. These tests parse them and pin the properties the release gate depends
on, so an edit that drops `needs: test`, unpins an action or widens a token fails the unit suite locally.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

SUB_TESTING = "sub_testing.yml"
SUB_TESTING_REF = "./.github/workflows/sub_testing.yml"
# Every workflow a GitHub event can start. Each must call sub_testing.yml and default its token to read-only.
ENTRYPOINTS = [
    "entrypoint_pull_request.yml",
    "entrypoint_nightly.yml",
    "entrypoint_prerelease.yml",
    "entrypoint_release.yml",
]
ALL_WORKFLOWS = [SUB_TESTING, *ENTRYPOINTS]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
MYPY_PLATFORMS = ["win32", "linux", "darwin"]
PUBLISH_ENVIRONMENTS = {"nightly", "release"}
DEAD_DOCKER_HOST = "unix:///nonexistent/docker.sock"

# `# - owner/repo vX.Y @ <sha>` lines of the "# Pinned GitHub Actions" header.
_PIN_HEADER_LINE = re.compile(r"^# - (?P<action>[\w.-]+/[\w.-]+) \S+ @ (?P<sha>[0-9a-f]{40})$")
_PINNED_USES = re.compile(r"^(?P<action>[\w.-]+/[\w.-]+)@(?P<sha>[0-9a-f]{40})$")


def _load(name: str) -> dict[str, Any]:
    path = WORKFLOW_DIR / name
    assert path.is_file(), f"{path.relative_to(REPO_ROOT)} is missing"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict), f"{name} does not parse to a mapping"
    return workflow


def _triggers(wf: dict[str, Any]) -> dict[str, Any]:
    # PyYAML (YAML 1.1) reads a bare `on:` key as the boolean True.
    triggers = wf.get("on", wf.get(True))
    if isinstance(triggers, str):
        return {triggers: None}
    if isinstance(triggers, list):
        return {t: None for t in triggers}
    assert isinstance(triggers, dict), f"unexpected trigger block {triggers!r}"
    return triggers


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job.get("steps") or [])


def _runs(job: dict[str, Any]) -> list[str]:
    return [step["run"] for step in _steps(job) if "run" in step]


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [step for step in _steps(job) if step.get("name") == name]
    assert len(matches) == 1, f"expected exactly one step named {name!r}, got {len(matches)}"
    return matches[0]


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _setup_pdm_python(job: dict[str, Any]) -> str:
    for step in _steps(job):
        if str(step.get("uses", "")).startswith("pdm-project/setup-pdm@"):
            return str(step["with"]["python-version"])
    raise AssertionError("job has no pdm-project/setup-pdm step")


def _pin_header(name: str) -> dict[str, str]:
    lines = (WORKFLOW_DIR / name).read_text(encoding="utf-8").splitlines()
    pins: dict[str, str] = {}
    if not lines or lines[0] != "# Pinned GitHub Actions":
        return pins
    for line in lines[1:]:
        if not line.startswith("#"):
            break
        match = _PIN_HEADER_LINE.match(line)
        if match:
            pins[match["action"]] = match["sha"]
    return pins


def _external_uses(wf: dict[str, Any]) -> list[str]:
    uses: list[str] = []
    for job in wf["jobs"].values():
        if "uses" in job and not str(job["uses"]).startswith("./"):
            uses.append(str(job["uses"]))
        uses += [str(step["uses"]) for step in _steps(job) if "uses" in step and not str(step["uses"]).startswith("./")]
    return uses


def _publishes(job: dict[str, Any]) -> bool:
    return any("pdm publish" in run for run in _runs(job))


def _write_permissions(job: dict[str, Any]) -> set[str]:
    permissions = job.get("permissions") or {}
    if isinstance(permissions, str):
        return {"*"} if permissions == "write-all" else set()
    return {scope for scope, level in permissions.items() if level == "write"}


def _uses_step(job: dict[str, Any], action: str) -> dict[str, Any]:
    matches = [step for step in _steps(job) if str(step.get("uses", "")).startswith(f"{action}@")]
    assert len(matches) == 1, f"expected exactly one {action} step, got {len(matches)}"
    return matches[0]


def _github_filter_regex(pattern: str) -> re.Pattern[str]:
    """Translate the subset of GitHub's branch/tag filter syntax the workflows use into a regular expression.

    Supported: character classes such as `[0-9]`, `+` (one or more of the preceding character or class) and literal
    characters, matched against the whole ref name. Any other filter special character fails the helper, so a pattern
    this translator cannot model faithfully never passes by accident.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "[":
            end = pattern.find("]", i)
            assert end > i, f"unterminated character class in {pattern!r}"
            body = pattern[i + 1 : end]
            assert re.fullmatch(r"(?:[0-9A-Za-z]-[0-9A-Za-z]|[0-9A-Za-z])+", body), f"unsupported class [{body}]"
            out.append(f"[{body}]")
            i = end + 1
        elif char == "+":
            assert out and out[-1] != "+", f"'+' must follow a character or class in {pattern!r}"
            out.append("+")
            i += 1
        else:
            assert char not in "*?!\\{}()|^$]", f"unsupported filter character {char!r} in {pattern!r}"
            out.append(re.escape(char))
            i += 1
    return re.compile("".join(out))


def _filter_admits(patterns: list[str], ref: str) -> bool:
    return any(_github_filter_regex(pattern).fullmatch(ref) for pattern in patterns)


# --- sub_testing.yml -------------------------------------------------------------------------------------------------


def test_sub_testing_is_only_a_reusable_workflow() -> None:
    assert set(_triggers(_load(SUB_TESTING))) == {"workflow_call"}


def test_sub_testing_jobs_all_run_after_code_testing() -> None:
    jobs = _load(SUB_TESTING)["jobs"]
    assert set(jobs) == {"testing", "unit-tests", "compatibility", "build-n-install"}
    for name in ("unit-tests", "compatibility", "build-n-install"):
        assert _needs(jobs[name]) == ["testing"], name
    assert all(job["runs-on"] == "ubuntu-latest" for job in jobs.values())


def test_testing_job_runs_mypy_then_pre_commit() -> None:
    job = _load(SUB_TESTING)["jobs"]["testing"]
    assert job["name"] == "Code testing"
    assert _setup_pdm_python(job) == "3.12"
    assert _runs(job) == [
        "pdm update && pdm sync -d -G testing",
        "pdm run -v mypy --check-untyped-defs --pretty",
        "pdm run -v pre-commit run --all-files --hook-stage pre-commit",
    ]


def test_unit_tests_run_every_supported_python_with_docker_unreachable() -> None:
    job = _load(SUB_TESTING)["jobs"]["unit-tests"]
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"] == {"version": PYTHON_VERSIONS}
    assert _setup_pdm_python(job) == "${{ matrix.version }}"
    assert job["env"]["DOCKER_HOST"] == DEAD_DOCKER_HOST
    runs = _runs(job)
    assert runs[0] == "pdm update && pdm sync -d -G testing"
    assert runs[-1] == "pdm run -v pytest -q"
    # The live-suite guard runs before pytest, so a live variable can never reach the test run.
    guard = _step(job, "Refuse live tests")
    assert _steps(job).index(guard) < _steps(job).index(_step(job, "Run unit tests"))


def _run_guard(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = _step(_load(SUB_TESTING)["jobs"]["unit-tests"], "Refuse live tests")["run"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXEGOL_MCP_LIVE_")}
    env.update(extra_env)
    # GitHub's default shell on ubuntu runners: bash with errexit and pipefail.
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash, the runner's default shell")
def test_live_guard_passes_without_live_variables() -> None:
    result = _run_guard({})
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash, the runner's default shell")
@pytest.mark.parametrize("variable", ["EXEGOL_MCP_LIVE_IMAGE", "EXEGOL_MCP_LIVE_CONTAINER", "EXEGOL_MCP_LIVE_NEW"])
def test_live_guard_fails_on_any_live_variable(variable: str) -> None:
    result = _run_guard({variable: "set-by-test"})
    assert result.returncode == 1
    assert variable in result.stderr
    assert "set-by-test" not in result.stdout + result.stderr  # names only, never values


def test_compatibility_matrix_type_checks_the_package_everywhere() -> None:
    job = _load(SUB_TESTING)["jobs"]["compatibility"]
    assert job["name"] == "Compatibility checks"
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"] == {"version": PYTHON_VERSIONS, "os": MYPY_PLATFORMS}
    assert _setup_pdm_python(job) == "${{ matrix.version }}"
    assert _runs(job)[-1] == (
        "pdm run -v mypy exegol_mcp/ --check-untyped-defs"
        " --python-version ${{ matrix.version }} --platform ${{ matrix.os }}"
    )


def test_build_n_install_installs_and_runs_both_distributions() -> None:
    job = _load(SUB_TESTING)["jobs"]["build-n-install"]
    runs = _runs(job)
    assert "pdm build" in runs
    assert "pdm venv create -n vtest-source --with-pip && pdm venv create -n vtest-wheel --with-pip" in runs
    # One install per distribution: the wrapper installs the sdist twice and never tests the wheel.
    assert _step(job, "Install pip source package")["run"] == "pdm run --venv vtest-source -v pip install ./dist/*.tar.gz"
    assert _step(job, "Install pip wheel package")["run"] == "pdm run --venv vtest-wheel -v pip install ./dist/*.whl"
    assert _step(job, "Smoke test")["run"] == (
        "pdm run --venv vtest-source exegol-mcp --help && pdm run --venv vtest-wheel exegol-mcp --help"
    )
    assert runs.index("pdm build") < runs.index(_step(job, "Smoke test")["run"])


# --- every workflow ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_WORKFLOWS)
def test_external_actions_are_pinned_and_listed_in_the_header(name: str) -> None:
    wf = _load(name)
    pins = _pin_header(name)
    for uses in _external_uses(wf):
        match = _PINNED_USES.match(uses)
        assert match, f"{name}: {uses!r} is not pinned to a full commit SHA"
        assert pins.get(match["action"]) == match["sha"], f"{name}: {uses!r} is missing from the pin header"


@pytest.mark.parametrize("name", ALL_WORKFLOWS)
def test_only_publish_jobs_get_a_write_token(name: str) -> None:
    for job_name, job in _load(name)["jobs"].items():
        writes = _write_permissions(job)
        if _publishes(job):
            assert writes == {"id-token"}, f"{name}:{job_name} must hold exactly id-token: write"
        else:
            assert not writes, f"{name}:{job_name} runs tests or tooling and must not hold {sorted(writes)} write"


@pytest.mark.parametrize("name", ALL_WORKFLOWS)
def test_no_run_script_splices_an_event_value(name: str) -> None:
    # A `${{ github.* }}` expression in run text is pasted into the script before the shell parses it (script
    # injection through a tag or branch name); event values reach scripts only through env.
    for job_name, job in _load(name)["jobs"].items():
        for run in _runs(job):
            assert not re.search(r"\$\{\{\s*github\.", run), f"{name}:{job_name} splices an event value: {run!r}"


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoint_defaults_the_token_to_read_only(name: str) -> None:
    assert _load(name).get("permissions") == {"contents": "read"}


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoint_runs_the_reusable_gate_and_publishes_only_after_it(name: str) -> None:
    jobs = _load(name)["jobs"]
    gate_jobs = {job_name for job_name, job in jobs.items() if job.get("uses") == SUB_TESTING_REF}
    assert gate_jobs, f"{name} never calls {SUB_TESTING}"
    for job_name, job in jobs.items():
        if not _publishes(job):
            continue
        assert gate_jobs & set(_needs(job)), f"{name}:{job_name} publishes without needing {sorted(gate_jobs)}"
        assert job.get("environment") in PUBLISH_ENVIRONMENTS, f"{name}:{job_name} publishes outside an environment"


# --- entrypoint_pull_request.yml --------------------------------------------------------------------------------------


def test_pull_request_entrypoint_covers_prs_and_branch_pushes() -> None:
    wf = _load("entrypoint_pull_request.yml")
    triggers = _triggers(wf)
    ignored_paths = [".github/**", "**.md"]
    assert triggers["pull_request"] == {"branches-ignore": ["main"], "paths-ignore": ignored_paths}
    assert triggers["push"] == {"branches-ignore": ["dev", "main"], "paths-ignore": ignored_paths}
    assert wf["concurrency"] == {"group": "${{ github.ref }}", "cancel-in-progress": True}
    assert wf["jobs"] == {"test": {"name": "Python tests and checks", "uses": SUB_TESTING_REF}}


# --- publishing entrypoints -------------------------------------------------------------------------------------------

TEST_PYPI_PUBLISH = "pdm publish --no-build --repository https://test.pypi.org/legacy/ --skip-existing"
PYPI_PUBLISH = "pdm publish --no-build"
PUBLISH_PERMISSIONS = {"id-token": "write", "contents": "read"}
BUILD_PERMISSIONS = {"contents": "read"}
DIST_ARTIFACT = {"name": "dist", "path": "dist/"}
RELEASE_TAG_PATTERNS = ["[0-9]+.[0-9]+.[0-9]+", "[0-9]+.[0-9]+.[0-9]+b[0-9]+", "[0-9]+.[0-9]+.[0-9]+rc[0-9]+"]
TAG_CHECK = "Tag must equal the package version"
# Release tags and package versions: X.Y.Z, PEP 440 betas and release candidates only (beta-first v2, UAT G-05-3).
RELEASE_TAGS = [
    ("2.0.0", True),
    ("10.20.30", True),
    ("2.0.0b1", True),
    ("2.0.0rc12", True),
    ("2.0.0a1", False),
    ("2.0.0.dev1", False),
    ("2.0.0.post1", False),
    ("v2.0.0", False),
    ("2.0", False),
    ("2.0.0b", False),
    ("2.0.0-beta.1", False),
    ("2.0.0B1", False),
    ("2.0.0b1+local", False),
]
OIDC_WORKFLOWS = ["entrypoint_release.yml", "entrypoint_nightly.yml"]


def _assert_unprivileged_build(job: dict[str, Any], runs: list[str]) -> None:
    """The build job: after the gate, read-only token, no persisted git credentials, uploads what it built."""
    assert job["name"] == "Build Python 🐍 distributions 📦"
    assert job["runs-on"] == "ubuntu-latest"
    assert _needs(job) == ["test"]
    assert job["permissions"] == BUILD_PERMISSIONS
    assert _uses_step(job, "actions/checkout")["with"]["persist-credentials"] is False
    assert _setup_pdm_python(job) == "3.12"
    assert _runs(job) == runs
    upload = _uses_step(job, "actions/upload-artifact")
    assert upload["name"] == "Upload distributions"
    assert upload["with"] == {**DIST_ARTIFACT, "if-no-files-found": "error"}
    build_step = next(step for step in _steps(job) if step.get("run") == "pdm build")
    assert _steps(job).index(build_step) < _steps(job).index(upload)


def _assert_upload_only_publish(job: dict[str, Any], name: str, environment: str, runs: list[str]) -> None:
    """The publish job: the only OIDC holder, after the gate and the build, uploading only the downloaded dist/."""
    assert job["name"] == name
    assert job["runs-on"] == "ubuntu-latest"
    assert _needs(job) == ["test", "build"]
    assert job["environment"] == environment
    assert job["permissions"] == PUBLISH_PERMISSIONS
    assert _uses_step(job, "actions/checkout")["with"]["persist-credentials"] is False
    assert _setup_pdm_python(job) == "3.12"
    download = _uses_step(job, "actions/download-artifact")
    assert download["name"] == "Download distributions"
    assert download["with"] == DIST_ARTIFACT
    assert _runs(job) == runs
    first_publish = next(step for step in _steps(job) if "run" in step)
    assert _steps(job).index(download) < _steps(job).index(first_publish)


def test_only_nightly_and_release_publish() -> None:
    publishers = {
        (name, job_name) for name in ENTRYPOINTS for job_name, job in _load(name)["jobs"].items() if _publishes(job)
    }
    assert publishers == {("entrypoint_nightly.yml", "publish"), ("entrypoint_release.yml", "publish")}


@pytest.mark.parametrize("name", OIDC_WORKFLOWS)
def test_no_oidc_job_builds_and_every_oidc_job_downloads(name: str) -> None:
    oidc_jobs = {job_name: job for job_name, job in _load(name)["jobs"].items() if "id-token" in _write_permissions(job)}
    assert oidc_jobs, f"{name} has no OIDC job"
    for job_name, job in oidc_jobs.items():
        # Build-time code (the unpinned pdm-backend, the project) must never run next to the publishing token.
        for run in _runs(job):
            assert run.startswith("pdm publish --no-build"), f"{name}:{job_name} runs {run!r} while holding the token"
        _uses_step(job, "actions/download-artifact")


def test_nightly_publishes_dev_to_test_pypi_after_the_gate() -> None:
    wf = _load("entrypoint_nightly.yml")
    assert _triggers(wf) == {"push": {"branches": ["dev"]}}
    jobs = wf["jobs"]
    assert set(jobs) == {"test", "build", "publish"}
    assert jobs["test"] == {"name": "Python tests and checks", "uses": SUB_TESTING_REF}
    _assert_unprivileged_build(jobs["build"], ["pdm build"])
    # Test PyPI only, never production.
    _assert_upload_only_publish(
        jobs["publish"], "Publish Python 🐍 distributions to TestPyPI 📦", "nightly", [TEST_PYPI_PUBLISH]
    )


def test_release_publishes_tags_to_test_pypi_then_pypi_after_the_gate() -> None:
    wf = _load("entrypoint_release.yml")
    assert _triggers(wf) == {"push": {"tags": RELEASE_TAG_PATTERNS}}
    jobs = wf["jobs"]
    assert set(jobs) == {"test", "build", "publish"}
    assert jobs["test"] == {"name": "Python tests and checks", "uses": SUB_TESTING_REF}
    build = jobs["build"]
    tag_check = _step(build, TAG_CHECK)
    _assert_unprivileged_build(build, [tag_check["run"], "pdm build"])
    _assert_upload_only_publish(
        jobs["publish"], "Publish Python 🐍 distributions to PyPI 📦", "release", [TEST_PYPI_PUBLISH, PYPI_PUBLISH]
    )


def test_release_tag_check_reads_the_tag_from_the_environment() -> None:
    tag_check = _step(_load("entrypoint_release.yml")["jobs"]["build"], TAG_CHECK)
    assert tag_check["env"] == {"TAG": "${{ github.ref_name }}"}
    assert "${{" not in tag_check["run"]
    assert 'os.environ["TAG"]' in tag_check["run"]


@pytest.mark.parametrize(("tag", "releasable"), RELEASE_TAGS)
def test_release_tag_filter_admits_releases_betas_and_release_candidates(tag: str, releasable: bool) -> None:
    patterns = _triggers(_load("entrypoint_release.yml"))["push"]["tags"]
    assert _filter_admits(patterns, tag) is releasable


@pytest.mark.parametrize("pattern", ["v*", "[0-9]+.**", "2.0.0?", "!2.0.0", "[0-9", "+1", "[^0]"])
def test_github_filter_helper_refuses_syntax_it_does_not_model(pattern: str) -> None:
    with pytest.raises(AssertionError):
        _github_filter_regex(pattern)


def _python_c_code(job: dict[str, Any], step_name: str) -> str:
    run = _step(job, step_name)["run"].strip()
    match = re.fullmatch(r"python3 -c '(?P<code>[^']+)'", run)
    assert match, f"unexpected {step_name!r} command {run!r}"
    return match["code"]


def _run_against_stand_in(
    tmp_path: Path, code: str, version: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    # Runs a workflow's own command against a stand-in package, exactly as the runner would from the checkout root.
    package = tmp_path / "exegol_mcp"
    package.mkdir()
    (package / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONSAFEPATH", "TAG"}}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _tag_check_code() -> str:
    return _python_c_code(_load("entrypoint_release.yml")["jobs"]["build"], TAG_CHECK)


@pytest.mark.parametrize(
    ("tag", "version", "exit_code"),
    [
        ("2.0.0", "2.0.0", 0),
        ("2.0.0b1", "2.0.0b1", 0),
        ("2.0.0rc1", "2.0.0rc1", 0),
        ("2.0.1", "2.0.0", 1),
        ("2.0.0", "2.0.0b1", 1),
        ("2.0.0b1", "2.0.0b2", 1),
    ],
)
def test_release_tag_check_requires_the_tag_to_equal_the_version(
    tmp_path: Path, tag: str, version: str, exit_code: int
) -> None:
    result = _run_against_stand_in(tmp_path, _tag_check_code(), version, {"TAG": tag})
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert tag in result.stdout and version in result.stdout, result.stdout  # the log names both values


VERSION_CHECK = "Check package version (only X.Y.Z, X.Y.ZbN or X.Y.ZrcN can be released)"
PRERELEASE_VERSIONS = [
    ("2.0.0", True),
    ("10.20.30", True),
    ("2.0.0b1", True),
    ("2.0.0b2", True),
    ("2.0.0rc1", True),
    ("2.0.0a1", False),
    ("2.0.0.dev1", False),
    ("2.0.0.post1", False),
    ("v2.0.0", False),
    ("2.0", False),
    ("2.0.0b", False),
    ("2.0.0-beta.1", False),
    ("2.0.0B1", False),
    ("2.0.0b1+local", False),
]


def test_prerelease_checks_the_version_then_runs_the_gate() -> None:
    wf = _load("entrypoint_prerelease.yml")
    assert _triggers(wf) == {"pull_request": {"branches": ["main"], "paths-ignore": [".github/**", "**.md"]}}
    assert wf["concurrency"] == {"group": "${{ github.ref }}", "cancel-in-progress": True}
    jobs = wf["jobs"]
    assert set(jobs) == {"preprod_test", "code_test"}
    preprod = jobs["preprod_test"]
    assert preprod["if"] == "github.event.pull_request.draft == false"
    assert [str(step.get("uses", "")).split("@")[0] for step in _steps(preprod)][:2] == [
        "actions/checkout",
        "actions/setup-python",
    ]
    assert _steps(preprod)[1]["with"] == {"python-version": "3.12"}
    _step(preprod, VERSION_CHECK)
    # Only this repository's own dev branch skips the gate (its pushes already ran it); a fork's dev branch does not.
    assert jobs["code_test"] == {
        "name": "Python tests and checks",
        "needs": "preprod_test",
        "if": (
            "github.event.pull_request.head.ref != 'dev'"
            " || github.event.pull_request.head.repo.full_name != github.repository"
        ),
        "uses": SUB_TESTING_REF,
    }


def _version_check_code() -> str:
    return _python_c_code(_load("entrypoint_prerelease.yml")["jobs"]["preprod_test"], VERSION_CHECK)


@pytest.mark.parametrize(("version", "releasable"), PRERELEASE_VERSIONS)
def test_prerelease_version_check_accepts_releases_betas_and_release_candidates(
    tmp_path: Path, version: str, releasable: bool
) -> None:
    result = _run_against_stand_in(tmp_path, _version_check_code(), version)
    assert result.stdout.strip() == version, result.stderr  # the stand-in, not the installed package, was checked
    assert result.returncode == (0 if releasable else 1), result.stderr


@pytest.mark.parametrize("version", sorted({v for v, _ in RELEASE_TAGS} | {v for v, _ in PRERELEASE_VERSIONS}))
def test_prerelease_check_and_release_tag_filter_admit_the_same_versions(tmp_path: Path, version: str) -> None:
    # One rule in two places: a version the PR check lets into main must be taggable, and only such a version.
    patterns = _triggers(_load("entrypoint_release.yml"))["push"]["tags"]
    result = _run_against_stand_in(tmp_path, _version_check_code(), version)
    assert result.returncode in (0, 1), result.stderr
    assert (result.returncode == 0) is _filter_admits(patterns, version)


# --- pre-push mypy hook (WR-04) ---------------------------------------------------------------------------------------

PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
RUN_MYPY = REPO_ROOT / "scripts" / "run_mypy.py"
MYPY_ARGS = ["--check-untyped-defs", "--pretty"]


def test_mypy_hook_runs_the_project_environment_launcher() -> None:
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    assert config["default_install_hook_types"] == ["pre-commit", "pre-push"]
    local_hooks = [hook for repo in config["repos"] if repo["repo"] == "local" for hook in repo["hooks"]]
    mypy_hooks = [hook for hook in local_hooks if hook["id"] == "mypy"]
    assert len(mypy_hooks) == 1
    hook = mypy_hooks[0]
    # language python starts the launcher with pre-commit's own interpreter on every OS, Windows included; the hook
    # environment stays empty, because the launcher only starts the project's own mypy.
    assert hook["entry"] == "python scripts/run_mypy.py"
    assert hook["language"] == "python"
    assert "additional_dependencies" not in hook
    assert hook["pass_filenames"] is False
    assert hook["types"] == ["python"]
    assert hook["args"] == MYPY_ARGS
    assert hook["stages"] == ["pre-push"]


def _stub(path: Path, label: str, exit_code: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/sh\necho "{label} $*"\necho "VIRTUAL_ENV=${{VIRTUAL_ENV-unset}}"\nexit {exit_code}\n', encoding="utf-8"
    )
    path.chmod(0o755)


def _run_launcher(project: Path, path_dirs: list[Path]) -> subprocess.CompletedProcess[str]:
    # A copy of the launcher inside a fake project, so its .venv lookup sees only the fake tree.
    (project / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN_MYPY, project / "scripts" / "run_mypy.py")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONSAFEPATH"}}
    env["PATH"] = os.pathsep.join(str(d) for d in path_dirs)
    env["VIRTUAL_ENV"] = str(project / "pre-commit-hook-env")  # what pre-commit's language: python sets
    return subprocess.run(
        [sys.executable, str(project / "scripts" / "run_mypy.py"), *MYPY_ARGS],
        cwd=project,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


posix_stubs = pytest.mark.skipif(os.name == "nt", reason="the stub executables are POSIX shell scripts")


@posix_stubs
def test_launcher_prefers_the_project_venv_mypy(tmp_path: Path) -> None:
    _stub(tmp_path / ".venv" / "bin" / "mypy", "venv-mypy", 3)
    _stub(tmp_path / "bin" / "pdm", "pdm", 4)
    _stub(tmp_path / "bin" / "uv", "uv", 5)
    result = _run_launcher(tmp_path, [tmp_path / "bin"])
    assert result.stdout.splitlines()[0] == "venv-mypy --check-untyped-defs --pretty", result.stderr
    assert result.returncode == 3  # mypy's own exit status


@posix_stubs
def test_launcher_falls_back_to_pdm_run(tmp_path: Path) -> None:
    _stub(tmp_path / "bin" / "pdm", "pdm", 4)
    _stub(tmp_path / "bin" / "uv", "uv", 5)
    result = _run_launcher(tmp_path, [tmp_path / "bin"])
    assert result.stdout.splitlines() == ["pdm run mypy --check-untyped-defs --pretty", "VIRTUAL_ENV=unset"]
    assert result.returncode == 4


@posix_stubs
def test_launcher_falls_back_to_uv_run_without_syncing(tmp_path: Path) -> None:
    _stub(tmp_path / "bin" / "uv", "uv", 5)
    result = _run_launcher(tmp_path, [tmp_path / "bin"])
    # --no-sync: the launcher runs the environment as it is and never installs anything.
    assert result.stdout.splitlines() == ["uv run --no-sync mypy --check-untyped-defs --pretty", "VIRTUAL_ENV=unset"]
    assert result.returncode == 5


@posix_stubs
def test_launcher_names_the_three_ways_when_none_is_found(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    result = _run_launcher(tmp_path, [tmp_path / "empty"])
    assert result.returncode != 0
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 1, result.stderr
    for way in (".venv", "pdm", "uv"):
        assert way in lines[0]
