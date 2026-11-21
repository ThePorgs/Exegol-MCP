"""Model-facing text carries no internal planning references (QUAL-04). Docker-free.

Everything a model or a user reads from the server is checked here against PLANNING_ID: the live tools/list
definitions (names, titles, descriptions, annotations, input and output schemas), resources/templates/list,
resources/list, the class docstrings and field descriptions of the schema-visible result models, and the
`exegol-mcp --help` text. Requirement, decision, threat, review, phase, pitfall and backlog references belong in
code comments, module docstrings and the planning files, not in the published interface.

A false positive is fixed by tightening PLANNING_ID and adding the case to the negative examples below, never by an
allowlist, a skip or an xfail.
"""

import importlib
import inspect
import re
from typing import Any

import pytest
from mcp import Client
from pydantic import BaseModel

from exegol_mcp.app import build_server
from exegol_mcp.bootstrap.cli import parse_args
from exegol_mcp.resources import RESOURCE_REGISTRY
from tests.fakes import V2_TOOL_NAMES, FakeExegolAdapter

# Requirement prefixes of .planning/REQUIREMENTS.md, D- decisions, T- threats (T-04-113), IN-/WR-/CR- review
# findings, then phase, pitfall and backlog (999.x) references.
PLANNING_ID = re.compile(
    r"\b(?:COMP|CONF|CORE|EXEC|FILE|INTR|LIFE|LONG|NOTF|PROTO|QUAL|RES|SEC|D|T|IN|WR|CR)-\d+"
    r"|\b[Pp]hase[- ]?\d+|\bPitfall \d+|\b999\.\d+"
)

_SNIPPET_RADIUS = 30


def _hits(value: Any, path: str) -> list[str]:
    """Walk str/dict/list values and return one "path: match: snippet" line per PLANNING_ID match."""
    found: list[str] = []
    if isinstance(value, str):
        for match in PLANNING_ID.finditer(value):
            start = max(0, match.start() - _SNIPPET_RADIUS)
            end = min(len(value), match.end() + _SNIPPET_RADIUS)
            snippet = value[start:end].replace("\n", " ")
            found.append(f"{path}: {match.group(0)}: ...{snippet}...")
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_hits(str(key), f"{path}.<key>"))
            found.extend(_hits(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_hits(item, f"{path}[{index}]"))
    return found


def _report(hits: list[str]) -> str:
    return "planning IDs in model-facing text:\n" + "\n".join(hits)


# --- The pattern itself ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["EXEC-05", "D-12", "FILE-01", "SEC-02", "QUAL-01", "RES-03", "T-04-113", "IN-31", "Phase 4", "Phase-2",
     "phase 5", "Pitfall 6", "999.3"],
)
def test_planning_id_pattern_matches_planning_references(text: str) -> None:
    assert PLANNING_ID.search(f"see ({text}) here"), text


@pytest.mark.parametrize(
    "text",
    ["UTF-8", "UTF-16", "SHA-256", "2026-07-28", "-32602", "127.0.0.1:2187", "base64", "Python 3.10",
     "exit_code -1", "/workspace/notes-2024.txt", "D-Bus"],
)
def test_planning_id_pattern_ignores_ordinary_text(text: str) -> None:
    assert not PLANNING_ID.search(f"see ({text}) here"), text


# --- CLI help -------------------------------------------------------------------------------------------------------


def test_cli_help_has_no_planning_ids(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # A wide terminal so textwrap never breaks a token at a hyphen (which would hide it), and no colour codes.
    monkeypatch.setenv("COLUMNS", "10000")
    monkeypatch.setenv("NO_COLOR", "1")
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--auth" in help_text
    assert "--accept-eula" in help_text
    hits = _hits(help_text, "exegol-mcp --help")
    assert not hits, _report(hits)


# --- Tool definitions -----------------------------------------------------------------------------------------------


async def test_tool_definitions_have_no_planning_ids() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tools = (await client.list_tools()).tools
    assert {tool.name for tool in tools} == V2_TOOL_NAMES
    hits: list[str] = []
    for tool in tools:
        dumped = tool.model_dump(mode="json", exclude_none=True, exclude={"output_schema"})
        hits.extend(_hits(dumped, tool.name))
    assert not hits, _report(hits)


# Tools whose output schema must exist, so a lost schema cannot make the output-schema guard pass vacuously.
_TOOLS_WITH_OUTPUT_SCHEMA = {"exec_command", "read_file", "list_directory", "cleanup_containers", "list_installed_tools"}

# Model modules whose classes become tool or resource schemas (pydantic publishes a class docstring as the schema
# description and each Field(description=...) as a property description).
_SCHEMA_MODEL_MODULES = ["exec", "job", "catalog", "files", "lifecycle"]


async def test_tool_output_schemas_have_no_planning_ids() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        tools = (await client.list_tools()).tools
    with_schema = {tool.name for tool in tools if tool.output_schema}
    assert _TOOLS_WITH_OUTPUT_SCHEMA <= with_schema, _TOOLS_WITH_OUTPUT_SCHEMA - with_schema
    hits: list[str] = []
    for tool in tools:
        if tool.output_schema:
            hits.extend(_hits(tool.output_schema, f"{tool.name}.output_schema"))
    assert not hits, _report(hits)


@pytest.mark.parametrize("module_name", _SCHEMA_MODEL_MODULES)
def test_schema_model_classes_have_no_planning_ids(module_name: str) -> None:
    # Catches a model before it is wired into a tool: every BaseModel defined in the module, docstring and fields.
    module = importlib.import_module(f"exegol_mcp.models.{module_name}")
    classes = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__
    ]
    assert classes, f"no pydantic model defined in {module.__name__}"
    hits: list[str] = []
    for cls in classes:
        hits.extend(_hits(cls.__doc__ or "", f"{module_name}.{cls.__name__}.__doc__"))
        for field_name, field in cls.model_fields.items():
            hits.extend(_hits(field.description or "", f"{module_name}.{cls.__name__}.{field_name}"))
    assert not hits, _report(hits)


# --- Resources ------------------------------------------------------------------------------------------------------


async def test_resource_listings_have_no_planning_ids() -> None:
    async with Client(build_server(None, FakeExegolAdapter())) as client:
        templates = (await client.list_resource_templates()).resource_templates
        resources = (await client.list_resources()).resources
    assert [template.uri_template for template in templates] == [entry.uri_template for entry in RESOURCE_REGISTRY]
    assert resources, "resources/list is empty on the default fake; the guard would pass vacuously"
    hits: list[str] = []
    for template in templates:
        hits.extend(_hits(template.model_dump(mode="json", exclude_none=True), f"template {template.name}"))
    for resource in resources:
        hits.extend(_hits(resource.model_dump(mode="json", exclude_none=True), f"resource {resource.uri}"))
    assert not hits, _report(hits)
