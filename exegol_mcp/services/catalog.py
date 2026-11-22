"""Tool/resource catalog service (CORE-05): installed-tools CSV parse, tool help, offline-resources listing.

SDK-free: it only talks to an ExegolPort. The CSV, help output and find output are read through
adapter.exec_command. v1 bugs fixed while porting: the short-row AttributeError (.strip() on a None field),
the unquoted tool_name in get_tool_help, and the exit-code coercion that turned a missing code into 1.
"""

import csv
import io
import re
import shlex

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.catalog import InstalledToolsPage, OfflineResourcesPage
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.container import InstalledTool
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.services.execution import require_running

# The opaque-cursor helpers live in services/paging.py (D-19), shared with list_directory. The private names are
# aliases of them, so the call sites below and tests/test_pagination.py are unchanged.
from exegol_mcp.services.paging import decode_cursor as _decode_cursor  # noqa: F401
from exegol_mcp.services.paging import encode_cursor as _encode_cursor  # noqa: F401
from exegol_mcp.services.paging import page as _page

# Help flags to try in order, as in v1.
HELP_FLAGS = ["--help", "-h", "help"]
# OS subdirectories under /opt/resources that offline resources are listed from.
VALID_TARGET_OS = ("linux", "windows")
# The built-in listing page size (D-23), the single source for both the model default and the service default.
_DEFAULT_PAGE_SIZE = ExecutionLimits().listing_page_size
# SEC-07/D-07: a tool name is validated against this ASCII charset (letters, digits, '.', '_', '+', '-') and a
# length cap before it is ever interpolated into a shell command. The shlex.quote in get_tool_help stays as
# defense-in-depth; this rejects clearly invalid names up front with a ValueError (-> ToolError).
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
_TOOL_NAME_MAX_LEN = 128


def _field(row: dict[str, str | None], *keys: str) -> str | None:
    """Return the first non-empty CSV field among the given header variants, or None.

    Uses (row.get(k) or "").strip() so a short row (csv.DictReader fills missing fields with None) never
    raises AttributeError, the v1 crash.
    """
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return None


def parse_installed_tools(csv_text: str) -> list[InstalledTool]:
    """Parse the text of an installed_tools.csv into InstalledTool items, in file order.

    Pure: no I/O. The header Exegol images write is Tool,Version,Link,Description (older and free images write
    Tool,Link,Description). Accepted header variants are name/Tool/tool, version/Version, link/Link/url/URL and
    description/Description. A column the CSV does not have, or a short row, gives a None field instead of
    raising. Two rows with the same tool name are both kept, in file order. An empty text, or a header with no
    row, gives an empty list.

    Args:
        csv_text: The whole CSV text, header line first.

    Returns:
        Every row as an InstalledTool, in file order.
    """
    tools: list[InstalledTool] = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        tools.append(
            InstalledTool(
                name=_field(row, "name", "Tool", "tool") or "",
                version=_field(row, "version", "Version"),
                link=_field(row, "link", "Link", "url", "URL"),
                description=_field(row, "description", "Description"),
            )
        )
    return tools


async def read_installed_tools(adapter: ExegolPort, name: str) -> list[InstalledTool]:
    """Read /.exegol/installed_tools.csv from a running container and parse the whole catalog, unpaged.

    The single read path of the catalog: the paged list_installed_tools tool and the installed-tools resource both
    use it, so they always agree.

    Args:
        adapter: The Exegol port.
        name: Name of the container, as returned by list_containers.

    Returns:
        Every installed tool, in file order (see parse_installed_tools).

    Raises:
        ValueError: If the container name is unknown.
        RuntimeError: If the container is not running.
    """
    await require_running(adapter, name)
    result = await adapter.exec_command(name, "cat /.exegol/installed_tools.csv")
    return parse_installed_tools(result.stdout)


async def list_installed_tools(
    adapter: ExegolPort, name: str, *, cursor: str | None = None, page_size: int = _DEFAULT_PAGE_SIZE
) -> InstalledToolsPage:
    """Parse /.exegol/installed_tools.csv from a running container into one page of InstalledTool items.

    The whole catalog comes from read_installed_tools. An empty file or a short row yields an empty or partial page
    without raising. CSV order is preserved, two rows with the same tool name but different versions are both kept,
    and the full catalog is sliced into pages by an opaque cursor (EXEC-10, D-23): walking every page reconstructs
    the full unpaged list, in order.

    Args:
        adapter: The Exegol port.
        name: Name of the container, as returned by list_containers.
        cursor: Opaque cursor from a previous page's next_cursor, or None for the first page.
        page_size: Items per page (the tool passes the operator-configured value).

    Returns:
        An InstalledToolsPage: the tools on this page in file order, plus the cursor for the next page (None on
        the last page).

    Raises:
        ValueError: If the container name is unknown, or the cursor is malformed / out of range.
        RuntimeError: If the container is not running.
    """
    tools = await read_installed_tools(adapter, name)
    page, next_cursor = _page(tools, cursor, page_size)
    return InstalledToolsPage(items=page, next_cursor=next_cursor)


async def get_tool_help(adapter: ExegolPort, name: str, tool_name: str) -> ExecResult:
    """Fetch a tool's help text from a running container, trying --help, -h then help.

    tool_name is validated against an ASCII charset and length cap (SEC-07) and then shlex-quoted before it is
    interpolated into the shell command. The first flag that exits 0 wins; if every flag fails the last ExecResult
    is returned as-is, without coercing its exit code (v1 turned a missing code into 1).

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        tool_name: Name of the tool to get help for.

    Returns:
        The ExecResult of the first successful help flag, or of the last attempt when all fail.

    Raises:
        ValueError: If the tool name is empty, over-length or out of charset, or the container name is unknown.
        RuntimeError: If the container is not running.
    """
    # SEC-07/D-07: reject an empty, over-length or out-of-charset tool name before touching the container or
    # building any shell command (mirrors the list_offline_resources ValueError idiom).
    if not tool_name or len(tool_name) > _TOOL_NAME_MAX_LEN or not _TOOL_NAME_RE.match(tool_name):
        raise ValueError(
            f"Invalid tool name {tool_name!r}: expected 1-{_TOOL_NAME_MAX_LEN} characters from letters, digits, "
            "'.', '_', '+' or '-'."
        )
    await require_running(adapter, name)
    quoted = shlex.quote(tool_name)
    last: ExecResult | None = None
    for flag in HELP_FLAGS:
        result = await adapter.exec_command(name, f"{quoted} {flag}")
        if result.exit_code == 0:
            return result
        last = result
    # Every flag failed: return the last result unchanged (no exit-code coercion).
    return last if last is not None else ExecResult(stdout="", stderr="", exit_code=None)


async def list_offline_resources(
    adapter: ExegolPort,
    name: str,
    target_os: str = "linux",
    *,
    cursor: str | None = None,
    page_size: int = _DEFAULT_PAGE_SIZE,
) -> OfflineResourcesPage:
    """List one page of the offline resources under /opt/resources/<target_os> in a running container.

    The full find output is sliced into pages by an opaque cursor (EXEC-10, D-23): find order is preserved and
    walking every page reconstructs the full unpaged list, in order.

    Args:
        adapter: The Exegol port.
        name: Name of the container.
        target_os: OS subdirectory to list, "linux" (default) or "windows".
        cursor: Opaque cursor from a previous page's next_cursor, or None for the first page.
        page_size: Items per page (the tool passes the operator-configured value).

    Returns:
        An OfflineResourcesPage: the resource paths on this page in find order, plus the cursor for the next page
        (None on the last page).

    Raises:
        ValueError: If target_os is invalid, the container name is unknown, or the cursor is malformed.
        RuntimeError: If the container is not running.
    """
    if target_os not in VALID_TARGET_OS:
        raise ValueError(
            f"Invalid target OS '{target_os}'. Only {' and '.join(repr(o) for o in VALID_TARGET_OS)} are supported."
        )
    await require_running(adapter, name)
    # v1 find command, copied verbatim from exegol_utils.read_installed_resources.
    command = (
        f"find /opt/resources/{target_os} -type f | grep -iv '/.git' | "
        "grep -ivE '\\.(md|png|jpg|txt|py|c|cpp|cs|h|sln|vcxproj(.filters)?|xml|csproj|yml|pdb|dll|idl|sys|"
        "chm|desktop|json|pssproj|rst|spec|xsd|config|asm)$' | "
        "grep -ivE '/(LICEN[SC]E|makefile|bootstrap|dockerfile)$'"
    )
    result = await adapter.exec_command(name, command)
    page, next_cursor = _page(result.stdout.splitlines(), cursor, page_size)
    return OfflineResourcesPage(items=page, next_cursor=next_cursor)
