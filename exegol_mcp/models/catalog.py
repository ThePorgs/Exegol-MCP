"""Paged result models for the catalog listings (EXEC-10, D-23).

``list_installed_tools`` and ``list_offline_resources`` return one page plus an opaque ``next_cursor`` the agent
passes back as ``cursor`` to fetch the next page; ``next_cursor`` is null on the last page. These shapes reach the
LLM (FastMCP derives the tools' output schema from them), so every field carries a description.
"""

from pydantic import BaseModel, Field

from exegol_mcp.models.container import InstalledTool


# EXEC-10
class InstalledToolsPage(BaseModel):
    """One page of installed tools plus the cursor to fetch the next page."""

    items: list[InstalledTool] = Field(description="The installed tools on this page, in catalog (CSV file) order")
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page: pass it back as 'cursor'. Null when this is the last page.",
    )


# EXEC-10
class OfflineResourcesPage(BaseModel):
    """One page of offline-resource paths plus the cursor to fetch the next page."""

    items: list[str] = Field(description="The offline-resource paths on this page, in the find command's order")
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page: pass it back as 'cursor'. Null when this is the last page.",
    )
