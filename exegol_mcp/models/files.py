"""Result models for the container file tools (FILE-01, FILE-02, FILE-03, D-16, D-18, D-19).

These shapes reach the LLM (FastMCP derives the tools' output schema from them), so every field carries a
description. FileContent's encoding field is the published D-16 contract: "utf-8" text or "base64" bytes.
"""

from typing import Literal

from pydantic import BaseModel, Field


# FILE-01, D-16
class FileContent(BaseModel):
    """The whole content of one container file, as UTF-8 text or base64."""

    path: str = Field(description="The in-container path that was read, with symlinks resolved")
    encoding: Literal["utf-8", "base64"] = Field(
        description=(
            'How content is encoded: "utf-8" when the file is valid UTF-8 text (content is that text), '
            '"base64" for anything else or when base64 was asked for (decode content to get the exact bytes)'
        )
    )
    content: str = Field(description="The file content, encoded as the encoding field says. Never redacted or cut")
    size: int = Field(description="The file size in raw bytes (before any base64 encoding)")


# FILE-02, D-18
class FileWritten(BaseModel):
    """The outcome of one verified container file write."""

    path: str = Field(description="The in-container path the bytes landed at (a symlink's target, not the link)")
    size: int = Field(description="The number of raw bytes written, checked against the file afterwards")
    created: bool = Field(description="True when the file did not exist before the write, false when it was replaced")


# FILE-03, D-19
class DirEntry(BaseModel):
    """One entry of a container directory listing."""

    name: str = Field(description="The entry's own name (the last segment of path)")
    path: str = Field(
        description=(
            'The entry\'s path relative to the listed directory ("certs/ca.pem" at depth 2). Entries with the same '
            "name in different subdirectories stay distinct through it"
        )
    )
    type: Literal["file", "dir", "symlink", "other"] = Field(
        description=(
            'The entry type: "file", "dir", "symlink" (reported with its target, never followed or descended into) '
            'or "other" (device, socket, FIFO)'
        )
    )
    size: int = Field(description="Size in bytes as stat reports it (a symlink's own size, not its target's)")
    mode: str = Field(description='Permission bits as a 4-digit octal string, for example "0644"')
    mtime: str = Field(description="Last modification time, ISO-8601 in UTC")
    symlink_target: str | None = Field(
        default=None, description="A symlink's target as stored in the link; null for anything that is not a symlink"
    )


# FILE-03, D-19
class DirectoryPage(BaseModel):
    """One page of a container directory listing plus the cursor to fetch the next page."""

    path: str = Field(description="The in-container path of the listed directory, with a symlinked start resolved")
    items: list[DirEntry] = Field(description="The entries on this page, sorted by their relative path")
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page: pass it back as 'cursor'. Null when this is the last page.",
    )
    truncated: bool = Field(
        default=False,
        description=(
            "True when the tree held more entries than the server's listing cap (files.max_list_entries): only the "
            "first ones are listed. Narrow the path or lower depth to see the rest"
        ),
    )
