"""Shared opaque-cursor pagination (D-23, D-19): one validated cursor module for every paged listing.

The catalog listings (installed tools, offline resources) and list_directory page through these helpers, so a
malformed, forged, negative or out-of-range cursor is a ValueError everywhere (T-02-04, T-04-40).
"""

import base64
import binascii
from typing import TypeVar

_T = TypeVar("_T")


def encode_cursor(offset: int) -> str:
    """Mint an opaque cursor for a zero-based offset (urlsafe-base64 of the decimal offset)."""
    return base64.urlsafe_b64encode(str(offset).encode("ascii")).decode("ascii")


def decode_cursor(cursor: str) -> int:
    """Decode an opaque cursor back to its offset, raising ValueError on anything malformed or forged.

    The cursor is opaque to the agent and encodes only an offset; it is re-validated against the list length by
    the caller (page), so a tampered-but-well-formed cursor still cannot drive an out-of-range read (T-02-04).
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        return int(raw.decode("ascii"))
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"Invalid pagination cursor '{cursor}'.") from exc


def page(items: list[_T], cursor: str | None, page_size: int) -> tuple[list[_T], str | None]:
    """Slice ``items`` by an opaque ``cursor``, returning ``(page, next_cursor)``.

    A ``None`` cursor starts at offset 0; a valid cursor decodes to the next offset. ``next_cursor`` is the
    encoded ``offset + page_size`` only when more items remain, else ``None`` — so a list whose length is an
    exact multiple of ``page_size`` ends with a final page carrying ``next_cursor`` None and never a trailing
    empty page. A malformed, forged, negative or out-of-range cursor raises ``ValueError`` (which the tool layer
    turns into a ToolError), never an unhandled crash (T-02-04).

    Args:
        items: The full ordered list to page (built by the caller exactly as before).
        cursor: The opaque cursor from a previous page's ``next_cursor``, or None for the first page.
        page_size: Number of items per page; must be a positive integer.

    Returns:
        A tuple of (the items on this page, the cursor to fetch the next page or None on the last page).

    Raises:
        ValueError: If ``page_size`` is not positive, or ``cursor`` is malformed / out of range.
    """
    if page_size <= 0:
        raise ValueError(f"Invalid page size {page_size}: it must be a positive integer.")
    offset = 0 if cursor is None else decode_cursor(cursor)
    if offset < 0 or offset > len(items):
        raise ValueError(f"Invalid pagination cursor: offset {offset} is out of range.")
    sliced = items[offset : offset + page_size]
    next_offset = offset + page_size
    next_cursor = encode_cursor(next_offset) if next_offset < len(items) else None
    return sliced, next_cursor
