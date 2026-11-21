"""Bounded per-stream tail buffer and the shared SDK-free output decoder (EXEC-03, D-20/D-21/D-22).

A command's stdout (and stderr, independently) can be arbitrarily large. Instead of buffering the whole
output and truncating it afterwards, each stream is appended into a TailBuffer that keeps at most `cap` bytes:
on overflow it drops the oldest bytes from the front (security-tool payoff lands at the end) and counts how
many bytes it dropped. Reads run forward from an absolute byte offset with an optional per-read limit, so a
caller can page toward the tail; a read whose offset is behind the retained window starts at the earliest
retained byte and reports the gap it skipped.

The buffer is pure: no event loop, no SDK. It is unit-tested directly.
"""


def decode_output(data: bytes) -> str:
    """Decode command output bytes to text, replacing undecodable bytes with U+FFFD instead of raising.

    A byte cap can land in the middle of a multi-byte UTF-8 character; errors="replace" keeps the decode from
    ever crashing. Whitespace is preserved exactly. This is the one decoder the real adapter, the fake and the
    services share (adapter/exec.py's _decode delegates here).
    """
    return data.decode("utf-8", errors="replace")


class TailBuffer:
    """A bounded byte buffer that keeps the most recent `cap` bytes of one stream.

    Attributes:
        dropped: Total number of bytes dropped from the front so far (also the absolute offset of the earliest
            retained byte). Counts bytes, not characters (EXEC-03 precision edge).
    """

    def __init__(self, cap: int) -> None:
        if cap <= 0:
            raise ValueError(f"TailBuffer cap must be positive, got {cap}")
        self._cap = cap
        self._buf = bytearray()
        # Absolute offset of the earliest byte still retained == total bytes dropped from the front.
        self._start = 0
        # Total bytes ever appended (the absolute offset just past the newest byte).
        self._end = 0

    @property
    def dropped(self) -> int:
        """How many bytes have been dropped from the front (a byte count, never a character count)."""
        return self._start

    @property
    def end(self) -> int:
        """The absolute offset just past the newest byte (total bytes ever appended)."""
        return self._end

    def append(self, chunk: bytes) -> None:
        """Append bytes; keep at most `cap`, dropping the oldest from the front on overflow (D-21)."""
        if not chunk:
            return
        self._buf += chunk
        self._end += len(chunk)
        overflow = len(self._buf) - self._cap
        if overflow > 0:
            del self._buf[:overflow]
            self._start += overflow

    def read(self, offset: int, limit: int | None = None) -> tuple[bytes, int, int]:
        """Return retained bytes from `offset` forward, the next absolute offset, and the skipped-gap count.

        Args:
            offset: Absolute byte offset to read from. An offset behind the retained window starts at the
                earliest retained byte and the returned gap reports how many bytes were skipped (D-22). An
                offset at or beyond the end returns empty data.
            limit: At most this many bytes when given; None reads everything retained from the offset forward.

        Returns:
            (data, next_offset, gap): the bytes read, the absolute offset to pass to the next read, and the
            number of bytes skipped because `offset` was behind the retained window (0 otherwise).
        """
        if offset >= self._end:
            return b"", self._end, 0
        effective = max(offset, self._start)
        gap = self._start - offset if offset < self._start else 0
        buf_index = effective - self._start
        available = len(self._buf) - buf_index
        take = available if limit is None else min(available, max(0, limit))
        data = bytes(self._buf[buf_index : buf_index + take])
        return data, effective + take, gap
