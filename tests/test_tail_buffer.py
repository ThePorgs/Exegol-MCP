"""Bounded per-stream tail buffer (EXEC-03 edges, D-20/D-21/D-22). Pure: no event loop, no SDK."""

from exegol_mcp.adapter.tail_buffer import TailBuffer, decode_output


def test_output_exactly_at_cap_keeps_everything_with_zero_dropped() -> None:
    buf = TailBuffer(4)
    buf.append(b"abcd")
    assert buf.dropped == 0
    data, next_offset, gap = buf.read(0)
    assert data == b"abcd"
    assert next_offset == 4
    assert gap == 0


def test_one_byte_over_the_cap_drops_exactly_one_from_the_front() -> None:
    buf = TailBuffer(4)
    buf.append(b"abcde")
    assert buf.dropped == 1
    data, next_offset, gap = buf.read(0)
    # The oldest byte (a) is gone; the tail is kept.
    assert data == b"bcde"
    assert next_offset == 5
    assert gap == 1


def test_a_large_stream_keeps_only_the_last_cap_bytes_with_the_exact_dropped_count() -> None:
    buf = TailBuffer(8)
    for _ in range(100):
        buf.append(b"0123456789")  # 1000 bytes total
    assert buf.dropped == 1000 - 8
    data, next_offset, _ = buf.read(0)
    assert len(data) == 8
    assert data == b"23456789"  # the last 8 bytes of a stream of repeated 0-9
    assert next_offset == 1000


def test_a_multibyte_utf8_stream_capped_mid_character_decodes_without_raising() -> None:
    # "é" is two bytes (0xc3 0xa9). Dropping the front mid-character leaves the retained window starting on a
    # lone continuation byte; the cap counts bytes, not characters (EXEC-03 precision edge).
    buf = TailBuffer(3)
    buf.append(b"ab\xc3\xa9cd")  # "abécd" in UTF-8 (é is \xc3\xa9), 6 bytes
    assert buf.dropped == 3  # byte count, not character count: the last 3 bytes are kept
    data, _, _ = buf.read(0)
    assert data == b"\xa9cd"
    # decode_output never raises; the half-character becomes the replacement char.
    text = decode_output(data)
    assert isinstance(text, str)
    assert text == "�cd"
    assert "�" in text


def test_a_read_behind_the_window_reports_the_gap_and_starts_at_the_earliest_retained_byte() -> None:
    buf = TailBuffer(4)
    buf.append(b"abcdefgh")  # keeps efgh, dropped 4, start offset 4
    data, next_offset, gap = buf.read(0)
    assert data == b"efgh"
    assert gap == 4  # the reader asked from 0 but 4 bytes are already gone
    assert next_offset == 8


def test_forward_reads_with_next_offset_walk_the_stream_with_no_dup_or_skip() -> None:
    buf = TailBuffer(100)
    buf.append(b"hello world")
    first, next_offset, gap = buf.read(0, limit=5)
    assert first == b"hello"
    assert gap == 0
    assert next_offset == 5
    second, next_offset2, gap2 = buf.read(next_offset)
    assert second == b" world"
    assert gap2 == 0
    assert next_offset2 == 11
    # Reassembled, it is exactly the stream with nothing duplicated or skipped.
    assert first + second == b"hello world"


def test_limit_caps_a_read_and_the_next_read_resumes_at_next_offset() -> None:
    buf = TailBuffer(100)
    buf.append(b"0123456789")
    data, next_offset, _ = buf.read(2, limit=2)
    assert data == b"23"
    assert next_offset == 4
    data2, next_offset2, _ = buf.read(next_offset, limit=3)
    assert data2 == b"456"
    assert next_offset2 == 7


def test_a_read_past_the_end_is_empty() -> None:
    buf = TailBuffer(100)
    buf.append(b"abc")
    data, next_offset, gap = buf.read(10)
    assert data == b""
    assert next_offset == 3  # the end of the stream
    assert gap == 0
