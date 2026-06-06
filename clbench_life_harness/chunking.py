from __future__ import annotations

import re
from dataclasses import dataclass

SPEAKER_TIME_RE = re.compile(
    r"^(?P<speaker>.{1,90}?)\s+[—-]\s+(?P<time>\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s+[AP]M)",
)
DOCUMENT_RE = re.compile(r"^\s*(?P<label>(document|doc|outline|table|chapter|email|hand)\s+\w+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    start_line: int
    end_line: int
    char_start: int
    char_end: int
    source_label: str | None
    speakers: tuple[str, ...]
    timestamps: tuple[str, ...]

    @property
    def line_span(self) -> str:
        return f"{self.start_line}-{self.end_line}"


def chunk_text(
    text: str,
    *,
    row_id: int,
    max_chars: int = 3500,
    overlap_lines: int = 6,
    overlap_char_budget: int = 700,
) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    current: list[tuple[int, str, int, int]] = []
    current_chars = 0
    char_cursor = 0
    current_source: str | None = None

    for line_no, line in enumerate(lines, start=1):
        line_start = char_cursor
        line_end = char_cursor + len(line)
        char_cursor = line_end + 1

        source_match = DOCUMENT_RE.match(line)
        starts_new_source = source_match is not None and current_chars > 0
        exceeds_size = current_chars + len(line) + 1 > max_chars and current_chars > 0
        if starts_new_source or exceeds_size:
            chunks.append(_make_chunk(row_id=row_id, index=len(chunks), entries=current, source_label=current_source))
            current = _tail_overlap(current, overlap_lines=overlap_lines, overlap_char_budget=overlap_char_budget)
            current_chars = sum(len(item[1]) + 1 for item in current)

        if source_match:
            current_source = source_match.group("label").strip()

        current.append((line_no, line, line_start, line_end))
        current_chars += len(line) + 1

    if current:
        chunks.append(_make_chunk(row_id=row_id, index=len(chunks), entries=current, source_label=current_source))
    return chunks


def _tail_overlap(
    entries: list[tuple[int, str, int, int]],
    *,
    overlap_lines: int,
    overlap_char_budget: int,
) -> list[tuple[int, str, int, int]]:
    if overlap_lines <= 0 or overlap_char_budget <= 0:
        return []

    kept: list[tuple[int, str, int, int]] = []
    char_count = 0
    for entry in reversed(entries[-overlap_lines:]):
        entry_chars = len(entry[1]) + 1
        if kept and char_count + entry_chars > overlap_char_budget:
            break
        if not kept and entry_chars > overlap_char_budget:
            kept.append(entry)
            break
        kept.append(entry)
        char_count += entry_chars
    kept.reverse()
    return kept


def _make_chunk(
    *,
    row_id: int,
    index: int,
    entries: list[tuple[int, str, int, int]],
    source_label: str | None,
) -> Chunk:
    start_line = entries[0][0]
    end_line = entries[-1][0]
    char_start = entries[0][2]
    char_end = entries[-1][3]
    text = "\n".join(entry[1] for entry in entries).strip()

    speakers: list[str] = []
    timestamps: list[str] = []
    for _, line, _, _ in entries:
        match = SPEAKER_TIME_RE.match(line)
        if match:
            speaker = match.group("speaker").strip()
            timestamp = match.group("time").strip()
            if speaker not in speakers:
                speakers.append(speaker)
            if timestamp not in timestamps:
                timestamps.append(timestamp)

    return Chunk(
        chunk_id=f"{row_id:04d}-{index:04d}",
        text=text,
        start_line=start_line,
        end_line=end_line,
        char_start=char_start,
        char_end=char_end,
        source_label=source_label,
        speakers=tuple(speakers[:12]),
        timestamps=tuple(timestamps[:12]),
    )
