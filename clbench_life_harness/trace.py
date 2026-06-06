from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .data import ClbenchTask, task_snippet
from .deterministic import build_deterministic_findings
from .retrieval import RetrievalHit


def classify_task(task: str) -> str:
    lowered = task.lower()
    if any(marker in lowered for marker in ["exact quote", "exact supporting", "snippet", "quote"]):
        return "quote_or_evidence"
    if any(marker in lowered for marker in ["count", "percentage", "total", "how many", "number of"]):
        return "count_or_calculation"
    if any(marker in lowered for marker in ["timeline", "chronological", "date", "when", "status"]):
        return "timeline_or_status"
    if any(marker in lowered for marker in ["compare", "only appear", "not the others", "difference"]):
        return "comparison"
    if any(marker in lowered for marker in ["contradict", "plot hole", "inconsistent"]):
        return "contradiction_detection"
    if any(marker in lowered for marker in ["recommend", "should i", "best candidates", "design"]):
        return "recommendation"
    if any(marker in lowered for marker in ["summarize", "outline", "rundown", "analyze"]):
        return "synthesis"
    return "general_context_learning"


def write_task_trace(
    *,
    output_dir: Path,
    task: ClbenchTask,
    chunks: list[Chunk],
    hits: list[RetrievalHit],
) -> dict[str, Any]:
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    record = build_trace_record(task=task, chunks=chunks, hits=hits)
    trace_path = trace_dir / f"task_{task.row_id:04d}.json"
    trace_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_trace_record(*, task: ClbenchTask, chunks: list[Chunk], hits: list[RetrievalHit]) -> dict[str, Any]:
    deterministic_findings = build_deterministic_findings(task)
    return {
        "id": task.row_id,
        "task_type": classify_task(task.task),
        "task": task.task,
        "task_snippet": task_snippet(task.task),
        "metadata": task.metadata,
        "observability": {
            "turns": task.turns,
            "chars": task.chars,
            "context_chars": len(task.context),
            "retrieval_text_chars": len(task.retrieval_text),
            "rubric_count": len(task.rubrics),
            "chunk_count": len(chunks),
            "top_hit_count": len(hits),
        },
        "rubrics_sample": task.rubrics[:8],
        "deterministic_findings": [
            {"kind": finding.kind, "summary": finding.summary, "details": finding.details}
            for finding in deterministic_findings
        ],
        "top_hits": [
            {
                "rank": index + 1,
                "score": hit.score,
                "matched_terms": hit.matched_terms,
                "chunk": asdict(hit.chunk) | {"preview": _preview(hit.chunk.text)},
            }
            for index, hit in enumerate(hits)
        ],
    }


def _preview(text: str, *, max_chars: int = 900) -> str:
    compact = "\n".join(line.rstrip() for line in text.splitlines())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."
