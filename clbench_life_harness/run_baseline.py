from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .chunking import chunk_text
from .data import load_tasks, load_tasks_by_ids, read_ids_file
from .retrieval import retrieve
from .trace import append_jsonl, classify_task, write_task_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Build observable CL-bench Life baseline traces.")
    parser.add_argument("--input", required=True, type=Path, help="Input CL-bench JSONL file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for trace outputs.")
    parser.add_argument("--start", type=int, default=0, help="0-based row offset.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional JSON/text file of original row ids to process.")
    parser.add_argument("--max-chars", type=int, default=3500, help="Maximum characters per retrieval chunk.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of retrieval hits to keep.")
    args = parser.parse_args()

    if args.ids_file:
        tasks = load_tasks_by_ids(args.input, read_ids_file(args.ids_file))
    else:
        tasks = load_tasks(args.input, start=args.start, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    task_type_counts: Counter[str] = Counter()
    chunk_counts: list[int] = []

    for task in tasks:
        chunks = chunk_text(task.retrieval_text, row_id=task.row_id, max_chars=args.max_chars)
        hits = retrieve(chunks, task.task, top_k=args.top_k)
        record = write_task_trace(output_dir=args.output_dir, task=task, chunks=chunks, hits=hits)
        task_type_counts[record["task_type"]] += 1
        chunk_counts.append(len(chunks))
        append_jsonl(
            summary_path,
            {
                "id": task.row_id,
                "task_type": record["task_type"],
                "chars": task.chars,
                "turns": task.turns,
                "rubric_count": len(task.rubrics),
                "chunk_count": len(chunks),
                "top_score": hits[0].score if hits else 0,
                "task_snippet": record["task_snippet"],
            },
        )

    summary = {
        "input": str(args.input),
        "processed": len(tasks),
        "start": args.start,
        "limit": args.limit,
        "task_type_counts": dict(task_type_counts),
        "avg_chunks": round(sum(chunk_counts) / len(chunk_counts), 2) if chunk_counts else 0,
        "max_chunks": max(chunk_counts) if chunk_counts else 0,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
