from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .data import load_tasks
from .trace import classify_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified CL-bench Life dev sample.")
    parser.add_argument("--input", required=True, type=Path, help="Dev JSONL file with metadata/rubrics.")
    parser.add_argument("--ids-output", required=True, type=Path, help="Output JSON file containing selected row ids.")
    parser.add_argument("--summary-output", required=True, type=Path, help="Output JSON summary for the sample.")
    parser.add_argument("--per-subcategory", type=int, default=1, help="Rows to sample from each official subcategory.")
    parser.add_argument("--seed", type=int, default=20260605, help="Random seed.")
    parser.add_argument(
        "--prefer-task-type-diversity",
        action="store_true",
        help="Try to avoid choosing the same harness task type repeatedly within each subcategory.",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.input)
    rng = random.Random(args.seed)
    by_subcategory: dict[str, list] = defaultdict(list)
    for task in tasks:
        subcategory = str(task.metadata.get("context_subcategory", "unknown"))
        by_subcategory[subcategory].append(task)

    selected = []
    for subcategory in sorted(by_subcategory):
        candidates = list(by_subcategory[subcategory])
        rng.shuffle(candidates)
        if args.prefer_task_type_diversity:
            selected.extend(select_diverse(candidates, args.per_subcategory))
        else:
            selected.extend(candidates[: args.per_subcategory])

    selected.sort(key=lambda task: task.row_id)
    ids = [task.row_id for task in selected]
    args.ids_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.ids_output.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input": str(args.input),
        "seed": args.seed,
        "per_subcategory": args.per_subcategory,
        "selected_count": len(selected),
        "selected_ids": ids,
        "items": [
            {
                "id": task.row_id,
                "category": task.metadata.get("context_category"),
                "subcategory": task.metadata.get("context_subcategory"),
                "task_type": classify_task(task.task),
                "chars": task.chars,
                "turns": task.turns,
                "rubric_count": len(task.rubrics),
                "task_snippet": " ".join(task.task.split())[:220],
            }
            for task in selected
        ],
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def select_diverse(candidates: list, count: int) -> list:
    chosen = []
    used_types: set[str] = set()
    for task in candidates:
        task_type = classify_task(task.task)
        if task_type in used_types and len(candidates) - len(chosen) > count - len(chosen):
            continue
        chosen.append(task)
        used_types.add(task_type)
        if len(chosen) == count:
            return chosen

    for task in candidates:
        if task in chosen:
            continue
        chosen.append(task)
        if len(chosen) == count:
            return chosen
    return chosen


if __name__ == "__main__":
    main()
