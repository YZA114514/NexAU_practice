from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .data import ClbenchTask, load_tasks, task_snippet
from .retrieval import tokenize
from .task_router import route_task
from .trace import classify_task

LENGTH_BUCKETS = (
    ("0-30k", 0, 30_000),
    ("30k-60k", 30_000, 60_000),
    ("60k-120k", 60_000, 120_000),
    ("120k-300k", 120_000, 300_000),
    ("300k+", 300_000, 10**12),
)

FORMAT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("watch_history", ("watch history", "youtube watch", "video title", "watched videos")),
    ("music_history", ("spotify", "last.fm", "music listening", "playlist", "album history")),
    ("apple_health", ("apple health", "health app", "step count", "daily steps", "heart rate")),
    ("finance", ("bank statement", "checking account", "credit card", "paypal", "venmo", "transaction log", "receipt")),
    ("chat_messages", ("text message", "message thread", "group chat", "dm thread", "chat transcript")),
    ("email", ("email thread", "inbox", "subject:", "from:", "to:")),
    ("game_logs", ("game log", "session log", "match log", "level log", "xp log", "score log")),
    ("poker", ("poker", "big blind", "small blind", "flop", "river", "showdown", "winnings")),
    ("tabletop_rpg", ("d&d", "dnd", "session notes", "campaign notes", "dm notes")),
    ("revision_history", ("revision history", "draft history", "outline", "editor comments", "feedback notes")),
    ("calendar", ("calendar", "appointment", "meeting transcript", "scheduled for")),
    ("shopping", ("order history", "wishlist", "shopping cart", "purchase history", "shopping list")),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CL-bench Life dev/test distributions and test-similar dev items.")
    parser.add_argument("--dev-input", required=True, type=Path, help="Dev JSONL file with rubrics/metadata.")
    parser.add_argument("--test-input", required=True, type=Path, help="Test JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON analysis path.")
    parser.add_argument("--top-similar", type=int, default=5, help="Dev candidates to list for each test item.")
    parser.add_argument("--test-driven-dev-count", type=int, default=45, help="Unique dev ids to recommend from test similarity.")
    args = parser.parse_args()

    dev_tasks = load_tasks(args.dev_input)
    test_tasks = load_tasks(args.test_input)
    dev_profiles = [build_profile(task) for task in dev_tasks]
    dev_profile_by_id = {int(profile["id"]): profile for profile in dev_profiles}
    test_profiles = [build_profile(task) for task in test_tasks]

    test_items = []
    candidate_votes: Counter[int] = Counter()
    for test_task, test_profile in zip(test_tasks, test_profiles, strict=True):
        similar = most_similar_dev(test_profile, dev_profiles, top_k=args.top_similar)
        for rank, item in enumerate(similar, start=1):
            candidate_votes[int(item["id"])] += max(args.top_similar - rank + 1, 1)
        test_items.append(
            {
                "test_id": test_task.row_id,
                "task_type": test_profile["task_type"],
                "chars": test_profile["chars"],
                "turns": test_profile["turns"],
                "length_bucket": test_profile["length_bucket"],
                "format_hints": sorted(test_profile["format_hints"]),
                "task_snippet": task_snippet(test_task.task, max_chars=320),
                "nearest_dev": similar,
            }
        )

    analysis = {
        "dev_input": str(args.dev_input),
        "test_input": str(args.test_input),
        "output": str(args.output),
        "dev_summary": summarize_profiles(dev_profiles),
        "test_summary": summarize_profiles(test_profiles),
        "test_items": test_items,
        "test_similarity_dev_set": [
            {"id": row_id, "vote_score": score, **dev_profile_by_id[row_id]["public"]}
            for row_id, score in candidate_votes.most_common(args.test_driven_dev_count)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(short_console_summary(analysis), ensure_ascii=False, indent=2))


def build_profile(task: ClbenchTask) -> dict[str, Any]:
    context_probe = (task.task + "\n" + task.retrieval_text[:80_000] + "\n" + task.retrieval_text[-20_000:]).lower()
    hints = {name for name, markers in FORMAT_HINTS if any(marker in context_probe for marker in markers)}
    tokens = set(tokenize(task.task))
    subcategory = str(task.metadata.get("context_subcategory", "unknown"))
    route = route_task(task)
    public = {
        "task_type": classify_task(task.task),
        "route": route.route,
        "route_confidence": route.confidence,
        "route_signals": list(route.signals),
        "subcategory": subcategory,
        "chars": task.chars,
        "turns": task.turns,
        "length_bucket": length_bucket(task.chars),
        "format_hints": sorted(hints),
        "task_snippet": task_snippet(task.task, max_chars=220),
    }
    return {
        "id": task.row_id,
        "task": task,
        "task_type": public["task_type"],
        "route": public["route"],
        "subcategory": subcategory,
        "chars": task.chars,
        "turns": task.turns,
        "length_bucket": public["length_bucket"],
        "format_hints": hints,
        "task_tokens": tokens,
        "public": public,
    }


def summarize_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    chars = [int(profile["chars"]) for profile in profiles]
    return {
        "count": len(profiles),
        "chars": numeric_summary(chars),
        "turns": dict(sorted(Counter(int(profile["turns"]) for profile in profiles).items())),
        "length_buckets": dict(Counter(str(profile["length_bucket"]) for profile in profiles).most_common()),
        "task_types": dict(Counter(str(profile["task_type"]) for profile in profiles).most_common()),
        "routes": dict(Counter(str(profile["route"]) for profile in profiles).most_common()),
        "subcategories": dict(Counter(str(profile["subcategory"]) for profile in profiles).most_common()),
        "format_hints": dict(Counter(hint for profile in profiles for hint in profile["format_hints"]).most_common()),
    }


def numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p25": percentile(sorted_values, 0.25),
        "median": percentile(sorted_values, 0.5),
        "mean": round(sum(sorted_values) / len(sorted_values), 1),
        "p75": percentile(sorted_values, 0.75),
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[int], ratio: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = index - lower
    return round(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction, 1)


def length_bucket(chars: int) -> str:
    for label, start, end in LENGTH_BUCKETS:
        if start <= chars < end:
            return label
    return "unknown"


def most_similar_dev(
    test_profile: dict[str, Any],
    dev_profiles: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    scored = []
    for dev_profile in dev_profiles:
        score = similarity_score(test_profile, dev_profile)
        scored.append((score, dev_profile))
    scored.sort(key=lambda item: (item[0], -int(item[1]["id"])), reverse=True)
    return [
        {
            "id": int(profile["id"]),
            "score": round(score, 3),
            **profile["public"],
        }
        for score, profile in scored[:top_k]
    ]


def similarity_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    score = 0.0
    if a["task_type"] == b["task_type"]:
        score += 5.0
    if a["length_bucket"] == b["length_bucket"]:
        score += 2.0
    if a["turns"] == b["turns"]:
        score += 1.0

    shared_hints = set(a["format_hints"]) & set(b["format_hints"])
    score += 2.0 * len(shared_hints)
    score += token_similarity(set(a["task_tokens"]), set(b["task_tokens"])) * 4.0
    score += length_similarity(int(a["chars"]), int(b["chars"])) * 2.0
    return score


def token_similarity(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union else 0.0


def length_similarity(a_chars: int, b_chars: int) -> float:
    if a_chars <= 0 or b_chars <= 0:
        return 0.0
    ratio = max(a_chars, b_chars) / max(min(a_chars, b_chars), 1)
    return max(0.0, 1.0 - math.log(ratio, 4) / 3.0)


def short_console_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "dev_summary": analysis["dev_summary"],
        "test_summary": analysis["test_summary"],
        "test_similarity_dev_ids": [item["id"] for item in analysis["test_similarity_dev_set"]],
        "output": analysis.get("output"),
    }


if __name__ == "__main__":
    main()
