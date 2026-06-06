from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .data import ClbenchTask


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class TaskConstraints:
    quote_required: bool
    required_count: int | None
    count_basis: str | None
    format_required: str | None
    answer_type: str
    must_include: tuple[str, ...]
    forbidden: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_constraints(task: ClbenchTask) -> TaskConstraints:
    text = task.task.lower()
    required_count, count_basis = extract_required_count(text)
    must_include = extract_must_include(text)
    return TaskConstraints(
        quote_required=any(marker in text for marker in ("exact quote", "exact quotes", "quote", "snippet")),
        required_count=required_count,
        count_basis=count_basis,
        format_required=extract_format(text),
        answer_type=extract_answer_type(text),
        must_include=must_include,
        forbidden=extract_forbidden(text),
    )


def extract_required_count(text: str) -> tuple[int | None, str | None]:
    patterns = (
        (r"\bat least\s+(\d+)\b", "at_least"),
        (r"\btop\s+(\d+)\b", "top_n"),
        (r"\bfirst\s+(\d+)\b", "first_n"),
        (r"\b(\d+)\s*[- ]?step\b", "step_count"),
        (r"\b(\d+)\s+(?:examples|incidents|items|reasons|candidates|details|quotes)\b", "explicit_count"),
    )
    for pattern, basis in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), basis

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b(?:at least|top|first)\s+{word}\b", text):
            return value, "word_count"
    return None, None


def extract_format(text: str) -> str | None:
    if "json" in text:
        return "json"
    if "table" in text:
        return "table"
    if "csv" in text:
        return "csv"
    if "bullet" in text or "list" in text:
        return "list"
    return None


def extract_answer_type(text: str) -> str:
    if any(marker in text for marker in ("rank", "top ", "best", "most likely")):
        return "ranked_list"
    if any(marker in text for marker in ("timeline", "chronological")):
        return "timeline"
    if any(marker in text for marker in ("compare", "difference", "not the others")):
        return "comparison"
    if any(marker in text for marker in ("count", "calculate", "total", "percentage")):
        return "calculation"
    return "freeform"


def extract_must_include(text: str) -> tuple[str, ...]:
    wanted: list[str] = []
    for label, markers in (
        ("who", ("who", "which person", "which user")),
        ("why", ("why", "reason", "rationale")),
        ("how", ("how", "method")),
        ("when", ("when", "date", "time")),
        ("evidence", ("evidence", "supporting", "quote", "snippet")),
        ("status", ("status", "complete", "completion")),
    ):
        if any(marker in text for marker in markers):
            wanted.append(label)
    return tuple(dict.fromkeys(wanted))


def extract_forbidden(text: str) -> tuple[str, ...]:
    forbidden: list[str] = []
    if any(marker in text for marker in ("do not invent", "don't invent", "no invented")):
        forbidden.append("invented_facts")
    if "username" in text or "usernames" in text:
        forbidden.append("invented_usernames")
    return tuple(forbidden)


def render_constraints(constraints: TaskConstraints) -> str:
    data = constraints.to_dict()
    return json.dumps(data, ensure_ascii=False, indent=2)


def verify_answer_constraints(
    *,
    task: ClbenchTask,
    answer: str,
    constraints: TaskConstraints,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stripped = answer.strip()
    checks.append({"name": "non_empty_answer", "passed": bool(stripped)})

    if constraints.format_required == "json":
        checks.append({"name": "valid_json", "passed": is_valid_json(stripped)})
    if constraints.format_required == "table":
        checks.append({"name": "markdown_table_present", "passed": "|" in stripped and "---" in stripped})

    if constraints.required_count is not None:
        item_count = estimate_answer_item_count(stripped)
        checks.append(
            {
                "name": "required_count",
                "passed": item_count >= constraints.required_count,
                "observed": item_count,
                "required": constraints.required_count,
                "basis": constraints.count_basis,
            }
        )

    if constraints.quote_required:
        quoted = extract_quoted_strings(stripped)
        missing = [quote for quote in quoted if quote not in task.retrieval_text]
        checks.append(
            {
                "name": "quotes_present",
                "passed": bool(quoted),
                "observed": len(quoted),
            }
        )
        checks.append(
            {
                "name": "quotes_verbatim_in_context",
                "passed": not missing,
                "missing_count": len(missing),
                "missing_preview": missing[:3],
            }
        )

    if "invented_usernames" in constraints.forbidden:
        usernames = sorted(set(re.findall(r"\bu/[A-Za-z0-9_-]+\b", stripped)))
        missing_usernames = [username for username in usernames if username not in task.retrieval_text]
        checks.append(
            {
                "name": "usernames_in_context",
                "passed": not missing_usernames,
                "missing_usernames": missing_usernames[:10],
            }
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except Exception:
        return False
    return True


def extract_quoted_strings(text: str) -> list[str]:
    quoted = re.findall(r'"([^"]{8,})"', text)
    quoted.extend(re.findall(r"'([^']{8,})'", text))
    return [quote.strip() for quote in quoted if quote.strip()]


def estimate_answer_item_count(text: str) -> int:
    if not text:
        return 0
    bullet_lines = [
        line
        for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+\S", line)
    ]
    if bullet_lines:
        return len(bullet_lines)
    table_rows = [
        line
        for line in text.splitlines()
        if line.strip().startswith("|") and "---" not in line
    ]
    if len(table_rows) >= 2:
        return len(table_rows) - 1
    paragraphs = [part for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return len(paragraphs)
