from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .constraints import TaskConstraints, estimate_answer_item_count, extract_quoted_strings, is_valid_json
from .data import ClbenchTask
from .workflows import StructuredState


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    severity: str = "warning"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_answer_hard(
    *,
    task: ClbenchTask,
    answer: str,
    constraints: TaskConstraints,
    structured_state: StructuredState,
    truncation: dict[str, Any],
) -> dict[str, Any]:
    checks: list[VerificationCheck] = []
    stripped = answer.strip()

    checks.append(VerificationCheck("non_empty_answer", bool(stripped), "error"))
    checks.extend(check_format(stripped, constraints))
    checks.extend(check_required_count(stripped, constraints))
    checks.extend(check_quotes(task, stripped, constraints, structured_state))
    checks.extend(check_usernames(task, stripped, constraints, structured_state))
    checks.extend(check_numbers(stripped, structured_state))
    checks.extend(check_workflow_state(structured_state))
    checks.extend(check_workflow_rubric_risks(task, stripped, structured_state))
    checks.extend(check_truncation(truncation))

    failure_types = attribute_failures(checks=checks, structured_state=structured_state)
    return {
        "passed": all(check.passed or check.severity != "error" for check in checks),
        "error_count": sum(1 for check in checks if not check.passed and check.severity == "error"),
        "warning_count": sum(1 for check in checks if not check.passed and check.severity == "warning"),
        "checks": [check.to_dict() for check in checks],
        "failure_type_hints": failure_types,
    }


def check_format(answer: str, constraints: TaskConstraints) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    if constraints.format_required == "json":
        checks.append(VerificationCheck("valid_json", is_valid_json(answer), "error"))
    elif constraints.format_required == "table":
        has_table = "|" in answer and "---" in answer
        checks.append(VerificationCheck("markdown_table_present", has_table, "warning"))
    elif constraints.format_required == "list":
        item_count = estimate_answer_item_count(answer)
        checks.append(
            VerificationCheck(
                "list_items_present",
                item_count > 0,
                "warning",
                {"observed_items": item_count},
            )
        )
    return checks


def check_required_count(answer: str, constraints: TaskConstraints) -> list[VerificationCheck]:
    if constraints.required_count is None:
        return []
    observed = estimate_answer_item_count(answer)
    return [
        VerificationCheck(
            "required_count_met",
            observed >= constraints.required_count,
            "warning",
            {
                "observed": observed,
                "required": constraints.required_count,
                "basis": constraints.count_basis,
            },
        )
    ]


def check_quotes(
    task: ClbenchTask,
    answer: str,
    constraints: TaskConstraints,
    structured_state: StructuredState,
) -> list[VerificationCheck]:
    if not constraints.quote_required:
        return []
    quoted = extract_quoted_strings(answer)
    context = task.retrieval_text
    evidence_quotes = [row.quote for row in structured_state.evidence_table if row.quote]
    missing = [
        quote
        for quote in quoted
        if quote not in context and quote not in evidence_quotes
    ]
    return [
        VerificationCheck("quotes_present", bool(quoted), "error", {"observed": len(quoted)}),
        VerificationCheck(
            "quotes_verbatim_in_context_or_state",
            not missing,
            "error",
            {"missing_count": len(missing), "missing_preview": missing[:3]},
        ),
    ]


def check_usernames(
    task: ClbenchTask,
    answer: str,
    constraints: TaskConstraints,
    structured_state: StructuredState,
) -> list[VerificationCheck]:
    if "invented_usernames" not in constraints.forbidden and "username" not in task.task.lower():
        return []
    usernames = sorted(extract_username_like_strings(answer))
    if not usernames:
        return [VerificationCheck("username_check_not_applicable", True, "info")]
    known = known_names_from_state(structured_state) | extract_username_like_strings(task.retrieval_text)
    missing = [username for username in usernames if username not in known and username not in task.retrieval_text]
    return [
        VerificationCheck(
            "usernames_grounded",
            not missing,
            "error",
            {"observed": usernames[:20], "missing": missing[:20]},
        )
    ]


def check_numbers(answer: str, structured_state: StructuredState) -> list[VerificationCheck]:
    answer_numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", answer)))
    if not answer_numbers:
        return []
    state_text = json.dumps(structured_state.to_dict(), ensure_ascii=False)
    computed_available = bool(structured_state.computed_values or structured_state.tables)
    unsupported = [
        number
        for number in answer_numbers
        if number not in state_text and len(number) > 2
    ]
    return [
        VerificationCheck(
            "numbers_have_structured_context",
            computed_available,
            "warning",
            {"answer_numbers": answer_numbers[:30]},
        ),
        VerificationCheck(
            "large_numbers_seen_in_state",
            not unsupported,
            "warning",
            {"unsupported_large_numbers": unsupported[:20]},
        ),
    ]


def check_workflow_state(structured_state: StructuredState) -> list[VerificationCheck]:
    checks = [
        VerificationCheck(
            "workflow_selected",
            structured_state.workflow != "general_evidence_workflow",
            "warning",
            {"workflow": structured_state.workflow},
        ),
        VerificationCheck(
            "evidence_or_candidate_state_present",
            bool(structured_state.evidence_table or structured_state.candidate_answers or structured_state.computed_values),
            "error",
            {
                "evidence_rows": len(structured_state.evidence_table),
                "candidate_answers": len(structured_state.candidate_answers),
                "computed_values": len(structured_state.computed_values),
            },
        ),
    ]
    if structured_state.warnings:
        checks.append(
            VerificationCheck(
                "workflow_warnings_present",
                False,
                "warning",
                {"warnings": structured_state.warnings[:10]},
            )
        )
    return checks


def check_workflow_rubric_risks(
    task: ClbenchTask,
    answer: str,
    structured_state: StructuredState,
) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    task_text = task.task.lower()
    answer_lower = answer.lower()

    speaker_activity = structured_state.tables.get("speaker_activity", [])
    if speaker_activity and any(marker in task_text for marker in ("most active", "top ", "message count", "number of messages")):
        expected_rows = speaker_activity[:10]
        lower_ranked_rows = speaker_activity[10:20]
        required_speakers = [str(row.get("speaker")) for row in expected_rows if row.get("speaker")]
        missing = [speaker for speaker in required_speakers if speaker.lower() not in answer_lower]
        extra_active_like = [
            str(row.get("speaker"))
            for row in lower_ranked_rows
            if row.get("speaker") and str(row.get("speaker")).lower() in answer_lower
        ]
        checks.append(
            VerificationCheck(
                "top_speakers_covered",
                not missing and not extra_active_like,
                "warning",
                {
                    "expected_top_speakers": required_speakers,
                    "missing": missing,
                    "lower_ranked_speakers_present": extra_active_like,
                    "source_table": "speaker_activity",
                },
            )
        )

    sentiment_evidence = structured_state.tables.get("speaker_sentiment_evidence", [])
    if sentiment_evidence and any(marker in task_text for marker in ("sentiment", "positive", "negative", "rating")):
        avoid_nei = [
            str(row.get("speaker"))
            for row in sentiment_evidence[:20]
            if row.get("speaker")
            and bool(row.get("has_enough_evidence_for_rating", True))
            and answer_row_has_nei_for_label(answer, str(row.get("speaker")))
        ]
        checks.append(
            VerificationCheck(
                "sentiment_evidence_not_nei",
                not avoid_nei,
                "warning",
                {"speakers_with_evidence_given_nei": avoid_nei},
            )
        )

    commenter_registry = structured_state.tables.get("commenter_registry", [])
    if commenter_registry and any(marker in task_text for marker in ("speaker registry", "commenter", "post count", "first appearance")):
        required_commenters = [str(row.get("commenter")) for row in commenter_registry[:20] if row.get("commenter")]
        missing = [name for name in required_commenters if name.lower() not in answer_lower]
        checks.append(
            VerificationCheck(
                "commenter_registry_covered",
                not missing,
                "warning",
                {"expected_commenters": required_commenters, "missing": missing[:20]},
            )
        )

    planned_coverage = structured_state.tables.get("llm_query_coverage", [])
    if planned_coverage:
        required_items = [
            str(row.get("name"))
            for row in planned_coverage[:30]
            if row.get("name") and bool(row.get("required", True))
        ]
        missing_planned = [item for item in required_items if not loose_label_mentioned(item, answer_lower)]
        checks.append(
            VerificationCheck(
                "planned_coverage_covered",
                not missing_planned,
                "warning",
                {"expected": required_items, "missing": missing_planned[:20]},
            )
        )

    recurring_items = structured_state.tables.get("recurring_items", [])
    if recurring_items and any(marker in task_text for marker in ("calendar", "subscribe", "delivery", "next 6 months")):
        planning_calendar = structured_state.tables.get("planning_calendar", [])
        month_names = tuple(str(row.get("month", "")).lower() for row in planning_calendar if row.get("month"))
        if not month_names:
            month_names = ("april", "may", "june", "july", "august", "september")
        missing_months = [month for month in month_names if month not in answer_lower]
        cadence_plan = structured_state.tables.get("subscription_cadence_plan", [])
        if cadence_plan:
            important_items = [str(row.get("display_name")) for row in cadence_plan[:16] if row.get("display_name")]
            missing_items = [
                item
                for item in important_items
                if not loose_label_mentioned(item, answer_lower)
            ]
        else:
            important_items = [
                str(row.get("item_key"))
                for row in recurring_items
                if str(row.get("cadence_hint", "")).endswith(("4_weeks", "6_to_8_weeks", "12_weeks_or_more"))
            ][:12]
            missing_items = [item for item in important_items if item.replace("_", " ") not in answer_lower and item not in answer_lower]
        checks.append(
            VerificationCheck(
                "delivery_calendar_months_present",
                not missing_months,
                "warning",
                {"missing_months": missing_months},
            )
        )
        checks.append(
            VerificationCheck(
                "recurring_items_covered",
                not missing_items,
                "warning",
                {"expected_item_keys": important_items, "missing": missing_items},
            )
        )

    canonical_items = structured_state.tables.get("canonical_item_matrix", [])
    if canonical_items and any(marker in task_text for marker in ("exactly", "master recipe list", "table of contents", "outline")):
        non_tentative = [row for row in canonical_items if not row.get("tentative")]
        checks.append(
            VerificationCheck(
                "canonical_item_matrix_available_for_exact_list",
                bool(non_tentative),
                "warning",
                {"canonical_item_count": len(non_tentative)},
            )
        )
    return checks


def loose_label_mentioned(label: str, answer_lower: str) -> bool:
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", label.lower())
        if len(token) >= 4 and token not in {"count", "pack", "bottle", "container", "with", "free"}
    ]
    if not tokens:
        return label.lower() in answer_lower
    required = tokens[:2] if len(tokens) <= 3 else tokens[:3]
    return all(token in answer_lower for token in required)


def answer_row_has_nei_for_label(answer: str, label: str) -> bool:
    label_lower = label.lower()
    for line in answer.splitlines():
        if "|" not in line:
            continue
        lowered = line.lower()
        if label_lower in lowered and re.search(r"\bnei\b", lowered):
            return True
    return False


def check_truncation(truncation: dict[str, Any]) -> list[VerificationCheck]:
    suspected = bool(truncation.get("suspected"))
    return [
        VerificationCheck(
            "not_suspected_truncated",
            not suspected,
            "error",
            {"reasons": truncation.get("reasons", [])},
        )
    ]


def attribute_failures(*, checks: list[VerificationCheck], structured_state: StructuredState) -> list[str]:
    hints: list[str] = []
    failed_names = {check.name for check in checks if not check.passed}
    if "evidence_or_candidate_state_present" in failed_names:
        hints.append("context_location")
    if (
        "quotes_present" in failed_names
        or "quotes_verbatim_in_context_or_state" in failed_names
        or "usernames_grounded" in failed_names
        or "planned_coverage_covered" in failed_names
    ):
        hints.append("fact_extraction")
    if "required_count_met" in failed_names or "large_numbers_seen_in_state" in failed_names:
        hints.append("calculation")
    if "valid_json" in failed_names or "markdown_table_present" in failed_names or "list_items_present" in failed_names:
        hints.append("format")
    if any(
        name in failed_names
        for name in (
            "top_speakers_covered",
            "commenter_registry_covered",
            "delivery_calendar_months_present",
            "recurring_items_covered",
            "canonical_item_matrix_available_for_exact_list",
            "sentiment_evidence_not_nei",
        )
    ):
        hints.append("final_validation")
    if "not_suspected_truncated" in failed_names or "non_empty_answer" in failed_names:
        hints.append("final_generation")
    if structured_state.warnings:
        hints.append("workflow_state")
    return list(dict.fromkeys(hints))


def extract_username_like_strings(text: str) -> set[str]:
    usernames = set(re.findall(r"\bu/[A-Za-z0-9_-]+\b", text))
    for parenthetical in re.findall(r"\(([^)]{2,300})\)", text):
        for part in parenthetical.split(","):
            value = part.strip()
            if re.match(r"^[A-Za-z][A-Za-z0-9_ .'-]{1,60}$", value):
                usernames.add(value)
    common_words = {
        "Top",
        "All",
        "Green",
        "Red",
        "Grey",
        "Blue",
        "Pink",
        "Knight",
        "Necromancer",
        "Blacksmith",
        "Industrialist",
        "Fencer",
        "The",
        "This",
        "That",
        "Evidence",
        "Answer",
    }
    return {username for username in usernames if username not in common_words}


def known_names_from_state(structured_state: StructuredState) -> set[str]:
    names: set[str] = set(structured_state.entity_profiles)
    for table in structured_state.tables.values():
        for row in table:
            for key in ("user", "speaker", "actor", "entity", "commenter"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
    return names
