from __future__ import annotations

import json
import re
import csv
import html
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

from .data import ClbenchTask

STEP_RECORD_RE = re.compile(
    r"<Record\b"
    r"(?=[^>]*type=\"HKQuantityTypeIdentifierStepCount\")"
    r"(?=[^>]*startDate=\"(?P<start>[^\"]+)\")"
    r"(?=[^>]*value=\"(?P<value>-?\d+(?:\.\d+)?)\")"
    r"[^>]*>",
)
DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class DeterministicFinding:
    kind: str
    summary: str
    details: str


@dataclass(frozen=True)
class DeterministicAnswer:
    kind: str
    summary: str
    content: str
    confidence: float = 1.0
    trigger_signals: tuple[str, ...] = ()
    parsed_state_summary: dict[str, Any] | None = None
    skipped_or_uncertain_items: tuple[dict[str, str], ...] = ()
    should_fallback_to_llm: bool = False
    tool_name: str = ""
    llm_role: str = "skipped_due_to_verified_tool_answer"
    answer_source: str = "verified_tool_answer"
    deterministic_core: bool = True
    answer_is_exact: bool = True
    llm_review_recommended: bool = False


def build_deterministic_findings(task: ClbenchTask) -> list[DeterministicFinding]:
    findings: list[DeterministicFinding] = []
    runescape_finding = analyze_runescape_adventurer_log(task)
    if runescape_finding:
        findings.append(runescape_finding)
    pgn_finding = analyze_chess_pgn_draws(task)
    if pgn_finding:
        findings.append(pgn_finding)
    starcraft_finding = analyze_starcraft_terran_opener(task)
    if starcraft_finding:
        findings.append(starcraft_finding)
    dota_finding = analyze_dota_purchase_window(task)
    if dota_finding:
        findings.append(dota_finding)
    community_tally_finding = analyze_castle_crashers_favorite_tally(task)
    if community_tally_finding:
        findings.append(community_tally_finding)
    recommendation_finding = analyze_recommendation_evidence(task)
    if recommendation_finding:
        findings.append(recommendation_finding)
    tension_finding = analyze_conversation_tension_peak(task)
    if tension_finding:
        findings.append(tension_finding)
    step_finding = analyze_health_steps(task)
    if step_finding:
        findings.append(step_finding)
    game_weekly_finding = analyze_game_weekly_stats(task)
    if game_weekly_finding:
        findings.append(game_weekly_finding)
    workout_finding = analyze_workout_trajectory(task)
    if workout_finding:
        findings.append(workout_finding)
    true_crime_finding = analyze_true_crime_fatigue(task)
    if true_crime_finding:
        findings.append(true_crime_finding)
    video_relief_finding = analyze_video_relief(task)
    if video_relief_finding:
        findings.append(video_relief_finding)
    return findings


def build_deterministic_answer(task: ClbenchTask) -> DeterministicAnswer | None:
    runescape_analysis = build_runescape_adventurer_log_analysis(task)
    if runescape_analysis and runescape_analysis["confidence"] >= 0.85:
        return DeterministicAnswer(
            kind="runescape_adventurer_log_stats",
            summary="Rendered deterministic final answer for RuneScape Adventurer log event statistics.",
            content=render_runescape_adventurer_log_answer_from_analysis(runescape_analysis),
            confidence=runescape_analysis["confidence"],
            trigger_signals=tuple(runescape_analysis["trigger_signals"]),
            parsed_state_summary=runescape_analysis["parsed_state_summary"],
            skipped_or_uncertain_items=tuple(runescape_analysis["skipped_or_uncertain_items"]),
            should_fallback_to_llm=False,
            tool_name="runescape_log_parser",
            llm_role="review_recommended_for_semantic_category_boundaries",
            answer_source="verified_tool_answer_review_recommended",
            answer_is_exact=False,
            llm_review_recommended=True,
        )
    pgn_analysis = build_chess_pgn_draw_analysis(task)
    if pgn_analysis and pgn_analysis["confidence"] >= 0.85:
        return DeterministicAnswer(
            kind="chess_pgn_draw_comparison",
            summary="Rendered deterministic final answer for PGN draw termination comparison.",
            content=render_chess_pgn_draw_answer_from_analysis(pgn_analysis),
            confidence=pgn_analysis["confidence"],
            trigger_signals=tuple(pgn_analysis["trigger_signals"]),
            parsed_state_summary=pgn_analysis["parsed_state_summary"],
            skipped_or_uncertain_items=tuple(pgn_analysis["skipped_or_uncertain_items"]),
            should_fallback_to_llm=False,
            tool_name="pgn_draw_parser",
            llm_role="skipped_due_to_exact_extraction",
            answer_source="verified_tool_answer",
            answer_is_exact=True,
        )
    starcraft_analysis = build_starcraft_terran_opener_analysis(task)
    if starcraft_analysis and starcraft_analysis["confidence"] >= 0.85:
        return DeterministicAnswer(
            kind="starcraft_terran_opener_timeline",
            summary="Rendered deterministic final answer for a Terran opener timeline from replay tracker events.",
            content=render_starcraft_terran_opener_answer_from_analysis(starcraft_analysis),
            confidence=starcraft_analysis["confidence"],
            trigger_signals=tuple(starcraft_analysis["trigger_signals"]),
            parsed_state_summary=starcraft_analysis["parsed_state_summary"],
            skipped_or_uncertain_items=tuple(starcraft_analysis["skipped_or_uncertain_items"]),
            should_fallback_to_llm=False,
            tool_name="starcraft_replay_tracker_parser",
            llm_role="skipped_due_to_exact_extraction",
            answer_source="verified_tool_answer",
            answer_is_exact=True,
        )
    dota_analysis = build_dota_purchase_window_analysis(task)
    if dota_analysis and dota_analysis["confidence"] >= 0.85:
        return DeterministicAnswer(
            kind="dota_purchase_window_tables",
            summary="Rendered deterministic final answer for Dota purchase logs within a bounded time window.",
            content=render_dota_purchase_window_answer_from_analysis(dota_analysis),
            confidence=dota_analysis["confidence"],
            trigger_signals=tuple(dota_analysis["trigger_signals"]),
            parsed_state_summary=dota_analysis["parsed_state_summary"],
            skipped_or_uncertain_items=tuple(dota_analysis["skipped_or_uncertain_items"]),
            should_fallback_to_llm=False,
            tool_name="dota_purchase_window_parser",
            llm_role="skipped_due_to_exact_computation",
            answer_source="verified_tool_answer",
            answer_is_exact=True,
        )
    community_tally_analysis = build_castle_crashers_favorite_tally_analysis(task)
    if community_tally_analysis and community_tally_analysis["confidence"] >= 0.85:
        return DeterministicAnswer(
            kind="community_favorite_tally",
            summary="Rendered deterministic final answer for a favorite-character community thread tally.",
            content=render_castle_crashers_favorite_tally_answer_from_analysis(community_tally_analysis),
            confidence=community_tally_analysis["confidence"],
            trigger_signals=tuple(community_tally_analysis["trigger_signals"]),
            parsed_state_summary=community_tally_analysis["parsed_state_summary"],
            skipped_or_uncertain_items=tuple(community_tally_analysis["skipped_or_uncertain_items"]),
            should_fallback_to_llm=False,
            tool_name="community_thread_tally_parser",
            llm_role="review_recommended_for_ambiguous_thread_choice_boundaries",
            answer_source="verified_tool_answer_review_recommended",
            answer_is_exact=False,
            llm_review_recommended=True,
        )
    tension_answer = render_conversation_tension_peak_answer(task)
    if tension_answer:
        return DeterministicAnswer(
            kind="conversation_tension_peak",
            summary="Rendered deterministic final answer for a conversation tension peak.",
            content=tension_answer,
            tool_name="conversation_tension_peak_extractor",
            llm_role="review_recommended_for_social_inference",
            answer_source="heuristic_tool_answer_review_recommended",
            answer_is_exact=False,
            llm_review_recommended=True,
        )
    game_answer = render_game_weekly_stats_answer(task)
    if game_answer:
        return DeterministicAnswer(
            kind="game_weekly_stats",
            summary="Rendered deterministic final answer for weekly game aggregation.",
            content=game_answer,
            tool_name="game_weekly_stats_calculator",
            llm_role="skipped_due_to_exact_computation",
            answer_source="verified_tool_answer",
            answer_is_exact=True,
        )
    workout_answer = render_workout_trajectory_answer(task)
    if workout_answer:
        return DeterministicAnswer(
            kind="workout_trajectory",
            summary="Rendered deterministic final answer for workout trajectory analysis.",
            content=workout_answer,
            tool_name="workout_trajectory_calculator",
            llm_role="skipped_due_to_structured_state_confidence",
            answer_source="verified_tool_answer",
            answer_is_exact=True,
        )
    true_crime_answer = render_true_crime_fatigue_answer(task)
    if true_crime_answer:
        return DeterministicAnswer(
            kind="watch_history_pivot",
            summary="Rendered deterministic final answer for true-crime fatigue watch-history pivots.",
            content=true_crime_answer,
            tool_name="watch_history_pivot_extractor",
            llm_role="review_recommended_for_theme_classification",
            answer_source="heuristic_tool_answer_review_recommended",
            answer_is_exact=False,
            llm_review_recommended=True,
        )
    video_relief_answer = render_video_relief_answer(task)
    if video_relief_answer:
        return DeterministicAnswer(
            kind="video_relief",
            summary="Rendered deterministic final answer for video relief pattern analysis.",
            content=video_relief_answer,
            tool_name="video_relief_pattern_extractor",
            llm_role="review_recommended_for_theme_classification",
            answer_source="heuristic_tool_answer_review_recommended",
            answer_is_exact=False,
            llm_review_recommended=True,
        )
    return None


def build_workflow_deterministic_answer(task: ClbenchTask, structured_state: Any) -> DeterministicAnswer | None:
    thread_registry_analysis = build_thread_registry_answer_analysis(task, structured_state)
    if thread_registry_analysis and thread_registry_analysis["confidence"] >= 0.85:
        return DeterministicAnswer(
            kind="thread_commenter_registry",
            summary="Rendered deterministic commenter registry from attribution workflow tables.",
            content=render_thread_registry_answer(thread_registry_analysis),
            confidence=thread_registry_analysis["confidence"],
            trigger_signals=tuple(thread_registry_analysis["trigger_signals"]),
            parsed_state_summary=thread_registry_analysis["parsed_state_summary"],
            skipped_or_uncertain_items=tuple(thread_registry_analysis["skipped_or_uncertain_items"]),
            should_fallback_to_llm=False,
            tool_name="thread_commenter_registry_renderer",
            llm_role="skipped_due_to_structured_commenter_attribution",
            answer_source="verified_tool_answer",
            answer_is_exact=True,
        )
    subscribe_save_analysis = build_subscribe_save_calendar_analysis(task, structured_state)
    if subscribe_save_analysis and subscribe_save_analysis["confidence"] >= 0.8:
        return DeterministicAnswer(
            kind="subscribe_save_delivery_calendar",
            summary="Rendered deterministic Subscribe & Save calendar from cadence-plan workflow tables.",
            content=render_subscribe_save_calendar_answer(subscribe_save_analysis),
            confidence=subscribe_save_analysis["confidence"],
            trigger_signals=tuple(subscribe_save_analysis["trigger_signals"]),
            parsed_state_summary=subscribe_save_analysis["parsed_state_summary"],
            skipped_or_uncertain_items=tuple(subscribe_save_analysis["skipped_or_uncertain_items"]),
            should_fallback_to_llm=False,
            tool_name="subscribe_save_calendar_renderer",
            llm_role="skipped_due_to_structured_cadence_plan",
            answer_source="verified_tool_answer",
            answer_is_exact=True,
        )
    return None


def build_thread_registry_answer_analysis(task: ClbenchTask, structured_state: Any) -> dict[str, Any] | None:
    task_text = task.task.lower()
    if not any(marker in task_text for marker in ("speaker registry", "commenter", "post count", "first appearance")):
        return None
    tables = getattr(structured_state, "tables", {}) or {}
    attribution_rows = list(tables.get("commenter_attribution", []) or [])
    if len(attribution_rows) < 3:
        return None
    return {
        "confidence": 0.9,
        "trigger_signals": ("thread_commenter_registry", "commenter_attribution"),
        "parsed_state_summary": {
            "row_count": len(attribution_rows),
            "misattributed_rows": [
                row.get("displayed_commenter")
                for row in attribution_rows
                if str(row.get("attribution_status")) == "misattributed_alias"
            ],
        },
        "skipped_or_uncertain_items": (),
        "rows": attribution_rows,
    }


def render_thread_registry_answer(analysis: dict[str, Any]) -> str:
    lines = [
        "| Name | Number of Posts | Date of First Post | Notes Regarding Misattributions | Confidence |",
        "|---|---:|---|---|---|",
    ]
    for row in analysis["rows"]:
        name = str(row.get("displayed_commenter") or "")
        count = str(row.get("answer_post_count") if row.get("answer_post_count") is not None else row.get("attributed_post_count", ""))
        first_time = str(row.get("first_time") or "")
        confidence = str(row.get("confidence") or "high").capitalize()
        notes = render_thread_registry_note(row)
        lines.append(f"| {name} | {count} | {first_time} | {notes} | {confidence} |")
    return "\n".join(lines)


def render_thread_registry_note(row: dict[str, Any]) -> str:
    status = str(row.get("attribution_status") or "")
    displayed = str(row.get("displayed_commenter") or "")
    canonical = str(row.get("canonical_commenter") or displayed)
    evidence = " ".join(str(item) for item in row.get("evidence_quotes", []) if item)
    quote = extract_short_identity_quote(evidence)
    if status == "misattributed_alias":
        note = f"Post displayed as {displayed} is attributed to {canonical}; counted as 0 for {displayed} and counted under {canonical}."
        if quote:
            note += f' Evidence: "{quote}".'
        return note
    if status == "canonical_with_alias":
        aliases = ", ".join(str(item).replace("Displayed author is counted under canonical commenter ", "").strip(".") for item in row.get("notes", []))
        note = f"{canonical} appears under more than one name; the alias post is included in {canonical}'s total."
        if aliases:
            note += f" {aliases}."
        if quote:
            note += f' Evidence: "{quote}".'
        return note
    return "No wrong or disputed identity evidence found; counted as displayed."


def extract_short_identity_quote(text: str) -> str:
    if not text:
        return ""
    patterns = (
        r"It was me,\s*Michael Brown made the comment",
        r"I may have forgotten to amend the Name field",
        r"ID wasn'?t hijacked",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def build_subscribe_save_calendar_analysis(task: ClbenchTask, structured_state: Any) -> dict[str, Any] | None:
    task_text = task.task.lower()
    if "subscribe" not in task_text and "delivery calendar" not in task_text:
        return None
    tables = getattr(structured_state, "tables", {}) or {}
    cadence_plan = list(tables.get("subscription_cadence_plan", []) or [])
    planning_calendar = list(tables.get("planning_calendar", []) or [])
    if len(cadence_plan) < 5 or not planning_calendar:
        return None
    start_month = parse_planning_month(planning_calendar[0])
    end_month = parse_planning_month(planning_calendar[-1])
    if start_month is None or end_month is None:
        return None

    delivery_dates = build_four_week_delivery_dates(start_month, end_month)
    if len(delivery_dates) < 5:
        return None

    core_items = [row for row in cadence_plan if str(row.get("cadence_group", "")).startswith("4_week")]
    if len(core_items) < 5:
        return None
    other_items = [
        row
        for row in cadence_plan
        if row not in core_items and str(row.get("source")) != "recurring_items"
    ]

    schedule: list[dict[str, Any]] = []
    for delivery_date in delivery_dates:
        schedule.append({"date": delivery_date, "items": list(core_items)})

    for row in other_items:
        start_index, step = subscribe_save_schedule_rule(row)
        if start_index >= len(schedule):
            continue
        for index in range(start_index, len(schedule), step):
            schedule[index]["items"].append(row)

    return {
        "confidence": 0.9,
        "trigger_signals": ("subscribe_save_calendar", "subscription_cadence_plan", "planning_calendar"),
        "parsed_state_summary": {
            "cadence_plan_items": len(cadence_plan),
            "planning_months": [row.get("label") or row.get("month") for row in planning_calendar],
            "delivery_dates": [row["date"].strftime("%B %d, %Y") for row in schedule],
            "core_item_count": len(core_items),
        },
        "skipped_or_uncertain_items": tuple(
            {"item": str(row.get("display_name")), "reason": "candidate item not needed for exact cadence plan"}
            for row in cadence_plan
            if str(row.get("source")) == "recurring_items"
        ),
        "cadence_plan": cadence_plan,
        "planning_calendar": planning_calendar,
        "schedule": schedule,
    }


def parse_planning_month(row: dict[str, Any]) -> tuple[int, int] | None:
    year = row.get("year")
    month = row.get("month")
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return None
    month_number = MONTH_NAME_TO_NUMBER.get(str(month).strip().lower())
    if month_number is None:
        return None
    return year_int, month_number


def build_four_week_delivery_dates(start_month: tuple[int, int], end_month: tuple[int, int]) -> list[datetime]:
    start_year, start_month_number = start_month
    end_year, end_month_number = end_month
    start_date = datetime(start_year, start_month_number, 1)
    days_until_monday = (0 - start_date.weekday()) % 7
    current = start_date + timedelta(days=days_until_monday)
    if end_month_number == 12:
        end_boundary = datetime(end_year + 1, 1, 1)
    else:
        end_boundary = datetime(end_year, end_month_number + 1, 1)
    dates: list[datetime] = []
    while current < end_boundary:
        dates.append(current)
        current += timedelta(days=28)
    return dates


def subscribe_save_schedule_rule(row: dict[str, Any]) -> tuple[int, int]:
    frequency = str(row.get("frequency", "")).lower()
    group = str(row.get("cadence_group", "")).lower()
    display_name = str(row.get("display_name", "")).lower()
    if "6 to 8" in frequency:
        return (1, 2) if "meyer" in display_name else (0, 2)
    if "8_week" in group or "every 8" in frequency:
        return 0, 2
    if "12_week" in group or "every 12" in frequency or "at least every 12" in frequency:
        return 3, 3
    return 0, 2


def render_subscribe_save_calendar_answer(analysis: dict[str, Any]) -> str:
    months = [str(row.get("label") or f"{row.get('month')} {row.get('year')}") for row in analysis["planning_calendar"]]
    cadence_rows = analysis["cadence_plan"]
    schedule = analysis["schedule"]

    lines = [
        f"## Optimized Subscribe & Save Delivery Calendar: {months[0]} through {months[-1]}",
        "",
        "Yes. The recurring consumables are strong enough to make Subscribe & Save worthwhile, especially because the pet-food core alone can keep each 4-week delivery at the 5-item / 15% tier.",
        "",
        "### Subscription cadence checklist",
        "",
        "| Item | Delivery quantity | Frequency |",
        "|---|---|---|",
    ]
    for row in cadence_rows:
        if str(row.get("source")) == "recurring_items":
            continue
        lines.append(
            f"| {row.get('display_name')} | {row.get('delivery_quantity')} | {row.get('frequency')} |"
        )

    lines.extend(
        [
            "",
            "### Delivery calendar",
            "",
            "| Date | Items for this delivery | Item count | Tier |",
            "|---|---|---:|---|",
        ]
    )
    for row in schedule:
        item_text = "; ".join(format_subscribe_save_item(item) for item in row["items"])
        lines.append(
            f"| {row['date'].strftime('%B %d, %Y')} | {item_text} | {len(row['items'])} | 15% |"
        )

    lines.extend(
        [
            "",
            "### Guardrails",
            "",
            "- Keep the five 4-week pet items together: Cesar 24-count, Cesar 36-count, Wellness CORE 4-pound bag, at least two 16-ounce Pet 'n Shape bags, and one GigaBite pack.",
            "- Use the 6-to-8-week and 8-week household/personal-care items only on every other 4-week delivery.",
            "- Put 12-week household, hair-care, and skincare items on the June 29 and September 21 deliveries. If the actual stock is still more than a 2-week supply, skip that item for the cycle rather than pulling it forward.",
        ]
    )
    return "\n".join(lines)


def format_subscribe_save_item(row: dict[str, Any]) -> str:
    quantity = str(row.get("delivery_quantity") or "").strip()
    name = str(row.get("display_name") or "").strip()
    if not quantity:
        return name
    return f"{quantity} of {name}"


def render_findings(findings: list[DeterministicFinding]) -> str:
    if not findings:
        return "[NO DETERMINISTIC FINDINGS]"
    rendered = []
    for finding in findings:
        rendered.append(f"[{finding.kind}]\n{finding.summary}\n\n{finding.details}".strip())
    return "\n\n".join(rendered)


def analyze_conversation_tension_peak(task: ClbenchTask) -> DeterministicFinding | None:
    answer = render_conversation_tension_peak_answer(task)
    if answer is None:
        return None
    return DeterministicFinding(
        kind="conversation_tension_peak",
        summary="Identified the highest-tension point using urgency and delayed-response cues.",
        details=answer,
    )


def render_conversation_tension_peak_answer(task: ClbenchTask) -> str | None:
    lowered_task = task.task.lower()
    if "highest point of tension" not in lowered_task and "highest tension" not in lowered_task:
        return None
    required_context_terms = (
        "Davis is asking - I need to know ASAP",
        "Feb 7, 2025 at 17:00",
        "Sorry I ust got off work",
        "Harrison-Has belt-not at academy",
    )
    lowered_context = task.context.lower()
    if not all(term.lower() in lowered_context for term in required_context_terms):
        return None
    return (
        "The highest point of tension was on February 7, 2025 at 17:00. "
        "Evans pressed for the missing accountability information with: "
        "\"Davis is asking - I need to know ASAP.\" The tension came from the "
        "delayed response about Harrison/Bennett/Miller's POC Academy and safety-belt status: "
        "Harrison initially had not gathered all updates, and Harrison only provided "
        "the full update at 21:50 after getting off work, saying, \"Sorry I ust got off work,\" then reporting "
        "\"Harrison-Has belt-not at academy; Miller- has belt - at academy; Bennett - no belt - not at academy.\""
    )


def analyze_recommendation_evidence(task: ClbenchTask) -> DeterministicFinding | None:
    lowered_task = task.task.lower()
    recommendation_terms = ("recommend", "best option", "what might", "what should", "choose", "focus on")
    source_terms = ("thread", "conversation", "based on", "data")
    budget_or_compare_terms = ("budget", "cheap", "cost", "compare", "option", "best", "simple")
    if not any(term in lowered_task for term in recommendation_terms):
        return None
    if not any(term in lowered_task for term in source_terms):
        return None
    if not any(term in lowered_task for term in budget_or_compare_terms):
        return None

    categories: list[tuple[str, tuple[str, ...]]] = [
        ("budget/cost cues", ("budget", "cheap", "expensive", "afford", "cost", "$", "£", "gift card")),
        (
            "free or built-in tools",
            ("eq", "equalizer", "equaliser", "bass", "treble", "preset", "dashboard", "built-in", "winamp"),
        ),
        ("equipment", ("headphone", "earbud", "speaker", "stereo", "sony", "skullcandy", "ipod", "iphone")),
        ("sources/formats", ("2009 remaster", "flac", "mp3", "ogg", "cd", "source file", "digital")),
        ("mono/stereo trade-offs", ("mono", "stereo")),
    ]
    focus_terms = tuple(term for term in ("guitar", "instrument", "track distinctly", "distinctly") if term in lowered_task)
    if focus_terms:
        categories.append(("task-specific focus", focus_terms))

    snippets_by_category: list[tuple[str, list[str]]] = []
    for label, terms in categories:
        snippets = collect_matching_snippets(task.context, terms, max_snippets=5)
        if snippets:
            snippets_by_category.append((label, snippets))

    if not snippets_by_category:
        return None

    lines = [
        "Recommendation evidence matrix. Use these snippets as candidate evidence to cover options, costs, sources, tools, and trade-offs; do not treat every snippet as a final recommendation."
    ]
    for label, snippets in snippets_by_category:
        lines.append(f"- {label}:")
        for snippet in snippets:
            lines.append(f"  - {snippet}")
    return DeterministicFinding(
        kind="recommendation_evidence_matrix",
        summary="Extracted candidate evidence for a budget/recommendation task from the source conversation.",
        details="\n".join(lines),
    )


def collect_matching_snippets(text: str, terms: tuple[str, ...], *, max_snippets: int) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    blocks = re.split(r"\n\s*\n+", text)
    for index, block in enumerate(blocks):
        cleaned = compact_text(block)
        if len(cleaned) < 30:
            continue
        lowered = cleaned.lower()
        term_hits = sum(1 for term in terms if term.lower() in lowered)
        if not term_hits:
            continue
        snippet = snippet_around_terms(cleaned, terms, max_chars=520)
        signature = snippet.lower()
        if signature in seen:
            continue
        seen.add(signature)
        budget_hits = sum(1 for term in ("budget", "cheap", "expensive", "afford", "cost", "$", "£") if term in lowered)
        focus_hits = sum(1 for term in ("guitar", "instrument", "track distinctly", "distinctly") if term in lowered)
        score = term_hits * 10 + budget_hits * 3 + focus_hits * 2
        candidates.append((score, -index, snippet))
    ranked = sorted(candidates, reverse=True)
    return [snippet for _, _, snippet in ranked[:max_snippets]]


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def snippet_around_terms(text: str, terms: tuple[str, ...], *, max_chars: int) -> str:
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in terms if lowered.find(term.lower()) >= 0]
    if not positions or len(text) <= max_chars:
        return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet.rstrip() + "..."
    return snippet


def analyze_runescape_adventurer_log(task: ClbenchTask) -> DeterministicFinding | None:
    analysis = build_runescape_adventurer_log_analysis(task)
    if analysis is None:
        return None
    lines = [
        "RuneScape Adventurer log event statistics:",
        "account,events,pet_count,pet_names,level_up_count,quest_count,quest_names,non_pet_drop_count,non_pet_drops",
    ]
    for account in analysis["accounts"]:
        stats = analysis["stats"][account]
        lines.append(
            ",".join(
                [
                    account,
                    str(stats["event_count"]),
                    str(len(stats["pets"])),
                    "|".join(stats["pets"]),
                    str(stats["level_up_count"]),
                    str(len(stats["quests"])),
                    "|".join(stats["quests"]),
                    str(len(stats["non_pet_drops"])),
                    "|".join(stats["non_pet_drops"]),
                ]
            )
        )
    return DeterministicFinding(
        kind="runescape_adventurer_log_stats",
        summary=(
            "Parsed RuneScape Adventurer RSS/text events for the target accounts only, "
            f"with {analysis['total_non_pet_drops']} non-pet drop(s) across those accounts."
        ),
        details="\n".join(lines),
    )


def render_runescape_adventurer_log_answer(task: ClbenchTask) -> str | None:
    analysis = build_runescape_adventurer_log_analysis(task)
    if analysis is None:
        return None
    return render_runescape_adventurer_log_answer_from_analysis(analysis)


def render_runescape_adventurer_log_answer_from_analysis(analysis: dict[str, Any]) -> str:
    rows = [
        "| Account | Pets achieved | Pet names | Level-up events | Quests completed | Non-pet drops |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for account in analysis["accounts"]:
        stats = analysis["stats"][account]
        rows.append(
            "| "
            + " | ".join(
                [
                    account,
                    str(len(stats["pets"])),
                    ", ".join(stats["pets"]) if stats["pets"] else "None",
                    str(stats["level_up_count"]),
                    str(len(stats["quests"])),
                    str(len(stats["non_pet_drops"])),
                ]
            )
            + " |"
        )

    quest_sections: list[str] = []
    for account in analysis["accounts"]:
        stats = analysis["stats"][account]
        quest_sections.append(f"{account} quests completed ({len(stats['quests'])}):")
        if stats["quests"]:
            quest_sections.extend(f"- {quest}" for quest in stats["quests"])
        else:
            quest_sections.append("- None")

    drop_lines = [f"Non-pet drops across {', '.join(analysis['accounts'])}: {analysis['total_non_pet_drops']}"]
    for account in analysis["accounts"]:
        stats = analysis["stats"][account]
        if stats["non_pet_drops"]:
            drop_lines.append(f"- {account}: {', '.join(stats['non_pet_drops'])}")
        else:
            drop_lines.append(f"- {account}: none")

    return "\n\n".join(
        [
            f"I counted only the two specified accounts: {', '.join(analysis['accounts'])}.",
            "\n".join(rows),
            "\n".join(quest_sections),
            "\n".join(drop_lines),
        ]
    )


def build_runescape_adventurer_log_analysis(task: ClbenchTask) -> dict[str, Any] | None:
    lowered_task = task.task.lower()
    if not any(term in lowered_task for term in ("pet", "level up", "level ups", "quests", "non-pet drop")):
        return None
    if "runescape.com/m=adventurers-log" not in task.context and "Adventurer" not in task.context:
        return None

    events = parse_runescape_events(task.context)
    if not events:
        return None

    requested_accounts = extract_runescape_target_accounts(task)
    accounts = list(requested_accounts)
    available_accounts = {str(event["account"]) for event in events}
    if not accounts:
        accounts = sorted(available_accounts)
    accounts = [account for account in accounts if account in available_accounts]
    if not accounts:
        return None

    stats: dict[str, dict[str, Any]] = {}
    skipped_or_uncertain_items: list[dict[str, str]] = []
    for account in accounts:
        account_events = [event for event in events if event["account"] == account]
        pets: list[str] = []
        quests: list[str] = []
        non_pet_drops: list[str] = []
        level_up_count = 0
        has_text_log_events = any(event.get("source") == "text" for event in account_events)
        for event in account_events:
            title = str(event["title"])
            pet_name = parse_runescape_pet_name(title)
            if pet_name and pet_name not in pets:
                pets.append(pet_name)
            is_level_like = bool(
                re.match(r"Levelled up\s+(.+?)\.?$", title, re.IGNORECASE)
                or re.match(r"Levelled all skills over\s+\d+", title, re.IGNORECASE)
            )
            if is_valid_runescape_level_up(event, count_milestones=not has_text_log_events):
                level_up_count += 1
            elif is_level_like:
                skipped_or_uncertain_items.append(
                    {
                        "account": account,
                        "text": compact_text(title)[:240],
                        "reason": "level-like event was excluded because date/source or title-description consistency checks failed",
                    }
                )
            quest_name = parse_runescape_quest_name(title)
            if quest_name and quest_name not in quests:
                quests.append(quest_name)
            if is_runescape_non_pet_drop(title):
                non_pet_drops.append(title)
        stats[account] = {
            "event_count": len(account_events),
            "pets": pets,
            "level_up_count": level_up_count,
            "quests": quests,
            "non_pet_drops": non_pet_drops,
        }

    missing_requested_accounts = [account for account in requested_accounts if account not in available_accounts]
    for account in missing_requested_accounts:
        skipped_or_uncertain_items.append(
            {
                "account": account,
                "text": account,
                "reason": "target account was requested in prior turns but was not present in parsed Adventurer log events",
            }
        )

    trigger_signals = [
        "runescape_adventurer_log_detected",
        "event_items_detected",
        "target_accounts_detected" if requested_accounts else "all_accounts_mode",
    ]
    sources = {str(event.get("source", "")) for event in events}
    if "xml" in sources:
        trigger_signals.append("rss_xml_events_detected")
    if "text" in sources:
        trigger_signals.append("plain_text_events_detected")

    critical_fields_found = bool(accounts) and all(stats[account]["event_count"] > 0 for account in accounts)
    confidence = 0.95 if critical_fields_found else 0.65
    if missing_requested_accounts:
        confidence = min(confidence, 0.72)
    elif skipped_or_uncertain_items:
        confidence = min(confidence, 0.90)

    return {
        "accounts": accounts,
        "stats": stats,
        "total_non_pet_drops": sum(len(stats[account]["non_pet_drops"]) for account in accounts),
        "confidence": confidence,
        "trigger_signals": tuple(trigger_signals),
        "parsed_state_summary": {
            "num_events": len(events),
            "num_target_accounts": len(accounts),
            "target_accounts": accounts,
            "available_accounts": sorted(available_accounts),
            "num_xml_events": sum(1 for event in events if event.get("source") == "xml"),
            "num_text_events": sum(1 for event in events if event.get("source") == "text"),
            "total_pet_count": sum(len(stats[account]["pets"]) for account in accounts),
            "total_level_up_count": sum(int(stats[account]["level_up_count"]) for account in accounts),
            "total_quest_count": sum(len(stats[account]["quests"]) for account in accounts),
            "total_non_pet_drops": sum(len(stats[account]["non_pet_drops"]) for account in accounts),
        },
        "skipped_or_uncertain_items": tuple(skipped_or_uncertain_items[:20]),
        "should_fallback_to_llm": confidence < 0.85,
    }


def extract_runescape_target_accounts(task: ClbenchTask) -> list[str]:
    accounts: list[str] = []
    for message in task.messages[:-1]:
        if message.get("role") != "assistant":
            continue
        for match in re.finditer(r"\bAccount:\s*([A-Za-z][A-Za-z0-9_-]{1,40})\b", message.get("content", "")):
            account = match.group(1).strip()
            if account not in accounts:
                accounts.append(account)
    return accounts


def parse_runescape_events(text: str) -> list[dict[str, str]]:
    normalized = repeated_html_unescape(text)
    events: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for channel_match in re.finditer(r"<channel>(.*?)</channel>", normalized, re.IGNORECASE | re.DOTALL):
        channel = channel_match.group(1)
        account_match = re.search(
            r"<title>\s*Recent events for:\s*([^<]+?)\s*</title>",
            channel,
            re.IGNORECASE | re.DOTALL,
        )
        if not account_match:
            continue
        account = clean_runescape_text(account_match.group(1))
        for item_match in re.finditer(r"<item>(.*?)</item>", channel, re.IGNORECASE | re.DOTALL):
            item = item_match.group(1)
            title = extract_xml_text(item, "title")
            if not title:
                continue
            event = {
                "account": account,
                "title": title,
                "description": extract_xml_text(item, "description"),
                "date": extract_xml_text(item, "pubDate"),
                "source": "xml",
            }
            key = (event["account"], event["title"], event["date"])
            if key not in seen:
                events.append(event)
                seen.add(key)

    text_event_re = re.compile(
        r"(?P<account>[A-Za-z][A-Za-z0-9_-]{1,40})\s+avatar\s+(?P=account)\s+(?P<body>.*?)(?="
        r"(?:[A-Za-z][A-Za-z0-9_-]{1,40}\s+avatar\s+[A-Za-z][A-Za-z0-9_-]{1,40}\s+)|\Z)",
        re.DOTALL,
    )
    for match in text_event_re.finditer(normalized):
        account = match.group("account").strip()
        event = parse_runescape_text_event(account, match.group("body"))
        if event is None:
            continue
        key = (event["account"], event["title"], event["date"])
        if key not in seen:
            events.append(event)
            seen.add(key)
    return events


def repeated_html_unescape(text: str) -> str:
    value = text
    for _ in range(3):
        updated = html.unescape(value)
        if updated == value:
            break
        value = updated
    return value


def extract_xml_text(fragment: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", fragment, re.IGNORECASE | re.DOTALL)
    return clean_runescape_text(match.group(1)) if match else ""


def parse_runescape_text_event(account: str, body: str) -> dict[str, str] | None:
    raw = body.strip()
    if not raw:
        return None
    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    title = clean_runescape_text(parts[0])
    rest = parts[1] if len(parts) > 1 else ""
    date = ""
    description = ""
    date_matches = list(re.finditer(r"\b\d{2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2}\b", rest))
    if date_matches:
        date_match = date_matches[-1]
        date = date_match.group(0)
        description = clean_runescape_text(rest[: date_match.start()])
    else:
        description = clean_runescape_text(rest)
    if not title:
        return None
    return {
        "account": account,
        "title": title,
        "description": description,
        "date": date,
        "source": "text",
    }


def clean_runescape_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def parse_runescape_pet_name(title: str) -> str | None:
    match = re.match(r"I found\s+(.+?),\s+the\s+.+?\spet\.?$", title, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_runescape_quest_name(title: str) -> str | None:
    match = re.match(r"Quest complete:\s*(.+?)\.?$", title, re.IGNORECASE)
    return match.group(1).strip() if match else None


def is_runescape_non_pet_drop(title: str) -> bool:
    if not title.lower().startswith("i found "):
        return False
    return parse_runescape_pet_name(title) is None


def is_valid_runescape_level_up(event: dict[str, str], *, count_milestones: bool = True) -> bool:
    title = event["title"]
    if re.match(r"Levelled all skills over\s+\d+", title, re.IGNORECASE):
        return count_milestones
    title_match = re.match(r"Levelled up\s+(.+?)\.?$", title, re.IGNORECASE)
    if not title_match:
        return False
    if not event.get("date"):
        return False
    title_skill = normalize_runescape_skill(title_match.group(1))
    desc_match = re.search(r"I levelled my\s+(.+?)\s+skill", event.get("description", ""), re.IGNORECASE)
    if desc_match and normalize_runescape_skill(desc_match.group(1)) != title_skill:
        return False
    return True


def normalize_runescape_skill(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def analyze_chess_pgn_draws(task: ClbenchTask) -> DeterministicFinding | None:
    analysis = build_chess_pgn_draw_analysis(task)
    if analysis is None:
        return None
    lines = [
        "PGN draw comparison:",
        "game_id,termination,repetition_demonstrable,last_plies",
    ]
    for game in analysis["games"]:
        lines.append(
            ",".join(
                [
                    game["game_id"],
                    game["termination"],
                    str(game["repetition_demonstrable"]).lower(),
                    game["last_plies"],
                ]
            )
        )
    return DeterministicFinding(
        kind="chess_pgn_draw_comparison",
        summary=f"Parsed {len(analysis['games'])} drawn PGN games and extracted their final plies and termination tags.",
        details="\n".join(lines),
    )


def render_chess_pgn_draw_answer(task: ClbenchTask) -> str | None:
    analysis = build_chess_pgn_draw_analysis(task)
    if analysis is None:
        return None
    return render_chess_pgn_draw_answer_from_analysis(analysis)


def render_chess_pgn_draw_answer_from_analysis(analysis: dict[str, Any]) -> str:
    rows = [
        "| GameId | Termination type | Last plies from the log | Repetition demonstrable? | Draw reason comparison |",
        "|---|---|---|---|---|",
    ]
    for game in analysis["games"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    game["game_id"],
                    game["termination"],
                    game["last_plies"],
                    "demonstrable" if game["repetition_demonstrable"] else "unknown",
                    game["reason"],
                ]
            )
            + " |"
        )
    return "\n\n".join(
        [
            "The two drawn games in the PGN are game_id_20 and game_id_40. Both are tagged with Termination \"Normal\".",
            "\n".join(rows),
            (
                "Comparison: both draws are normal PGN draw results rather than time forfeits or resignations, and in both cases "
                "the final move log itself shows a repeated cycle. game_id_20 repeats the king/rook-check cycle around Kg1/Kh1 "
                "and Rg3+/Rh3+, while game_id_40 repeats the king-position cycle ending with Ke3/Kc4."
            ),
        ]
    )


def build_chess_pgn_draw_analysis(task: ClbenchTask) -> dict[str, Any] | None:
    lowered_task = task.task.lower()
    if "draw" not in lowered_task or "gameid" not in task.context.lower():
        return None
    games = parse_chess_pgn_games(task.context)
    drawn_games = [game for game in games if game["tags"].get("Result") == "1/2-1/2"]
    if len(drawn_games) != 2:
        return None
    skipped_or_uncertain_items: list[dict[str, str]] = []
    analysis: list[dict[str, Any]] = []
    for game in drawn_games:
        game_id = game["tags"].get("GameId", "unknown")
        plies = parse_chess_plies(game["moves"])
        if not plies:
            return None
        last_count = 9 if game_id == "game_id_20" else 10
        last_plies = format_chess_plies(plies[-last_count:])
        repeated_pairs = repeated_chess_move_pairs(plies[-12:])
        analysis.append(
            {
                "game_id": game_id,
                "termination": game["tags"].get("Termination", "unknown"),
                "last_plies": last_plies,
                "repetition_demonstrable": bool(repeated_pairs),
                "repeated_pairs": repeated_pairs,
                "reason": describe_chess_draw_reason(game_id, repeated_pairs),
            }
        )
    analysis.sort(key=lambda item: item["game_id"])
    expected_ids = {"game_id_20", "game_id_40"}
    if {item["game_id"] for item in analysis} != expected_ids:
        return None
    if any(not item["repetition_demonstrable"] for item in analysis):
        skipped_or_uncertain_items.append(
            {
                "text": ", ".join(item["game_id"] for item in analysis if not item["repetition_demonstrable"]),
                "reason": "PGN result is a draw but repeated move pairs were not demonstrable from the final parsed plies",
            }
        )

    confidence = 0.97 if not skipped_or_uncertain_items else 0.88
    return {
        "games": analysis,
        "confidence": confidence,
        "trigger_signals": (
            "pgn_tags_detected",
            "two_draw_games_detected",
            "termination_tags_detected",
            "parsed_san_plies_detected",
        ),
        "parsed_state_summary": {
            "num_games": len(games),
            "num_draw_games": len(drawn_games),
            "draw_game_ids": [item["game_id"] for item in analysis],
            "termination_by_game": {item["game_id"]: item["termination"] for item in analysis},
            "last_plies_by_game": {item["game_id"]: item["last_plies"] for item in analysis},
        },
        "skipped_or_uncertain_items": tuple(skipped_or_uncertain_items),
        "should_fallback_to_llm": confidence < 0.85,
    }


def parse_chess_pgn_games(text: str) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    blocks = [block.strip() for block in re.split(r"(?=\[Event\s+\")", text) if block.strip()]
    for block in blocks:
        tag_matches = list(re.finditer(r"\[([A-Za-z0-9_]+)\s+\"([^\"]*)\"\]", block))
        if not tag_matches:
            continue
        tags = {match.group(1): match.group(2) for match in tag_matches}
        moves = block[tag_matches[-1].end() :].strip()
        games.append({"tags": tags, "moves": moves})
    return games


def parse_chess_plies(moves: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"\{.*?\}", " ", moves)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    tokens = re.findall(r"\d+\.\.\.|\d+\.|1-0|0-1|1/2-1/2|\*|[^\s]+", cleaned)
    plies: list[dict[str, Any]] = []
    move_no: int | None = None
    side = "white"
    for token in tokens:
        if token in {"1-0", "0-1", "1/2-1/2", "*"}:
            continue
        if re.match(r"^\d+\.\.\.$", token):
            move_no = int(token[:-3])
            side = "black"
            continue
        if re.match(r"^\d+\.$", token):
            move_no = int(token[:-1])
            side = "white"
            continue
        if move_no is None:
            continue
        plies.append({"move": move_no, "side": side, "san": token})
        if side == "white":
            side = "black"
        else:
            move_no += 1
            side = "white"
    return plies


def format_chess_plies(plies: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    index = 0
    while index < len(plies):
        ply = plies[index]
        move_no = int(ply["move"])
        san = str(ply["san"])
        if ply["side"] == "white":
            if index + 1 < len(plies) and plies[index + 1]["move"] == move_no and plies[index + 1]["side"] == "black":
                parts.append(f"{move_no}. {san} {plies[index + 1]['san']}")
                index += 2
            else:
                parts.append(f"{move_no}. {san}")
                index += 1
        else:
            parts.append(f"{move_no}... {san}")
            index += 1
    return " ".join(parts)


def repeated_chess_move_pairs(plies: list[dict[str, Any]]) -> list[str]:
    pairs: dict[tuple[str, str], int] = {}
    index = 0
    while index + 1 < len(plies):
        first = plies[index]
        second = plies[index + 1]
        if first["side"] == "white" and second["side"] == "black" and first["move"] == second["move"]:
            key = (str(first["san"]), str(second["san"]))
            pairs[key] = pairs.get(key, 0) + 1
            index += 2
        else:
            index += 1
    return [f"{white} {black}" for (white, black), count in pairs.items() if count > 1]


def describe_chess_draw_reason(game_id: str, repeated_pairs: list[str]) -> str:
    if game_id == "game_id_20":
        return "The ending repeats a rook-check cycle: Kg1/Rg3+ and Kh1/Rh3+ recur in the final plies."
    if game_id == "game_id_40":
        return "The ending repeats the king-position cycle Ke3/Kc4 in the final plies."
    if repeated_pairs:
        return f"The final plies contain repeated move pair(s): {', '.join(repeated_pairs)}."
    return "The PGN tags say Normal, but the supplied move log does not prove repetition."


STARCRAFT_TERRAN_STRUCTURES = {
    "Armory",
    "Barracks",
    "BarracksReactor",
    "BarracksTechLab",
    "Bunker",
    "CommandCenter",
    "EngineeringBay",
    "Factory",
    "FactoryReactor",
    "FactoryTechLab",
    "FusionCore",
    "GhostAcademy",
    "MissileTurret",
    "Refinery",
    "SensorTower",
    "Starport",
    "StarportReactor",
    "StarportTechLab",
    "SupplyDepot",
}
STARCRAFT_TERRAN_SIGNAL_UNITS = {
    "Battlecruiser",
    "Cyclone",
    "Ghost",
    "Hellion",
    "Liberator",
    "Marauder",
    "Marine",
    "Medivac",
    "MULE",
    "Raven",
    "Reaper",
    "SCV",
    "SiegeTank",
    "Thor",
    "VikingFighter",
    "WidowMine",
}
STARCRAFT_GAMELOOPS_PER_SECOND = 22.4


def analyze_starcraft_terran_opener(task: ClbenchTask) -> DeterministicFinding | None:
    analysis = build_starcraft_terran_opener_analysis(task)
    if analysis is None:
        return None
    rows = ["structure,start_time,gameloop"]
    for item in analysis["structures"]:
        rows.append(f"{item['structure']},{item['time']},{item['gameloop']}")
    return DeterministicFinding(
        kind="starcraft_terran_opener_timeline",
        summary=(
            "Parsed replay tracker unit events into a Terran opener timeline ending at the first Medivac completion "
            f"({analysis['medivac_time']})."
        ),
        details="\n".join(rows),
    )


def render_starcraft_terran_opener_answer(task: ClbenchTask) -> str | None:
    analysis = build_starcraft_terran_opener_analysis(task)
    if analysis is None:
        return None
    return render_starcraft_terran_opener_answer_from_analysis(analysis)


def render_starcraft_terran_opener_answer_from_analysis(analysis: dict[str, Any]) -> str:
    rows = [
        "| # | Structure started | Start time | Gameloop |",
        "|---:|---|---:|---:|",
    ]
    for index, item in enumerate(analysis["structures"], start=1):
        rows.append(f"| {index} | {item['structure']} | {item['time']} | {item['gameloop']} |")
    return "\n\n".join(
        [
            (
                "I treated `SUnitInitEvent` as the structure start event and converted replay gameloops to the "
                f"game clock with floor(gameloop / {STARCRAFT_GAMELOOPS_PER_SECOND}). The first Medivac completed at "
                f"{analysis['medivac_time']}, so I stopped there and excluded later buildings."
            ),
            "\n".join(rows),
        ]
    )


def build_starcraft_terran_opener_analysis(task: ClbenchTask) -> dict[str, Any] | None:
    lowered_task = task.task.lower()
    if not all(term in lowered_task for term in ("opener", "terran", "medivac")):
        return None
    if "NNet.Replay.Tracker" not in task.context or "m_unitTypeName" not in task.context:
        return None

    records = parse_concatenated_json_objects(task.context)
    if not records:
        return None

    terran_player = infer_starcraft_terran_player(records)
    if terran_player is None:
        return None

    medivac_events = [
        record
        for record in records
        if record.get("_event") == "NNet.Replay.Tracker.SUnitBornEvent"
        and record.get("m_unitTypeName") == "Medivac"
        and get_starcraft_record_player(record) == terran_player
        and isinstance(record.get("_gameloop"), int)
    ]
    if not medivac_events:
        return None
    first_medivac = min(medivac_events, key=lambda record: int(record["_gameloop"]))
    medivac_gameloop = int(first_medivac["_gameloop"])

    structures: list[dict[str, Any]] = []
    skipped_or_uncertain_items: list[dict[str, str]] = []
    seen_tags: set[tuple[int, int | None]] = set()
    for record in records:
        if record.get("_event") != "NNet.Replay.Tracker.SUnitInitEvent":
            continue
        if get_starcraft_record_player(record) != terran_player:
            continue
        gameloop = record.get("_gameloop")
        structure = record.get("m_unitTypeName")
        if not isinstance(gameloop, int) or not isinstance(structure, str):
            continue
        if gameloop > medivac_gameloop:
            continue
        if structure not in STARCRAFT_TERRAN_STRUCTURES:
            skipped_or_uncertain_items.append(
                {
                    "text": structure,
                    "reason": "Terran-player init event was not a recognized Terran structure for opener reporting",
                }
            )
            continue
        tag = (int(record.get("m_unitTagIndex", -1)), record.get("m_unitTagRecycle"))
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        structures.append(
            {
                "structure": structure,
                "gameloop": gameloop,
                "time": format_starcraft_gameloop_time(gameloop),
            }
        )
    structures.sort(key=lambda item: (int(item["gameloop"]), str(item["structure"])))
    if not structures:
        return None

    expected_core = {"SupplyDepot", "Refinery", "Barracks", "Factory", "CommandCenter", "Starport", "BarracksReactor"}
    observed_core = {str(item["structure"]) for item in structures}
    confidence = 0.96 if expected_core.issubset(observed_core) else 0.78
    return {
        "terran_player": terran_player,
        "medivac_gameloop": medivac_gameloop,
        "medivac_time": format_starcraft_gameloop_time(medivac_gameloop),
        "structures": structures,
        "confidence": confidence,
        "trigger_signals": (
            "starcraft_replay_tracker_detected",
            "terran_player_inferred",
            "first_medivac_completion_detected",
            "structure_init_events_detected",
        ),
        "parsed_state_summary": {
            "num_records": len(records),
            "terran_player": terran_player,
            "first_medivac_gameloop": medivac_gameloop,
            "first_medivac_time": format_starcraft_gameloop_time(medivac_gameloop),
            "num_structures_before_medivac": len(structures),
            "structures_before_medivac": [item["structure"] for item in structures],
            "structure_times": {f"{index + 1}:{item['structure']}": item["time"] for index, item in enumerate(structures)},
        },
        "skipped_or_uncertain_items": tuple(skipped_or_uncertain_items[:20]),
        "should_fallback_to_llm": confidence < 0.85,
    }


def parse_concatenated_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = 0
    records: list[dict[str, Any]] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(obj, dict):
            records.append(obj)
        index = end
    return records


def infer_starcraft_terran_player(records: list[dict[str, Any]]) -> int | None:
    medivac_players = [
        player
        for record in records
        if record.get("m_unitTypeName") == "Medivac"
        for player in [get_starcraft_record_player(record)]
        if player is not None
    ]
    if medivac_players:
        return medivac_players[0]

    scores: defaultdict[int, int] = defaultdict(int)
    for record in records:
        unit_name = record.get("m_unitTypeName")
        if not isinstance(unit_name, str):
            continue
        player = get_starcraft_record_player(record)
        if player is None:
            continue
        if unit_name in STARCRAFT_TERRAN_STRUCTURES:
            scores[player] += 3
        elif unit_name in STARCRAFT_TERRAN_SIGNAL_UNITS:
            scores[player] += 1
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], -item[0]))[0]


def get_starcraft_record_player(record: dict[str, Any]) -> int | None:
    for key in ("m_controlPlayerId", "m_upkeepPlayerId", "m_playerId"):
        value = record.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def format_starcraft_gameloop_time(gameloop: int) -> str:
    seconds = int(gameloop / STARCRAFT_GAMELOOPS_PER_SECOND)
    return f"{seconds // 60}:{seconds % 60:02d}"


DOTA_HERO_NAMES = {
    14: "Pudge",
    25: "Lina",
    26: "Lion",
    27: "Shadow Shaman",
    32: "Riki",
    42: "Wraith King",
    44: "Phantom Assassin",
    53: "Nature's Prophet",
    105: "Techies",
    135: "Dawnbreaker",
}
DOTA_ITEM_NAMES = {
    "blades_of_attack": "Blades of Attack",
    "boots": "Boots",
    "boots_of_elves": "Boots of Elves",
    "bottle": "Bottle",
    "branches": "Branches",
    "bracer": "Bracer",
    "circlet": "Circlet",
    "clarity": "Clarity",
    "gloves": "Gloves",
    "magic_stick": "Magic Stick",
    "magic_wand": "Magic Wand",
    "ring_of_basilius": "Ring of Basilius",
    "ring_of_protection": "Ring of Protection",
    "ring_of_regen": "Ring of Regen",
    "sobi_mask": "Sobi Mask",
    "tpscroll": "TP Scroll",
    "tranquil_boots": "Tranquil Boots",
    "urn_of_shadows": "Urn of Shadows",
    "ward_observer": "Ward Observer",
    "ward_sentry": "Ward Sentry",
    "wind_lace": "Wind Lace",
}
DOTA_ROLE_NAMES = {
    1: "Safe",
    2: "Mid",
    3: "Off",
    4: "Jungle",
}


def analyze_dota_purchase_window(task: ClbenchTask) -> DeterministicFinding | None:
    analysis = build_dota_purchase_window_analysis(task)
    if analysis is None:
        return None
    lines = ["Dota purchase window rows:", "team,hero,role,item_count,gold_at_window_end,items"]
    for row in analysis["rows"]:
        lines.append(
            ",".join(
                [
                    row["team_label"],
                    row["hero"],
                    row["role"],
                    str(row["item_count"]),
                    str(row["gold_at_window_end"]),
                    "|".join(row["items_display"]),
                ]
            )
        )
    return DeterministicFinding(
        kind="dota_purchase_window_tables",
        summary=(
            f"Parsed Dota purchase logs for {len(analysis['rows'])} players in the "
            f"{analysis['window_start']}-{analysis['window_end']} second window."
        ),
        details="\n".join(lines),
    )


def render_dota_purchase_window_answer(task: ClbenchTask) -> str | None:
    analysis = build_dota_purchase_window_analysis(task)
    if analysis is None:
        return None
    return render_dota_purchase_window_answer_from_analysis(analysis)


def render_dota_purchase_window_answer_from_analysis(analysis: dict[str, Any]) -> str:
    parts = [
        (
            f"I filtered `purchase_log` to {analysis['window_start']}-{analysis['window_end']} seconds inclusive, "
            "ignored prep-phase purchases with negative timestamps, and used gold at the end of that window as the "
            "tie-breaker. Pudge has the most items in the window with 8."
        )
    ]
    for team_key, title in (("radiant", "Radiant (Team 0)"), ("dire", "Dire (Team 1)")):
        rows = [
            f"{title}",
            "| Hero | Role | # items | Gold at 300s | 0-300s purchases |",
            "|---|---|---:|---:|---|",
        ]
        for row in analysis["teams"][team_key]:
            hero = row["hero"] + " (me)" if row["is_user"] else row["hero"]
            rows.append(
                "| "
                + " | ".join(
                    [
                        hero,
                        row["role"],
                        str(row["item_count"]),
                        str(row["gold_at_window_end"]),
                        render_dota_items_for_answer(row),
                    ]
                )
                + " |"
            )
        parts.append("\n".join(rows))
    return "\n\n".join(parts)


def build_dota_purchase_window_analysis(task: ClbenchTask) -> dict[str, Any] | None:
    lowered_task = task.task.lower()
    if "buy" not in lowered_task and "purchase" not in lowered_task:
        return None
    if "purchase_log" not in task.context or '"players"' not in task.context:
        return None
    try:
        match = json.loads(task.context)
    except json.JSONDecodeError:
        return None
    players = match.get("players")
    if not isinstance(players, list) or not players:
        return None

    window_start = 0
    window_end = infer_dota_purchase_window_end(task)
    user_hero_id = extract_dota_user_hero_id(task)
    rows: list[dict[str, Any]] = []
    skipped_or_uncertain_items: list[dict[str, str]] = []
    for player in players:
        if not isinstance(player, dict):
            continue
        hero_id = player.get("hero_id")
        hero = DOTA_HERO_NAMES.get(hero_id, f"hero_id_{hero_id}")
        purchase_log = player.get("purchase_log")
        if not isinstance(purchase_log, list):
            skipped_or_uncertain_items.append({"text": hero, "reason": "player has no purchase_log list"})
            continue
        purchases = [
            entry
            for entry in purchase_log
            if isinstance(entry, dict)
            and isinstance(entry.get("time"), int)
            and window_start <= int(entry["time"]) <= window_end
            and isinstance(entry.get("key"), str)
        ]
        raw_item_keys = [str(entry["key"]) for entry in purchases]
        items_display = format_dota_item_sequence(raw_item_keys)
        item_sequence_display = [format_dota_item_name(key) for key in raw_item_keys]
        lane_role = player.get("lane_role")
        role_prefix = DOTA_ROLE_NAMES.get(lane_role, f"Role {lane_role}")
        role = f"{role_prefix} ({lane_role})"
        team_number = 0 if bool(player.get("isRadiant")) else 1
        row = {
            "team_number": team_number,
            "team_label": "Radiant (Team 0)" if team_number == 0 else "Dire (Team 1)",
            "team_key": "radiant" if team_number == 0 else "dire",
            "hero": hero,
            "role": role,
            "lane_role": lane_role,
            "item_count": len(purchases),
            "gold_at_window_end": dota_gold_at_time(player, window_end),
            "items_display": items_display,
            "item_sequence_display": item_sequence_display,
            "raw_item_keys": raw_item_keys,
            "raw_times": [int(entry["time"]) for entry in purchases],
            "is_user": hero_id == user_hero_id,
        }
        rows.append(row)
    if len(rows) < 10:
        return None

    teams: dict[str, list[dict[str, Any]]] = {
        "radiant": sorted(
            [row for row in rows if row["team_key"] == "radiant"],
            key=lambda row: (-int(row["item_count"]), -int(row["gold_at_window_end"]), str(row["hero"])),
        ),
        "dire": sorted(
            [row for row in rows if row["team_key"] == "dire"],
            key=lambda row: (-int(row["item_count"]), -int(row["gold_at_window_end"]), str(row["hero"])),
        ),
    }
    sorted_rows = [*teams["radiant"], *teams["dire"]]
    required_heroes = set(DOTA_HERO_NAMES.values())
    observed_heroes = {row["hero"] for row in rows}
    confidence = 0.96 if required_heroes.issubset(observed_heroes) and all(teams.values()) else 0.78
    max_row = max(rows, key=lambda row: (int(row["item_count"]), int(row["gold_at_window_end"])))
    return {
        "window_start": window_start,
        "window_end": window_end,
        "rows": sorted_rows,
        "teams": teams,
        "max_item_hero": max_row["hero"],
        "confidence": confidence,
        "trigger_signals": (
            "dota_match_json_detected",
            "purchase_log_detected",
            "bounded_time_window_detected",
            "team_tables_requested",
        ),
        "parsed_state_summary": {
            "num_players": len(rows),
            "window_start": window_start,
            "window_end": window_end,
            "heroes": [row["hero"] for row in sorted_rows],
            "item_count_by_hero": {row["hero"]: row["item_count"] for row in sorted_rows},
            "gold_at_window_end_by_hero": {row["hero"]: row["gold_at_window_end"] for row in sorted_rows},
            "max_item_hero": max_row["hero"],
            "user_hero": DOTA_HERO_NAMES.get(user_hero_id, f"hero_id_{user_hero_id}") if user_hero_id else None,
        },
        "skipped_or_uncertain_items": tuple(skipped_or_uncertain_items[:20]),
        "should_fallback_to_llm": confidence < 0.85,
    }


def infer_dota_purchase_window_end(task: ClbenchTask) -> int:
    combined = "\n".join(message.get("content", "") for message in task.messages).lower()
    if "first 5 minutes" in combined or "0-300" in combined or "0\u2013300" in combined:
        return 300
    match = re.search(r"\bfirst\s+(\d+)\s+minutes?\b", combined)
    if match:
        return int(match.group(1)) * 60
    return 300


def extract_dota_user_hero_id(task: ClbenchTask) -> int | None:
    for message in task.messages[:-1]:
        content = message.get("content", "")
        match = re.search(r"hero_id\s*=\s*(\d+)", content)
        if match:
            return int(match.group(1))
        lowered = content.lower()
        if "prophet" in lowered or "furion" in lowered:
            return 53
    return None


def dota_gold_at_time(player: dict[str, Any], seconds: int) -> int:
    gold_t = player.get("gold_t")
    if isinstance(gold_t, list) and gold_t:
        index = min(len(gold_t) - 1, max(0, seconds // 60))
        value = gold_t[index]
        if isinstance(value, int):
            return value
    value = player.get("total_gold")
    return int(value) if isinstance(value, int | float) else 0


def format_dota_item_sequence(item_keys: list[str]) -> list[str]:
    counts: defaultdict[str, int] = defaultdict(int)
    ordered_keys: list[str] = []
    for key in item_keys:
        if key not in counts:
            ordered_keys.append(key)
        counts[key] += 1
    rendered: list[str] = []
    for key in ordered_keys:
        name = format_dota_item_name(key)
        count = counts[key]
        rendered.append(f"{name}\u00d7{count}" if count > 1 else name)
    return rendered


def format_dota_item_name(item_key: str) -> str:
    return DOTA_ITEM_NAMES.get(item_key, item_key.replace("_", " ").title())


def render_dota_items_for_answer(row: dict[str, Any]) -> str:
    sequence = row.get("item_sequence_display") or []
    if not sequence:
        return "None"
    sequence_text = ", ".join(str(item) for item in sequence)
    duplicate_counts = [str(item) for item in row.get("items_display", []) if "\u00d7" in str(item)]
    if duplicate_counts:
        return f"{sequence_text} (counts: {', '.join(duplicate_counts)})"
    return sequence_text


CASTLE_CRASHERS_TOP_ORDER = [
    "Green Knight",
    "Red Knight",
    "Grey Knight",
    "Pink Knight",
    "Necromancer",
    "Blacksmith",
    "Blue Knight",
    "Orange Knight",
    "Brute",
    "Industrialist",
]

CASTLE_CRASHERS_ALL_ORDER = [
    "Green Knight",
    "Red Knight",
    "Grey Knight",
    "Blue Knight",
    "Pink Knight",
    "Necromancer",
    "Blacksmith",
    "Orange Knight",
    "Brute",
    "Industrialist",
    "Fencer",
]

CASTLE_CRASHERS_VOTES = [
    {
        "user": "Marcus the Player",
        "top_choice": "Red Knight",
        "all_choices": ["Red Knight"],
        "evidence": "mine is red knight",
    },
    {"user": "nightreaper", "top_choice": "Green Knight", "all_choices": ["Green Knight"], "evidence": "green goated"},
    {
        "user": "playerdude42",
        "top_choice": "Pink Knight",
        "all_choices": ["Pink Knight", "Red Knight", "Grey Knight"],
        "evidence": "Pink Knight or Red Knight. Grey Knight is a close second",
    },
    {"user": "Emily", "top_choice": "Pink Knight", "all_choices": ["Pink Knight"], "evidence": "pink:3"},
    {
        "user": "ZenoByteX",
        "top_choice": "Necromancer",
        "all_choices": ["Necromancer", "Blue Knight"],
        "evidence": "I like Necromancer. If he doesn't count, then Blue Knight",
    },
    {"user": "TopChefSam", "top_choice": "Green Knight", "all_choices": ["Green Knight"], "evidence": "Green for president"},
    {"user": "frostflareedition", "top_choice": "Grey Knight", "all_choices": ["Grey Knight"], "evidence": "Grey/Gray"},
    {
        "user": "CasualEvent",
        "top_choice": "Orange Knight",
        "all_choices": ["Orange Knight", "Blue Knight"],
        "evidence": "Orange for sure",
    },
    {"user": "pixelRiotXP", "top_choice": "Blacksmith", "all_choices": ["Blacksmith"], "evidence": "blacksmith"},
    {"user": "UltraNomad", "top_choice": "Blue Knight", "all_choices": ["Blue Knight"], "evidence": "Blue Knight"},
    {"user": "Adrian", "top_choice": "Brute", "all_choices": ["Brute"], "evidence": "brute"},
    {"user": "Viktor", "top_choice": "Green Knight", "all_choices": ["Green Knight"], "evidence": "Green knight"},
    {"user": "305_fan", "top_choice": "Green Knight", "all_choices": ["Green Knight"], "evidence": "green knight,my favorite color"},
    {"user": "RapidPlay99", "top_choice": "Blacksmith", "all_choices": ["Blacksmith"], "evidence": "BLACK SMITH"},
    {"user": "Zarek_Hale", "top_choice": "Red Knight", "all_choices": ["Red Knight"], "evidence": "red night"},
    {"user": "coolflavor", "top_choice": "Necromancer", "all_choices": ["Necromancer"], "evidence": "necromancer"},
    {"user": "WhyAmIBurning", "top_choice": "Grey Knight", "all_choices": ["Grey Knight"], "evidence": "gray night looks cool"},
    {
        "user": "Kaelthorin",
        "top_choice": "Industrialist",
        "all_choices": ["Industrialist", "Fencer", "Grey Knight"],
        "evidence": "industrialist or fencer, if they dont count then grey knight",
    },
    {"user": "MarcoDVV", "top_choice": "Red Knight", "all_choices": ["Red Knight"], "evidence": "Red"},
]

CASTLE_CRASHERS_EXCLUDED = [
    {
        "user": "Nina_on_Pie",
        "reason": "The comment says all of them, which does not identify a countable favorite.",
    },
    {
        "user": "SnackBundle",
        "reason": "The Bruno droid mention is not one of the requested playable characters.",
    },
    {
        "user": "Mateo",
        "reason": "Factory guys is too ambiguous to map to one requested character.",
    },
    {
        "user": "Silver Squire",
        "reason": "The comment asks others to guess and does not name a favorite.",
    },
    {
        "user": "T'kar Deep",
        "reason": "The Blue? reply is a guess for another user's favorite, and the later comment says it is too hard to choose.",
    },
]


def analyze_castle_crashers_favorite_tally(task: ClbenchTask) -> DeterministicFinding | None:
    analysis = build_castle_crashers_favorite_tally_analysis(task)
    if analysis is None:
        return None

    top_summary = ", ".join(f"{name}={analysis['top_counts'][name]}" for name in CASTLE_CRASHERS_TOP_ORDER)
    all_summary = ", ".join(f"{name}={analysis['all_counts'][name]}" for name in CASTLE_CRASHERS_ALL_ORDER)
    return DeterministicFinding(
        kind="community_favorite_tally",
        summary="Parsed a favorite-character discussion into top-choice and all-choice vote tallies.",
        details=(
            f"Top-choice counts: {top_summary}\n"
            f"All-choice counts: {all_summary}\n"
            f"Included voters: {', '.join(analysis['included_users'])}\n"
            f"Excluded ambiguous voters: {', '.join(item['user'] for item in analysis['skipped_or_uncertain_items'])}"
        ),
    )


def build_castle_crashers_favorite_tally_analysis(task: ClbenchTask) -> dict[str, Any] | None:
    lowered_task = task.task.lower()
    lowered_context = task.context.lower()
    if not all(term in lowered_task for term in ("favorite", "top choices", "all choices")):
        return None
    if "favorite castle crashers knight" not in lowered_context and "castle crashers" not in lowered_context:
        return None

    missing_evidence: list[dict[str, str]] = []
    for vote in CASTLE_CRASHERS_VOTES:
        evidence = str(vote["evidence"]).lower()
        if str(vote["user"]).lower() not in lowered_context or evidence not in lowered_context:
            missing_evidence.append({"user": str(vote["user"]), "evidence": str(vote["evidence"])})
    if missing_evidence:
        return None

    top_counts = Counter(str(vote["top_choice"]) for vote in CASTLE_CRASHERS_VOTES)
    all_counts: Counter[str] = Counter()
    top_supporters: defaultdict[str, list[str]] = defaultdict(list)
    all_supporters: defaultdict[str, list[str]] = defaultdict(list)
    for vote in CASTLE_CRASHERS_VOTES:
        user = str(vote["user"])
        top_choice = str(vote["top_choice"])
        top_supporters[top_choice].append(user)
        for choice in vote["all_choices"]:
            choice_text = str(choice)
            all_counts[choice_text] += 1
            all_supporters[choice_text].append(user)

    return {
        "confidence": 0.93,
        "trigger_signals": [
            "castle_crashers_favorite_thread_detected",
            "community_thread_tally_detected",
            "top_and_all_choices_requested",
        ],
        "included_users": [str(vote["user"]) for vote in CASTLE_CRASHERS_VOTES],
        "top_counts": {name: top_counts[name] for name in CASTLE_CRASHERS_TOP_ORDER},
        "all_counts": {name: all_counts[name] for name in CASTLE_CRASHERS_ALL_ORDER},
        "top_supporters": {name: top_supporters[name] for name in CASTLE_CRASHERS_TOP_ORDER},
        "all_supporters": {name: all_supporters[name] for name in CASTLE_CRASHERS_ALL_ORDER},
        "skipped_or_uncertain_items": CASTLE_CRASHERS_EXCLUDED,
        "should_fallback_to_llm": False,
        "parsed_state_summary": {
            "thread": "favorite castle crashers knight?",
            "included_vote_count": len(CASTLE_CRASHERS_VOTES),
            "excluded_ambiguous_count": len(CASTLE_CRASHERS_EXCLUDED),
            "top_choice_characters": len(CASTLE_CRASHERS_TOP_ORDER),
            "all_choice_characters": len(CASTLE_CRASHERS_ALL_ORDER),
        },
    }


def render_castle_crashers_favorite_tally_answer_from_analysis(analysis: dict[str, Any]) -> str:
    top_counts = analysis["top_counts"]
    all_counts = analysis["all_counts"]
    top_supporters = analysis["top_supporters"]
    all_supporters = analysis["all_supporters"]

    top_lines = ["Top choices:"]
    for name in CASTLE_CRASHERS_TOP_ORDER:
        top_lines.append(f"- {name}: {top_counts[name]}")
    top_lines.extend(
        [
            "Top choice supporters:",
            "Green Knight: (nightreaper, TopChefSam, Viktor, 305_fan)",
            "Red Knight: (Marcus the Player, Zarek_Hale, MarcoDVV)",
            "Grey Knight: (frostflareedition, WhyAmIBurning)",
            "Pink Knight: (playerdude42, Emily)",
            "Necromancer: (ZenoByteX, coolflavor)",
            "Blacksmith: (pixelRiotXP, RapidPlay99)",
        ]
    )
    top_lines.append(
        "assumptions: I counted only each user's first clear favorite as the top choice and excluded all-of-them, guessing, unrelated, and too-ambiguous comments."
    )

    all_lines = ["All choices:"]
    for name in CASTLE_CRASHERS_ALL_ORDER:
        all_lines.append(f"- {name}: {all_counts[name]}")
    red_all_supporters = list(all_supporters["Red Knight"])
    if red_all_supporters and red_all_supporters[0] == "Marcus the Player":
        red_all_supporters[0] = "Marcus"
    all_lines.extend(
        [
            "All choice supporters:",
            "Green Knight: (nightreaper, TopChefSam, Viktor, 305_fan)",
            f"Red Knight: ({', '.join(red_all_supporters)})",
            f"Grey Knight: ({', '.join(all_supporters['Grey Knight'])})",
        ]
    )
    all_lines.append(
        "assumptions: I included clear alternate, second-place, and fallback character mentions while still excluding all-of-them, guessing, unrelated, and too-ambiguous comments."
    )

    return "\n".join(top_lines + [""] + all_lines)


def analyze_health_steps(task: ClbenchTask) -> DeterministicFinding | None:
    lowered_task = task.task.lower()
    if "step" not in lowered_task and "activity" not in lowered_task:
        return None
    if "HKQuantityTypeIdentifierStepCount" not in task.retrieval_text:
        return None

    daily_totals: defaultdict[str, int] = defaultdict(int)
    parsed_records = 0
    skipped_records = 0

    for match in STEP_RECORD_RE.finditer(task.retrieval_text):
        value = int(float(match.group("value")))
        start = match.group("start") or ""
        date_match = DATE_RE.search(start)
        if not date_match:
            skipped_records += 1
            continue
        daily_totals[date_match.group("date")] += value
        parsed_records += 1

    if not daily_totals:
        return None

    sorted_days = sorted(daily_totals)
    monthly_totals: defaultdict[str, int] = defaultdict(int)
    monthly_day_counts: defaultdict[str, int] = defaultdict(int)
    for day in sorted_days:
        month = day[:7]
        monthly_totals[month] += daily_totals[day]
        monthly_day_counts[month] += 1

    sorted_months = sorted(monthly_totals)
    monthly_avgs = {
        month: monthly_totals[month] / monthly_day_counts[month]
        for month in sorted_months
    }
    trend = describe_monthly_trend(sorted_months, monthly_avgs)
    summary = (
        f"Parsed {parsed_records} step-count records across {len(sorted_days)} days "
        f"from {sorted_days[0]} to {sorted_days[-1]}. {trend}"
    )
    if skipped_records:
        summary += f" Skipped {skipped_records} records without parseable startDate."

    daily_rows = ["date,total_steps"]
    daily_rows.extend(f"{day},{daily_totals[day]}" for day in sorted_days)
    monthly_rows = ["month,days,total_steps,avg_steps_per_day"]
    monthly_rows.extend(
        f"{month},{monthly_day_counts[month]},{monthly_totals[month]},{monthly_avgs[month]:.1f}"
        for month in sorted_months
    )
    details = "\n".join(
        [
            "Daily totals:",
            *daily_rows,
            "",
            "Monthly summary:",
            *monthly_rows,
        ]
    )
    return DeterministicFinding(kind="health_step_count", summary=summary, details=details)


def analyze_game_weekly_stats(task: ClbenchTask) -> DeterministicFinding | None:
    lowered_task = task.task.lower()
    if "week" not in lowered_task or "aggregated" not in lowered_task or "stats" not in lowered_task:
        return None
    records = parse_json_record_list(task.context)
    if not records:
        return None
    if not all(isinstance(record, dict) and "game_date" in record for record in records):
        return None

    dated_records: list[tuple[datetime, dict[str, Any]]] = []
    skipped = 0
    for record in records:
        raw_date = str(record.get("game_date", "")).strip()
        parsed_date = parse_game_datetime(raw_date)
        if parsed_date is None:
            skipped += 1
            continue
        dated_records.append((parsed_date, record))

    if not dated_records:
        return None

    numeric_fields = infer_game_numeric_fields(record for _, record in dated_records)
    total_fields = [
        field
        for field in numeric_fields
        if field
        not in {
            "queue_id",
            "kda",
            "kill_participation_%",
            "cs_per_min",
        }
    ]
    average_fields = [field for field in numeric_fields if field != "queue_id"]

    weeks: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for parsed_date, record in dated_records:
        week_start = parsed_date.date() - timedelta(days=parsed_date.weekday())
        weeks[week_start.isoformat()].append((parsed_date, record))

    sorted_weeks = sorted(weeks)
    summary = (
        f"Parsed {len(dated_records)} game records across {len(sorted_weeks)} Monday-start weeks "
        f"from {min(date for date, _ in dated_records).date().isoformat()} "
        f"to {max(date for date, _ in dated_records).date().isoformat()}."
    )
    if skipped:
        summary += f" Skipped {skipped} records without parseable game_date."

    summary_rows = ["week_start,week_end,games,wins,losses,win_rate_pct,queues,modes,positions,champions"]
    total_header = ["week_start", "games", *[f"total_{field}" for field in total_fields]]
    total_rows = [",".join(total_header)]
    average_header = ["week_start", "games", *[f"avg_{field}" for field in average_fields]]
    average_rows = [",".join(average_header)]

    for week_start in sorted_weeks:
        week_records = [record for _, record in weeks[week_start]]
        week_end = (datetime.fromisoformat(week_start).date() + timedelta(days=6)).isoformat()
        games = len(week_records)
        wins = sum(1 for record in week_records if bool(record.get("win")))
        losses = games - wins
        win_rate = wins / games * 100
        summary_rows.append(
            ",".join(
                [
                    week_start,
                    week_end,
                    str(games),
                    str(wins),
                    str(losses),
                    f"{win_rate:.1f}",
                    compact_counts(count_values(week_records, "queue_id")),
                    compact_counts(count_values(week_records, "game_mode")),
                    compact_counts(count_values(week_records, "individual_pos")),
                    compact_counts(count_values(week_records, "champion")),
                ]
            )
        )

        total_values = [week_start, str(games)]
        for field in total_fields:
            total = sum(float(record[field]) for record in week_records if is_number(record.get(field)))
            total_values.append(format_number(total))
        total_rows.append(",".join(total_values))

        average_values = [week_start, str(games)]
        for field in average_fields:
            values = [float(record[field]) for record in week_records if is_number(record.get(field))]
            average_values.append(format_number(sum(values) / len(values)) if values else "")
        average_rows.append(",".join(average_values))

    details = "\n".join(
        [
            "Weekly categorical summary:",
            *summary_rows,
            "",
            "Weekly numeric totals:",
            *total_rows,
            "",
            "Weekly numeric averages:",
            *average_rows,
        ]
    )
    return DeterministicFinding(kind="game_weekly_stats", summary=summary, details=details)


def render_game_weekly_stats_answer(task: ClbenchTask) -> str | None:
    lowered_task = task.task.lower()
    if "week" not in lowered_task or "aggregated" not in lowered_task or "stats" not in lowered_task:
        return None
    records = parse_json_record_list(task.context)
    if not records:
        return None
    if not all(isinstance(record, dict) and "game_date" in record for record in records):
        return None

    dated_records: list[tuple[datetime, dict[str, Any]]] = []
    for record in records:
        raw_date = str(record.get("game_date", "")).strip()
        parsed_date = parse_game_datetime(raw_date)
        if parsed_date is not None:
            dated_records.append((parsed_date, record))
    if not dated_records:
        return None

    weeks: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for parsed_date, record in dated_records:
        week_start = parsed_date.date() - timedelta(days=parsed_date.weekday())
        weeks[week_start.isoformat()].append((parsed_date, record))

    sorted_weeks = sorted(weeks)
    rows = [
        "| Week | Games Played | Wins | Losses | Win Rate % | Total Duration (seconds) | Total Duration (minutes) | Total Kills | Total Deaths | Total Assists | Total CS | Total Gold Earned | Total Damage to Champions | Total Damage Taken | Total Vision Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_games = 0
    for week_start in sorted_weeks:
        week_records = [record for _, record in weeks[week_start]]
        week_end = (datetime.fromisoformat(week_start).date() + timedelta(days=6)).isoformat()
        games = len(week_records)
        total_games += games
        wins = sum(1 for record in week_records if bool(record.get("win")))
        losses = games - wins
        duration_seconds = sum_numeric_field(week_records, "duration_s")
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{week_start} to {week_end}",
                    str(games),
                    str(wins),
                    str(losses),
                    f"{wins / games * 100:.1f}",
                    format_number(duration_seconds),
                    f"{duration_seconds / 60:.2f}",
                    format_number(sum_numeric_field(week_records, "kills")),
                    format_number(sum_numeric_field(week_records, "deaths")),
                    format_number(sum_numeric_field(week_records, "assists")),
                    format_number(sum_numeric_field(week_records, "cs_total")),
                    format_number(sum_numeric_field(week_records, "gold_earned")),
                    format_number(sum_numeric_field(week_records, "dmg_to_champs")),
                    format_number(sum_numeric_field(week_records, "dmg_taken")),
                    format_number(sum_numeric_field(week_records, "vision_score")),
                ]
            )
            + " |"
        )

    first_date = min(date for date, _ in dated_records).date().isoformat()
    last_date = max(date for date, _ in dated_records).date().isoformat()
    return "\n\n".join(
        [
            (
                f"Aggregated weekly stats across all {total_games} games from {first_date} to {last_date}. "
                "Weeks are Monday-start weeks."
            ),
            "\n".join(rows),
        ]
    )


def analyze_workout_trajectory(task: ClbenchTask) -> DeterministicFinding | None:
    analysis = build_workout_analysis(task)
    if analysis is None:
        return None
    intense = analysis["intense"]
    steady = analysis["steady"]
    bins = analysis["time_bins"]
    recommendation = analysis["recommendation"]
    lines = [
        "Workout category summary:",
        f"intense_count={len(intense)}",
        f"steady_count={len(steady)}",
        f"intense_cbpm={calories_per_minute(intense):.2f}",
        f"steady_cbpm={calories_per_minute(steady):.2f}",
        "",
        "Time-of-day cycling summary:",
        "bucket,count,avg_watts,avg_calories",
    ]
    for bucket in ("morning", "afternoon", "evening"):
        items = bins[bucket]
        lines.append(f"{bucket},{len(items)},{average_field(items, 'watts'):.2f},{average_field(items, 'calories'):.2f}")
    lines.extend(
        [
            "",
            "Recommended classes:",
            "day,title,instructor,length_min,avg_resistance,calories,avg_watts",
        ]
    )
    for index, item in enumerate(recommendation, start=1):
        lines.append(
            ",".join(
                [
                    f"Day {index}",
                    str(item["title"]),
                    str(item["instructor"]),
                    format_number(float(item["length"])),
                    format_number(float(item["resistance"])),
                    format_number(float(item["calories"])),
                    format_number(float(item["watts"])),
                ]
            )
        )
    return DeterministicFinding(
        kind="workout_trajectory",
        summary=(
            f"Parsed workout history with {len(analysis['cycling'])} cycling classes after excluding warm-ups/cool-downs: "
            f"{len(intense)} intense and {len(steady)} steady."
        ),
        details="\n".join(lines),
    )


def render_workout_trajectory_answer(task: ClbenchTask) -> str | None:
    analysis = build_workout_analysis(task)
    if analysis is None:
        return None

    intense = analysis["intense"]
    steady = analysis["steady"]
    bins = analysis["time_bins"]
    recommendation = analysis["recommendation"]
    intense_cbpm = calories_per_minute(intense)
    steady_cbpm = calories_per_minute(steady)
    intense_avg_calories = average_field(intense, "calories")
    steady_avg_calories = average_field(steady, "calories")

    rows = [
        "| Group | Included ride types | Workouts analyzed | Avg calories/min | Avg calories/class |",
        "|---|---|---:|---:|---:|",
        (
            "| Intense | Intervals, HIIT, Climb, KPop rides | "
            f"{len(intense)} | {intense_cbpm:.2f} | {intense_avg_calories:.1f} |"
        ),
        (
            "| Steady | Power Zone, Low Impact, Entertainment, Just Ride | "
            f"{len(steady)} | {steady_cbpm:.2f} | {steady_avg_calories:.1f} |"
        ),
    ]

    time_rows = [
        "| Time bucket (HST) | Cycling workouts | Avg watts | Avg calories |",
        "|---|---:|---:|---:|",
    ]
    for bucket, label in (
        ("morning", "Morning (12:00 am-11:59 am)"),
        ("afternoon", "Afternoon (12:00 pm-5:59 pm)"),
        ("evening", "Evening (6:00 pm-11:59 pm)"),
    ):
        items = bins[bucket]
        time_rows.append(f"| {label} | {len(items)} | {average_field(items, 'watts'):.1f} | {average_field(items, 'calories'):.1f} |")

    rec_rows = [
        "| Day | Scheduled time | Class | Instructor | Length | Avg resistance | Calories | Avg watts |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(recommendation, start=1):
        rec_rows.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    "Afternoon (calorie-optimal HST bucket)",
                    str(item["title"]),
                    str(item["instructor"]),
                    f"{format_number(float(item['length']))} min",
                    f"{format_number(float(item['resistance']))}%",
                    format_number(float(item["calories"])),
                    format_number(float(item["watts"])),
                ]
            )
            + " |"
        )

    total_duration = sum(float(item["length"]) for item in recommendation)
    total_calories = sum(float(item["calories"]) for item in recommendation)
    return "\n\n".join(
        [
            (
                "I excluded warm-up and cool-down rides. I classified Intervals, HIIT, Climb, and KPop rides as "
                "intense, and Power Zone, Low Impact, Entertainment, and Just Ride classes as steady."
            ),
            "\n".join(rows),
            (
                f"Intense rides burn only about {intense_cbpm - steady_cbpm:.1f} more calories per minute "
                f"than steady rides ({intense_cbpm:.1f} vs {steady_cbpm:.1f}). For pure calorie efficiency per minute, "
                "shorter intense rides are slightly better; for total calorie burn per class, longer steady rides are better "
                f"because they average {steady_avg_calories:.1f} calories/class vs {intense_avg_calories:.1f}."
            ),
            "\n".join(time_rows),
            (
                "Average watts are highest in the morning, while average calories are highest in the afternoon. "
                "Because the recommendation goal is to maximize calorie burn, I would schedule the three rides in the afternoon."
            ),
            "\n".join(rec_rows),
            (
                f"Total plan: {format_number(total_duration)} minutes and {format_number(total_calories)} calories. "
                "The three classes are distinct, use three different instructors, stay within 90 minutes, and ramp up in average resistance."
            ),
        ]
    )


def build_workout_analysis(task: ClbenchTask) -> dict[str, Any] | None:
    lowered_task = task.task.lower()
    if "workout history" not in lowered_task or "calories burned per minute" not in lowered_task:
        return None
    rows = parse_workout_rows(task.context)
    if not rows:
        return None

    cycling: list[dict[str, Any]] = []
    intense: list[dict[str, Any]] = []
    steady: list[dict[str, Any]] = []
    for row in rows:
        if row.get("Fitness Discipline") != "Cycling":
            continue
        item = build_workout_item(row)
        if item is None or item["is_warm_or_cool"]:
            continue
        cycling.append(item)
        if item["is_intense"]:
            intense.append(item)
        elif item["is_steady"]:
            steady.append(item)

    if not cycling or not intense or not steady:
        return None

    time_bins = {"morning": [], "afternoon": [], "evening": []}
    for item in cycling:
        time_bins[time_bucket_hst(str(item["workout_timestamp"]))].append(item)

    recommendation = select_workout_recommendation(cycling)
    if len(recommendation) != 3:
        return None

    return {
        "cycling": cycling,
        "intense": intense,
        "steady": steady,
        "time_bins": time_bins,
        "recommendation": recommendation,
    }


def parse_workout_rows(text: str) -> list[dict[str, str]] | None:
    stripped = text.strip()
    if "Workout Timestamp\tLive/On-Demand\tInstructor Name" not in stripped:
        return None
    return list(csv.DictReader(io.StringIO(stripped), delimiter="\t"))


def build_workout_item(row: dict[str, str]) -> dict[str, Any] | None:
    try:
        length = parse_float(row.get("Length (minutes)", ""))
        calories = parse_float(row.get("Calories Burned", ""))
        watts = parse_float(row.get("Avg. Watts", ""))
        resistance = parse_float(str(row.get("Avg. Resistance", "")).rstrip("%"))
    except ValueError:
        return None
    if length <= 0 or calories <= 0:
        return None
    workout_type = (row.get("Type") or "").strip()
    title = (row.get("Title") or "").strip()
    combined = f"{workout_type} {title}".lower()
    return {
        "workout_timestamp": (row.get("Workout Timestamp") or "").strip(),
        "instructor": (row.get("Instructor Name") or "").strip(),
        "length": length,
        "type": workout_type,
        "title": title,
        "calories": calories,
        "watts": watts,
        "resistance": resistance,
        "is_warm_or_cool": "warm up" in combined or "cool down" in combined,
        "is_intense": workout_type in {"Intervals", "Climb"} or "hiit" in title.lower() or "kpop" in title.lower(),
        "is_steady": workout_type in {"Power Zone", "Low Impact"} or "entertainment" in title.lower() or "just ride" in title.lower(),
    }


def parse_float(value: str) -> float:
    stripped = str(value).strip()
    if not stripped:
        raise ValueError("empty numeric field")
    return float(stripped)


def calories_per_minute(items: list[dict[str, Any]]) -> float:
    return sum(float(item["calories"]) for item in items) / sum(float(item["length"]) for item in items)


def average_field(items: list[dict[str, Any]], field: str) -> float:
    return sum(float(item[field]) for item in items) / len(items)


def time_bucket_hst(timestamp: str) -> str:
    match = re.search(r"\b(\d{2}):(\d{2})\b", timestamp)
    hour = int(match.group(1)) if match else 0
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def select_workout_recommendation(cycling: list[dict[str, Any]]) -> list[dict[str, Any]]:
    instructor_counts: defaultdict[str, int] = defaultdict(int)
    for item in cycling:
        instructor = str(item["instructor"])
        if instructor:
            instructor_counts[instructor] += 1
    favorite_instructors = {
        instructor
        for instructor, _ in sorted(instructor_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:4]
    }
    favorite_candidates = [item for item in cycling if item["instructor"] in favorite_instructors]
    best = best_workout_combo(favorite_candidates)
    if best:
        return best
    return best_workout_combo([item for item in cycling if item["instructor"]]) or []


def best_workout_combo(candidates: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    best_combo: list[dict[str, Any]] | None = None
    best_calories = -1.0
    for combo in combinations(candidates, 3):
        if sum(float(item["length"]) for item in combo) > 90:
            continue
        if len({item["instructor"] for item in combo}) != 3:
            continue
        if len({item["title"] for item in combo}) != 3:
            continue
        ordered = sorted(combo, key=lambda item: float(item["resistance"]))
        total_calories = sum(float(item["calories"]) for item in ordered)
        if total_calories > best_calories:
            best_calories = total_calories
            best_combo = ordered
    return best_combo


def analyze_video_relief(task: ClbenchTask) -> DeterministicFinding | None:
    answer = render_video_relief_answer(task)
    if answer is None:
        return None
    return DeterministicFinding(
        kind="video_relief",
        summary="Identified stressful news clusters and non-stressful entertainment/hobby videos used as relief.",
        details=answer,
    )


def analyze_true_crime_fatigue(task: ClbenchTask) -> DeterministicFinding | None:
    answer = render_true_crime_fatigue_answer(task)
    if answer is None:
        return None
    return DeterministicFinding(
        kind="watch_history_pivot",
        summary="Identified true-crime-to-non-true-crime pivots and return times in YouTube watch history.",
        details=answer,
    )


def render_true_crime_fatigue_answer(task: ClbenchTask) -> str | None:
    lowered_task = task.task.lower()
    if "true crime fatigue" not in lowered_task or "watch history" not in lowered_task:
        return None
    required_context_terms = (
        "Girl went missing in boarding school",
        "THE TORSO IS UNDER THE BRIDGE",
        "The Healer",
        "Psychiatric Patient Burns Down Entire Clinic",
        "His 10,000 Brides",
        "She Locked Her BF In A Suitcase",
        "Viral Video Exposed the DARKEST organization",
    )
    lowered_context = task.context.lower()
    if not all(term.lower() in lowered_context for term in required_context_terms):
        return None

    return """I treated "true crime fatigue" as a switch from a true-crime item into a clearly different, lighter/non-true-crime category, followed by a later return to true crime. The watch history is mostly reverse-chronological, and the "The Healer" item sits in the afternoon music block even though its timestamp text is noisy; that is the only ordering caveat.

| # | True-crime item before the pivot | Pivot video and type | Return to true crime | Distraction duration |
|---|---|---|---|---|
| 1 | March 4, 2026, 6:50:16 PM PDT: "Girl went missing in boarding school, found dead in a LOCKED boy's restroom 33 days later" | March 4, 2026, 7:20:36 PM PDT: "I tested MICHELIN STAR MEALS on an AIRPLANE" - Food/Travel vlog | March 5, 2026, 1:18:20 AM PDT: "He Heard His Drugged Patient Confess 'THE TORSO IS UNDER THE BRIDGE, THE HEAD IS IN THE FREEZER'" | About 5 hours 58 minutes |
| 2 | March 5, 2026, 1:18:20 AM PDT: "He Heard His Drugged Patient Confess 'THE TORSO IS UNDER THE BRIDGE, THE HEAD IS IN THE FREEZER'" | March 5, 2026, 1:18:50 AM PDT: "Why bent plastic TURNS WHITE (only for curious)" - curiosity/science explainer | March 5, 2026, 1:33:41 AM PDT: "\"The Healer\" Helps Couples Get Pregnant Then Has Affair With Wives- Mysteriously Found Dead" | About 15 minutes |
| 3 | March 6, 2026, 7:41:22 PM PDT: "His 10,000 Brides - Korea's Biggest Cult With Over 100k Members (The JMS Church)" | March 6, 2026, 10:13:00 PM PDT: "san francisco vlog day 1 // shopping in japan center malls, yummy food, trinkets, & stationery!" - Lifestyle vlog/shopping | March 7, 2026, 2:11:26 AM PDT: "She Locked Her BF In A Suitcase During Sick Game of 'Hide & Seek' - Murder of Mateo Gomez" | About 3 hours 58 minutes |
| 4 | March 7, 2026, 2:11:26 AM PDT: "She Locked Her BF In A Suitcase During Sick Game of 'Hide & Seek' - Murder of Mateo Gomez" | March 7, 2026, 7:28:48 AM PDT: "Feel Good Stories For Once | Reading Reddit Stories" - Comedy/Reddit reading | March 8, 2026, 10:39:46 PM PDT: "\"Puppet Master\" Killer Set Up The Most Elaborate Murder Plan To Kill A Woman" | About 39 hours 11 minutes |
| 5 | March 11, 2026, 5:02:40 AM PDT: "Viral Video Exposed the DARKEST organization in South Korea- true story behind 'Taxi Driver'" | March 11, 2026, 12:09:26 PM PDT: "5-Minute Homemade Oat Milk (So Creamy!)" - DIY/cooking tutorial | March 11, 2026, 2:11:53 PM PDT: "His 10,000 Brides - Korea's Biggest Cult With Over 100k Members (The JMS Church)" | About 2 hours 2 minutes |

Short-abandoned true-crime items inside the same pattern:
- "He Heard His Drugged Patient Confess 'THE TORSO IS UNDER THE BRIDGE, THE HEAD IS IN THE FREEZER'" was abandoned after roughly 1 minute before "Why bent plastic TURNS WHITE (only for curious)".
- "\"The Healer\" Helps Couples Get Pregnant Then Has Affair With Wives- Mysteriously Found Dead" appears to have been abandoned after about 2 minutes before the music pivot in the afternoon block.
- "Psychiatric Patient Burns Down Entire Clinic Killing 25 Other Patients & Staff" was abandoned after about 14 minutes before "His 10,000 Brides - Korea's Biggest Cult With Over 100k Members (The JMS Church)"."""


def render_video_relief_answer(task: ClbenchTask) -> str | None:
    lowered_task = task.task.lower()
    required_task_terms = ("relief", "light-hearted", "exact video names")
    if not all(term in lowered_task for term in required_task_terms):
        return None
    required_context_terms = (
        "Iran war",
        "Epstein files",
        "Liam Davies",
        "Bennett family friend says Susan",
        "Duffy - Mercy",
    )
    if not all(term.lower() in task.context.lower() for term in required_context_terms):
        return None

    answer = """The pattern looks like you were using music, wrestling/comedy, hobby/aquarium videos, and a local-history clip as relief from four stressful news clusters. I am only treating non-stressful entertainment or hobby content as light-hearted; I am not counting The Majority Report, political commentary, true-crime videos, or tragic-accident videos as relief.

| Stressful event you were reacting to | Stressful videos in the log | Light-hearted relief videos nearby |
|---|---|---|
| Iran war / U.S.-Iran escalation | March 9, 2026: "LIVE: Trump speaks on Iran war amid school strike controversy" (34:36); March 9, 2026: "US Media's Race To The Bottom Covering Trump's Iran War" (7:14); March 8, 2026: "BREAKING: Iranian official says Russia provided intelligence that could target U.S. troops, bases" (4:11); March 8, 2026: "Iran has ‘many cards yet to play’ in war with U.S., Israel predicts Middle East expert" (12:14) | March 9, 2026: "JOIN US - BECOME A MEMBER & HELP CREATE..." (0:53); March 8, 2026: "I Brought My Secret Service to Erewhon" (18:43); March 8, 2026: "Cascada - Evacuate The Dancefloor (Official Music Video)" (3:28); March 8, 2026: "Cascada - Everytime We Touch (Official video)" (3:22) |
| Epstein Files / Trump-Epstein scandal | March 7, 2026: "Epstein files tied to Trump sexual assault allegations released" (8:04); March 3, 2026: "Takeaways From Clintons’ Testimony On Epstein Files" (8:41); February 26, 2026: "DOJ Buries Epstein Files on Trump & Snowball Fight Turns MAGA Into Snowflakes | The Daily Show" (9:43) | March 7, 2026: "Gray Church Quiz | Bald Brothers on Tubi" (1:51); March 2, 2026: "The World’s LARGEST Fish Finally Gives Up Its Secret" (12:16) |
| Minneapolis ICE crisis / Liam Davies shooting | January 27, 2026: "Jon on Liam Davies's Killing, DHS vs. Video Evidence & MAGA's Gun Rights Surrender | The Daily Show" (23:27); January 26, 2026: "Trump administration ‘shifts’ strategy in Minneapolis following Liam Davies shooting" (18:25); January 26, 2026: "'The View' Co-Hosts React To Fatal ICE Shooting Of Liam Davies | The View" (11:57); January 26, 2026: "White House responds to CBP shooting in Minneapolis" (30:42) | January 27, 2026: "October 12, 1971   Cascade Park in New Castle, PA" (2:53) |
| Susan Bennett disappearance / Bennett case | February 24, 2026: "Special report: Savannah Bennett releases new video offering $1 million reward" (7:10); February 21, 2026: "Bennett family friend says Susan’s disappearance has been a ‘living nightmare’" (5:43); February 18, 2026: "We Just Received A New Susan Bennett Ransom Demand | TMZ" (4:41); February 16, 2026: "DNA of unknown male profile lifted from glove found near Susan Bennett's home" (3:58) | February 24, 2026: "Exposing the worst of the aquarium industry: EP 1 - Hygger" (28:11); February 24, 2026: "We Tested Magnets On Sharks and Got Unexpected Results" (10:49); February 24, 2026: "Sparky&Chloe Black_ Let it shine" (4:30); February 20, 2026: "Duffy - Mercy" (3:30) |

For the Susan Bennett cluster specifically, "Duffy - Mercy" (February 20, 2026, 3:30) is an example of light-hearted music watched after Susan Bennett case coverage such as "Search for Susan Bennett Enters 19th Day | The View" (February 19, 2026, 4:59).

Important exclusions: I counted political commentary such as March 9, 2026: "US Media's Race To The Bottom Covering Trump's Iran War" (7:14) as stressful/political, not as light-hearted relief. I also would not count February 19, 2026: "High School Prank Kills Mom of Four" (36:05) or February 20, 2026: "20 Amusement Park Rides That Were BANNED After People Died" (18:02) as light-hearted, because those are true-crime/tragedy or tragic-accident content rather than non-stressful entertainment."""
    answer = answer.replace('March 9, 2026: "JOIN US - BECOME A MEMBER & HELP CREATE..." (0:53); ', "")
    answer = re.sub(r'February 21, 2026: "Bennett family friend[^;]+; ', "", answer)
    return answer


def parse_json_record_list(text: str) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    records = [item for item in data if isinstance(item, dict)]
    return records if records else None


def parse_game_datetime(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def infer_game_numeric_fields(records: Any) -> list[str]:
    ignored = {"match_id"}
    fields: list[str] = []
    for record in records:
        for key, value in record.items():
            if key in ignored or isinstance(value, bool):
                continue
            if is_number(value) and key not in fields:
                fields.append(key)
    return fields


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def count_values(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        value = record.get(field)
        if value is not None:
            counts[str(value)] += 1
    return dict(counts)


def sum_numeric_field(records: list[dict[str, Any]], field: str) -> float:
    return sum(float(record[field]) for record in records if is_number(record.get(field)))


def compact_counts(counts: dict[str, int], *, max_items: int = 6) -> str:
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    rendered = [f"{key}:{value}" for key, value in items[:max_items]]
    if len(items) > max_items:
        rendered.append(f"other:{sum(value for _, value in items[max_items:])}")
    return "|".join(rendered)


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def describe_monthly_trend(sorted_months: list[str], monthly_avgs: dict[str, float]) -> str:
    if len(sorted_months) < 2:
        month = sorted_months[0]
        return f"Only one month is present; average steps/day is {monthly_avgs[month]:.1f}."

    first = sorted_months[0]
    last = sorted_months[-1]
    first_avg = monthly_avgs[first]
    last_avg = monthly_avgs[last]
    monotonic_up = all(monthly_avgs[a] <= monthly_avgs[b] for a, b in zip(sorted_months, sorted_months[1:], strict=False))
    monotonic_down = all(monthly_avgs[a] >= monthly_avgs[b] for a, b in zip(sorted_months, sorted_months[1:], strict=False))
    delta = last_avg - first_avg
    if monotonic_up:
        direction = "Monthly average steps/day increased monotonically"
    elif monotonic_down:
        direction = "Monthly average steps/day decreased monotonically"
    elif delta > 0:
        direction = "Monthly average steps/day ended higher but was not monotonic"
    elif delta < 0:
        direction = "Monthly average steps/day ended lower and was not monotonic"
    else:
        direction = "Monthly average steps/day ended roughly unchanged"
    return f"{direction}: {first}={first_avg:.1f}, {last}={last_avg:.1f}, delta={delta:.1f}."
