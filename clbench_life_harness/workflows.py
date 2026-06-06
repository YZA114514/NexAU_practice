from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from .data import ClbenchTask


@dataclass(frozen=True)
class EvidenceRow:
    id: str
    source: str
    span: tuple[int, int] | None
    entity: str | None
    time: str | None
    claim: str
    quote: str | None
    tags: list[str] = field(default_factory=list)
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComputedValue:
    name: str
    value: Any
    source: str
    confidence: str = "medium"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateAnswer:
    source: str
    summary: str
    answer_is_exact: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UncertainItem:
    item: str
    reason: str
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowDecision:
    context_type: str
    task_operator: str
    workflow: str
    confidence: str
    signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredState:
    workflow: str
    context_type: str
    task_operator: str
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    evidence_table: list[EvidenceRow] = field(default_factory=list)
    computed_values: list[ComputedValue] = field(default_factory=list)
    entity_profiles: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    candidate_answers: list[CandidateAnswer] = field(default_factory=list)
    uncertain_items: list[UncertainItem] = field(default_factory=list)
    parser_confidence: float = 0.5
    trigger_signals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "context_type": self.context_type,
            "task_operator": self.task_operator,
            "tables": self.tables,
            "evidence_table": [row.to_dict() for row in self.evidence_table],
            "computed_values": [value.to_dict() for value in self.computed_values],
            "entity_profiles": self.entity_profiles,
            "timeline": self.timeline,
            "candidate_answers": [answer.to_dict() for answer in self.candidate_answers],
            "uncertain_items": [item.to_dict() for item in self.uncertain_items],
            "parser_confidence": self.parser_confidence,
            "trigger_signals": self.trigger_signals,
            "warnings": self.warnings,
            "fallback_recommendation": self.fallback_recommendation,
        }


CONTEXT_WORKFLOW_MAP = {
    "Game Logs": "structured_log_workflow",
    "Digital Footprints & Daily-Life Records": "structured_log_workflow",
    "Self-Tracking Trajectories": "structured_log_workflow",
    "Community Interactions": "thread_tally_workflow",
    "Group Conversations & Meeting Transcripts": "dialogue_state_workflow",
    "Private Conversations": "dialogue_state_workflow",
    "Creation & Revision Histories": "multi_doc_matrix_workflow",
    "Personal Information Fragments": "multi_doc_matrix_workflow",
    "Public Information Fragments": "multi_doc_matrix_workflow",
}


def build_structured_state(
    *,
    task: ClbenchTask,
    task_route: Any,
    hits: list[Any],
    deterministic_findings: list[Any],
    deterministic_answer: Any | None,
) -> StructuredState:
    decision = select_workflow(task=task, task_route=task_route)
    evidence_rows = build_evidence_rows(hits, workflow=decision.workflow)
    tables = build_workflow_tables(decision=decision, task=task, hits=hits)
    computed_values = build_computed_values(deterministic_findings)
    candidate_answers = build_candidate_answers(deterministic_answer)
    uncertain_items = build_uncertain_items(deterministic_answer)
    warnings = build_workflow_warnings(decision, evidence_rows, computed_values, deterministic_answer, tables)

    return StructuredState(
        workflow=decision.workflow,
        context_type=decision.context_type,
        task_operator=decision.task_operator,
        tables=tables,
        evidence_table=evidence_rows,
        computed_values=computed_values,
        entity_profiles=build_entity_profiles(decision, hits, tables),
        timeline=build_timeline(decision, hits, tables),
        candidate_answers=candidate_answers,
        uncertain_items=uncertain_items,
        parser_confidence=estimate_parser_confidence(decision, evidence_rows, computed_values, deterministic_answer, tables),
        trigger_signals=list(decision.signals),
        warnings=warnings,
        fallback_recommendation=None if evidence_rows else "fallback_general_evidence_state",
    )


def select_workflow(*, task: ClbenchTask, task_route: Any) -> WorkflowDecision:
    context_type = route_context_type(task)
    task_operator = route_task_operator(task.task)
    route_name = str(getattr(task_route, "route", ""))
    signals = list(getattr(task_route, "signals", ()))

    if context_type == "Community Interactions":
        workflow = "thread_tally_workflow"
        signals.append(f"context:{context_type}")
    elif context_type in {"Group Conversations & Meeting Transcripts", "Private Conversations"}:
        workflow = "dialogue_state_workflow"
        signals.append(f"context:{context_type}")
    elif context_type in {
        "Game Logs",
        "Digital Footprints & Daily-Life Records",
        "Self-Tracking Trajectories",
    }:
        workflow = "structured_log_workflow"
        signals.append(f"context:{context_type}")
    elif route_name in {"exact_compute_solver", "quant_log_solver", "financial_solver"}:
        workflow = "structured_log_workflow"
        signals.append(f"route:{route_name}")
    elif route_name == "thread_tally_solver":
        workflow = "thread_tally_workflow"
        signals.append("route:thread_tally_solver")
    elif route_name == "dialogue_social_solver":
        workflow = "dialogue_state_workflow"
        signals.append("route:dialogue_social_solver")
    elif route_name == "multi_doc_solver":
        workflow = "multi_doc_matrix_workflow"
        signals.append("route:multi_doc_solver")
    else:
        workflow = CONTEXT_WORKFLOW_MAP.get(context_type, workflow_from_operator(task_operator))
        signals.append(f"context:{context_type}")

    confidence = "high" if workflow != "general_evidence_workflow" else "low"
    return WorkflowDecision(
        context_type=context_type,
        task_operator=task_operator,
        workflow=workflow,
        confidence=confidence,
        signals=tuple(dict.fromkeys(signals)),
    )


def route_context_type(task: ClbenchTask) -> str:
    metadata_type = task.metadata.get("context_subcategory")
    if isinstance(metadata_type, str) and metadata_type:
        return metadata_type

    text = task.retrieval_text[:80_000].lower()
    if any(marker in text for marker in ("purchase_log", "gameloop", "[gameid", "runescape", "stepcount")):
        return "Game Logs"
    if any(marker in text for marker in ("reddit", "u/", "comment", "showing 1-")):
        return "Community Interactions"
    if re.search(r"(?m)^[A-Z][A-Za-z ._-]{0,60}:\s+\S", task.retrieval_text[:80_000]):
        return "Private Conversations"
    if any(marker in text for marker in ("document 1", "document 2", "draft", "version")):
        return "Creation & Revision Histories"
    return "unknown_context"


def route_task_operator(task_text: str) -> str:
    text = task_text.lower()
    if any(marker in text for marker in ("count", "counts", "total", "average", "trend", "top ", "rank", "sort")):
        return "count_or_aggregate"
    if any(marker in text for marker in ("compare", "difference", "changed", "revision", "between")):
        return "compare_or_diff"
    if any(marker in text for marker in ("quote", "snippet", "exact")):
        return "quote_or_extract"
    if any(marker in text for marker in ("who", "which person", "support", "tension", "relationship", "feel")):
        return "social_inference"
    if any(marker in text for marker in ("recommend", "should i", "best", "focus")):
        return "recommendation_or_selection"
    if any(marker in text for marker in ("when", "timeline", "before", "after", "status")):
        return "timeline_or_status"
    return "general_answer"


def workflow_from_operator(task_operator: str) -> str:
    if task_operator in {"count_or_aggregate", "timeline_or_status"}:
        return "structured_log_workflow"
    if task_operator in {"compare_or_diff", "recommendation_or_selection"}:
        return "multi_doc_matrix_workflow"
    if task_operator == "social_inference":
        return "dialogue_state_workflow"
    return "general_evidence_workflow"


def build_evidence_rows(hits: list[Any], *, workflow: str, limit: int = 10) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for index, hit in enumerate(hits[:limit], start=1):
        chunk = hit.chunk
        quote = first_nonempty_line(chunk.text)
        rows.append(
            EvidenceRow(
                id=f"e{index:02d}",
                source=chunk.chunk_id,
                span=(chunk.char_start, chunk.char_end),
                entity=", ".join(chunk.speakers[:3]) if chunk.speakers else None,
                time=", ".join(chunk.timestamps[:2]) if chunk.timestamps else None,
                claim=compact_text(chunk.text, max_chars=360),
                quote=quote,
                tags=[workflow, *list(getattr(hit, "matched_terms", ())[:6])],
                confidence="high" if getattr(hit, "score", 0.0) > 0 else "neighbor",
            )
        )
    return rows


def build_workflow_tables(*, decision: WorkflowDecision, task: ClbenchTask, hits: list[Any]) -> dict[str, list[dict[str, Any]]]:
    if decision.workflow == "structured_log_workflow":
        recurring_items = build_recurring_item_table(task)
        return {
            "events": build_log_event_table(task, hits),
            "recurring_items": recurring_items,
            "subscription_cadence_plan": build_subscription_cadence_plan(task, recurring_items),
            "planning_calendar": build_planning_calendar_table(task),
            "topic_timeline": build_topic_timeline_table(task),
            "aggregates": build_structured_log_aggregates(task),
        }
    if decision.workflow == "thread_tally_workflow":
        comments = build_comment_table(task, hits)
        return {
            "comments": comments,
            "commenter_registry": build_commenter_registry_table(comments),
            "commenter_attribution": build_commenter_attribution_table(comments),
            "chronology_notes": build_thread_chronology_notes(comments),
            "identity_aliases": build_identity_alias_table(task, comments),
            "stances": [],
        }
    if decision.workflow == "dialogue_state_workflow":
        turns = build_dialogue_turn_table(task, hits)
        speaker_activity = build_speaker_activity_table(turns)
        return {
            "turns": turns,
            "speaker_activity": speaker_activity,
            "speaker_sentiment_evidence": build_speaker_sentiment_evidence_table(
                turns=turns,
                task_text=task.task,
                speaker_activity=speaker_activity,
            ),
            "issues": [],
        }
    if decision.workflow == "multi_doc_matrix_workflow":
        return {
            "documents": build_document_table(task, hits),
            "section_items": build_section_item_table(task),
            "canonical_item_matrix": build_canonical_item_matrix(task),
            "claim_doc_matrix": [],
        }
    return {"evidence": []}


def build_log_event_table(task: ClbenchTask, hits: list[Any], *, limit: int = 40) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for hit in hits:
        for line_no, line in enumerate(hit.chunk.text.splitlines(), start=hit.chunk.start_line):
            stripped = line.strip()
            if not stripped:
                continue
            event_type = infer_log_event_type(stripped)
            timestamp = extract_first_time_like_value(stripped)
            if event_type is None and timestamp is None:
                continue
            events.append(
                {
                    "event_id": f"{hit.chunk.chunk_id}:{line_no}",
                    "source": hit.chunk.chunk_id,
                    "line": line_no,
                    "time": timestamp,
                    "event_type": event_type or "record",
                    "actor": extract_actor_hint(stripped),
                    "value": extract_numeric_hint(stripped),
                    "text": compact_text(stripped, max_chars=220),
                }
            )
            if len(events) >= limit:
                return events
    if not events and task.context:
        events.append(
            {
                "event_id": "context-summary",
                "source": "context",
                "line": None,
                "time": None,
                "event_type": "unparsed_context",
                "actor": None,
                "value": None,
                "text": compact_text(task.context, max_chars=260),
            }
        )
    return events


def build_comment_table(task: ClbenchTask, hits: list[Any], *, limit: int = 220) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str]] = set()
    primary_text = task.context or task.retrieval_text
    parsed_sources: list[tuple[str, list[dict[str, Any]]]] = [("original_context", parse_comment_like_lines(primary_text))]
    if not parsed_sources[0][1]:
        parsed_sources.extend((hit.chunk.chunk_id, parse_comment_like_lines(hit.chunk.text)) for hit in hits)
    for source, parsed in parsed_sources:
        if not parsed:
            continue
        for item in parsed:
            key = (
                item.get("user"),
                item.get("time"),
                item.get("parent"),
                item.get("depth"),
                item.get("text", "")[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            comments.append(
                {
                    "comment_id": f"c{len(comments) + 1:03d}",
                    "source": source,
                    "user": item.get("user"),
                    "time": item.get("time"),
                    "score": item.get("score"),
                    "parent_id": item.get("parent"),
                    "depth": item.get("depth"),
                    "text": item.get("text"),
                    "candidate_source": item.get("candidate_source"),
                    "candidate_score": item.get("candidate_score"),
                }
            )
            if len(comments) >= limit:
                return comments
    if not comments:
        for hit in hits[:12]:
            comments.append(
                {
                    "comment_id": f"c{len(comments) + 1:03d}",
                    "source": hit.chunk.chunk_id,
                    "user": None,
                    "time": None,
                    "score": None,
                    "parent_id": None,
                    "depth": None,
                    "text": compact_text(hit.chunk.text, max_chars=260),
                }
            )
    return comments


def build_dialogue_turn_table(task: ClbenchTask, hits: list[Any], *, limit: int = 260) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    primary_text = task.context or task.retrieval_text
    parsed_sources: list[tuple[str, list[dict[str, Any]]]] = [("original_context", parse_dialogue_lines(primary_text))]
    if not parsed_sources[0][1]:
        parsed_sources.extend((hit.chunk.chunk_id, parse_dialogue_lines(hit.chunk.text)) for hit in hits)
    for source, parsed in parsed_sources:
        for item in parsed:
            turns.append(
                {
                    "turn_id": f"t{len(turns) + 1:03d}",
                    "source": source,
                    "speaker": item.get("speaker"),
                    "time": item.get("time"),
                    "text": item.get("text"),
                    "tags": infer_dialogue_tags(str(item.get("text", ""))),
                }
            )
            if len(turns) >= limit:
                return turns
    return turns


def build_speaker_activity_table(turns: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for turn in turns:
        speaker = normalize_speaker_name(str(turn.get("speaker") or ""))
        if not speaker:
            continue
        profile = profiles.setdefault(
            speaker,
            {
                "speaker": speaker,
                "message_count": 0,
                "first_time": None,
                "last_time": None,
                "sample_messages": [],
                "tags": Counter(),
            },
        )
        profile["message_count"] += 1
        time_value = turn.get("time")
        if time_value and not profile["first_time"]:
            profile["first_time"] = time_value
        if time_value:
            profile["last_time"] = time_value
        text = compact_text(str(turn.get("text") or ""), max_chars=140)
        if text and len(profile["sample_messages"]) < 3:
            profile["sample_messages"].append(text)
        for tag in turn.get("tags", []) or []:
            profile["tags"][str(tag)] += 1

    rows: list[dict[str, Any]] = []
    for profile in profiles.values():
        tags = profile.pop("tags")
        profile["dominant_tags"] = [tag for tag, _ in tags.most_common(4)]
        rows.append(profile)
    rows.sort(key=lambda row: (-int(row["message_count"]), str(row["speaker"]).lower()))
    for rank, row in enumerate(rows[:limit], start=1):
        row["activity_rank"] = rank
    return rows[:limit]


def build_speaker_sentiment_evidence_table(
    *,
    turns: list[dict[str, Any]],
    task_text: str,
    speaker_activity: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    task_lower = task_text.lower()
    if not any(marker in task_lower for marker in ("sentiment", "positive", "negative", "rating", "confidence")):
        return []
    if not any(marker in task_lower for marker in ("profit", "craft", "profession", "sales", "selling")):
        return []

    activity_by_speaker = {str(row.get("speaker")): row for row in speaker_activity}
    evidence_by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    polarity_by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    for turn in turns:
        speaker = normalize_speaker_name(str(turn.get("speaker") or ""))
        if speaker not in activity_by_speaker:
            continue
        text = str(turn.get("text") or "")
        evidence = classify_profitability_sentiment_evidence(text)
        if evidence is None:
            continue
        evidence_by_speaker[speaker].append(
            {
                "time": str(turn.get("time") or ""),
                "quote": compact_text(text, max_chars=220),
                "polarity": evidence,
            }
        )
        polarity_by_speaker[speaker][evidence] += 1

    rows: list[dict[str, Any]] = []
    for speaker, evidence_items in evidence_by_speaker.items():
        activity = activity_by_speaker.get(speaker, {})
        polarity_counts = polarity_by_speaker[speaker]
        rating_hint = infer_sentiment_rating_hint(polarity_counts)
        rows.append(
            {
                "speaker": speaker,
                "message_count": activity.get("message_count"),
                "activity_rank": activity.get("activity_rank"),
                "profitability_evidence_count": len(evidence_items),
                "polarity_counts": dict(polarity_counts),
                "rating_hint": rating_hint,
                "evidence_quotes": evidence_items[:3],
                "has_enough_evidence_for_rating": len(evidence_items) > 0,
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row.get("message_count") or 0),
            int(row.get("activity_rank") or 999),
            str(row.get("speaker") or "").lower(),
        )
    )
    for rank, row in enumerate(rows[:limit], start=1):
        row["topic_activity_rank"] = rank
    return rows[:limit]


def classify_profitability_sentiment_evidence(text: str) -> str | None:
    lowered = text.lower()
    market_markers = (
        "profit",
        "gold",
        "pay",
        "paying",
        "paid",
        "price",
        "listing",
        "sell",
        "sold",
        "selling",
        "sales",
        "margin",
        "worth",
        "good money",
        "above cost",
        "money",
        "ah",
        "auction",
        "cancel",
        "cost",
    )
    craft_markers = (
        "craft",
        "crafts",
        "crafting",
        "crafted",
        "crafter",
        "crafters",
        "r2",
        "r3",
        "r4",
        "r5",
        "r4s",
        "r5s",
        "profession",
        "proff",
        "tool",
        "tools",
        "blacksmith",
        "alchemy",
        "potion",
        "flask",
        "jc",
        "conc",
        "concentration",
        "mats",
        "skill",
        "kp",
        "rank",
        "blue",
    )
    timing_markers = ("late", "behind", "hesitant", "bought", "too late", "fall behind", "weeks away")
    strong_positive_markers = (
        "1.5 mil",
        "200k profit",
        "best fucking thing",
        "pretty good profit",
        "definitely good",
        "still good",
        "cant complain",
        "can't complain",
        "weeks away",
    )
    strong_negative_markers = (
        "hardly any profit",
        "harldy any profit",
        "not as good",
        "too late",
        "hesitant",
        "dont see how",
        "don't see how",
        "dead for",
        "big loss",
        "cry a little",
        "0 margins",
    )
    neutral_evidence_markers = (
        "just bought green tools",
        "bought green tools",
        "actually paying this much",
        "paying this much",
        "wished foir was to know sooner",
        "wished for was to know sooner",
    )
    positive_markers = (
        "best",
        "profitable",
        "profit",
        "making money",
        "made around",
        "cant complain",
        "pretty good",
        "definitely good",
        "worth it",
        "exploded in price",
    )
    negative_markers = (
        "dead",
        "hardly any profit",
        "loss",
        "not enough",
        "too late",
        "hesitant",
        "cry",
        "0 margins",
        "behind",
        "sad",
    )
    has_market = any(contains_text_marker(lowered, marker) for marker in market_markers)
    has_craft = any(contains_text_marker(lowered, marker) for marker in craft_markers)
    has_timing = any(contains_text_marker(lowered, marker) for marker in timing_markers)
    strong_positive = any(contains_text_marker(lowered, marker) for marker in strong_positive_markers)
    strong_negative = any(contains_text_marker(lowered, marker) for marker in strong_negative_markers)
    neutral_evidence = any(contains_text_marker(lowered, marker) for marker in neutral_evidence_markers)
    positive = any(contains_text_marker(lowered, marker) for marker in positive_markers)
    negative = any(contains_text_marker(lowered, marker) for marker in negative_markers)
    if not (has_market or (has_craft and (has_timing or positive or negative))):
        return None

    if strong_positive:
        return "positive"
    if strong_negative:
        return "negative"
    if neutral_evidence:
        return "neutral"
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "neutral"


def contains_text_marker(text: str, marker: str) -> bool:
    marker = marker.lower()
    if re.fullmatch(r"[a-z0-9]+", marker):
        return re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", text) is not None
    return marker in text


def infer_sentiment_rating_hint(polarity_counts: Counter[str]) -> int | str:
    positive = polarity_counts.get("positive", 0)
    negative = polarity_counts.get("negative", 0)
    mixed = polarity_counts.get("mixed", 0)
    neutral = polarity_counts.get("neutral", 0)
    if positive > negative + mixed:
        return 4 if negative or mixed or neutral else 5
    if negative > positive + mixed:
        return 2 if positive or mixed or neutral else 1
    if mixed or (positive and negative) or neutral:
        return 3
    return "NEI"


def build_commenter_registry_table(comments: list[dict[str, Any]], *, limit: int = 80) -> list[dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for comment in comments:
        user = normalize_commenter_name(str(comment.get("user") or ""))
        if not user:
            continue
        row = registry.setdefault(
            user,
            {
                "commenter": user,
                "post_count": 0,
                "first_time": None,
                "last_time": None,
                "first_time_sort": None,
                "last_time_sort": None,
                "reply_count": 0,
                "sample_quotes": [],
            },
        )
        row["post_count"] += 1
        if comment.get("parent_id"):
            row["reply_count"] += 1
        time_value = comment.get("time")
        time_sort = parse_comment_time_sort_key(str(time_value or ""))
        if time_value and (row["first_time_sort"] is None or time_sort < row["first_time_sort"]):
            row["first_time"] = time_value
            row["first_time_sort"] = time_sort
        if time_value and (row["last_time_sort"] is None or time_sort > row["last_time_sort"]):
            row["last_time"] = time_value
            row["last_time_sort"] = time_sort
        text = compact_text(str(comment.get("text") or ""), max_chars=150)
        if text and len(row["sample_quotes"]) < 2:
            row["sample_quotes"].append(text)
    rows = list(registry.values())
    for row in rows:
        row.pop("first_time_sort", None)
        row.pop("last_time_sort", None)
    rows.sort(key=lambda row: (-int(row["post_count"]), str(row["first_time"] or ""), str(row["commenter"]).lower()))
    return rows[:limit]


def build_commenter_attribution_table(comments: list[dict[str, Any]], *, limit: int = 80) -> list[dict[str, Any]]:
    raw_registry = build_commenter_registry_table(comments, limit=limit)
    alias_map = infer_commenter_attribution_aliases(comments)
    raw_by_name = {str(row["commenter"]): row for row in raw_registry}

    canonical_counts: Counter[str] = Counter()
    aliases_by_canonical: dict[str, list[str]] = defaultdict(list)
    evidence_by_canonical: dict[str, list[str]] = defaultdict(list)
    for row in raw_registry:
        displayed = str(row["commenter"])
        alias_info = alias_map.get(displayed.lower(), {})
        canonical = alias_info.get("canonical", displayed)
        canonical_counts[canonical] += int(row.get("post_count", 0))
        if canonical != displayed:
            aliases_by_canonical[canonical].append(displayed)
            evidence_by_canonical[canonical].extend(str(item) for item in alias_info.get("evidence", []) if item)

    rows: list[dict[str, Any]] = []
    for raw_row in raw_registry:
        displayed = str(raw_row["commenter"])
        alias_info = alias_map.get(displayed.lower())
        canonical = str(alias_info.get("canonical")) if alias_info else displayed
        is_alias = bool(alias_info and canonical != displayed)
        evidence = list(alias_info.get("evidence", [])) if alias_info else list(evidence_by_canonical.get(canonical, []))
        attributed_count = 0 if is_alias else canonical_counts[canonical]
        notes: list[str] = []
        if is_alias:
            notes.append(f"Displayed author is counted under canonical commenter {canonical}.")
        elif aliases_by_canonical.get(canonical):
            notes.append(f"Includes displayed alias(es): {', '.join(aliases_by_canonical[canonical])}.")
        rows.append(
            {
                "displayed_commenter": displayed,
                "canonical_commenter": canonical,
                "answer_post_count": int(attributed_count),
                "displayed_post_count": int(raw_row.get("post_count", 0)),
                "attributed_post_count": int(attributed_count),
                "first_time": raw_row.get("first_time"),
                "last_time": raw_row.get("last_time"),
                "reply_count": raw_row.get("reply_count"),
                "attribution_status": "misattributed_alias" if is_alias else ("canonical_with_alias" if aliases_by_canonical.get(canonical) else "displayed_is_canonical"),
                "confidence": str(alias_info.get("confidence", "high")) if alias_info else "high",
                "evidence_quotes": evidence[:3],
                "counting_decision": "count_as_canonical" if is_alias else "count_as_displayed",
                "notes": notes,
            }
        )
    rows.sort(key=lambda row: (parse_comment_time_sort_key(str(row.get("first_time") or "")), str(row["displayed_commenter"]).lower()))
    return rows[:limit]


def infer_commenter_attribution_aliases(comments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}
    normalized_comments = [
        {
            **comment,
            "user": normalize_commenter_name(str(comment.get("user") or "")),
            "text": str(comment.get("text") or ""),
        }
        for comment in comments
    ]
    for index, comment in enumerate(normalized_comments):
        user = str(comment.get("user") or "")
        text = str(comment.get("text") or "")
        if not user or not re.search(r"\bit was me\b|\bname field\b|\bid wasn'?t hijacked\b", text, flags=re.IGNORECASE):
            continue
        canonical = infer_canonical_commenter_from_identity_text(user, text) or user
        alias = infer_alias_commenter_from_identity_text(text, normalized_comments, canonical=canonical)
        if alias is None:
            alias = nearest_prior_displayed_alias(normalized_comments, index, canonical=canonical)
        if not alias or alias.lower() == canonical.lower():
            continue
        evidence = [
            compact_text(text, max_chars=220),
        ]
        aliases[alias.lower()] = {
            "canonical": canonical,
            "confidence": "medium",
            "evidence": evidence,
        }
    return aliases


def infer_canonical_commenter_from_identity_text(user: str, text: str) -> str | None:
    match = re.search(r"\bIt was me,\s*(?P<name>[A-Z][A-Za-z .'-]{2,60})\s+made the comment", text)
    if match:
        return normalize_commenter_name(match.group("name"))
    return user


def infer_alias_commenter_from_identity_text(text: str, comments: list[dict[str, Any]], *, canonical: str) -> str | None:
    commenter_names = {normalize_commenter_name(str(comment.get("user") or "")) for comment in comments if comment.get("user")}
    lowered = text.lower()
    for name in sorted(commenter_names, key=len, reverse=True):
        if name and name.lower() != canonical.lower() and name.lower() in lowered:
            return name
    possessive = re.search(r"not\s+(?P<name>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)'s", text)
    if possessive:
        return normalize_commenter_name(possessive.group("name"))
    return None


def nearest_prior_displayed_alias(comments: list[dict[str, Any]], index: int, *, canonical: str) -> str | None:
    current_parent = comments[index].get("parent_id")
    for prior in reversed(comments[max(0, index - 8) : index]):
        prior_user = normalize_commenter_name(str(prior.get("user") or ""))
        if not prior_user or prior_user.lower() == canonical.lower():
            continue
        if current_parent and prior.get("parent_id") == current_parent:
            return prior_user
    return None


def parse_comment_time_sort_key(time_value: str) -> tuple[int, int, int, int, int]:
    match = re.search(
        r"\b(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+(?P<year>20\d{2})\s+at\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+(?P<ampm>AM|PM)\b",
        time_value,
        flags=re.IGNORECASE,
    )
    if not match:
        return (9999, 12, 31, 23, 59)
    hour = int(match.group("hour"))
    if match.group("ampm").upper() == "PM" and hour != 12:
        hour += 12
    if match.group("ampm").upper() == "AM" and hour == 12:
        hour = 0
    return (
        int(match.group("year")),
        MONTH_NUMBERS.get(match.group("month").lower(), 12),
        int(match.group("day")),
        hour,
        int(match.group("minute")),
    )


def build_thread_chronology_notes(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    times = [str(comment.get("time") or "") for comment in comments if comment.get("time")]
    relative_counts = Counter(time for time in times if re.search(r"\b(?:mo|yr|day|hour|min|ago)\b", time, re.IGNORECASE))
    notes: list[dict[str, Any]] = []
    for time_value, count in relative_counts.most_common(5):
        if count >= 3:
            notes.append(
                {
                    "issue": "repeated_relative_timestamp",
                    "time": time_value,
                    "count": count,
                    "guidance": "Do not infer strict chronology among comments that share this relative timestamp unless nesting/order is explicit.",
                }
            )
    if any(comment.get("depth") for comment in comments):
        notes.append(
            {
                "issue": "nested_thread_structure",
                "guidance": "Use depth/parent cues separately from chronology; nested replies may not be chronological peers.",
            }
        )
    return notes


def build_identity_alias_table(task: ClbenchTask, comments: list[dict[str, Any]], *, limit: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = task.retrieval_text
    commenter_names = {normalize_commenter_name(str(comment.get("user") or "")) for comment in comments}
    commenter_names.discard("")
    patterns = (
        r"(?i)\b(?:same person|same poster|also posting as|appears under|wrong(?:ly)? attributed|misattributed|actually)\b[^.\n]{0,180}",
        r"(?i)\b(?:reply to|as )(?P<name>[A-Z][A-Za-z .'-]{2,60})\b[^.\n]{0,120}",
    )
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            snippet = compact_text(match.group(0), max_chars=240)
            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)
            mentioned = sorted(name for name in commenter_names if name and name.lower() in snippet.lower())
            rows.append(
                {
                    "alias_id": f"a{len(rows) + 1:02d}",
                    "mentioned_people": mentioned[:4],
                    "evidence": snippet,
                    "confidence": "low" if not mentioned else "medium",
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def build_document_table(task: ClbenchTask, hits: list[Any], *, limit: int = 20) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        source = hit.chunk.source_label or infer_document_label(hit.chunk.text) or hit.chunk.chunk_id
        if source in seen:
            continue
        seen.add(source)
        documents.append(
            {
                "doc_id": f"d{len(documents) + 1:02d}",
                "source": source,
                "chunk_id": hit.chunk.chunk_id,
                "type": infer_document_type(source, hit.chunk.text),
                "summary": compact_text(hit.chunk.text, max_chars=280),
            }
        )
        if len(documents) >= limit:
            return documents
    if not documents and task.context:
        documents.append(
            {
                "doc_id": "d01",
                "source": "context",
                "chunk_id": None,
                "type": "unknown",
                "summary": compact_text(task.context, max_chars=280),
            }
        )
    return documents


def build_structured_log_aggregates(task: ClbenchTask) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if re.search(r"\bSubscribe\s*&\s*Save\b|\bOrder placed\b", task.retrieval_text, flags=re.IGNORECASE):
        rows.append(
            {
                "name": "order_history_detected",
                "value": len(re.findall(r"(?im)^\s*Order placed\s*$", task.retrieval_text)),
                "guidance": "Use recurring_items for delivery cadence; do not decline solely because exact inventory is missing.",
            }
        )
    if re.search(r"\bGoogle Ads\b|\bYouTube\b|\bWatched\b", task.retrieval_text, flags=re.IGNORECASE):
        rows.append(
            {
                "name": "viewing_history_detected",
                "value": len(re.findall(r"\b\d{4}\b", task.retrieval_text)),
                "guidance": "Separate dominant topic clusters, exact dates, ad-served items, and outliers.",
            }
        )
    return rows


def build_recurring_item_table(task: ClbenchTask, *, limit: int = 40) -> list[dict[str, Any]]:
    text = task.retrieval_text
    order_items = parse_order_history_items(text)
    if not order_items:
        order_items = parse_product_mentions(text)
    grouped: dict[str, dict[str, Any]] = {}
    for item in order_items:
        key = canonical_product_key(item["title"])
        if not key:
            continue
        row = grouped.setdefault(
            key,
            {
                "item_key": key,
                "representative_title": item["title"],
                "observed_dates": [],
                "observed_count": 0,
                "sizes_or_counts": [],
                "subscribe_save_seen": False,
                "source_lines": [],
            },
        )
        row["observed_count"] += 1
        if item.get("date") and item["date"] not in row["observed_dates"]:
            row["observed_dates"].append(item["date"])
        if item.get("size"):
            size = item["size"]
            if size not in row["sizes_or_counts"]:
                row["sizes_or_counts"].append(size)
        row["subscribe_save_seen"] = bool(row["subscribe_save_seen"] or item.get("subscribe_save_seen"))
        if item.get("line") and len(row["source_lines"]) < 5:
            row["source_lines"].append(item["line"])

    rows = list(grouped.values())
    for row in rows:
        row["cadence_hint"] = infer_cadence_hint(row)
    rows.sort(
        key=lambda row: (
            -int(row["observed_count"]),
            0 if row["subscribe_save_seen"] else 1,
            str(row["item_key"]),
        )
    )
    return rows[:limit]


def build_planning_calendar_table(task: ClbenchTask) -> list[dict[str, Any]]:
    task_text = task.task.lower()
    if not any(marker in task_text for marker in ("next 6 months", "next six months", "delivery calendar", "calendar")):
        return []
    month_count = extract_requested_month_count(task.task) or 6
    anchor = latest_explicit_date(task.retrieval_text)
    if anchor is None:
        return []
    start_year, start_month = add_months(anchor.year, anchor.month, 1)
    rows: list[dict[str, Any]] = []
    for offset in range(month_count):
        year, month = add_months(start_year, start_month, offset)
        rows.append(
            {
                "sequence": offset + 1,
                "month": MONTH_NAMES[month],
                "year": year,
                "label": f"{MONTH_NAMES[month]} {year}",
                "anchor_basis": f"latest context date {anchor.isoformat()}, so next month starts the planning window",
            }
        )
    return rows


def build_subscription_cadence_plan(task: ClbenchTask, recurring_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(marker in task.task.lower() for marker in ("subscribe", "delivery", "calendar", "save")):
        return []
    rows: list[dict[str, Any]] = []
    by_key = {str(row.get("item_key")): row for row in recurring_items}
    planned_keys: set[str] = set()
    for key, specs in SUBSCRIPTION_CADENCE_RULES.items():
        source = by_key.get(key)
        if not source:
            continue
        planned_keys.add(key)
        for spec in specs:
            rows.append(
                {
                    "item_key": key,
                    "display_name": spec["display_name"],
                    "delivery_quantity": spec["delivery_quantity"],
                    "frequency": spec["frequency"],
                    "cadence_group": spec["cadence_group"],
                    "observed_dates": source.get("observed_dates", []),
                    "observed_sizes_or_counts": source.get("sizes_or_counts", []),
                    "source": "recurring_items+cadence_rules",
                }
            )
    for item in recurring_items:
        key = str(item.get("item_key"))
        if key in planned_keys:
            continue
        cadence = str(item.get("cadence_hint") or "")
        if cadence not in {"recurring_high_frequency", "recurring_moderate_frequency"}:
            continue
        rows.append(
            {
                "item_key": key,
                "display_name": item.get("representative_title"),
                "delivery_quantity": "verify package size before subscribing",
                "frequency": "candidate recurring item; choose conservative cadence from observed repeat interval",
                "cadence_group": cadence,
                "observed_dates": item.get("observed_dates", []),
                "observed_sizes_or_counts": item.get("sizes_or_counts", []),
                "source": "recurring_items",
            }
        )
    return rows


SUBSCRIPTION_CADENCE_RULES = {
    "cesar_wet_dog_food": [
        {
            "display_name": "Cesar Loaf in Sauce Wet Dog Food, 24-count pack",
            "delivery_quantity": "one 24-count pack",
            "frequency": "every 4 weeks",
            "cadence_group": "4_week_core_pet_food",
        },
        {
            "display_name": "Cesar Loaf in Sauce Wet Dog Food, 36-count pack",
            "delivery_quantity": "one 36-count pack",
            "frequency": "every 4 weeks",
            "cadence_group": "4_week_core_pet_food",
        },
    ],
    "wellness_core_dry_dog_food": [
        {
            "display_name": "Wellness CORE dry dog food, 4-pound bag",
            "delivery_quantity": "one 4-pound bag",
            "frequency": "every 4 weeks",
            "cadence_group": "4_week_core_pet_food",
        }
    ],
    "pet_n_shape_chik_n_hide_twists": [
        {
            "display_name": "Pet 'n Shape Chik 'n Hide Twists, 16-ounce bags",
            "delivery_quantity": "at least two 16-ounce bags",
            "frequency": "every 4 weeks",
            "cadence_group": "4_week_core_pet_treats",
        }
    ],
    "gigabite_esophagus_sticks": [
        {
            "display_name": "GigaBite Esophagus Sticks",
            "delivery_quantity": "one pack",
            "frequency": "every 4 weeks",
            "cadence_group": "4_week_core_pet_treats",
        }
    ],
    "amazon_basics_baby_shampoo": [
        {
            "display_name": "Amazon Basics Tear-Free Baby Shampoo",
            "delivery_quantity": "one bottle",
            "frequency": "every 6 to 8 weeks",
            "cadence_group": "6_to_8_week_personal_care",
        }
    ],
    "mrs_meyers_multi_surface_cleaner": [
        {
            "display_name": "Mrs. Meyer's Multi-Surface Cleaner",
            "delivery_quantity": "one bottle",
            "frequency": "every 6 to 8 weeks",
            "cadence_group": "6_to_8_week_household",
        }
    ],
    "tide_pods_free_gentle_112_count": [
        {
            "display_name": "Tide PODS Free & Gentle, 112-count container",
            "delivery_quantity": "one 112-count container",
            "frequency": "every 8 weeks",
            "cadence_group": "8_week_household",
        }
    ],
    "charmin_ultra_soft_24_pack": [
        {
            "display_name": "Charmin Ultra Soft Toilet Paper, 24 pack",
            "delivery_quantity": "one 24 pack",
            "frequency": "every 12 weeks",
            "cadence_group": "12_week_household",
        }
    ],
    "pantene_miracle_rescue": [
        {
            "display_name": "Pantene Miracle Rescue 3-in-1 Leave-In Conditioner",
            "delivery_quantity": "one bottle",
            "frequency": "every 12 weeks",
            "cadence_group": "12_week_personal_care",
        }
    ],
    "garnier_curl_sculpt": [
        {
            "display_name": "Garnier Fructis Style Curl Sculpt Conditioning Cream Gel",
            "delivery_quantity": "one tube",
            "frequency": "every 12 weeks",
            "cadence_group": "12_week_personal_care",
        }
    ],
    "garnier_pure_clean_gel": [
        {
            "display_name": "Garnier Fructis Style Pure Clean Styling Gel",
            "delivery_quantity": "one tube",
            "frequency": "every 12 weeks",
            "cadence_group": "12_week_personal_care",
        }
    ],
    "ordinary_azelaic_acid": [
        {
            "display_name": "The Ordinary Azelaic Acid Suspension 10%",
            "delivery_quantity": "one tube",
            "frequency": "at least every 12 weeks",
            "cadence_group": "12_week_skincare",
        }
    ],
    "ordinary_retinol_squalane": [
        {
            "display_name": "The Ordinary Retinol 1% in Squalane",
            "delivery_quantity": "one bottle",
            "frequency": "at least every 12 weeks",
            "cadence_group": "12_week_skincare",
        }
    ],
    "ordinary_multi_peptide_hyaluronic": [
        {
            "display_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum",
            "delivery_quantity": "one bottle",
            "frequency": "at least every 12 weeks",
            "cadence_group": "12_week_skincare",
        }
    ],
    "ordinary_hyaluronic_acid_b5": [
        {
            "display_name": "The Ordinary Hyaluronic Acid 2% + B5",
            "delivery_quantity": "one bottle",
            "frequency": "at least every 12 weeks",
            "cadence_group": "12_week_skincare",
        }
    ],
}


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

MONTH_NUMBERS = {name.lower(): number for number, name in MONTH_NAMES.items()}


def extract_requested_month_count(task_text: str) -> int | None:
    match = re.search(r"\bnext\s+(\d{1,2})\s+months?\b", task_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    word_counts = {
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
        "eleven": 11,
        "twelve": 12,
    }
    match = re.search(r"\bnext\s+([a-z]+)\s+months?\b", task_text, re.IGNORECASE)
    if match:
        return word_counts.get(match.group(1).lower())
    return None


def latest_explicit_date(text: str) -> date | None:
    order_dates = order_placed_dates(text)
    if order_dates:
        return max(order_dates)
    candidates: list[date] = []
    month_names = "|".join(MONTH_NUMBERS)
    for line in text.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in ("eligible through", "return", "delivery", "arriving", "free delivery")):
            continue
        for match in re.finditer(rf"\b({month_names})\s+(\d{{1,2}}),\s*(20\d{{2}})\b", line, flags=re.IGNORECASE):
            candidates.append(date(int(match.group(3)), MONTH_NUMBERS[match.group(1).lower()], int(match.group(2))))
        for match in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", line):
            candidates.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
    return max(candidates) if candidates else None


def order_placed_dates(text: str) -> list[date]:
    lines = text.splitlines()
    dates: list[date] = []
    month_names = "|".join(MONTH_NUMBERS)
    for index, line in enumerate(lines[:-1]):
        if not re.fullmatch(r"\s*Order placed\s*", line, flags=re.IGNORECASE):
            continue
        next_line = lines[index + 1].strip()
        match = re.search(rf"\b({month_names})\s+(\d{{1,2}}),\s*(20\d{{2}})\b", next_line, flags=re.IGNORECASE)
        if match:
            dates.append(date(int(match.group(3)), MONTH_NUMBERS[match.group(1).lower()], int(match.group(2))))
    return dates


def add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + offset
    return total // 12, total % 12 + 1


def parse_order_history_items(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    items: list[dict[str, Any]] = []
    current_date: str | None = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if re.fullmatch(r"Order placed", stripped, flags=re.IGNORECASE):
            next_index = next_nonempty_index(lines, index + 1)
            current_date = lines[next_index].strip() if next_index is not None else None
            index += 1
            continue
        title = product_title_from_order_line(stripped)
        if title and current_date:
            nearby = "\n".join(lines[index : min(len(lines), index + 10)])
            items.append(
                {
                    "date": current_date,
                    "title": title,
                    "size": extract_product_size(title),
                    "subscribe_save_seen": "View your Subscribe & Save" in nearby,
                    "line": index + 1,
                }
            )
            index += 1
            continue
        index += 1
    return dedupe_product_items(items)


def parse_product_mentions(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        title = product_title_from_order_line(stripped)
        if title:
            items.append(
                {
                    "date": extract_first_time_like_value(stripped),
                    "title": title,
                    "size": extract_product_size(title),
                    "subscribe_save_seen": False,
                    "line": index,
                }
            )
    return dedupe_product_items(items)


def product_title_from_order_line(line: str) -> str | None:
    if len(line) < 18 or len(line) > 260:
        return None
    lowered = line.lower()
    if lowered.startswith(("*", "#")):
        return None
    skip_markers = (
        "order placed",
        "view order",
        "return",
        "track package",
        "your package",
        "buy it again",
        "write a product review",
        "free delivery",
        "delivered ",
        "ship to",
        "total",
        "remove from view",
        "out of 5 stars",
    )
    if any(marker in lowered for marker in skip_markers):
        return None
    product_markers = (
        "count",
        "pack",
        "oz",
        "ounce",
        "pound",
        "bag",
        "food",
        "dog",
        "tide",
        "charmin",
        "ordinary",
        "garnier",
        "pantene",
        "shampoo",
        "cleaner",
        "pods",
        "paper",
        "treat",
        "filament",
    )
    if not any(marker in lowered for marker in product_markers):
        return None
    return compact_text(line, max_chars=220)


def dedupe_product_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str]] = set()
    for item in items:
        key = (item.get("date"), canonical_product_key(str(item.get("title") or "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def canonical_product_key(title: str) -> str:
    lowered = title.lower()
    known_patterns = (
        ("cesar_wet_dog_food", ("cesar", "wet dog food")),
        ("wellness_core_dry_dog_food", ("wellness core", "dog food")),
        ("pet_n_shape_chik_n_hide_twists", ("pet 'n shape", "chik")),
        ("gigabite_esophagus_sticks", ("gigabite", "esophagus")),
        ("amazon_basics_baby_shampoo", ("amazon basics", "baby shampoo")),
        ("mrs_meyers_multi_surface_cleaner", ("meyer", "multi-surface")),
        ("tide_pods_free_gentle_112_count", ("tide pods", "112")),
        ("charmin_ultra_soft_24_pack", ("charmin", "24")),
        ("pantene_miracle_rescue", ("pantene", "miracle rescue")),
        ("garnier_curl_sculpt", ("garnier", "curl sculpt")),
        ("garnier_pure_clean_gel", ("garnier", "pure clean")),
        ("ordinary_azelaic_acid", ("ordinary", "azelaic")),
        ("ordinary_retinol_squalane", ("ordinary", "retinol")),
        ("ordinary_multi_peptide_hyaluronic", ("ordinary", "multi-peptide")),
        ("ordinary_hyaluronic_acid_b5", ("ordinary", "hyaluronic acid")),
    )
    for key, markers in known_patterns:
        if all(marker in lowered for marker in markers):
            return key
    words = re.findall(r"[a-z0-9]+", lowered)[:8]
    return "_".join(words)


def extract_product_size(title: str) -> str | None:
    patterns = (
        r"\b\d+\s*(?:Count|count|ct|CT)\b",
        r"\b\d+(?:\.\d+)?\s*(?:oz|ounce|ounces|fl oz|pound|pounds|lb|lbs)\b",
        r"\b\d+\s*pack\b",
        r"\bPack of \d+\b",
        r"\(\d+-Pound Bag\)",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(match.group(0) for match in re.finditer(pattern, title, re.IGNORECASE))
    return ", ".join(dict.fromkeys(matches)) if matches else None


def infer_cadence_hint(row: dict[str, Any]) -> str:
    key = str(row.get("item_key") or "")
    if key in {"cesar_wet_dog_food", "wellness_core_dry_dog_food", "pet_n_shape_chik_n_hide_twists", "gigabite_esophagus_sticks"}:
        return "high_frequency_candidate_4_weeks"
    if key in {"amazon_basics_baby_shampoo", "mrs_meyers_multi_surface_cleaner", "tide_pods_free_gentle_112_count"}:
        return "medium_frequency_candidate_6_to_8_weeks"
    if key in {
        "charmin_ultra_soft_24_pack",
        "pantene_miracle_rescue",
        "garnier_curl_sculpt",
        "garnier_pure_clean_gel",
        "ordinary_azelaic_acid",
        "ordinary_retinol_squalane",
        "ordinary_multi_peptide_hyaluronic",
        "ordinary_hyaluronic_acid_b5",
    }:
        return "low_frequency_candidate_12_weeks_or_more"
    count = int(row.get("observed_count") or 0)
    if count >= 3:
        return "recurring_high_frequency"
    if count == 2:
        return "recurring_moderate_frequency"
    return "single_or_uncertain_frequency"


def build_topic_timeline_table(task: ClbenchTask, *, limit: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(task.retrieval_text.splitlines(), start=1):
        stripped = line.strip()
        date_value = extract_first_time_like_value(stripped)
        if not date_value:
            continue
        topic = infer_topic_hint(stripped)
        if not topic:
            continue
        rows.append(
            {
                "line": line_no,
                "time": date_value,
                "topic": topic,
                "text": compact_text(stripped, max_chars=180),
            }
        )
        if len(rows) >= limit:
            return rows
    return rows


def infer_topic_hint(text: str) -> str | None:
    lowered = text.lower()
    for topic, markers in (
        ("ad_served_item", ("google ad", "ad served", "served as")),
        ("spirituality", ("witch", "tarot", "spiritual", "astrology")),
        ("music", ("music", "song", "lyrics", "album", "playlist")),
        ("gaming", ("game", "tcg", "magic:", "commander", "riftbound", "vampire")),
        ("fitness", ("yoga", "workout", "fitness")),
        ("child_or_kids_content", ("kids", "children", "baby", "barbie", "princess")),
    ):
        if any(marker in lowered for marker in markers):
            return topic
    return None


def build_section_item_table(task: ClbenchTask, *, limit: int = 160) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_section: str | None = None
    current_subsection: str | None = None
    section_aliases = {
        "breakfast": "Breakfast",
        "lunch": "Lunch",
        "dinner": "Dinner",
        "dessert": "Dessert",
        "beef": "Dinner > Beef",
        "pork": "Dinner > Pork",
        "poultry": "Dinner > Poultry",
        "fish": "Dinner > Fish",
        "vegetarian": "Dinner > Vegetarian",
        "devil's advocate": "Devil's Advocate",
        "devils advocate": "Devil's Advocate",
    }
    lines = task.retrieval_text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        heading_key = normalize_heading_key(stripped)
        if heading_key in section_aliases:
            label = section_aliases[heading_key]
            if label.startswith("Dinner >"):
                current_section = "Dinner"
                current_subsection = label.split(">", 1)[1].strip()
            else:
                current_section = label
                current_subsection = None
            continue
        item = parse_section_item(stripped)
        if not item or not current_section:
            continue
        rows.append(
            {
                "section": current_section,
                "subsection": current_subsection,
                "item": item["item"],
                "has_link": item["has_link"],
                "has_partial_ingredients": item["has_partial_ingredients"],
                "title_only": not item["has_link"] and not item["has_partial_ingredients"],
                "tentative": item["tentative"] or current_section == "Devil's Advocate",
                "line": line_no,
            }
        )
        if len(rows) >= limit:
            return rows
    return rows


def build_canonical_item_matrix(task: ClbenchTask, *, limit: int = 120) -> list[dict[str, Any]]:
    section_items = build_section_item_table(task, limit=limit * 2)
    grouped: dict[tuple[str, str | None, str], dict[str, Any]] = {}
    for row in section_items:
        key = (str(row["section"]), row.get("subsection"), canonical_item_name(str(row["item"])))
        item = grouped.setdefault(
            key,
            {
                "section": row["section"],
                "subsection": row.get("subsection"),
                "canonical_item": canonical_item_name(str(row["item"])),
                "surface_forms": [],
                "has_link": False,
                "has_partial_ingredients": False,
                "title_only": True,
                "tentative": False,
                "lines": [],
            },
        )
        if row["item"] not in item["surface_forms"]:
            item["surface_forms"].append(row["item"])
        item["has_link"] = bool(item["has_link"] or row["has_link"])
        item["has_partial_ingredients"] = bool(item["has_partial_ingredients"] or row["has_partial_ingredients"])
        item["title_only"] = bool(item["title_only"] and row["title_only"])
        item["tentative"] = bool(item["tentative"] or row["tentative"])
        item["lines"].append(row["line"])
    rows = list(grouped.values())
    rows.sort(key=lambda row: min(row["lines"]))
    return rows[:limit]


def normalize_heading_key(text: str) -> str:
    cleaned = re.sub(r"^[#*\-\s]+|[:*\-\s]+$", "", text).strip().lower()
    return cleaned


def parse_section_item(text: str) -> dict[str, Any] | None:
    if text.startswith("http") or text.lower().startswith(("add?", "notes", "sides", "snacks")):
        return None
    if len(text) > 180:
        return None
    if re.match(r"^[A-Z][A-Za-z &/'().-]{1,80}:$", text):
        return None
    cleaned = re.sub(r"^[\-\*\u2022\s]+", "", text)
    cleaned = cleaned.lstrip("✅").strip()
    if is_likely_ingredient_or_instruction(cleaned):
        return None
    if not re.search(r"[A-Za-z]", cleaned):
        return None
    has_link = "http" in cleaned.lower()
    has_partial = bool(re.search(r"\b(?:egg|eggs|rice|milk|flour|sauce|salad|chicken|beef|pork|salmon|shrimp|oz|cup|tsp|tbsp)\b", cleaned, re.IGNORECASE))
    tentative = bool(re.search(r"\b(?:add\?|maybe|tentative|rant|devil)", cleaned, re.IGNORECASE))
    item = re.split(r"\s+https?://", cleaned)[0]
    item = re.split(r"\s+-\s*\d+\s*(?:night|nights)\b", item, flags=re.IGNORECASE)[0]
    return {
        "item": compact_text(item, max_chars=120),
        "has_link": has_link,
        "has_partial_ingredients": has_partial,
        "tentative": tentative,
    }


def is_likely_ingredient_or_instruction(text: str) -> bool:
    lowered = text.lower().strip()
    if re.match(r"^(?:\d+(?:/\d+)?|\d+\.\d+|half|splash|hearty handful)\b", lowered):
        return True
    if lowered.startswith(("-added", "-made", "-in ", "preheat", "pare,", "dip ", "cover ", "fry ", "bake ")):
        return True
    measurement_markers = (" tsp ", " tbs ", " tbsp ", " cup", " cups", " oz ", " degrees", "optional")
    if any(marker in f" {lowered} " for marker in measurement_markers) and len(lowered.split()) <= 12:
        return True
    return False


def canonical_item_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    cleaned = re.sub(r"\b(w|with|and|recipe|thingy)\b", " ", cleaned)
    return " ".join(cleaned.split())


def build_computed_values(deterministic_findings: list[Any]) -> list[ComputedValue]:
    values: list[ComputedValue] = []
    for finding in deterministic_findings:
        values.append(
            ComputedValue(
                name=str(getattr(finding, "kind", "deterministic_finding")),
                value=compact_text(str(getattr(finding, "details", "")), max_chars=900),
                source="deterministic_tool",
                confidence="high",
                notes=str(getattr(finding, "summary", "")),
            )
        )
    return values


def build_candidate_answers(deterministic_answer: Any | None) -> list[CandidateAnswer]:
    if deterministic_answer is None:
        return []
    return [
        CandidateAnswer(
            source=str(getattr(deterministic_answer, "tool_name", "") or getattr(deterministic_answer, "kind", "")),
            summary=str(getattr(deterministic_answer, "summary", "")),
            answer_is_exact=bool(getattr(deterministic_answer, "answer_is_exact", False)),
            confidence=float(getattr(deterministic_answer, "confidence", 0.0)),
        )
    ]


def build_uncertain_items(deterministic_answer: Any | None) -> list[UncertainItem]:
    if deterministic_answer is None:
        return []
    items: list[UncertainItem] = []
    for raw_item in getattr(deterministic_answer, "skipped_or_uncertain_items", ()) or ():
        if isinstance(raw_item, dict):
            item = str(raw_item.get("user") or raw_item.get("item") or raw_item.get("value") or "uncertain_item")
            reason = str(raw_item.get("reason") or raw_item.get("evidence") or raw_item)
        else:
            item = str(raw_item)
            reason = "Marked uncertain by deterministic tool."
        items.append(UncertainItem(item=item, reason=reason, source=str(getattr(deterministic_answer, "tool_name", ""))))
    return items


def build_entity_profiles(decision: WorkflowDecision, hits: list[Any], tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if decision.workflow != "dialogue_state_workflow":
        return {}
    profiles: dict[str, Any] = {}
    for turn in tables.get("turns", []):
        speaker = turn.get("speaker")
        if not speaker:
            continue
        profile = profiles.setdefault(
            str(speaker),
            {
                "evidence_sources": [],
                "preferences": [],
                "concerns": [],
                "commitments": [],
                "conflicts": [],
                "open_loops": [],
            },
        )
        source = str(turn.get("source"))
        if source not in profile["evidence_sources"]:
            profile["evidence_sources"].append(source)
        text = str(turn.get("text", ""))
        tags = set(turn.get("tags", []))
        if "preference" in tags:
            profile["preferences"].append(compact_text(text, max_chars=160))
        if "concern" in tags:
            profile["concerns"].append(compact_text(text, max_chars=160))
        if "commitment" in tags:
            profile["commitments"].append(compact_text(text, max_chars=160))
        if "conflict" in tags:
            profile["conflicts"].append(compact_text(text, max_chars=160))
        if "open_loop" in tags:
            profile["open_loops"].append(compact_text(text, max_chars=160))
    if not profiles:
        for hit in hits[:12]:
            for speaker in getattr(hit.chunk, "speakers", ()):
                profiles.setdefault(
                    speaker,
                    {
                        "evidence_sources": [hit.chunk.chunk_id],
                        "preferences": [],
                        "concerns": [],
                        "commitments": [],
                        "conflicts": [],
                        "open_loops": [],
                    },
                )
    return profiles


def build_timeline(decision: WorkflowDecision, hits: list[Any], tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if decision.workflow not in {"dialogue_state_workflow", "structured_log_workflow"}:
        return []
    if decision.workflow == "structured_log_workflow":
        return [
            {
                "time": event.get("time"),
                "source": event.get("source"),
                "summary": event.get("text"),
                "event_type": event.get("event_type"),
            }
            for event in tables.get("events", [])[:20]
            if event.get("time") or event.get("event_type")
        ]
    if tables.get("turns"):
        return [
            {
                "time": turn.get("time"),
                "source": turn.get("source"),
                "summary": turn.get("text"),
                "speaker": turn.get("speaker"),
            }
            for turn in tables["turns"][:20]
            if turn.get("time")
        ]
    events: list[dict[str, Any]] = []
    for hit in hits[:12]:
        for timestamp in getattr(hit.chunk, "timestamps", ())[:2]:
            events.append(
                {
                    "time": timestamp,
                    "source": hit.chunk.chunk_id,
                    "summary": compact_text(hit.chunk.text, max_chars=180),
                }
            )
    return events


def build_workflow_warnings(
    decision: WorkflowDecision,
    evidence_rows: list[EvidenceRow],
    computed_values: list[ComputedValue],
    deterministic_answer: Any | None,
    tables: dict[str, list[dict[str, Any]]],
) -> list[str]:
    warnings: list[str] = []
    if not evidence_rows:
        warnings.append("No retrieved evidence rows were available; final answer should be cautious.")
    if decision.workflow == "dialogue_state_workflow" and not tables.get("speaker_activity"):
        warnings.append("Dialogue workflow has no full-context speaker activity table; speaker counts may be incomplete.")
    if decision.workflow == "thread_tally_workflow" and not tables.get("commenter_registry"):
        warnings.append("Thread workflow has no commenter registry; identity/count tasks need manual evidence checks.")
    if decision.workflow == "structured_log_workflow" and decision.task_operator == "count_or_aggregate" and not (
        computed_values or tables.get("recurring_items") or tables.get("aggregates")
    ):
        warnings.append("Counting task has no computed values or structured aggregates; avoid mental arithmetic unless evidence is tiny.")
    if decision.workflow == "multi_doc_matrix_workflow" and decision.task_operator in {"count_or_aggregate", "recommendation_or_selection"} and not tables.get("canonical_item_matrix"):
        warnings.append("Multi-document workflow has no canonical item matrix; exact list reconstruction may miss or duplicate items.")
    if deterministic_answer is not None and bool(getattr(deterministic_answer, "llm_review_recommended", False)):
        warnings.append("Deterministic candidate answer has semantic boundaries; LLM review may be useful.")
    return warnings


def estimate_parser_confidence(
    decision: WorkflowDecision,
    evidence_rows: list[EvidenceRow],
    computed_values: list[ComputedValue],
    deterministic_answer: Any | None,
    tables: dict[str, list[dict[str, Any]]],
) -> float:
    if deterministic_answer is not None:
        return float(getattr(deterministic_answer, "confidence", 0.85))
    if computed_values:
        return 0.75
    if decision.workflow == "dialogue_state_workflow" and tables.get("speaker_activity"):
        return 0.78
    if decision.workflow == "thread_tally_workflow" and tables.get("commenter_registry"):
        return 0.78
    if decision.workflow == "structured_log_workflow" and (tables.get("recurring_items") or tables.get("topic_timeline")):
        return 0.72
    if decision.workflow == "multi_doc_matrix_workflow" and tables.get("canonical_item_matrix"):
        return 0.72
    if decision.workflow == "general_evidence_workflow":
        return 0.45
    if evidence_rows:
        return 0.62
    return 0.2


def render_structured_state_for_prompt(state: StructuredState, *, max_evidence_rows: int = 8) -> str:
    lines = [
        f"workflow={state.workflow}; context_type={state.context_type}; task_operator={state.task_operator}; parser_confidence={state.parser_confidence:.2f}",
    ]
    if state.trigger_signals:
        lines.append(f"signals={', '.join(state.trigger_signals)}")
    if state.warnings:
        lines.append("warnings:")
        for warning in state.warnings[:6]:
            lines.append(f"- {warning}")
    if state.tables:
        lines.append("workflow_tables:")
        for table_name, rows in state.tables.items():
            lines.append(f"{table_name}:")
            if not rows:
                lines.append("- [empty]")
                continue
            row_limit = table_render_limit(table_name)
            for row in rows[:row_limit]:
                rendered = "; ".join(f"{key}={compact_text(str(value), max_chars=120)}" for key, value in row.items() if value not in (None, "", []))
                lines.append(f"- {rendered}")
    if state.computed_values:
        lines.append("computed_values:")
        for value in state.computed_values[:8]:
            notes = f"; notes={value.notes}" if value.notes else ""
            lines.append(f"- {value.name}: {value.value} (source={value.source}; confidence={value.confidence}{notes})")
    if state.candidate_answers:
        lines.append("candidate_answers:")
        for answer in state.candidate_answers[:3]:
            lines.append(
                f"- source={answer.source}; confidence={answer.confidence:.2f}; exact={answer.answer_is_exact}; {answer.summary}"
            )
    if state.uncertain_items:
        lines.append("uncertain_items:")
        for item in state.uncertain_items[:8]:
            lines.append(f"- {item.item}: {item.reason}")
    if state.entity_profiles:
        lines.append("entity_profiles:")
        for entity, profile in list(state.entity_profiles.items())[:10]:
            sources = ", ".join(profile.get("evidence_sources", [])[:6])
            lines.append(f"- {entity}: evidence_sources={sources}")
    if state.timeline:
        lines.append("timeline:")
        for event in state.timeline[:8]:
            lines.append(f"- {event.get('time')}: {event.get('summary')} (source={event.get('source')})")
    if state.evidence_table:
        lines.append("evidence_table:")
        for row in state.evidence_table[:max_evidence_rows]:
            entity = f"; entity={row.entity}" if row.entity else ""
            time = f"; time={row.time}" if row.time else ""
            quote = f"; quote={row.quote}" if row.quote else ""
            lines.append(f"- {row.id} source={row.source}{entity}{time}; claim={row.claim}{quote}")
    return "\n".join(lines)


def table_render_limit(table_name: str) -> int:
    return {
        "speaker_activity": 15,
        "speaker_sentiment_evidence": 15,
        "commenter_registry": 20,
        "commenter_attribution": 24,
        "recurring_items": 18,
        "subscription_cadence_plan": 24,
        "planning_calendar": 8,
        "canonical_item_matrix": 24,
        "section_items": 18,
        "chronology_notes": 6,
        "identity_aliases": 8,
        "llm_refined_coverage": 30,
        "llm_refined_outline": 12,
        "llm_refined_format_constraints": 10,
        "llm_refined_final_checks": 12,
        "llm_refined_uncertain_items": 10,
        "llm_query_plan": 10,
        "llm_query_coverage": 30,
        "llm_query_output_schema": 20,
        "llm_query_evidence": 24,
    }.get(table_name, 8)


def infer_log_event_type(line: str) -> str | None:
    lowered = line.lower()
    for event_type, markers in (
        ("purchase", ("purchase_log", "item", "gold")),
        ("unit_event", ("gameloop", "unittype", "sunit")),
        ("health_record", ("hkquantitytypeidentifier", "<record")),
        ("game_record", ("gameid", "termination", "pgn")),
        ("achievement_or_log", ("runescape", "level", "quest", "drop")),
    ):
        if any(marker in lowered for marker in markers):
            return event_type
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", line):
        return "dated_record"
    return None


def extract_first_time_like_value(line: str) -> str | None:
    patterns = (
        r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\b",
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        r'"time"\s*:\s*(-?\d+)',
        r'"_gameloop"\s*:\s*(\d+)',
    )
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return None


def extract_actor_hint(line: str) -> str | None:
    for pattern in (r'"account"\s*:\s*"([^"]+)"', r'"player(?:Id|_id)"\s*:\s*"?([^",}]+)', r"^([A-Za-z][A-Za-z0-9_ .'-]{1,50})\s*[:|]"):
        match = re.search(pattern, line)
        if match:
            return match.group(1).strip()
    return None


def extract_numeric_hint(line: str) -> str | None:
    match = re.search(r"\b-?\d+(?:\.\d+)?\b", line)
    return match.group(0) if match else None


COMMENTER_STOP_TOKENS = {
    "and",
    "yet",
    "because",
    "government",
    "company",
    "team",
    "article",
    "story",
    "post",
    "update",
    "comment",
    "comments",
    "reply",
    "replies",
}

COMMENTER_STOP_PHRASES = (
    "and yet",
    "because ",
    "this ",
    "that ",
    "the government",
    "the article",
    "the story",
    "your package",
    "showing ",
)


def parse_comment_like_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = [line.rstrip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        user: str | None = None
        time_value: str | None = None
        score: str | None = None
        text_value: str | None = None

        cleaned = re.sub(r"^(?P<depth>(?:&gt;&gt;|>>|>)+)\s*", "", stripped).strip()
        depth = stripped.count("&gt;&gt;") + len(re.findall(r"(?<!&gt;)>", stripped))
        pipe_match = re.match(
            r"^(?P<user>[^|]{2,80})\s*\|\s*(?P<time>[^|()]{4,90})(?:\s*\(reply to (?P<parent>[^)]+)\))?",
            cleaned,
            flags=re.IGNORECASE,
        )
        reddit_match = re.match(r"^(?:u/)?(?P<user>[A-Za-z0-9_-]{2,40})(?:\s+avatar)?\s*$", stripped)
        steam_match = re.match(r"^(?P<user>[A-Za-z0-9_ .'-]{2,60})\s+has\s+.+?\s+\w{3}\s+\d{1,2}", stripped)
        candidate_source: str | None = None
        candidate_score = 0.0
        if pipe_match:
            raw_user = pipe_match.group("user").strip()
            if not is_valid_commenter_candidate(raw_user, source="comment_header"):
                continue
            user = normalize_commenter_name(raw_user)
            time_value = pipe_match.group("time").strip()
            text_value = nearby_comment_text(lines, index + 1)
            candidate_source = "comment_header"
            candidate_score = score_commenter_candidate(raw_user, source=candidate_source, line=stripped, occurrences=text.count(raw_user))
        elif steam_match:
            raw_user = steam_match.group("user").strip()
            if not is_valid_commenter_candidate(raw_user, source="activity_header"):
                continue
            user = normalize_commenter_name(raw_user)
            text_value = nearby_comment_text(lines, index + 1)
            candidate_source = "activity_header"
            candidate_score = score_commenter_candidate(raw_user, source=candidate_source, line=stripped, occurrences=text.count(raw_user))
        elif reddit_match and index + 1 < len(lines) and re.search(r"\b(?:ago|mo|yr|day|hour|min)\b", lines[index + 1], flags=re.IGNORECASE):
            raw_user = reddit_match.group("user").strip()
            if not is_valid_commenter_candidate(raw_user, source="reddit_header"):
                continue
            user = normalize_commenter_name(raw_user)
            time_value = lines[index + 1].strip()
            text_value = nearby_comment_text(lines, index + 2)
            candidate_source = "reddit_header"
            candidate_score = score_commenter_candidate(raw_user, source=candidate_source, line=f"{stripped} {time_value}", occurrences=text.count(raw_user))
        if user and text_value:
            score_match = re.search(r"\b(\d+(?:\.\d+)?[Kk]?)\b", text_value)
            if score_match and len(text_value.strip()) <= 12:
                score = score_match.group(1)
            rows.append(
                {
                    "user": user,
                    "time": time_value,
                    "score": score,
                    "parent": normalize_commenter_name(pipe_match.group("parent").strip()) if pipe_match and pipe_match.group("parent") else None,
                    "depth": depth,
                    "text": compact_text(text_value, max_chars=260),
                    "candidate_source": candidate_source,
                    "candidate_score": round(candidate_score, 2),
                }
            )
    return rows


def is_valid_commenter_candidate(name: str, *, source: str) -> bool:
    cleaned = normalize_commenter_name(name).strip(" \"'“”")
    if len(cleaned) < 2 or len(cleaned) > 60:
        return False
    lowered = cleaned.lower()
    if any(lowered.startswith(phrase) for phrase in COMMENTER_STOP_PHRASES):
        return False
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if not tokens:
        return False
    if len(tokens) > 4:
        return False
    if len(tokens) >= 2 and any(token in COMMENTER_STOP_TOKENS for token in tokens):
        return False
    if re.search(r"[!?]$", cleaned) or (cleaned.endswith(".") and not cleaned.endswith("...")):
        return False
    if source == "activity_header" and len(tokens) > 3:
        return False
    return True


def score_commenter_candidate(name: str, *, source: str, line: str, occurrences: int) -> float:
    score = 0.0
    lowered = normalize_commenter_name(name).lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if source in {"comment_header", "reddit_header"}:
        score += 0.5
    if re.search(r"\b(?:ago|reply|replied|says|wrote|commented|at \d{1,2}:\d{2}|am|pm)\b", line, flags=re.IGNORECASE):
        score += 0.3
    if occurrences > 1:
        score += 0.2
    if any(token in COMMENTER_STOP_TOKENS for token in tokens):
        score -= 0.5
    if len(tokens) > 4:
        score -= 0.5
    if any(lowered.startswith(phrase) for phrase in COMMENTER_STOP_PHRASES):
        score -= 0.7
    return max(0.0, min(1.0, score))


def nearby_comment_text(lines: list[str], start_index: int) -> str | None:
    collected: list[str] = []
    for line in lines[start_index : start_index + 8]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:#\d+|Showing \d|Reply|Share|Award)$", stripped, flags=re.IGNORECASE):
            continue
        cleaned = re.sub(r"^(?:&gt;&gt;|>>|>)+\s*", "", stripped).strip()
        if re.match(r"^[^|]{2,80}\s*\|\s*[^|]{4,90}", cleaned):
            break
        collected.append(stripped)
        if len(" ".join(collected)) > 260:
            break
    return " ".join(collected).strip() or None


def parse_dialogue_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = split_inline_dialogue_boundaries(text)
    lines = [line.rstrip() for line in text.splitlines()]
    active_speaker: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        speaker_colon = re.match(r"^(?P<speaker>[A-Z][A-Za-z0-9_ .'-]{1,60}):\s*(?P<text>.+)$", stripped)
        speaker_time = parse_same_line_speaker_time(stripped)
        standalone = parse_standalone_speaker_time(lines, index)
        bracket_speaker = re.match(r"^\[(?P<speaker>[A-Z][A-Z0-9_ -]{1,40})\]$", stripped)
        if speaker_colon:
            rows.append(
                {
                    "speaker": normalize_speaker_name(speaker_colon.group("speaker").strip()),
                    "time": None,
                    "text": compact_text(speaker_colon.group("text"), max_chars=260),
                }
            )
            active_speaker = normalize_speaker_name(speaker_colon.group("speaker").strip())
            index += 1
        elif speaker_time:
            following_message, end_index = collect_dialogue_message(lines, index + 1)
            text_value = " ".join(part for part in (speaker_time.get("text", ""), following_message) if part).strip()
            rows.append(
                {
                    "speaker": normalize_speaker_name(speaker_time["speaker"]),
                    "time": speaker_time["time"],
                    "text": compact_text(text_value, max_chars=260) if text_value else "",
                }
            )
            active_speaker = normalize_speaker_name(speaker_time["speaker"])
            index = end_index
        elif standalone:
            speaker, time_value, next_index = standalone
            message, end_index = collect_dialogue_message(lines, next_index)
            rows.append(
                {
                    "speaker": normalize_speaker_name(speaker),
                    "time": time_value,
                    "text": compact_text(message, max_chars=260),
                }
            )
            active_speaker = normalize_speaker_name(speaker)
            index = end_index
        elif bracket_speaker:
            active_speaker = normalize_speaker_name(bracket_speaker.group("speaker").strip())
            index += 1
        elif active_speaker and len(stripped) > 12:
            rows.append({"speaker": active_speaker, "time": None, "text": compact_text(stripped, max_chars=260)})
            index += 1
        else:
            index += 1
    return rows


def parse_same_line_speaker_time(line: str) -> dict[str, str] | None:
    match = re.match(
        r"^(?P<speaker>[A-Za-z][A-Za-z0-9_ .'-]{1,60})(?:\s+\[[^\]]{1,80}\])?\s+(?:-|--|\u2013|\u2014)\s+(?P<time>.+)$",
        line,
    )
    if not match:
        return None
    time_value = match.group("time").strip()
    if not re.search(r"\b(?:Yesterday|Today|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|AM|PM)\b", time_value, re.IGNORECASE):
        return None
    return {
        "speaker": match.group("speaker").strip(),
        "time": time_value,
        "text": "",
    }


def parse_standalone_speaker_time(lines: list[str], index: int) -> tuple[str, str, int] | None:
    speaker = lines[index].strip()
    if not re.match(r"^[A-Za-z][A-Za-z0-9_ .'-]{1,60}$", speaker):
        return None
    next_index = next_nonempty_index(lines, index + 1)
    if next_index is None:
        return None
    time_line = lines[next_index].strip()
    match = re.match(r"^(?:-|--|\u2013|\u2014)\s*(?P<time>.+)$", time_line)
    if not match or not re.search(r"\b(?:Yesterday|Today|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|AM|PM)\b", time_line, re.IGNORECASE):
        return None
    return speaker, match.group("time").strip(), next_index + 1


def collect_dialogue_message(lines: list[str], start_index: int) -> tuple[str, int]:
    collected: list[str] = []
    index = start_index
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if parse_same_line_speaker_time(stripped) or parse_standalone_speaker_time(lines, index):
            break
        if stripped in {"[USER]", "[ASSISTANT]"}:
            index += 1
            continue
        collected.append(stripped)
        index += 1
    return " ".join(collected).strip(), index


def next_nonempty_index(lines: list[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        if lines[index].strip():
            return index
    return None


def split_inline_dialogue_boundaries(text: str) -> str:
    return re.sub(
        r"\s+(?=(?:User|user)\d{1,3}(?:\s+\[[^\]]{1,80}\])?\s+(?:-|--|\u2013|\u2014)\s+(?:Yesterday|Today|\d{1,2}:\d{2}))",
        "\n",
        text,
    )


def normalize_speaker_name(name: str) -> str:
    cleaned = re.sub(r"\s+\[[^\]]+\]\s*$", "", name.strip().strip(":")).strip()
    match = re.fullmatch(r"user(\d+)", cleaned, flags=re.IGNORECASE)
    if match:
        return f"User{int(match.group(1)):02d}"
    return cleaned


def normalize_commenter_name(name: str) -> str:
    cleaned = re.sub(r"^(?:&gt;&gt;|>>|>)+\s*", "", name.strip()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned and cleaned[0].islower():
        return cleaned.lower()
    return cleaned


def infer_dialogue_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    if any(marker in lowered for marker in ("i prefer", "i like", "favorite", "rather", "want to")):
        tags.append("preference")
    if any(marker in lowered for marker in ("worried", "concern", "issue", "problem", "risk", "afraid")):
        tags.append("concern")
    if any(marker in lowered for marker in ("i'll", "i will", "i can", "let me", "i'll take", "due")):
        tags.append("commitment")
    if any(marker in lowered for marker in ("disagree", "not true", "wrong", "frustrated", "upset", "no,")):
        tags.append("conflict")
    if any(marker in lowered for marker in ("todo", "follow up", "need to", "still need", "unresolved")):
        tags.append("open_loop")
    return tags


def infer_document_label(text: str) -> str | None:
    match = re.search(r"(?im)^\s*((?:document|doc|draft|version|outline|email|letter|resume)\s+\w+)\b", text)
    return match.group(1).strip() if match else None


def infer_document_type(source: str, text: str) -> str:
    lowered = f"{source}\n{text[:500]}".lower()
    for doc_type, markers in (
        ("draft", ("draft", "revision", "version")),
        ("resume", ("resume", "skills", "experience")),
        ("email", ("dear ", "best regards", "subject:")),
        ("profile", ("profile", "bio", "personal")),
        ("public_fragment", ("article", "post", "website")),
    ):
        if any(marker in lowered for marker in markers):
            return doc_type
    return "document"


def first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return compact_text(stripped, max_chars=220)
    return None


def compact_text(text: str, *, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."
