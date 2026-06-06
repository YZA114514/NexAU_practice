from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .data import ClbenchTask
from .workflows import StructuredState, compact_text, render_structured_state_for_prompt


@dataclass(frozen=True)
class StateRefinement:
    enabled: bool
    used: bool
    status: str
    raw_response: str | None = None
    parsed: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "used": self.used,
            "status": self.status,
            "parsed": self.parsed,
            "errors": self.errors,
            "raw_response_preview": self.raw_response[:2000] if self.raw_response else None,
        }


REFINABLE_WORKFLOWS = {
    "dialogue_state_workflow",
    "thread_tally_workflow",
    "structured_log_workflow",
    "multi_doc_matrix_workflow",
}


def should_refine_state(state: StructuredState, *, deterministic_answer_available: bool) -> bool:
    if deterministic_answer_available:
        return False
    if state.workflow not in REFINABLE_WORKFLOWS:
        return False
    if state.candidate_answers:
        return False
    useful_tables = {
        "speaker_activity",
        "commenter_registry",
        "recurring_items",
        "canonical_item_matrix",
        "section_items",
        "topic_timeline",
    }
    return any(state.tables.get(table_name) for table_name in useful_tables)


def build_state_refiner_prompt(*, task: ClbenchTask, state: StructuredState) -> str:
    state_text = render_structured_state_for_prompt(state, max_evidence_rows=4)
    return f"""You are a CL-bench Life harness state refiner.

Your job is NOT to answer the user. Your job is to convert deterministic workflow tables into a compact answer contract that the final answer writer must follow.

Rules:
- Use only the structured state below.
- Do not invent facts absent from the structured state.
- Prefer exact names, counts, dates, section headings, months, and required output fields.
- If the state is insufficient, say exactly what is missing in `uncertain_items`.
- Return strict JSON only, with this schema:
{{
  "coverage_items": [
    {{"kind": "person|item|month|section|field|event|other", "name": "...", "value": "...", "required": true, "source_table": "..."}}
  ],
  "answer_outline": ["..."],
  "format_constraints": ["..."],
  "uncertain_items": [
    {{"item": "...", "reason": "...", "source_table": "..."}}
  ],
  "final_checks": ["..."]
}}

FINAL USER TASK:
{task.task}

WORKFLOW STRUCTURED STATE:
{state_text}
"""


def parse_state_refinement_response(response: str) -> StateRefinement:
    parsed = parse_json_object(response)
    if parsed is None:
        return StateRefinement(
            enabled=True,
            used=False,
            status="parse_error",
            raw_response=response,
            errors=["refiner_response_not_valid_json"],
        )
    errors = validate_refinement(parsed)
    if errors:
        return StateRefinement(enabled=True, used=False, status="validation_error", raw_response=response, parsed=parsed, errors=errors)
    return StateRefinement(enabled=True, used=True, status="ok", raw_response=response, parsed=parsed)


def parse_json_object(response: str) -> dict[str, Any] | None:
    stripped = response.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    elif stripped.startswith("```"):
        stripped = stripped[3:].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def validate_refinement(parsed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("coverage_items", "answer_outline", "format_constraints", "uncertain_items", "final_checks"):
        if key not in parsed:
            errors.append(f"missing_key:{key}")
        elif not isinstance(parsed[key], list):
            errors.append(f"non_list_key:{key}")
    return errors


def merge_refinement_into_state(state: StructuredState, refinement: StateRefinement) -> StructuredState:
    if not refinement.used or not refinement.parsed:
        return state
    tables = {key: [dict(row) for row in rows] for key, rows in state.tables.items()}
    parsed = refinement.parsed
    tables["llm_refined_coverage"] = normalize_coverage_items(parsed.get("coverage_items", []))
    tables["llm_refined_outline"] = normalize_string_rows(parsed.get("answer_outline", []), key="step")
    tables["llm_refined_format_constraints"] = normalize_string_rows(parsed.get("format_constraints", []), key="constraint")
    tables["llm_refined_final_checks"] = normalize_string_rows(parsed.get("final_checks", []), key="check")
    tables["llm_refined_uncertain_items"] = normalize_uncertain_items(parsed.get("uncertain_items", []))
    warnings = list(state.warnings)
    if tables["llm_refined_uncertain_items"]:
        warnings.append("LLM state refiner reported uncertain items; final answer should preserve these caveats.")
    return StructuredState(
        workflow=state.workflow,
        context_type=state.context_type,
        task_operator=state.task_operator,
        tables=tables,
        evidence_table=state.evidence_table,
        computed_values=state.computed_values,
        entity_profiles=state.entity_profiles,
        timeline=state.timeline,
        candidate_answers=state.candidate_answers,
        uncertain_items=state.uncertain_items,
        parser_confidence=state.parser_confidence,
        trigger_signals=[*state.trigger_signals, "llm_state_refiner"],
        warnings=warnings,
        fallback_recommendation=state.fallback_recommendation,
    )


def normalize_coverage_items(items: list[Any], *, limit: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items[:limit], start=1):
        if isinstance(item, dict):
            rows.append(
                {
                    "coverage_id": f"cov{index:03d}",
                    "kind": compact_text(str(item.get("kind", "other")), max_chars=40),
                    "name": compact_text(str(item.get("name", "")), max_chars=120),
                    "value": compact_text(str(item.get("value", "")), max_chars=160),
                    "required": bool(item.get("required", True)),
                    "source_table": compact_text(str(item.get("source_table", "")), max_chars=80),
                }
            )
        else:
            rows.append(
                {
                    "coverage_id": f"cov{index:03d}",
                    "kind": "other",
                    "name": compact_text(str(item), max_chars=120),
                    "value": "",
                    "required": True,
                    "source_table": "",
                }
            )
    return [row for row in rows if row["name"] or row["value"]]


def normalize_string_rows(items: list[Any], *, key: str, limit: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items[:limit], start=1):
        rows.append({"rank": index, key: compact_text(str(item), max_chars=220)})
    return [row for row in rows if row[key]]


def normalize_uncertain_items(items: list[Any], *, limit: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items[:limit], start=1):
        if isinstance(item, dict):
            rows.append(
                {
                    "uncertain_id": f"unc{index:03d}",
                    "item": compact_text(str(item.get("item", "")), max_chars=120),
                    "reason": compact_text(str(item.get("reason", "")), max_chars=220),
                    "source_table": compact_text(str(item.get("source_table", "")), max_chars=80),
                }
            )
        else:
            rows.append({"uncertain_id": f"unc{index:03d}", "item": compact_text(str(item), max_chars=120), "reason": "", "source_table": ""})
    return [row for row in rows if row["item"] or row["reason"]]
