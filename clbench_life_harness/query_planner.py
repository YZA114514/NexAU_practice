from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .chunking import Chunk
from .data import ClbenchTask
from .retrieval import RetrievalHit, retrieve
from .state_refiner import parse_json_object
from .workflows import StructuredState, compact_text, render_structured_state_for_prompt


@dataclass(frozen=True)
class QueryRequest:
    query_id: str
    purpose: str
    query: str
    expected_fields: tuple[str, ...] = ()
    quote_required: bool = False
    priority: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "purpose": self.purpose,
            "query": self.query,
            "expected_fields": list(self.expected_fields),
            "quote_required": self.quote_required,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class QueryPlan:
    enabled: bool
    used: bool
    status: str
    requests: tuple[QueryRequest, ...] = ()
    coverage_items: tuple[dict[str, Any], ...] = ()
    output_schema: tuple[str, ...] = ()
    raw_response: str | None = None
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "used": self.used,
            "status": self.status,
            "requests": [request.to_dict() for request in self.requests],
            "coverage_items": list(self.coverage_items),
            "output_schema": list(self.output_schema),
            "raw_response_preview": self.raw_response[:2000] if self.raw_response else None,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class QueryExecution:
    plan: QueryPlan
    hits: tuple[RetrievalHit, ...] = ()
    evidence_rows: tuple[dict[str, Any], ...] = ()
    query_hit_map: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "hit_ids": [hit.chunk.chunk_id for hit in self.hits],
            "query_hit_map": self.query_hit_map,
            "evidence_rows": list(self.evidence_rows),
        }


def should_plan_queries(task: ClbenchTask, state: StructuredState, *, deterministic_answer_available: bool) -> bool:
    if deterministic_answer_available:
        return False
    if state.candidate_answers:
        return False
    task_text = task.task.lower()
    if state.workflow == "dialogue_state_workflow":
        if state.tables.get("speaker_activity") and is_exact_speaker_activity_task(task_text):
            return False
        return True
    if state.workflow == "thread_tally_workflow":
        return any(
            marker in task_text
            for marker in (
                "identity",
                "same person",
                "different names",
                "misattribution",
                "misattributed",
                "wrong",
                "disputed",
                "quote",
                "stance",
                "sentiment",
                "opinion",
            )
        )
    if state.workflow == "multi_doc_matrix_workflow":
        return not is_exact_list_task(task_text)
    if state.workflow == "structured_log_workflow":
        return False
    if state.workflow == "general_evidence_workflow":
        return True
    return False


def is_exact_list_task(task_text: str) -> bool:
    return any(
        marker in task_text
        for marker in (
            "exact list",
            "all items",
            "which items",
            "ingredients",
            "titles",
            "names",
            "do not paraphrase",
            "from section",
            "table of contents",
        )
    )


def is_exact_speaker_activity_task(task_text: str) -> bool:
    return any(marker in task_text for marker in ("most active", "top ")) and any(
        marker in task_text for marker in ("message", "messages", "posted", "sent")
    )


def build_query_planner_prompt(*, task: ClbenchTask, state: StructuredState, max_queries: int = 6) -> str:
    state_text = render_structured_state_for_prompt(state, max_evidence_rows=5)
    return f"""You are a CL-bench Life query planner for a retrieval harness.

Your job is NOT to answer the task. Your job is to decide which targeted evidence queries the harness should execute against the full context before the final answer writer responds.

Return strict JSON only with this schema:
{{
  "queries": [
    {{
      "purpose": "why this evidence is needed",
      "query": "short retrieval query with exact names, dates, phrases, entities, or fields",
      "expected_fields": ["field_or_fact_to_extract"],
      "quote_required": true,
      "priority": 1
    }}
  ],
  "coverage_items": [
    {{"kind": "person|item|date|field|section|event|other", "name": "...", "required": true}}
  ],
  "output_schema": ["column or section that the final answer should include"]
}}

Rules:
- Create at most {max_queries} queries.
- Prefer targeted queries over broad summaries.
- Include exact names/entities already visible in workflow tables when they matter.
- Query raw-context words, exact names, timestamps, and quoted phrases. Do not query internal table names such as `commenter_registry`, `speaker_activity`, or `llm_query_plan`.
- For identity, sentiment, rationale, recommendation, or chronology tasks, ask for direct quote evidence.
- For count/list/table tasks, include every required output field in `output_schema`.
- Do not use hidden rubrics or assume a known answer.

FINAL USER TASK:
{task.task}

CURRENT WORKFLOW STATE:
{state_text}
"""


def parse_query_plan_response(response: str, *, max_queries: int = 6) -> QueryPlan:
    parsed = parse_json_object(response)
    if parsed is None:
        return QueryPlan(enabled=True, used=False, status="parse_error", raw_response=response, errors=("query_plan_not_valid_json",))
    queries = parsed.get("queries")
    if not isinstance(queries, list):
        return QueryPlan(enabled=True, used=False, status="validation_error", raw_response=response, errors=("missing_or_non_list_queries",))

    requests: list[QueryRequest] = []
    for index, item in enumerate(queries[:max_queries], start=1):
        if not isinstance(item, dict):
            continue
        query = compact_text(str(item.get("query", "")), max_chars=180)
        if not query:
            continue
        fields = item.get("expected_fields", [])
        if not isinstance(fields, list):
            fields = [fields]
        try:
            priority = int(item.get("priority", index))
        except (TypeError, ValueError):
            priority = index
        requests.append(
            QueryRequest(
                query_id=f"q{index:02d}",
                purpose=compact_text(str(item.get("purpose", "")), max_chars=180),
                query=query,
                expected_fields=tuple(compact_text(str(field), max_chars=80) for field in fields[:8] if str(field).strip()),
                quote_required=bool(item.get("quote_required", False)),
                priority=priority,
            )
        )

    if not requests:
        return QueryPlan(enabled=True, used=False, status="validation_error", raw_response=response, errors=("no_valid_queries",))
    coverage_items = normalize_coverage_items(parsed.get("coverage_items", []))
    output_schema = normalize_string_tuple(parsed.get("output_schema", []), limit=20)
    return QueryPlan(
        enabled=True,
        used=True,
        status="ok",
        requests=tuple(sorted(requests, key=lambda request: request.priority)),
        coverage_items=coverage_items,
        output_schema=output_schema,
        raw_response=response,
    )


def execute_query_plan(
    *,
    chunks: list[Chunk],
    plan: QueryPlan,
    top_k_per_query: int = 4,
    neighbor_chunks: int = 1,
    max_total_hits: int = 28,
) -> QueryExecution:
    if not plan.used:
        return QueryExecution(plan=plan)
    index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    merged_hits: list[RetrievalHit] = []
    hit_ids: set[str] = set()
    query_hit_map: dict[str, list[str]] = {}
    evidence_rows: list[dict[str, Any]] = []

    for request in plan.requests:
        direct_hits = retrieve(chunks, request.query, top_k=top_k_per_query)
        expanded_hits = expand_hits_with_neighbors(chunks, direct_hits, index_by_id=index_by_id, neighbor_chunks=neighbor_chunks)
        query_hit_map[request.query_id] = [hit.chunk.chunk_id for hit in expanded_hits]
        for rank, hit in enumerate(expanded_hits, start=1):
            evidence_rows.append(
                {
                    "query_id": request.query_id,
                    "rank": rank,
                    "chunk_id": hit.chunk.chunk_id,
                    "lines": hit.chunk.line_span,
                    "score": hit.score,
                    "matched_terms": list(hit.matched_terms[:8]),
                    "purpose": request.purpose,
                    "snippet": compact_text(hit.chunk.text, max_chars=420),
                }
            )
            if hit.chunk.chunk_id in hit_ids:
                continue
            hit_ids.add(hit.chunk.chunk_id)
            merged_hits.append(hit)
            if len(merged_hits) >= max_total_hits:
                return QueryExecution(plan=plan, hits=tuple(merged_hits), evidence_rows=tuple(evidence_rows), query_hit_map=query_hit_map)
    return QueryExecution(plan=plan, hits=tuple(merged_hits), evidence_rows=tuple(evidence_rows), query_hit_map=query_hit_map)


def merge_query_execution_into_state(state: StructuredState, execution: QueryExecution) -> StructuredState:
    if not execution.plan.used:
        return state
    tables = {key: [dict(row) for row in rows] for key, rows in state.tables.items()}
    tables["llm_query_plan"] = [request.to_dict() for request in execution.plan.requests]
    tables["llm_query_coverage"] = list(execution.plan.coverage_items)
    tables["llm_query_output_schema"] = [{"rank": index + 1, "field": field} for index, field in enumerate(execution.plan.output_schema)]
    tables["llm_query_evidence"] = list(execution.evidence_rows[:80])
    warnings = list(state.warnings)
    if execution.plan.requests and not execution.hits:
        warnings.append("LLM query planner produced queries but retrieval found no matching chunks.")
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
        trigger_signals=[*state.trigger_signals, "llm_query_planner"],
        warnings=warnings,
        fallback_recommendation=state.fallback_recommendation,
    )


def merge_query_hits(base_hits: list[RetrievalHit], query_hits: tuple[RetrievalHit, ...], *, max_hits: int = 34) -> list[RetrievalHit]:
    merged: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in [*base_hits, *query_hits]:
        chunk_id = hit.chunk.chunk_id
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(hit)
        if len(merged) >= max_hits:
            break
    return merged


def expand_hits_with_neighbors(
    chunks: list[Chunk],
    hits: list[RetrievalHit],
    *,
    index_by_id: dict[str, int],
    neighbor_chunks: int,
) -> list[RetrievalHit]:
    if neighbor_chunks <= 0 or not hits:
        return hits
    direct_hit_by_id = {hit.chunk.chunk_id: hit for hit in hits}
    expanded: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:
        center = index_by_id.get(hit.chunk.chunk_id)
        if center is None:
            continue
        for index in range(max(0, center - neighbor_chunks), min(len(chunks), center + neighbor_chunks + 1)):
            chunk = chunks[index]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            if chunk.chunk_id in direct_hit_by_id:
                expanded.append(direct_hit_by_id[chunk.chunk_id])
            else:
                expanded.append(RetrievalHit(chunk=chunk, score=0.0, matched_terms=(f"neighbor_of:{hit.chunk.chunk_id}",)))
    return expanded


def normalize_coverage_items(items: Any, *, limit: int = 80) -> tuple[dict[str, Any], ...]:
    if not isinstance(items, list):
        return ()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items[:limit], start=1):
        if isinstance(item, dict):
            name = compact_text(str(item.get("name", "")), max_chars=140)
            kind = compact_text(str(item.get("kind", "other")), max_chars=40)
            required = bool(item.get("required", True))
        else:
            name = compact_text(str(item), max_chars=140)
            kind = "other"
            required = True
        if name:
            rows.append({"coverage_id": f"plan{index:03d}", "kind": kind, "name": name, "required": required})
    return tuple(rows)


def normalize_string_tuple(items: Any, *, limit: int) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    return tuple(compact_text(str(item), max_chars=120) for item in items[:limit] if str(item).strip())
