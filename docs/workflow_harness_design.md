# CL-bench Life Workflow Harness Design

本项目不把 9 个上下文类别实现成 9 个独立 solver，而是抽象成 4 条可复用 workflow 加 2 个横向模块。

核心原则：

- Python 负责让信息可靠：解析、结构化、去重、计数、计算、格式硬校验。
- LLM 负责让信息有意义：语义抽取、社会关系判断、跨文档归纳、最终答案表达。
- Deterministic solver 只处理可验证子问题，不作为默认答题方式。
- 每次运行都产出可观测中间态：workflow、structured state、evidence table、computed values、uncertain items、constraints、verification result。

## System Flow

```text
parse_task
  -> route_context_type
  -> route_task_operator
  -> select_workflow
  -> build_structured_state
  -> render_evidence_pack
  -> LLM answer writer or exact tool answer
  -> constraint verification
  -> trace and failure attribution
```

## Four Workflows

| Workflow | Covers | Main state |
| --- | --- | --- |
| `structured_log_workflow` | Game Logs, Digital Footprints, Self-Tracking | event table, aggregates, computed values |
| `thread_tally_workflow` | Community Interactions | comments, user/stance table, tally, ambiguous comments |
| `dialogue_state_workflow` | Group Conversations, Private Conversations | speaker profiles, issues, commitments, conflicts, open loops |
| `multi_doc_matrix_workflow` | Creation/Revisions, Personal Info, Public Info | document table, claim matrix, candidate-condition matrix |

## Nine Categories To Workflow Mapping

| CL-bench Life category | Primary workflow | Notes |
| --- | --- | --- |
| Game Logs | `structured_log_workflow` | Exact parsers for PGN, replay logs, match JSON, RSS-like game logs. |
| Digital Footprints & Daily-Life Records | `structured_log_workflow` | Time aggregation and trend detection before LLM explanation. |
| Self-Tracking Trajectories | `structured_log_workflow` | Numeric trends, anomalies, goals, health/activity trajectories. |
| Community Interactions | `thread_tally_workflow` | Comment/user table first, then tally or stance reasoning. |
| Group Conversations & Meeting Transcripts | `dialogue_state_workflow` | Speaker state, agenda, action items, social roles. |
| Private Conversations | `dialogue_state_workflow` | Intent, concern, support, tension, relationship state. |
| Creation & Revision Histories | `multi_doc_matrix_workflow` | Version separation, diff, changed/unchanged claims. |
| Personal Information Fragments | `multi_doc_matrix_workflow` | Entity profile and evidence matrix across fragments. |
| Public Information Fragments | `multi_doc_matrix_workflow` | Candidate filtering and claim-source matrix. |

## Structured State

All workflows return the same outer shape:

```text
StructuredState(
  workflow,
  context_type,
  task_operator,
  evidence_table,
  computed_values,
  entity_profiles,
  timeline,
  candidate_answers,
  uncertain_items,
  parser_confidence,
  trigger_signals,
  warnings,
  fallback_recommendation,
)
```

This makes traces comparable across task types. A failure can be attributed to:

- wrong workflow selection;
- weak evidence retrieval;
- missing structured state;
- wrong computation;
- semantic inference failure;
- final format or constraint failure.

## Current Implementation Status

Implemented:

- `clbench_life_harness.workflows` with `StructuredState`, `EvidenceRow`, `ComputedValue`, `CandidateAnswer`, `UncertainItem`, and `WorkflowDecision`.
- Two-level routing: `context_type + task_operator -> workflow`.
- Prediction traces now include `workflow_decision` and full `structured_state`.
- Prompt now includes `WORKFLOW STRUCTURED STATE` before raw evidence.
- Existing deterministic tools are exposed as computed/candidate state rather than being framed as the whole system.
- Workflow-specific starter tables:
  - `structured_log_workflow`: `events`, `aggregates`
  - `thread_tally_workflow`: `comments`, `stances`, `tally`
  - `dialogue_state_workflow`: `turns`, `issues`, plus speaker `entity_profiles`
  - `multi_doc_matrix_workflow`: `documents`, `claim_doc_matrix`, `candidate_matrix`
- `clbench_life_harness.verifier` provides a Python hard verifier for:
  - non-empty output;
  - JSON/table/list shape;
  - required count;
  - exact quote substring checks;
  - username grounding;
  - number/state consistency warnings;
  - truncation;
  - workflow-state warnings and failure attribution.
- Harness-level model retries now support exponential backoff through:
  - `--model-call-attempts`
  - `--retry-backoff-base`
  - `--retry-backoff-max`

Still needed:

- Move structured log parsers into a workflow package.
- Add stronger `dialogue_state_workflow` speaker state extraction.
- Add stronger `multi_doc_matrix_workflow` document and claim matrix builders.
- Add optional LLM semantic verifier/repair for non-exact workflows.
