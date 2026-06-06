from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
NEXAU_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]

for path in (str(NEXAU_ROOT), str(WORKSPACE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nexau import Agent, AgentConfig, LLMConfig  # noqa: E402
from nexau.archs.main_sub.execution.hooks import AfterModelHookInput, HookResult  # noqa: E402

from clbench_life_harness.chunking import Chunk, chunk_text  # noqa: E402
from clbench_life_harness.constraints import extract_constraints, render_constraints, verify_answer_constraints  # noqa: E402
from clbench_life_harness.data import ClbenchTask, load_tasks, load_tasks_by_ids, read_ids_file  # noqa: E402
from clbench_life_harness.deterministic import (  # noqa: E402
    build_deterministic_answer,
    build_deterministic_findings,
    build_workflow_deterministic_answer,
    render_findings,
)
from clbench_life_harness.query_planner import (  # noqa: E402
    QueryExecution,
    QueryPlan,
    build_query_planner_prompt,
    execute_query_plan,
    is_exact_list_task,
    merge_query_execution_into_state,
    merge_query_hits,
    parse_query_plan_response,
    should_plan_queries,
)
from clbench_life_harness.retrieval import RetrievalHit, retrieve  # noqa: E402
from clbench_life_harness.state_refiner import (  # noqa: E402
    StateRefinement,
    build_state_refiner_prompt,
    merge_refinement_into_state,
    parse_state_refinement_response,
    should_refine_state,
)
from clbench_life_harness.task_router import route_task  # noqa: E402
from clbench_life_harness.trace import build_trace_record  # noqa: E402
from clbench_life_harness.verifier import verify_answer_hard  # noqa: E402
from clbench_life_harness.workflows import build_structured_state, render_structured_state_for_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NexAU CL-bench Life predictions.")
    parser.add_argument("--input", required=True, type=Path, help="Input CL-bench JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Prediction JSONL output path.")
    parser.add_argument("--trace-dir", required=True, type=Path, help="Directory for per-task traces.")
    parser.add_argument("--start", type=int, default=0, help="0-based row offset.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional JSON/text file of original row ids to process.")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel model requests. Keep <= 5 for Nex-N2.")
    parser.add_argument("--max-chars", type=int, default=3500, help="Maximum characters per retrieval chunk.")
    parser.add_argument("--top-k", type=int, default=14, help="Number of evidence chunks to keep.")
    parser.add_argument(
        "--neighbor-chunks",
        type=int,
        default=1,
        help="Include this many adjacent chunks around each retrieved hit in the evidence pack.",
    )
    parser.add_argument("--evidence-char-budget", type=int, default=52000, help="Maximum evidence text chars in the prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature.")
    parser.add_argument("--max-tokens", type=int, default=12000, help="Maximum completion tokens.")
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum API retries per model request.")
    parser.add_argument("--model-call-attempts", type=int, default=2, help="Harness-level LLM attempts with exponential backoff.")
    parser.add_argument("--retry-backoff-base", type=float, default=2.0, help="Base seconds for harness-level exponential backoff.")
    parser.add_argument("--retry-backoff-max", type=float, default=30.0, help="Maximum seconds for harness-level exponential backoff.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum NexAU agent iterations per task.")
    parser.add_argument(
        "--thinking-mode",
        choices=("auto", "off", "budget"),
        default="off",
        help="Qwen-compatible thinking control: auto leaves the provider default, off disables, budget caps thinking tokens.",
    )
    parser.add_argument("--thinking-budget", type=int, default=2048, help="Thinking token budget used when --thinking-mode budget.")
    parser.add_argument(
        "--thinking-param-shape",
        choices=("chat_template_kwargs", "top_level", "both"),
        default="chat_template_kwargs",
        help="Where to put Qwen/SGLang thinking controls inside OpenAI extra_body.",
    )
    parser.add_argument(
        "--deterministic-answer-mode",
        choices=("auto", "off"),
        default="auto",
        help="Use deterministic final answers when the harness has a reliable renderer.",
    )
    parser.add_argument(
        "--state-refiner",
        choices=("off", "auto"),
        default="off",
        help="Optionally run a narrow LLM JSON state refiner before final answer generation.",
    )
    parser.add_argument("--state-refiner-max-tokens", type=int, default=4000, help="Maximum completion tokens for state refiner JSON.")
    parser.add_argument(
        "--query-planner",
        choices=("off", "auto"),
        default="off",
        help="Optionally run an LLM JSON query planner and execute targeted retrieval before final answer generation.",
    )
    parser.add_argument("--query-planner-max-tokens", type=int, default=3000, help="Maximum completion tokens for query planner JSON.")
    parser.add_argument("--query-planner-max-queries", type=int, default=6, help="Maximum targeted retrieval queries generated by the planner.")
    parser.add_argument("--query-planner-top-k", type=int, default=4, help="Chunks to retrieve per planner query before neighbor expansion.")
    parser.add_argument("--query-planner-max-hits", type=int, default=28, help="Maximum extra evidence chunks kept from planner queries.")
    parser.add_argument(
        "--answer-repair",
        choices=("off", "auto"),
        default="off",
        help="Optionally run bounded LLM repair passes when verifier detects malformed or incomplete output.",
    )
    parser.add_argument("--answer-repair-max-tokens", type=int, default=7000, help="Maximum completion tokens for each repair pass.")
    parser.add_argument("--answer-repair-max-attempts", type=int, default=2, help="Maximum verifier-repair loop attempts per task.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and traces without calling the model.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output/partial prediction records.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow empty model predictions to be written as successful records.")
    parser.add_argument("--allow-framework-error", action="store_true", help="Allow known NexAU framework error strings to be written as successful records.")
    parser.add_argument("--allow-truncation", action="store_true", help="Allow predictions that appear to hit the max token limit.")
    parser.add_argument("--continue-on-error", action="store_true", help="Write an error prediction record and continue when a task fails.")
    args = parser.parse_args()

    if args.concurrency < 1 or args.concurrency > 5:
        raise ValueError("--concurrency must be between 1 and 5")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be at least 1")
    if args.neighbor_chunks < 0:
        raise ValueError("--neighbor-chunks must be non-negative")
    if args.thinking_budget < 0:
        raise ValueError("--thinking-budget must be non-negative")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    if args.model_call_attempts < 1:
        raise ValueError("--model-call-attempts must be at least 1")
    if args.retry_backoff_base < 0 or args.retry_backoff_max < 0:
        raise ValueError("--retry backoff values must be non-negative")
    if args.state_refiner_max_tokens < 500:
        raise ValueError("--state-refiner-max-tokens must be at least 500")
    if args.query_planner_max_tokens < 500:
        raise ValueError("--query-planner-max-tokens must be at least 500")
    if args.query_planner_max_queries < 1:
        raise ValueError("--query-planner-max-queries must be at least 1")
    if args.query_planner_top_k < 1:
        raise ValueError("--query-planner-top-k must be at least 1")
    if args.query_planner_max_hits < 1:
        raise ValueError("--query-planner-max-hits must be at least 1")
    if args.answer_repair_max_tokens < 500:
        raise ValueError("--answer-repair-max-tokens must be at least 500")
    if args.answer_repair_max_attempts < 1:
        raise ValueError("--answer-repair-max-attempts must be at least 1")

    model = os.getenv("LLM_MODEL", "nex-agi/Nex-N2-Pro")
    base_url = os.getenv("LLM_BASE_URL", "https://northgate.xiaobei.top/v1")

    if args.ids_file:
        tasks = load_tasks_by_ids(args.input, read_ids_file(args.ids_file))
    else:
        tasks = load_tasks(args.input, start=args.start, limit=args.limit)
    if not args.dry_run and requires_llm(
        tasks,
        args.deterministic_answer_mode,
        query_planner_mode=args.query_planner,
        state_refiner_mode=args.state_refiner,
        answer_repair_mode=args.answer_repair,
    ):
        require_env("LLM_API_KEY")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(args.output.name + ".partial")
    completed_records: dict[int, dict[str, Any]] = {}
    if args.resume:
        completed_records.update(read_prediction_records(partial_output))
        completed_records.update(read_prediction_records(args.output))
    else:
        for path in (args.output, partial_output):
            if path.exists():
                path.unlink()
    tasks_to_run = [task for task in tasks if task.row_id not in completed_records]
    task_by_id = {task.row_id: task for task in tasks}

    results: list[dict[str, Any]] = [
        {"id": row_id, "prediction_record": record, "latency_sec": 0.0, "resumed": True}
        for row_id, record in completed_records.items()
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_one_task,
                task,
                model,
                base_url,
                args.temperature,
                args.max_tokens,
                args.request_timeout,
                args.max_retries,
                args.model_call_attempts,
                args.retry_backoff_base,
                args.retry_backoff_max,
                args.max_iterations,
                args.thinking_mode,
                args.thinking_budget,
                args.thinking_param_shape,
                args.deterministic_answer_mode,
                args.state_refiner,
                args.state_refiner_max_tokens,
                args.query_planner,
                args.query_planner_max_tokens,
                args.query_planner_max_queries,
                args.query_planner_top_k,
                args.query_planner_max_hits,
                args.answer_repair,
                args.answer_repair_max_tokens,
                args.answer_repair_max_attempts,
                args.max_chars,
                args.top_k,
                args.neighbor_chunks,
                args.evidence_char_budget,
                args.trace_dir,
                args.dry_run,
                args.allow_empty,
                args.allow_framework_error,
                args.allow_truncation,
            ): task.row_id
            for task in tasks_to_run
        }
        for future in as_completed(futures):
            row_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                append_prediction_record(partial_output, result["prediction_record"])
                print(f"finished task {row_id}")
            except Exception as exc:
                print(f"failed task {row_id}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
                error_record = build_error_prediction_record(task_by_id[row_id], exc)
                result = {
                    "id": row_id,
                    "prediction_record": error_record,
                    "latency_sec": 0.0,
                    "error": str(exc),
                }
                results.append(result)
                append_prediction_record(partial_output, error_record)

    result_by_id = {int(item["id"]): item for item in results}
    missing_ids = [task.row_id for task in tasks if task.row_id not in result_by_id]
    if missing_ids:
        raise RuntimeError(f"Missing prediction records for ids: {missing_ids}. Re-run with --resume.")
    with args.output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            result = result_by_id[task.row_id]
            handle.write(json.dumps(result["prediction_record"], ensure_ascii=False) + "\n")

    run_summary = {
        "input": str(args.input),
        "output": str(args.output),
        "trace_dir": str(args.trace_dir),
        "processed": len(results),
        "newly_processed": len(tasks_to_run),
        "resumed": len(completed_records),
        "start": args.start,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "model_call_attempts": args.model_call_attempts,
        "retry_backoff_base": args.retry_backoff_base,
        "retry_backoff_max": args.retry_backoff_max,
        "max_iterations": args.max_iterations,
        "neighbor_chunks": args.neighbor_chunks,
        "thinking_mode": args.thinking_mode,
        "thinking_budget": args.thinking_budget if args.thinking_mode == "budget" else None,
        "thinking_param_shape": args.thinking_param_shape,
        "deterministic_answer_mode": args.deterministic_answer_mode,
        "state_refiner": args.state_refiner,
        "state_refiner_max_tokens": args.state_refiner_max_tokens,
        "query_planner": args.query_planner,
        "query_planner_max_tokens": args.query_planner_max_tokens,
        "query_planner_max_queries": args.query_planner_max_queries,
        "query_planner_top_k": args.query_planner_top_k,
        "query_planner_max_hits": args.query_planner_max_hits,
        "answer_repair": args.answer_repair,
        "answer_repair_max_tokens": args.answer_repair_max_tokens,
        "answer_repair_max_attempts": args.answer_repair_max_attempts,
        "model": model,
        "base_url": base_url,
        "dry_run": args.dry_run,
        "continue_on_error": args.continue_on_error,
        "error_records": sum(1 for item in results if "error" in item or item["prediction_record"].get("harness_error")),
        "avg_latency_sec": round(sum(item["latency_sec"] for item in results) / len(results), 2) if results else 0,
    }
    (args.trace_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


def require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Missing environment variable: {name}")


def requires_llm(
    tasks: list[ClbenchTask],
    deterministic_answer_mode: str,
    *,
    query_planner_mode: str,
    state_refiner_mode: str,
    answer_repair_mode: str,
) -> bool:
    if query_planner_mode == "auto" or state_refiner_mode == "auto" or answer_repair_mode == "auto":
        return True
    if deterministic_answer_mode == "off":
        return True
    return any(build_deterministic_answer(task) is None for task in tasks)


def read_prediction_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records[int(row["id"])] = row
    return records


def append_prediction_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_error_prediction_record(task: ClbenchTask, exc: Exception) -> dict[str, Any]:
    message = f"[HARNESS_ERROR] {type(exc).__name__}: {exc}"
    return {
        "id": task.row_id,
        "prediction": "",
        "messages": [*task.messages, {"role": "assistant", "content": ""}],
        "harness_error": message,
    }


def build_tool_observations(deterministic_answer: Any | None) -> list[dict[str, Any]]:
    if deterministic_answer is None:
        return []
    return [
        {
            "tool": deterministic_answer.tool_name or deterministic_answer.kind,
            "kind": deterministic_answer.kind,
            "summary": deterministic_answer.summary,
            "confidence": deterministic_answer.confidence,
            "trigger_signals": deterministic_answer.trigger_signals,
            "parsed_state_summary": deterministic_answer.parsed_state_summary,
            "skipped_or_uncertain_items": deterministic_answer.skipped_or_uncertain_items,
            "answer_is_exact": deterministic_answer.answer_is_exact,
            "llm_review_recommended": deterministic_answer.llm_review_recommended,
        }
    ]


class LlmObservationCollector:
    def __init__(self, observations: list[dict[str, Any]]) -> None:
        self.observations = observations

    def __call__(self, hook_input: AfterModelHookInput) -> HookResult:
        self.observations.append(summarize_after_model_hook(hook_input))
        return HookResult.no_changes()


def summarize_after_model_hook(hook_input: AfterModelHookInput) -> dict[str, Any]:
    model_response = hook_input.model_response
    if model_response is None:
        return {
            "iteration": hook_input.current_iteration,
            "model_response_type": None,
            "original_response_len": len(hook_input.original_response or ""),
            "original_response_preview": (hook_input.original_response or "")[:500],
        }

    content = model_response.content or ""
    reasoning_content = model_response.reasoning_content or ""
    usage = summarize_usage(model_response.usage)
    return {
        "iteration": hook_input.current_iteration,
        "max_iterations": hook_input.max_iterations,
        "model_response_type": type(model_response).__name__,
        "content_len": len(content),
        "content_preview": content[:1000],
        "reasoning_content_len": len(reasoning_content),
        "has_reasoning_content": bool(reasoning_content),
        "tool_calls_count": len(model_response.tool_calls),
        "usage": usage,
        "raw_message_summary": summarize_raw_message(model_response.raw_message),
    }


def summarize_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    to_dict = getattr(usage, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return data if isinstance(data, dict) else {"repr_preview": repr(usage)[:500]}
    data = object_to_plain_dict(usage)
    if data is not None:
        return data
    return {"repr_preview": repr(usage)[:500]}


def summarize_agent_response(response: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(response).__name__,
        "repr_preview": repr(response)[:1000],
    }
    if isinstance(response, tuple):
        summary["tuple_len"] = len(response)
        summary["tuple_item_types"] = [type(item).__name__ for item in response]
    if isinstance(response, str):
        summary["str_len"] = len(response)
        summary["str_preview"] = response[:1000]
    return summary


def summarize_raw_message(raw_message: Any) -> dict[str, Any]:
    if raw_message is None:
        return {"type": None}
    data = object_to_plain_dict(raw_message)
    if not isinstance(data, dict):
        return {"type": type(raw_message).__name__, "repr_preview": repr(raw_message)[:500]}

    sensitive_reasoning_keys = {"reasoning", "reasoning_content", "reasoning_details"}
    summary: dict[str, Any] = {"type": type(raw_message).__name__, "keys": sorted(str(key) for key in data)}
    for key, value in data.items():
        key_text = str(key)
        if key_text in sensitive_reasoning_keys:
            summary[key_text] = summarize_value_shape(value)
        elif key_text in {"content", "text", "role", "finish_reason", "refusal"}:
            summary[key_text] = summarize_value_preview(value)
        elif key_text in {"tool_calls", "function_call"}:
            summary[key_text] = summarize_value_shape(value)
    return summary


def object_to_plain_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return dumped if isinstance(dumped, dict) else None
    return None


def summarize_value_shape(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": None, "present": False}
    if isinstance(value, str):
        return {"type": "str", "present": bool(value), "len": len(value)}
    if isinstance(value, list):
        return {"type": "list", "len": len(value), "item_types": sorted({type(item).__name__ for item in value})}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value)}
    return {"type": type(value).__name__, "present": True}


def summarize_value_preview(value: Any, *, max_chars: int = 1000) -> dict[str, Any]:
    if value is None:
        return {"type": None, "present": False}
    if isinstance(value, str):
        return {"type": "str", "len": len(value), "preview": value[:max_chars]}
    if isinstance(value, list):
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        return {"type": "list", "len": len(value), "preview": rendered[:max_chars]}
    rendered = str(value)
    return {"type": type(value).__name__, "len": len(rendered), "preview": rendered[:max_chars]}


def build_agent_config(
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
    max_iterations: int,
    after_model_hooks: list[Any] | None = None,
) -> AgentConfig:
    llm_config_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": os.getenv("LLM_API_KEY"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_type": "openai_chat_completion",
        "stream": False,
        "timeout": request_timeout,
        "max_retries": max_retries,
    }
    if thinking_request:
        llm_config_kwargs["extra_body"] = thinking_request

    return AgentConfig(
        name="clbench_life_answer_agent",
        max_context_tokens=120000,
        max_iterations=max_iterations,
        system_prompt=str(SCRIPT_DIR / "systemprompt.md"),
        system_prompt_type="file",
        tool_call_mode="xml",
        llm_config=LLMConfig(**llm_config_kwargs),
        tools=[],
        skills=[],
        after_model_hooks=after_model_hooks,
    )


def build_thinking_request(thinking_mode: str, thinking_budget: int, thinking_param_shape: str) -> dict[str, Any]:
    if thinking_mode == "auto":
        return {}
    template_kwargs: dict[str, Any]
    if thinking_mode == "off":
        template_kwargs = {"enable_thinking": False}
    else:
        template_kwargs = {"enable_thinking": True, "thinking_budget": thinking_budget}

    if thinking_param_shape == "chat_template_kwargs":
        return {"chat_template_kwargs": template_kwargs}
    if thinking_param_shape == "top_level":
        return dict(template_kwargs)
    if thinking_param_shape == "both":
        return {**template_kwargs, "chat_template_kwargs": template_kwargs}
    raise ValueError(f"Unsupported thinking_param_shape: {thinking_param_shape}")


def apply_prompt_thinking_control(prompt: str, thinking_mode: str) -> str:
    if thinking_mode == "off":
        return "/no_think\n\n" + prompt
    return prompt


def run_one_task(
    task: ClbenchTask,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    model_call_attempts: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
    max_iterations: int,
    thinking_mode: str,
    thinking_budget: int,
    thinking_param_shape: str,
    deterministic_answer_mode: str,
    state_refiner_mode: str,
    state_refiner_max_tokens: int,
    query_planner_mode: str,
    query_planner_max_tokens: int,
    query_planner_max_queries: int,
    query_planner_top_k: int,
    query_planner_max_hits: int,
    answer_repair_mode: str,
    answer_repair_max_tokens: int,
    answer_repair_max_attempts: int,
    max_chars: int,
    top_k: int,
    neighbor_chunks: int,
    evidence_char_budget: int,
    trace_dir: Path,
    dry_run: bool,
    allow_empty: bool,
    allow_framework_error: bool,
    allow_truncation: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_route = route_task(task)
    task_constraints = extract_constraints(task)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id, max_chars=max_chars)
    retrieved_hits = retrieve(chunks, task.task, top_k=top_k)
    hits = expand_hits_with_neighbors(chunks, retrieved_hits, neighbor_chunks=neighbor_chunks)
    deterministic_findings = build_deterministic_findings(task)
    deterministic_answer = build_deterministic_answer(task) if deterministic_answer_mode == "auto" else None
    structured_state = build_structured_state(
        task=task,
        task_route=task_route,
        hits=hits,
        deterministic_findings=deterministic_findings,
        deterministic_answer=deterministic_answer,
    )
    if deterministic_answer is None and deterministic_answer_mode == "auto":
        deterministic_answer = build_workflow_deterministic_answer(task, structured_state)
        if deterministic_answer is not None:
            structured_state = build_structured_state(
                task=task,
                task_route=task_route,
                hits=hits,
                deterministic_findings=deterministic_findings,
                deterministic_answer=deterministic_answer,
            )
    thinking_request = build_thinking_request(thinking_mode, thinking_budget, thinking_param_shape)
    query_plan = QueryPlan(enabled=query_planner_mode != "off", used=False, status="disabled")
    query_execution = QueryExecution(plan=query_plan)
    query_planner_debug: dict[str, Any] = {"mode": query_planner_mode}
    if (
        not dry_run
        and query_planner_mode == "auto"
        and should_plan_queries(task, structured_state, deterministic_answer_available=deterministic_answer is not None)
    ):
        query_planner_prompt = apply_prompt_thinking_control(
            build_query_planner_prompt(task=task, state=structured_state, max_queries=query_planner_max_queries),
            thinking_mode,
        )
        try:
            query_planner_raw, query_planner_attempts = run_query_planner_with_retries(
                prompt=query_planner_prompt,
                model=model,
                base_url=base_url,
                temperature=0.0,
                max_tokens=query_planner_max_tokens,
                request_timeout=request_timeout,
                max_retries=max_retries,
                thinking_request=thinking_request,
                model_call_attempts=model_call_attempts,
                retry_backoff_base=retry_backoff_base,
                retry_backoff_max=retry_backoff_max,
            )
            query_plan = parse_query_plan_response(query_planner_raw, max_queries=query_planner_max_queries)
            query_execution = execute_query_plan(
                chunks=chunks,
                plan=query_plan,
                top_k_per_query=query_planner_top_k,
                neighbor_chunks=neighbor_chunks,
                max_total_hits=query_planner_max_hits,
            )
            structured_state = merge_query_execution_into_state(structured_state, query_execution)
            if query_execution.hits:
                hits = merge_query_hits(hits, query_execution.hits)
            query_planner_debug = {
                "mode": query_planner_mode,
                "prompt_chars": len(query_planner_prompt),
                "attempts": query_planner_attempts,
                "execution": query_execution.to_dict(),
            }
        except Exception as exc:
            query_plan = QueryPlan(
                enabled=True,
                used=False,
                status="runtime_error",
                errors=(f"{type(exc).__name__}: {exc}",),
            )
            query_execution = QueryExecution(plan=query_plan)
            query_planner_debug = {
                "mode": query_planner_mode,
                "prompt_chars": len(query_planner_prompt),
                "execution": query_execution.to_dict(),
            }
    elif query_planner_mode == "auto":
        query_planner_debug = {
            "mode": query_planner_mode,
            "skipped": True,
            "reason": "dry_run_or_not_plannable_or_deterministic_answer",
        }
    state_refinement = StateRefinement(enabled=state_refiner_mode != "off", used=False, status="disabled")
    state_refiner_debug: dict[str, Any] = {"mode": state_refiner_mode}
    if (
        not dry_run
        and state_refiner_mode == "auto"
        and should_refine_state(structured_state, deterministic_answer_available=deterministic_answer is not None)
    ):
        refiner_prompt = apply_prompt_thinking_control(
            build_state_refiner_prompt(task=task, state=structured_state),
            thinking_mode,
        )
        try:
            refiner_raw, refiner_attempts = run_state_refiner_with_retries(
                prompt=refiner_prompt,
                model=model,
                base_url=base_url,
                temperature=0.0,
                max_tokens=state_refiner_max_tokens,
                request_timeout=request_timeout,
                max_retries=max_retries,
                thinking_request=thinking_request,
                model_call_attempts=model_call_attempts,
                retry_backoff_base=retry_backoff_base,
                retry_backoff_max=retry_backoff_max,
            )
            state_refinement = parse_state_refinement_response(refiner_raw)
            structured_state = merge_refinement_into_state(structured_state, state_refinement)
            state_refiner_debug = {
                "mode": state_refiner_mode,
                "prompt_chars": len(refiner_prompt),
                "attempts": refiner_attempts,
                "result": state_refinement.to_dict(),
            }
        except Exception as exc:
            state_refinement = StateRefinement(
                enabled=True,
                used=False,
                status="runtime_error",
                errors=[f"{type(exc).__name__}: {exc}"],
            )
            state_refiner_debug = {
                "mode": state_refiner_mode,
                "prompt_chars": len(refiner_prompt),
                "result": state_refinement.to_dict(),
            }
    elif state_refiner_mode == "auto":
        state_refiner_debug = {
            "mode": state_refiner_mode,
            "skipped": True,
            "reason": "dry_run_or_not_refinable_or_deterministic_answer",
        }
    prompt = build_prompt(
        task=task,
        hits=hits,
        task_route=task_route,
        task_constraints=task_constraints,
        deterministic_findings_text=render_findings(deterministic_findings),
        structured_state_text=render_structured_state_for_prompt(structured_state),
        evidence_char_budget=evidence_char_budget,
    )
    prompt = apply_prompt_thinking_control(prompt, thinking_mode)

    llm_observations: list[dict[str, Any]] = []
    agent_response_debug: dict[str, Any] | None = None
    agent_attempts_debug: list[dict[str, Any]] = []
    if dry_run:
        prediction = "[DRY RUN] Model call skipped. Inspect trace prompt and evidence pack."
    elif deterministic_answer:
        prediction = deterministic_answer.content.strip()
        agent_response_debug = {
            "type": "deterministic_answer",
            "kind": deterministic_answer.kind,
            "summary": deterministic_answer.summary,
            "tool_name": deterministic_answer.tool_name,
            "llm_role": deterministic_answer.llm_role,
            "answer_source": deterministic_answer.answer_source,
            "answer_is_exact": deterministic_answer.answer_is_exact,
        }
    else:
        prediction, agent_response_debug, llm_observations, agent_attempts_debug = run_agent_with_retries(
            prompt=prompt,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_retries=max_retries,
            thinking_request=thinking_request,
            max_iterations=max_iterations,
            model_call_attempts=model_call_attempts,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

    prediction_record = {
        "id": task.row_id,
        "prediction": prediction,
        "messages": [*task.messages, {"role": "assistant", "content": prediction}],
    }
    latency_sec = round(time.perf_counter() - started, 2)
    truncation = detect_suspected_truncation(
        prediction=prediction,
        llm_observations=llm_observations,
        max_tokens=max_tokens,
    )
    constraint_verification = verify_answer_constraints(
        task=task,
        answer=prediction,
        constraints=task_constraints,
    )
    if dry_run:
        hard_verification = {
            "skipped": True,
            "reason": "dry_run_model_call_skipped",
            "passed": False,
            "checks": [],
            "failure_type_hints": [],
        }
    else:
        hard_verification = verify_answer_hard(
            task=task,
            answer=prediction,
            constraints=task_constraints,
            structured_state=structured_state,
            truncation=truncation,
        )
    answer_repair_debug: dict[str, Any] = {
        "mode": answer_repair_mode,
        "used": False,
        "max_attempts": answer_repair_max_attempts,
        "repair_rounds": [],
    }
    if not dry_run and not deterministic_answer and answer_repair_mode == "auto":
        repair_rounds: list[dict[str, Any]] = []
        for repair_index in range(1, answer_repair_max_attempts + 1):
            if not should_repair_answer(
                task=task,
                structured_state=structured_state,
                truncation=truncation,
                hard_verification=hard_verification,
            ):
                answer_repair_debug["stop_reason"] = "verification_not_repairable_or_already_clean"
                break

            repair_prompt = apply_prompt_thinking_control(
                build_answer_repair_prompt(
                task=task,
                original_answer=prediction,
                structured_state=structured_state,
                structured_state_text=render_structured_state_for_prompt(structured_state, max_evidence_rows=6),
                truncation=truncation,
                hard_verification=hard_verification,
                ),
                thinking_mode,
            )
            repair_round: dict[str, Any] = {
                "repair_index": repair_index,
                "prompt_chars": len(repair_prompt),
                "pre_repair_failed_checks": failed_verification_check_names(hard_verification),
                "pre_repair_truncation": truncation,
            }
            try:
                repaired_prediction, repair_attempts = run_answer_repair_with_retries(
                    prompt=repair_prompt,
                    model=model,
                    base_url=base_url,
                    temperature=0.0,
                    max_tokens=answer_repair_max_tokens,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    thinking_request=thinking_request,
                    model_call_attempts=model_call_attempts,
                    retry_backoff_base=retry_backoff_base,
                    retry_backoff_max=retry_backoff_max,
                )
                repair_round["attempts"] = repair_attempts
                repair_round["response_len"] = len(repaired_prediction or "")
                if repaired_prediction and not is_framework_failure(repaired_prediction):
                    prediction = repaired_prediction
                    truncation = detect_suspected_truncation(
                        prediction=prediction,
                        llm_observations=llm_observations,
                        max_tokens=max_tokens,
                    )
                    constraint_verification = verify_answer_constraints(
                        task=task,
                        answer=prediction,
                        constraints=task_constraints,
                    )
                    hard_verification = verify_answer_hard(
                        task=task,
                        answer=prediction,
                        constraints=task_constraints,
                        structured_state=structured_state,
                        truncation=truncation,
                    )
                    repair_round["accepted"] = True
                    repair_round["post_repair_failed_checks"] = failed_verification_check_names(hard_verification)
                    repair_round["post_repair_truncation"] = truncation
                    repair_round["post_repair_hard_verification"] = hard_verification
                    answer_repair_debug["used"] = True
                    repair_rounds.append(repair_round)
                    if not should_repair_answer(
                        task=task,
                        structured_state=structured_state,
                        truncation=truncation,
                        hard_verification=hard_verification,
                    ):
                        answer_repair_debug["stop_reason"] = "verification_converged_or_no_repairable_failures"
                        break
                else:
                    repair_round["accepted"] = False
                    repair_round["stop_reason"] = "empty_or_framework_failure_repair_response"
                    repair_rounds.append(repair_round)
                    answer_repair_debug["stop_reason"] = "empty_or_framework_failure_repair_response"
                    break
            except Exception as exc:
                repair_round["accepted"] = False
                repair_round["error"] = f"{type(exc).__name__}: {exc}"
                repair_rounds.append(repair_round)
                answer_repair_debug["stop_reason"] = "repair_runtime_error"
                break
        answer_repair_debug["repair_rounds"] = repair_rounds
    latency_sec = round(time.perf_counter() - started, 2)
    prediction_record = {
        "id": task.row_id,
        "prediction": prediction,
        "messages": [*task.messages, {"role": "assistant", "content": prediction}],
    }
    trace = build_prediction_trace(
        task=task,
        chunks=chunks,
        retrieved_hits=retrieved_hits,
        hits=hits,
        prompt=prompt,
        prediction=prediction,
        latency_sec=latency_sec,
        dry_run=dry_run,
        run_config={
            "max_tokens_requested": max_tokens,
            "request_timeout_sec": request_timeout,
            "max_retries": max_retries,
            "model_call_attempts": model_call_attempts,
            "retry_backoff_base": retry_backoff_base,
            "retry_backoff_max": retry_backoff_max,
            "agent_attempts_debug": agent_attempts_debug,
            "max_iterations": max_iterations,
            "top_k": top_k,
            "neighbor_chunks": neighbor_chunks,
            "retrieved_hit_count": len(retrieved_hits),
            "evidence_hit_count": len(hits),
            "retrieved_hit_ids": [hit.chunk.chunk_id for hit in retrieved_hits],
            "evidence_hit_ids": [hit.chunk.chunk_id for hit in hits],
            "task_route": asdict(task_route),
            "workflow_decision": {
                "workflow": structured_state.workflow,
                "context_type": structured_state.context_type,
                "task_operator": structured_state.task_operator,
                "parser_confidence": structured_state.parser_confidence,
                "trigger_signals": structured_state.trigger_signals,
            },
            "structured_state": structured_state.to_dict(),
            "task_constraints": task_constraints.to_dict(),
            "constraint_verification": constraint_verification,
            "hard_verification": hard_verification,
            "thinking_mode": thinking_mode,
            "thinking_budget": thinking_budget if thinking_mode == "budget" else None,
            "thinking_param_shape": thinking_param_shape,
            "thinking_request": thinking_request,
            "query_planner": query_planner_debug,
            "state_refiner": state_refiner_debug,
            "answer_repair": answer_repair_debug,
            "prompt_no_think": thinking_mode == "off",
            "deterministic_answer_mode": deterministic_answer_mode,
            "deterministic_answer_available": deterministic_answer is not None,
            "deterministic_answer_used": bool(deterministic_answer and not dry_run),
            "deterministic_answer_kind": deterministic_answer.kind if deterministic_answer else None,
            "deterministic_answer_summary": deterministic_answer.summary if deterministic_answer else None,
            "deterministic_answer_confidence": deterministic_answer.confidence if deterministic_answer else None,
            "deterministic_answer_trigger_signals": deterministic_answer.trigger_signals if deterministic_answer else (),
            "deterministic_answer_parsed_state_summary": (
                deterministic_answer.parsed_state_summary if deterministic_answer else None
            ),
            "deterministic_answer_skipped_or_uncertain_items": (
                deterministic_answer.skipped_or_uncertain_items if deterministic_answer else ()
            ),
            "deterministic_answer_should_fallback_to_llm": (
                deterministic_answer.should_fallback_to_llm if deterministic_answer else None
            ),
            "tool_observations": build_tool_observations(deterministic_answer),
            "llm_role": deterministic_answer.llm_role if deterministic_answer else "final_answer_writer",
            "answer_source": deterministic_answer.answer_source if deterministic_answer else "llm_answer_from_evidence_pack",
            "deterministic_core": deterministic_answer.deterministic_core if deterministic_answer else False,
            "deterministic_answer_tool_name": deterministic_answer.tool_name if deterministic_answer else None,
            "deterministic_answer_is_exact": deterministic_answer.answer_is_exact if deterministic_answer else None,
            "deterministic_answer_llm_review_recommended": (
                deterministic_answer.llm_review_recommended if deterministic_answer else None
            ),
        },
        truncation=truncation,
        llm_observations=llm_observations,
        agent_response_debug=agent_response_debug,
    )
    trace_path = trace_dir / f"task_{task.row_id:04d}.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    if not dry_run and not allow_empty and not prediction:
        raise RuntimeError(f"Empty prediction for task {task.row_id}; inspect trace: {trace_path}")
    if not dry_run and not allow_framework_error and is_framework_failure(prediction):
        raise RuntimeError(f"Framework error prediction for task {task.row_id}; inspect trace: {trace_path}")
    if not dry_run and not allow_truncation and truncation["suspected"]:
        reasons = "; ".join(truncation["reasons"])
        raise RuntimeError(f"Suspected truncated prediction for task {task.row_id}: {reasons}; inspect trace: {trace_path}")
    return {"id": task.row_id, "prediction_record": prediction_record, "latency_sec": latency_sec}


def run_agent_with_retries(
    *,
    prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
    max_iterations: int,
    model_call_attempts: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_observations: list[dict[str, Any]] = []
    attempts_debug: list[dict[str, Any]] = []
    last_error: Exception | None = None
    last_prediction = ""
    last_response_debug: dict[str, Any] = {}

    for attempt in range(1, model_call_attempts + 1):
        attempt_started = time.perf_counter()
        attempt_observations: list[dict[str, Any]] = []
        observation_hook = LlmObservationCollector(attempt_observations)
        try:
            config = build_agent_config(
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                max_retries=max_retries,
                thinking_request=thinking_request,
                max_iterations=max_iterations,
                after_model_hooks=[observation_hook],
            )
            agent = Agent(config=config)
            response = agent.run(message=prompt)
            response_debug = summarize_agent_response(response)
            prediction = str(response[0] if isinstance(response, tuple) else response).strip()
            for observation in attempt_observations:
                observation["attempt"] = attempt
            all_observations.extend(attempt_observations)
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "prediction_len": len(prediction),
                    "empty_prediction": not bool(prediction),
                    "framework_error": is_framework_failure(prediction),
                    "response_summary": response_debug,
                    "observation_count": len(attempt_observations),
                }
            )
            last_prediction = prediction
            last_response_debug = response_debug
            if prediction and not is_framework_failure(prediction):
                return prediction, response_debug, all_observations, attempts_debug
        except Exception as exc:
            last_error = exc
            for observation in attempt_observations:
                observation["attempt"] = attempt
            all_observations.extend(attempt_observations)
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                    "observation_count": len(attempt_observations),
                }
            )

        if attempt < model_call_attempts:
            delay = retry_delay(attempt, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max)
            if delay > 0:
                time.sleep(delay)

    if last_prediction or last_response_debug:
        return last_prediction, last_response_debug, all_observations, attempts_debug
    if last_error is not None:
        raise last_error
    return "", {"type": "no_response"}, all_observations, attempts_debug


def run_state_refiner_with_retries(
    *,
    prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
    model_call_attempts: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
) -> tuple[str, list[dict[str, Any]]]:
    attempts_debug: list[dict[str, Any]] = []
    last_response = ""
    last_error: Exception | None = None

    for attempt in range(1, model_call_attempts + 1):
        attempt_started = time.perf_counter()
        try:
            config = build_state_refiner_agent_config(
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                max_retries=max_retries,
                thinking_request=thinking_request,
            )
            agent = Agent(config=config)
            response = agent.run(message=prompt)
            raw = str(response[0] if isinstance(response, tuple) else response).strip()
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "response_len": len(raw),
                    "empty_response": not bool(raw),
                    "framework_error": is_framework_failure(raw),
                    "response_summary": summarize_agent_response(response),
                }
            )
            last_response = raw
            if raw and not is_framework_failure(raw):
                return raw, attempts_debug
        except Exception as exc:
            last_error = exc
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if attempt < model_call_attempts:
            delay = retry_delay(attempt, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max)
            if delay > 0:
                time.sleep(delay)
    if last_response:
        return last_response, attempts_debug
    if last_error is not None:
        raise last_error
    return "", attempts_debug


def run_query_planner_with_retries(
    *,
    prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
    model_call_attempts: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
) -> tuple[str, list[dict[str, Any]]]:
    attempts_debug: list[dict[str, Any]] = []
    last_response = ""
    last_error: Exception | None = None

    for attempt in range(1, model_call_attempts + 1):
        attempt_started = time.perf_counter()
        try:
            config = build_query_planner_agent_config(
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                max_retries=max_retries,
                thinking_request=thinking_request,
            )
            agent = Agent(config=config)
            response = agent.run(message=prompt)
            raw = str(response[0] if isinstance(response, tuple) else response).strip()
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "response_len": len(raw),
                    "empty_response": not bool(raw),
                    "framework_error": is_framework_failure(raw),
                    "response_summary": summarize_agent_response(response),
                }
            )
            last_response = raw
            if raw and not is_framework_failure(raw):
                return raw, attempts_debug
        except Exception as exc:
            last_error = exc
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if attempt < model_call_attempts:
            delay = retry_delay(attempt, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max)
            if delay > 0:
                time.sleep(delay)
    if last_response:
        return last_response, attempts_debug
    if last_error is not None:
        raise last_error
    return "", attempts_debug


def should_repair_answer(
    *,
    task: ClbenchTask,
    structured_state: Any,
    truncation: dict[str, Any],
    hard_verification: dict[str, Any],
) -> bool:
    if (
        getattr(structured_state, "workflow", "") == "multi_doc_matrix_workflow"
        and is_exact_list_task(task.task.lower())
    ):
        return False
    if truncation.get("suspected"):
        return True
    checks = hard_verification.get("checks", [])
    repairable_failures = {
        "required_count_met",
        "valid_json_when_requested",
        "quotes_present",
        "quotes_verbatim_in_context_or_state",
        "commenter_registry_covered",
        "top_speakers_covered",
        "sentiment_evidence_not_nei",
        "recurring_items_covered",
        "planned_coverage_covered",
        "not_suspected_truncated",
    }
    for check in checks:
        if check.get("name") == "top_speakers_covered" and is_soft_activity_sentiment_task(task, structured_state):
            continue
        if not check.get("passed") and check.get("name") in repairable_failures:
            return True
    return False


def is_soft_activity_sentiment_task(task: ClbenchTask, structured_state: Any) -> bool:
    task_text = task.task.lower()
    return bool(structured_state.tables.get("speaker_sentiment_evidence")) and any(
        marker in task_text for marker in ("sentiment", "positive", "negative", "rating")
    )


def failed_verification_check_names(hard_verification: dict[str, Any]) -> list[str]:
    return [
        str(check.get("name"))
        for check in hard_verification.get("checks", [])
        if not check.get("passed")
    ]


def build_answer_repair_prompt(
    *,
    task: ClbenchTask,
    original_answer: str,
    structured_state: Any,
    structured_state_text: str,
    truncation: dict[str, Any],
    hard_verification: dict[str, Any],
) -> str:
    soft_activity_sentiment = is_soft_activity_sentiment_task(task, structured_state)
    failed_checks = [
        check
        for check in hard_verification.get("checks", [])
        if not check.get("passed")
        and not (soft_activity_sentiment and check.get("name") == "top_speakers_covered")
    ]
    top_speaker_rule = (
        "- If the verifier reports `top_speakers_covered`, replace the table's user rows with the exact `expected_top_speakers` list and message counts from `speaker_activity`; do not keep lower-ranked speakers as substitutes."
        if not soft_activity_sentiment
        else "- For this activity/sentiment task, `top_speakers_covered` is treated as a soft ranking warning; do not change the table's user set just to match raw `speaker_activity` if the current users are supported by profitability-relevant evidence."
    )
    return f"""You are a CL-bench Life final-answer repair agent.

Repair the answer below. Return only the repaired final user-facing answer.

Rules:
- Preserve facts that are supported by the workflow state or original answer.
- Do not invent new facts.
- Fix malformed markdown tables. Every table row must have the same number of columns as the header.
- For registry/list/table tasks, use one complete table with one row per requested entity/item. Do not add duplicate rows for repeated appearances. Do not use "see above" as a value.
- If quotes are required, use short exact quotes already present in the workflow state or original answer.
{top_speaker_rule}
- If `speaker_sentiment_evidence` is present, use it as candidate evidence for tied/borderline user selection and to replace `NEI` ratings when profitability evidence exists; do not treat its row order as a fixed answer.
- If the verifier reports `sentiment_evidence_not_nei`, keep the same user set unless another check requires otherwise, and replace those `NEI` cells with 1-5 ratings grounded in `speaker_sentiment_evidence` or raw quotes.
- If the verifier says planned coverage is missing, add the missing required names/items when supported by the workflow state.

FINAL USER TASK:
{task.task}

TRUNCATION OR FORMAT SIGNALS:
{json.dumps(truncation, ensure_ascii=False, indent=2)}

FAILED VERIFIER CHECKS:
{json.dumps(failed_checks, ensure_ascii=False, indent=2)}

WORKFLOW STATE:
{structured_state_text}

ORIGINAL ANSWER:
{original_answer}
"""


def run_answer_repair_with_retries(
    *,
    prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
    model_call_attempts: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
) -> tuple[str, list[dict[str, Any]]]:
    attempts_debug: list[dict[str, Any]] = []
    last_response = ""
    last_error: Exception | None = None

    for attempt in range(1, model_call_attempts + 1):
        attempt_started = time.perf_counter()
        try:
            config = build_answer_repair_agent_config(
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                max_retries=max_retries,
                thinking_request=thinking_request,
            )
            agent = Agent(config=config)
            response = agent.run(message=prompt)
            raw = str(response[0] if isinstance(response, tuple) else response).strip()
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "response_len": len(raw),
                    "empty_response": not bool(raw),
                    "framework_error": is_framework_failure(raw),
                    "response_summary": summarize_agent_response(response),
                }
            )
            last_response = raw
            if raw and not is_framework_failure(raw):
                return raw, attempts_debug
        except Exception as exc:
            last_error = exc
            attempts_debug.append(
                {
                    "attempt": attempt,
                    "latency_sec": round(time.perf_counter() - attempt_started, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if attempt < model_call_attempts:
            delay = retry_delay(attempt, retry_backoff_base=retry_backoff_base, retry_backoff_max=retry_backoff_max)
            if delay > 0:
                time.sleep(delay)
    if last_response:
        return last_response, attempts_debug
    if last_error is not None:
        raise last_error
    return "", attempts_debug


def build_answer_repair_agent_config(
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
) -> AgentConfig:
    llm_config_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": os.getenv("LLM_API_KEY"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_type": "openai_chat_completion",
        "stream": False,
        "timeout": request_timeout,
        "max_retries": max_retries,
    }
    if thinking_request:
        llm_config_kwargs["extra_body"] = thinking_request
    return AgentConfig(
        name="clbench_life_answer_repair",
        max_context_tokens=120000,
        max_iterations=3,
        system_prompt=(
            "You repair malformed or incomplete CL-bench Life answers. "
            "You return only the corrected final answer, never analysis or JSON."
        ),
        system_prompt_type="string",
        tool_call_mode="xml",
        llm_config=LLMConfig(**llm_config_kwargs),
        tools=[],
        skills=[],
    )


def build_query_planner_agent_config(
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
) -> AgentConfig:
    llm_config_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": os.getenv("LLM_API_KEY"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_type": "openai_chat_completion",
        "stream": False,
        "timeout": request_timeout,
        "max_retries": max_retries,
    }
    if thinking_request:
        llm_config_kwargs["extra_body"] = thinking_request
    return AgentConfig(
        name="clbench_life_query_planner",
        max_context_tokens=120000,
        max_iterations=3,
        system_prompt=(
            "You are a strict JSON-only retrieval query planner for a CL-bench Life harness. "
            "You never answer the user directly. You only produce the requested JSON query plan."
        ),
        system_prompt_type="string",
        tool_call_mode="xml",
        llm_config=LLMConfig(**llm_config_kwargs),
        tools=[],
        skills=[],
    )


def build_state_refiner_agent_config(
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_request: dict[str, Any],
) -> AgentConfig:
    llm_config_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": os.getenv("LLM_API_KEY"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_type": "openai_chat_completion",
        "stream": False,
        "timeout": request_timeout,
        "max_retries": max_retries,
    }
    if thinking_request:
        llm_config_kwargs["extra_body"] = thinking_request
    return AgentConfig(
        name="clbench_life_state_refiner",
        max_context_tokens=120000,
        max_iterations=3,
        system_prompt=(
            "You are a strict JSON-only state refiner for a CL-bench Life harness. "
            "You never answer the user directly. You only produce the requested JSON contract."
        ),
        system_prompt_type="string",
        tool_call_mode="xml",
        llm_config=LLMConfig(**llm_config_kwargs),
        tools=[],
        skills=[],
    )


def retry_delay(attempt: int, *, retry_backoff_base: float, retry_backoff_max: float) -> float:
    if retry_backoff_base <= 0:
        return 0.0
    return min(retry_backoff_max, retry_backoff_base * (2 ** max(0, attempt - 1)))


def is_framework_failure(prediction: str) -> bool:
    return any(
        marker in prediction
        for marker in (
            "[Error: Maximum iteration limit reached.]",
            "[Note: Maximum iteration limit reached]",
            "Error in agent execution:",
        )
    )


def detect_suspected_truncation(
    *,
    prediction: str,
    llm_observations: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    for index, observation in enumerate(llm_observations, start=1):
        usage = observation.get("usage")
        if not isinstance(usage, dict):
            continue
        completion_tokens = first_int_value(usage, ("completion_tokens", "output_tokens"))
        if completion_tokens is not None and completion_tokens >= max_tokens:
            reasons.append(f"llm_observation_{index}.completion_tokens={completion_tokens} reached max_tokens={max_tokens}")

    stripped = prediction.rstrip()
    if stripped.count("```") % 2 == 1:
        reasons.append("prediction has an unclosed fenced code block")

    non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1]
        if last_line.startswith("|"):
            table_block: list[str] = []
            for line in reversed(non_empty_lines):
                if not line.startswith("|"):
                    break
                table_block.append(line)
            table_block.reverse()
            separator = next((line for line in table_block if re.fullmatch(r"\|[\s:|\-]+", line)), "")
            expected_pipes = separator.count("|") if separator else table_block[0].count("|")
            if not last_line.endswith("|") or last_line.count("|") < expected_pipes:
                reasons.append(
                    f"last markdown table row has {last_line.count('|')} pipe chars, expected about {expected_pipes}"
                )

    return {"suspected": bool(reasons), "reasons": reasons}


def first_int_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int):
            return value
    return None


def expand_hits_with_neighbors(
    chunks: list[Chunk],
    hits: list[RetrievalHit],
    *,
    neighbor_chunks: int,
) -> list[RetrievalHit]:
    if neighbor_chunks <= 0 or not hits:
        return hits

    index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    direct_hit_by_id = {hit.chunk.chunk_id: hit for hit in hits}
    ordered_ids: list[str] = []
    relation_by_id: dict[str, str] = {}
    seen: set[str] = set()

    for hit in hits:
        center = index_by_id.get(hit.chunk.chunk_id)
        if center is None:
            continue
        start = max(0, center - neighbor_chunks)
        end = min(len(chunks), center + neighbor_chunks + 1)
        for index in range(start, end):
            chunk = chunks[index]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            ordered_ids.append(chunk.chunk_id)
            if chunk.chunk_id != hit.chunk.chunk_id:
                relation_by_id[chunk.chunk_id] = f"neighbor_of:{hit.chunk.chunk_id}"

    expanded: list[RetrievalHit] = []
    for chunk_id in ordered_ids:
        direct_hit = direct_hit_by_id.get(chunk_id)
        if direct_hit is not None:
            expanded.append(direct_hit)
            continue
        chunk = chunks[index_by_id[chunk_id]]
        expanded.append(
            RetrievalHit(
                chunk=chunk,
                score=0.0,
                matched_terms=(relation_by_id.get(chunk_id, "neighbor_context"),),
            )
        )
    return expanded


def build_prompt(
    *,
    task: ClbenchTask,
    hits: list[RetrievalHit],
    task_route: Any,
    task_constraints: Any,
    deterministic_findings_text: str,
    structured_state_text: str,
    evidence_char_budget: int,
) -> str:
    evidence = render_evidence(hits, char_budget=evidence_char_budget)
    conversation_note = render_conversation_note(task)
    route_note = render_route_note(task_route)
    answer_planning = render_answer_planning_instructions(deterministic_findings_text)
    return f"""You are solving one CL-bench Life task.

{conversation_note}

TASK ROUTE:
{route_note}

WORKFLOW STRUCTURED STATE:
{structured_state_text}

TASK CONSTRAINTS:
{render_constraints(task_constraints)}

FINAL USER TASK:
{task.task}

DETERMINISTIC FINDINGS:
{deterministic_findings_text}

RAW EVIDENCE PACK:
{evidence}

Answer the final user task using only the workflow structured state, deterministic findings, and raw evidence pack. Treat computed values and deterministic findings as more reliable than model-estimated counts or arithmetic. Use raw evidence to verify or supplement structured state, not to invent unsupported facts. If the answer needs quotes, quote exactly from the evidence pack or evidence_table. If you need to make an inference, keep it tightly tied to the evidence.
Use the task constraints and any llm_refined_* or llm_query_* workflow tables as a coverage checklist before writing the final answer.
Do not explain the workflow tables themselves. Produce only the user-facing answer. Keep headings and requested labels exact when the task asks for a table, calendar, list, or named sections.
For registry/list/table tasks, prefer one complete table with one row per requested entity/item. Do not add duplicate rows for repeated appearances, do not write "see above" as a value, and do not add extra summary tables unless the user explicitly asked for them.
Every markdown table row must have the same number of columns as the header. If the task asks for direct quotes, keep quote cells short and put longer rationale in a short note after the table.
If `llm_refined_coverage` is present, cover every required item unless it conflicts with stronger deterministic evidence; if there is a conflict, state the uncertainty briefly.
If `llm_query_plan` or `llm_query_evidence` is present, use those targeted retrieval results to fill exact names, counts, dates, fields, and direct quotes.
If `commenter_attribution` is present for a thread registry task, use its `answer_post_count` as the final "Posts counted" value, and use `canonical_commenter` plus `counting_decision` for disputed identities; do not use `displayed_post_count` as the final count when it conflicts with `answer_post_count`, and do not recount commenter posts from prose.
If `speaker_sentiment_evidence` is present for an activity/sentiment rating task, treat it as a candidate evidence table, not as a fixed answer. Use `speaker_activity` for message counts, then decide any tied or borderline users by interpreting which speakers have profitability-relevant evidence. Prefer its quotes and `rating_hint`, avoid `NEI` when a listed speaker has profitability evidence, and do not include purely technical speakers unless they really belong under the final task wording.
{answer_planning}
"""


def render_route_note(task_route: Any) -> str:
    route = getattr(task_route, "route", "unknown")
    task_type = getattr(task_route, "task_type", "unknown")
    confidence = getattr(task_route, "confidence", "unknown")
    signals = getattr(task_route, "signals", ())
    signal_text = ", ".join(signals) if signals else "none"
    return (
        f"route={route}; task_type={task_type}; confidence={confidence}; signals={signal_text}. "
        "Use this route as a processing hint, not as an answer."
    )


def render_conversation_note(task: ClbenchTask) -> str:
    if task.is_multiturn:
        prior_assistant_count = sum(1 for message in task.messages[:-1] if message["role"] == "assistant")
        return (
            "This is a multi-turn item. Earlier turns are included only through retrieved evidence. "
            f"There are {task.turns} total turns and {prior_assistant_count} prior assistant response(s). "
            "Answer only the final user task below."
        )
    return "This is a single-turn item. The context before the task delimiter was indexed and retrieved."


def render_answer_planning_instructions(deterministic_findings_text: str) -> str:
    if "[recommendation_evidence_matrix]" not in deterministic_findings_text:
        return ""
    return """

For this recommendation/comparison task, use the recommendation evidence matrix as an answer coverage checklist. In the final answer, explicitly cover:
- the primary recommendation;
- free or built-in tools/settings if present;
- if the user wants to focus on or isolate something, explain how available free settings/tools can serve as the primary no-cost focusing method, including any cautions from evidence;
- low-cost hardware or budget cues if present;
- source formats or media versions if present;
- mono/stereo or other trade-offs if present, comparing the alternatives even when one is recommended;
- why headphones or the recommended equipment reveal details that cheap speakers, stock earbuds, or phone/laptop speakers may miss if that contrast appears in evidence;
- why the recommendation fits the user's stated goal.
"""


def render_evidence(hits: list[RetrievalHit], *, char_budget: int) -> str:
    if not hits:
        return "[NO RETRIEVED EVIDENCE]"

    parts: list[str] = []
    used_chars = 0
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = (
            f"\n[CHUNK {rank} | id={chunk.chunk_id} | lines={chunk.line_span} | "
            f"score={hit.score} | source={chunk.source_label or 'unknown'}]\n"
        )
        meta = ""
        if chunk.speakers:
            meta += f"speakers={', '.join(chunk.speakers)}\n"
        if chunk.timestamps:
            meta += f"timestamps={', '.join(chunk.timestamps)}\n"
        body = chunk.text.strip()
        item = header + meta + body + "\n"
        remaining = char_budget - used_chars
        if remaining <= 0:
            break
        if len(item) > remaining:
            item = item[: max(0, remaining - 40)] + "\n[TRUNCATED]\n"
        parts.append(item)
        used_chars += len(item)
    return "\n".join(parts).strip()


def build_prediction_trace(
    *,
    task: ClbenchTask,
    chunks: list[Any],
    retrieved_hits: list[RetrievalHit],
    hits: list[RetrievalHit],
    prompt: str,
    prediction: str,
    latency_sec: float,
    dry_run: bool,
    run_config: dict[str, Any],
    truncation: dict[str, Any],
    llm_observations: list[dict[str, Any]],
    agent_response_debug: dict[str, Any] | None,
) -> dict[str, Any]:
    base = build_trace_record(task=task, chunks=chunks, hits=hits)
    base["model_run"] = {
        "latency_sec": latency_sec,
        "prompt_chars": len(prompt),
        "prediction_chars": len(prediction),
        "prediction_preview": prediction[:1200],
        "top_hit_ids": [hit.chunk.chunk_id for hit in hits],
        "retrieved_hit_ids": [hit.chunk.chunk_id for hit in retrieved_hits],
        "dry_run": dry_run,
        "empty_prediction": not bool(prediction),
        "framework_error": is_framework_failure(prediction),
        "suspected_truncation": truncation["suspected"],
        "truncation_reasons": truncation["reasons"],
        **run_config,
    }
    base["agent_response_debug"] = agent_response_debug
    base["llm_observations"] = llm_observations
    base["prompt_preview"] = prompt[:3000]
    base["prediction"] = prediction
    base["raw_top_hits"] = [
        {
            "rank": index + 1,
            "score": hit.score,
            "matched_terms": hit.matched_terms,
            "chunk": asdict(hit.chunk),
        }
        for index, hit in enumerate(hits)
    ]
    base["raw_retrieved_hits"] = [
        {
            "rank": index + 1,
            "score": hit.score,
            "matched_terms": hit.matched_terms,
            "chunk": asdict(hit.chunk),
        }
        for index, hit in enumerate(retrieved_hits)
    ]
    return base


if __name__ == "__main__":
    main()
