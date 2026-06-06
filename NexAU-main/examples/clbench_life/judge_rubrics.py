from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
NEXAU_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]

for path in (str(NEXAU_ROOT), str(WORKSPACE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from nexau import Agent, AgentConfig, LLMConfig  # noqa: E402

from clbench_life_harness.data import load_tasks_by_ids, read_ids_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate dev rubric judging for CL-bench Life predictions.")
    parser.add_argument("--dev-input", required=True, type=Path, help="Dev JSONL with rubrics.")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Judgement JSONL output path.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional selected original row ids.")
    parser.add_argument("--max-rubrics-per-task", type=int, default=None, help="Limit rubrics for smoke tests.")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel judge requests. Keep <= 5 for Nex-N2.")
    parser.add_argument("--max-tokens", type=int, default=12000, help="Maximum judge completion tokens.")
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum API retries per judge request.")
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
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output/partial judgement file.")
    parser.add_argument("--dry-run", action="store_true", help="Write judge prompts without calling model.")
    args = parser.parse_args()

    if args.concurrency < 1 or args.concurrency > 5:
        raise ValueError("--concurrency must be between 1 and 5")
    if args.max_rubrics_per_task is not None and args.max_rubrics_per_task < 1:
        raise ValueError("--max-rubrics-per-task must be positive")
    if args.thinking_budget < 0:
        raise ValueError("--thinking-budget must be non-negative")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")

    predictions = read_predictions(args.predictions)
    ids = read_ids_file(args.ids_file) if args.ids_file else sorted(predictions)
    tasks = load_tasks_by_ids(args.dev_input, ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(args.output.name + ".partial")
    completed_records: dict[int, dict[str, Any]] = {}
    if args.resume:
        completed_records.update(read_judgement_records(partial_output))
        completed_records.update(read_judgement_records(args.output))
    else:
        for path in (args.output, partial_output):
            if path.exists():
                path.unlink()

    tasks_to_run = [task for task in tasks if task.row_id not in completed_records]
    results: list[dict[str, Any]] = list(completed_records.values())
    judge_config = {
        "model": os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "nex-agi/Nex-N2-Pro"),
        "base_url": os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "https://northgate.xiaobei.top/v1"),
        "max_tokens": args.max_tokens,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "thinking_mode": args.thinking_mode,
        "thinking_budget": args.thinking_budget,
        "thinking_param_shape": args.thinking_param_shape,
    }

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                judge_one_task,
                task,
                predictions,
                args.max_rubrics_per_task,
                args.dry_run,
                judge_config,
            ): task.row_id
            for task in tasks_to_run
        }
        for future in as_completed(futures):
            row_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                append_judgement_record(partial_output, result)
                print(f"judged task {row_id}")
            except Exception as exc:
                print(f"failed judge task {row_id}: {exc}", file=sys.stderr)
                raise

    result_by_id = {int(item["id"]): item for item in results}
    missing_ids = [task.row_id for task in tasks if task.row_id not in result_by_id]
    if missing_ids:
        raise RuntimeError(f"Missing judgement records for ids: {missing_ids}. Re-run with --resume.")
    with args.output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(result_by_id[task.row_id], ensure_ascii=False) + "\n")

    summary = build_run_summary(
        records=[result_by_id[task.row_id] for task in tasks],
        args=args,
        judge_config=judge_config,
    )
    summary_path = args.output.with_name(args.output.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_predictions(path: Path) -> dict[int, str]:
    predictions: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions[int(row["id"])] = str(row["prediction"])
    return predictions


def read_judgement_records(path: Path) -> dict[int, dict[str, Any]]:
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


def append_judgement_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def judge_one_task(
    task: Any,
    predictions: dict[int, str],
    max_rubrics_per_task: int | None,
    dry_run: bool,
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    prediction = predictions.get(task.row_id)
    if prediction is None:
        raise ValueError(f"missing prediction for row id {task.row_id}")
    rubrics = task.rubrics[:max_rubrics_per_task] if max_rubrics_per_task else task.rubrics
    prompt = build_judge_prompt(task_id=task.row_id, final_task=task.task, prediction=prediction, rubrics=rubrics)
    started = time.perf_counter()
    if dry_run:
        judgement = {
            "dry_run": True,
            "prompt_preview": prompt[:5000],
            "rubric_count": len(rubrics),
            "note": "Model call skipped. This is not an evaluation result.",
        }
    else:
        config = build_judge_config(
            max_tokens=int(judge_config["max_tokens"]),
            request_timeout=float(judge_config["request_timeout"]),
            max_retries=int(judge_config["max_retries"]),
            thinking_mode=str(judge_config["thinking_mode"]),
            thinking_budget=int(judge_config["thinking_budget"]),
            thinking_param_shape=str(judge_config["thinking_param_shape"]),
        )
        agent = Agent(config=config)
        response = agent.run(message=prompt)
        judgement = parse_judge_response(str(response[0] if isinstance(response, tuple) else response))
    normalized = normalize_judgement(judgement, rubric_count=len(rubrics), dry_run=dry_run)
    return {
        "id": task.row_id,
        "rubric_count": len(rubrics),
        "latency_sec": round(time.perf_counter() - started, 2),
        "judgement": judgement,
        "normalized": normalized,
    }


def build_judge_config(
    *,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    thinking_mode: str,
    thinking_budget: int,
    thinking_param_shape: str,
) -> AgentConfig:
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable: JUDGE_API_KEY or LLM_API_KEY")
    llm_config_kwargs: dict[str, Any] = {
        "model": os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "nex-agi/Nex-N2-Pro"),
        "base_url": os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "https://northgate.xiaobei.top/v1"),
        "api_key": api_key,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "api_type": "openai_chat_completion",
        "stream": False,
        "timeout": request_timeout,
        "max_retries": max_retries,
    }
    thinking_request = build_thinking_request(thinking_mode, thinking_budget, thinking_param_shape)
    if thinking_request:
        llm_config_kwargs["extra_body"] = thinking_request
    return AgentConfig(
        name="clbench_life_rubric_judge",
        max_context_tokens=100000,
        max_iterations=3,
        system_prompt=(
            "You are an approximate CL-bench Life dev rubric judge. "
            "Evaluate whether the prediction satisfies each rubric using only the final task, prediction, and rubric text. "
            "Return strict JSON only."
        ),
        system_prompt_type="string",
        tool_call_mode="xml",
        llm_config=LLMConfig(**llm_config_kwargs),
        tools=[],
        skills=[],
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


def build_judge_prompt(*, task_id: int, final_task: str, prediction: str, rubrics: list[str]) -> str:
    rubric_lines = "\n".join(f"{index + 1}. {rubric}" for index, rubric in enumerate(rubrics))
    return f"""Judge this CL-bench Life dev prediction against rubrics.

Important limitations:
- You are an approximate local judge, not the official CL-bench judge.
- Be strict: a rubric passes only if the prediction clearly satisfies it.
- If a rubric requires a specific fact and the prediction omits it, mark fail.
- If uncertain, mark fail and explain briefly.

Return JSON with this schema:
{{
  "all_pass": true/false,
  "passed": number,
  "failed": number,
  "rubrics": [
    {{"index": 1, "pass": true/false, "reason": "brief reason"}}
  ],
  "failure_type_hints": ["context_location" | "fact_extraction" | "reasoning" | "calculation" | "format" | "final_validation"]
}}

Task id: {task_id}

FINAL TASK:
{final_task}

PREDICTION:
{prediction}

RUBRICS:
{rubric_lines}
"""


def parse_judge_response(response: str) -> dict[str, Any]:
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
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"parse_error": True, "raw_response": response}


def normalize_judgement(judgement: dict[str, Any], *, rubric_count: int, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "all_pass": False,
            "passed": 0,
            "failed": 0,
            "parse_error": False,
            "dry_run": True,
            "rubric_pass_rate": None,
            "failure_type_hints": [],
        }
    if judgement.get("parse_error"):
        return {
            "all_pass": False,
            "passed": 0,
            "failed": rubric_count,
            "parse_error": True,
            "dry_run": False,
            "rubric_pass_rate": 0.0 if rubric_count else None,
            "failure_type_hints": ["final_validation"],
        }

    rubric_rows = judgement.get("rubrics")
    if not isinstance(rubric_rows, list):
        return {
            "all_pass": False,
            "passed": 0,
            "failed": rubric_count,
            "parse_error": True,
            "dry_run": False,
            "rubric_pass_rate": 0.0 if rubric_count else None,
            "failure_type_hints": ["final_validation"],
        }

    pass_values = [bool(row.get("pass")) for row in rubric_rows if isinstance(row, dict)]
    passed = sum(pass_values)
    observed_count = len(pass_values)
    missing = max(0, rubric_count - observed_count)
    failed = (observed_count - passed) + missing
    hints = judgement.get("failure_type_hints")
    if not isinstance(hints, list):
        hints = []
    normalized_hints = [str(hint) for hint in hints]
    return {
        "all_pass": failed == 0 and observed_count == rubric_count,
        "passed": passed,
        "failed": failed,
        "parse_error": False,
        "dry_run": False,
        "rubric_pass_rate": round(passed / rubric_count, 4) if rubric_count else None,
        "observed_rubrics": observed_count,
        "missing_rubric_judgements": missing,
        "failure_type_hints": normalized_hints,
    }


def build_run_summary(records: list[dict[str, Any]], args: argparse.Namespace, judge_config: dict[str, Any]) -> dict[str, Any]:
    normalized_rows = [record.get("normalized", {}) for record in records]
    completed_rows = [row for row in normalized_rows if not row.get("dry_run")]
    total_tasks = len(records)
    solved = sum(1 for row in completed_rows if row.get("all_pass") is True)
    parse_errors = sum(1 for row in completed_rows if row.get("parse_error"))
    rubric_total = sum(int(record.get("rubric_count", 0)) for record in records if not record.get("normalized", {}).get("dry_run"))
    rubric_passed = sum(int(row.get("passed", 0)) for row in completed_rows)
    failure_types: Counter[str] = Counter()
    for row in completed_rows:
        for hint in row.get("failure_type_hints", []):
            failure_types[str(hint)] += 1
    return {
        "dev_input": str(args.dev_input),
        "predictions": str(args.predictions),
        "output": str(args.output),
        "ids_file": str(args.ids_file) if args.ids_file else None,
        "processed": total_tasks,
        "dry_run": args.dry_run,
        "concurrency": args.concurrency,
        "max_rubrics_per_task": args.max_rubrics_per_task,
        **judge_config,
        "task_pass_rate": round(solved / len(completed_rows), 4) if completed_rows else None,
        "solved_tasks": solved,
        "judged_tasks": len(completed_rows),
        "rubric_pass_rate": round(rubric_passed / rubric_total, 4) if rubric_total else None,
        "passed_rubrics": rubric_passed,
        "total_rubrics": rubric_total,
        "parse_errors": parse_errors,
        "failure_type_counts": dict(failure_types),
        "avg_latency_sec": round(sum(float(record.get("latency_sec", 0)) for record in records) / total_tasks, 2)
        if total_tasks
        else 0,
    }


if __name__ == "__main__":
    main()
