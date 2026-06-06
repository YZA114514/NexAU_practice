from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_DELIMITER = "<|TASK|>"


@dataclass(frozen=True)
class ClbenchTask:
    row_id: int
    messages: list[dict[str, str]]
    context: str
    task: str
    retrieval_text: str
    rubrics: list[str]
    metadata: dict[str, Any]
    is_multiturn: bool

    @property
    def chars(self) -> int:
        return sum(len(message.get("content", "")) for message in self.messages)

    @property
    def turns(self) -> int:
        return len(self.messages)


def read_jsonl(path: Path, *, start: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_id, line in enumerate(handle):
            if row_id < start:
                continue
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_tasks(path: Path, *, start: int = 0, limit: int | None = None) -> list[ClbenchTask]:
    tasks: list[ClbenchTask] = []
    rows = read_jsonl(path, start=start, limit=limit)
    for offset, row in enumerate(rows):
        tasks.append(parse_task(row, row_id=start + offset))
    return tasks


def load_tasks_by_ids(path: Path, row_ids: list[int]) -> list[ClbenchTask]:
    wanted = set(row_ids)
    tasks_by_id: dict[int, ClbenchTask] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_id, line in enumerate(handle):
            if row_id not in wanted:
                continue
            line = line.strip()
            if not line:
                continue
            tasks_by_id[row_id] = parse_task(json.loads(line), row_id=row_id)
            if len(tasks_by_id) == len(wanted):
                break

    missing = sorted(wanted - set(tasks_by_id))
    if missing:
        raise ValueError(f"row ids not found in {path}: {missing}")
    return [tasks_by_id[row_id] for row_id in row_ids]


def read_ids_file(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("ids file JSON must be a list")
        return [int(item) for item in data]
    ids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            ids.extend(int(part.strip()) for part in line.split(",") if part.strip())
        else:
            ids.append(int(line))
    return ids


def parse_task(row: dict[str, Any], *, row_id: int) -> ClbenchTask:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"row {row_id} has no messages")

    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError(f"row {row_id} has a non-object message")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        messages.append({"role": role, "content": content})

    first_user = next((message["content"] for message in messages if message["role"] == "user"), "")
    final_user = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
    context, first_task = split_first_user(first_user)
    is_multiturn = len(messages) > 1
    if is_multiturn and TASK_DELIMITER in final_user:
        repeated_context, final_task = split_first_user(final_user)
        if repeated_context:
            context = repeated_context
        task = final_task.strip()
    elif is_multiturn:
        task = final_user.strip()
    else:
        task = first_task.strip()
    if not task:
        task = final_user.strip()

    retrieval_text = build_retrieval_text(messages, context=context, is_multiturn=is_multiturn)
    rubrics_raw = row.get("rubrics", [])
    rubrics = [str(item) for item in rubrics_raw] if isinstance(rubrics_raw, list) else []
    metadata_raw = row.get("metadata", {})
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    return ClbenchTask(
        row_id=row_id,
        messages=messages,
        context=context,
        task=task,
        retrieval_text=retrieval_text,
        rubrics=rubrics,
        metadata=metadata,
        is_multiturn=is_multiturn,
    )


def split_first_user(content: str) -> tuple[str, str]:
    if TASK_DELIMITER not in content:
        return content.strip(), ""
    context, task = content.split(TASK_DELIMITER, 1)
    return context.strip(), task.strip()


def build_retrieval_text(messages: list[dict[str, str]], *, context: str, is_multiturn: bool) -> str:
    if not is_multiturn:
        return context

    final_user_index = max(index for index, message in enumerate(messages) if message["role"] == "user")
    prior_turns = messages[:final_user_index]
    rendered_turns = []
    for index, message in enumerate(prior_turns):
        role = message["role"].upper()
        content = message["content"]
        if index == 0 and TASK_DELIMITER in content:
            _, first_task = split_first_user(content)
            content = f"{context}\n\n[INITIAL_TASK]\n{first_task}".strip()
        rendered_turns.append(f"[{role}]\n{content}")
    return "\n\n".join(rendered_turns).strip()


def task_snippet(task: str, *, max_chars: int = 240) -> str:
    compact = " ".join(task.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."
