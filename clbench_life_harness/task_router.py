from __future__ import annotations

import re
from dataclasses import dataclass

from .data import ClbenchTask
from .trace import classify_task


@dataclass(frozen=True)
class TaskRoute:
    route: str
    task_type: str
    confidence: str
    signals: tuple[str, ...]


def route_task(task: ClbenchTask) -> TaskRoute:
    task_type = classify_task(task.task)
    probe = build_probe(task)
    signals: list[str] = []

    if contains_any(probe, ("runescape.com/m=adventurers-log", "adventurer's log", "recent events for:")):
        signals.append("runescape_adventurer_log_records")
        return TaskRoute("quant_log_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ('[gameid "', '[termination "', '[event "')) and "1/2-1/2" in probe:
        signals.append("chess_pgn_records")
        return TaskRoute("exact_compute_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ("nnet.replay.tracker", "m_unittype", "gameloop")):
        signals.append("starcraft_replay_tracker_records")
        return TaskRoute("exact_compute_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ("hkquantitytypeidentifierstepcount", "<record")):
        signals.append("xml_health_or_quantity_records")
        return TaskRoute("quant_log_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ('"purchase_log"', '"hero_id"', '"lane_role"', '"players"')):
        signals.append("dota_purchase_log_records")
        return TaskRoute("exact_compute_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ("favorite castle crashers knight", "favorite playable characters")) and contains_any(
        probe, ("top choices", "all choices", "supporters")
    ):
        signals.append("community_favorite_tally_records")
        return TaskRoute("thread_tally_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ('"artistname"', '"msplayed"', '"trackname"', "spotify", "last.fm")):
        signals.append("music_or_listening_history_records")
        return TaskRoute("quant_log_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ("watch history", "youtube watch", "netflix", "viewing history")):
        signals.append("watch_or_viewing_history_records")
        return TaskRoute("quant_log_solver", task_type, "medium", tuple(signals))

    if contains_any(probe, ("match_id", "game_date", "champion")) and contains_any(
        probe, ("kills", "deaths", "assists", "duration_s", "vision_score")
    ):
        signals.append("game_match_json_records")
        return TaskRoute("quant_log_solver", task_type, "high", tuple(signals))

    if contains_any(probe, ("winnings", "big blind", "small blind", "showdown")) and contains_any(
        probe, ("actions", "d db", "players")
    ):
        signals.append("poker_hand_history_with_actions_and_winnings")
        return TaskRoute("exact_compute_solver", task_type, "high", tuple(signals))

    if contains_any(
        probe,
        (
            "beginning balance",
            "statement period",
            "bank statement",
            "checking account",
            "credit card",
            "transaction log",
        ),
    ):
        signals.append("financial_statement_or_transaction_log")
        return TaskRoute("financial_solver", task_type, "high", tuple(signals))

    if has_multi_document_markers(probe):
        signals.append("multi_document_markers")
        return TaskRoute("multi_doc_solver", task_type, "medium", tuple(signals))

    if contains_any(probe, ("r/", "u/", "reddit", "comment tree", "more replies")):
        signals.append("thread_or_reddit_markers")
        return TaskRoute("thread_tree_solver", task_type, "medium", tuple(signals))

    if looks_like_dialogue(task.retrieval_text):
        signals.append("speaker_turn_or_timestamp_pattern")
        return TaskRoute("dialogue_social_solver", task_type, "medium", tuple(signals))

    if task_type in {"count_or_calculation", "timeline_or_status"}:
        signals.append(f"task_type:{task_type}")
        return TaskRoute("general_structured_solver", task_type, "low", tuple(signals))

    signals.append("fallback_general_evidence")
    return TaskRoute("general_evidence_solver", task_type, "low", tuple(signals))


def build_probe(task: ClbenchTask) -> str:
    context = task.retrieval_text
    head = context[:80_000]
    tail = context[-20_000:] if len(context) > 20_000 else ""
    return f"{task.task}\n{head}\n{tail}".lower()


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def has_multi_document_markers(text: str) -> bool:
    if any(marker in text for marker in ("document one", "document two", "document 1", "document 2", "draft one", "draft two")):
        return True
    return bool(re.search(r"(?i)<doc(?:ument)?\s*\d+\s*>", text))


def looks_like_dialogue(text: str) -> bool:
    head = text[:80_000]
    if re.search(r"(?m)^[A-Z][A-Za-z ._-]{0,60}\s+[—-]\s+\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}", head):
        return True
    if re.search(r"(?m)^\[[A-Z]+\]\s*$", head):
        return True
    return bool(re.search(r"(?m)^[A-Z][A-Za-z ._-]{0,60}:\s+\S", head))
