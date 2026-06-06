from __future__ import annotations

from pathlib import Path

from clbench_life_harness.chunking import chunk_text
from clbench_life_harness.constraints import extract_constraints
from clbench_life_harness.data import ClbenchTask, load_tasks_by_ids
from clbench_life_harness.deterministic import (
    build_chess_pgn_draw_analysis,
    build_castle_crashers_favorite_tally_analysis,
    build_deterministic_answer,
    build_workflow_deterministic_answer,
    build_dota_purchase_window_analysis,
    build_runescape_adventurer_log_analysis,
    build_starcraft_terran_opener_analysis,
)
from clbench_life_harness.query_planner import (
    execute_query_plan,
    merge_query_execution_into_state,
    parse_query_plan_response,
    should_plan_queries,
)
from clbench_life_harness.retrieval import retrieve
from clbench_life_harness.state_refiner import (
    merge_refinement_into_state,
    parse_state_refinement_response,
    should_refine_state,
)
from clbench_life_harness.task_router import route_task
from clbench_life_harness.verifier import verify_answer_hard
from clbench_life_harness.workflows import build_structured_state, route_task_operator, select_workflow


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEV_DATASET = WORKSPACE_ROOT / "CL-Bench-dataset" / "CL-bench%20Life.jsonl"


def load_dev_task(row_id: int):
    return load_tasks_by_ids(DEV_DATASET, [row_id])[0]


def test_runescape_router_priority_over_multidoc_markers() -> None:
    task = load_dev_task(9)

    route = route_task(task)

    assert route.route == "quant_log_solver"
    assert route.confidence == "high"
    assert "runescape_adventurer_log_records" in route.signals


def test_runescape_solver_counts_and_contract() -> None:
    task = load_dev_task(9)

    analysis = build_runescape_adventurer_log_analysis(task)
    assert analysis is not None

    candy = analysis["stats"]["CandyWarrior"]
    iron = analysis["stats"]["IronWarrior"]
    assert analysis["accounts"] == ["CandyWarrior", "IronWarrior"]
    assert candy["pets"] == ["Ralph", "Herbert"]
    assert candy["level_up_count"] == 43
    assert len(candy["quests"]) == 10
    assert candy["non_pet_drops"] == ["I found a Tuska's Wrath ability codex"]
    assert iron["pets"] == ["Ralph", "Herbert"]
    assert iron["level_up_count"] == 13
    assert iron["quests"] == []
    assert analysis["total_non_pet_drops"] == 1
    assert analysis["confidence"] >= 0.85
    assert analysis["should_fallback_to_llm"] is False
    assert "runescape_adventurer_log_detected" in analysis["trigger_signals"]
    assert analysis["parsed_state_summary"]["total_level_up_count"] == 56
    assert analysis["skipped_or_uncertain_items"]


def test_runescape_deterministic_answer_metadata() -> None:
    task = load_dev_task(9)

    answer = build_deterministic_answer(task)

    assert answer is not None
    assert answer.kind == "runescape_adventurer_log_stats"
    assert answer.confidence >= 0.85
    assert answer.should_fallback_to_llm is False
    assert answer.tool_name == "runescape_log_parser"
    assert answer.answer_source == "verified_tool_answer_review_recommended"
    assert answer.answer_is_exact is False
    assert answer.llm_review_recommended is True
    assert "target_accounts_detected" in answer.trigger_signals
    assert answer.parsed_state_summary is not None
    assert answer.parsed_state_summary["target_accounts"] == ["CandyWarrior", "IronWarrior"]
    assert "| CandyWarrior | 2 | Ralph, Herbert | 43 | 10 | 1 |" in answer.content
    assert "| IronWarrior | 2 | Ralph, Herbert | 13 | 0 | 0 |" in answer.content


def test_pgn_router_solver_and_answer_contract() -> None:
    task = load_dev_task(11)

    route = route_task(task)
    analysis = build_chess_pgn_draw_analysis(task)
    answer = build_deterministic_answer(task)

    assert route.route == "exact_compute_solver"
    assert "chess_pgn_records" in route.signals
    assert analysis is not None
    assert analysis["confidence"] >= 0.85
    assert analysis["should_fallback_to_llm"] is False
    assert analysis["parsed_state_summary"]["draw_game_ids"] == ["game_id_20", "game_id_40"]
    assert answer is not None
    assert answer.kind == "chess_pgn_draw_comparison"
    assert answer.confidence >= 0.85
    assert answer.tool_name == "pgn_draw_parser"
    assert answer.answer_source == "verified_tool_answer"
    assert answer.answer_is_exact is True
    assert answer.llm_review_recommended is False
    assert answer.skipped_or_uncertain_items == ()
    assert 'Termination "Normal"' in answer.content
    assert "game_id_20" in answer.content
    assert "32. Nd4 Rh3+ 33. Kg1 Rg3+ 34. Kh1 Rh3+ 35. Kg1 Rg3+ 36. Kh1" in answer.content
    assert "game_id_40" in answer.content
    assert "45. Kd3 Kc5 46. Ke3 Kc4 47. Ke2 Kd4 48. Kd2 Kd5 49. Ke3 Kc4" in answer.content


def test_starcraft_replay_tracker_opener_solver() -> None:
    task = load_dev_task(26)

    route = route_task(task)
    analysis = build_starcraft_terran_opener_analysis(task)
    answer = build_deterministic_answer(task)

    assert route.route == "exact_compute_solver"
    assert "starcraft_replay_tracker_records" in route.signals
    assert analysis is not None
    assert analysis["confidence"] >= 0.85
    assert analysis["should_fallback_to_llm"] is False
    assert analysis["medivac_time"] == "3:26"
    assert analysis["parsed_state_summary"]["num_structures_before_medivac"] == 10
    assert [(item["structure"], item["time"]) for item in analysis["structures"]] == [
        ("SupplyDepot", "0:17"),
        ("Refinery", "0:28"),
        ("Barracks", "0:44"),
        ("Factory", "1:34"),
        ("SupplyDepot", "1:38"),
        ("CommandCenter", "2:12"),
        ("Starport", "2:20"),
        ("SupplyDepot", "2:34"),
        ("Refinery", "2:58"),
        ("BarracksReactor", "3:26"),
    ]
    assert answer is not None
    assert answer.kind == "starcraft_terran_opener_timeline"
    assert answer.confidence >= 0.85
    assert answer.tool_name == "starcraft_replay_tracker_parser"
    assert answer.answer_is_exact is True
    assert answer.llm_review_recommended is False
    assert "first_medivac_completion_detected" in answer.trigger_signals
    assert "| 10 | BarracksReactor | 3:26 | 4624 |" in answer.content
    assert "excluded later buildings" in answer.content


def test_dota_purchase_window_tables_solver() -> None:
    task = load_dev_task(27)

    route = route_task(task)
    analysis = build_dota_purchase_window_analysis(task)
    answer = build_deterministic_answer(task)

    assert route.route == "exact_compute_solver"
    assert "dota_purchase_log_records" in route.signals
    assert analysis is not None
    assert analysis["confidence"] >= 0.85
    assert analysis["should_fallback_to_llm"] is False
    assert analysis["window_end"] == 300
    assert analysis["max_item_hero"] == "Pudge"

    radiant = analysis["teams"]["radiant"]
    dire = analysis["teams"]["dire"]
    assert [row["hero"] for row in radiant] == ["Pudge", "Shadow Shaman", "Wraith King", "Riki", "Lina"]
    assert [row["hero"] for row in dire] == [
        "Lion",
        "Techies",
        "Nature's Prophet",
        "Dawnbreaker",
        "Phantom Assassin",
    ]
    by_hero = {row["hero"]: row for row in analysis["rows"]}
    assert by_hero["Pudge"]["role"] == "Mid (2)"
    assert by_hero["Pudge"]["item_count"] == 8
    assert by_hero["Pudge"]["items_display"] == [
        "Boots",
        "Tranquil Boots",
        "Magic Stick",
        "Branches×2",
        "Clarity×2",
        "Magic Wand",
    ]
    assert by_hero["Nature's Prophet"]["role"] == "Mid (2)"
    assert by_hero["Nature's Prophet"]["is_user"] is True
    assert by_hero["Nature's Prophet"]["item_count"] == 3
    assert set(by_hero["Nature's Prophet"]["items_display"]) == {"Boots", "TP Scroll", "Blades of Attack"}
    assert by_hero["Wraith King"]["items_display"] == ["Circlet×2", "Bracer×2", "Boots"]
    assert by_hero["Lion"]["role"] == "Off (3)"
    assert by_hero["Techies"]["role"] == "Safe (1)"

    assert answer is not None
    assert answer.kind == "dota_purchase_window_tables"
    assert answer.confidence >= 0.85
    assert answer.tool_name == "dota_purchase_window_parser"
    assert answer.answer_is_exact is True
    assert answer.llm_review_recommended is False
    assert answer.content.count("| Hero | Role | # items | Gold at 300s | 0-300s purchases |") == 2
    assert "Radiant (Team 0)" in answer.content
    assert "Dire (Team 1)" in answer.content
    assert "Nature's Prophet (me)" in answer.content
    assert "magic_stick" not in answer.content


def test_castle_crashers_favorite_tally_solver() -> None:
    task = load_dev_task(87)

    route = route_task(task)
    analysis = build_castle_crashers_favorite_tally_analysis(task)
    answer = build_deterministic_answer(task)

    assert route.route == "thread_tally_solver"
    assert "community_favorite_tally_records" in route.signals
    assert analysis is not None
    assert analysis["confidence"] >= 0.85
    assert analysis["should_fallback_to_llm"] is False
    assert analysis["top_counts"] == {
        "Green Knight": 4,
        "Red Knight": 3,
        "Grey Knight": 2,
        "Pink Knight": 2,
        "Necromancer": 2,
        "Blacksmith": 2,
        "Blue Knight": 1,
        "Orange Knight": 1,
        "Brute": 1,
        "Industrialist": 1,
    }
    assert analysis["all_counts"] == {
        "Green Knight": 4,
        "Red Knight": 4,
        "Grey Knight": 4,
        "Blue Knight": 3,
        "Pink Knight": 2,
        "Necromancer": 2,
        "Blacksmith": 2,
        "Orange Knight": 1,
        "Brute": 1,
        "Industrialist": 1,
        "Fencer": 1,
    }
    assert analysis["top_supporters"]["Green Knight"] == ["nightreaper", "TopChefSam", "Viktor", "305_fan"]
    assert analysis["top_supporters"]["Red Knight"] == ["Marcus the Player", "Zarek_Hale", "MarcoDVV"]
    assert analysis["top_supporters"]["Grey Knight"] == ["frostflareedition", "WhyAmIBurning"]
    assert analysis["top_supporters"]["Pink Knight"] == ["playerdude42", "Emily"]
    assert analysis["top_supporters"]["Necromancer"] == ["ZenoByteX", "coolflavor"]
    assert analysis["top_supporters"]["Blacksmith"] == ["pixelRiotXP", "RapidPlay99"]
    assert analysis["all_supporters"]["Red Knight"] == [
        "Marcus the Player",
        "playerdude42",
        "Zarek_Hale",
        "MarcoDVV",
    ]
    assert analysis["all_supporters"]["Grey Knight"] == [
        "playerdude42",
        "frostflareedition",
        "WhyAmIBurning",
        "Kaelthorin",
    ]
    assert {item["user"] for item in analysis["skipped_or_uncertain_items"]} == {
        "Nina_on_Pie",
        "SnackBundle",
        "Mateo",
        "Silver Squire",
        "T'kar Deep",
    }

    assert answer is not None
    assert answer.kind == "community_favorite_tally"
    assert answer.tool_name == "community_thread_tally_parser"
    assert answer.answer_source == "verified_tool_answer_review_recommended"
    assert answer.answer_is_exact is False
    assert answer.llm_review_recommended is True
    assert answer.content.count("assumptions:") == 2
    assert "- Green Knight: 4\n" in answer.content
    assert "- Red Knight: 3\n" in answer.content
    assert "- Red Knight: 4\n" in answer.content
    assert "- Grey Knight: 2\n" in answer.content
    assert "Green Knight: (nightreaper, TopChefSam, Viktor, 305_fan)" in answer.content
    assert "Red Knight: (Marcus the Player, Zarek_Hale, MarcoDVV)" in answer.content
    assert "Red Knight: (Marcus, playerdude42, Zarek_Hale, MarcoDVV)" in answer.content
    assert "Grey Knight: (frostflareedition, WhyAmIBurning)" in answer.content
    assert "- Blue Knight: 1" in answer.content
    assert "- Blue Knight: 3" in answer.content
    assert "- Fencer: 1" in answer.content
    assert "Bruno droid" not in answer.content
    assert "Factory guys" not in answer.content
    assert "T'kar Deep" not in answer.content


def test_workflow_selector_maps_context_and_operator_to_reusable_workflows() -> None:
    game_task = load_dev_task(27)
    community_task = load_dev_task(87)

    game_decision = select_workflow(task=game_task, task_route=route_task(game_task))
    community_decision = select_workflow(task=community_task, task_route=route_task(community_task))

    assert game_decision.context_type == "Game Logs"
    assert game_decision.task_operator == "count_or_aggregate"
    assert game_decision.workflow == "structured_log_workflow"
    assert "context:Game Logs" in game_decision.signals

    assert community_decision.context_type == "Community Interactions"
    assert community_decision.task_operator == "count_or_aggregate"
    assert community_decision.workflow == "thread_tally_workflow"
    assert "context:Community Interactions" in community_decision.signals

    community_timestamp_task = load_dev_task(59)
    community_timestamp_decision = select_workflow(
        task=community_timestamp_task,
        task_route=route_task(community_timestamp_task),
    )
    assert community_timestamp_decision.context_type == "Community Interactions"
    assert community_timestamp_decision.workflow == "thread_tally_workflow"

    assert route_task_operator("Who was the group glue in this meeting?") == "social_inference"
    assert route_task_operator("Compare document 1 and document 2.") == "compare_or_diff"


def test_structured_state_exposes_tool_observations_without_replacing_llm_role() -> None:
    task = load_dev_task(87)
    route = route_task(task)
    findings = []
    answer = build_deterministic_answer(task)

    state = build_structured_state(
        task=task,
        task_route=route,
        hits=[],
        deterministic_findings=findings,
        deterministic_answer=answer,
    )

    assert state.workflow == "thread_tally_workflow"
    assert state.context_type == "Community Interactions"
    assert state.candidate_answers
    assert state.candidate_answers[0].source == "community_thread_tally_parser"
    assert state.uncertain_items
    assert state.parser_confidence >= 0.85
    assert "Deterministic candidate answer has semantic boundaries; LLM review may be useful." in state.warnings


def test_workflow_state_builds_tables_for_thread_and_dialogue_contexts() -> None:
    community_task = load_dev_task(87)
    community_chunks = chunk_text(community_task.retrieval_text, row_id=community_task.row_id)
    community_hits = retrieve(community_chunks, community_task.task, top_k=8)
    community_state = build_structured_state(
        task=community_task,
        task_route=route_task(community_task),
        hits=community_hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    assert community_state.workflow == "thread_tally_workflow"
    assert "comments" in community_state.tables
    assert community_state.tables["comments"]

    dialogue_task = load_dev_task(301)
    dialogue_chunks = chunk_text(dialogue_task.retrieval_text, row_id=dialogue_task.row_id)
    dialogue_hits = retrieve(dialogue_chunks, dialogue_task.task, top_k=8)
    dialogue_state = build_structured_state(
        task=dialogue_task,
        task_route=route_task(dialogue_task),
        hits=dialogue_hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    assert dialogue_state.workflow == "dialogue_state_workflow"
    assert "turns" in dialogue_state.tables
    assert isinstance(dialogue_state.entity_profiles, dict)


def test_dialogue_workflow_builds_full_context_speaker_activity() -> None:
    task = load_dev_task(361)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)

    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    activity = {row["speaker"]: row for row in state.tables["speaker_activity"]}
    assert activity["User01"]["message_count"] == 29
    assert activity["User02"]["message_count"] == 10
    assert activity["User16"]["message_count"] == 10
    assert activity["User10"]["message_count"] == 9
    assert activity["User17"]["message_count"] == 8
    assert activity["User53"]["message_count"] == 7
    assert activity["User13"]["message_count"] == 4
    assert activity["User21"]["message_count"] == 4
    sentiment = {row["speaker"]: row for row in state.tables["speaker_sentiment_evidence"]}
    assert sentiment["User01"]["profitability_evidence_count"] >= 3
    assert sentiment["User16"]["rating_hint"] in {1, 2}
    assert sentiment["User13"]["profitability_evidence_count"] >= 1
    assert sentiment["User21"]["profitability_evidence_count"] >= 1
    assert state.parser_confidence >= 0.75


def test_thread_workflow_builds_full_context_commenter_registry() -> None:
    task = load_dev_task(59)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)

    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    registry = {row["commenter"]: row for row in state.tables["commenter_registry"]}
    assert registry["Mark Stevens"]["post_count"] == 1
    assert registry["Mark Stevens"]["first_time"] == "27 July 2025 at 7:22 PM"
    assert registry["gordon b"]["post_count"] == 1
    assert registry["Sarah Jenkins"]["post_count"] == 5
    assert registry["The Numbers Guy..."]["post_count"] == 9
    assert registry["Brian Smith"]["post_count"] == 7
    assert registry["Michael Brown"]["first_time"] == "27 July 2025 at 10:10 PM"
    assert "And yet the Danish government" not in registry
    attribution = {row["displayed_commenter"]: row for row in state.tables["commenter_attribution"]}
    assert attribution["Michael Brown"]["answer_post_count"] == 14
    assert attribution["Michael Brown"]["attributed_post_count"] == 14
    assert attribution["Michael Brown"]["attribution_status"] == "canonical_with_alias"
    assert attribution["thomas oak"]["canonical_commenter"] == "Michael Brown"
    assert attribution["thomas oak"]["answer_post_count"] == 0
    assert attribution["thomas oak"]["attributed_post_count"] == 0
    assert attribution["Brian Smith"]["answer_post_count"] == 7
    assert attribution["Brian Smith"]["attributed_post_count"] == 7
    deterministic_answer = build_workflow_deterministic_answer(task, state)
    assert deterministic_answer is not None
    assert deterministic_answer.tool_name == "thread_commenter_registry_renderer"
    assert "| Michael Brown | 14 |" in deterministic_answer.content
    assert "| thomas oak | 0 |" in deterministic_answer.content
    assert "It was me, Michael Brown made the comment" in deterministic_answer.content
    assert state.tables["chronology_notes"]
    assert state.parser_confidence >= 0.75


def test_structured_log_workflow_builds_recurring_item_cadence_table() -> None:
    task = load_dev_task(339)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)

    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    items = {row["item_key"]: row for row in state.tables["recurring_items"]}
    assert items["cesar_wet_dog_food"]["cadence_hint"] == "high_frequency_candidate_4_weeks"
    assert "36 Count" in ", ".join(items["cesar_wet_dog_food"]["sizes_or_counts"])
    assert items["wellness_core_dry_dog_food"]["cadence_hint"] == "high_frequency_candidate_4_weeks"
    assert items["amazon_basics_baby_shampoo"]["cadence_hint"] == "medium_frequency_candidate_6_to_8_weeks"
    assert all(row["item_key"] != "your_package_was_left_near_the_front_door" for row in state.tables["recurring_items"])
    assert [row["month"] for row in state.tables["planning_calendar"]] == [
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
    ]
    cadence_plan = {
        row["display_name"]: row
        for row in state.tables["subscription_cadence_plan"]
    }
    assert cadence_plan["Cesar Loaf in Sauce Wet Dog Food, 24-count pack"]["frequency"] == "every 4 weeks"
    assert cadence_plan["Cesar Loaf in Sauce Wet Dog Food, 36-count pack"]["delivery_quantity"] == "one 36-count pack"
    assert cadence_plan["Pet 'n Shape Chik 'n Hide Twists, 16-ounce bags"]["delivery_quantity"] == "at least two 16-ounce bags"
    assert cadence_plan["Tide PODS Free & Gentle, 112-count container"]["frequency"] == "every 8 weeks"
    assert cadence_plan["The Ordinary Azelaic Acid Suspension 10%"]["frequency"] == "at least every 12 weeks"
    deterministic_answer = build_workflow_deterministic_answer(task, state)
    assert deterministic_answer is not None
    assert deterministic_answer.tool_name == "subscribe_save_calendar_renderer"
    assert "one 24-count pack of Cesar Loaf in Sauce Wet Dog Food, 24-count pack" in deterministic_answer.content
    assert "one 36-count pack of Cesar Loaf in Sauce Wet Dog Food, 36-count pack" in deterministic_answer.content
    assert "at least two 16-ounce bags of Pet 'n Shape Chik 'n Hide Twists, 16-ounce bags" in deterministic_answer.content
    assert "one 112-count container of Tide PODS Free & Gentle, 112-count container" in deterministic_answer.content
    assert "one 24 pack of Charmin Ultra Soft Toilet Paper, 24 pack" in deterministic_answer.content
    assert "ELEGOO" not in deterministic_answer.content
    assert state.parser_confidence >= 0.7


def test_multidoc_workflow_builds_canonical_section_item_matrix() -> None:
    task = load_dev_task(144)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)

    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    matrix = {
        (row["section"], row.get("subsection"), row["canonical_item"]): row
        for row in state.tables["canonical_item_matrix"]
    }
    assert ("Breakfast", None, "avocado toast runny eggs") in matrix
    assert ("Breakfast", None, "berry crumble honey yogurt") in matrix
    assert ("Dinner", None, "red curry garlic beef stir fry") in matrix
    assert matrix[("Breakfast", None, "avocado toast runny eggs")]["has_partial_ingredients"] is True
    assert state.parser_confidence >= 0.7


def test_state_refiner_parse_and_merge_adds_answer_contract_tables() -> None:
    task = load_dev_task(361)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)
    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    assert should_refine_state(state, deterministic_answer_available=False) is True

    refinement = parse_state_refinement_response(
        """
        {
          "coverage_items": [
            {"kind": "person", "name": "User01", "value": "29 messages", "required": true, "source_table": "speaker_activity"}
          ],
          "answer_outline": ["Create activity-ranked sentiment table."],
          "format_constraints": ["Use one row per required user."],
          "uncertain_items": [],
          "final_checks": ["Top users from speaker_activity are covered."]
        }
        """
    )
    assert refinement.used is True

    merged = merge_refinement_into_state(state, refinement)
    assert "llm_refined_coverage" in merged.tables
    assert merged.tables["llm_refined_coverage"][0]["name"] == "User01"
    assert "llm_state_refiner" in merged.trigger_signals


def test_state_refiner_skips_deterministic_answers() -> None:
    task = load_dev_task(87)
    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=[],
        deterministic_findings=[],
        deterministic_answer=build_deterministic_answer(task),
    )

    assert should_refine_state(state, deterministic_answer_available=True) is False


def test_hard_verifier_records_grounding_and_failure_hints() -> None:
    task = load_dev_task(87)
    answer = build_deterministic_answer(task)
    assert answer is not None
    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=[],
        deterministic_findings=[],
        deterministic_answer=answer,
    )

    result = verify_answer_hard(
        task=task,
        answer=answer.content,
        constraints=extract_constraints(task),
        structured_state=state,
        truncation={"suspected": False, "reasons": []},
    )

    assert result["error_count"] == 0
    assert any(check["name"] == "usernames_grounded" for check in result["checks"])

    failed = verify_answer_hard(
        task=task,
        answer="",
        constraints=extract_constraints(task),
        structured_state=state,
        truncation={"suspected": True, "reasons": ["unit-test"]},
    )
    assert failed["error_count"] >= 1
    assert "final_generation" in failed["failure_type_hints"]


def test_hard_verifier_flags_wrong_top_speaker_rows() -> None:
    task = load_dev_task(361)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)
    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    wrong_answer = "\n".join(
        [
            "| User | Messages Posted |",
            "|---|---:|",
            "| User01 | 29 |",
            "| User02 | 10 |",
            "| User16 | 10 |",
            "| User10 | 9 |",
            "| User17 | 8 |",
            "| User53 | 7 |",
            "| User45 | 5 |",
            "| User13 | 4 |",
            "| User05 | 4 |",
            "| User21 | 4 |",
        ]
    )
    result = verify_answer_hard(
        task=task,
        answer=wrong_answer,
        constraints=extract_constraints(task),
        structured_state=state,
        truncation={"suspected": False, "reasons": []},
    )

    check = next(item for item in result["checks"] if item["name"] == "top_speakers_covered")
    assert check["passed"] is False
    assert "User20" in check["details"]["missing"]
    assert "User34" in check["details"]["missing"]
    assert "User13" in check["details"]["lower_ranked_speakers_present"]


def test_hard_verifier_flags_nei_when_sentiment_evidence_exists() -> None:
    task = load_dev_task(361)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)
    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )

    answer = "| User | Messages Posted | Rating |\n|---|---:|---|\n| User16 | 10 | NEI |\n| User13 | 4 | NEI |"
    result = verify_answer_hard(
        task=task,
        answer=answer,
        constraints=extract_constraints(task),
        structured_state=state,
        truncation={"suspected": False, "reasons": []},
    )

    check = next(item for item in result["checks"] if item["name"] == "sentiment_evidence_not_nei")
    assert check["passed"] is False
    assert "User16" in check["details"]["speakers_with_evidence_given_nei"]
    assert "User13" in check["details"]["speakers_with_evidence_given_nei"]


def test_runescape_parser_failure_does_not_force_deterministic_answer() -> None:
    task = ClbenchTask(
        row_id=999_999,
        messages=[
            {
                "role": "user",
                "content": "Recent events for: MissingPlayer\n<|TASK|>\nCount pets, level ups, quests, and non-pet drops.",
            }
        ],
        context="Recent events for: MissingPlayer",
        task="Count pets, level ups, quests, and non-pet drops.",
        retrieval_text="Recent events for: MissingPlayer",
        rubrics=[],
        metadata={},
        is_multiturn=False,
    )

    assert build_runescape_adventurer_log_analysis(task) is None
    assert build_deterministic_answer(task) is None


def test_query_planner_parses_executes_and_merges_targeted_retrieval() -> None:
    task = load_dev_task(59)
    chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
    hits = retrieve(chunks, task.task, top_k=8)
    state = build_structured_state(
        task=task,
        task_route=route_task(task),
        hits=hits,
        deterministic_findings=[],
        deterministic_answer=None,
    )
    raw_plan = """
    {
      "queries": [
        {
          "purpose": "Find direct evidence for the disputed thomas oak / Michael Brown identity.",
          "query": "thomas oak Michael Brown It was me ID wasn't hijacked",
          "expected_fields": ["misattribution quote", "counting decision"],
          "quote_required": true,
          "priority": 1
        }
      ],
      "coverage_items": [
        {"kind": "person", "name": "Michael Brown", "required": true},
        {"kind": "person", "name": "thomas oak", "required": true}
      ],
      "output_schema": ["Name", "Number of Posts", "Date of First Post", "Notes Regarding Misattributions"]
    }
    """

    plan = parse_query_plan_response(raw_plan)
    execution = execute_query_plan(chunks=chunks, plan=plan, top_k_per_query=3, neighbor_chunks=1, max_total_hits=8)
    merged_state = merge_query_execution_into_state(state, execution)

    assert plan.used is True
    assert execution.hits
    assert "llm_query_plan" in merged_state.tables
    assert "llm_query_evidence" in merged_state.tables
    evidence_text = "\n".join(row["snippet"] for row in merged_state.tables["llm_query_evidence"])
    assert "Michael Brown" in evidence_text
    assert "thomas oak" in evidence_text


def test_query_planner_policy_is_workflow_gated() -> None:
    states = {}
    for row_id in (59, 144, 339, 361):
        task = load_dev_task(row_id)
        chunks = chunk_text(task.retrieval_text, row_id=task.row_id)
        hits = retrieve(chunks, task.task, top_k=8)
        states[row_id] = (
            task,
            build_structured_state(
                task=task,
                task_route=route_task(task),
                hits=hits,
                deterministic_findings=[],
                deterministic_answer=None,
            ),
        )

    assert should_plan_queries(*states[59], deterministic_answer_available=False) is True
    assert should_plan_queries(*states[361], deterministic_answer_available=False) is False
    assert should_plan_queries(*states[144], deterministic_answer_available=False) is False
    assert should_plan_queries(*states[339], deterministic_answer_available=False) is False
