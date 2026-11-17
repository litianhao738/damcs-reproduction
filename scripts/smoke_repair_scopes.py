"""Verify repair witnesses with real Crafter execution, without an LLM."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import crafter
from crafter import constants
from memory_system.agent import Agent
from memory_system.coordination import Plan, diagnose
from memory_system.llm_api import ResponseEvent
from memory_system.memory_system import Experience
from memory_system.repair_scopes import (
    build_repair_targets, generate_scope_candidates, verify_joint_repair,
)
from utils import AgentActionProcessor, AgentThinkingProcessor


def plan(agent_id, operation="noop", material="not_applicable",
         item="not_applicable", recipient=-1):
    return Plan(agent_id, operation, material, item, recipient)


class RepairScopeChecks(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory(prefix="mcrafter-scopes-")
        self.addCleanup(self.output.cleanup)
        self.make_env()

    def make_env(self, count=3):
        self.env = crafter.Env(n_players=count, length=4, seed=4)
        with patch("crafter.env.worldgen.generate_world", side_effect=lambda world, player:
                   world._mat_map.fill(world._mat_ids["grass"])):
            self.env.reset()
        self.world = self.env._world
        self.players = self.env._players
        for agent_id, player in enumerate(self.players):
            self.world.move(player, (8 + agent_id * 4, 8))

    def table(self, agent_id):
        x, y = self.players[agent_id].pos
        self.world[(x + 1, y + 1)] = "table"

    def facing(self, agent_id, material):
        player = self.players[agent_id]
        target = tuple(player.pos + player.facing)
        self.world[target] = material

    def observations(self):
        semantic = self.env._sem_view()
        return {agent_id: {
            "inventory": player.inventory.copy(), "semantic": semantic.copy(),
            "achievements": player.achievements.copy(),
            "player_pos": player.pos.copy(), "player_facing": player.facing,
            "sleeping": player.sleeping, "dead": False,
        } for agent_id, player in enumerate(self.players)}

    def agents(self, plans):
        agents = []
        for current in plans:
            with patch("memory_system.agent.TextModel"):
                agent = Agent(current.agent_id, results_dir=self.output.name)
            agent.update_current_skill(current.operation, current.material, current.item, current.recipient)
            agent.step = 0
            agent.context = f"Context for agent {agent.id}."
            agent.experience = Experience(path=str(agent.agent_dir))
            agents.append(agent)
        return agents

    def execute(self, plans, edits=()):
        replacements = {edit.agent_id: edit for edit in edits}
        actual = [replacements.get(current.agent_id, current) for current in plans]
        obs = self.observations()
        agents = self.agents(actual)
        # Share the existing execution guard, then let the real executor,
        # exchange_item and Env.step independently determine action effects.
        held = {check.agent_id for check in diagnose(actual, obs)
                if check.immediate and check.status == "violated"}
        for agent in agents:
            agent.info = obs[agent.id]
            agent.coordination_blocked = agent.id in held
        with patch.object(Agent, "create_experience"), redirect_stdout(io.StringIO()):
            AgentActionProcessor.process_all_agent_actions(agents, self.env, len(agents))
            actions = AgentActionProcessor.collect_agent_actions(agents, len(agents))
            return self.env.step(actions)

    def target(self, plans):
        obs = self.observations()
        return build_repair_targets(plans, obs, diagnose(plans, obs))

    def response(self, current):
        data = json.loads((PROJECT_ROOT / "scripts/fixtures/agentscope_plan.json").read_text(encoding="utf-8"))
        action = data["action"]
        action["next_action"] = action["final_next_action"] = current.operation
        action["final_target_material_to_collect"] = current.material
        action["final_target_material_to_share"] = current.item
        action["final_target_agent_id"] = current.recipient
        return ResponseEvent.model_validate(data)

    def test_diagnosis_exposes_structured_stock_facts(self):
        self.table(0)
        checks = diagnose([plan(0, "make_wood_pickaxe"), plan(1), plan(2)], self.observations())
        check = next(check for check in checks if check.rule == "inventory")
        self.assertEqual((check.item, check.required, check.available), ("wood", 1, 0))

    def test_only_provider_changes_and_real_crafting_resumes(self):
        self.table(0)
        self.table(1)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "make_wood_pickaxe"), plan(2)]
        observations = self.observations()
        report = generate_scope_candidates(plans, observations)
        selected = report["selected"]
        self.assertEqual(selected["edited_agents"], [1])
        self.assertEqual(set(selected["evidence_agents"]), {0, 1})
        self.assertEqual(selected["covered_targets"], ("0:make:wood_pickaxe",))
        self.assertEqual(observations[1]["inventory"]["wood"], 1)
        np.testing.assert_array_equal(observations[0]["semantic"], self.env._sem_view())
        self.assertEqual(self.players[0].inventory["wood_pickaxe"], 0)
        self.execute(plans, [Plan(**edit) for edit in selected["edits"]])
        self.assertEqual(self.players[0].achievements["make_wood_pickaxe"], 1)
        self.assertEqual(self.players[1].achievements["make_wood_pickaxe"], 0)

    def test_joint_tool_transfer_and_collection_need_two_editable_agents(self):
        self.players[0].inventory["wood_pickaxe"] = 1
        self.facing(1, "stone")
        plans = [plan(0), plan(1, "Navigator", "stone"), plan(2)]
        one = generate_scope_candidates(plans, self.observations(), max_scope_agents=1)
        self.assertIsNone(one["selected"])
        self.assertTrue(all(scope["status"] == "infeasible" for scope in one["scope_results"]))
        two = generate_scope_candidates(plans, self.observations())
        self.assertEqual(two["selected"]["edited_agents"], [0, 1])
        self.execute(plans, [Plan(**edit) for edit in two["selected"]["edits"]])
        self.assertEqual(self.players[1].achievements["collect_stone"], 1)

    def test_single_unit_cannot_cover_two_crafting_targets(self):
        self.table(0)
        self.table(1)
        self.players[2].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "make_wood_pickaxe"), plan(2)]
        report = generate_scope_candidates(plans, self.observations())
        self.assertEqual(len(report["targets"]), 2)
        self.assertTrue(all(len(candidate["covered_targets"]) <= 1 for candidate in report["candidates"]))
        self.execute(plans, [Plan(**edit) for edit in report["selected"]["edits"]])
        self.assertEqual(sum(player.achievements["make_wood_pickaxe"] for player in self.players), 1)

    def test_known_zero_supply_is_infeasible_in_the_declared_step(self):
        self.table(0)
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        report = generate_scope_candidates(plans, self.observations())
        self.assertIsNone(report["selected"])
        self.assertTrue(all(scope["status"] == "infeasible" for scope in report["scope_results"]))

    def test_unsubmitted_inventory_keeps_possible_transfers_unknown(self):
        self.table(0)
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        obs = self.observations()
        obs[1]["inventory"] = None
        report = generate_scope_candidates(plans, obs)
        self.assertIsNone(report["selected"])
        scope = next(scope for scope in report["scope_results"] if scope["agents"] == (1,))
        self.assertEqual(scope["status"], "unknown")

    def test_missing_facility_observation_is_unknown(self):
        self.table(0)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        obs = self.observations()
        obs[0]["semantic"] = None
        report = generate_scope_candidates(plans, obs)
        self.assertIsNone(report["selected"])
        self.assertTrue(any(scope["status"] == "unknown" for scope in report["scope_results"]))

    def test_unrelated_missing_observation_does_not_poison_a_repair(self):
        self.table(0)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        obs = self.observations()
        obs[2]["semantic"] = None
        report = generate_scope_candidates(plans, obs)
        self.assertEqual(report["selected"]["edited_agents"], [1])
        self.assertNotIn(2, report["selected"]["evidence_agents"])

    def test_unobserved_earlier_collection_keeps_a_shared_target_unknown(self):
        self.world.move(self.players[1], (8, 10))
        self.players[1].facing = (0, -1)
        self.facing(0, "stone")
        self.players[0].inventory["wood_pickaxe"] = 1
        self.players[2].inventory["wood_pickaxe"] = 1
        plans = [plan(0, "do"), plan(1, "Navigator", "stone"), plan(2)]
        obs = self.observations()
        obs[0]["semantic"] = None
        targets = build_repair_targets(plans, obs, diagnose(plans, obs))
        result = verify_joint_repair(plans, obs, targets, [
            replace(plans[1], operation="do"),
            replace(plans[2], operation="share", item="wood_pickaxe", recipient=1),
        ])
        self.assertEqual(result.status, "unknown")

    def test_unobserved_navigation_only_taints_cells_it_can_reach(self):
        self.table(1)
        self.players[2].inventory["wood"] = 1
        plans = [plan(0, "Navigator", "tree"), plan(1, "make_wood_pickaxe"), plan(2)]
        obs = self.observations()
        obs[0]["semantic"] = None
        report = generate_scope_candidates(plans, obs)
        self.assertEqual(report["selected"]["edited_agents"], [2])

    def test_noop_does_not_satisfy_the_fixed_manufacturing_goal(self):
        self.table(0)
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        result = verify_joint_repair(plans, self.observations(), self.target(plans), [plan(0)])
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.covered_targets, ())

    def test_unselected_crafting_commitment_cannot_lose_its_incoming_stock(self):
        self.table(0)
        self.table(2)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "share", item="wood", recipient=2),
                 plan(2, "make_wood_pickaxe")]
        edit = replace(plans[1], recipient=0)
        result = verify_joint_repair(plans, self.observations(), self.target(plans), [edit])
        self.assertEqual(result.status, "infeasible")
        self.assertTrue(any("Unedited agent 2" in reason for reason in result.reasons))
        report = generate_scope_candidates(plans, self.observations(), editable_agents={0, 1})
        self.assertIsNone(report["selected"])

    def test_a_full_energy_sleeper_wakes_and_keeps_its_manufacturing_commitment(self):
        self.table(0)
        self.table(2)
        self.players[1].inventory["wood"] = 1
        self.players[2].sleeping = True
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "share", item="wood", recipient=2),
                 plan(2, "make_wood_pickaxe")]
        result = verify_joint_repair(plans, self.observations(), self.target(plans),
                                     [replace(plans[1], recipient=0)])
        self.assertEqual(result.status, "infeasible")
        self.execute(plans)
        self.assertEqual(self.players[2].achievements["make_wood_pickaxe"], 1)

    def test_permissions_exclude_uneditable_agents_and_undeclared_operations(self):
        self.table(0)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        report = generate_scope_candidates(plans, self.observations(), editable_agents={0})
        self.assertIsNone(report["selected"])
        self.assertEqual(set(report["domains"]), {0})
        result = verify_joint_repair(plans, self.observations(), self.target(plans),
                                     [replace(plans[0], operation="make_iron_pickaxe")])
        self.assertEqual(result.status, "infeasible")
        self.assertIn("permission domain", result.reasons[0])

    def test_share_chain_order_matches_the_real_executor(self):
        self.table(0)
        self.players[2].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2, "share", item="wood", recipient=1)]
        edit = replace(plans[1], operation="share", item="wood", recipient=0)
        result = verify_joint_repair(plans, self.observations(), self.target(plans), [edit])
        self.assertEqual(result.status, "infeasible")
        self.execute(plans, [edit])
        self.assertEqual(self.players[0].achievements["make_wood_pickaxe"], 0)

    def test_earlier_collection_removes_material_before_second_collector(self):
        self.world.move(self.players[1], (8, 10))
        self.players[1].facing = (0, -1)
        self.facing(0, "stone")
        self.players[0].inventory["wood_pickaxe"] = 1
        self.players[2].inventory["wood_pickaxe"] = 1
        plans = [plan(0, "do"), plan(1, "Navigator", "stone"), plan(2)]
        edits = [replace(plans[1], operation="do"),
                 replace(plans[2], operation="share", item="wood_pickaxe", recipient=1)]
        result = verify_joint_repair(plans, self.observations(), self.target(plans), edits)
        self.assertEqual(result.status, "infeasible")
        self.execute(plans, edits)
        self.assertEqual(self.players[0].achievements["collect_stone"], 1)
        self.assertEqual(self.players[1].achievements["collect_stone"], 0)

    def test_existing_construction_must_execute_before_crafting(self):
        for builder, maker in [(0, 1), (1, 0)]:
            with self.subTest(builder=builder):
                self.make_env()
                self.world.move(self.players[1], (9, 8))
                self.players[builder].inventory["wood"] = 2
                self.players[2].inventory["wood"] = 1
                plans = [plan(0), plan(1), plan(2)]
                plans[builder] = plan(builder, "place_table")
                plans[maker] = plan(maker, "make_wood_pickaxe")
                edit = plan(2, "share", item="wood", recipient=maker)
                result = verify_joint_repair(plans, self.observations(), self.target(plans), [edit])
                self.assertEqual(result.status == "repairable", builder < maker)
                self.execute(plans, [edit])
                self.assertEqual(self.players[maker].achievements["make_wood_pickaxe"], int(builder < maker))

    def test_two_providers_can_restore_an_unedited_manufacturing_plan(self):
        x, y = self.players[0].pos
        self.world[(x + 1, y + 1)] = "furnace"
        self.players[0].inventory["iron"] = 1
        self.players[1].inventory["wood"] = 1
        self.players[2].inventory["coal"] = 1
        plans = [plan(0, "make_iron_pickaxe"), plan(1), plan(2)]
        one = generate_scope_candidates(plans, self.observations(), max_scope_agents=1)
        self.assertIsNone(one["selected"])
        two = generate_scope_candidates(plans, self.observations())
        self.assertEqual(two["selected"]["edited_agents"], [1, 2])
        self.execute(plans, [Plan(**edit) for edit in two["selected"]["edits"]])
        self.assertEqual(self.players[0].achievements["make_iron_pickaxe"], 1)

    def three_provider_iron_pickaxe(self):
        self.make_env(4)
        x, y = self.players[0].pos
        self.world[(x + 1, y + 1)] = "furnace"
        self.players[1].inventory["wood"] = 1
        self.players[2].inventory["coal"] = 1
        self.players[3].inventory["iron"] = 1
        return [plan(0, "make_iron_pickaxe"), plan(1), plan(2), plan(3)]

    def test_uncapped_search_finds_a_three_provider_repair(self):
        plans = self.three_provider_iron_pickaxe()
        capped = generate_scope_candidates(plans, self.observations(), max_scope_agents=2)
        self.assertIsNone(capped["selected"])
        report = generate_scope_candidates(plans, self.observations())
        self.assertIsNone(report["max_scope_agents"])
        self.assertEqual(report["selected"]["edited_agents"], [1, 2, 3])
        # Every target is covered by three agents, so four-agent scopes are skipped.
        self.assertEqual((report["searched_scope_agents"], report["search_stop"]), (3, "all_targets_covered"))
        self.execute(plans, [Plan(**edit) for edit in report["selected"]["edits"]])
        self.assertEqual(self.players[0].achievements["make_iron_pickaxe"], 1)

    def test_uncapped_search_keeps_the_smallest_scope_when_targets_cannot_all_be_covered(self):
        self.table(0)
        self.table(1)
        self.players[2].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "make_wood_pickaxe"), plan(2)]
        report = generate_scope_candidates(plans, self.observations())
        self.assertEqual((report["searched_scope_agents"], report["search_stop"]), (3, "scope_limit"))
        self.assertEqual(report["selected"]["edited_agents"], [2])
        self.assertEqual(len(report["selected"]["covered_targets"]), 1)

    def test_assignment_budget_stops_deepening_after_the_complete_small_scopes(self):
        self.table(0)
        self.table(1)
        self.players[2].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "make_wood_pickaxe"), plan(2)]
        with patch("memory_system.repair_scopes.ASSIGNMENT_BUDGET", 1):
            report = generate_scope_candidates(plans, self.observations())
        self.assertEqual((report["searched_scope_agents"], report["search_stop"]), (2, "assignment_budget"))
        self.assertEqual(report["selected"]["edited_agents"], [2])
        plans = self.three_provider_iron_pickaxe()
        with patch("memory_system.repair_scopes.ASSIGNMENT_BUDGET", 1):
            report = generate_scope_candidates(plans, self.observations())
        self.assertIsNone(report["selected"])
        self.assertIn("assignment budget", report["reason"])

    def test_any_selection_ablation_picks_a_reproducible_verified_repair(self):
        self.table(0)
        self.table(1)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "make_wood_pickaxe"), plan(2)]
        minimal = generate_scope_candidates(plans, self.observations())
        first = generate_scope_candidates(plans, self.observations(), selection="any")
        second = generate_scope_candidates(plans, self.observations(), selection="any")
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["selected"]["status"], "repairable")
        # The ablation does not stop deepening once every target is covered.
        self.assertEqual((minimal["search_stop"], first["search_stop"]), ("all_targets_covered", "scope_limit"))
        self.assertEqual(first["selection"], "any")
        self.assertGreaterEqual(len(first["selected"]["edited_agents"]), len(minimal["selected"]["edited_agents"]))
        self.execute(plans, [Plan(**edit) for edit in first["selected"]["edits"]])
        self.assertEqual(self.players[0].achievements["make_wood_pickaxe"], 1)
        with self.assertRaises(ValueError):
            generate_scope_candidates(plans, self.observations(), selection="largest")

    def test_unobserved_incoming_share_propagates_through_a_chain(self):
        self.table(2)
        plans = [plan(0, "share", item="wood", recipient=1),
                 plan(1, "share", item="wood", recipient=2), plan(2, "make_wood_pickaxe")]
        obs = self.observations()
        obs[0]["inventory"] = None
        checks = diagnose(plans, obs)
        requirement = next(check for check in checks if check.agent_id == 2 and check.rule == "inventory")
        self.assertEqual(requirement.status, "unknown")
        result = verify_joint_repair(plans, obs, build_repair_targets(plans, obs, checks), [])
        self.assertEqual(result.status, "unknown")

    def test_six_agent_capacity_enumeration_and_repeatable_choice(self):
        self.make_env(6)
        self.table(0)
        self.players[1].inventory["wood"] = 1
        plans = [plan(i, "make_wood_pickaxe" if i == 0 else "noop") for i in range(6)]
        first = generate_scope_candidates(plans, self.observations())
        second = generate_scope_candidates(plans, self.observations())
        # Scopes of up to two agents are always complete (1 + 6 + 15); one agent
        # already covers the target, so the uncapped search stops there.
        self.assertEqual(len(first["scope_results"]), 22)
        self.assertEqual(first["search_stop"], "all_targets_covered")
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["selected"]["edited_agents"], [1])

    def test_pipeline_records_suggestion_without_extra_calls_or_applying_it(self):
        self.table(0)
        self.table(1)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1, "make_wood_pickaxe"), plan(2)]
        agents = self.agents([plan(i) for i in range(3)])
        calls = []

        async def completion(messages):
            text = messages[-1].content[-1]["text"]
            agent_id = next(i for i in range(3) if f"for agent {i}" in text)
            calls.append(agent_id)
            return self.response(plans[agent_id])

        obs = self.observations()
        trace_path = Path(self.output.name) / "coordination.jsonl"
        with patch("memory_system.agent.aget_completion", side_effect=completion), redirect_stdout(io.StringIO()):
            replies = AgentThinkingProcessor.process_agent_thinking_parallel(
                agents, {agent.id: agent.context for agent in agents}, obs,
                coordination_log=trace_path,
            )
        self.assertEqual(len(calls), 7)  # 3 initial + 2 affected agents x 2 rounds.
        self.assertTrue(agents[0].coordination_blocked)
        self.assertEqual(replies[0]["scope_suggestion"]["edited_agents"], [1])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(len(trace["rounds"]), 3)
        self.assertEqual(trace["scope_analysis"]["selected"]["edited_agents"], [1])
        AgentThinkingProcessor.update_agents_from_responses(agents, replies)
        self.assertEqual(agents[1].op, "make_wood_pickaxe")
        self.assertEqual(len(agents[0].memory_system), 0)
        self.assertEqual(self.players[0].inventory["wood"], 0)
        self.assertEqual(self.players[1].inventory["wood"], 1)

    def test_scope_off_and_diagnose_modes_skip_the_search(self):
        self.table(0)
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        for mode, scope_mode in [("repair", "off"), ("diagnose", "suggest"), ("off", "suggest")]:
            with self.subTest(mode=mode, scope=scope_mode):
                agents = self.agents([plan(i) for i in range(3)])

                async def completion(messages):
                    text = messages[-1].content[-1]["text"]
                    agent_id = next(i for i in range(3) if f"for agent {i}" in text)
                    return self.response(plans[agent_id])

                with patch("memory_system.agent.aget_completion", side_effect=completion), \
                     patch("utils.generate_scope_candidates") as search, redirect_stdout(io.StringIO()):
                    AgentThinkingProcessor.process_agent_thinking_parallel(
                        agents, {agent.id: agent.context for agent in agents}, self.observations(),
                        coordination_mode=mode, max_repair_rounds=0, scope_mode=scope_mode,
                    )
                search.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
