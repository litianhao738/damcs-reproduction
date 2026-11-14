"""Check cooperation rules against Crafter mechanics and the planning pipeline."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from crafter import constants, engine, objects
from crafter.env import Env
from memory_system.agent import ActionStatus, Agent
from memory_system.coordination import Plan, coordinate, diagnose
from memory_system.llm_api import ResponseEvent, run_async
from memory_system.memory_system import Experience
from utils import AgentActionProcessor, AgentThinkingProcessor


def plan(agent_id, operation="noop", material="not_applicable",
         item="not_applicable", recipient=-1):
    return Plan(agent_id, operation, material, item, recipient)


class CooperationChecks(unittest.TestCase):
    def setUp(self):
        self.world = engine.World((9, 9), constants.materials, (12, 12))
        self.world._mat_map[:] = self.world._mat_ids["grass"]
        self.players = [
            objects.Player(self.world, position)
            for position in [(3, 3), (5, 3), (3, 5)]
        ]
        for player in self.players:
            self.world.add(player)
        self.output = tempfile.TemporaryDirectory(prefix="mcrafter-cooperation-")
        self.addCleanup(self.output.cleanup)
        self.fixture = json.loads(
            (PROJECT_ROOT / "scripts/fixtures/agentscope_plan.json").read_text(encoding="utf-8")
        )

    def observations(self):
        semantic = engine.SemanticView(self.world, [objects.Player, objects.Cow])()
        return {
            index: {
                "inventory": player.inventory.copy(),
                "semantic": semantic.copy(),
                "player_pos": player.pos.copy(),
                "player_facing": player.facing,
                "sleeping": False,
                "dead": False,
            }
            for index, player in enumerate(self.players)
        }

    def response(self, operation="noop", material="not_applicable",
                 item="not_applicable", recipient=-1):
        data = json.loads(json.dumps(self.fixture))
        action = data["action"]
        action["next_action"] = action["final_next_action"] = operation
        action["final_target_material_to_collect"] = material
        action["final_target_material_to_share"] = item
        action["final_target_agent_id"] = recipient
        return ResponseEvent.model_validate(data)

    def agent(self, agent_id):
        with patch("memory_system.agent.TextModel"):
            agent = Agent(agent_id, results_dir=self.output.name)
        agent.step = 0
        agent.context = f"Current context for agent {agent_id}."
        agent.experience = Experience(path=str(agent.agent_dir))
        return agent

    def share(self, sender, recipient, item):
        # Execute the real exchange method; the method only needs these fields.
        Env.exchange_item(SimpleNamespace(player_id=sender, _players=self.players), recipient, item)

    def test_crafting_facility_can_be_diagonal_and_not_facing(self):
        player = self.players[0]
        player.inventory["wood"] = 1
        self.world[(4, 4)] = "table"
        checks = diagnose([plan(0, "make_wood_pickaxe")], self.observations())
        self.assertTrue(all(check.status == "satisfied" for check in checks))
        player.action = "make_wood_pickaxe"
        player.update()
        self.assertEqual(player.inventory["wood_pickaxe"], 1)
        self.assertEqual(player.inventory["wood"], 0)

    def test_missing_facility_matches_game_failure(self):
        self.players[0].inventory["wood"] = 1
        checks = diagnose([plan(0, "make_wood_pickaxe")], self.observations())
        self.assertTrue(any(
            check.rule == "crafting_facility" and check.status == "violated" for check in checks
        ))
        self.players[0].action = "make_wood_pickaxe"
        self.players[0].update()
        self.assertEqual(self.players[0].inventory["wood_pickaxe"], 0)
        self.assertEqual(self.players[0].inventory["wood"], 1)

    def test_collection_uses_actual_tool_requirement(self):
        player = self.players[0]
        self.world[(3, 4)] = "iron"
        player.inventory["wood_pickaxe"] = 1
        checks = diagnose([plan(0, "do")], self.observations())
        self.assertTrue(any(check.status == "violated" for check in checks))
        player.action = "do"
        player.update()
        self.assertEqual(player.inventory["iron"], 0)
        player.inventory["stone_pickaxe"] = 1
        checks = diagnose([plan(0, "do")], self.observations())
        self.assertTrue(all(check.status == "satisfied" for check in checks))
        player.update()
        self.assertEqual(player.inventory["iron"], 1)

    def test_placement_matches_real_terrain_and_occupancy_rules(self):
        player = self.players[0]
        self.world[(3, 4)] = "water"
        player.inventory["wood"] = 2
        checks = diagnose([plan(0, "place_table")], self.observations())
        self.assertTrue(any(check.rule == "placement" and check.status == "violated" for check in checks))
        player.action = "place_table"
        player.update()
        self.assertEqual(player.inventory["wood"], 2)

        player.inventory["stone"] = 1
        checks = diagnose([plan(0, "place_stone")], self.observations())
        self.assertTrue(all(check.status == "satisfied" for check in checks))
        player.action = "place_stone"
        player.update()
        self.assertEqual(self.world[(3, 4)][0], "stone")

        self.world[(3, 4)] = "grass"
        cow = objects.Cow(self.world, (3, 4))
        self.world.add(cow)
        checks = diagnose([plan(0, "place_table")], self.observations())
        self.assertTrue(any(check.rule == "placement" and check.status == "violated" for check in checks))
        player.action = "place_table"
        player.update()
        self.assertEqual(player.inventory["wood"], 2)

    def test_do_on_a_facility_is_not_a_crafting_operation(self):
        self.world[(3, 4)] = "table"
        player = self.players[0]
        player.inventory["wood"] = 1
        checks = diagnose([plan(0, "do")], self.observations())
        self.assertTrue(any(check.rule == "collection" and check.status == "violated" for check in checks))
        player.action = "do"
        player.update()
        self.assertEqual(player.inventory["wood"], 1)
        self.assertEqual(player.inventory["wood_pickaxe"], 0)

    def test_planned_share_supplies_crafting_without_distance_rule(self):
        self.players[0].inventory["wood"] = 1
        self.world[(5, 4)] = "table"
        plans = [plan(0, "share", item="wood", recipient=1), plan(1, "make_wood_pickaxe")]
        observations = self.observations()
        checks = diagnose(plans, observations)
        self.assertTrue(all(check.status == "satisfied" for check in checks))
        self.assertEqual(observations[0]["inventory"]["wood"], 1)
        self.assertEqual(observations[1]["inventory"]["wood"], 0)
        self.share(0, 1, "wood")
        self.players[1].action = "make_wood_pickaxe"
        self.players[1].update()
        self.assertEqual(self.players[1].inventory["wood_pickaxe"], 1)

    def test_share_chains_follow_actual_agent_order(self):
        self.players[0].inventory["wood"] = 1
        plans = [
            plan(0, "share", item="wood", recipient=1),
            plan(1, "share", item="wood", recipient=2),
        ]
        self.assertFalse(any(check.status == "violated" for check in diagnose(plans, self.observations())))
        self.share(0, 1, "wood")
        self.share(1, 2, "wood")
        self.assertEqual(self.players[2].inventory["wood"], 1)

        self.players[0].inventory["wood"] = 0
        self.players[1].inventory["wood"] = 1
        plans = [
            plan(0, "share", item="wood", recipient=2),
            plan(1, "share", item="wood", recipient=0),
        ]
        checks = diagnose(plans, self.observations())
        self.assertTrue(any(check.agent_id == 0 and check.status == "violated" for check in checks))

    def test_invalid_share_recipient_and_empty_stock_are_diagnosed(self):
        for recipient in [-1, 0, 99]:
            with self.subTest(recipient=recipient):
                checks = diagnose([plan(0, "share", item="wood", recipient=recipient)], self.observations())
                self.assertTrue(any(
                    check.rule == "share_target" and check.status == "violated" for check in checks
                ))
        checks = diagnose([plan(0, "share", item="wood", recipient=1)], self.observations())
        self.assertTrue(any(check.rule == "inventory" and check.status == "violated" for check in checks))

    def test_absent_observation_is_unknown_not_zero(self):
        observations = self.observations()
        observations[0]["inventory"] = None
        observations[0]["semantic"] = None
        checks = diagnose([plan(0, "make_wood_pickaxe")], observations)
        self.assertTrue(checks)
        self.assertTrue(all(check.status == "unknown" for check in checks))

    def test_pending_unverified_share_keeps_requirement_unknown(self):
        observations = self.observations()
        observations[0]["inventory"] = None
        self.world[(5, 4)] = "table"
        observations[1] = self.observations()[1]
        plans = [plan(0, "share", item="wood", recipient=1), plan(1, "make_wood_pickaxe")]
        checks = diagnose(plans, observations)
        self.assertTrue(any(
            check.agent_id == 1 and check.rule == "inventory" and check.status == "unknown"
            for check in checks
        ))

    def test_pending_facility_is_not_assumed_built(self):
        self.players[0].inventory["wood"] = 2
        self.players[1].inventory["wood"] = 1
        plans = [plan(0, "place_table"), plan(1, "make_wood_pickaxe")]
        # The proposed table is diagonally nearby, but has not been built.
        self.world.move(self.players[1], (4, 3))
        checks = diagnose(plans, self.observations())
        self.assertTrue(any(
            check.agent_id == 1 and check.rule == "crafting_facility" and check.status == "unknown"
            for check in checks
        ))

    def test_same_resource_target_is_not_a_conflict(self):
        checks = diagnose(
            [plan(0, "Navigator", "tree"), plan(1, "Navigator", "tree")],
            self.observations(),
        )
        self.assertFalse(any(check.status == "violated" for check in checks))

    def test_targeted_repair_can_replan_a_provider_and_preserve_other_work(self):
        self.world[(3, 4)] = "table"
        self.players[1].inventory["wood"] = 2
        agents = [self.agent(i) for i in range(3)]
        agents[1].op = "Navigator"
        agents[1].rss_to_collect = "tree"
        agents[2].op = "Navigator"
        agents[2].rss_to_collect = "tree"
        agents[2].action_status = ActionStatus.IN_PROGRESS
        responses = [
            {"id": 0, "thoughts": self.response("make_wood_pickaxe"), "error": None},
            {"id": 1, "thoughts": None, "error": None},
            {"id": 2, "thoughts": None, "error": None},
        ]
        called = []

        async def replan(agent_id, context):
            called.append(agent_id)
            self.assertIn("Verified cooperation feedback", context)
            if agent_id == 0:
                return self.response("make_wood_pickaxe")
            return self.response("share", item="wood", recipient=0)

        revised, history = run_async(coordinate(
            agents, self.observations(), {0: agents[0].context}, responses, replan,
        ))
        self.assertEqual(called, [0, 1])
        self.assertEqual(len(history) - 1, 1)
        self.assertFalse(any(agent.coordination_blocked for agent in agents))
        self.assertIsNone(revised[2]["thoughts"])
        self.assertEqual(agents[2].action_status, ActionStatus.IN_PROGRESS)
        AgentThinkingProcessor.update_agents_from_responses(agents, revised)
        self.assertEqual(len(agents[0].memory_system), 1)
        self.assertEqual(len(agents[1].memory_system), 1)

    def test_two_failed_repairs_hold_action_without_committing_failed_memories(self):
        agent = self.agent(0)
        replies = [{"id": 0, "thoughts": self.response("share", item="wood", recipient=-1), "error": None}]
        called = []

        async def replan(agent_id, context):
            called.append(agent_id)
            return self.response("share", item="wood", recipient=-1)

        revised, history = run_async(coordinate(
            [agent], {0: self.observations()[0]}, {0: agent.context}, replies, replan,
        ))
        self.assertEqual(called, [0, 0])
        self.assertEqual(len(history) - 1, 2)
        AgentThinkingProcessor.update_agents_from_responses([agent], revised)
        self.assertEqual(len(agent.memory_system), 0)
        self.assertEqual(agent.memory_system.history_actions, [])
        self.assertEqual(agent.action_status, ActionStatus.INTERRUPTED)
        before = [player.inventory.copy() for player in self.players]
        AgentActionProcessor.process_all_agent_actions([agent], object(), 1)
        self.assertEqual(agent.action, "noop")
        self.assertEqual([player.inventory for player in self.players], before)

    def test_navigation_is_not_held_and_does_not_repeat_repairs_each_move(self):
        agent = self.agent(0)
        initial = [{"id": 0, "thoughts": self.response("Navigator", "diamond"), "error": None}]
        called = []

        async def replan(agent_id, context):
            called.append(agent_id)
            return self.response("Navigator", "diamond")

        revised, _ = run_async(coordinate(
            [agent], {0: self.observations()[0]}, {0: agent.context}, initial, replan,
        ))
        self.assertEqual(len(called), 2)
        self.assertFalse(agent.coordination_blocked)
        AgentThinkingProcessor.update_agents_from_responses([agent], revised)
        called.clear()
        run_async(coordinate(
            [agent], {0: self.observations()[0]}, {},
            [{"id": 0, "thoughts": None, "error": None}], replan,
        ))
        self.assertEqual(called, [])
        self.assertFalse(agent.coordination_blocked)

    def test_diagnose_only_does_not_replan_or_hold(self):
        agent = self.agent(0)

        async def unexpected_replan(agent_id, context):
            self.fail("Diagnosis-only mode must not invoke the model.")

        revised, _ = run_async(coordinate(
            [agent], {0: self.observations()[0]}, {0: agent.context},
            [{"id": 0, "thoughts": self.response("make_wood_pickaxe"), "error": None}],
            unexpected_replan, repair=False,
        ))
        self.assertFalse(revised[0]["blocked"])
        self.assertTrue(any(check["status"] == "violated" for check in revised[0]["constraints"]))

    def test_held_agent_can_resume_after_preconditions_change(self):
        agent = self.agent(0)
        self.world[(3, 4)] = "table"

        async def replan(agent_id, context):
            return self.response("make_wood_pickaxe")

        inputs = [{"id": 0, "thoughts": self.response("make_wood_pickaxe"), "error": None}]
        revised, _ = run_async(coordinate(
            [agent], {0: self.observations()[0]}, {0: agent.context}, inputs, replan,
        ))
        AgentThinkingProcessor.update_agents_from_responses([agent], revised)
        self.assertTrue(agent.coordination_blocked)
        self.players[0].inventory["wood"] = 1
        revised, history = run_async(coordinate(
            [agent], {0: self.observations()[0]}, {0: agent.context}, inputs, replan,
        ))
        self.assertEqual(len(history), 1)
        AgentThinkingProcessor.update_agents_from_responses([agent], revised)
        self.assertFalse(agent.coordination_blocked)
        self.assertIsNone(agent.thinking_error)
        self.assertEqual(len(agent.memory_system), 1)

    def test_real_agentscope_planning_pipeline_revises_only_affected_agent(self):
        agents = [self.agent(0), self.agent(1)]
        calls = []

        async def completion(messages):
            text = messages[-1].content[-1]["text"]
            agent_id = 0 if "for agent 0" in text else 1
            is_repair = "Verified cooperation feedback" in text
            calls.append((agent_id, is_repair))
            if agent_id == 0:
                return self.response("Navigator", "tree") if is_repair else self.response("make_wood_pickaxe")
            return self.response()

        observations = self.observations()
        with patch("memory_system.agent.aget_completion", side_effect=completion):
            with redirect_stdout(io.StringIO()):
                replies = AgentThinkingProcessor.process_agent_thinking_parallel(
                    agents, {agent.id: agent.context for agent in agents},
                    [observations[0], observations[1]],
                )
        self.assertEqual(calls, [(0, False), (1, False), (0, True)])
        AgentThinkingProcessor.update_agents_from_responses(agents, replies)
        self.assertEqual(agents[0].op, "Navigator")
        self.assertEqual(agents[1].op, "noop")
        self.assertEqual(len(agents[0].memory_system), 1)

    def test_invalid_repair_reply_is_reported_and_retried_within_two_rounds(self):
        agents = [self.agent(0), self.agent(1)]
        repairs = []

        async def completion(messages):
            text = messages[-1].content[-1]["text"]
            if "for agent 1" in text:
                return self.response()
            if "Verified cooperation feedback" not in text:
                return self.response("make_wood_pickaxe")
            repairs.append(text)
            if len(repairs) == 1:
                invalid = self.response().model_dump(mode="json")
                invalid["goal"]["current_goal"] = "navigate to diamond"
                return ResponseEvent.model_validate(invalid)
            self.assertIn("ValidationError", text)
            return self.response("Navigator", "tree")

        observations = self.observations()
        trace_path = Path(self.output.name) / "coordination.jsonl"
        with patch("memory_system.agent.aget_completion", side_effect=completion):
            with redirect_stdout(io.StringIO()):
                replies = AgentThinkingProcessor.process_agent_thinking_parallel(
                    agents, {agent.id: agent.context for agent in agents},
                    [observations[0], observations[1]], coordination_log=trace_path,
                )
        AgentThinkingProcessor.update_agents_from_responses(agents, replies)
        self.assertEqual(len(repairs), 2)
        self.assertFalse(agents[0].coordination_blocked)
        self.assertEqual(agents[0].op, "Navigator")
        self.assertEqual(agents[1].op, "noop")
        self.assertEqual(len(agents[0].memory_system), 1)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertIn("ValidationError", trace["rounds"][0]["replan_errors"]["0"])
        self.assertEqual(trace["held_agents"], [])

    def test_two_invalid_repair_replies_hold_last_valid_but_infeasible_plan(self):
        agent = self.agent(0)

        async def replan(agent_id, context):
            invalid = self.response().model_dump(mode="json")
            invalid["goal"]["current_goal"] = "navigate to diamond"
            return ResponseEvent.model_validate(invalid)

        replies, history = run_async(coordinate(
            [agent], {0: self.observations()[0]}, {0: agent.context},
            [{"id": 0, "thoughts": self.response("make_wood_pickaxe"), "error": None}],
            replan,
        ))
        AgentThinkingProcessor.update_agents_from_responses([agent], replies)
        self.assertEqual(len(history) - 1, 2)
        self.assertTrue(all(history[index]["replan_errors"] for index in (0, 1)))
        self.assertTrue(agent.coordination_blocked)
        self.assertEqual(agent.op, "make_wood_pickaxe")
        self.assertEqual(len(agent.memory_system), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
