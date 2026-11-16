"""Verify scope witnesses and the shared controller against upstream CUBE physics."""

import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_system.coordination import Plan
from memory_system.cube_adapter import CubeAdapter, observations_from_infos
from memory_system.cube_agent import CubeAgent
from memory_system.experiments import ScriptedReviewer
from memory_system.intervention import InterventionController, write_json
from memory_system.repair_scopes import RepairTarget
from memory_system.review_server import ReviewServer
from run_cube import CubeSimulation
from third_party.coop2_cube.env.env import Block, CoopBlockPush


class FixedCube(CoopBlockPush):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._blocks = [Block(0, 2, 3, 5)]
        self._agent_positions = {'agent_0': (3, 4), 'agent_1': (4, 4), 'agent_2': (7, 0)}
        self._rebuild_occupied()
        self.task_tracker.reset()
        return self._obs(), self._info_for_agents()


class CubeChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='cube_scope_')
        self.addCleanup(self.directory.cleanup)
        self.env = FixedCube(grid_size=9, num_agents=3, block_specs={2: 1}, max_steps=8)
        _, infos = self.env.reset(seed=4)
        self.obs = observations_from_infos(infos)
        self.plans = [Plan(i, 'move_right' if i == 0 else 'noop', 'not_applicable', 'not_applicable', -1)
                      for i in range(3)]
        self.adapter = CubeAdapter()
        self.agents = [CubeAgent(i) for i in range(3)]
        for agent, plan in zip(self.agents, self.plans):
            agent.op = plan.operation

    def controller(self, **kwargs):
        return InterventionController(self.directory.name, environment='cube', **kwargs)

    def test_one_helper_edit_is_verified_and_pushes_the_real_block(self):
        report = self.adapter.generate(self.plans, self.obs, max_scope_agents=1)
        self.assertEqual(report['selected']['edited_agents'], [1])
        self.assertEqual(report['selected']['edits'][0]['operation'], 'move_right')
        _, _, _, _, infos = self.env.step({'agent_0': 4, 'agent_1': 4, 'agent_2': 0})
        actual = observations_from_infos(infos)
        self.assertEqual(actual[0]['cube']['blocks'][0]['pos'], (3, 6))
        target = RepairTarget(**report['targets'][0])
        self.assertTrue(self.adapter.target_met(target, self.obs, actual))

    def test_single_actor_force_fails_and_helper_is_in_evidence(self):
        targets = self.adapter.targets(self.plans, self.obs)
        self.assertEqual(set(targets[0].evidence_agents), {0, 1})
        _, _, _, _, infos = self.env.step({'agent_0': 4, 'agent_1': 0, 'agent_2': 0})
        self.assertEqual(infos['agent_0']['last_step_events']['failed_pushes'][0]['reason'], 'insufficient_force')

    def test_immutable_helper_cannot_be_silently_changed(self):
        result = self.adapter.generate(self.plans, self.obs, editable_agents={0, 2})
        self.assertIsNone(result['selected'])
        self.assertEqual(result['status'], 'infeasible')

    def test_destination_blockage_is_not_fixed_by_simultaneous_move_away(self):
        self.env._agent_positions['agent_2'] = (3, 7)
        self.env._rebuild_occupied()
        obs = observations_from_infos(self.env._info_for_agents())
        report = self.adapter.generate(self.plans, obs)
        self.assertIsNone(report['selected'])
        self.assertEqual(report['status'], 'infeasible')

    def test_public_snapshot_round_trip_has_same_joint_result(self):
        report = self.adapter.generate(self.plans, self.obs)
        snapshot = json.loads(json.dumps(self.adapter.snapshot(self.obs)))
        restored = self.adapter.restore(snapshot)
        result = self.adapter.verify(self.plans, restored,
            [RepairTarget(**target) for target in report['targets']],
            [Plan(**edit) for edit in report['selected']['edits']],
            {key: tuple(Plan(**p) for p in domain) for key, domain in report['domains'].items()})
        self.assertEqual(result.status, 'repairable')

    def scripted_planner(self, observations, plans, agent_ids, feedback):
        # A deliberately uncoordinated planner leaves helper 1 idle.
        return [replace(plans[i], operation='move_right' if i == 0 else 'noop') for i in agent_ids]

    def test_auto_runner_delivers_and_records_actual_terminal_success(self):
        controller = self.controller(mode='auto')
        simulation = CubeSimulation(self.env, self.agents, controller, 6, repair_rounds=2,
                                    planner=self.scripted_planner, review_step_seconds=0)
        summary = asyncio.run(simulation.run())
        self.assertTrue(summary['task_success'])
        self.assertEqual(summary['team_score'], 2)
        self.assertEqual(summary['total_steps'], 2)
        self.assertTrue(all(outcome['recovered'] for outcome in summary['outcomes']))

    def test_human_runner_waits_locally_then_applies_joint_plan(self):
        controller = self.controller(mode='human', pause_mode='local')
        reviewer = ScriptedReviewer(delay_steps=1)
        simulation = CubeSimulation(self.env, self.agents, controller, 7, repair_rounds=0,
                                    planner=self.scripted_planner, review_step_seconds=0, reviewer=reviewer)
        summary = asyncio.run(simulation.run())
        self.assertTrue(summary['task_success'])
        self.assertEqual(summary['requests_issued'], 2)
        self.assertEqual(summary['paused_agent_steps'], 4)
        self.assertEqual(summary['human_seconds'], 0)
        events = [json.loads(line) for line in (Path(self.directory.name) / 'events.jsonl').read_text().splitlines()]
        replies = [event for event in events if event['event'] == 'review_resolved']
        self.assertTrue(all(event['provider'] == 'scripted' for event in replies))

    def test_http_preview_uses_cube_rules_and_changes_nothing(self):
        controller = self.controller(mode='human')
        analysis = self.adapter.generate(self.plans, self.obs)
        responses = [{'id': p.agent_id, 'plan': asdict(p), 'thoughts': None, 'error': None} for p in self.plans]
        controller.process(self.agents, self.obs, responses, analysis)
        server = ReviewServer(controller.review_dir, port=0)
        url = server.start()
        self.addCleanup(server.close)
        body = urlencode({'id': 'review_0001', 'decision': 'preview', 'agent_1': '4'}).encode()
        html = urlopen(Request(url + '/respond', data=body)).read().decode()
        self.assertIn('repairable', html)
        self.assertFalse((controller.review_dir / 'review_0001.response.json').exists())
        self.assertEqual(self.env._blocks[0].c, 5)


if __name__ == '__main__':
    unittest.main()
