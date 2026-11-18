"""Progress repair: legal but slow plans, checked against real Crafter and CUBE transitions."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import smoke_repair_scopes as fixtures
from scripts.smoke_repair_scopes import plan
from memory_system.agent import Agent
from memory_system.coordination import Plan
from memory_system.cube_adapter import CubeAdapter, observations_from_infos
from memory_system.cube_agent import CubeAgent
from memory_system.intervention import InterventionController, slow_margin_for, write_json
from memory_system.llm_api import ResponseEvent, run_async
from memory_system.progress_repair import CrafterProgress, frontier
from memory_system.repair_scopes import generate_scope_candidates
from memory_system.review_domains import CrafterAdapter
from memory_system.review_server import ReviewServer
from third_party.coop2_cube.env.env import Block, CoopBlockPush
from utils import AgentActionProcessor, AgentThinkingProcessor

PLAN_FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'agentscope_plan.json'


def response(current, goal):
    data = json.loads(PLAN_FIXTURE.read_text(encoding='utf-8'))
    action = data['action']
    action['next_action'] = action['final_next_action'] = current.operation
    action['final_target_material_to_collect'] = current.material
    action['final_target_material_to_share'] = current.item
    action['final_target_agent_id'] = current.recipient
    data['goal']['current_goal'] = goal
    return ResponseEvent.model_validate(data)


class CrafterProgressChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RepairScopeChecks()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.env._length = 500
        self.players = self.fixture.players
        self.output = Path(self.fixture.output.name)

    def reach(self, *milestones):
        for name in milestones:
            self.players[0].achievements[name] = 1

    def stone_pickaxe_scene(self):
        """Agent 0 holds wood and stone, but the only table stands next to Agent 2, eight cells away."""
        self.reach('collect_wood', 'place_table', 'make_wood_pickaxe', 'collect_stone')
        self.fixture.table(2)
        self.players[0].inventory.update(wood=1, stone=1)
        self.players[2].inventory['stone'] = 1
        self.plans = [plan(0, 'Navigator', 'table'), plan(1), plan(2)]
        self.goals = {0: 'make_stone_pickaxe', 1: 'share', 2: 'share'}
        self.agents = self.fixture.agents(self.plans)
        self.obs = self.fixture.observations()

    def wood_scene(self):
        """Nobody works on the first milestone; a tree stands four cells below Agent 0."""
        x, y = self.players[0].pos
        self.fixture.world[(x, y + 4)] = 'tree'
        self.plans = [plan(0), plan(1), plan(2)]
        self.goals = {0: 'share', 1: 'share', 2: 'share'}
        self.agents = self.fixture.agents(self.plans)
        self.obs = self.fixture.observations()

    def replies(self):
        return [{'id': current.agent_id, 'thoughts': response(current, self.goals[current.agent_id]), 'error': None}
                for current in self.plans]

    def controller(self, **kwargs):
        return InterventionController(self.output, **{'slow_margin': 4, **kwargs})

    def plan_step(self, controller):
        replies = controller.process(self.agents, self.obs, self.replies(),
                                     generate_scope_candidates(self.plans, self.obs))
        AgentThinkingProcessor.update_agents_from_responses(self.agents, replies)
        return replies

    def transition(self, controller):
        for agent in self.agents:
            agent.info = self.obs[agent.id]
        with redirect_stdout(io.StringIO()), patch.object(Agent, 'create_experience'):
            AgentActionProcessor.process_all_agent_actions(self.agents, self.fixture.env, 3)
            actions = AgentActionProcessor.collect_agent_actions(self.agents, 3)
            _, _, _, info = self.fixture.env.step(actions)
        controller.observe(list(self.obs.values()), info, self.agents, self.fixture.env._step)
        self.obs = dict(enumerate(info))
        for agent in self.agents:
            agent.step += 1
        return info

    def events(self):
        return [json.loads(line) for line in (self.output / 'events.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_frontier_and_stall_progress_follow_the_diamond_chain(self):
        self.reach('collect_wood', 'place_table', 'make_wood_pickaxe', 'collect_stone')
        self.players[1].achievements['collect_drink'] = 3
        obs = self.fixture.observations()
        self.assertEqual(frontier(obs), ['make_stone_pickaxe', 'collect_coal', 'place_furnace'])
        # Drinking is not progress toward the diamond.
        self.assertEqual(CrafterAdapter.progress(list(obs.values())), 4)

    def test_slow_margin_scales_with_episode_length(self):
        self.assertEqual(slow_margin_for(None, 350), 10)
        self.assertEqual(slow_margin_for(None, 30), 2)
        self.assertEqual(slow_margin_for(7, 350), 7)
        self.assertEqual(slow_margin_for(None, None), 10)

    def test_exact_substitute_hands_the_wood_to_the_agent_at_the_table(self):
        self.stone_pickaxe_scene()
        analysis = CrafterProgress(self.plans, self.obs, self.goals, {0, 1, 2}, slow_margin=4).analyse()
        problem = next(entry for entry in analysis['problems'] if entry['milestone'] == 'make_stone_pickaxe')
        self.assertEqual(problem['type'], 'exact')
        self.assertEqual((problem['now_estimate'], problem['now_agents']), (10, [0]))
        best = problem['options'][0]
        self.assertEqual((best['template'], best['executor'], best['estimate'], best['exact'], best['agents']),
                         ('substitute', 2, 1, True, [0, 2]))
        self.assertEqual(best['first_plans'], {'0': asdict(plan(0, 'share', item='wood', recipient=2)),
                                               '2': asdict(plan(2, 'make_stone_pickaxe'))})
        # Real Crafter: the share happens first, so Agent 2 makes the pickaxe in this very step.
        self.fixture.execute(self.plans, [Plan(**data) for data in best['first_plans'].values()])
        self.assertEqual(self.players[2].achievements['make_stone_pickaxe'], 1)

    def test_a_b_c_only_record_progress_problems(self):
        self.stone_pickaxe_scene()
        controller = self.controller(mode='auto', progress_mode='log')
        replies = self.plan_step(controller)
        self.assertFalse(any('plan_override' in reply for reply in replies))
        self.assertEqual(self.agents[0].op, 'Navigator')
        summary = controller.finish(list(self.obs.values()), 1, 'step_limit')
        self.assertEqual((summary['progress_target_steps'], summary['progress_targets_by_type']), (1, {'exact': 1}))
        self.assertEqual(summary['progress_commits'], 0)
        self.assertEqual(next(event for event in self.events() if event['event'] == 'progress_problems')
                         ['problems'][0]['milestone'], 'make_stone_pickaxe')

    def test_d_applies_the_exact_option_and_measures_it(self):
        self.stone_pickaxe_scene()
        controller = self.controller(mode='auto', progress_mode='auto')
        self.plan_step(controller)
        self.assertEqual((self.agents[0].op, self.agents[2].op), ('share', 'make_stone_pickaxe'))
        self.assertIn('Team assignment', self.agents[2].intervention_notes[-2])
        info = self.transition(controller)
        self.assertEqual(self.players[2].achievements['make_stone_pickaxe'], 1)
        summary = controller.finish(info, 1, 'step_limit')
        self.assertEqual((summary['progress_exact_commits'], summary['progress_exact_success']), (1, 1))
        self.assertEqual(summary['progress_estimate_errors'], [0])
        self.assertEqual(summary['progress_commits_by_template'], {'substitute': 1})

    def test_locked_task_walks_collects_and_releases_without_asking_the_model(self):
        self.wood_scene()
        controller = self.controller(mode='auto', progress_mode='auto')
        self.plan_step(controller)
        self.assertEqual(set(controller.tasks), {0})
        self.assertEqual(self.agents[0].op, 'Navigator')
        self.assertIsNone(run_async(Agent.process_agent(self.agents[0], {}, list(self.obs.values())))['thoughts'])
        for _ in range(8):
            self.transition(controller)
            controller.prepare(self.agents, self.obs)
            if not controller.tasks:
                break
        self.assertEqual(self.players[0].achievements['collect_wood'], 1)
        self.assertEqual(controller.tasks, {})
        self.assertIsNone(self.agents[0].progress_task)
        released = next(event for event in self.events() if event['event'] == 'progress_task_released')
        self.assertEqual(released['reason'], 'milestone_reached')
        summary = controller.finish(list(self.obs.values()), self.fixture.env._step, 'step_limit')
        self.assertEqual((summary['progress_tasks'], summary['progress_tasks_completed']), (1, 1))
        self.assertGreater(summary['committed_agent_steps'], 0)

    def test_locked_task_is_released_after_its_budget(self):
        self.wood_scene()
        controller = self.controller(mode='auto', progress_mode='auto')
        self.plan_step(controller)
        controller.tasks[0]['budget'] = 1
        self.transition(controller)
        controller.prepare(self.agents, self.obs)
        self.assertEqual(controller.tasks, {})
        self.assertEqual(dict(controller.task_releases), {'budget': 1})
        self.assertIn('ended: budget', self.agents[0].intervention_notes[-1])

    def test_no_commit_ablation_only_writes_the_assignment_into_the_context(self):
        self.wood_scene()
        controller = self.controller(mode='auto', progress_mode='auto', progress_commit=False)
        replies = self.plan_step(controller)
        self.assertEqual(controller.tasks, {})
        self.assertFalse(any('plan_override' in reply for reply in replies))
        self.assertIn('collect 1 tree', self.agents[0].intervention_notes[-1])

    def respond(self, controller, **payload):
        write_json(controller.review_dir / f"{controller.pending['id']}.response.json",
                   {'request_id': controller.pending['id'], 'read_seconds': 3, 'provider': 'scripted', **payload})
        replies = controller.before_actions(self.agents, self.obs)
        AgentThinkingProcessor.update_agents_from_responses(self.agents, replies)
        return replies

    def test_h_does_not_ask_when_the_first_option_is_clearly_better(self):
        # The exact one-step handover beats the walking alternatives: no close call to make.
        self.stone_pickaxe_scene()
        controller = self.controller(mode='human', progress_mode='human')
        self.plan_step(controller)
        self.assertEqual((controller.issued, controller.pending), (0, None))
        self.assertEqual((self.agents[0].op, self.agents[2].op), ('share', 'make_stone_pickaxe'))
        skipped = next(event for event in self.events() if event['event'] == 'review_skipped')
        self.assertEqual((skipped['reason'], skipped['milestone']), ('not_critical', 'make_stone_pickaxe'))

    def test_choice_signal_needs_a_close_call_between_different_agents(self):
        controller = self.controller(mode='human', progress_mode='human')

        def option(agents, cost, exact=True):
            return {'agents': agents, 'cost': cost, 'exact': exact}

        best = option([0, 2], 9)
        cases = {
            'close, different agents': ([best, option([1, 2], 12)], ['choice']),
            'one agent apart in cost (Δ = 4)': ([best, option([1, 2], 13)], []),
            'exact against an estimate': ([best, option([1, 2], 10, exact=False)], []),
            'same agents': ([best, option([2, 0], 10)], []),
            'only one option': ([best], []),
        }
        for name, (options, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(controller._progress_signals({'milestone': 'make_stone_pickaxe'}, options, 0),
                                 expected)

    def test_progress_review_holds_nobody_and_closes_once_the_milestone_is_reached(self):
        self.stone_pickaxe_scene()
        controller = self.controller(mode='human', progress_mode='human', review_trigger='all', pause_mode='local')
        self.plan_step(controller)
        request = json.loads((controller.review_dir / 'review_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(request['pause_agents'], [])
        self.assertFalse(any(agent.review_paused for agent in self.agents))
        # While the review is open the team keeps its own plans for this milestone; no second request.
        self.plan_step(controller)
        self.assertEqual((controller.issued, self.agents[0].op), (1, 'Navigator'))
        # The team makes the pickaxe on its own before anyone answers: the request closes by itself.
        self.players[0].achievements['make_stone_pickaxe'] = 1
        self.obs = self.fixture.observations()
        controller.before_actions(self.agents, self.obs)
        self.assertIsNone(controller.pending)
        self.assertEqual(dict(controller.progress_replies), {'reached': 1})
        status = json.loads((controller.review_dir / 'review_0001.status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['status'], 'already_reached')
        self.assertEqual(controller.paused_agent_steps, 0)

    def test_progress_pause_ablation_holds_the_option_agents(self):
        self.stone_pickaxe_scene()
        controller = self.controller(mode='human', progress_mode='human', review_trigger='all', pause_mode='local',
                                     progress_pause=True)
        self.plan_step(controller)
        request = json.loads((controller.review_dir / 'review_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(request['pause_agents'], request['edit_agents'])
        self.assertTrue(all(self.agents[agent_id].review_paused for agent_id in request['pause_agents']))

    def test_h_applies_the_option_the_reviewer_chooses(self):
        self.stone_pickaxe_scene()
        controller = self.controller(mode='human', progress_mode='human', review_trigger='all')
        self.plan_step(controller)
        request = json.loads((controller.review_dir / 'review_0001.json').read_text(encoding='utf-8'))
        self.assertEqual((request['kind'], request['milestone']), ('progress', 'make_stone_pickaxe'))
        self.assertEqual(request['trigger_signals'], {'progress:make_stone_pickaxe': ['all']})
        other = request['options'][1]
        self.assertEqual((other['executor'], other['exact']), (1, False))
        self.respond(controller, decision='choose', option_id=other['option_id'])
        self.assertIsNone(controller.pending)
        self.assertEqual(set(controller.tasks), {0, 1, 2})
        self.assertEqual((self.agents[0].op, self.agents[2].op, self.agents[1].op), ('share', 'share', 'Navigator'))
        self.assertEqual(dict(controller.progress_replies), {'other': 1})
        summary = controller.finish(list(self.obs.values()), 1, 'step_limit')
        self.assertEqual((summary['requests_progress'], summary['requests_legal']), (1, 0))

    def test_h_keep_holds_the_milestone_and_skip_uses_the_recommended_option(self):
        self.stone_pickaxe_scene()
        controller = self.controller(mode='human', progress_mode='human', review_trigger='all')
        self.plan_step(controller)
        self.respond(controller, decision='keep')
        self.assertEqual((controller.tasks, controller.progress_hold), ({}, {'make_stone_pickaxe': 10}))
        self.plan_step(controller)
        self.assertEqual(controller.issued, 1)
        self.assertEqual(self.agents[0].op, 'Navigator')
        # Once the hold ends, the problem is asked about again; skipping applies the recommendation.
        for agent in self.agents:
            agent.step = 10
        self.plan_step(controller)
        self.assertEqual(controller.issued, 2)
        self.respond(controller, decision='reject')
        self.assertEqual((self.agents[0].op, self.agents[2].op), ('share', 'make_stone_pickaxe'))
        self.assertEqual(dict(controller.progress_replies), {'keep': 1, 'skip': 1})

    def test_review_page_lists_the_options_and_records_keep(self):
        self.stone_pickaxe_scene()
        controller = self.controller(mode='human', progress_mode='human', review_trigger='all')
        self.plan_step(controller)
        server = ReviewServer(controller.review_dir, 0)
        address = server.start()
        self.addCleanup(server.close)
        with urlopen(address + '/review?id=review_0001') as page:
            text = page.read().decode('utf-8')
        for expected in ('进度问题', '做石镐', '换人做', '把 1 个木头送给 Agent 2', '不改', '没有人停下等你'):
            self.assertIn(expected, text)
        data = urlencode({'id': 'review_0001', 'decision': 'choose', 'option': '__keep__'}).encode()
        with urlopen(Request(address + '/respond', data=data)) as reply:
            reply.read()
        saved = json.loads((controller.review_dir / 'review_0001.response.json').read_text(encoding='utf-8'))
        self.assertEqual((saved['decision'], saved['option_id']), ('keep', None))

    def test_agent_with_the_model_off_waits_without_any_model_call(self):
        self.wood_scene()
        agent = self.agents[0]
        agent.llm_enabled = False
        agent.update_state(self.obs[0], 3, self.fixture.env, self.obs[0], 0, 3)
        with patch('memory_system.agent.aget_completion', side_effect=AssertionError('chat called')), \
                patch('memory_system.memory_system.Experience.generate_embedding',
                      side_effect=AssertionError('embedding called')):
            self.assertEqual(agent.create_experience(), '')
            reply = run_async(agent.reply(None))
        thoughts = ResponseEvent.model_validate(reply.metadata)
        self.assertEqual((thoughts.action.final_next_action.value, thoughts.goal.current_goal.value, thoughts.timestep),
                         ('noop', 'share', 3))

    def test_controller_with_locked_tasks_survives_a_checkpoint(self):
        self.wood_scene()
        controller = self.controller(mode='auto', progress_mode='auto')
        self.plan_step(controller)
        restored = pickle.loads(pickle.dumps(controller))
        self.assertEqual(restored.tasks[0]['macros'], [['collect', 'tree', 1]])


class FarCube(CoopBlockPush):
    """A weight-2 block at rows 3-4, columns 5-6; agents placed by the test."""
    positions = {'agent_0': (0, 0), 'agent_1': (7, 0)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._blocks = [Block(0, 2, 3, 5)]
        self._agent_positions = dict(self.positions)
        self._rebuild_occupied()
        self.task_tracker.reset()
        return self._obs(), self._info_for_agents()


class CubeProgressChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='cube_progress_')
        self.addCleanup(self.directory.cleanup)
        self.adapter = CubeAdapter()

    def start(self, positions):
        FarCube.positions = positions
        self.env = FarCube(grid_size=9, num_agents=2, block_specs={2: 1}, max_steps=30)
        _, infos = self.env.reset(seed=4)
        self.obs = observations_from_infos(infos)
        self.agents = [CubeAgent(agent_id) for agent_id in range(2)]
        self.controller = InterventionController(self.directory.name, environment='cube', mode='auto',
                                                 progress_mode='auto', slow_margin=2, step_limit=30)

    def step(self, step):
        for agent in self.agents:
            agent.step = step
        self.controller.prepare(self.agents, self.obs)
        plans = [Plan.from_agent(agent) for agent in self.agents]
        replies = [{'id': current.agent_id, 'plan': asdict(current), 'thoughts': None, 'error': None}
                   for current in plans]
        analysis = self.adapter.generate(plans, self.obs, editable_agents={agent.id for agent in self.agents
                                                                            if agent.progress_task is None})
        for reply in self.controller.process(self.agents, self.obs, replies, analysis):
            self.agents[reply['id']].accept(reply)
        actions = {agent.name: ('noop', 'move_up', 'move_down', 'move_left', 'move_right').index(agent.op)
                   for agent in self.agents}
        _, _, terminated, _, infos = self.env.step(actions)
        after = observations_from_infos(infos)
        self.controller.observe(self.obs, after, self.agents, step + 1)
        self.obs = after
        return all(terminated.values())

    def test_agents_walk_to_the_face_and_push_together_until_delivery(self):
        self.start({'agent_0': (0, 0), 'agent_1': (7, 0)})
        for step in range(20):
            if self.step(step):
                break
        self.assertEqual(self.env._team_score, 2.0)
        summary = self.controller.finish(self.obs, step + 1, 'success')
        self.assertEqual(summary['progress_commits_by_template'], {'push': 1})
        self.assertEqual((summary['progress_tasks'], summary['progress_tasks_completed']), (2, 2))
        # Seven steps to line up, two pushes: as estimated.
        self.assertEqual(summary['progress_estimate_errors'], [0])

    def test_a_block_with_another_block_in_its_path_gets_no_option(self):
        class Jammed(FarCube):
            def reset(self, seed=None, options=None):
                observation, info = super().reset(seed=seed, options=options)
                # Block 1 sits in block 0's rows to its right and also covers part of its own face.
                self._blocks = [Block(0, 2, 4, 5), Block(1, 3, 2, 7)]
                self._rebuild_occupied()
                return self._obs(), self._info_for_agents()

        self.start({'agent_0': (0, 0), 'agent_1': (7, 0)})
        self.env = Jammed(grid_size=15, num_agents=2, block_specs={2: 1, 3: 1}, max_steps=30)
        _, infos = self.env.reset(seed=4)
        obs = observations_from_infos(infos)
        plans = [Plan(agent_id, 'noop', 'not_applicable', 'not_applicable', -1) for agent_id in range(2)]
        analysis = self.adapter.progress_analysis(plans, obs, {}, {0, 1}, 2)
        self.assertEqual(analysis['problems'], [])
        stalled = self.adapter.progress_analysis(plans, obs, {}, {0, 1}, 2, stalled=True)
        self.assertEqual({problem['type'] for problem in stalled['problems']}, {'unknown'})

    def test_agents_already_at_the_face_push_now(self):
        self.start({'agent_0': (3, 4), 'agent_1': (4, 4)})
        self.step(0)
        summary = self.controller.finish(self.obs, 1, 'step_limit')
        self.assertEqual((summary['progress_exact_commits'], summary['progress_exact_success']), (1, 1))
        self.assertEqual(self.env._blocks[0].c, 6)


if __name__ == '__main__':
    unittest.main()
