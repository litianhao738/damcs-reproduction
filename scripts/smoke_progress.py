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
from memory_system.progress_repair import STAGES_AFTER, CrafterProgress, crafter_task_plans, frontier
from memory_system.repair_scopes import generate_scope_candidates
from memory_system.review_domains import CrafterAdapter
from memory_system.review_server import ReviewServer
from memory_system.team_board import TeamBoard
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

    def furnace_tasks(self):
        """Agent 0 places the furnace (4 stone): it holds 1, Agents 1 and 2 hand over 1 each, it collects 1."""
        self.reach('collect_wood', 'place_table', 'make_wood_pickaxe', 'collect_stone')
        x, y = self.players[0].pos
        self.fixture.world[(x, y + 3)] = 'stone'
        self.players[0].inventory.update(stone=1, wood_pickaxe=1)
        for agent_id in (1, 2):
            self.players[agent_id].inventory['stone'] = 1

        def task(agent_id, macros):
            return {'agent_id': agent_id, 'option_id': 'furnace', 'milestone': 'place_furnace', 'macros': macros,
                    'index': 0, 'state': {}, 'started': 0, 'budget': 20}
        return {0: task(0, [['collect', 'stone', 1, 4], ['place', 'furnace']]),
                1: task(1, [['share', 'stone', 0, 1]]), 2: task(2, [['share', 'stone', 0, 1]])}

    def test_executor_counts_handed_over_units_once(self):
        # Pilot run: the stones handed over counted as collected, so the executor stopped at 3 of 4
        # and waited at the place step until its budget ran out.
        tasks = self.furnace_tasks()
        first = crafter_task_plans(tasks, self.fixture.observations(), 0)
        self.assertEqual([first[agent_id][0].operation for agent_id in (1, 2)], ['share', 'share'])
        self.assertEqual((first[0][0].operation, first[0][0].material), ('Navigator', 'stone'))
        self.players[0].inventory['stone'] = 3
        for agent_id in (1, 2):
            self.players[agent_id].inventory['stone'] = 0
        second = crafter_task_plans({0: tasks[0]}, self.fixture.observations(), 1)
        self.assertEqual((second[0][0].operation, second[0][0].material, tasks[0]['index']),
                         ('Navigator', 'stone', 0))
        self.players[0].inventory['stone'] = 4
        third = crafter_task_plans({0: tasks[0]}, self.fixture.observations(), 2)
        self.assertEqual((third[0][0].operation, tasks[0]['index']), ('place_furnace', 1))

    def test_place_step_waits_only_for_units_a_teammate_still_promises(self):
        tasks = self.furnace_tasks()
        tasks[0]['index'] = 1
        self.players[0].inventory['stone'] = 3
        tasks[1]['macros'] = [['collect', 'stone', 1], ['share', 'stone', 0, 1]]
        waiting = crafter_task_plans({0: tasks[0], 1: tasks[1]}, self.fixture.observations(), 0)
        self.assertEqual(waiting[0][0].operation, 'noop')
        # A giver that cannot hand over ends its task; nobody brings the last stone, so Agent 0 fetches it.
        tasks[2]['macros'] = [['share', 'stone', 0, 1]]
        self.players[2].inventory['stone'] = 0
        fetching = crafter_task_plans({0: tasks[0], 2: tasks[2]}, self.fixture.observations(), 1)
        self.assertEqual(fetching[2], (None, 'item_missing'))
        self.assertEqual((fetching[0][0].operation, fetching[0][0].material), ('Navigator', 'stone'))

    def test_options_leave_the_materials_of_a_more_critical_milestone(self):
        # Pilot run: the furnace option took the only stone of the agent about to make the stone pickaxe.
        self.assertEqual((STAGES_AFTER['collect_wood'], STAGES_AFTER['make_stone_pickaxe'],
                          STAGES_AFTER['place_furnace'], STAGES_AFTER['collect_diamond']), (7, 3, 2, 0))
        self.reach('collect_wood', 'place_table', 'make_wood_pickaxe', 'collect_stone')
        x, y = self.players[0].pos
        self.fixture.world[(x, y + 3)] = 'stone'
        self.players[0].inventory.update(wood=1, stone=1, wood_pickaxe=1)
        self.players[1].inventory.update(stone=2, wood_pickaxe=1)
        self.players[2].inventory.update(stone=1, wood_pickaxe=1)
        plans = [plan(0, 'Navigator', 'table'), plan(1), plan(2)]
        obs = self.fixture.observations()
        # Kept back as its goal, or because it already holds every material for the stone pickaxe.
        for goal in ('make_stone_pickaxe', 'share'):
            progress = CrafterProgress(plans, obs, {0: goal, 1: 'share', 2: 'share'}, {0, 1, 2}, slow_margin=4)
            options = progress.analyse(only=['place_furnace'], force=True)['problems'][0]['options']
            self.assertTrue(any(option['executor'] == 0 for option in options))
            for option in options:
                self.assertNotIn('0', option['givers'])
                if option['executor'] == 0:
                    brought = sum(items.get('stone', 0) for group in ('givers', 'gatherers')
                                  for items in option[group].values()) + option['own'].get('stone', 0)
                    self.assertEqual(brought, 4)
                    if option['own']:
                        self.assertEqual(option['tasks']['0'][0], ['collect', 'stone', 1, 5])

    def test_the_critical_milestone_may_still_take_from_a_less_critical_one(self):
        self.reach('collect_wood', 'place_table', 'make_wood_pickaxe', 'collect_stone')
        self.fixture.table(2)
        self.players[2].inventory['wood'] = 1
        self.players[1].inventory['stone'] = 3
        plans = [plan(0), plan(1, 'Navigator', 'stone'), plan(2)]
        goals = {0: 'share', 1: 'place_furnace', 2: 'share'}
        analysis = CrafterProgress(plans, self.fixture.observations(), goals, {0, 1, 2},
                                   slow_margin=4).analyse(only=['make_stone_pickaxe'])
        best = analysis['problems'][0]['options'][0]
        self.assertEqual((best['executor'], best['exact'], best['givers']), (2, True, {'1': {'stone': 1}}))

    def test_no_commit_ablation_only_writes_the_assignment_into_the_context(self):
        self.wood_scene()
        controller = self.controller(mode='auto', progress_mode='auto', progress_commit=False)
        replies = self.plan_step(controller)
        self.assertEqual(controller.tasks, {})
        self.assertFalse(any('plan_override' in reply for reply in replies))
        self.assertIn('collect 1 tree', self.agents[0].intervention_notes[-1])

    def iron_scene(self, iron=(12, 22)):
        """Stone is the next stage; the only iron lies far below Agent 1 (14 steps to face it)."""
        self.reach('collect_wood', 'place_table', 'make_wood_pickaxe')
        self.fixture.world[iron] = 'iron'
        self.plans = [plan(0, 'Navigator', 'stone'), plan(1), plan(2)]
        self.goals = {0: 'collect_stone', 1: 'share', 2: 'share'}
        self.agents = self.fixture.agents(self.plans)
        self.obs = self.fixture.observations()

    def test_prestage_sends_a_free_agent_ahead_and_ends_when_the_tool_exists(self):
        self.iron_scene()
        controller = self.controller(mode='auto', progress_mode='auto')
        self.plan_step(controller)
        # Agent 0 works on the frontier (stone); Agent 1 is free and nearest to the iron.
        self.assertEqual(set(controller.tasks), {1})
        task = controller.tasks[1]
        self.assertEqual((task['prestage'], task['milestone'], task['macros'], task['estimate']),
                         ('collect_iron', 'make_stone_pickaxe', [['approach', 'iron']], 14))
        self.assertEqual((Plan.from_agent(self.agents[1]).operation, Plan.from_agent(self.agents[1]).material),
                         ('Navigator', 'iron'))
        self.assertTrue(any('wait there' in note for note in self.agents[1].intervention_notes))
        board = TeamBoard(controller).render(self.agents, self.obs, step=0)[0]
        self.assertIn('team task: wait near iron for collect_iron', board)
        start = int(self.players[1].pos[1])
        for _ in range(3):
            self.transition(controller)
            controller.prepare(self.agents, self.obs)
        self.assertGreater(int(self.players[1].pos[1]), start)
        # The stone pickaxe is made: the walk ends, ready for the hand-over.
        self.reach('collect_stone', 'make_stone_pickaxe')
        self.transition(controller)
        self.assertEqual(controller.tasks, {})
        self.assertEqual(dict(controller.prestage_releases), {'milestone_reached': 1})
        summary = controller.finish(list(self.obs.values()), 4, 'step_limit')
        self.assertEqual((summary['progress_prestaged'], summary['progress_tasks_completed']), (1, 0))
        self.assertEqual(next(event for event in self.events() if event['event'] == 'progress_prestaged')
                         ['milestone'], 'collect_iron')

    def test_prestage_trusts_routes_not_stated_goals(self):
        # Pilot run: one agent wrote "collect_diamond" as its goal all game, also while locked on
        # the walk to iron, so the diamond was never prepared.
        self.iron_scene()
        goals = {0: 'collect_stone', 1: 'collect_iron', 2: 'collect_iron'}
        proposals = CrafterProgress(self.plans, self.obs, goals, {1, 2}, slow_margin=4).prestage()
        self.assertEqual([(proposal['milestone'], proposal['agent']) for proposal in proposals],
                         [('collect_iron', 1)])
        walking = [self.plans[0], self.plans[1], plan(2, 'Navigator', 'iron')]
        self.assertEqual(CrafterProgress(walking, self.obs, goals, {1}, slow_margin=4).prestage(), [])
        # A locked agent's stated goal is stale and left out of every analysis.
        controller = self.controller(mode='auto', progress_mode='auto')
        controller.tasks[1] = {'agent_id': 1, 'milestone': 'collect_stone', 'macros': [], 'index': 0}
        self.assertEqual(set(controller._goals(self.agents, self.replies())), {0, 2})

    def test_h_keeps_the_agent_sent_ahead_waiting_until_the_review_resolves(self):
        # Pilot run: the stone pickaxe was made, the walk to iron ended, and while the reviewer
        # chose who collects the iron, the agent walked away from it for three steps.
        self.iron_scene()
        controller = self.controller(mode='human', progress_mode='human', review_trigger='all')
        self.plan_step(controller)
        self.assertEqual(controller.tasks[1]['prestage'], 'collect_iron')
        self.reach('collect_stone', 'make_stone_pickaxe')
        self.players[0].inventory['stone_pickaxe'] = 1
        self.transition(controller)
        self.assertEqual((controller.tasks, dict(controller.prestage_releases)), ({}, {'milestone_reached': 1}))
        self.plan_step(controller)
        request = json.loads((controller.review_dir / 'review_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(request['milestone'], 'collect_iron')
        held = controller.tasks[1]
        self.assertEqual((held['hold_for'], held['milestone'], held['macros']),
                         ('review_0001', 'collect_iron', [['approach', 'iron']]))
        current = Plan.from_agent(self.agents[1])
        self.assertEqual((current.operation, current.material), ('Navigator', 'iron'))
        board = TeamBoard(controller).render(self.agents, self.obs, step=1)[0]
        self.assertIn('a reviewer is choosing who collects', board)
        self.transition(controller)
        controller.prepare(self.agents, self.obs)
        self.assertEqual(self.agents[1].progress_task, held['option_id'])
        # The reply frees the agent first, so the chosen option can still use it.
        chosen = next(option for option in request['options'] if option['executor'] == 1)
        self.respond(controller, decision='choose', option_id=chosen['option_id'])
        self.assertEqual(controller.tasks[1]['option_id'], chosen['option_id'])
        self.assertNotIn('hold_for', controller.tasks[1])
        self.assertEqual(dict(controller.prestage_releases), {'milestone_reached': 1, 'review_resolved': 1})
        self.assertNotIn('stale', controller.progress_replies)
        summary = controller.finish(list(self.obs.values()), 3, 'step_limit')
        self.assertEqual(summary['progress_prestage_held'], 1)

    def test_prestage_leaves_alone_the_agents_a_waiting_problem_needs(self):
        # Coal takes Agents 0 and 1; iron's best option also needs Agent 1, so iron waits a step.
        # Agent 2, which that iron option would use, must not be sent ahead elsewhere meanwhile.
        self.iron_scene()
        controller = self.controller(mode='auto', progress_mode='auto')
        problems = [{'milestone': 'collect_coal', 'type': 'slow', 'now_estimate': None, 'now_agents': [],
                     'options': [{'option_id': 'coal', 'agents': [0, 1], 'estimate': 3, 'exact': False,
                                  'saves': None}]},
                    {'milestone': 'collect_iron', 'type': 'slow', 'now_estimate': None, 'now_agents': [],
                     'options': [{'option_id': 'iron', 'agents': [1, 2], 'estimate': 9, 'exact': False,
                                  'saves': None}]}]
        offered = {}

        def prestage_analysis(plans, observations, goals, free_agents, slow_margin, skip=()):
            offered['free'] = set(free_agents)
            return []

        with patch.object(controller.adapter, 'progress_analysis', return_value={'problems': problems}), \
                patch.object(controller.adapter, 'prestage_analysis', side_effect=prestage_analysis), \
                patch.object(controller, '_apply_progress') as applied:
            self.plan_step(controller)
        self.assertEqual([call.args[4]['option_id'] for call in applied.call_args_list], ['coal'])
        self.assertEqual(offered['free'], set())

    def test_prestage_skips_near_materials_logging_conditions_and_the_ablation(self):
        cases = {
            'iron already near an agent': ({'mode': 'auto', 'progress_mode': 'auto'}, (16, 10)),
            'progress only recorded (B)': ({'mode': 'auto', 'progress_mode': 'log'}, (12, 22)),
            'switched off': ({'mode': 'auto', 'progress_mode': 'auto', 'progress_prestage': False}, (12, 22)),
        }
        for name, (settings, iron) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.iron_scene(iron)
                controller = self.controller(**settings)
                self.plan_step(controller)
                self.assertEqual(controller.tasks, {})

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
