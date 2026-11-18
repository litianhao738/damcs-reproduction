"""Shared team board: each agent's own status and the team's progress, shown to all agents."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import smoke_repair_scopes as fixtures
from scripts.smoke_repair_scopes import plan
from scripts.smoke_progress import response
from scripts.run_experiments import build_matrix, get_parser
from memory_system.agent import Agent
from memory_system.experiments import apply_condition
from memory_system.intervention import InterventionController
from memory_system.team_board import TeamBoard, seen_nearby
from utils import SimulationContextManager, get_arg_parser


class TeamBoardChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RepairScopeChecks()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.players = self.fixture.players  # at (8, 8), (12, 8) and (16, 8) on open grass

    def scene(self):
        self.fixture.world[(6, 8)] = 'tree'       # 2 cells from Agent 0, 6 from Agent 1 (out of its view)
        self.fixture.world[(14, 8)] = 'diamond'   # 6 cells from Agent 0 (out of its view), 2 from Agents 1 and 2
        self.players[0].inventory.update(wood=2, health=2)
        for name in ('collect_wood', 'place_table', 'make_wood_pickaxe'):
            self.players[0].achievements[name] = 1
        plans = [plan(0, 'Navigator', 'tree'), plan(1, 'share', item='stone', recipient=0), plan(2)]
        agents = self.fixture.agents(plans)
        goals = {0: 'collect_stone', 1: 'share', 2: 'collect_coal'}
        for current, agent in zip(plans, agents):
            agent.thoughts = response(current, goals[current.agent_id])
        return agents, self.fixture.observations()

    def test_each_agent_reports_only_its_own_view(self):
        _, observations = self.scene()
        self.assertEqual(seen_nearby(observations[0]), {'tree': 2})
        self.assertEqual(seen_nearby(observations[1]), {'diamond': 2})

    def test_board_lists_status_and_progress_for_every_reader(self):
        agents, observations = self.scene()
        controller = InterventionController(self.fixture.output.name, mode='auto', progress_mode='auto')
        controller.last_progress_step = 3
        controller.tasks = {2: {'milestone': 'collect_coal'}}
        boards = TeamBoard(controller).render(agents, observations, step=10)
        self.assertEqual(set(boards), {0, 1, 2})
        text = boards[1]
        self.assertIn('Done: collect_wood, place_table, make_wood_pickaxe.', text)
        # Agent 0 and the locked Agent 2 work on the next milestones; nobody yet on the furnace chain.
        self.assertIn('collect_stone (agents 0)', text)
        self.assertIn('collect_coal (agents 2)', text)
        self.assertIn('Steps since the last new milestone: 7.', text)
        self.assertIn('- Agent 0: at (8, 8); LOW health 2; goal: collect_stone; plan: go to tree; has wood 2; '
                      'sees tree (2 away)', text)
        self.assertIn('- Agent 1 (you): at (12, 8); goal: share; plan: share stone to agent 0; '
                      'has nothing useful; sees diamond (2 away)', text)
        self.assertIn('- Agent 2: at (16, 8); goal: collect_coal; team task for collect_coal: noop', text)
        self.assertIn('(you)', boards[0].split('- Agent 0')[1].split('\n')[0])

    def test_contexts_carry_the_board_only_when_it_is_on(self):
        agents, observations = self.scene()
        manager = SimulationContextManager(3)
        with patch.object(Agent, 'create_experience', return_value='own context'):
            plain = manager.create_agent_contexts(agents, observations, 5)
            manager.board = TeamBoard()
            shared = manager.create_agent_contexts(agents, observations, 5)
        self.assertNotIn('Team board', plain[0])
        self.assertIn('### Team board', shared[0])
        self.assertTrue(shared[0].startswith('own context'))

    def test_board_follows_the_progress_module_unless_set(self):
        for condition, expected in (('A', 'off'), ('B', 'off'), ('C', 'on'), ('H', 'on')):
            args = get_arg_parser().parse_args(['--condition', condition])
            apply_condition(args)
            self.assertEqual(args.team_board, expected, condition)
        args = get_arg_parser().parse_args(['--condition', 'C', '--team_board', 'off'])
        apply_condition(args)
        self.assertEqual(args.team_board, 'off')

    def test_ablation_suite_hides_the_board_in_one_row(self):
        args = get_parser().parse_args(['--suite', 'ablation', '--seeds', '4', '--results_dir',
                                        self.fixture.output.name])
        rows = {row['variant']: row['command'] for row in build_matrix(args)}
        command = rows['C-no-board']
        self.assertEqual((command[command.index('--condition') + 1], command[command.index('--team_board') + 1]),
                         ('C', 'off'))
        self.assertTrue(command[command.index('--results_dir') + 1].endswith('noboard'))
        self.assertEqual(rows['C-no-llm'][rows['C-no-llm'].index('--team_board') + 1], 'auto')


if __name__ == '__main__':
    unittest.main()
