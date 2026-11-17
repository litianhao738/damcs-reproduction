"""Check that an interrupted Crafter episode resumes from its checkpoint exactly, offline."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

from agentscope.message import Msg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import smoke_repair_scopes as fixtures
from memory_system.agent import Agent
from memory_system.checkpoint import (
    CheckpointError, checkpoint_path, load_checkpoint, save_checkpoint, trim_after_checkpoint,
)
from memory_system.intervention import InterventionController
from memory_system.llm_api import TextModel
from memory_system.memory_system import Experience
from run import CompleteMulitAgentSimulation


class SimulatedCrash(RuntimeError):
    pass


class CheckpointChecks(unittest.TestCase):
    def episode(self):
        """A fresh, identical 3-agent episode where Agent 0 needs Agent 1's wood."""
        fixture = fixtures.RepairScopeChecks()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.table(0)
        fixture.table(1)
        fixture.players[1].inventory['wood'] = 1
        plans = [fixtures.plan(0, 'make_wood_pickaxe'), fixtures.plan(1, 'make_wood_pickaxe'),
                 fixtures.plan(2, 'move_right')]
        agents = fixture.agents(plans)
        for agent in agents:
            # Real text models: the checkpoint must pickle what a real run holds.
            agent.text_model = TextModel()
            agent.update_current_skill('noop', 'not_applicable', 'not_applicable', -1)
        controller = InterventionController(fixture.output.name, mode='auto', step_limit=4)
        return fixture, plans, agents, controller

    def run_episode(self, fixture, plans, agents, controller, checkpoint_every=0, crash_at=None,
                    resume_state=None):
        async def reply(agent, msg):
            response = fixture.response(plans[agent.id])
            agent.think_context = msg.get_text_content()
            return Msg(agent.name, response.summary, 'assistant', metadata=response.model_dump(mode='json'))

        def context(agent):
            agent.experience = Experience(path=str(agent.agent_dir))
            agent.context = agent.think_context = f'Context for agent {agent.id}'
            return agent.context

        env = resume_state['env'] if resume_state else fixture.env
        simulation = CompleteMulitAgentSimulation(
            agents, env, 4, 3, coordination_mode='repair', repair_rounds=0, scope_mode='auto',
            intervention=controller, execution_guard=True, review_step_seconds=0,
            checkpoint_dir=fixture.output.name, checkpoint_every=checkpoint_every,
            source_sha256='test-source', resume_state=resume_state,
        )
        step_environment = simulation.env_manager.step_environment

        def flaky_step(environment, actions):
            if crash_at is not None and environment._step == crash_at:
                raise SimulatedCrash(f'interrupted at step {crash_at}')
            return step_environment(environment, actions)

        simulation.env_manager.step_environment = flaky_step
        # Keep the prepared world: reset returns the current view. Patched on the class so
        # the checkpoint, which pickles the environment instance, holds no test double.
        with patch.object(type(fixture.env), 'reset', new=lambda environment: environment._obs()), \
                patch.object(Agent, 'reply', reply), patch.object(Agent, 'create_experience', context), \
                redirect_stdout(io.StringIO()):
            return simulation.run_simulation()

    @staticmethod
    def outcome(result):
        summary = result['intervention_summary']
        return {
            'total_steps': result['total_steps'],
            'players': [(info['inventory'], info['achievements'], list(map(int, info['player_pos'])))
                        for info in result['final_info']],
            'commits': summary['commits'], 'recovered': [item['recovered'] for item in summary['outcomes']],
        }

    def test_interrupted_episode_resumes_to_the_same_result(self):
        reference = self.outcome(self.run_episode(*self.episode()))

        fixture, plans, agents, controller = self.episode()
        with self.assertRaises(SimulatedCrash):
            self.run_episode(fixture, plans, agents, controller, checkpoint_every=1, crash_at=2)
        directory = Path(fixture.output.name)
        self.assertTrue(checkpoint_path(directory).exists())
        logged_steps = {json.loads(line)['step'] for line in (directory / 'events.jsonl').read_text().splitlines()}
        self.assertIn(2, logged_steps)  # the interrupted step left partial logs

        state = load_checkpoint(directory, 'test-source')
        self.assertEqual(state['step'], 2)
        self.assertIs(state['agents'][0].env, state['env'])
        trim_after_checkpoint(directory, state['loop']['step'] - 1, state['intervention'].issued)
        logged_steps = {json.loads(line)['step'] for line in (directory / 'events.jsonl').read_text().splitlines()}
        self.assertEqual(logged_steps, {0, 1})
        resumed = self.run_episode(fixture, plans, state['agents'], state['intervention'],
                                   checkpoint_every=1, resume_state=state)
        self.assertEqual(self.outcome(resumed), reference)
        self.assertEqual(resumed['final_info'][0]['achievements']['make_wood_pickaxe'], 1)
        self.assertFalse(checkpoint_path(directory).exists())  # removed once the episode finished

    def test_checkpoint_from_other_code_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, {'source_sha256': 'old-code', 'step': 3})
            with self.assertRaisesRegex(CheckpointError, 'different code'):
                load_checkpoint(directory, 'new-code')
            self.assertEqual(load_checkpoint(directory, 'old-code')['step'], 3)
            self.assertIsNone(load_checkpoint(Path(directory) / 'missing', 'old-code'))

    def test_saving_is_atomic_and_keeps_the_previous_checkpoint_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, {'source_sha256': 'code', 'step': 1})
            with self.assertRaises(Exception):
                save_checkpoint(directory, {'source_sha256': 'code', 'step': 2, 'bad': lambda: None})
            self.assertEqual(load_checkpoint(directory, 'code')['step'], 1)

    def test_trimming_drops_later_reviews_and_failure_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reviews').mkdir()
            (root / 'failures').mkdir()
            for name in ('review_0001.json', 'review_0002.json', 'review_0002.response.json'):
                (root / 'reviews' / name).write_text('{}')
            for step in (3, 7):
                (root / 'failures' / f'step_{step:05d}.json').write_text('{}')
            (root / 'coordination.jsonl').write_text(
                '\n'.join(json.dumps({'step': step}) for step in range(6)) + '\n')
            trim_after_checkpoint(root, 4, 1)
            self.assertEqual(sorted(path.name for path in (root / 'reviews').iterdir()), ['review_0001.json'])
            self.assertEqual([path.name for path in (root / 'failures').iterdir()], ['step_00003.json'])
            steps = [json.loads(line)['step'] for line in (root / 'coordination.jsonl').read_text().splitlines()]
            self.assertEqual(steps, [0, 1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main(verbosity=2)
