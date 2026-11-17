"""Exercise automatic and human plan commits with real Crafter transitions."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import unittest
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from unittest.mock import patch

from agentscope.message import Msg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import smoke_repair_scopes as fixtures
from memory_system.coordination import Plan, diagnose
from memory_system.intervention import InterventionController, write_json
from memory_system.repair_scopes import RepairTarget, generate_scope_candidates
from memory_system.review_evidence import build_review_evidence
from memory_system.review_server import ReviewServer
from utils import AgentActionProcessor, AgentThinkingProcessor
from memory_system.agent import Agent
from memory_system.memory_system import Experience
from memory_system.experiments import ScriptedReviewer
from run import CompleteMulitAgentSimulation


class InterventionChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RepairScopeChecks()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.table(0)
        self.fixture.table(1)
        self.fixture.players[1].inventory['wood'] = 1
        self.plans = [fixtures.plan(0, 'make_wood_pickaxe'), fixtures.plan(1, 'make_wood_pickaxe'),
                      fixtures.plan(2, 'move_right')]
        self.agents = self.fixture.agents(self.plans)
        self.obs = self.fixture.observations()
        self.analysis = generate_scope_candidates(self.plans, self.obs)
        self.output = Path(self.fixture.output.name)

    def replies(self):
        return [{'id': plan.agent_id, 'thoughts': self.fixture.response(plan), 'error': None}
                for plan in self.plans]

    def controller(self, **kwargs):
        # The wood-pickaxe scenario is not a critical moment, so most checks ask for every problem.
        return InterventionController(self.output, **{'review_trigger': 'all', **kwargs})

    def use_iron_pickaxe_scenario(self):
        """A critical moment: Agent 0 can make the iron pickaxe once Agents 1 and 2 share wood and coal."""
        self.fixture.make_env()
        x, y = self.fixture.players[0].pos
        self.fixture.world[(x + 1, y + 1)] = 'furnace'
        self.fixture.players[0].inventory['iron'] = 1
        self.fixture.players[1].inventory['wood'] = 1
        self.fixture.players[2].inventory['coal'] = 1
        self.plans = [fixtures.plan(0, 'make_iron_pickaxe'), fixtures.plan(1), fixtures.plan(2)]
        self.agents = self.fixture.agents(self.plans)
        self.obs = self.fixture.observations()
        self.analysis = generate_scope_candidates(self.plans, self.obs)

    def test_problem_outside_critical_moments_is_repaired_without_asking(self):
        controller = self.controller(mode='human', review_trigger='critical')
        self.prepare(controller)
        self.assertEqual(controller.issued, 0)
        self.assertIsNone(controller.pending)
        self.assertEqual(controller.commits[-1]['source'], 'auto')
        skipped = next(event for event in self.events() if event['event'] == 'review_skipped')
        self.assertEqual(skipped['reason'], 'not_critical')

    def test_critical_moment_is_asked_without_a_request_budget(self):
        self.use_iron_pickaxe_scenario()
        controller = InterventionController(self.output, mode='human')
        self.assertEqual((controller.review_trigger, controller.request_budget), ('critical', None))
        self.prepare(controller)
        self.assertEqual(controller.issued, 1)
        request = json.loads((controller.review_dir / 'review_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(request['edit_agents'], [1, 2])
        self.assertEqual([target['target_id'] for target in request['targets']], ['0:make:iron_pickaxe'])
        self.assertIsNone(request['remaining_requests'])
        self.respond(controller)
        self.transition(controller)
        self.assertEqual(self.fixture.players[0].achievements['make_iron_pickaxe'], 1)

    def test_same_critical_problem_waits_for_the_cooldown(self):
        self.use_iron_pickaxe_scenario()
        controller = InterventionController(self.output, mode='human', review_cooldown=10)
        self.prepare(controller)
        self.respond(controller, decision='reject')
        self.prepare(controller)
        self.assertEqual(controller.issued, 1)
        reasons = [event['reason'] for event in self.events() if event['event'] == 'review_skipped']
        self.assertEqual(reasons, ['cooldown'])
        # Ten steps later, with the situation changed, the same problem may be asked again.
        self.fixture.players[0].inventory['stone'] = 1
        self.obs = self.fixture.observations()
        self.analysis = generate_scope_candidates(self.plans, self.obs)
        for agent in self.agents:
            agent.step = 10
        self.prepare(controller)
        self.assertEqual(controller.issued, 2)

    def prepare(self, controller):
        replies = controller.process(self.agents, self.obs, self.replies(), self.analysis)
        AgentThinkingProcessor.update_agents_from_responses(self.agents, replies)
        return replies

    def transition(self, controller):
        for agent in self.agents:
            agent.info = self.obs[agent.id]
        with redirect_stdout(io.StringIO()):
            AgentActionProcessor.process_all_agent_actions(
                self.agents, self.fixture.env, 3,
                event_sink=lambda kind, **data: controller.event(kind, 0, **data),
            )
            actions = AgentActionProcessor.collect_agent_actions(self.agents, 3)
            _, _, _, info = self.fixture.env.step(actions)
        controller.observe(list(self.obs.values()), info, self.agents, self.fixture.env._step)
        self.obs = dict(enumerate(info))
        for agent in self.agents:
            agent.step += 1
        return info

    def respond(self, controller, decision='approve', edits=None, read_seconds=2):
        request = controller.pending
        payload = {'request_id': request['id'], 'decision': decision,
                   'read_seconds': read_seconds, 'provider': 'scripted'}
        if edits is not None:
            payload['edits'] = [asdict(plan) for plan in edits]
        write_json(controller.review_dir / (request['id'] + '.response.json'), payload)
        replies = controller.before_actions(self.agents, self.obs)
        AgentThinkingProcessor.update_agents_from_responses(self.agents, replies)
        return replies

    def test_auto_commit_changes_the_accepted_plan_and_real_execution(self):
        controller = self.controller(mode='auto')
        self.prepare(controller)
        self.assertEqual(self.agents[1].op, 'share')
        self.assertEqual(self.agents[1].thoughts.action.final_next_action.value, 'share')
        self.assertFalse(self.agents[0].coordination_blocked)
        self.assertIn('auto', self.agents[1].intervention_notes[-1])
        self.assertEqual(controller.outcomes, [])
        self.transition(controller)
        self.assertEqual(self.fixture.players[0].achievements['make_wood_pickaxe'], 1)
        self.assertTrue(controller.outcomes[0]['recovered'])
        events = [json.loads(line) for line in (self.output / 'events.jsonl').read_text().splitlines()]
        transfer = next(event for event in events if event['event'] == 'item_transferred')
        self.assertEqual(transfer['quantity'], 1)

    def test_human_local_pause_keeps_independent_agent_running(self):
        controller = self.controller(mode='human', pause_mode='local')
        self.prepare(controller)
        self.assertEqual(controller.issued, 1)
        self.assertEqual(controller.pending['paused'], {0, 1})
        self.assertEqual(len(self.agents[0].memory_system), 0)
        position = tuple(self.fixture.players[2].pos)
        self.transition(controller)
        self.assertEqual(tuple(self.fixture.players[2].pos), (position[0] + 1, position[1]))
        self.assertEqual(self.fixture.players[1].inventory['wood'], 1)
        self.respond(controller)
        self.assertIsNone(controller.pending)
        self.assertEqual(controller.human_seconds, 2)
        self.assertEqual(self.agents[1].op, 'share')
        self.transition(controller)
        self.assertTrue(controller.outcomes[-1]['recovered'])
        self.assertEqual(controller.outcomes[-1]['source'], 'human')

    def test_global_pause_waits_all_agents_but_keeps_the_world_clock(self):
        controller = self.controller(mode='human', pause_mode='global')
        self.prepare(controller)
        self.assertTrue(all(agent.review_paused for agent in self.agents))
        position = tuple(self.fixture.players[2].pos)
        self.transition(controller)
        self.assertEqual(tuple(self.fixture.players[2].pos), position)
        self.assertEqual(self.fixture.env._step, 1)
        self.assertEqual(controller.paused_agent_steps, 3)

    def test_zero_budget_uses_the_same_automatic_fallback(self):
        controller = self.controller(mode='human', request_budget=0)
        self.prepare(controller)
        self.assertEqual(controller.issued, 0)
        self.assertIsNone(controller.pending)
        self.assertEqual(self.agents[1].op, 'share')

    def events(self):
        return [json.loads(line) for line in (self.output / 'events.jsonl').read_text().splitlines()]

    def test_applied_repair_logs_edited_and_interrupted_helpers(self):
        controller = self.controller(mode='auto')
        self.prepare(controller)
        applied = next(event for event in self.events() if event['event'] == 'plan_applied')
        # Agent 1 was making its own pickaxe; the repair replaced that with a share.
        self.assertEqual((applied['edited_agents'], applied['interrupted_agents']), ([1], [1]))

    def stale_reply(self, revalidation):
        controller = self.controller(mode='human', revalidation=revalidation)
        self.prepare(controller)
        # The provider loses its wood while the review is pending.
        self.fixture.players[1].inventory['wood'] = 0
        self.obs = self.fixture.observations()
        self.respond(controller)
        return controller, next(event for event in self.events() if event['event'] == 'review_resolved')

    def test_recheck_rejects_a_stale_reviewed_edit(self):
        controller, resolved = self.stale_reply('incremental')
        self.assertEqual(resolved['status'], 'invalid')
        self.assertNotIn('unchecked', controller.validation_counts)

    def test_no_recheck_ablation_commits_the_stale_edit_and_counts_it(self):
        controller, resolved = self.stale_reply('none')
        self.assertEqual(resolved['status'], 'accepted')
        self.assertEqual(controller.validation_counts['unchecked'], 1)
        self.assertEqual(controller.validation_counts['unchecked_would_fail'], 1)
        applied = next(event for event in self.events() if event['event'] == 'plan_applied')
        self.assertTrue(applied['unchecked'] and applied['would_fail_check'])
        self.assertEqual(self.agents[1].op, 'share')
        with self.assertRaises(ValueError):
            self.controller(revalidation='never')

    def test_repair_larger_than_review_capacity_skips_review_and_uses_auto(self):
        self.plans = self.fixture.three_provider_iron_pickaxe()
        self.agents = self.fixture.agents(self.plans)
        self.obs = self.fixture.observations()
        self.analysis = generate_scope_candidates(self.plans, self.obs)
        self.assertEqual(self.analysis['selected']['edited_agents'], [1, 2, 3])
        controller = self.controller(mode='human', capacity=2)
        self.prepare(controller)
        self.assertEqual(controller.issued, 0)
        self.assertIsNone(controller.pending)
        self.assertEqual(controller.commits[-1]['source'], 'auto')
        events = [json.loads(line) for line in (self.output / 'events.jsonl').read_text().splitlines()]
        skipped = next(event for event in events if event['event'] == 'review_skipped')
        self.assertEqual((skipped['reason'], skipped['scope_agents'], skipped['review_capacity']),
                         ('over_review_capacity', [1, 2, 3], 2))
        self.assertEqual([agent.op for agent in self.agents[1:]], ['share'] * 3)

    def test_rejection_consumes_one_request_and_uses_auto(self):
        controller = self.controller(mode='human')
        self.prepare(controller)
        self.respond(controller, decision='reject')
        self.assertEqual(controller.issued, 1)
        self.assertEqual(controller.commits[-1]['source'], 'auto')

    def test_timeout_consumes_one_request_without_a_new_model_call(self):
        now = [0]
        controller = self.controller(mode='human', timeout=5, clock=lambda: now[0])
        self.prepare(controller)
        now[0] = 6
        replies = controller.before_actions(self.agents, self.obs)
        AgentThinkingProcessor.update_agents_from_responses(self.agents, replies)
        self.assertEqual(controller.issued, 1)
        self.assertIsNone(controller.pending)
        self.assertEqual(controller.commits[-1]['source'], 'auto')

    def test_unselected_edit_is_rejected_as_a_whole(self):
        controller = self.controller(mode='human')
        self.prepare(controller)
        edits = controller.pending['proposal'] + [replace(self.plans[2], operation='noop')]
        self.respond(controller, decision='edit', edits=edits)
        status = json.loads((controller.review_dir / 'review_0001.status.json').read_text())
        self.assertEqual(status['status'], 'permission')
        self.assertEqual(self.agents[2].op, 'move_right')
        self.assertEqual(controller.commits[-1]['source'], 'auto')

    def test_unpaused_consumption_invalidates_the_old_share_proposal(self):
        controller = self.controller(mode='human', pause_mode='none')
        self.prepare(controller)
        self.transition(controller)
        self.assertEqual(self.fixture.players[1].inventory['wood'], 0)
        self.respond(controller)
        status = json.loads((controller.review_dir / 'review_0001.status.json').read_text())
        self.assertEqual(status['status'], 'invalid')
        self.assertEqual(controller.commit_count, 0)
        self.assertNotEqual(self.agents[1].op, 'share')

    def test_new_share_into_review_evidence_is_paused_before_exchange(self):
        controller = self.controller(mode='human', pause_mode='local')
        self.prepare(controller)
        self.fixture.players[2].inventory['wood'] = 1
        self.obs = self.fixture.observations()
        new_plan = replace(self.plans[2], operation='share', item='wood', recipient=0)
        replies = [{'id': 0, 'thoughts': None, 'error': None}, {'id': 1, 'thoughts': None, 'error': None},
                   {'id': 2, 'thoughts': self.fixture.response(new_plan), 'error': None}]
        replies = controller.process(self.agents, self.obs, replies, None)
        AgentThinkingProcessor.update_agents_from_responses(self.agents, replies)
        self.assertTrue(self.agents[2].review_paused)
        self.transition(controller)
        self.assertEqual(self.fixture.players[2].inventory['wood'], 1)

    def test_compact_evidence_removes_an_unneeded_view(self):
        candidate = self.analysis['selected']
        targets = [RepairTarget(**target) for target in self.analysis['targets']]
        edits = [Plan(**edit) for edit in candidate['edits']]
        domains = {key: tuple(Plan(**plan) for plan in values) for key, values in self.analysis['domains'].items()}
        compact = build_review_evidence(self.plans, self.obs, targets, edits, domains,
                                        candidate['covered_targets'], 'compact')
        full = build_review_evidence(self.plans, self.obs, targets, edits, domains,
                                     candidate['covered_targets'], 'full')
        self.assertTrue(compact['sufficient'])
        self.assertLess(compact['fact_groups_after'], full['fact_groups_after'])
        self.assertNotIn('view:1', compact['facts'])

    def test_real_http_form_writes_a_human_response_and_it_executes(self):
        controller = self.controller(mode='human')
        self.prepare(controller)
        server = ReviewServer(controller.review_dir, 0)
        address = server.start()
        self.addCleanup(server.close)
        with urlopen(address + '/review?id=review_0001') as response:
            html = response.read().decode('utf-8')
        self.assertIn('agent_1', html)
        self.assertIn('make_wood_pickaxe', html)
        data = urlencode({'id': 'review_0001', 'decision': 'approve'}).encode()
        with urlopen(Request(address + '/respond', data=data)) as response:
            self.assertEqual(response.status, 200)
        replies = controller.before_actions(self.agents, self.obs)
        AgentThinkingProcessor.update_agents_from_responses(self.agents, replies)
        self.transition(controller)
        self.assertTrue(controller.outcomes[-1]['recovered'])

    def serve(self, controller):
        server = ReviewServer(controller.review_dir, 0)
        address = server.start()
        self.addCleanup(server.close)
        return address

    def test_review_list_polls_for_open_requests_with_time_left(self):
        controller = self.controller(mode='human')
        self.prepare(controller)
        address = self.serve(controller)
        with urlopen(address + '/') as response:
            html = response.read().decode('utf-8')
        self.assertIn('/api/requests', html)
        self.assertIn('setInterval(poll, 2000)', html)
        with urlopen(address + '/api/requests') as response:
            data = json.loads(response.read().decode('utf-8'))
        self.assertEqual(data['run'], self.output.name)
        [request] = data['requests']
        self.assertEqual((request['id'], request['status'], request['edit_agents']), ('review_0001', 'pending', [1]))
        self.assertTrue(0 < request['remaining_seconds'] <= controller.timeout)
        with urlopen(address + '/review?id=review_0001') as response:
            page = response.read().decode('utf-8')
        self.assertIn('id="countdown"', page)
        self.assertNotIn(' disabled', page)

    def post_reply(self, address):
        from urllib.error import HTTPError
        data = urlencode({'id': 'review_0001', 'decision': 'approve'}).encode()
        try:
            with urlopen(Request(address + '/respond', data=data)) as response:
                return response.status
        except HTTPError as error:
            return error.code

    def test_expired_request_refuses_a_late_reply(self):
        controller = self.controller(mode='human')
        self.prepare(controller)
        path = controller.review_dir / 'review_0001.json'
        request = json.loads(path.read_text(encoding='utf-8'))
        request['issued_at'] -= controller.timeout + 1
        write_json(path, request)
        address = self.serve(controller)
        with urlopen(address + '/review?id=review_0001') as response:
            page = response.read().decode('utf-8')
        self.assertIn('已超时', page)
        self.assertIn(' disabled', page)
        self.assertEqual(self.post_reply(address), 409)
        self.assertFalse((controller.review_dir / 'review_0001.response.json').exists())

    def test_ended_episode_refuses_a_reply_and_reports_why(self):
        controller = self.controller(mode='human')
        self.prepare(controller)
        address = self.serve(controller)
        controller.finish(self.transition(controller), 1, 'step_limit')
        with urlopen(address + '/api/requests') as response:
            [request] = json.loads(response.read().decode('utf-8'))['requests']
        self.assertEqual((request['status'], request['status_name']), ('episode_ended', '本局已结束'))
        self.assertEqual(self.post_reply(address), 409)
        self.assertFalse((controller.review_dir / 'review_0001.response.json').exists())

    def test_pending_review_is_closed_at_episode_end_without_claiming_success(self):
        controller = self.controller(mode='human')
        self.prepare(controller)
        info = self.transition(controller)
        summary = controller.finish(info, 1, 'step_limit')
        self.assertEqual(summary['requests_issued'], 1)
        self.assertFalse(summary['task_success'])
        self.assertEqual(json.loads((controller.review_dir / 'review_0001.status.json').read_text())['status'], 'episode_ended')

    def test_unchanged_evidence_reuses_the_joint_certificate(self):
        controller = self.controller(mode='human', revalidation='incremental')
        self.prepare(controller)
        self.respond(controller)
        self.assertEqual(controller.validation_counts['reused'], 1)
        self.assertEqual(controller.validation_counts['full_checks'], 0)

    def test_full_revalidation_applies_the_same_witness(self):
        controller = self.controller(mode='human', revalidation='full')
        self.prepare(controller)
        self.respond(controller)
        self.assertEqual(controller.validation_counts['full_checks'], 1)
        self.transition(controller)
        self.assertTrue(controller.outcomes[-1]['recovered'])

    def test_changed_stock_is_rechecked_instead_of_using_the_certificate(self):
        controller = self.controller(mode='human', revalidation='incremental')
        self.prepare(controller)
        self.fixture.players[1].inventory['wood'] = 2
        self.obs = self.fixture.observations()
        self.respond(controller)
        self.assertEqual(controller.validation_counts['reused'], 0)
        self.assertEqual(controller.validation_counts['related_checks'], 1)
        self.transition(controller)
        self.assertTrue(controller.outcomes[-1]['recovered'])

    def test_evidence_cannot_hide_an_unedited_crafting_commitment(self):
        self.fixture.table(2)
        plans = [self.plans[0], fixtures.plan(1, 'share', item='wood', recipient=2),
                 fixtures.plan(2, 'make_wood_pickaxe')]
        edits = [replace(plans[1], recipient=0)]
        targets = self.fixture.target(plans)
        domains = {1: (plans[1], edits[0])}
        evidence = build_review_evidence(plans, self.fixture.observations(), targets, edits, domains,
                                         ['0:make:wood_pickaxe'])
        self.assertFalse(evidence['sufficient'])
        self.assertIn('inventory:2', evidence['facts'])

    def run_complete_loop(self, mode):
        controller = self.controller(mode=mode, pause_mode='local')
        for agent in self.agents:
            agent.update_current_skill('noop', 'not_applicable', 'not_applicable', -1)

        async def reply(agent, msg):
            response = self.fixture.response(self.plans[agent.id])
            agent.think_context = msg.get_text_content()
            return Msg(agent.name, response.summary, 'assistant', metadata=response.model_dump(mode='json'))

        def context(agent):
            agent.experience = Experience(path=str(agent.agent_dir))
            agent.context = agent.think_context = f'Context for agent {agent.id}'
            return agent.context

        simulation = CompleteMulitAgentSimulation(
            self.agents, self.fixture.env, 3, 3, coordination_mode='repair', repair_rounds=0,
            scope_mode=mode, intervention=controller, execution_guard=True, review_step_seconds=0,
            reviewer=ScriptedReviewer(delay_steps=2) if mode == 'human' else None,
        )
        with patch.object(self.fixture.env, 'reset', return_value=self.fixture.env._obs()), \
                patch.object(Agent, 'reply', reply), patch.object(Agent, 'create_experience', context), \
                redirect_stdout(io.StringIO()):
            return simulation.run_simulation()

    def test_complete_crafter_auto_loop_commits_and_executes(self):
        result = self.run_complete_loop('auto')
        self.assertTrue(result['intervention_summary']['outcomes'][0]['recovered'])
        self.assertEqual(result['final_info'][0]['achievements']['make_wood_pickaxe'], 1)
        self.assertFalse(result['intervention_summary']['task_success'])

    def test_complete_crafter_human_loop_waits_then_executes(self):
        result = self.run_complete_loop('human')
        summary = result['intervention_summary']
        self.assertEqual(summary['requests_issued'], 1)
        self.assertTrue(summary['outcomes'][0]['recovered'])
        self.assertEqual(summary['outcomes'][0]['source'], 'human')
        self.assertGreater(summary['paused_agent_steps'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
