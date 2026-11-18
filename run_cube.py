"""Run the vendored COOP2 CUBE rules with the common review/commit loop."""

import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from memory_system.coordination import Plan
from memory_system.cube_adapter import CubeAdapter, OPERATIONS, observations_from_infos
from memory_system.cube_agent import CubeAgent
from memory_system.experiments import apply_condition, experiment_config, ScriptedReviewer
from memory_system.intervention import InterventionController, write_json
from memory_system.llm_api import get_model_usage
from memory_system.review_server import ReviewServer
from third_party.coop2_cube.env.env import CoopBlockPush
from utils import get_arg_parser


class CubeSimulation:
    def __init__(self, env, agents, controller, max_steps, repair_rounds=2,
                 review_step_seconds=1, reviewer=None, planner=None):
        self.env, self.agents, self.controller = env, agents, controller
        self.max_steps, self.repair_rounds = max_steps, repair_rounds
        self.review_step_seconds, self.reviewer = review_step_seconds, reviewer
        self.planner = planner
        self.adapter = CubeAdapter()

    async def plan(self, observations, plans, agent_ids, feedback=None):
        if self.planner is not None:
            return self.planner(observations, plans, agent_ids, feedback)
        replies = await asyncio.gather(*(
            self.agents[agent_id](self.agents[agent_id].context(observations, plans, feedback))
            for agent_id in agent_ids
        ), return_exceptions=True)
        current = {plan.agent_id: plan for plan in plans}
        result = []
        for agent_id, reply in zip(agent_ids, replies):
            if isinstance(reply, BaseException):
                # Same as Crafter: a failed planning call keeps the agent's previous plan.
                self.controller.event('planning_failed', self.agents[agent_id].step, agent=agent_id,
                                      error=f'{type(reply).__name__}: {reply}')
                result.append(current[agent_id])
            else:
                result.append(Plan(**reply.metadata))
        return result

    async def run(self, seed=4):
        _, infos = self.env.reset(seed=seed)
        observations = observations_from_infos(infos)
        usage_before = get_model_usage()
        total_reward, analysis = 0.0, None
        for step in range(self.max_steps):
            for agent in self.agents:
                agent.step = step
            had_pending = self.controller.pending is not None
            if self.reviewer is not None:
                self.reviewer.poll(self.controller, step)
            if had_pending:
                for response in self.controller.before_actions(self.agents, observations):
                    self.agents[response['id']].accept(response)
            # A just accepted repair gets its next execution before another LLM revision.
            if not had_pending or self.controller.pending is not None:
                # Locked progress tasks set their agents' plans; those agents do not plan.
                self.controller.prepare(self.agents, observations)
                plans = [Plan.from_agent(agent) for agent in self.agents]
                editable = [agent.id for agent in self.agents
                            if not agent.review_paused and agent.progress_task is None]
                new = {plan.agent_id: plan for plan in await self.plan(observations, plans, editable)}
                plans = [new.get(plan.agent_id, plan) for plan in plans]
                rounds = []
                if self.controller.pending is None:
                    for revision in range(self.repair_rounds + 1):
                        targets = self.adapter.targets(plans, observations)
                        rounds.append({'round': revision, 'plans': [asdict(plan) for plan in plans],
                                       'targets': [asdict(target) for target in targets]})
                        if not targets or revision == self.repair_rounds:
                            break
                        affected = sorted({agent_id for target in targets for agent_id in target.evidence_agents
                                           if agent_id in editable})
                        if not affected:
                            break
                        revised = await self.plan(observations, plans, affected, [asdict(target) for target in targets])
                        new = {plan.agent_id: plan for plan in revised}
                        plans = [new.get(plan.agent_id, plan) for plan in plans]
                    analysis = self.adapter.generate(plans, observations,
                                                     max_scope_agents=self.controller.search_agents,
                                                     selection=self.controller.selection,
                                                     editable_agents=set(editable)) \
                        if self.controller.mode in ('auto', 'human', 'suggest') else None
                else:
                    analysis = None
                replies = [{'id': plan.agent_id, 'plan': asdict(plan), 'thoughts': None, 'error': None}
                           for plan in plans]
                replies = self.controller.process(self.agents, observations, replies, analysis)
                for response in replies:
                    self.agents[response['id']].accept(response)
                self.controller.event('planning_completed', step, rounds=rounds, scope_analysis=analysis)
            if self.controller.pending is not None:
                await asyncio.sleep(self.review_step_seconds)
            actions = {agent.name: 0 if agent.review_paused or agent.coordination_blocked else OPERATIONS.index(agent.op)
                       for agent in self.agents}
            for agent in self.agents:
                agent.action = OPERATIONS[actions[agent.name]]
            self.controller.event('action_attempt', step, actions=actions)
            _, rewards, terminated, truncated, infos = self.env.step(actions)
            after = observations_from_infos(infos)
            self.controller.observe(observations, after, self.agents, step)
            public = next(iter(after.values()))['cube']
            self.controller.event('cube_result', step, board=public, rewards=rewards)
            observations = after
            total_reward += sum(rewards.values())
            if all(terminated.values()) or all(truncated.values()):
                break
        reason = 'success' if self.adapter.success(observations) else 'step_limit'
        summary = self.controller.finish(observations, step + 1, reason)
        summary.update({'total_steps': step + 1, 'team_score': public['team_score'], 'total_reward': total_reward})
        write_json(self.controller.output_dir / 'intervention_summary.json', summary)
        usage = {key: value - usage_before.get(key, 0) for key, value in get_model_usage().items()}
        write_json(self.controller.output_dir / 'model_usage.json', usage)
        return summary


def main():
    parser = get_arg_parser()
    parser.add_argument('--grid_size', type=int, default=15)
    parser.add_argument('--block_specs', default='2:1,3:1', help='Weight:count pairs, e.g. 2:1,3:1')
    args = parser.parse_args()
    apply_condition(args)
    args.team_board = 'off'  # the shared team board is Crafter-only
    output = Path(args.results_dir)
    controller = InterventionController(
        output, mode=args.scope_mode, strategy=args.review_strategy, capacity=args.review_max_agents,
        search_agents=args.scope_max_agents, selection=args.scope_selection,
        request_budget=args.human_requests, review_trigger=args.review_trigger,
        review_cooldown=args.review_cooldown, stall_steps=args.stall_steps, stall_fraction=args.stall_fraction,
        step_limit=args.step_num, timeout=args.review_timeout,
        pause_mode=args.pause_mode,
        evidence_mode=args.evidence_mode, revalidation=args.revalidation, environment='cube',
        progress_mode=args.progress_mode, progress_commit=args.progress_commit == 'on',
        slow_margin=args.slow_margin, progress_pause=args.progress_pause == 'on',
        progress_prestage=args.progress_prestage == 'on',
    )
    write_json(output / 'config.json', experiment_config(args, 'cube'))
    env = CoopBlockPush(grid_size=args.grid_size, num_agents=args.agent_num,
                        block_specs={int(weight): int(count) for weight, count in
                                     (entry.split(':') for entry in args.block_specs.split(','))},
                        max_steps=args.step_num)
    agents = [CubeAgent(agent_id) for agent_id in range(args.agent_num)]
    server = ReviewServer(controller.review_dir, args.review_port) \
        if args.scope_mode == 'human' and args.reviewer == 'human' and args.review_port else None
    if server is not None:
        print('Plan review: ' + server.start())
    simulation = CubeSimulation(
        env, agents, controller, args.step_num,
        repair_rounds=args.repair_rounds if args.coordination_mode == 'repair' else 0,
        review_step_seconds=args.review_step_seconds,
        reviewer=ScriptedReviewer(args.scripted_decision, args.scripted_delay_steps) if args.reviewer == 'scripted' else None,
    )
    try:
        print(json.dumps(asyncio.run(simulation.run(args.seed)), indent=2))
    finally:
        env.close()
        if server is not None:
            server.close()


if __name__ == '__main__':
    main()
