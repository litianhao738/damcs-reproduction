"""Scope generation for the upstream COOP2 CUBE environment."""

from collections import Counter
from dataclasses import asdict, replace
from itertools import product
from time import perf_counter

from third_party.coop2_cube.env.env import Block, CoopBlockPush
from .coordination import Plan
from .repair_scopes import (
    SCOPE_SELECTIONS, RepairTarget, ScopeCandidate, ScopeSearch, select_any_repair, select_repair_scope,
)


OPERATIONS = ('noop', 'move_up', 'move_down', 'move_left', 'move_right')
DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def observations_from_infos(infos):
    return {int(name.split('_')[1]): {
        'cube': info, 'inventory': {}, 'semantic': None,
        'player_pos': tuple(next(agent['pos'] for agent in info['agents'] if agent['name'] == name)),
        'player_facing': (0, 1), 'sleeping': False, 'dead': False, 'achievements': {},
    } for name, info in infos.items()}


def _project(plans, observations):
    # Reconstruct only the public, fully observed entity state. This is a
    # model instance, never a copy or query of the live simulator internals.
    state = next(iter(observations.values()))['cube']
    env = CoopBlockPush(grid_size=state['grid_size'], num_agents=len(state['agents']),
                        block_specs=dict(Counter(block['weight'] for block in state['blocks'])), max_steps=2)
    env.reset(seed=0)
    env._blocks = [Block(block['id'], block['weight'], *block['pos']) for block in state['blocks']]
    env._agent_positions = {agent['name']: tuple(agent['pos']) for agent in state['agents']}
    env._rebuild_occupied()
    env.task_tracker.reset()
    env.step({f'agent_{plan.agent_id}': OPERATIONS.index(plan.operation) for plan in plans})
    pushes = {(event['block_id'], event['direction']) for event in env._last_step_events['successful_pushes']}
    return pushes, env._agent_positions, env._last_step_events


class CubeAdapter:
    name = 'cube'

    @staticmethod
    def critical(target):
        # Every failed push is critical: an episode has only a few blocks, and each
        # delivery is a direct share of the team score.
        return True

    def moves(self, plans, observations):
        return {}

    def targets(self, plans, observations):
        state = next(iter(observations.values()))['cube']
        _, _, events = _project(plans, observations)
        blocks = {block['id']: Block(block['id'], block['weight'], *block['pos']) for block in state['blocks']}
        targets = []
        for failure in events['failed_pushes']:
            block, direction = blocks[failure['block_id']], failure['direction']
            dr, dc = DELTAS[direction]
            adjacent = {(r - dr, c - dc) for r, c in block.border_facing(direction)}
            nr, nc = block.next_pos(direction)
            destination = {(nr + r, nc + c) for r in range(block.weight) for c in range(block.weight)}
            related = tuple(agent['id'] for agent in state['agents']
                            if tuple(agent['pos']) in adjacent | destination)
            owner = int(failure['agents'][0].split('_')[1])
            targets.append(RepairTarget(f'block:{block.id}:{direction}', owner, 'push',
                                        f'{block.id}:{direction}', True, {'participants': block.weight}, related))
        return targets

    def generate(self, plans, observations, max_scope_agents=None, editable_agents=None, selection='minimal'):
        if selection not in SCOPE_SELECTIONS:
            raise ValueError(f'selection must be one of {SCOPE_SELECTIONS}')
        started = perf_counter()
        targets = self.targets(plans, observations)
        editable = set(observations) if editable_agents is None else set(editable_agents)
        domains = {plan.agent_id: tuple(replace(plan, operation=operation) for operation in OPERATIONS)
                   for plan in plans if plan.agent_id in editable}
        original = {plan.agent_id: plan for plan in plans}
        candidates, exact, scopes = {}, {}, []
        count = 0
        search = ScopeSearch(domains, original, max_scope_agents)
        all_targets = {target.target_id for target in targets}

        def covered_all():
            return selection == 'minimal' and any(
                set(result.covered_targets) >= all_targets for result in candidates.values())

        for scope in (scope for level in search.levels(covered_all) for scope in level) if targets else []:
            scopes.append(scope)
            results = []
            choices = [tuple(plan for plan in domains[agent_id] if plan != original[agent_id]) for agent_id in scope]
            for edits in product(*choices):
                result = self.verify(plans, observations, targets, edits, domains)
                count += 1
                results.append(result.status)
                if result.status == 'repairable':
                    signature = (result.covered_targets, tuple(result.changed_fields))
                    candidates.setdefault(signature, result)
            exact[scope] = results
        values = list(candidates.values())
        minimal = [candidate for candidate in values if not any(
            set(other.changed_fields) < set(candidate.changed_fields)
            and set(other.covered_targets) >= set(candidate.covered_targets) for other in values
        )]
        selected = select_repair_scope(minimal, targets, max_scope_agents) if selection == 'minimal' \
            else select_any_repair(values, targets)
        summaries = [{'agents': scope, 'status': 'repairable' if any(
            'repairable' in results for subset, results in exact.items() if set(subset) <= set(scope)
        ) else 'infeasible'} for scope in scopes] if targets else []
        return {'environment': 'cube', 'targets': [asdict(target) for target in targets],
                'domains': {key: [asdict(plan) for plan in domain] for key, domain in domains.items()},
                'candidates': [candidate.to_dict() for candidate in minimal], 'scope_results': summaries,
                'selected': selected.to_dict() if selected else None, 'horizon': 1,
                **search.report(), 'selection': selection, 'assignments_evaluated': count,
                'status': 'repairable' if selected else 'infeasible' if targets else 'no_targets',
                'elapsed_seconds': perf_counter() - started}

    def verify(self, plans, observations, targets, edits, allowed_domains, navigator_actions=None,
               protected_agents=None):
        original = {plan.agent_id: plan for plan in plans}
        if any(edit.agent_id not in allowed_domains or edit not in allowed_domains[edit.agent_id] for edit in edits):
            return ScopeCandidate(tuple(edits), {}, 'infeasible', {}, (), (), ('Outside permission domain.',))
        changed = {edit.agent_id: ('operation',) for edit in edits if edit != original[edit.agent_id]}
        replacements = {edit.agent_id: edit for edit in edits}
        proposed = [replacements.get(plan.agent_id, plan) for plan in plans]
        before, baseline_positions, _ = _project(plans, observations)
        after, projected_positions, _ = _project(proposed, observations)
        protected = set(observations) - set(changed) if protected_agents is None else set(protected_agents) - set(changed)
        state = next(iter(observations.values()))['cube']
        initial_positions = {agent['name']: tuple(agent['pos']) for agent in state['agents']}
        preserve = all(projected_positions[f'agent_{agent_id}'] == baseline_positions[f'agent_{agent_id}']
                       for agent_id in protected if baseline_positions[f'agent_{agent_id}'] != initial_positions[f'agent_{agent_id}'])
        preserve &= before <= after
        results = {target.target_id: 'satisfied' if tuple(map(int, target.subject.split(':'))) in after else 'violated'
                   for target in targets}
        covered = tuple(sorted(key for key, value in results.items() if value == 'satisfied'))
        return ScopeCandidate(tuple(edits), changed, 'repairable' if covered and preserve else 'infeasible',
                              results, covered, tuple(sorted(observations)),
                              () if preserve else ('An unedited movement or successful push was lost.',))

    def dependencies(self, plans, observations, edits, targets):
        replacements = {edit.agent_id: edit for edit in edits}
        proposed = [replacements.get(plan.agent_id, plan) for plan in plans]
        expected, positions, _ = _project(proposed, observations)
        _, original_positions, _ = _project(plans, observations)
        related = set(replacements) | {target.agent_id for target in targets}
        related.update(agent_id for agent_id in observations
                       if positions[f'agent_{agent_id}'] != original_positions[f'agent_{agent_id}'])
        for plan in proposed:
            if plan.agent_id in related:
                continue
            for operation in OPERATIONS:
                altered = [replace(current, operation=operation) if current.agent_id == plan.agent_id else current
                           for current in proposed]
                pushes, moved, _ = _project(altered, observations)
                if pushes != expected or any(moved[f'agent_{agent_id}'] != positions[f'agent_{agent_id}']
                                             for agent_id in related):
                    related.add(plan.agent_id)
                    break
        return related

    def facts(self, plans, observations, agents, navigator_actions=None):
        state = next(iter(observations.values()))['cube']
        facts = {'board': {'grid_size': state['grid_size'], 'agents': state['agents'], 'blocks': state['blocks']}}
        facts.update({f'agent:{plan.agent_id}': asdict(plan) for plan in plans if plan.agent_id in agents})
        return facts

    def evidence(self, plans, observations, targets, edits, domains, expected, mode='compact'):
        related = self.dependencies(plans, observations, edits, targets)
        result = self.verify(plans, observations, targets, edits, domains)
        return {'mode': mode, 'agents': sorted(related), 'facts': self.facts(plans, observations, related),
                'sufficient': result.status == 'repairable' and set(expected) <= set(result.covered_targets),
                'fact_groups_before': 1, 'fact_groups_after': 1,
                'minimality': 'Complete observed board retained as one transition-model fact group.',
                'navigator_actions': {}}

    def snapshot(self, observations, full=False):
        return {str(key): value['cube'] for key, value in observations.items()}

    def restore(self, data):
        return observations_from_infos({f'agent_{key}': value for key, value in data.items()})

    def target_met(self, target, before, after):
        block_id, direction = map(int, target.subject.split(':'))
        events = next(iter(after.values() if isinstance(after, dict) else after))['cube']['last_step_events']
        return any(event['block_id'] == block_id and event['direction'] == direction
                   for event in events['successful_pushes'])

    def success(self, info):
        return not next(iter(info.values() if isinstance(info, dict) else info))['cube']['blocks']
