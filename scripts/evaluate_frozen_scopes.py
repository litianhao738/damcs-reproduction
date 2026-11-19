"""Compare scope selection and joint/independent claims on identical captured inputs."""

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_system.coordination import Plan
from memory_system.intervention import choose_review_scope, write_json
from memory_system.repair_scopes import RepairTarget, ScopeSearch
from memory_system.review_domains import get_review_adapter


def independent_merge(adapter, plans, observations, targets, domains, capacity):
    """Select each target's witness in isolation from the same action domains."""
    original = {plan.agent_id: plan for plan in plans}
    options = {agent: tuple(plan for plan in choices if plan != original[agent])
               for agent, choices in domains.items()}
    choices = {}
    search = ScopeSearch(domains, original, capacity)
    if len(targets) >= 2:
        for level in search.levels(lambda: len(choices) == len(targets)):
            for agents in level:
                if not agents:
                    continue
                size = len(agents)
                for edits in product(*(options[agent] for agent in agents)):
                    for target in targets:
                        if target.target_id in choices and choices[target.target_id][0][0] < size:
                            continue
                        result = adapter.verify(plans, observations, [target], edits, domains)
                        if result.status != 'repairable' or target.target_id not in result.covered_targets:
                            continue
                        rank = (len(result.changed_fields), sum(map(len, result.changed_fields.values())))
                        if target.target_id not in choices or rank < choices[target.target_id][0]:
                            choices[target.target_id] = (rank, result)
    fields, conflicts = {}, set()
    for _, witness in choices.values():
        for plan in witness.edits:
            changes = fields.setdefault(plan.agent_id, {})
            for field in witness.changed_fields[plan.agent_id]:
                value = getattr(plan, field)
                if field in changes and changes[field] != value:
                    conflicts.add((plan.agent_id, field))
                changes[field] = value
    claims = sorted(choices)
    covered, status = [], 'fewer_than_two_witnesses'
    if len(claims) >= 2:
        if conflicts:
            status = 'conflicting_fields'
        elif len(fields) > capacity:
            status = 'over_capacity'
        else:
            merged = adapter.verify(plans, observations, [target for target in targets if target.target_id in choices],
                                    [replace(original[agent], **patch) for agent, patch in fields.items()], domains)
            covered = list(merged.covered_targets)
            status = 'verified' if merged.status == 'repairable' and set(claims) <= set(covered) else 'joint_check_failed'
    return {'claimed_targets': claims, 'conflicting_fields': sorted(conflicts), 'edited_agents': sorted(fields),
            'jointly_valid_within_capacity': status == 'verified' if len(claims) >= 2 else None,
            'verified_merged_targets': covered, 'status': status, 'search_stop': search.stop}


def assess_snapshot(document, capacity):
    adapter = get_review_adapter(document['environment'])
    plans = [Plan(**plan) for plan in document['plans']]
    observations = adapter.restore(document['snapshot'])
    report = adapter.generate(plans, observations, max_scope_agents=capacity)
    targets_by_id = {target['target_id']: RepairTarget(**target) for target in report['targets']}
    if 'targets' in document and {target['target_id'] for target in document['targets']} != set(targets_by_id):
        raise ValueError("The replayed targets differ from the captured episode; check the source version.")
    # A globally sufficient one-agent witness can stop the minimal search before an
    # owner-only two-agent witness was examined. Search the owner domain separately.
    owner_scope = choose_review_scope(report, 'owner', capacity)
    sources = {'owner': adapter.generate(plans, observations, max_scope_agents=capacity, editable_agents=owner_scope),
               'verified': report}
    ranges = {}
    for strategy, source in sources.items():
        scope = owner_scope if strategy == 'owner' else (report['selected'] or {}).get('edited_agents', [])
        witnesses = [candidate for candidate in source['candidates'] if set(candidate['edited_agents']) <= set(scope)]
        ranges[strategy] = {'agents': scope, 'has_witness': bool(witnesses),
                            'target_coverage': max((len(candidate['covered_targets']) for candidate in witnesses), default=0),
                            'search_stop': source['search_stop']}
    domains = {int(key): tuple(Plan(**plan) for plan in options) for key, options in report['domains'].items()}
    independent = independent_merge(adapter, plans, observations, list(targets_by_id.values()), domains, capacity)
    selected = report['selected']
    evidence = None
    if selected and document['environment'] == 'crafter':
        edits = [Plan(**edit) for edit in selected['edits']]
        full = adapter.evidence(plans, observations, list(targets_by_id.values()), edits, domains,
                                selected['covered_targets'], mode='full')
        compact = adapter.evidence(plans, observations, list(targets_by_id.values()), edits, domains,
                                   selected['covered_targets'], mode='compact')
        evidence = {'full_groups': full['fact_groups_after'], 'compact_groups': compact['fact_groups_after'],
                    'full_sufficient': full['sufficient'], 'compact_sufficient': compact['sufficient']}
    edited = selected['edited_agents'] if selected else []
    covered = selected['covered_targets'] if selected else []
    owners = {targets_by_id[target].agent_id for target in covered}
    return {'step': document['step'], 'environment': document['environment'], 'capacity': capacity,
            'targets': len(report['targets']), 'strategies': ranges,
            'joint': {'covered_targets': covered, 'edited_agents': edited,
                      'non_owner_only': bool(edited) and not owners.intersection(edited),
                      'changed_fields': selected['changed_fields'] if selected else {},
                      'edited_field_count': sum(map(len, selected['changed_fields'].values())) if selected else 0,
                      'search_stop': report['search_stop']},
            'independent': independent,
            'evidence': evidence, 'search_seconds': report['elapsed_seconds'],
            'assignments_evaluated': report['assignments_evaluated']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Episode or experiment directory containing failures/*.json')
    parser.add_argument('--capacities', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--completed-only', action='store_true', help='Use only completed experiment episodes.')
    parser.add_argument('--output', default='results/frozen_scope_comparison.json')
    args = parser.parse_args()
    if any(capacity < 1 for capacity in args.capacities):
        parser.error('Capacities must be positive.')
    rows = []
    for path in sorted(Path(args.input).rglob('failures/step_*.json')):
        if any('.failed' in part for part in path.parts):
            continue
        episode = path.parent.parent
        if args.completed_only and (not (episode / 'process.json').is_file()
                or json.loads((episode / 'process.json').read_text(encoding='utf-8')).get('exit_code') != 0
                or not (episode / 'intervention_summary.json').is_file()):
            continue
        config_path = episode / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
        document = json.loads(path.read_text(encoding='utf-8'))
        for capacity in args.capacities:
            rows.append({'snapshot': str(path), 'condition': config.get('condition'), 'seed': config.get('seed'),
                         'repeat_id': config.get('repeat_id'), **assess_snapshot(document, capacity)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, rows)
    print(f'Saved {len(rows)} snapshot/capacity comparisons to {output}')


if __name__ == '__main__':
    main()
