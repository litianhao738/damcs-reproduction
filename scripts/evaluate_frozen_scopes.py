"""Compare scope selection and joint/independent claims on identical captured inputs."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_system.coordination import Plan
from memory_system.intervention import choose_review_scope, write_json
from memory_system.repair_scopes import RepairTarget
from memory_system.review_domains import get_review_adapter


def assess_snapshot(document, capacity):
    adapter = get_review_adapter(document['environment'])
    plans = [Plan(**plan) for plan in document['plans']]
    observations = adapter.restore(document['snapshot'])
    report = adapter.generate(plans, observations, max_scope_agents=capacity)
    ranges = {}
    for strategy in ('owner', 'related', 'verified'):
        scope = choose_review_scope(report, strategy, capacity)
        witnesses = [candidate for candidate in report['candidates'] if set(candidate['edited_agents']) <= set(scope)]
        ranges[strategy] = {'agents': scope, 'has_witness': bool(witnesses),
                            'target_coverage': max((len(candidate['covered_targets']) for candidate in witnesses), default=0)}
    # Each problem takes its cheapest individual witness without checking whether
    # those witnesses can coexist. This deliberately removes only joint checking.
    chosen, claims, conflicting = {}, set(), set()
    for target in report['targets']:
        options = [candidate for candidate in report['candidates'] if target['target_id'] in candidate['covered_targets']]
        if not options:
            continue
        witness = min(options, key=lambda candidate: (len(candidate['edited_agents']),
                                                     sum(map(len, candidate['changed_fields'].values()))))
        claims.add(target['target_id'])
        for patch in witness['edits']:
            plan = Plan(**patch)
            if plan.agent_id in chosen and chosen[plan.agent_id] != plan:
                conflicting.add(plan.agent_id)
            chosen[plan.agent_id] = plan
    targets = [RepairTarget(**target) for target in report['targets'] if target['target_id'] in claims]
    domains = {int(key): tuple(Plan(**plan) for plan in options) for key, options in report['domains'].items()}
    merged = adapter.verify(plans, observations, targets, list(chosen.values()), domains)
    consistent = (not conflicting and len(chosen) <= capacity and merged.status == 'repairable'
                  and claims <= set(merged.covered_targets)) if claims else None
    return {'step': document['step'], 'environment': document['environment'], 'capacity': capacity,
            'targets': len(report['targets']), 'strategies': ranges,
            'independent': {'claimed_targets': sorted(claims), 'conflicting_agents': sorted(conflicting),
                            'edited_agents': sorted(chosen), 'jointly_valid_within_capacity': consistent,
                            'verified_merged_targets': list(merged.covered_targets)},
            'search_seconds': report['elapsed_seconds'], 'assignments_evaluated': report['assignments_evaluated']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Episode or experiment directory containing failures/*.json')
    parser.add_argument('--capacities', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--output', default='results/frozen_scope_comparison.json')
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.input).rglob('failures/step_*.json')):
        document = json.loads(path.read_text(encoding='utf-8'))
        for capacity in args.capacities:
            rows.append({'snapshot': str(path), **assess_snapshot(document, capacity)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, rows)
    print(f'Saved {len(rows)} snapshot/capacity comparisons to {output}')


if __name__ == '__main__':
    main()
