"""Sequential paired-seed experiment matrix and episode-level result export."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_system.experiments import CONDITIONS
from memory_system.intervention import write_json


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def build_matrix(args):
    variants = [(args.scope_max_agents, args.evidence_mode, args.pause_mode, args.revalidation)]
    if args.suite == 'capacity':
        variants = [(k, args.evidence_mode, args.pause_mode, args.revalidation) for k in (1, 2, None)]
    elif args.suite == 'evidence':
        variants = [(args.scope_max_agents, evidence, args.pause_mode, args.revalidation) for evidence in ('full', 'compact')]
    elif args.suite == 'waiting':
        variants = [(args.scope_max_agents, args.evidence_mode, pause, validation)
                    for pause, validation in (('global', 'full'), ('none', 'full'), ('local', 'incremental'))]
    matrix = []
    for condition, seed, repeat, (k, evidence, pause, validation) in product(
            args.conditions, args.seeds, range(args.repeats), variants):
        label = f'{condition}_seed{seed}_r{repeat}_k{"all" if k is None else k}_{evidence}_{pause}_{validation}'
        directory = Path(args.results_dir).resolve() / label
        command = [sys.executable, '-B', str(ROOT / ('run_cube.py' if args.environment == 'cube' else 'run.py')),
                   '--condition', condition, '--seed', str(seed), '--repeat_id', str(repeat),
                   '--agent_num', str(args.agent_num), '--step_num', str(args.step_num),
                   *(['--scope_max_agents', str(k)] if k is not None else []),
                   '--review_max_agents', str(args.review_max_agents), '--human_requests', str(args.human_requests),
                   '--evidence_mode', evidence, '--pause_mode', pause, '--revalidation', validation,
                   '--reviewer', args.reviewer, '--review_timeout', str(args.review_timeout),
                   '--review_step_seconds', str(args.review_step_seconds),
                   '--scripted_delay_steps', str(args.scripted_delay_steps),
                   '--results_dir', str(directory), '--no_plot']
        if args.environment == 'cube':
            command += ['--grid_size', str(args.grid_size), '--block_specs', args.block_specs]
        matrix.append({'run_id': label, 'directory': str(directory), 'command': command,
                       'environment': args.environment, 'condition': condition, 'seed': seed, 'repeat_id': repeat,
                       'reviewer': args.reviewer, 'scope_max_agents': k,
                       'review_max_agents': args.review_max_agents, 'evidence_mode': evidence,
                       'pause_mode': pause, 'revalidation': validation, 'human_requests': args.human_requests})
    return matrix


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def json_lines(path):
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                yield json.loads(line)


def summarize_run(directory):
    directory = Path(directory)
    config = read_json(directory / 'config.json')
    summary = read_json(directory / 'intervention_summary.json')
    events = list(json_lines(directory / 'events.jsonl'))
    analyses = [entry['scope_analysis'] for entry in json_lines(directory / 'coordination.jsonl')
                if entry.get('scope_analysis') is not None]
    analyses += [event['scope_analysis'] for event in events
                 if event['event'] == 'planning_completed' and event.get('scope_analysis') is not None]
    target_snapshots = [analysis for analysis in analyses if analysis['targets']]
    reviews = [read_json(path) for path in sorted((directory / 'reviews').glob('review_[0-9][0-9][0-9][0-9].json'))]
    replies = [event for event in events if event['event'] == 'review_resolved']
    row = {key: config.get(key) for key in (
        'environment', 'condition', 'seed', 'repeat_id', 'reviewer', 'scope_max_agents', 'review_max_agents',
        'human_requests', 'evidence_mode', 'pause_mode', 'revalidation', 'source_sha256',
    )}
    row.update({key: summary.get(key) for key in (
        'total_steps', 'task_success', 'termination_reason', 'requests_issued', 'human_seconds', 'commits',
        'paused_agent_steps', 'team_score', 'total_reward',
        'validation_seconds', 'review_preparation_seconds',
    )})
    outcomes = summary.get('outcomes', [])
    row.update({
        'run_dir': str(directory), 'completed': bool(summary),
        'target_snapshots': len(target_snapshots),
        'selected_snapshots': sum(analysis['selected'] is not None for analysis in target_snapshots),
        'assignments_evaluated': sum(analysis['assignments_evaluated'] for analysis in analyses),
        'search_seconds': sum(analysis['elapsed_seconds'] for analysis in analyses),
        'reviews_with_verified_witness': sum(review['evidence']['sufficient'] for review in reviews),
        'review_agent_slots': sum(len(review['edit_agents']) for review in reviews),
        'evidence_groups_before': sum(review['evidence']['fact_groups_before'] for review in reviews),
        'evidence_groups_after': sum(review['evidence']['fact_groups_after'] for review in reviews),
        'real_human_replies': sum(reply.get('provider') == 'human_web' for reply in replies),
        'scripted_replies': sum(reply.get('provider') == 'scripted' for reply in replies),
        'invalid_replies': sum(reply['status'] in ('invalid', 'permission', 'stale_plan') for reply in replies),
        'timeouts': sum(reply['status'] == 'timeout' for reply in replies),
        'recoveries': sum(outcome['recovered'] for outcome in outcomes), 'recovery_attempts': len(outcomes),
        **read_json(directory / 'model_usage.json'),
    })
    row.update({f'validation_{key}': value for key, value in summary.get('validation_counts', {}).items()})
    return row


def export_rows(output, rows):
    write_json(output / 'episodes.json', rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['crafter', 'cube'], default='crafter')
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=['A', 'B', 'C', 'H0', 'H1', 'H2'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[4, 5, 6])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--suite', choices=['main', 'capacity', 'evidence', 'waiting'], default='main')
    parser.add_argument('--agent_num', type=int, default=6)
    parser.add_argument('--step_num', type=int, default=350)
    parser.add_argument('--scope_max_agents', type=positive_int, default=None,
                        help='Repair search cap; default none. The capacity suite compares 1, 2 and none.')
    parser.add_argument('--review_max_agents', type=positive_int, default=2,
                        help='Most agents one human review may open.')
    parser.add_argument('--human_requests', type=int, default=3)
    parser.add_argument('--pause_mode', choices=['global', 'local', 'none'], default='global')
    parser.add_argument('--evidence_mode', choices=['full', 'compact'], default='compact')
    parser.add_argument('--revalidation', choices=['incremental', 'full'], default='incremental')
    parser.add_argument('--reviewer', choices=['human', 'scripted'], default='human')
    parser.add_argument('--review_timeout', type=float, default=120)
    parser.add_argument('--review_step_seconds', type=float, default=1)
    parser.add_argument('--scripted_delay_steps', type=int, default=2)
    parser.add_argument('--grid_size', type=int, default=15)
    parser.add_argument('--block_specs', default='2:1,3:1')
    parser.add_argument('--results_dir', default='results/experiments')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    return parser


def main():
    args = get_parser().parse_args()
    output = Path(args.results_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(args)
    if not args.summarize_only:
        write_json(output / 'matrix.json', matrix)
    if args.dry_run:
        for run in matrix:
            print(subprocess.list2cmdline(run['command']))
        return
    if args.summarize_only:
        matrix = read_json(output / 'matrix.json')
    rows = []
    for run in matrix:
        directory = Path(run['directory'])
        if not args.summarize_only:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Running {run['run_id']}", flush=True)
            started = perf_counter()
            with (directory / 'console.log').open('w', encoding='utf-8') as stream:
                process = subprocess.run(run['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            write_json(directory / 'process.json', {'exit_code': process.returncode,
                                                    'process_seconds': perf_counter() - started})
        row = {**summarize_run(directory), **run, **read_json(directory / 'process.json')}
        row.pop('command')
        rows.append(row)
        export_rows(output, rows)
    print(f"Saved {len(rows)} episodes to {output / 'episodes.csv'}")


if __name__ == '__main__':
    main()
