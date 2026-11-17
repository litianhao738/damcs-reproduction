"""Sequential paired-seed experiment matrix and episode-level result export.

Suites:
  main       conditions x seeds x repeats (tables 1 and 3)
  ablation   the fixed component ablations of table 2 (ignores --conditions)
  capacity / evidence / waiting   earlier single-factor sweeps
With --model_profiles, every run is repeated per named profile from
config/model_profiles.json (table 3). Keys are resolved only when a run starts.
"""

import argparse
from collections import Counter
import csv
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_system.experiments import CONDITIONS
from memory_system.intervention import write_json
from memory_system.model_profiles import episode_cost, load_profiles, profile_env, profile_record
from memory_system.checkpoint import checkpoint_path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


# Table 2: each row changes one component of the full system and nothing else.
# The automatic rows start from C; the review rows start from H2 with a scripted reviewer.
ABLATIONS = [
    ('C-no-replan', 'custom', {'coordination_mode': 'repair', 'repair_rounds': 0,
                               'scope_mode': 'auto', 'review_strategy': 'verified'}),
    ('C-any-scope', 'C', {'scope_selection': 'any'}),
    ('H2script', 'H2', {'reviewer': 'scripted'}),
    ('H2script-global-wait', 'H2', {'reviewer': 'scripted', 'pause_mode': 'global'}),
    ('H2script-no-wait', 'H2', {'reviewer': 'scripted', 'pause_mode': 'none'}),
    ('H2script-full-recheck', 'H2', {'reviewer': 'scripted', 'revalidation': 'full'}),
    ('H2script-no-recheck', 'H2', {'reviewer': 'scripted', 'revalidation': 'none'}),
]

# Crafter milestone columns written by run.py, in task order.
MILESTONES = ['Collect wood', 'Place table', 'Make wood pickaxe', 'Collect stone', 'Make stone pickaxe',
              'Collect iron', 'Collect coal', 'Place furnace', 'Make iron pickaxe', 'Collect diamond']


def variants(args):
    base = {'scope_max_agents': args.scope_max_agents, 'evidence_mode': args.evidence_mode,
            'pause_mode': args.pause_mode, 'revalidation': args.revalidation,
            'scope_selection': 'minimal', 'reviewer': args.reviewer}
    if args.suite == 'ablation':
        return [(tag, condition, {**base, **override}) for tag, condition, override in ABLATIONS]
    if args.suite == 'capacity':
        settings = [{**base, 'scope_max_agents': k} for k in (1, 2, None)]
    elif args.suite == 'evidence':
        settings = [{**base, 'evidence_mode': evidence} for evidence in ('full', 'compact')]
    elif args.suite == 'waiting':
        settings = [{**base, 'pause_mode': pause, 'revalidation': validation}
                    for pause, validation in (('global', 'full'), ('none', 'full'), ('local', 'incremental'))]
    else:
        settings = [base]
    return [(condition, condition, setting) for condition in args.conditions for setting in settings]


def build_matrix(args):
    profiles = args.model_profiles or [None]
    settings_file = load_profiles() if args.model_profiles else None
    matrix = []
    for profile, (tag, condition, s), seed, repeat in product(
            profiles, variants(args), args.seeds, range(args.repeats)):
        k = s['scope_max_agents']
        parts = ([profile] if profile else []) + [
            tag, f'seed{seed}', f'r{repeat}', f'k{"all" if k is None else k}',
            s['evidence_mode'], s['pause_mode'], s['revalidation']]
        label = '_'.join(parts + (['any'] if s['scope_selection'] == 'any' else []))
        directory = Path(args.results_dir).resolve() / label
        command = [sys.executable, '-B', str(ROOT / ('run_cube.py' if args.environment == 'cube' else 'run.py')),
                   '--condition', condition, '--seed', str(seed), '--repeat_id', str(repeat),
                   '--agent_num', str(args.agent_num), '--step_num', str(args.step_num),
                   *(['--scope_max_agents', str(k)] if k is not None else []),
                   '--review_max_agents', str(args.review_max_agents),
                   *(['--human_requests', str(args.human_requests)] if args.human_requests is not None else []),
                   '--review_trigger', args.review_trigger, '--review_cooldown', str(args.review_cooldown),
                   '--stall_fraction', str(args.stall_fraction),
                   *(['--stall_steps', str(args.stall_steps)] if args.stall_steps is not None else []),
                   '--evidence_mode', s['evidence_mode'], '--pause_mode', s['pause_mode'],
                   '--revalidation', s['revalidation'], '--scope_selection', s['scope_selection'],
                   '--reviewer', s['reviewer'], '--reviewer_id', args.reviewer_id,
                   '--review_timeout', str(args.review_timeout),
                   '--review_step_seconds', str(args.review_step_seconds),
                   '--scripted_delay_steps', str(args.scripted_delay_steps),
                   '--results_dir', str(directory), '--no_plot']
        if condition == 'custom':
            command += ['--coordination_mode', s['coordination_mode'], '--repair_rounds', str(s['repair_rounds']),
                        '--scope_mode', s['scope_mode'], '--review_strategy', s['review_strategy']]
        if args.environment == 'cube':
            command += ['--grid_size', str(args.grid_size), '--block_specs', args.block_specs]
        else:
            command += ['--checkpoint_every', str(args.checkpoint_every)]
        run = {'run_id': label, 'directory': str(directory), 'command': command,
               'environment': args.environment, 'variant': tag, 'condition': condition, 'seed': seed,
               'repeat_id': repeat, 'reviewer': s['reviewer'], 'reviewer_id': args.reviewer_id,
               'scope_max_agents': k, 'review_max_agents': args.review_max_agents,
               'scope_selection': s['scope_selection'], 'evidence_mode': s['evidence_mode'],
               'pause_mode': s['pause_mode'], 'revalidation': s['revalidation'],
               'human_requests': args.human_requests, 'review_trigger': args.review_trigger,
               'review_cooldown': args.review_cooldown, 'stall_steps': args.stall_steps,
               'stall_fraction': args.stall_fraction, 'step_limit': args.step_num}
        if profile:
            run.update(profile_record(profile, settings_file))
        matrix.append(run)
    return matrix


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def json_lines(path):
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                yield json.loads(line)


def milestone_steps(directory, step_limit):
    """First step of each Crafter milestone from run.py's all_rounds_stats.csv."""
    path = Path(directory) / 'all_rounds_stats.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    row = rows[-1]
    steps = {name: float(row[name]) if row.get(name) not in (None, '') else None for name in MILESTONES}
    diamond = steps['Collect diamond']
    result = {'milestones_reached': sum(value is not None for value in steps.values()),
              'diamond_step': diamond, 'diamond_censored': diamond is None,
              # Censored at the step limit when no diamond was collected.
              'diamond_step_or_limit': diamond if diamond is not None else step_limit}
    result.update({'step_' + name.lower().replace(' ', '_'): value for name, value in steps.items()})
    return result


def summarize_run(directory, step_limit=None):
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
    applied = [event for event in events if event['event'] == 'plan_applied']
    skipped = Counter(event.get('reason') for event in events if event['event'] == 'review_skipped')
    row = {key: config.get(key) for key in (
        'environment', 'condition', 'seed', 'repeat_id', 'reviewer', 'reviewer_id', 'scope_max_agents',
        'review_max_agents', 'scope_selection', 'human_requests', 'review_trigger', 'review_cooldown', 'stall_steps', 'stall_fraction',
        'evidence_mode', 'pause_mode',
        'revalidation', 'source_sha256', 'git_head',
    )}
    models = config.get('models') or {}
    row['config_chat_model'] = models.get('LLM_CHAT_MODEL') or models.get('QWEN_CHAT_MODEL') \
        or models.get('OPENAI_MODEL')
    row['config_model_profile'] = models.get('MODEL_PROFILE')
    row.update({key: summary.get(key) for key in (
        'total_steps', 'task_success', 'termination_reason', 'requests_issued', 'human_seconds', 'commits',
        'paused_agent_steps', 'team_score', 'total_reward',
        'validation_seconds', 'review_preparation_seconds',
    )})
    outcomes = summary.get('outcomes', [])
    edited = [len(event.get('edited_agents') or {edit['agent_id'] for edit in event['edits']})
              for event in applied]
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
        'mean_edited_agents': sum(edited) / len(edited) if edited else None,
        'interrupted_plans': sum(len(event.get('interrupted_agents', [])) for event in applied),
        'over_review_capacity': skipped.get('over_review_capacity', 0),
        **{f'requests_{signal}': sum(any(signal in found for found in review.get('trigger_signals', {}).values())
                                     for review in reviews) for signal in ('choice', 'unknown', 'stalled')},
        'not_critical_skips': skipped.get('not_critical', 0),
        'planning_failures': sum(event['event'] == 'planning_failed' for event in events),
        **read_json(directory / 'model_usage.json'),
    })
    row.update({f'reply_{status}': count for status, count in
                Counter(reply['status'] for reply in replies).items()})
    row.update({f'validation_{key}': value for key, value in summary.get('validation_counts', {}).items()})
    row.update(milestone_steps(directory, step_limit))
    return row


def export_rows(output, rows):
    write_json(output / 'episodes.json', rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--environment', choices=['crafter', 'cube'], default='crafter')
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[4, 5, 6])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--suite', choices=['main', 'ablation', 'capacity', 'evidence', 'waiting'], default='main')
    parser.add_argument('--model_profiles', nargs='+', default=None,
                        help='Named profiles from config/model_profiles.json; default: the current environment.')
    parser.add_argument('--agent_num', type=int, default=6)
    parser.add_argument('--step_num', type=int, default=350)
    parser.add_argument('--scope_max_agents', type=positive_int, default=None,
                        help='Repair search cap; default none. The capacity suite compares 1, 2 and none.')
    parser.add_argument('--review_max_agents', type=positive_int, default=2,
                        help='Most agents one human review may open.')
    parser.add_argument('--human_requests', type=int, default=None,
                        help='Optional cap on review requests per episode. Default: no cap.')
    parser.add_argument('--review_trigger', choices=['signals', 'stage', 'all'], default='signals',
                        help="'signals' asks at choice, unknown or stalled moments; 'stage' uses the fixed list; "
                             "'all' is for practice.")
    parser.add_argument('--review_cooldown', type=int, default=10,
                        help='Steps before the same problem may be asked about again.')
    parser.add_argument('--stall_fraction', type=float, default=0.1,
                        help='Share of the episode length without team progress before open problems count as critical.')
    parser.add_argument('--stall_steps', type=int, default=None,
                        help='Fixed stall window in steps; overrides --stall_fraction.')
    parser.add_argument('--pause_mode', choices=['global', 'local', 'none'], default='local')
    parser.add_argument('--evidence_mode', choices=['full', 'compact'], default='compact')
    parser.add_argument('--revalidation', choices=['incremental', 'full', 'none'], default='incremental')
    parser.add_argument('--reviewer', choices=['human', 'scripted'], default='human')
    parser.add_argument('--reviewer_id', default='', help='Recorded with every run of this batch.')
    parser.add_argument('--review_timeout', type=float, default=120)
    parser.add_argument('--review_step_seconds', type=float, default=1)
    parser.add_argument('--scripted_delay_steps', type=int, default=2)
    parser.add_argument('--grid_size', type=int, default=15)
    parser.add_argument('--block_specs', default='2:1,3:1')
    parser.add_argument('--results_dir', default='results/experiments')
    parser.add_argument('--checkpoint_every', type=int, default=5,
                        help='Crafter: save each episode every this many steps so it can resume (0 disables).')
    parser.add_argument('--resume', action='store_true',
                        help='Skip finished episodes, continue interrupted Crafter episodes from their '
                             'checkpoint, and rerun the rest; failed attempts are kept as <dir>.failedN.')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    return parser


def resume_action(run):
    """With --resume: 'done' (finished), 'resume' (has a checkpoint) or 'fresh'."""
    directory = Path(run['directory'])
    if read_json(directory / 'process.json').get('exit_code') == 0             and (directory / 'intervention_summary.json').exists():
        return 'done'
    if run['environment'] == 'crafter' and checkpoint_path(directory).exists():
        return 'resume'
    return 'fresh'


def set_aside(directory):
    """Keep a failed attempt without a checkpoint under <dir>.failedN instead of mixing its logs."""
    number = 1
    while directory.with_name(f'{directory.name}.failed{number}').exists():
        number += 1
    target = directory.with_name(f'{directory.name}.failed{number}')
    directory.rename(target)
    return target


def main():
    args = get_parser().parse_args()
    output = Path(args.results_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(args)
    if not args.summarize_only:
        write_json(output / 'matrix.json', matrix)
    if args.dry_run:
        for run in matrix:
            prefix = f"[{run['model_profile']}] " if run.get('model_profile') else ''
            print(prefix + subprocess.list2cmdline(run['command']))
        return
    if args.summarize_only:
        matrix = read_json(output / 'matrix.json')
    else:
        # Fail before the first run if any profile lacks a model name or key.
        for name in sorted({run['model_profile'] for run in matrix if run.get('model_profile')}):
            profile_env(name)
    rows = []
    for run in matrix:
        directory = Path(run['directory'])
        action = resume_action(run) if args.resume else 'fresh'
        if args.summarize_only:
            pass
        elif action == 'done':
            print(f"Skipping {run['run_id']} (finished)", flush=True)
        else:
            env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
            if run.get('model_profile'):
                env.update(profile_env(run['model_profile']))
            if action == 'fresh' and args.resume and directory.exists() and any(directory.iterdir()):
                print(f"Keeping the failed attempt as {set_aside(directory).name}", flush=True)
            directory.mkdir(parents=True, exist_ok=True)
            previous = read_json(directory / 'process.json') if action == 'resume' else {}
            command = run['command'] + (['--resume'] if action == 'resume' else [])
            print(f"{'Resuming' if action == 'resume' else 'Running'} {run['run_id']}", flush=True)
            started = perf_counter()
            with (directory / 'console.log').open('a' if action == 'resume' else 'w', encoding='utf-8') as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
            write_json(directory / 'process.json', {
                'exit_code': process.returncode,
                'process_seconds': previous.get('process_seconds', 0) + perf_counter() - started,
                'attempts': previous.get('attempts', 1) + 1 if action == 'resume' else 1,
            })
        row = {**summarize_run(directory, run.get('step_limit')), **run, **read_json(directory / 'process.json')}
        row.pop('command')
        row['episode_cost'] = episode_cost(row, row)
        rows.append(row)
        export_rows(output, rows)
    print(f"Saved {len(rows)} episodes to {output / 'episodes.csv'}")


if __name__ == '__main__':
    main()
