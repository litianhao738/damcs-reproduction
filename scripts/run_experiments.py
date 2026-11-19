"""Sequential paired-seed experiment matrix and episode-level result export.

Conditions: A (DAMCS base), B (legality repair; progress problems recorded only),
C (adds automatic progress repair) and H (C with human review at critical moments).

Suites:
  main       conditions x seeds x repeats (tables 1 and 3)
  ablation   table 2: remove one module at a time from C, plus C with the LLM off (ignores --conditions)
  appendix   design choices inside each module (appendix table; ignores --conditions)
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
from memory_system.experiments import CONDITIONS, write_json
from memory_system.model_profiles import episode_cost, load_profiles, profile_env, profile_record
from memory_system.checkpoint import checkpoint_path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be at least 0')
    return number


# Table 2: remove one module at a time from C, the automatic full system. The rows
# "- progress repair" (= B) and "- human review" (H vs C) reuse table 1 and are not run
# here. C-no-legality turns the whole legality repair off, both replanning and the repair
# scope S: illegal plans are only recorded, as in A, and progress repair stays on. C-no-llm
# switches every agent's language model off: what the mechanisms do on their own.
ABLATIONS = [
    ('C-no-legality', 'custom', {'coordination_mode': 'diagnose', 'repair_rounds': 0, 'scope_mode': 'off',
                                 'review_strategy': 'verified', 'progress_mode': 'auto'}),
    ('C-no-llm', 'C', {'agent_llm': 'off'}),
]

# Appendix: design choices inside progress repair and human review, each changing one choice
# and nothing else. The repair scope S is an internal fallback of legality repair and has no
# row; scripts/evaluate_frozen_scopes.py can still measure it offline when replanning leaves
# legality problems unsolved.
APPENDIX = [
    ('C-no-commit', 'C', {'progress_commit': 'off'}),
    ('Hscript', 'H', {'reviewer': 'scripted'}),
    ('Hscript-global-wait', 'H', {'reviewer': 'scripted', 'pause_mode': 'global'}),
    ('Hscript-no-recheck', 'H', {'reviewer': 'scripted', 'revalidation': 'none'}),
]

# Crafter milestone columns written by run.py, in task order.
MILESTONES = ['Collect wood', 'Place table', 'Make wood pickaxe', 'Collect stone', 'Make stone pickaxe',
              'Collect iron', 'Collect coal', 'Place furnace', 'Make iron pickaxe', 'Collect diamond']


def variants(args):
    base = {'scope_max_agents': args.scope_max_agents, 'evidence_mode': args.evidence_mode,
            'pause_mode': args.pause_mode, 'revalidation': args.revalidation,
            'scope_selection': 'minimal', 'reviewer': args.reviewer, 'progress_commit': 'on',
            'agent_llm': 'on'}
    if args.suite in ('ablation', 'appendix'):
        rows = ABLATIONS if args.suite == 'ablation' else APPENDIX
        return [(tag, condition, {**base, **override}) for tag, condition, override in rows]
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
    selected = variants(args)
    if args.variants:
        unknown = set(args.variants) - {tag for tag, _, _ in selected}
        if unknown:
            raise ValueError(f"Unknown variants for suite {args.suite}: {', '.join(sorted(unknown))}")
        selected = [row for row in selected if row[0] in args.variants]
    repeats = range(args.repeat_offset, args.repeat_offset + args.repeats)
    matrix = []
    for profile, (tag, condition, s), seed, repeat in product(
            profiles, selected, args.seeds, repeats):
        k = s['scope_max_agents']
        parts = ([profile] if profile else []) + [
            tag, f'seed{seed}', f'r{repeat}', f'k{"all" if k is None else k}',
            s['evidence_mode'], s['pause_mode'], s['revalidation']]
        label = '_'.join(parts + (['any'] if s['scope_selection'] == 'any' else [])
                         + (['nocommit'] if s['progress_commit'] == 'off' else [])
                         + (['nollm'] if s['agent_llm'] == 'off' else []))
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
                   '--progress_commit', s['progress_commit'], '--agent_llm', s['agent_llm'],
                   *(['--slow_margin', str(args.slow_margin)] if args.slow_margin is not None else []),
                   '--reviewer', s['reviewer'], '--reviewer_id', args.reviewer_id,
                   '--review_timeout', str(args.review_timeout),
                   '--review_step_seconds', str(args.review_step_seconds),
                   '--scripted_delay_steps', str(args.scripted_delay_steps),
                   '--results_dir', str(directory), '--no_plot']
        if condition == 'custom':
            command += ['--coordination_mode', s['coordination_mode'], '--repair_rounds', str(s['repair_rounds']),
                        '--scope_mode', s['scope_mode'], '--review_strategy', s['review_strategy'],
                        '--progress_mode', s['progress_mode']]
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
               'stall_fraction': args.stall_fraction, 'step_limit': args.step_num,
               'progress_commit': s['progress_commit'], 'agent_llm': s['agent_llm'],
               'slow_margin_arg': args.slow_margin}
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
    coordination = list(json_lines(directory / 'coordination.jsonl'))
    analyses = [entry['scope_analysis'] for entry in coordination if entry.get('scope_analysis') is not None]
    # Agents whose plan cannot be carried out this step: in the plans as first written,
    # and after the last replanning round (before any repair scope S is applied).
    illegal = [[{check['agent_id'] for check in round_.get('checks', [])
                 if check.get('immediate') and check.get('status') == 'violated'}
                for round_ in (entry['rounds'][0], entry['rounds'][-1])]
               for entry in coordination if entry.get('rounds')]
    analyses += [event['scope_analysis'] for event in events
                 if event['event'] == 'planning_completed' and event.get('scope_analysis') is not None]
    target_snapshots = [analysis for analysis in analyses if analysis['targets']]
    reviews = [read_json(path) for path in sorted((directory / 'reviews').glob('review_[0-9][0-9][0-9][0-9].json'))]
    # Evidence, witnesses and field permissions exist only for legality reviews.
    legal_reviews = [review for review in reviews if review.get('kind', 'legal') == 'legal']
    replies = [event for event in events if event['event'] == 'review_resolved']
    applied = [event for event in events if event['event'] == 'plan_applied']
    skipped = Counter(event.get('reason') for event in events if event['event'] == 'review_skipped')
    row = {key: config.get(key) for key in (
        'environment', 'condition', 'seed', 'repeat_id', 'agent_num', 'reviewer', 'reviewer_id', 'scope_max_agents',
        'review_max_agents', 'scope_selection', 'human_requests', 'review_trigger', 'review_cooldown', 'stall_steps', 'stall_fraction',
        'evidence_mode', 'pause_mode',
        'revalidation', 'source_sha256', 'git_head',
        'coordination_mode', 'repair_rounds', 'scope_mode', 'review_strategy', 'execution_guard',
    )}
    models = config.get('models') or {}
    row['config_chat_model'] = models.get('LLM_CHAT_MODEL') or models.get('QWEN_CHAT_MODEL') \
        or models.get('OPENAI_MODEL')
    row['config_input_mode'] = models.get('LLM_INPUT_MODE')
    row['config_model_profile'] = models.get('MODEL_PROFILE')
    row['config_agent_llm'] = config.get('agent_llm')
    row['config_progress_mode'] = config.get('progress_mode')
    row['config_progress_commit'] = config.get('progress_commit')
    row.update({key: summary.get(key) for key in (
        'total_steps', 'task_success', 'termination_reason', 'requests_issued', 'human_seconds', 'commits',
        'paused_agent_steps', 'team_score', 'total_reward',
        'validation_seconds', 'review_preparation_seconds',
    )})
    outcomes = summary.get('outcomes', [])
    edited = [len(event.get('edited_agents') or {edit['agent_id'] for edit in event['edits']})
              for event in applied]
    usage_path = directory / 'model_usage.json'
    usage = read_json(usage_path)
    if usage_path.is_file() and config.get('agent_llm') == 'off':
        for counter in ('chat_requests', 'embedding_requests', 'chat_input_tokens',
                        'chat_output_tokens', 'embedding_tokens'):
            usage.setdefault(counter, 0)
    row.update({
        'run_dir': str(directory), 'completed': bool(summary),
        'illegal_plan_steps': sum(bool(first) for first, _ in illegal),
        'illegal_plans_first': sum(len(first) for first, _ in illegal),
        'illegal_plans_after_replan': sum(len(last) for _, last in illegal),
        'target_snapshots': len(target_snapshots),
        'selected_snapshots': sum(analysis['selected'] is not None for analysis in target_snapshots),
        'assignments_evaluated': sum(analysis['assignments_evaluated'] for analysis in analyses),
        'search_seconds': sum(analysis['elapsed_seconds'] for analysis in analyses),
        'reviews_with_verified_witness': sum(review['evidence']['sufficient'] for review in legal_reviews),
        'review_agent_slots': sum(len(review['edit_agents']) for review in reviews),
        'evidence_groups_before': sum(review['evidence']['fact_groups_before'] for review in legal_reviews),
        'evidence_groups_after': sum(review['evidence']['fact_groups_after'] for review in legal_reviews),
        'real_human_replies': sum(reply.get('provider') == 'human_web' for reply in replies),
        'ai_external_replies': sum(reply.get('provider') == 'ai_external' for reply in replies),
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
        **usage,
    })
    row.update({f'reply_{status}': count for status, count in
                Counter(reply['status'] for reply in replies).items()})
    row.update({f'validation_{key}': value for key, value in summary.get('validation_counts', {}).items()})
    # Progress repair (legal but slow plans).
    row['requests_legal'] = sum(review.get('kind', 'legal') == 'legal' for review in reviews)
    row['requests_progress'] = sum(review.get('kind') == 'progress' for review in reviews)
    row.update({key: summary.get(key) for key in (
        'progress_mode', 'progress_commit', 'slow_margin', 'progress_target_steps', 'progress_commits',
        'progress_exact_commits', 'progress_exact_success', 'progress_tasks', 'progress_tasks_completed',
        'stall_steps_total', 'committed_agent_steps', 'progress_seconds')})
    for field, prefix in (('progress_targets_by_type', 'progress_targets_'),
                          ('progress_commits_by_template', 'progress_commits_'),
                          ('progress_commits_by_source', 'progress_source_'),
                          ('progress_task_releases', 'progress_release_'),
                          ('progress_replies', 'progress_reply_')):
        row.update({prefix + key: value for key, value in (summary.get(field) or {}).items()})
    errors = summary.get('progress_estimate_errors') or []
    row['progress_estimate_error_mean'] = sum(errors) / len(errors) if errors else None
    row['progress_estimate_error_median'] = sorted(errors)[len(errors) // 2] if errors else None
    row['progress_exact_success_rate'] = (summary['progress_exact_success'] / summary['progress_exact_commits']
                                          if summary.get('progress_exact_commits') else None)
    row['progress_task_completion_rate'] = (summary['progress_tasks_completed'] / summary['progress_tasks']
                                            if summary.get('progress_tasks') else None)
    # Table 1's repair success rate: legality repairs (scope S) and progress repairs
    # (exact one-step edits and locked tasks) that reached their target, over all submitted.
    attempts = len(outcomes) + (summary.get('progress_exact_commits') or 0) + (summary.get('progress_tasks') or 0)
    successes = (sum(outcome['recovered'] for outcome in outcomes) + (summary.get('progress_exact_success') or 0)
                 + (summary.get('progress_tasks_completed') or 0))
    row['repair_success_rate'] = successes / attempts if attempts else None
    row['repair_attempts'] = attempts
    row['repair_successes'] = successes
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
    parser.add_argument('--repeats', type=positive_int, default=1)
    parser.add_argument('--repeat_offset', type=nonnegative_int, default=0,
                        help='Start repeat IDs here for separately scheduled runs.')
    parser.add_argument('--suite', choices=['main', 'ablation', 'appendix', 'capacity', 'evidence', 'waiting'],
                        default='main')
    parser.add_argument('--variants', nargs='+', help='Run only the named variants within the selected suite.')
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
    parser.add_argument('--slow_margin', type=positive_int, default=None,
                        help='Steps an option must save before a milestone counts as clearly slow '
                             '(default: 3%% of --step_num, at least 2).')
    parser.add_argument('--pause_mode', choices=['global', 'local', 'none'], default='local')
    parser.add_argument('--evidence_mode', choices=['full', 'compact'], default='compact')
    parser.add_argument('--revalidation', choices=['incremental', 'full', 'none'], default='incremental')
    parser.add_argument('--reviewer', choices=['human', 'scripted', 'ai'], default='human')
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
    if len(set(args.seeds)) != len(args.seeds):
        sys.exit('Seeds must be distinct within a batch.')
    try:
        matrix = build_matrix(args)
    except ValueError as error:
        sys.exit(str(error))
    if not args.summarize_only:
        matrix_path = output / 'matrix.json'
        if args.resume and matrix_path.is_file() and read_json(matrix_path) != matrix:
            sys.exit(f'{matrix_path}: settings changed. Use the original command or start a new batch.')
        write_json(matrix_path, matrix)
    if args.dry_run:
        for run in matrix:
            prefix = f"[{run['model_profile']}] " if run.get('model_profile') else ''
            print(prefix + subprocess.list2cmdline(run['command']))
        return
    if args.summarize_only:
        matrix = read_json(output / 'matrix.json')
    else:
        # Fail before the first run if any profile lacks a model name or key.
        model_envs = {name: profile_env(name) for name in sorted({
            run['model_profile'] for run in matrix if run.get('model_profile') and run['agent_llm'] == 'on'
        })}
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
            if run['agent_llm'] == 'on' and run.get('model_profile'):
                env.update(model_envs[run['model_profile']])
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
        row['episode_cost'] = (0.0 if row['agent_llm'] == 'off' and all(row.get(counter) == 0 for counter in (
            'chat_requests', 'embedding_requests', 'chat_input_tokens', 'chat_output_tokens', 'embedding_tokens',
        )) else episode_cost(row, row))
        rows.append(row)
        export_rows(output, rows)
    print(f"Saved {len(rows)} episodes to {output / 'episodes.csv'}")


if __name__ == '__main__':
    main()
