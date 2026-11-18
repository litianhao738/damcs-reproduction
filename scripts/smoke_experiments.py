"""Check paired conditions, snapshot replay, and episode-level cost exports."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import smoke_repair_scopes as fixtures
from scripts.evaluate_frozen_scopes import assess_snapshot
from scripts.run_experiments import build_matrix, get_parser, summarize_run, export_rows
from memory_system.experiments import apply_condition, experiment_config
from memory_system.intervention import write_json
from memory_system.review_evidence import snapshot_to_json, snapshot_from_json
from utils import get_arg_parser


class ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory(prefix='experiment_matrix_')
        self.addCleanup(self.output.cleanup)
        self.directory = Path(self.output.name)

    def test_paired_seed_matrix_uses_identical_shared_settings(self):
        args = get_parser().parse_args(['--seeds', '4', '5', '--repeats', '2', '--results_dir', self.output.name])
        rows = build_matrix(args)
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({row['run_id'] for row in rows}), 16)
        self.assertEqual({row['condition'] for row in rows}, {'A', 'B', 'C', 'H'})
        for condition in ('A', 'B', 'C', 'H'):
            self.assertEqual({(row['seed'], row['repeat_id']) for row in rows if row['condition'] == condition},
                             {(4, 0), (4, 1), (5, 0), (5, 1)})

    def test_waiting_ablation_contains_three_declared_controls(self):
        args = get_parser().parse_args(['--suite', 'waiting', '--conditions', 'H', '--seeds', '4'])
        commands = [row['command'] for row in build_matrix(args)]
        self.assertEqual({(cmd[cmd.index('--pause_mode') + 1], cmd[cmd.index('--revalidation') + 1]) for cmd in commands},
                         {('global', 'full'), ('none', 'full'), ('local', 'incremental')})

    def test_ablation_suite_removes_one_module_from_c(self):
        args = get_parser().parse_args(['--suite', 'ablation', '--seeds', '4', '--results_dir', self.output.name])
        rows = {row['variant']: row['command'] for row in build_matrix(args)}
        self.assertEqual(set(rows), {'C-no-legality', 'C-no-llm'})

        def flag(command, name):
            return command[command.index(name) + 1]

        def settings(command):
            return tuple(flag(command, name) for name in ('--coordination_mode', '--repair_rounds', '--scope_mode',
                                                          '--progress_mode'))

        # C itself: repair, 2 rounds, automatic legality and progress repair, model on. Without
        # legality repair, illegal plans are only recorded (as in A) and progress repair stays.
        self.assertEqual(settings(rows['C-no-legality']), ('diagnose', '0', 'off', 'auto'))
        self.assertEqual(flag(rows['C-no-legality'], '--agent_llm'), 'on')
        self.assertEqual((flag(rows['C-no-llm'], '--condition'), flag(rows['C-no-llm'], '--agent_llm')), ('C', 'off'))
        self.assertTrue(rows['C-no-llm'][rows['C-no-llm'].index('--results_dir') + 1].endswith('nollm'))

    def test_appendix_suite_changes_one_design_choice_per_row(self):
        args = get_parser().parse_args(['--suite', 'appendix', '--seeds', '4', '--results_dir', self.output.name])
        rows = {row['variant']: row['command'] for row in build_matrix(args)}
        self.assertEqual(set(rows), {'C-no-commit', 'Hscript', 'Hscript-progress-pause', 'Hscript-no-recheck'})

        def flag(command, name):
            return command[command.index(name) + 1]

        self.assertEqual((flag(rows['C-no-commit'], '--condition'), flag(rows['C-no-commit'], '--progress_commit')),
                         ('C', 'off'))
        base = rows['Hscript']
        self.assertEqual((flag(base, '--reviewer'), flag(base, '--pause_mode'), flag(base, '--revalidation'),
                          flag(base, '--evidence_mode')), ('scripted', 'local', 'incremental', 'compact'))
        self.assertEqual((flag(base, '--condition'), flag(base, '--progress_commit'), flag(base, '--progress_pause')),
                         ('H', 'on', 'off'))
        changed = {'Hscript-progress-pause': ('--progress_pause', 'on'),
                   'Hscript-no-recheck': ('--revalidation', 'none')}
        for variant, (name, value) in changed.items():
            with self.subTest(variant=variant):
                command = rows[variant]
                self.assertEqual(flag(command, name), value)
                differing = [option for option in ('--pause_mode', '--revalidation', '--evidence_mode',
                                                   '--reviewer', '--scope_selection', '--condition',
                                                   '--progress_commit', '--progress_pause')
                             if flag(command, option) != flag(base, option)]
                self.assertEqual(differing, [name])

    def test_model_profiles_repeat_the_matrix_and_record_prices(self):
        args = get_parser().parse_args(['--conditions', 'A', 'B', 'C', '--seeds', '4',
                                        '--model_profiles', 'qwen', 'deepseek', '--results_dir', self.output.name])
        rows = build_matrix(args)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row['model_profile'] for row in rows}, {'qwen', 'deepseek'})
        self.assertTrue(all(row['run_id'].startswith(row['model_profile'] + '_') for row in rows))
        self.assertTrue(all('input_price_per_million_tokens' in row for row in rows))
        self.assertTrue(all('KEY' not in json.dumps(row['command']) for row in rows))

    def test_crafter_milestones_cost_and_repair_counts_are_summarized(self):
        (self.directory / 'all_rounds_stats.csv').write_text(
            'Collect wood,Place table,Make wood pickaxe,Collect stone,Make stone pickaxe,Collect iron,'
            'Collect coal,Place furnace,Make iron pickaxe,Collect diamond\n'
            '2.0,7.0,8.0,17.0,26.0,70.0,122.0,129.0,,\n', encoding='utf-8')
        events = [
            {'event': 'plan_applied', 'edits': [], 'edited_agents': [1, 2], 'interrupted_agents': [2]},
            {'event': 'plan_applied', 'edits': [], 'edited_agents': [3], 'interrupted_agents': []},
            {'event': 'review_skipped', 'reason': 'over_review_capacity'},
            {'event': 'review_resolved', 'status': 'stale_plan'},
        ]
        (self.directory / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n', encoding='utf-8')

        def check(agent_id, status, immediate=True):
            return {'agent_id': agent_id, 'status': status, 'immediate': immediate}

        coordination = [
            # Two agents illegal at first; replanning fixes agent 1, agent 2 stays illegal.
            {'step': 1, 'scope_analysis': None, 'rounds': [
                {'checks': [check(1, 'violated'), check(2, 'violated'), check(3, 'violated', immediate=False)]},
                {'checks': [check(2, 'violated'), check(1, 'satisfied')]}]},
            # Only a future (not immediate) violation: not an illegal plan this step.
            {'step': 2, 'scope_analysis': None, 'rounds': [{'checks': [check(4, 'violated', immediate=False)]}]},
        ]
        (self.directory / 'coordination.jsonl').write_text('\n'.join(map(json.dumps, coordination)) + '\n',
                                                          encoding='utf-8')
        row = summarize_run(self.directory, step_limit=350)
        self.assertEqual((row['illegal_plan_steps'], row['illegal_plans_first'], row['illegal_plans_after_replan']),
                         (1, 2, 1))
        self.assertEqual((row['milestones_reached'], row['step_place_furnace']), (8, 129.0))
        self.assertIsNone(row['diamond_step'])
        self.assertTrue(row['diamond_censored'])
        self.assertEqual(row['diamond_step_or_limit'], 350)
        self.assertEqual((row['mean_edited_agents'], row['interrupted_plans'], row['over_review_capacity']),
                         (1.5, 1, 1))
        self.assertEqual((row['reply_stale_plan'], row['invalid_replies']), (1, 1))

    def test_resume_skips_finished_continues_checkpointed_and_sets_failures_aside(self):
        from scripts.run_experiments import resume_action, set_aside
        args = get_parser().parse_args(['--conditions', 'A', 'B', 'C', '--seeds', '4',
                                        '--results_dir', self.output.name])
        finished, interrupted, failed = build_matrix(args)
        self.assertIn('--checkpoint_every', finished['command'])
        for run in (finished, interrupted, failed):
            Path(run['directory']).mkdir(parents=True)
        write_json(Path(finished['directory']) / 'process.json', {'exit_code': 0})
        write_json(Path(finished['directory']) / 'intervention_summary.json', {})
        (Path(interrupted['directory']) / 'checkpoint.pkl').write_bytes(b'')
        write_json(Path(failed['directory']) / 'process.json', {'exit_code': 1})
        self.assertEqual([resume_action(run) for run in (finished, interrupted, failed)],
                         ['done', 'resume', 'fresh'])
        kept = set_aside(Path(failed['directory']))
        self.assertEqual(kept.name, Path(failed['directory']).name + '.failed1')
        self.assertTrue((kept / 'process.json').exists())
        self.assertFalse(Path(failed['directory']).exists())

    def test_condition_presets_and_config_capture_source_without_credentials(self):
        presets = {'A': ('diagnose', 0, 'off', 'verified', 'log'), 'B': ('repair', 2, 'auto', 'verified', 'log'),
                   'C': ('repair', 2, 'auto', 'verified', 'auto'), 'H': ('repair', 2, 'human', 'verified', 'human')}
        for condition, expected in presets.items():
            args = get_arg_parser().parse_args(['--condition', condition])
            apply_condition(args)
            self.assertEqual((args.coordination_mode, args.repair_rounds, args.scope_mode, args.review_strategy,
                              args.progress_mode), expected)
        config = experiment_config(args, 'crafter')
        self.assertTrue(config['execution_guard'])
        self.assertEqual(len(config['source_sha256']), 64)
        self.assertNotIn('local_config/qwen_api_key.csv', config['source_files'])
        self.assertTrue(all('KEY' not in name for name in config['models']))

    def test_export_preserves_failures_and_distinguishes_human_from_scripted(self):
        write_json(self.directory / 'config.json', {'condition': 'H', 'reviewer': 'scripted', 'seed': 4})
        write_json(self.directory / 'intervention_summary.json', {
            'task_success': False, 'total_steps': 20,
            'outcomes': [{'source': 'human', 'recovered': True}, {'source': 'auto', 'recovered': False}],
            'progress_exact_commits': 1, 'progress_exact_success': 1, 'progress_tasks': 3,
            'progress_tasks_completed': 2,
        })
        (self.directory / 'events.jsonl').write_text(json.dumps({
            'event': 'review_resolved', 'status': 'accepted', 'provider': 'scripted',
        }) + '\n', encoding='utf-8')
        row = summarize_run(self.directory)
        self.assertFalse(row['task_success'])
        self.assertEqual(row['real_human_replies'], 0)
        self.assertEqual((row['scripted_replies'], row['recoveries'], row['recovery_attempts']), (1, 1, 2))
        # Legality and progress repairs together: (1 + 1 + 2) of (2 + 1 + 3).
        self.assertAlmostEqual(row['repair_success_rate'], 4 / 6)
        export_rows(self.directory, [row, {'completed': False, 'exit_code': 1}])
        rows = json.loads((self.directory / 'episodes.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['exit_code'], 1)

    def test_frozen_snapshot_keeps_same_permissions_and_joint_resource_conflict(self):
        fixture = fixtures.RepairScopeChecks()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.table(0)
        fixture.table(1)
        fixture.players[2].inventory['wood'] = 1
        plans = [fixtures.plan(0, 'make_wood_pickaxe'), fixtures.plan(1, 'make_wood_pickaxe'), fixtures.plan(2)]
        snapshot = json.loads(json.dumps(snapshot_to_json(fixture.observations(), full=True)))
        restored = snapshot_from_json(snapshot)
        self.assertEqual(restored[0]['semantic'].tolist(), fixture.observations()[0]['semantic'].tolist())
        row = assess_snapshot({'step': 0, 'environment': 'crafter', 'plans': [asdict(p) for p in plans],
                               'snapshot': snapshot}, 2)
        self.assertEqual(len(row['independent']['claimed_targets']), 2)
        self.assertFalse(row['independent']['jointly_valid_within_capacity'])
        self.assertTrue(row['strategies']['verified']['has_witness'])
        self.assertEqual(row['strategies']['verified']['target_coverage'], 1)


if __name__ == '__main__':
    unittest.main()
