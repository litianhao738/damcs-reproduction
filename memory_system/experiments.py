"""Shared conditions, source provenance, and a labelled scripted reviewer."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .intervention import write_json


# (coordination_mode, repair_rounds, scope_mode, review_strategy, progress_mode).
# Each condition adds one thing to the previous one: replanning (B), verified legality
# repair (C), automatic progress repair for legal but slow plans (D), and a person
# choosing at critical moments (H). A, B and C record progress problems without acting.
# History: on 2026-11-18 the problem-owner review (H1) and the verified-scope review (H2)
# were merged into H, which keeps H2's review interface; the owner-versus-scope question
# moved to an offline comparison (scripts/evaluate_frozen_scopes.py). Older results
# named H2 correspond to H without progress repair.
CONDITIONS = {
    'A': ('diagnose', 0, 'off', 'verified', 'log'),
    'B': ('repair', 2, 'off', 'verified', 'log'),
    'C': ('repair', 2, 'auto', 'verified', 'log'),
    'D': ('repair', 2, 'auto', 'verified', 'auto'),
    'H': ('repair', 2, 'human', 'verified', 'human'),
}


def apply_condition(args):
    if args.condition != 'custom':
        (args.coordination_mode, args.repair_rounds, args.scope_mode, args.review_strategy,
         args.progress_mode) = CONDITIONS[args.condition]


def experiment_config(args, environment):
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root,
    ).decode('utf-8').split('\0')
    source = {}
    for name in sorted(set(paths) - {''}):
        path = root / name
        if path.suffix in ('.py', '.yml', '.yaml', '.txt', '.ps1', '.json', '.toml'):
            source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        **vars(args), 'environment': environment,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'source_sha256': hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        'source_files': source,
        # Model names and settings only; keys are never recorded.
        'models': {key: os.getenv(key) for key in (
            'MODEL_PROFILE', 'LLM_PROVIDER', 'OPENAI_MODEL', 'OPENAI_EMBEDDING_MODEL',
            'QWEN_CHAT_MODEL', 'QWEN_EMBEDDING_MODEL', 'LLM_CHAT_MODEL', 'LLM_BASE_URL',
            'EMBEDDING_MODEL', 'EMBEDDING_BASE_URL', 'LLM_JSON_MODE', 'LLM_GENERATE_KWARGS',
            'AZURE_OPENAI_CHAT_DEPLOYMENT', 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
        )},
        'execution_guard': True,
    }


class ScriptedReviewer:
    """Approve/reject after a fixed tick delay; never counted as human data.

    For a progress problem, 'approve' takes the recommended option and 'reject' skips
    (the automatic choice applies)."""

    def __init__(self, decision='approve', delay_steps=2):
        self.decision = decision
        self.delay_steps = delay_steps

    def poll(self, controller, step):
        pending = controller.pending
        if pending is None or step - pending['step'] < self.delay_steps:
            return
        path = controller.review_dir / f"{pending['id']}.response.json"
        if not path.exists():
            write_json(path, {
                'request_id': pending['id'], 'decision': self.decision,
                'read_seconds': 0, 'provider': 'scripted', 'delay_steps': self.delay_steps,
            })
