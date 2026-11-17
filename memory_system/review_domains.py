"""The two concrete environment adapters used by plan review."""

from .repair_scopes import build_repair_targets, generate_scope_candidates, verify_joint_repair
from .coordination import diagnose
from .repair_scopes import _JointModel
from .review_evidence import (
    build_review_evidence, dependency_agents, evidence_facts, snapshot_to_json, snapshot_from_json,
)


# Fixed stage rule, kept as the optional ``--review_trigger stage``: the last stage of the
# diamond chain (no iron pickaxe, no tool for iron or diamond). The default rule uses
# environment-independent signals instead (see InterventionController).
CRAFTER_CRITICAL_TARGETS = frozenset({"make:iron_pickaxe", "collect:iron", "collect:diamond"})


class CrafterAdapter:
    name = 'crafter'

    @staticmethod
    def critical(target):
        return f"{target['kind']}:{target['subject']}" in CRAFTER_CRITICAL_TARGETS

    @staticmethod
    def progress(info):
        """Team progress for the stall signal: how many distinct achievements any agent has unlocked.

        Repeating an achievement (collecting more wood) is not progress.
        """
        observations = info.values() if isinstance(info, dict) else info
        return len({name for obs in observations for name, count in obs['achievements'].items() if count > 0})
    generate = staticmethod(generate_scope_candidates)
    verify = staticmethod(verify_joint_repair)
    dependencies = staticmethod(dependency_agents)
    evidence = staticmethod(build_review_evidence)
    facts = staticmethod(evidence_facts)
    snapshot = staticmethod(snapshot_to_json)
    restore = staticmethod(snapshot_from_json)

    def targets(self, plans, observations):
        return build_repair_targets(plans, observations, diagnose(plans, observations))

    def moves(self, plans, observations):
        return _JointModel(plans, observations).navigator_actions

    def target_met(self, target, before, after):
        key = target.kind + '_' + target.subject
        return after[target.agent_id]['achievements'].get(key, 0) > before[target.agent_id]['achievements'].get(key, 0)

    def success(self, info):
        return any(obs['achievements'].get('collect_diamond', 0) > 0 for obs in info)


def get_review_adapter(environment):
    if environment == 'cube':
        from .cube_adapter import CubeAdapter
        return CubeAdapter()
    return CrafterAdapter()
