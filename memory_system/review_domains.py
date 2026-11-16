"""The two concrete environment adapters used by plan review."""

from .repair_scopes import build_repair_targets, generate_scope_candidates, verify_joint_repair
from .coordination import diagnose
from .repair_scopes import _JointModel
from .review_evidence import (
    build_review_evidence, dependency_agents, evidence_facts, snapshot_to_json, snapshot_from_json,
)


class CrafterAdapter:
    name = 'crafter'
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
