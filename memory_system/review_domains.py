"""The two concrete environment adapters used by plan review."""

from .repair_scopes import build_repair_targets, generate_scope_candidates, verify_joint_repair
from .coordination import diagnose
from .progress_repair import CrafterProgress, achieved_milestones, crafter_task_plans, milestone_count, \
    milestone_target
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
        """Team progress for the stall signal: how many of the ten diamond-chain milestones are reached.

        Side achievements (drinking, eating, waking up) and repeats are not progress.
        """
        return milestone_count(info)

    # Progress problems (legal but slow plans); see progress_repair.
    task_plans = staticmethod(crafter_task_plans)
    milestone_target = staticmethod(milestone_target)

    @staticmethod
    def milestone_reached(milestone, info):
        return milestone in achieved_milestones(info)

    @staticmethod
    def exact_effect(option, before, after):
        return option["milestone"] in achieved_milestones(after)

    @staticmethod
    def goals(agents, responses):
        """Each agent's current goal from its latest planning record."""
        replies = {response["id"]: response for response in responses}
        goals = {}
        for agent in agents:
            thoughts = replies.get(agent.id, {}).get("thoughts") or getattr(agent, "thoughts", None)
            goal = getattr(getattr(thoughts, "goal", None), "current_goal", None)
            if goal is not None:
                goals[agent.id] = getattr(goal, "value", goal)
        return goals

    @staticmethod
    def progress_analysis(plans, observations, goals, free_agents, slow_margin, stalled=False,
                          skip=(), only=None, force=False):
        return CrafterProgress(plans, observations, goals, free_agents, slow_margin).analyse(
            stalled, skip, only, force)

    @staticmethod
    def apply_plan(agent, plan):
        from .agent import ActionStatus  # imported here: the agent module loads the vision stack
        agent.update_current_skill(plan.operation, plan.material, plan.item, plan.recipient)
        agent.update_action_status(ActionStatus.IN_PROGRESS)

    @staticmethod
    def release_agent(agent):
        from .agent import ActionStatus
        # Plan afresh at the next step.
        agent.update_action_status(ActionStatus.DONE)

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
