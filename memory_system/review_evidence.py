"""Concrete review facts and dependency-scoped waiting for Crafter plans."""

from dataclasses import asdict

import numpy as np

from crafter import constants
from .constants import world_ids_item
from .coordination import Plan, _front
from .repair_scopes import _JointModel


def snapshot_copy(observations):
    return {agent_id: {
        **obs,
        "inventory": dict(obs["inventory"]) if obs.get("inventory") is not None else None,
        "semantic": obs["semantic"].copy() if obs.get("semantic") is not None else None,
        "player_pos": tuple(map(int, obs["player_pos"])) if obs.get("player_pos") is not None else None,
        "player_facing": tuple(map(int, obs["player_facing"])) if obs.get("player_facing") is not None else None,
    } for agent_id, obs in observations.items()}


def _cell_key(position):
    return "cell:" + ":".join(map(str, position))


def action_footprint(plan, observation, primitive=None):
    reads, writes = set(), set()
    agent_id = plan.agent_id
    item = lambda name: f"stock:{agent_id}:{name}"
    position = observation.get("player_pos")
    front, material, _ = _front(observation)
    operation = primitive if plan.operation == "Navigator" else plan.operation
    if operation == "share":
        reads.add(item(plan.item))
        writes.update((item(plan.item), f"stock:{plan.recipient}:{plan.item}"))
    elif operation and operation.startswith("make_"):
        recipe = constants.make[operation[5:]]
        reads.update(item(name) for name in recipe["uses"])
        writes.update(reads | {item(operation[5:])})
        if position is not None:
            reads.update(_cell_key((position[0] + dx, position[1] + dy))
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    elif operation == "do":
        if front is not None:
            reads.add(_cell_key(front))
            writes.add(_cell_key(front))
        recipe = constants.collect.get(material, {})
        reads.update(item(name) for name in recipe.get("require", {}))
        writes.update(item(name) for name in recipe.get("receive", {}))
    elif operation and operation.startswith("place_"):
        recipe = constants.place[operation[6:]]
        reads.update(item(name) for name in recipe["uses"])
        writes.update(reads)
        if front is not None:
            reads.add(_cell_key(front))
            writes.add(_cell_key(front))
    elif operation and operation.startswith("move_") and position is not None:
        direction = {"move_left": (-1, 0), "move_right": (1, 0),
                     "move_up": (0, -1), "move_down": (0, 1)}[operation]
        destination = (position[0] + direction[0], position[1] + direction[1])
        reads.add(_cell_key(destination))
        writes.update((_cell_key(position), _cell_key(destination)))
    return reads, writes


def dependency_agents(plans, observations, edits, targets=()):
    """Close actual resource/cell dependencies, including canceled actions."""
    replacements = {plan.agent_id: plan for plan in edits}
    model = _JointModel(plans, observations)
    footprints = {}
    for plan in plans:
        reads, writes = action_footprint(plan, observations[plan.agent_id],
                                        model.navigator_actions.get(plan.agent_id))
        if plan.agent_id in replacements:
            extra_reads, extra_writes = action_footprint(replacements[plan.agent_id], observations[plan.agent_id])
            reads |= extra_reads
            writes |= extra_writes
        footprints[plan.agent_id] = (reads, writes)
    related = set(replacements) | {target.agent_id for target in targets}
    while True:
        expanded = set(related)
        for agent_id, (reads, writes) in footprints.items():
            if agent_id in related:
                continue
            if any(writes & (footprints[other][0] | footprints[other][1])
                   or reads & footprints[other][1] for other in related):
                expanded.add(agent_id)
        if expanded == related:
            return related
        related = expanded


def _local_grid(observation):
    grid, position = observation.get("semantic"), observation.get("player_pos")
    if grid is None or position is None:
        return None
    x, y = map(int, position)
    return {f"{xx},{yy}": int(grid[xx, yy])
            for xx in range(max(0, x - 1), min(grid.shape[0], x + 2))
            for yy in range(max(0, y - 1), min(grid.shape[1], y + 2))}


def evidence_facts(plans, observations, agents, navigator_actions=None):
    by_id = {plan.agent_id: plan for plan in plans}
    baseline = _JointModel(plans, observations, navigator_actions=navigator_actions).baseline
    facts = {}
    for agent_id in sorted(agents):
        obs = observations[agent_id]
        facts[f"agent:{agent_id}"] = {
            "plan": asdict(by_id[agent_id]),
            "position": list(map(int, obs["player_pos"])) if obs.get("player_pos") is not None else None,
            "facing": list(map(int, obs["player_facing"])) if obs.get("player_facing") is not None else None,
            "sleeping": bool(obs.get("sleeping", False)), "dead": bool(obs.get("dead", False)),
            "next_move": (navigator_actions or {}).get(agent_id),
            "protected_event": {'kind': baseline[agent_id].kind, 'subject': baseline[agent_id].subject,
                                'status': baseline[agent_id].status},
        }
        facts[f"inventory:{agent_id}"] = obs.get("inventory")
        facts[f"view:{agent_id}"] = _local_grid(obs)
    return facts


def build_review_evidence(plans, observations, targets, edits, domains, expected, mode="compact"):
    """Subset-minimal inventory/view groups relative to fixed plan background.

    Original protection obligations are fixed before deletion. Hiding a fact
    cannot silently waive an unedited agent's already-verified work.
    """
    related = dependency_agents(plans, observations, edits, targets)
    if not edits:
        related.update(agent_id for target in targets for agent_id in target.evidence_agents)
    model = _JointModel(plans, observations)
    facts = evidence_facts(plans, observations, related, model.navigator_actions)
    optional = [key for key in facts if not key.startswith("agent:")]
    kept = set(optional)

    def check(keys):
        masked = snapshot_copy(observations)
        for agent_id, obs in masked.items():
            if f"inventory:{agent_id}" not in keys:
                obs["inventory"] = None
            if f"view:{agent_id}" not in keys:
                obs["semantic"] = None
        trial = _JointModel(plans, masked, navigator_actions=model.navigator_actions)
        trial.baseline = model.baseline
        result = trial.verify(targets, edits, domains, protected_agents=related)
        return result.status == "repairable" and set(expected) <= set(result.covered_targets)

    sufficient = check(kept)
    if mode == "compact" and sufficient:
        removed = True
        while removed:
            removed = False
            for key in optional:
                if key in kept and check(kept - {key}):
                    kept.remove(key)
                    removed = True
    retained = {key: value for key, value in facts.items()
                if key.startswith("agent:") or key in kept}
    return {"mode": mode, "agents": sorted(related), "facts": retained,
            "sufficient": sufficient, "fact_groups_before": len(optional),
            "fact_groups_after": len(kept),
            "minimality": "deletion-minimal inventory/view groups; fixed plans, positions and permissions",
            "navigator_actions": model.navigator_actions,
            "material_names": world_ids_item}


def snapshot_to_json(observations, full=False):
    return {str(agent_id): {
        "inventory": obs.get("inventory"), "position": list(map(int, obs["player_pos"])),
        "facing": list(map(int, obs["player_facing"])), "sleeping": bool(obs.get("sleeping", False)),
        "dead": bool(obs.get("dead", False)), "shape": list(obs["semantic"].shape)
        if obs.get("semantic") is not None else None,
        "cells": ({f'{x},{y}': int(cell) for (x, y), cell in np.ndenumerate(obs['semantic'])}
                  if full and obs.get('semantic') is not None else _local_grid(obs)),
    } for agent_id, obs in observations.items()}


def snapshot_from_json(data):
    result = {}
    for key, value in data.items():
        grid = np.zeros(value["shape"], dtype=np.uint8) if value["shape"] else None
        for position, material in (value["cells"] or {}).items():
            grid[tuple(map(int, position.split(",")))] = material
        result[int(key)] = {"inventory": value["inventory"], "semantic": grid,
                            "player_pos": value["position"], "player_facing": value["facing"],
                            "sleeping": value["sleeping"], "dead": value["dead"]}
    return result
