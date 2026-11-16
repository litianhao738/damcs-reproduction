"""Small controlled scope comparisons; not a task-success benchmark."""

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory_system.repair_scopes import generate_scope_candidates
from memory_system.coordination import Plan
from scripts.smoke_repair_scopes import RepairScopeChecks, plan


def related_scope(targets, capacity):
    # A transparent structural reference: cover complete related-agent groups.
    groups = [set(target["evidence_agents"]) for target in targets]
    scopes = set()
    for size in range(1, len(groups) + 1):
        for chosen in combinations(groups, size):
            scope = frozenset().union(*chosen)
            if len(scope) <= capacity:
                scopes.add(scope)
    return sorted(min(scopes, key=lambda scope: (
        -sum(group <= scope for group in groups), len(scope), tuple(sorted(scope)),
    ))) if scopes else []


def assess_range(report, scope):
    if not scope:
        return {"agents": [], "status": "no_selection", "verified_target_count": 0}
    summary = next(result for result in report["scope_results"] if set(result["agents"]) == set(scope))
    witnesses = [candidate for candidate in report["candidates"]
                 if set(candidate["edited_agents"]) <= set(scope)]
    return {"agents": scope, "status": summary["status"],
            "verified_target_count": max((len(candidate["covered_targets"]) for candidate in witnesses), default=0)}


def run_case(name):
    fixture = RepairScopeChecks()
    fixture.setUp()
    try:
        capacity = 1 if name == "provider_only" else 2
        plans = [plan(0, "make_wood_pickaxe"), plan(1), plan(2)]
        fixture.table(0)
        if name == "provider_only":
            fixture.players[1].inventory["wood"] = 1
        elif name == "joint_tool":
            fixture.players[0].inventory["wood_pickaxe"] = 1
            fixture.facing(1, "stone")
            plans = [plan(0), plan(1, "Navigator", "stone"), plan(2)]
        elif name == "two_providers":
            fixture.world[(9, 9)] = "furnace"
            fixture.players[0].inventory["iron"] = 1
            fixture.players[1].inventory["wood"] = 1
            fixture.players[2].inventory["coal"] = 1
            plans[0] = plan(0, "make_iron_pickaxe")
        elif name == "shared_unit":
            fixture.table(1)
            fixture.players[2].inventory["wood"] = 1
            plans[1] = plan(1, "make_wood_pickaxe")
        elif name == "six_agent_mixed":
            fixture.make_env(6)
            fixture.world[(9, 9)] = "furnace"
            fixture.table(1)
            fixture.facing(2, "diamond")
            for agent_id, stock in {
                0: {"iron": 1}, 3: {"wood": 2, "stone": 1},
                4: {"coal": 1, "iron_pickaxe": 1}, 5: {"wood_pickaxe": 1, "iron": 1},
            }.items():
                fixture.players[agent_id].inventory.update(stock)
            plans = [plan(0, "make_iron_pickaxe"), plan(1, "make_stone_pickaxe"),
                     plan(2, "Navigator", "diamond"), plan(3), plan(4), plan(5)]
        observations = fixture.observations()
        if name == "unknown_inventory":
            observations[1]["inventory"] = None
        report = generate_scope_candidates(plans, observations, max_scope_agents=capacity)
        owners = list(dict.fromkeys(target["agent_id"] for target in sorted(
            report["targets"], key=lambda target: (not target["immediate"], target["agent_id"]),
        )))[:capacity]
        selected = report["selected"]
        policies = {
            "problem_owner_order": assess_range(report, owners),
            "complete_related_groups": assess_range(report, related_scope(report["targets"], capacity)),
            "verified_scope": assess_range(report, selected["edited_agents"] if selected else []),
        }
        actual_events = []
        if selected:
            fixture.execute(plans, [Plan(**edit) for edit in selected["edits"]])
            for target in report["targets"]:
                if target["target_id"] not in selected["covered_targets"]:
                    continue
                event = target["kind"] + "_" + target["subject"]
                assert fixture.players[target["agent_id"]].achievements[event] > 0, target
                actual_events.append(target["target_id"])
        return {"case": name, "capacity": capacity, "analysis": report,
                "policies": policies, "real_crafter_verified_events": actual_events}
    finally:
        fixture.doCleanups()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/scope_examples_20261115")
    args = parser.parse_args()
    results = [run_case(name) for name in (
        "provider_only", "joint_tool", "two_providers", "shared_unit", "no_supply",
        "unknown_inventory", "six_agent_mixed",
    )]
    output = Path(args.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scope_examples.json").write_text(json.dumps({
        "purpose": "Controlled correctness and scope-selection examples; no human or task-success claims.",
        "source_sha256": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                          for name in ("memory_system/repair_scopes.py", "memory_system/coordination.py",
                                       "utils.py", "crafter/env.py", "crafter/objects.py", "crafter/data.yaml")},
        "cases": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        report = result["analysis"]
        print(json.dumps({
            "case": result["case"], "status": report["status"],
            "selected": report["selected"]["edited_agents"] if report["selected"] else None,
            "assignments": report["assignments_evaluated"],
            "elapsed_seconds": round(report["elapsed_seconds"], 4),
            "policies": result["policies"], "verified_events": result["real_crafter_verified_events"],
        }))


if __name__ == "__main__":
    main()
