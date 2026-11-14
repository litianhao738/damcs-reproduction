"""MA-Crafter cooperation checks and bounded, targeted replanning."""

import asyncio
from dataclasses import asdict, dataclass
from enum import Enum
import json

from crafter import constants
from crafter.env import NONE_EXCHANGE_ITEMS


MATERIAL_NAMES = [None, *constants.materials]


def _value(value):
    return value.value if isinstance(value, Enum) else value


@dataclass(frozen=True)
class Plan:
    agent_id: int
    operation: str
    material: str
    item: str
    recipient: int

    @classmethod
    def from_agent(cls, agent):
        return cls(
            agent.id, _value(agent.op), _value(agent.rss_to_collect),
            _value(agent.rss_to_share), agent.target_agent_id,
        )

    @classmethod
    def from_response(cls, agent_id, response):
        action = response.action
        return cls(
            agent_id, action.final_next_action.value,
            action.final_target_material_to_collect.value,
            action.final_target_material_to_share.value,
            action.final_target_agent_id,
        )


@dataclass(frozen=True)
class ConstraintCheck:
    agent_id: int
    rule: str
    status: str
    message: str
    related_agents: tuple[int, ...]
    immediate: bool = True


def _cell(observation, position):
    grid = observation.get("semantic")
    if grid is None or position is None:
        return None, None
    x, y = position
    if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
        return "outside", False
    cell = int(grid[x, y])
    if cell >= len(MATERIAL_NAMES):
        return None, True
    return MATERIAL_NAMES[cell], False if cell else None


def _front(observation):
    position = observation.get("player_pos")
    facing = observation.get("player_facing")
    if position is None or facing is None:
        return None, None, None
    target = (int(position[0] + facing[0]), int(position[1] + facing[1]))
    material, occupied = _cell(observation, target)
    return target, material, occupied


def _nearby(observation):
    grid = observation.get("semantic")
    position = observation.get("player_pos")
    if grid is None or position is None:
        return None
    x, y = position
    # Match World.nearby(..., 1), including diagonal crafting facilities.
    return {
        MATERIAL_NAMES[int(cell)]
        for cell in grid[x - 1:x + 2, y - 1:y + 2].flat
        if int(cell) < len(MATERIAL_NAMES)
    }


def diagnose(plans, observations):
    """Check submitted plans against submitted observations, without a simulator.

    A supplied inventory is complete (omitted items mean zero); an absent
    inventory or semantic observation is unknown. Valid planned shares occur
    before primitive actions, matching AgentActionProcessor's execution order.
    """
    checks = []
    inventories = {
        agent_id: dict(obs["inventory"]) if obs.get("inventory") is not None else None
        for agent_id, obs in observations.items()
    }
    uncertain_incoming = set()

    def add(plan, rule, status, message, related=(), immediate=True):
        checks.append(ConstraintCheck(
            plan.agent_id, rule, status, message,
            tuple(dict.fromkeys((plan.agent_id, *related))), immediate,
        ))

    def requirements(plan, required, label, immediate=True):
        inventory = inventories[plan.agent_id]
        for item, count in required.items():
            if inventory is None:
                add(plan, "inventory", "unknown",
                    f"{label}: inventory for {item} was not submitted.",
                    immediate=immediate)
                continue
            available = inventory.get(item, 0)
            if available >= count:
                add(plan, "inventory", "satisfied",
                    f"{label}: requires {count} {item}; {available} available.",
                    immediate=immediate)
            elif (plan.agent_id, item) in uncertain_incoming:
                add(plan, "inventory", "unknown",
                    f"{label}: needs {count} {item}; a planned incoming share is unverified.",
                    immediate=immediate)
            else:
                providers = [
                    other_id for other_id, stock in inventories.items()
                    if other_id != plan.agent_id and stock is not None
                    and stock.get(item, 0) > 0
                ]
                add(plan, "inventory", "violated",
                    f"{label}: requires {count} {item}; only {available} available.",
                    providers, immediate)

    # Sharing is a high-level side effect before env.step, in agent order.
    for plan in plans:
        if plan.operation != "share":
            continue
        valid_target = plan.recipient in observations and plan.recipient != plan.agent_id
        add(plan, "share_target", "satisfied" if valid_target else "violated",
            f"Share recipient {plan.recipient} "
            + ("is another team member." if valid_target else "must be another team member."))
        valid_item = plan.item in constants.items and plan.item not in NONE_EXCHANGE_ITEMS
        if not valid_item:
            add(plan, "share_item", "violated",
                f"{plan.item} is not a transferable inventory item.")
            continue
        inventory = inventories[plan.agent_id]
        if inventory is None:
            add(plan, "inventory", "unknown", f"Share: inventory for {plan.item} was not submitted.")
            if valid_target:
                uncertain_incoming.add((plan.recipient, plan.item))
            continue
        requirements(plan, {plan.item: 1}, "Share")
        if valid_target and inventory.get(plan.item, 0) >= 1:
            inventory[plan.item] -= 1
            recipient = inventories[plan.recipient]
            if recipient is not None:
                recipient[plan.item] = recipient.get(plan.item, 0) + 1

    # A teammate's proposed construction is not yet a confirmed facility.
    planned_facilities = []
    for plan in plans:
        if plan.operation not in ("place_table", "place_furnace"):
            continue
        target, material, occupied = _front(observations[plan.agent_id])
        recipe = constants.place[plan.operation[6:]]
        inventory = inventories[plan.agent_id]
        if target is not None and material in recipe["where"] and occupied is False:
            if inventory is not None and all(
                inventory.get(item, 0) >= count for item, count in recipe["uses"].items()
            ):
                planned_facilities.append((plan.operation[6:], target, plan.agent_id))

    for plan in plans:
        operation = plan.operation
        observation = observations[plan.agent_id]
        if operation.startswith("make_") and operation[5:] in constants.make:
            recipe = constants.make[operation[5:]]
            requirements(plan, recipe["uses"], operation)
            nearby = _nearby(observation)
            for facility in recipe["nearby"]:
                if nearby is None:
                    add(plan, "crafting_facility", "unknown",
                        f"{operation}: nearby facilities were not observed.")
                elif facility in nearby:
                    add(plan, "crafting_facility", "satisfied",
                        f"{operation}: {facility} is within the required 3x3 neighborhood.")
                else:
                    position = observation["player_pos"]
                    pending = [
                        provider for name, target, provider in planned_facilities
                        if name == facility
                        and max(abs(target[0] - position[0]), abs(target[1] - position[1])) <= 1
                    ]
                    add(plan, "crafting_facility", "unknown" if pending else "violated",
                        f"{operation}: {facility} "
                        + ("is only planned, not yet observed." if pending else "is not nearby."),
                        pending)
        elif operation.startswith("place_") and operation[6:] in constants.place:
            recipe = constants.place[operation[6:]]
            requirements(plan, recipe["uses"], operation)
            _, material, occupied = _front(observation)
            if occupied is True:
                add(plan, "placement", "violated", f"{operation}: the target cell is occupied.")
            elif material is None or occupied is None:
                add(plan, "placement", "unknown", f"{operation}: the target cell was not observed.")
            else:
                allowed = material in recipe["where"]
                add(plan, "placement", "satisfied" if allowed else "violated",
                    f"{operation}: target material is {material}; allowed: {recipe['where']}.")
        elif operation == "do":
            _, material, occupied = _front(observation)
            if occupied is True:
                # Object interactions (combat, eating) have different effects.
                continue
            if material is None:
                add(plan, "collection", "unknown", "The facing material was not observed.")
            elif material in constants.collect:
                requirements(plan, constants.collect[material]["require"], f"Collect {material}")
            else:
                add(plan, "collection", "violated",
                    f"The do action cannot collect {material}; choose a supported material or operation.")
        elif operation == "Navigator":
            if plan.material in ("", "not_applicable"):
                add(plan, "navigation_target", "violated", "Navigator needs a target material.")
            elif plan.material in constants.collect:
                # Missing tools are a future dependency, not illegal movement.
                requirements(
                    plan, constants.collect[plan.material]["require"],
                    f"Future collection of {plan.material}", immediate=False,
                )
    return checks


def _repair_context(agent_id, base_context, checks, plans, observations, attempt, max_rounds,
                    previous_error):
    relevant = [
        asdict(check) for check in checks
        if check.status == "violated" and agent_id in check.related_agents
    ]
    team = [
        {
            "agent_id": plan.agent_id,
            "plan": asdict(plan),
            "submitted_inventory": observations[plan.agent_id].get("inventory"),
        }
        for plan in plans
    ]
    return (
        base_context
        + f"\n\n### Verified cooperation feedback: repair {attempt}/{max_rounds}\n"
        + json.dumps({
            "problems": relevant, "team": team, "previous_reply_error": previous_error,
        }, ensure_ascii=False)
        + "\nRevise your own next action to address these concrete problems. "
        "Inventory facts above describe the current snapshot; a plan is not an executed transfer. "
        "Coordinate with the listed providers where useful. Do not assume missing items already exist. "
        "Missing tools do not make navigation itself illegal. Keep working toward the task. "
        "Return the same complete structured planning record."
    )


async def coordinate(agents, observations, contexts, responses, replan, max_rounds=2, repair=True):
    """Replan affected agents on one fixed observation snapshot.

    replan(agent_id, context) returns a ResponseEvent. Only the final response
    is passed to experience consolidation by the existing thinking processor.
    """
    by_id = {response["id"]: dict(response) for response in responses}
    proposed = {
        agent_id for agent_id, response in by_id.items()
        if response["thoughts"] is not None
    }

    def current_plans():
        return [
            Plan.from_response(agent.id, by_id[agent.id]["thoughts"])
            if agent.id in by_id and by_id[agent.id]["thoughts"] is not None
            else Plan.from_agent(agent)
            for agent in agents
        ]

    history = []
    rounds = 0
    replan_errors = {}
    while True:
        plans = current_plans()
        checks = diagnose(plans, observations)
        affected = {
            agent_id
            for check in checks
            if check.status == "violated" and (check.immediate or check.agent_id in proposed)
            for agent_id in check.related_agents
            if agent_id in by_id
            and not observations[agent_id].get("sleeping", False)
            and not observations[agent_id].get("dead", False)
        }
        targets = sorted(affected) if repair and rounds < max_rounds else []
        history.append({
            "round": rounds,
            "plans": [asdict(plan) for plan in plans],
            "checks": [asdict(check) for check in checks],
            "replanned_agents": targets,
            "replan_errors": {},
        })
        if not targets:
            break
        prompts = [
            _repair_context(
                agent_id,
                contexts[agent_id] if agent_id in contexts else agents[agent_id].context,
                checks, plans, observations, rounds + 1, max_rounds,
                replan_errors.get(agent_id),
            )
            for agent_id in targets
        ]
        revised = await asyncio.gather(
            *(replan(agent_id, context) for agent_id, context in zip(targets, prompts)),
            return_exceptions=True,
        )
        replan_errors = {}
        for agent_id, response in zip(targets, revised):
            if isinstance(response, BaseException):
                # Keep the last valid candidate and let the remaining repair
                # budget address the error, as in the initial planning path.
                error = f"{type(response).__name__}: {response}"
                replan_errors[agent_id] = error
                by_id[agent_id]["error"] = error
            else:
                by_id[agent_id]["thoughts"] = response
                by_id[agent_id]["error"] = None
        history[-1]["replan_errors"] = replan_errors
        proposed.update(targets)
        rounds += 1

    blocked = {
        check.agent_id for check in checks
        if repair and check.status == "violated" and check.immediate
    }
    for agent in agents:
        agent.coordination_blocked = agent.id in blocked
        agent.constraint_checks = [asdict(check) for check in checks if check.agent_id == agent.id]
    for agent_id, response in by_id.items():
        response["blocked"] = agent_id in blocked
        response["constraints"] = [
            asdict(check) for check in checks if check.agent_id == agent_id
        ]
        response["repair_rounds"] = rounds

    return [by_id[response["id"]] for response in responses], history
