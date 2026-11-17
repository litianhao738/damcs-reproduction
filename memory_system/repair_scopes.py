"""One-step, finite-domain repair scopes from submitted Crafter observations.

Witnesses are suggestions, never executed here. The projection models the
player action pass (shares first), not later autonomous-object/life dynamics.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import combinations, product
import json
from math import prod
from time import perf_counter

from crafter import constants
from .coordination import MATERIAL_NAMES, Plan, _cell, _front, diagnose
from .pathfinding import PathFinding


PLAN_FIELDS = ("operation", "material", "item", "recipient")
TRANSFERABLE = {"wood", "stone", "coal", "iron", "diamond",
                "wood_pickaxe", "stone_pickaxe", "iron_pickaxe"}


@dataclass(frozen=True)
class RepairTarget:
    target_id: str
    agent_id: int
    kind: str
    subject: str
    immediate: bool
    required: dict[str, int]
    evidence_agents: tuple[int, ...]


@dataclass
class ScopeCandidate:
    edits: tuple[Plan, ...]
    changed_fields: dict[int, tuple[str, ...]]
    status: str
    target_results: dict[str, str]
    covered_targets: tuple[str, ...]
    evidence_agents: tuple[int, ...]
    reasons: tuple[str, ...]

    def to_dict(self):
        return {**asdict(self), "edited_agents": sorted(self.changed_fields)}


def build_repair_targets(plans, observations, checks):
    """Fix the desired events before searching; no goal substitution."""
    targets = []
    for plan in plans:
        obs = observations[plan.agent_id]
        if obs.get("dead", False) or obs.get("sleeping", False):
            continue
        missing = [check for check in checks if check.agent_id == plan.agent_id
                   and check.rule == "inventory" and check.status != "satisfied"]
        if not missing:
            continue
        if plan.operation.startswith("make_") and plan.operation[5:] in constants.make:
            kind, subject = "make", plan.operation[5:]
            required = constants.make[subject]["uses"]
        elif plan.operation in ("do", "Navigator"):
            _, material, occupied = _front(obs)
            subject = plan.material if plan.operation == "Navigator" else material
            if subject not in constants.collect or not constants.collect[subject]["require"]:
                continue
            if occupied is True or (material is not None and material != subject):
                continue
            kind, required = "collect", constants.collect[subject]["require"]
        else:
            continue
        targets.append(RepairTarget(
            f"{plan.agent_id}:{kind}:{subject}", plan.agent_id, kind, subject,
            plan.operation != "Navigator", dict(required),
            tuple(sorted({agent_id for check in missing for agent_id in check.related_agents})),
        ))
    return targets


def _plan_key(plan):
    return (plan.agent_id, plan.operation, plan.material, plan.item, plan.recipient)


def _edit_domains(plans, observations, targets, editable_agents):
    items = set(item for target in targets for item in target.required)
    # Existing commitments may need a compensating transfer to stay feasible.
    for plan in plans:
        if plan.operation == "share":
            items.add(plan.item)
        elif plan.operation.startswith("make_") and plan.operation[5:] in constants.make:
            items.update(constants.make[plan.operation[5:]]["uses"])
        elif plan.operation.startswith("place_") and plan.operation[6:] in constants.place:
            items.update(constants.place[plan.operation[6:]]["uses"])
    items &= TRANSFERABLE
    inventories = [obs.get("inventory") for obs in observations.values()]
    # Shares precede all primitive actions: no item can be manufactured in time
    # for a share in this step. Known zero team supply can therefore be pruned.
    items = sorted(item for item in items if any(
        stock is None or stock.get(item, 0) > 0 for stock in inventories
    ))
    domains = {}
    for plan in plans:
        obs = observations[plan.agent_id]
        if plan.agent_id not in editable_agents or obs.get("sleeping", False) or obs.get("dead", False):
            continue
        options = {plan, replace(plan, operation="noop")}
        for target in targets:
            if target.agent_id == plan.agent_id:
                operation = "make_" + target.subject if target.kind == "make" else "do"
                options.add(replace(plan, operation=operation))
        for item in items:
            for recipient in observations:
                if recipient != plan.agent_id:
                    options.add(replace(plan, operation="share", item=item, recipient=recipient))
        domains[plan.agent_id] = tuple(sorted(options, key=_plan_key))
    return domains


@dataclass
class _Outcome:
    kind: str
    subject: str
    status: str
    evidence: set[int]
    reason: str = ""


class _JointModel:
    """Copy inventories and sparse cell changes; never clone or query an Env."""

    def __init__(self, plans, observations, navigator_actions=None):
        self.plans = plans
        self.observations = observations
        self.original = {plan.agent_id: plan for plan in plans}
        self.navigator_actions = {}
        for plan in plans:
            if plan.operation != "Navigator":
                continue
            if navigator_actions is not None:
                self.navigator_actions[plan.agent_id] = navigator_actions[plan.agent_id]
                continue
            obs = observations[plan.agent_id]
            grid, position = obs.get("semantic"), obs.get("player_pos")
            if grid is None or position is None or plan.material not in MATERIAL_NAMES:
                self.navigator_actions[plan.agent_id] = None
                continue
            # Derive only the next primitive move, using the same pathfinder
            # as the executor and the agent's already-submitted observation.
            pathfinder = PathFinding(
                {MATERIAL_NAMES.index(name) for name in constants.walkable},
                grid_size=grid.T.shape,
            )
            path = pathfinder.find(grid.T, (int(position[1]), int(position[0])),
                                   MATERIAL_NAMES.index(plan.material))
            self.navigator_actions[plan.agent_id] = constants.actions[path[0]] if path else "noop"
        self.baseline = self.project(plans)

    def project(self, plans):
        observations = self.observations
        stocks = {agent_id: dict(obs["inventory"]) if obs.get("inventory") is not None else None
                  for agent_id, obs in observations.items()}
        sources = {(agent_id, item): {agent_id} for agent_id, stock in stocks.items()
                   if stock is not None for item in stock}
        uncertain = set()
        cells = {}
        positions = {agent_id: tuple(map(int, obs["player_pos"]))
                     for agent_id, obs in observations.items()
                     if obs.get("player_pos") is not None and not obs.get("dead", False)}
        old_positions = dict(positions)
        uncertain_positions = {}
        unknown_locations = set()
        outcomes = {}
        checks = diagnose(plans, observations)
        held = {check.agent_id for check in checks
                if check.immediate and check.status == "violated"}

        def cell(agent_id, position):
            if position is None:
                return None, None, set()
            material, occupied = _cell(observations[agent_id], position)
            evidence = {agent_id}
            if position in cells:
                material, occupied, writers = cells[position]
                evidence |= writers
            for other_id, current in positions.items():
                if current == position:
                    return material, True, evidence | {other_id}
            if position in old_positions.values():
                # Semantic views hide the terrain under a player. A vacated
                # cell is free of that player but its terrain stays unknown.
                material, occupied = None, False
                evidence.update(i for i, pos in old_positions.items() if pos == position)
            possible_occupants = uncertain_positions.get(position, set()) | unknown_locations
            if possible_occupants and (material is None or material in constants.walkable + ["lava"]):
                occupied = None
                evidence |= possible_occupants
            return material, occupied, evidence

        def uncertain_move(agent_id, position, destinations):
            positions.pop(agent_id, None)
            if position is None:
                unknown_locations.add(agent_id)
            else:
                for target in (position, *destinations):
                    uncertain_positions.setdefault(target, set()).add(agent_id)

        def requirement(agent_id, needed):
            evidence = {agent_id}
            missing, unknown = False, False
            stock = stocks[agent_id]
            for item, amount in needed.items():
                evidence |= sources.get((agent_id, item), set())
                if stock is None or (agent_id, item) in uncertain:
                    unknown = True
                elif stock.get(item, 0) < amount:
                    missing = True
            return ("violated" if missing else "unknown" if unknown else "satisfied"), evidence

        def consume(agent_id, needed, gives):
            if stocks[agent_id] is None:
                return
            for item, amount in needed.items():
                stocks[agent_id][item] -= amount
            for item, amount in gives.items():
                stocks[agent_id][item] = min(constants.items[item]["max"],
                                           stocks[agent_id].get(item, 0) + amount)
                sources[agent_id, item] = {agent_id}

        # High-level transfers happen before any Player.update call.
        for plan in plans:
            if plan.operation != "share":
                continue
            agent_id = plan.agent_id
            subject = f"{plan.item}:{plan.recipient}"
            state, evidence = requirement(agent_id, {plan.item: 1})
            if agent_id in held:
                state = "violated"
            if state == "satisfied":
                stocks[agent_id][plan.item] -= 1
                if stocks[plan.recipient] is not None:
                    stocks[plan.recipient][plan.item] = stocks[plan.recipient].get(plan.item, 0) + 1
                sources.setdefault((plan.recipient, plan.item), {plan.recipient}).update(evidence)
            elif state == "unknown":
                uncertain.add((plan.recipient, plan.item))
                sources.setdefault((plan.recipient, plan.item), {plan.recipient}).update(evidence)
            outcomes[agent_id] = _Outcome("share", subject, state, evidence | {plan.recipient})

        for plan in plans:
            agent_id = plan.agent_id
            obs = observations[agent_id]
            operation = plan.operation
            if operation == "share":
                continue
            if agent_id in held or obs.get("dead", False):
                outcomes[agent_id] = _Outcome("held", "", "violated", {agent_id})
                continue
            if operation == "Navigator":
                operation = self.navigator_actions[agent_id]
            if obs.get("sleeping", False):
                energy = stocks[agent_id].get("energy", 0) if stocks[agent_id] is not None else None
                if energy is not None and energy < constants.items["energy"]["max"]:
                    operation = "sleep"
                elif energy is None and operation not in ("noop", "sleep"):
                    position = positions.get(agent_id)
                    if operation is None or operation.startswith("move_"):
                        adjacent = [(position[0] + dx, position[1] + dy)
                                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))] if position else []
                        uncertain_move(agent_id, position, adjacent)
                    elif operation == "do" or operation.startswith("place_"):
                        front, _, _ = _front(obs)
                        if front is not None:
                            cells[front] = (None, None, {agent_id})
                    outcomes[agent_id] = _Outcome("sleep", "unverified_wake", "unknown", {agent_id},
                                                 "Submitted energy does not establish whether the agent wakes.")
                    continue
            if operation in ("noop", "sleep"):
                outcomes[agent_id] = _Outcome("idle", "", "satisfied", {agent_id})
                continue
            position = positions.get(agent_id)
            if operation is None:
                destinations = [(position[0] + dx, position[1] + dy)
                                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))] if position else []
                uncertain_move(agent_id, position, destinations)
                outcomes[agent_id] = _Outcome("move", "unobserved", "unknown", {agent_id},
                                             "Next navigation move cannot be derived from the submitted observation.")
                continue
            facing = obs.get("player_facing")
            front = tuple(position[i] + int(facing[i]) for i in (0, 1)) \
                if position is not None and facing is not None else None
            material, occupied, evidence = cell(agent_id, front)
            kind, subject, state, reason = "unsupported", "", "unknown", "Unmodeled operation."
            if operation is not None and operation.startswith("make_"):
                kind, subject = "make", operation[5:]
                recipe = constants.make[subject]
                state, inventory_evidence = requirement(agent_id, recipe["uses"])
                evidence = inventory_evidence
                for facility in recipe["nearby"]:
                    nearby = [cell(agent_id, (position[0] + dx, position[1] + dy))
                              for dx in (-1, 0, 1) for dy in (-1, 0, 1)] if position else []
                    evidence |= set().union(*(entry[2] for entry in nearby))
                    names = {entry[0] for entry in nearby}
                    if facility not in names:
                        facility_state = "unknown" if not nearby or None in names else "violated"
                        if state != "violated":
                            state = facility_state
                if state == "satisfied":
                    consume(agent_id, recipe["uses"], {subject: recipe["gives"]})
                reason = "Crafting requires observed facilities and shared/projected stock."
            elif operation == "do":
                kind, subject = "collect", material or "unobserved"
                if occupied is True:
                    state, reason = "unknown", "Object interactions are outside the material model."
                    cells[front] = (None, None, {agent_id})
                elif occupied is None:
                    state, reason = "unknown", "Facing occupancy was not observed."
                    if front is not None:
                        cells[front] = (None, None, {agent_id})
                elif material in constants.collect:
                    recipe = constants.collect[material]
                    state, inventory_evidence = requirement(agent_id, recipe["require"])
                    evidence |= inventory_evidence
                    if recipe.get("probability", 1) < 1 and state != "violated":
                        state = "unknown"
                    if state == "satisfied":
                        consume(agent_id, {}, recipe["receive"])
                        cells[front] = (recipe["leaves"], False, {agent_id})
                    elif state == "unknown":
                        cells[front] = (None, None, {agent_id})
                    reason = "Material collection follows the declared tool and terrain rules."
                else:
                    state = "unknown" if material is None else "violated"
                    if state == "unknown" and front is not None:
                        cells[front] = (None, None, {agent_id})
                    reason = "Facing material is missing or no longer collectible."
            elif operation is not None and operation.startswith("place_"):
                kind, subject = "place", operation[6:]
                recipe = constants.place[subject]
                state, inventory_evidence = requirement(agent_id, recipe["uses"])
                evidence |= inventory_evidence
                if occupied is True or (material is not None and material not in recipe["where"]):
                    state = "violated"
                elif material is None or occupied is None:
                    if state != "violated":
                        state = "unknown"
                if state == "satisfied":
                    consume(agent_id, recipe["uses"], {})
                    cells[front] = (subject, False, {agent_id}) if recipe["type"] == "material" \
                        else (None, True, {agent_id})
                elif state == "unknown" and front is not None:
                    cells[front] = (None, None, {agent_id})
                reason = "Existing placements are projected to preserve their effects."
            elif operation is not None and operation.startswith("move_"):
                kind, subject = "move", operation
                direction = {"move_left": (-1, 0), "move_right": (1, 0),
                             "move_up": (0, -1), "move_down": (0, 1)}[operation]
                destination = tuple(position[i] + direction[i] for i in (0, 1)) if position else None
                terrain, blocked, evidence = cell(agent_id, destination)
                if blocked is True or (terrain is not None and terrain not in constants.walkable + ["lava"]):
                    state = "violated"
                elif terrain is None or blocked is None:
                    state = "unknown"
                else:
                    state = "satisfied"
                    positions[agent_id] = destination
                if state == "unknown":
                    uncertain_move(agent_id, position, [destination] if destination is not None else [])
                reason = "Only the already-planned next primitive move is projected."
            outcomes[agent_id] = _Outcome(kind, subject, state, evidence, reason)
        return outcomes

    def verify(self, targets, edits, domains, protected_agents=None):
        changed = {}
        for plan in edits:
            if plan.agent_id not in domains or plan not in domains[plan.agent_id]:
                return ScopeCandidate(tuple(edits), {}, "infeasible", {}, (), (),
                                      ("Edit is outside the declared finite permission domain.",))
            fields = tuple(name for name in PLAN_FIELDS
                           if getattr(plan, name) != getattr(self.original[plan.agent_id], name))
            if fields:
                changed[plan.agent_id] = fields
        replacements = {plan.agent_id: plan for plan in edits}
        proposed = [replacements.get(plan.agent_id, plan) for plan in self.plans]
        projected = self.project(proposed)
        reasons, evidence = [], set()
        preservation = "satisfied"
        for agent_id, before in self.baseline.items():
            if protected_agents is not None and agent_id not in protected_agents:
                continue
            if agent_id in changed or before.status != "satisfied" or before.kind == "idle":
                continue
            after = projected[agent_id]
            evidence |= before.evidence | after.evidence
            if after.status != "satisfied" or (before.kind, before.subject) != (after.kind, after.subject):
                if after.status != "unknown":
                    preservation = "violated"
                elif preservation != "violated":
                    preservation = "unknown"
                reasons.append(f"Unedited agent {agent_id}: {before.kind} {before.subject} is no longer verified.")
        target_results = {}
        for target in targets:
            outcome = projected[target.agent_id]
            evidence |= outcome.evidence | set(target.evidence_agents)
            same_event = (outcome.kind, outcome.subject) == (target.kind, target.subject)
            expected_operation = "make_" + target.subject if target.kind == "make" else "do"
            can_emit = next(plan for plan in proposed if plan.agent_id == target.agent_id).operation == expected_operation
            state = outcome.status if same_event else "unknown" if can_emit and outcome.status == "unknown" else "violated"
            target_results[target.target_id] = state
            if state != "satisfied":
                reasons.append(f"{target.target_id}: {state}; {outcome.reason or 'target event did not execute'}")
        covered = tuple(sorted(key for key, value in target_results.items() if value == "satisfied"))
        if preservation == "violated":
            status = "infeasible"
        elif preservation == "unknown":
            status = "unknown"
        elif covered:
            status = "repairable"
        else:
            status = "unknown" if "unknown" in target_results.values() else "infeasible"
        return ScopeCandidate(tuple(plan for plan in edits if plan.agent_id in changed), changed,
                              status, target_results, covered, tuple(sorted(evidence)), tuple(reasons))


def verify_joint_repair(plans, observations, targets, edits, allowed_domains=None,
                        navigator_actions=None, protected_agents=None):
    """Verify a specific joint witness; keep all unedited plans fixed."""
    domains = allowed_domains if allowed_domains is not None else _edit_domains(
        plans, observations, targets, set(observations),
    )
    return _JointModel(plans, observations, navigator_actions=navigator_actions).verify(
        targets, edits, domains, protected_agents=protected_agents,
    )


# Scopes up to this many agents are always searched completely, so review
# strategies that open a fixed small group still see every candidate inside it.
COMPLETE_SCOPE_AGENTS = 2
# Larger scopes are searched only while no witness covers every target, and
# only while the total number of joint checks stays within this budget
# (about 80 microseconds each in Crafter; 50k is roughly 4 seconds per step).
ASSIGNMENT_BUDGET = 50_000


class ScopeSearch:
    """Agent subsets by increasing size; ``max_scope_agents=None`` sets no cap.

    Selection prefers covering more targets, then editing fewer agents. Once a
    witness covers every target, a larger scope can only edit more agents, so
    deepening past ``COMPLETE_SCOPE_AGENTS`` stops there without changing the choice.
    """

    def __init__(self, domains, original, max_scope_agents=None, budget=None):
        self.agents = sorted(domains)
        self.options = {agent_id: sum(plan != original[agent_id] for plan in domains[agent_id])
                        for agent_id in self.agents}
        self.max_scope_agents = max_scope_agents
        self.limit = len(self.agents) if max_scope_agents is None else min(max_scope_agents, len(self.agents))
        self.budget = ASSIGNMENT_BUDGET if budget is None else budget
        self.evaluated = 0
        self.searched = 0
        self.stop = None

    def levels(self, covered_all):
        yield [()]
        self.evaluated += 1
        for size in range(1, self.limit + 1):
            level = list(combinations(self.agents, size))
            cost = sum(prod(self.options[agent_id] for agent_id in scope) for scope in level)
            if size > COMPLETE_SCOPE_AGENTS:
                if covered_all():
                    self.stop = "all_targets_covered"
                    return
                if self.evaluated + cost > self.budget:
                    self.stop = "assignment_budget"
                    return
            self.searched = size
            self.evaluated += cost
            yield level
        self.stop = "scope_limit"

    def report(self):
        return {"max_scope_agents": self.max_scope_agents, "searched_scope_agents": self.searched,
                "search_stop": self.stop}


SCOPE_SELECTIONS = ("minimal", "any")


def select_any_repair(candidates, targets):
    """Ablation: one verified witness chosen without preferring coverage or size.

    The choice is a hash of the state, so the same snapshot always gives the
    same pick and runs stay reproducible.
    """
    ordered = sorted((candidate for candidate in candidates if candidate.status == "repairable"
                      and candidate.changed_fields),
                     key=lambda candidate: tuple(_plan_key(plan) for plan in candidate.edits))
    if not ordered:
        return None
    state = {"targets": sorted(target.target_id for target in targets),
             "witnesses": [[_plan_key(plan) for plan in candidate.edits] for candidate in ordered]}
    digest = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def select_repair_scope(candidates, targets, max_scope_agents=None):
    priorities = {target.target_id: target.immediate for target in targets}

    def key(candidate):
        covered = candidate.covered_targets
        return (-sum(priorities[item] for item in covered),
                -sum(not priorities[item] for item in covered),
                len(candidate.changed_fields), sum(map(len, candidate.changed_fields.values())),
                tuple(_plan_key(plan) for plan in candidate.edits))

    eligible = [candidate for candidate in candidates if candidate.status == "repairable"
                and 0 < len(candidate.changed_fields)
                and (max_scope_agents is None or len(candidate.changed_fields) <= max_scope_agents)]
    return min(eligible, key=key) if eligible else None


def generate_scope_candidates(plans, observations, checks=None, max_scope_agents=None,
                              editable_agents=None, selection="minimal"):
    """Enumerate exact changed sets by increasing size, then summarize allowed supersets.

    ``selection="any"`` is the ablation: search every allowed scope (within the
    budget) and pick any verified witness instead of the fullest, smallest one.
    """
    if selection not in SCOPE_SELECTIONS:
        raise ValueError(f"selection must be one of {SCOPE_SELECTIONS}")
    started = perf_counter()
    checks = diagnose(plans, observations) if checks is None else checks
    targets = build_repair_targets(plans, observations, checks)
    editable = set(observations) if editable_agents is None else set(editable_agents)
    domains = _edit_domains(plans, observations, targets, editable) if targets else {}
    model = _JointModel(plans, observations) if targets else None
    original = {plan.agent_id: plan for plan in plans}
    exact, witnesses, scopes = {}, {}, []
    search = ScopeSearch(domains, original, max_scope_agents)
    all_targets = {target.target_id for target in targets}

    def covered_all():
        return selection == "minimal" and any(
            set(result.covered_targets) >= all_targets for result in witnesses.values())

    for level in search.levels(covered_all) if targets else []:
        for scope in level:
            scopes.append(scope)
            choices = [tuple(plan for plan in domains[agent_id] if plan != original[agent_id])
                       for agent_id in scope]
            counts = {"repairable": 0, "unknown": 0, "infeasible": 0}
            for edits in product(*choices):
                result = model.verify(targets, edits, domains)
                counts[result.status] += 1
                if result.status == "repairable":
                    signature = (result.covered_targets, tuple(sorted(result.changed_fields.items())))
                    witnesses.setdefault(signature, result)
            exact[scope] = counts
    candidates = list(witnesses.values())

    def dominates(left, right):
        return (left is not right and set(left.covered_targets) >= set(right.covered_targets)
                and set(left.changed_fields) <= set(right.changed_fields)
                and all(set(fields) <= set(right.changed_fields[agent_id])
                        for agent_id, fields in left.changed_fields.items()))

    minimal = [candidate for candidate in candidates
               if not any(dominates(other, candidate) for other in candidates)]
    summaries = []
    for scope in scopes if targets else []:
        counts = {key: sum(value[key] for subset, value in exact.items() if set(subset) <= set(scope))
                  for key in ("repairable", "unknown", "infeasible")}
        status = "repairable" if counts["repairable"] else "unknown" if counts["unknown"] else "infeasible"
        summaries.append({"agents": scope, "status": status, "assignments": sum(counts.values())})
    selected = select_repair_scope(minimal, targets, max_scope_agents) if selection == "minimal" \
        else select_any_repair(candidates, targets)
    if not targets:
        status, reason = "no_targets", "No outstanding inventory/tool goal is covered by the one-step model."
    elif selected is not None:
        status, reason = "repairable", "A joint witness verifies the listed covered targets and preserves unedited work."
    elif exact[()]["repairable"]:
        status, reason = "already_satisfied", "The projected target event needs no plan-field change."
    elif any(scope["status"] == "unknown" for scope in summaries):
        status, reason = "unknown", "Missing evidence or unmodeled effects prevent a verified recommendation."
    else:
        status, reason = "infeasible", "No witness exists in the declared finite domain and searched scopes."
    if status != "repairable" and search.stop == "assignment_budget":
        reason += " Larger scopes were not searched: assignment budget reached."
    return {
        "status": status, "reason": reason,
        "horizon": 1, **search.report(), "selection": selection,
        "model": "submitted-state player action pass; shares before primitives",
        "targets": [asdict(target) for target in targets],
        "domains": {agent_id: [asdict(plan) for plan in domain] for agent_id, domain in domains.items()},
        "scope_results": summaries,
        "candidates": [candidate.to_dict() for candidate in minimal],
        "selected": selected.to_dict() if selected is not None else None,
        "assignments_evaluated": sum(sum(counts.values()) for counts in exact.values()),
        "elapsed_seconds": perf_counter() - started,
    }
