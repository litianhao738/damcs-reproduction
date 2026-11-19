"""Progress problems in MA-Crafter: legal plans that leave the team's next milestone clearly slow.

Legality repair (``repair_scopes``) keeps each agent's own goal and edits one step.
Progress repair works on team milestones instead. Whoever reaches the next milestone
first counts, so an option may change who does it. Options come from a fixed set of
templates:

- ``supply``: teammates give the agent already working on the milestone what it lacks;
- ``substitute``: someone else makes or places it, with the materials handed over;
- ``build_here``: the maker places a new table or furnace where it stands;
- ``gather``: someone else is sent to collect the material.

An option that reaches the milestone in this very step is verified exactly with the
one-step joint model used by legality repair. Shares are applied before every other
action and ignore distance, so "hand the wood over and let the agent at the table make
it" can be exact. Anything that needs walking carries an estimated number of steps;
it runs as a locked task (see ``crafter_task_plans``) and is measured afterwards.
"""

from collections import Counter, deque
from dataclasses import asdict
import math

import numpy as np

from crafter import constants
from .coordination import MATERIAL_NAMES, Plan
from .repair_scopes import _JointModel


INF = math.inf
NA = "not_applicable"

# The diamond chain: the ten stages reported as milestone steps, with the stages each needs first.
CRAFTER_MILESTONES = {
    "collect_wood": ("collect", "tree", ()),
    "place_table": ("place", "table", ("collect_wood",)),
    "make_wood_pickaxe": ("make", "wood_pickaxe", ("place_table",)),
    "collect_stone": ("collect", "stone", ("make_wood_pickaxe",)),
    "make_stone_pickaxe": ("make", "stone_pickaxe", ("collect_stone",)),
    "collect_coal": ("collect", "coal", ("make_wood_pickaxe",)),
    "place_furnace": ("place", "furnace", ("collect_stone",)),
    "collect_iron": ("collect", "iron", ("make_stone_pickaxe",)),
    "make_iron_pickaxe": ("make", "iron_pickaxe", ("collect_iron", "collect_coal", "place_furnace")),
    "collect_diamond": ("collect", "diamond", ("make_iron_pickaxe",)),
}
MILESTONE_ORDER = tuple(CRAFTER_MILESTONES)


def _stages_after():
    """Longest chain of later stages each milestone gates (the diamond: 0, the stone pickaxe: 3)."""
    after = {}
    for name in reversed(MILESTONE_ORDER):
        later = [other for other in MILESTONE_ORDER if name in CRAFTER_MILESTONES[other][2]]
        after[name] = 1 + max((after[other] for other in later), default=-1)
    return after


# How critical a milestone is: a frontier milestone that gates more later stages lies on the critical path.
STAGES_AFTER = _stages_after()
# Inventory item received by collecting a material, and back.
COLLECTED_ITEM = {"tree": "wood", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond"}
SOURCE_OF = {item: material for material, item in COLLECTED_ITEM.items()}
IDLE_OPERATIONS = frozenset({"noop", "sleep"})
# Locked tasks stop when health falls to this level, so the agent can look after itself.
LOW_HEALTH = 3
# Options kept per problem (the review page lists them; the first is applied automatically).
LISTED_OPTIONS = 5
# A facility closer than this is simply walked to; placing a new one is not offered.
BUILD_HERE_MIN_STEPS = 3

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MOVES = {(-1, 0): "move_left", (1, 0): "move_right", (0, -1): "move_up", (0, 1): "move_down"}
_WALKABLE_IDS = [MATERIAL_NAMES.index(name) for name in constants.walkable]
_INT_MAX = np.iinfo(np.int32).max


def _as_list(info):
    return list(info.values()) if isinstance(info, dict) else list(info)


def achieved_milestones(info):
    """Milestones any agent has reached (``info``: a list or a dict of observations)."""
    observations = _as_list(info)
    return {name for name in CRAFTER_MILESTONES
            if any((obs.get("achievements") or {}).get(name, 0) > 0 for obs in observations)}


def milestone_count(info):
    return len(achieved_milestones(info))


def frontier(info):
    """Milestones not reached yet whose earlier stages are all done, in chain order."""
    done = achieved_milestones(info)
    return [name for name in MILESTONE_ORDER
            if name not in done and all(stage in done for stage in CRAFTER_MILESTONES[name][2])]


def requirements(milestone, build_here=False):
    """Items the executor must hold when it acts (consumed items and tools), and the facility it needs nearby."""
    kind, subject, _ = CRAFTER_MILESTONES[milestone]
    if kind == "make":
        recipe = constants.make[subject]
        needs, facility = dict(recipe["uses"]), recipe["nearby"][0]
        if build_here:
            for item, count in constants.place[facility]["uses"].items():
                needs[item] = needs.get(item, 0) + count
        return needs, facility
    if kind == "place":
        return dict(constants.place[subject]["uses"]), None
    return dict(constants.collect[subject]["require"]), None


def related_goals(milestone):
    """Goals that count as working toward ``milestone`` when nobody names it directly."""
    needs, facility = requirements(milestone)
    related = {milestone}
    related.update(f"collect_{item}" for item in needs if item in SOURCE_OF)
    if facility:
        related.add(f"place_{facility}")
    return related


def milestone_target(milestone):
    """The milestone in the shape of a legality problem, for the fixed ``stage`` list."""
    kind, subject, _ = CRAFTER_MILESTONES[milestone]
    return {"kind": kind, "subject": subject}


class Board:
    """Walking distances on the shared semantic map (indexed ``[x, y]``).

    This is the map the DAMCS navigator already plans on. Other players, animals and
    monsters block their cells; terrain changes are not predicted.
    """

    def __init__(self, observations):
        grid = next((obs["semantic"] for obs in observations.values() if obs.get("semantic") is not None), None)
        self.grid = None if grid is None else np.asarray(grid)
        self._distances = {}
        if self.grid is not None:
            self.width, self.height = self.grid.shape
            self._walkable = np.isin(self.grid, _WALKABLE_IDS).tolist()

    def inside(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def material(self, cell):
        """Material name of a cell; None outside the map or under a creature."""
        if self.grid is None or not self.inside(cell):
            return None
        value = int(self.grid[cell[0], cell[1]])
        return MATERIAL_NAMES[value] if value < len(MATERIAL_NAMES) else None

    def distances(self, start):
        """Steps from ``start`` to every walkable cell (-1 where unreachable)."""
        start = (int(start[0]), int(start[1]))
        if start not in self._distances:
            distance = [[-1] * self.height for _ in range(self.width)]
            distance[start[0]][start[1]] = 0
            walkable, queue = self._walkable, deque([start])
            while queue:
                x, y = queue.popleft()
                step = distance[x][y] + 1
                for dx, dy in _DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and distance[nx][ny] < 0 \
                            and walkable[nx][ny]:
                        distance[nx][ny] = step
                        queue.append((nx, ny))
            self._distances[start] = np.array(distance, dtype=np.int32)
        return self._distances[start]

    def _closest(self, start, candidates):
        distance = self.distances(start)
        reachable = candidates & (distance >= 0)
        if not reachable.any():
            return INF, None
        masked = np.where(reachable, distance, _INT_MAX)
        index = np.unravel_index(int(np.argmin(masked)), masked.shape)
        return int(masked[index]), (int(index[0]), int(index[1]))

    def face_steps(self, start, facing, material):
        """Steps to stand next to ``material`` and face it (0 when already facing it), and where."""
        if self.grid is None or material not in MATERIAL_NAMES:
            return INF, None
        start = (int(start[0]), int(start[1]))
        target = MATERIAL_NAMES.index(material)
        if facing is not None:
            front = (start[0] + int(facing[0]), start[1] + int(facing[1]))
            if self.inside(front) and int(self.grid[front]) == target:
                return 0, start
        mask = self.grid == target
        if not mask.any():
            return INF, None
        adjacent = np.zeros_like(mask)
        adjacent[1:, :] |= mask[:-1, :]
        adjacent[:-1, :] |= mask[1:, :]
        adjacent[:, 1:] |= mask[:, :-1]
        adjacent[:, :-1] |= mask[:, 1:]
        steps, stand = self._closest(start, adjacent)
        # One more move toward the material turns the agent to face it.
        return (steps + 1, stand) if stand is not None else (INF, None)

    def near_steps(self, start, facility):
        """Steps until ``facility`` is in the 3x3 neighbourhood crafting needs (0 when it already is)."""
        if self.grid is None:
            return INF, None
        start = (int(start[0]), int(start[1]))
        mask = self.grid == MATERIAL_NAMES.index(facility)
        if not mask.any():
            return INF, None
        near = mask.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                shifted = np.zeros_like(mask)
                shifted[max(dx, 0):self.width + min(dx, 0), max(dy, 0):self.height + min(dy, 0)] = \
                    mask[max(-dx, 0):self.width + min(-dx, 0), max(-dy, 0):self.height + min(-dy, 0)]
                near |= shifted
        if near[start]:
            return 0, start
        return self._closest(start, near)

    def place_steps(self, start, facing, facility):
        """Steps to place ``facility`` in front: 1 when the front cell allows it, else one move first."""
        where = constants.place[facility]["where"]
        if facing is not None:
            front = (int(start[0]) + int(facing[0]), int(start[1]) + int(facing[1]))
            if self.material(front) in where:
                return 1
        return 2


class CrafterProgress:
    """Progress problems and their options for one planning step."""

    def __init__(self, plans, observations, goals, free_agents, slow_margin):
        self.plans = {plan.agent_id: plan for plan in plans}
        self.observations = observations
        self.goals = dict(goals or {})
        self.margin = slow_margin
        self.board = Board(observations)
        self.front = frontier(observations)
        self.inventory = {agent_id: dict(obs.get("inventory") or {}) for agent_id, obs in observations.items()}
        self.free = {agent_id for agent_id in free_agents if self._active(agent_id)}
        self._spare_cache = {}
        # Shares already planned this step count as arriving for the current-plan estimate.
        self.incoming = Counter()
        for plan in plans:
            if plan.operation == "share" and plan.recipient in observations \
                    and self.inventory.get(plan.agent_id, {}).get(plan.item, 0) > 0:
                self.incoming[plan.recipient, plan.item] += 1

    def _kept(self, agent_id, milestone):
        """Units ``agent_id`` keeps back when an option for ``milestone`` uses its inventory.

        It keeps what it needs for another frontier milestone that gates at least as many
        later stages (``STAGES_AFTER``), when that milestone is its own goal or it already
        holds every material for it. Handing those over would trade a step on the critical
        path for one off it: in a pilot run the furnace option took the only stone of the
        agent about to make the stone pickaxe, and the pickaxe came 13 steps later.
        A tool counts for a collect milestone only when collecting is the agent's goal,
        since many agents hold the same pickaxe.
        """
        stock, kept = self.inventory[agent_id], Counter()
        for other in self.front:
            if other == milestone or STAGES_AFTER[other] < STAGES_AFTER[milestone]:
                continue
            needs, _ = requirements(other)
            ready = CRAFTER_MILESTONES[other][0] != "collect" and all(
                stock.get(item, 0) >= count for item, count in needs.items())
            if ready or self.goals.get(agent_id) == other:
                for item, count in needs.items():
                    kept[item] = max(kept[item], min(count, stock.get(item, 0)))
        return kept

    def _spare(self, milestone):
        """Per free agent: units kept back for ``milestone`` and units it can spare (cached)."""
        cache = self._spare_cache
        if milestone not in cache:
            kept = {agent_id: self._kept(agent_id, milestone) for agent_id in self.free}
            spare = {agent_id: {item: count - kept[agent_id][item]
                                for item, count in self.inventory[agent_id].items()}
                     for agent_id in self.free}
            cache[milestone] = (kept, spare)
        return cache[milestone]

    def _active(self, agent_id):
        obs = self.observations.get(agent_id) or {}
        return (not obs.get("dead", False) and not obs.get("sleeping", False)
                and obs.get("player_pos") is not None and obs.get("inventory") is not None)

    def _position(self, agent_id):
        return tuple(int(value) for value in self.observations[agent_id]["player_pos"])

    def _facing(self, agent_id):
        facing = self.observations[agent_id].get("player_facing")
        return None if facing is None else (int(facing[0]), int(facing[1]))

    def _gather(self, agent_id, item, units, position, facing):
        """Steps for ``agent_id`` to collect ``units`` of ``item`` from ``position``, and where it ends."""
        material = SOURCE_OF[item]
        tools = constants.collect[material]["require"]
        if any(self.inventory[agent_id].get(tool, 0) < count for tool, count in tools.items()):
            return INF, None
        steps, stand = self.board.face_steps(position, facing, material)
        if steps == INF:
            return INF, None
        # A tree stays a tree; a mined cell turns into path, so each further unit needs a turn or move.
        extra = 1 if material == "tree" else 2
        return steps + 1 + (units - 1) * extra, stand

    def _execute(self, milestone, position, facing, build_here):
        """Steps from ``position`` until the executor's milestone action (inclusive)."""
        kind, subject, _ = CRAFTER_MILESTONES[milestone]
        if build_here:
            facility = constants.make[subject]["nearby"][0]
            return self.board.place_steps(position, facing, facility) + 1
        if kind == "make":
            steps, _ = self.board.near_steps(position, constants.make[subject]["nearby"][0])
            return steps + 1
        if kind == "collect":
            steps, _ = self.board.face_steps(position, facing, subject)
            return steps + 1
        return self.board.place_steps(position, facing, subject)

    def _timeline(self, executor, milestone, build_here, givers, gatherers, own):
        """Estimated steps until the milestone; ``own``: items the executor collects itself first."""
        position, facing = self._position(executor), self._facing(executor)
        elapsed = 0
        for item, units in sorted(own.items()):
            cost, stand = self._gather(executor, item, units, position, facing)
            if cost == INF:
                return INF
            elapsed, position, facing = elapsed + cost, stand, None
        elapsed += self._execute(milestone, position, facing, build_here)
        # Each share takes one step of the giver; a share and the action may fall in the same step.
        times = [elapsed] + [sum(items.values()) for items in givers.values()]
        for agent_id, items in gatherers.items():
            total, position, facing = 0, self._position(agent_id), self._facing(agent_id)
            for item, units in sorted(items.items()):
                cost, stand = self._gather(agent_id, item, units, position, facing)
                if cost == INF:
                    return INF
                total, position, facing = total + cost + units, stand, None
            times.append(total)
        return max(times)

    def _now(self, milestone):
        """Estimate under the current plans: the agents working on it, each acting alone."""
        working = sorted(agent_id for agent_id, goal in self.goals.items()
                         if goal == milestone and self._active(agent_id))
        if not working:
            related = related_goals(milestone)
            working = sorted(agent_id for agent_id, goal in self.goals.items()
                             if goal in related and self._active(agent_id))
        needs, _ = requirements(milestone)
        best = INF
        for agent_id in working:
            stock = {item: self.inventory[agent_id].get(item, 0) + self.incoming[agent_id, item] for item in needs}
            own = {item: count - stock[item] for item, count in needs.items() if stock[item] < count}
            if any(item not in SOURCE_OF for item in own):
                continue  # a missing tool cannot be collected by this agent alone
            best = min(best, self._timeline(agent_id, milestone, False, {}, {}, own))
        return best, working

    def _action_plan(self, executor, milestone):
        kind, subject, _ = CRAFTER_MILESTONES[milestone]
        if kind == "make":
            return Plan(executor, f"make_{subject}", NA, NA, -1)
        if kind == "place":
            return Plan(executor, f"place_{subject}", NA, NA, -1)
        return Plan(executor, "do", subject, NA, -1)

    def verify(self, first_plans, executor, milestone):
        """Exact check with the joint one-step model: the milestone happens and unedited work survives."""
        plans = [self.plans[agent_id] for agent_id in sorted(self.plans)]
        model = _JointModel(plans, self.observations)
        projected = model.project([first_plans.get(plan.agent_id, plan) for plan in plans])
        kind, subject, _ = CRAFTER_MILESTONES[milestone]
        outcome = projected[executor]
        if (outcome.kind, outcome.subject, outcome.status) != (kind, subject, "satisfied"):
            return False
        for agent_id, before in model.baseline.items():
            if agent_id in first_plans or before.status != "satisfied" or before.kind == "idle":
                continue
            after = projected[agent_id]
            if after.status != "satisfied" or (before.kind, before.subject) != (after.kind, after.subject):
                return False
        return True

    def _option(self, milestone, executor, working, build_here=False):
        kind, subject, _ = CRAFTER_MILESTONES[milestone]
        needs, facility = requirements(milestone, build_here)
        kept, spare = self._spare(milestone)
        # A collect milestone only needs the tool in hand, so the executor's own tools all count.
        stock = self.inventory[executor] if kind == "collect" else spare[executor]
        missing = {item: count - stock.get(item, 0) for item, count in needs.items() if stock.get(item, 0) < count}
        givers, left = {}, {}
        for item, count in sorted(missing.items()):
            # Prefer idle holders, then those with most to spare.
            holders = sorted((agent_id for agent_id in self.free
                              if agent_id != executor and spare[agent_id].get(item, 0) > 0),
                             key=lambda agent_id: (self.plans[agent_id].operation not in IDLE_OPERATIONS,
                                                   -spare[agent_id][item], agent_id))
            for holder in holders:
                take = min(count, spare[holder][item])
                givers.setdefault(holder, {})[item] = take
                count -= take
                if not count:
                    break
            if count:
                left[item] = count
        own, gatherers = {}, {}
        for item, units in sorted(left.items()):
            if item not in SOURCE_OF:
                return None  # a tool nobody can hand over
            best = None
            for helper in [None] + sorted(self.free - {executor} - set(givers) - set(gatherers)):
                trial_own, trial_gatherers = dict(own), dict(gatherers)
                if helper is None:
                    trial_own[item] = units
                else:
                    trial_gatherers[helper] = {item: units}
                estimate = self._timeline(executor, milestone, build_here, givers, trial_gatherers, trial_own)
                # A helper is one more changed agent, worth ``margin`` steps.
                key = (estimate + (self.margin if helper is not None else 0), helper is not None, helper or -1)
                if best is None or key < best[0]:
                    best = (key, helper, estimate)
            if best[2] == INF:
                return None
            if best[1] is None:
                own[item] = units
            else:
                gatherers[best[1]] = {item: units}
        estimate = self._timeline(executor, milestone, build_here, givers, gatherers, own)
        if estimate == INF:
            return None
        changed = set(givers) | set(gatherers)
        if executor not in working or build_here:
            changed.add(executor)
        if not changed:
            return None  # that is the current plan
        exact, first_plans = False, {}
        if estimate == 1 and not own and not gatherers and all(sum(items.values()) == 1 for items in givers.values()):
            first_plans = {giver: Plan(giver, "share", NA, item, executor)
                           for giver, items in givers.items() for item in items}
            first_plans[executor] = self._action_plan(executor, milestone)
            exact = self.verify(first_plans, executor, milestone)
        if build_here:
            template = "build_here"
        elif executor in working:
            template = "supply"
        else:
            template = "gather" if kind == "collect" else "substitute"
        tasks = {}
        for giver, items in sorted(givers.items()):
            tasks[giver] = [["share", item, executor, count] for item, count in sorted(items.items())]
        for agent_id, items in sorted(gatherers.items()):
            tasks[agent_id] = [macro for item, units in sorted(items.items())
                               for macro in (["collect", SOURCE_OF[item], units], ["share", item, executor, units])]
        final = ([["place", facility], ["make", subject]] if build_here
                 else [["make", subject]] if kind == "make"
                 else [["collect", subject, 1]] if kind == "collect"
                 else [["place", subject]])
        # The executor collects until it holds the total it acts with (plus what it keeps back);
        # units teammates hand over count toward it, so it never stops short of the recipe.
        tasks[executor] = [["collect", SOURCE_OF[item], units, needs[item] + kept[executor][item]]
                           for item, units in sorted(own.items())] + final
        parts = [template, milestone, f"by{executor}",
                 ",".join(f"{giver}>{item}x{count}" for giver, items in sorted(givers.items())
                          for item, count in sorted(items.items())),
                 ",".join(f"{agent_id}^{item}x{units}" for agent_id, items in sorted(gatherers.items())
                          for item, units in sorted(items.items())),
                 ",".join(f"self^{item}x{units}" for item, units in sorted(own.items()))]
        interrupted = sorted(agent_id for agent_id in changed
                             if self.plans[agent_id].operation not in IDLE_OPERATIONS
                             and not (agent_id == executor and executor in working))
        return {
            "option_id": "|".join(parts), "template": template, "milestone": milestone, "executor": executor,
            "agents": sorted(changed), "interrupted": interrupted, "estimate": estimate, "exact": exact,
            "cost": estimate + self.margin * len(changed),
            "givers": {str(agent_id): items for agent_id, items in sorted(givers.items())},
            "gatherers": {str(agent_id): items for agent_id, items in sorted(gatherers.items())},
            "own": own, "first_plans": {str(agent_id): asdict(plan) for agent_id, plan in first_plans.items()}
            if exact else {},
            "tasks": {str(agent_id): macros for agent_id, macros in sorted(tasks.items())} if not exact else {},
        }

    def analyse(self, stalled=False, skip=(), only=None, force=False):
        """Progress problems at the frontier.

        A milestone has a problem when an option reaches it in this very step while the
        current plans would not (``exact``), or when the best option is at least
        ``slow_margin`` steps faster than the current plans (``slow``). When the team
        has stalled and no option can be estimated at all, the problem is ``unknown``
        and has no options. ``skip``: milestones already being handled; ``only`` limits
        the check; ``force`` returns the options of ``only`` even without a problem.
        """
        problems, now_by_milestone = [], {}
        steps = frontier(self.observations)
        for milestone in steps:
            if milestone in skip or (only is not None and milestone not in only):
                continue
            now, working = self._now(milestone)
            now_by_milestone[milestone] = {"estimate": None if now == INF else now, "agents": working}
            kind, subject, _ = CRAFTER_MILESTONES[milestone]
            options = []
            for executor in sorted(self.free):
                variants = [False]
                if kind == "make" and self.board.near_steps(self._position(executor),
                                                            constants.make[subject]["nearby"][0])[0] \
                        >= BUILD_HERE_MIN_STEPS:
                    variants.append(True)
                for build_here in variants:
                    option = self._option(milestone, executor, set(working), build_here)
                    if option is not None:
                        option["saves"] = None if now == INF else now - option["estimate"]
                        options.append(option)
            chosen = options if force else [
                option for option in options
                if (option["exact"] and now >= 2) or now - option["estimate"] >= self.margin]
            chosen.sort(key=lambda option: (not option["exact"], option["cost"], len(option["agents"]),
                                            option["estimate"], option["option_id"]))
            base = {"problem_id": f"progress:{milestone}", "milestone": milestone,
                    "now_estimate": None if now == INF else now, "now_agents": working}
            if chosen:
                problems.append({**base, "type": "exact" if chosen[0]["exact"] else "slow",
                                 "options": chosen if force else chosen[:LISTED_OPTIONS]})
            elif stalled and not options:
                problems.append({**base, "type": "unknown", "options": []})
        return {"frontier": steps, "now": now_by_milestone, "problems": problems, "slow_margin": self.margin}

    def prestage(self, skip=()):
        """Free agents to send ahead to a far material whose tool is about to be made.

        A collect milestone qualifies when its only missing earlier stage is making the
        tool it needs, and that tool is at most one stage from being makeable: iron while
        stone is being collected for the stone pickaxe, the diamond while the iron
        pickaxe's materials are being gathered. Shares ignore distance, so the tool can be
        handed over the moment it exists; walking there first saves the walk after it.

        One agent per material, chosen among free agents not working on a frontier
        milestone, nearest first. Nobody is sent when the walk is shorter than
        ``slow_margin``, when an agent is already that close, or when an agent is heading
        there (its goal is the milestone, or it is navigating to the material).
        ``skip``: milestones already being prepared.
        """
        done = achieved_milestones(self.observations)
        front = frontier(self.observations)
        busy = {agent_id for agent_id, goal in self.goals.items()
                if any(goal in related_goals(milestone) for milestone in front)}
        proposals, used = [], set()
        for milestone in MILESTONE_ORDER:
            kind, material, earlier = CRAFTER_MILESTONES[milestone]
            missing = [stage for stage in earlier if stage not in done]
            if kind != "collect" or milestone in done or milestone in skip or len(missing) != 1:
                continue
            tool = missing[0]
            if CRAFTER_MILESTONES[tool][0] != "make" or not all(
                    stage in done or stage in front for stage in CRAFTER_MILESTONES[tool][2]):
                continue
            walk = {agent_id: self.board.face_steps(self._position(agent_id), self._facing(agent_id), material)[0]
                    for agent_id in self.observations if self._active(agent_id)}
            heading = any(self.goals.get(agent_id) == milestone
                          or (plan.operation == "Navigator" and plan.material == material)
                          for agent_id, plan in self.plans.items())
            if heading or any(steps < self.margin for steps in walk.values()):
                continue
            candidates = [agent_id for agent_id in sorted(self.free - busy - used) if walk.get(agent_id, INF) < INF]
            if not candidates:
                continue
            agent_id = min(candidates, key=lambda candidate: (walk[candidate], candidate))
            proposals.append({
                "option_id": f"prestage|{milestone}|by{agent_id}", "milestone": milestone, "after": tool,
                "material": material, "agent": agent_id, "estimate": int(walk[agent_id]),
                "tasks": {str(agent_id): [["approach", material]]}})
            used.add(agent_id)
        return proposals


def _plan(agent_id, operation, material=NA, item=NA, recipient=-1):
    return Plan(agent_id, operation, material, item, recipient)


def crafter_task_step(task, observation, board, done, incoming, step, promised=None):
    """Next plan for one locked task, or ``(None, reason)`` when the task ends.

    ``task["macros"]`` run in order:

    - ``["share", item, recipient, count]``: hand over ``count`` units, one per step;
    - ``["collect", material, count]``: walk to the material and collect ``count`` units;
    - ``["collect", material, count, total]``: the executor's own part. It collects until
      it holds ``total``, counting units teammates' tasks still promise to hand over
      (``promised``), so a unit received counts once, not as collected;
    - ``["make", tool]``: walk next to the facility and make the tool once the materials are in hand;
    - ``["place", facility]``: place the facility in front (skipped when one is already nearby);
    - ``["approach", material]``: walk until facing the material, then wait there. Used to
      send an agent ahead (``CrafterProgress.prestage``); the task's milestone is the tool
      it waits for, so it ends when the tool is made.

    ``make`` and ``place`` wait for materials only while a teammate's task still promises
    them; otherwise the agent collects what is missing itself.
    """
    promised = Counter() if promised is None else promised
    if task["milestone"] in done:
        return None, "milestone_reached"
    if observation.get("dead", False):
        return None, "dead"
    if observation.get("sleeping", False):
        return None, "asleep"
    inventory = observation.get("inventory") or {}
    if inventory.get("health", constants.items["health"]["max"]) <= LOW_HEALTH:
        return None, "low_health"
    if step - task["started"] >= task["budget"]:
        return None, "budget"
    agent_id = task["agent_id"]
    position = tuple(int(value) for value in observation["player_pos"])
    facing = observation.get("player_facing")
    facing = None if facing is None else (int(facing[0]), int(facing[1]))

    def holds(needed):
        return all(inventory.get(item, 0) + incoming[agent_id, item] >= count for item, count in needed.items())

    def expected(item):
        """Units in hand, arriving this step, or still promised by a teammate's task."""
        return inventory.get(item, 0) + incoming[agent_id, item] + promised[agent_id, item]

    def toward(material):
        steps, _ = board.face_steps(position, facing, material)
        if steps == INF:
            return None
        return _plan(agent_id, "do", material) if steps == 0 else _plan(agent_id, "Navigator", material)

    def fetch(needed):
        """Collect a missing material nobody is bringing; None when only promised units are missing."""
        for item, count in sorted(needed.items()):
            material = SOURCE_OF.get(item)
            if expected(item) >= count or material is None or any(
                    inventory.get(tool, 0) < units for tool, units in constants.collect[material]["require"].items()):
                continue
            plan = toward(material)
            if plan is not None:
                return plan
        return None

    while task["index"] < len(task["macros"]):
        macro, state = task["macros"][task["index"]], task.setdefault("state", {})
        kind = macro[0]
        if kind == "share":
            _, item, recipient, count = macro
            if state.get("sent", 0) >= count:
                task["index"], task["state"] = task["index"] + 1, {}
                continue
            if inventory.get(item, 0) < 1:
                return None, "item_missing"
            state["sent"] = state.get("sent", 0) + 1
            incoming[recipient, item] += 1
            promised[recipient, item] -= 1  # now counted in ``incoming``
            return _plan(agent_id, "share", item=item, recipient=recipient), None
        if kind == "collect":
            material, count = macro[1], macro[2]
            item = COLLECTED_ITEM[material]
            if len(macro) > 3:
                collected = expected(item) >= macro[3]
            else:
                base = state.setdefault("base", inventory.get(item, 0))
                collected = inventory.get(item, 0) - base >= count
            if collected:
                task["index"], task["state"] = task["index"] + 1, {}
                continue
            plan = toward(material)
            return (plan, None) if plan is not None else (None, "unreachable")
        if kind == "make":
            _, tool = macro
            recipe = constants.make[tool]
            base = state.setdefault("base", inventory.get(tool, 0))
            if inventory.get(tool, 0) > base:
                task["index"], task["state"] = task["index"] + 1, {}
                continue
            if not holds(recipe["uses"]):
                plan = fetch(recipe["uses"])
                if plan is not None:
                    return plan, None
            facility = recipe["nearby"][0]
            steps, _ = board.near_steps(position, facility)
            if steps == INF:
                return None, "unreachable"
            if steps > 0:
                return _plan(agent_id, "Navigator", facility), None
            # Wait in place until the promised materials arrive.
            return (_plan(agent_id, f"make_{tool}") if holds(recipe["uses"]) else _plan(agent_id, "noop")), None
        if kind == "place":
            _, facility = macro
            recipe = constants.place[facility]
            if board.near_steps(position, facility)[0] == 0:
                task["index"], task["state"] = task["index"] + 1, {}
                continue
            if not holds(recipe["uses"]):
                return fetch(recipe["uses"]) or _plan(agent_id, "noop"), None
            if facing is not None and board.material((position[0] + facing[0], position[1] + facing[1])) \
                    in recipe["where"]:
                return _plan(agent_id, f"place_{facility}"), None
            for delta in _DIRECTIONS:
                if board.material((position[0] + delta[0], position[1] + delta[1])) in recipe["where"]:
                    return _plan(agent_id, _MOVES[delta]), None
            return None, "no_space"
        if kind == "approach":
            _, material = macro
            steps, _ = board.face_steps(position, facing, material)
            if steps == INF:
                return None, "unreachable"
            return (_plan(agent_id, "noop") if steps == 0 else _plan(agent_id, "Navigator", material)), None
        raise ValueError(f"Unknown task step {macro!r}")
    return None, "completed"


def owed_shares(task):
    """Units a locked task still has to hand over: ``{(recipient, item): units}``."""
    owed = Counter()
    for offset, macro in enumerate(task["macros"][task["index"]:]):
        if macro[0] == "share":
            _, item, recipient, count = macro
            sent = (task.get("state") or {}).get("sent", 0) if offset == 0 else 0
            owed[recipient, item] += max(count - sent, 0)
    return owed


def crafter_task_plans(tasks, observations, step):
    """Next plan for every locked task: ``{agent_id: (plan, None)}`` or ``(None, reason)``.

    Shares are planned first, so a teammate's planned share counts toward what the
    receiver holds in the same step, as in the environment. A task that ends stops
    promising its remaining shares, so the receiver collects them itself.
    """
    board, done, incoming, results = Board(observations), achieved_milestones(observations), Counter(), {}
    promised = Counter()
    for task in tasks.values():
        promised.update(owed_shares(task))

    def sharing(agent_id):
        task = tasks[agent_id]
        return task["index"] < len(task["macros"]) and task["macros"][task["index"]][0] == "share"

    for agent_id in sorted(tasks, key=lambda agent_id: (not sharing(agent_id), agent_id)):
        result = crafter_task_step(tasks[agent_id], observations[agent_id], board, done, incoming, step, promised)
        if result[0] is None:
            promised.subtract(owed_shares(tasks[agent_id]))
        results[agent_id] = result
    return results
