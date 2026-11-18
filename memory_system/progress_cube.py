"""Progress problems in CUBE: a block whose pushers are slow to line up at its face.

A block of weight W moves right only when W agents stand on its left face and push
right in the same step; the milestone is delivering the block to the goal column. An
option assigns W agents to the face cells. When they already stand there, the first
push is verified exactly with the one-step CUBE model. Otherwise the walk is
estimated, and the agents run a locked task: walk to the assigned cell, wait for the
others, then push together until the block is delivered.
"""

from collections import deque
from dataclasses import asdict
from itertools import combinations, permutations
import math

from .coordination import Plan


INF = math.inf
NA = "not_applicable"
PUSH = "move_right"
MOVES = {(-1, 0): "move_up", (1, 0): "move_down", (0, -1): "move_left", (0, 1): "move_right"}
DELTAS = {operation: delta for delta, operation in MOVES.items()}
LISTED_OPTIONS = 5


def cube_state(info):
    return next(iter(info.values() if isinstance(info, dict) else info))["cube"]


def milestone_name(block_id):
    return f"deliver_block_{block_id}"


def block_of(milestone):
    return int(milestone.rsplit("_", 1)[1])


def delivered(milestone, info):
    return all(block["id"] != block_of(milestone) for block in cube_state(info)["blocks"])


def block_cells(block):
    row, column = block["pos"]
    return [(row + i, column + j) for i in range(block["weight"]) for j in range(block["weight"])]


def face_cells(block):
    """Cells to stand on to push the block right, top to bottom."""
    row, column = block["pos"]
    return [(row + i, column - 1) for i in range(block["weight"])]


def path_clear(block, blocks):
    """No other block sits in this block's rows to its right, so pushing right can deliver it."""
    rows = range(block["pos"][0], block["pos"][0] + block["weight"])
    edge = block["pos"][1] + block["weight"]
    return not any(other["id"] != block["id"] and cell[0] in rows and cell[1] >= edge
                   for other in blocks for cell in block_cells(other))


def pushes_needed(block, size):
    """Pushes until the block's right edge reaches the goal column."""
    return max(1, (size - 1) - (block["pos"][1] + block["weight"] - 1))


class CubeBoard:
    """Walking distances on the CUBE grid. Blocks block cells; agents move, so they do not."""

    def __init__(self, state):
        self.size = state["grid_size"]
        self.blocked = {cell for block in state["blocks"] for cell in block_cells(block)}
        self._distances = {}

    def free(self, cell):
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size and cell not in self.blocked

    def distances(self, start):
        start = tuple(start)
        if start not in self._distances:
            distance, queue = {start: 0}, deque([start])
            while queue:
                cell = queue.popleft()
                for delta in MOVES:
                    neighbour = (cell[0] + delta[0], cell[1] + delta[1])
                    if neighbour not in distance and self.free(neighbour):
                        distance[neighbour] = distance[cell] + 1
                        queue.append(neighbour)
            self._distances[start] = distance
        return self._distances[start]

    def distance(self, start, cell):
        return self.distances(start).get(tuple(cell), INF)


class CubeProgress:
    """Progress problems and their options for one planning step."""

    def __init__(self, plans, observations, free_agents, slow_margin, project):
        self.state = cube_state(observations)
        self.board = CubeBoard(self.state)
        self.plans = {plan.agent_id: plan for plan in plans}
        self.observations = observations
        self.positions = {agent_id: tuple(obs["player_pos"]) for agent_id, obs in observations.items()}
        self.free = set(free_agents) & set(observations)
        self.margin = slow_margin
        self.project = project

    def _moves_toward(self, agent_id, cell):
        """Whether the current plan keeps the agent on ``cell`` or brings it closer."""
        position, operation = self.positions[agent_id], self.plans[agent_id].operation
        if position == cell:
            return operation in ("noop", PUSH)
        delta = DELTAS.get(operation)
        if delta is None:
            return False
        after = (position[0] + delta[0], position[1] + delta[1])
        return self.board.free(after) and self.board.distance(after, cell) < self.board.distance(position, cell)

    def _assign(self, group, cells):
        """Assignment of ``group`` to ``cells`` with the shortest makespan, then the least walking."""
        best = (INF, INF, None)
        for order in permutations(cells):
            times = [self.board.distance(self.positions[agent_id], cell) for agent_id, cell in zip(group, order)]
            key = (max(times), sum(times))
            if key < best[:2]:
                best = (*key, dict(zip(group, order)))
        return best[2], best[0]

    def _pushes_now(self, plans, block_id):
        pushes, _, _ = self.project([plans.get(agent_id) for agent_id in sorted(plans)], self.observations)
        return (block_id, 4) in pushes

    def analyse(self, stalled=False, skip=(), only=None, force=False):
        """Same contract as ``CrafterProgress.analyse``; one milestone per undelivered block."""
        problems, now_by_milestone = [], {}
        blocks = sorted(self.state["blocks"], key=lambda block: block["id"])
        steps = [milestone_name(block["id"]) for block in blocks]
        agents = sorted(self.observations)
        for block in blocks:
            milestone = milestone_name(block["id"])
            if milestone in skip or (only is not None and milestone not in only):
                continue
            cells, weight = face_cells(block), block["weight"]
            pushes = pushes_needed(block, self.state["grid_size"])
            base = {"problem_id": f"progress:{milestone}", "milestone": milestone}
            # A block behind another block cannot be delivered first; the front one is its own problem.
            if len(agents) < weight or not all(self.board.free(cell) for cell in cells)                     or not path_clear(block, blocks):
                now_by_milestone[milestone] = {"estimate": None, "agents": []}
                if stalled:
                    problems.append({**base, "type": "unknown", "now_estimate": None, "now_agents": [],
                                     "options": []})
                continue
            # The current plans: agents already moving to (or waiting at) a face cell.
            heading = [agent_id for agent_id in agents
                       if any(self._moves_toward(agent_id, cell) for cell in cells)]
            now = INF
            for group in combinations(heading, weight):
                assignment, makespan = self._assign(group, cells)
                if assignment and all(self._moves_toward(agent_id, cell) for agent_id, cell in assignment.items()):
                    now = min(now, makespan + pushes)
            pushed_now = self._pushes_now(self.plans, block["id"])
            now_by_milestone[milestone] = {"estimate": None if now == INF else now, "agents": heading}
            options = []
            for group in combinations(agents, weight):
                if not set(group) <= self.free:
                    continue
                assignment, makespan = self._assign(group, cells)
                if makespan == INF:
                    continue
                exact, first_plans = False, {}
                if makespan == 0 and not pushed_now:
                    first_plans = {agent_id: Plan(agent_id, PUSH, NA, NA, -1) for agent_id in group}
                    exact = self._pushes_now({**self.plans, **first_plans}, block["id"])
                    changed = sorted(agent_id for agent_id in group if self.plans[agent_id].operation != PUSH)
                else:
                    changed = sorted(agent_id for agent_id, cell in assignment.items()
                                     if not self._moves_toward(agent_id, cell))
                if not changed:
                    continue
                estimate = makespan + pushes
                rows = {agent_id: cell[0] - block["pos"][0] for agent_id, cell in assignment.items()}
                options.append({
                    "option_id": f"push|{milestone}|" + ",".join(f"{agent_id}@{rows[agent_id]}"
                                                                 for agent_id in sorted(rows)),
                    "template": "push", "milestone": milestone, "executor": group[0], "group": list(group),
                    "agents": changed,
                    "interrupted": [agent_id for agent_id in changed if self.plans[agent_id].operation != "noop"],
                    "estimate": estimate, "exact": exact, "cost": estimate + self.margin * len(changed),
                    "saves": None if now == INF else now - estimate,
                    "first_plans": {str(agent_id): asdict(plan) for agent_id, plan in first_plans.items()}
                    if exact else {},
                    "tasks": {str(agent_id): [["push", block["id"], rows[agent_id]]] for agent_id in sorted(rows)},
                })
            chosen = options if force else [option for option in options
                                            if option["exact"] or now - option["estimate"] >= self.margin]
            chosen.sort(key=lambda option: (not option["exact"], option["cost"], len(option["agents"]),
                                            option["estimate"], option["option_id"]))
            base.update({"now_estimate": None if now == INF else now, "now_agents": heading})
            if chosen:
                problems.append({**base, "type": "exact" if chosen[0]["exact"] else "slow",
                                 "options": chosen if force else chosen[:LISTED_OPTIONS]})
            elif stalled and not options:
                problems.append({**base, "type": "unknown", "options": []})
        return {"frontier": steps, "now": now_by_milestone, "problems": problems, "slow_margin": self.margin}


def cube_task_plans(tasks, observations, step):
    """Next plan for every locked push task: walk to the assigned face cell, wait, push together.

    A task is ``[["push", block_id, row]]``: stand on the block's left face in that row.
    Agents that share an option push only once every one of them is in place.
    """
    state = cube_state(observations)
    board = CubeBoard(state)
    blocks = {block["id"]: block for block in state["blocks"]}
    positions = {agent_id: tuple(obs["player_pos"]) for agent_id, obs in observations.items()}
    results, targets = {}, {}
    for agent_id, task in tasks.items():
        _, block_id, row = task["macros"][0]
        block = blocks.get(block_id)
        if block is None:
            results[agent_id] = (None, "milestone_reached")
        elif step - task["started"] >= task["budget"]:
            results[agent_id] = (None, "budget")
        elif not path_clear(block, state["blocks"]):
            results[agent_id] = (None, "blocked")
        else:
            cell = (block["pos"][0] + row, block["pos"][1] - 1)
            if board.free(cell):
                targets[agent_id] = cell
            else:
                results[agent_id] = (None, "unreachable")
    for agent_id, cell in targets.items():
        option = tasks[agent_id]["option_id"]
        members = [other for other, task in tasks.items() if task["option_id"] == option]
        if positions[agent_id] == cell:
            ready = all(other in targets and positions[other] == targets[other] for other in members)
            results[agent_id] = (Plan(agent_id, PUSH if ready else "noop", NA, NA, -1), None)
            continue
        distance = board.distances(cell)
        position = positions[agent_id]
        steps = [(distance.get((position[0] + delta[0], position[1] + delta[1]), INF), operation)
                 for delta, operation in MOVES.items()]
        best = min(steps)
        results[agent_id] = (Plan(agent_id, best[1], NA, NA, -1), None) if best[0] < INF \
            else (None, "unreachable")
    return results
