"""Shared team board for MA-Crafter: every agent's own status and the team's progress, shown to all.

Two decentralised-coordination ideas, kept as information only:

- a status board, after the shared context that connects each agent's local progress to
  the team (Microsoft Agensh): each agent reports its own inventory, current goal and
  plan, position, and what lies in its own field of view;
- a progress ledger (Magentic-One): milestones done, the next ones, who is working on
  each, and how long the team has gone without a new milestone.

The board holds no whole-map knowledge and gives no orders: everything on it is what
some agent already observes or intends, and each agent still plans for itself. The DAMCS
base passes each agent only the previous agent's goal (a chain); the board broadcasts
every agent's status to everyone. It is stateless, so checkpoints need nothing extra.
"""

import math

import numpy as np

from crafter import constants
from .coordination import MATERIAL_NAMES, Plan
from .progress_repair import MILESTONE_ORDER, achieved_milestones, frontier, related_goals

NA = "not_applicable"
# Inventory entries worth telling teammates about, in this order.
ITEMS = ("wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe")
# Materials and facilities worth reporting from an agent's own field of view.
SEEN_MATERIALS = ("tree", "stone", "coal", "iron", "diamond", "water", "lava", "table", "furnace")
# Semantic ids after the 13 material ids (crafter's SemanticView: Player, Cow, Zombie, Skeleton, Arrow, Plant).
CREATURES = {15: "zombie", 16: "skeleton"}
LOW_VITALS = 3


def field_of_view(observation):
    """Half width and half height of the agent's local view (9 x 7 cells by default: 4 and 3)."""
    view = observation.get("view")
    width, height = (9, 9) if view is None else (int(view[0]), int(view[1]))
    item_rows = int(math.ceil(len(constants.items) / width))
    return width // 2, (height - item_rows) // 2


def seen_nearby(observation):
    """Closest cell of each notable material or creature in the agent's own view: {name: steps}."""
    grid = observation.get("semantic")
    position = observation.get("player_pos")
    if grid is None or position is None:
        return {}
    grid = np.asarray(grid)
    x0, y0 = int(position[0]), int(position[1])
    half_width, half_height = field_of_view(observation)
    found = {}
    for x in range(max(0, x0 - half_width), min(grid.shape[0], x0 + half_width + 1)):
        for y in range(max(0, y0 - half_height), min(grid.shape[1], y0 + half_height + 1)):
            value = int(grid[x, y])
            name = MATERIAL_NAMES[value] if value < len(MATERIAL_NAMES) else CREATURES.get(value)
            if name in SEEN_MATERIALS or name in CREATURES.values():
                distance = abs(x - x0) + abs(y - y0)
                if distance and (name not in found or distance < found[name]):
                    found[name] = distance
    return found


def describe_plan(plan):
    if plan.operation == "share" and plan.item not in (None, NA):
        return f"share {plan.item} to agent {plan.recipient}"
    if plan.operation == "Navigator" and plan.material not in (None, NA):
        return f"go to {plan.material}"
    if plan.material not in (None, NA):
        return f"{plan.operation} {plan.material}"
    return plan.operation


def current_goal(agent):
    thoughts = getattr(agent, "thoughts", None)
    goal = getattr(getattr(thoughts, "goal", None), "current_goal", None)
    return None if goal is None else getattr(goal, "value", goal)


class TeamBoard:
    """Renders the board once per planning step; ``intervention`` (optional) supplies locked
    team tasks and the step of the last new milestone."""

    def __init__(self, intervention=None):
        self.intervention = intervention

    def _status_line(self, agent, observation, tasks):
        inventory = observation.get("inventory") or {}
        parts = [f"at {tuple(int(value) for value in observation['player_pos'])}"
                 if observation.get("player_pos") is not None else None]
        low = [f"{name} {inventory[name]}" for name in ("health", "food", "drink", "energy")
               if inventory.get(name, 9) <= LOW_VITALS]
        if low:
            parts.append("LOW " + ", ".join(low))
        if observation.get("sleeping"):
            parts.append("sleeping")
        goal = current_goal(agent)
        if goal:
            parts.append(f"goal: {goal}")
        task = tasks.get(agent.id)
        plan = describe_plan(Plan.from_agent(agent))
        if task and task.get("prestage"):
            parts.append(f"team task: wait near {task['macros'][0][1]} for {task['prestage']} "
                         f"(until {task['milestone']} is done): {plan}")
        elif task:
            parts.append(f"team task for {task['milestone']}: {plan}")
        else:
            parts.append(f"plan: {plan}")
        held = [f"{name} {inventory[name]}" for name in ITEMS if inventory.get(name, 0) > 0]
        parts.append("has " + (", ".join(held) if held else "nothing useful"))
        seen = seen_nearby(observation)
        if seen:
            parts.append("sees " + ", ".join(f"{name} ({steps} away)"
                                             for name, steps in sorted(seen.items(), key=lambda kv: kv[1])))
        return "; ".join(part for part in parts if part)

    def _ledger(self, agents, observations, tasks, step):
        done = achieved_milestones(observations)
        goals = {agent.id: current_goal(agent) for agent in agents}
        lines = ["Done: " + (", ".join(name for name in MILESTONE_ORDER if name in done) or "nothing yet") + "."]
        upcoming = []
        for milestone in frontier(observations):
            related = related_goals(milestone)
            workers = sorted(agent_id for agent_id in goals
                             if goals[agent_id] in related
                             or ((tasks.get(agent_id) or {}).get("milestone") == milestone
                                 and not tasks[agent_id].get("prestage")))
            upcoming.append(f"{milestone} ({'agents ' + ', '.join(map(str, workers)) if workers else 'nobody'})")
        if upcoming:
            lines.append("Next: " + "; ".join(upcoming) + ".")
        last = getattr(self.intervention, "last_progress_step", None)
        if last is not None and step is not None:
            lines.append(f"Steps since the last new milestone: {max(0, step - last)}.")
        return " ".join(lines)

    def render(self, agents, info, step=None):
        """Board text for each agent id, marking the reader's own line."""
        observations = {agent.id: info[agent.id] for agent in agents}
        tasks = dict(getattr(self.intervention, "tasks", None) or {})
        lines = {agent.id: self._status_line(agent, observations[agent.id], tasks) for agent in agents}
        ledger = self._ledger(agents, observations, tasks, step)
        boards = {}
        for reader in agents:
            body = "\n".join(f"- Agent {agent_id}{' (you)' if agent_id == reader.id else ''}: {line}"
                             for agent_id, line in lines.items())
            boards[reader.id] = (
                "\n### Team board (every agent reports its own status; you still decide your own plan)\n"
                f"Team progress. {ledger}\n{body}\n")
        return boards
