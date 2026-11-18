"""Budgeted plan review, joint commits, and observed execution feedback."""

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from .coordination import Plan, diagnose
from .progress_text import option_en
from .repair_scopes import PLAN_FIELDS, RepairTarget
from .review_domains import get_review_adapter
from .review_evidence import snapshot_copy


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def response_plans(agents, responses):
    replies = {response["id"]: response for response in responses}
    plans = []
    for agent in agents:
        response = replies.get(agent.id, {})
        if response.get("plan") is not None:
            plans.append(Plan(**response["plan"]))
        elif response.get("thoughts") is not None:
            plans.append(Plan.from_response(agent.id, response["thoughts"]))
        else:
            plans.append(Plan.from_agent(agent))
    return plans


def choose_review_scope(analysis, strategy, capacity):
    """Agents a reviewer may edit; ``capacity`` caps what one person is asked to read."""
    if strategy == "verified":
        selected = analysis["selected"]
        # A review view may widen the scope to verified alternatives (or to an unverifiable
        # scope); otherwise the selected repair. The search has no cap; a minimal repair
        # larger than the review cap is not sent.
        agents = analysis.get("review_agents") or (selected["edited_agents"] if selected else [])
        return list(agents) if agents and len(agents) <= capacity else []
    if strategy == "owner":
        counts = Counter(target["agent_id"] for target in analysis["targets"])
        return sorted(counts, key=lambda agent_id: (-counts[agent_id], agent_id))[:capacity]
    raise ValueError(f"Unknown review strategy '{strategy}'; use 'verified' or 'owner'.")


REVIEW_TRIGGERS = ("signals", "stage", "all")
# Steps without team progress that count as stalled, as a share of the episode length.
DEFAULT_STALL_FRACTION = 0.1
# Used only when the episode length is unknown.
FALLBACK_STALL_STEPS = 30


def stall_window(stall_steps, stall_fraction, step_limit):
    """Steps without team progress before the stall signal fires."""
    if stall_steps is not None:
        return stall_steps
    if step_limit:
        return max(1, round(stall_fraction * step_limit))
    return FALLBACK_STALL_STEPS

# Progress problems (legal plans, clearly slow team milestone). ``off``: not checked;
# ``log``: checked and recorded only (conditions A, B); ``auto``: the first option is
# applied (C); ``human``: a person chooses at critical moments, otherwise as ``auto`` (H).
PROGRESS_MODES = ("off", "log", "auto", "human")
# "Clearly slow" by default: this share of the episode length (10 of 350 steps), at least 2.
DEFAULT_SLOW_FRACTION = 0.03
FALLBACK_SLOW_MARGIN = 10
# A locked task is released after this multiple of its estimate plus a few steps.
TASK_BUDGET_FACTOR = 1.5
TASK_BUDGET_EXTRA = 3


def slow_margin_for(slow_margin, step_limit):
    """Steps an option must save over the current plans before the milestone counts as clearly slow."""
    if slow_margin is not None:
        return slow_margin
    if step_limit:
        return max(2, round(DEFAULT_SLOW_FRACTION * step_limit))
    return FALLBACK_SLOW_MARGIN


def option_agents(option):
    """Every agent an option changes, hands work to, or locks."""
    return {*option["agents"], *map(int, option.get("tasks", {})), *map(int, option.get("first_plans", {}))}


def effective_plans(responses, plans):
    replacements = {response["id"]: Plan(**response["plan_override"])
                    for response in responses if "plan_override" in response}
    return [replacements.get(plan.agent_id, plan) for plan in plans]


# Per-problem check results that mean "known not to be solved by this edit".
KNOWN_FAILURES = frozenset({"violated", "infeasible"})


def may_vouch(baseline, result):
    """Whether a reviewer may vouch for an edit the check cannot confirm.

    The check must be unknown overall (not broken somewhere), and the edit must turn
    at least one problem that fails without any edit into one that might now be solved.
    An edit that helps nothing is still rejected.
    """
    return result.status == "unknown" and any(
        baseline.target_results.get(target_id) in KNOWN_FAILURES and state not in KNOWN_FAILURES
        for target_id, state in result.target_results.items())


class InterventionController:
    def __init__(self, output_dir, mode="suggest", strategy="verified", capacity=2,
                 request_budget=None, timeout=120, pause_mode="global", evidence_mode="compact",
                 clock=time.monotonic, revalidation="incremental", environment="crafter",
                 search_agents=None, selection="minimal", review_trigger="signals", review_cooldown=10,
                 stall_steps=None, stall_fraction=DEFAULT_STALL_FRACTION, step_limit=None,
                 progress_mode="off", progress_commit=True, slow_margin=None, progress_pause=False):
        """``capacity``: most agents one review may open. ``search_agents``: cap on the
        automatic repair search (None searches the smallest sufficient scope, uncapped).
        ``selection``: ``minimal`` or the ``any`` ablation. ``revalidation``:
        ``incremental``, ``full``, or the ``none`` ablation, which commits a reviewed
        edit without rechecking the current state (the check still runs, only to log
        whether it would have failed).

        Human review has no request budget by default (``request_budget=None``) and asks
        only at critical moments. ``review_trigger`` decides which:

        - ``"signals"`` (default, environment-independent): a problem is critical when
          (1) ``choice``: verified repairs exist that change different agents, so someone
          must weigh whom to divert; (2) ``unknown``: no repair could be verified because
          facts are missing; or (3) ``stalled``: team progress (from the environment
          adapter) has not grown for ``stall_steps`` steps. By default that is
          ``stall_fraction`` of the episode's ``step_limit`` (10%: 35 of 350 Crafter
          steps, 3 of 30 CUBE steps), so the rule scales with episode length; an
          explicit ``stall_steps`` overrides it.
        - ``"stage"``: the adapter's fixed list of critical problems.
        - ``"all"``: every problem (practice episodes).

        ``review_cooldown``: steps before the same problem may be asked again after a
        request about it, so one stuck problem does not ask every step.

        Progress problems are plans that are all legal while the team's next milestone
        is clearly slow (see ``progress_repair``). ``progress_mode`` is one of
        ``PROGRESS_MODES``. ``progress_commit=False`` is the ablation that only writes
        multi-step assignments into the agents' context instead of locking them.
        ``slow_margin``: steps an option must save to count; by default 3% of
        ``step_limit`` (at least 2). The ``stalled`` signal works the same way for a
        progress problem, but ``choice`` is narrower: the two best options change
        different agents and the machine's ranking barely separates them (both exact or
        both estimated, costs less than ``slow_margin`` apart). A clear first option is
        applied without asking.

        A progress review holds nobody: the team keeps working, legality problems and
        the other milestones' progress problems get the automatic repair, and the chosen
        option is recomputed on the state when the reply arrives (a stale option falls
        back to the automatic choice). A review whose milestone is reached in the
        meantime closes on its own. ``progress_pause=True`` is the ablation that holds
        the agents of the listed options, following ``pause_mode``."""
        if revalidation not in ("incremental", "full", "none"):
            raise ValueError("revalidation must be 'incremental', 'full' or 'none'")
        if review_trigger not in REVIEW_TRIGGERS:
            raise ValueError(f"review_trigger must be one of {REVIEW_TRIGGERS}")
        if progress_mode not in PROGRESS_MODES:
            raise ValueError(f"progress_mode must be one of {PROGRESS_MODES}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir = self.output_dir / "reviews"
        self.review_dir.mkdir(exist_ok=True)
        self.mode, self.strategy, self.capacity = mode, strategy, capacity
        self.search_agents, self.selection = search_agents, selection
        self.request_budget, self.timeout = request_budget, timeout
        self.review_trigger, self.review_cooldown = review_trigger, review_cooldown
        self.stall_fraction, self.step_limit = stall_fraction, step_limit
        self.stall_steps = stall_window(stall_steps, stall_fraction, step_limit)
        self.last_asked = {}
        # Team progress and the last step it grew, for the stall signal.
        self.progress, self.last_progress_step = None, 0
        self.pause_mode, self.evidence_mode, self.clock = pause_mode, evidence_mode, clock
        self.revalidation = revalidation
        self.adapter = get_review_adapter(environment)
        self.validation_counts = Counter()
        self.validation_seconds = 0.0
        self.review_preparation_seconds = 0.0
        self.issued = 0
        self.human_seconds = 0.0
        self.pending = None
        self.last_request_key = None
        self.commits = []
        self.commit_count = 0
        self.outcomes = []
        self.paused_agent_steps = 0
        self.requests_by_kind = Counter()
        # Progress repair state.
        self.progress_mode, self.progress_commit = progress_mode, progress_commit
        self.progress_pause = progress_pause
        self.slow_margin = slow_margin_for(slow_margin, step_limit)
        self.tasks = {}            # agent id -> locked task
        self.progress_hold = {}    # milestone -> step until which a reviewer's "keep" holds
        self.progress_open = {}    # milestone -> last applied option, until the milestone is reached
        self.exact_checks = []     # exact options to check at the next transition
        self.progress_counts = Counter()
        self.progress_templates = Counter()
        self.progress_sources = Counter()
        self.task_releases = Counter()
        self.progress_replies = Counter()
        self.estimate_errors = []
        self.stall_steps_total = 0
        self.committed_agent_steps = 0
        self.progress_seconds = 0.0

    def event(self, kind, step, **data):
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "step": step, "time": time.time(), **data},
                                    ensure_ascii=False) + "\n")

    def _guard(self, agents, observations, responses, plans):
        from dataclasses import replace
        effective = [replace(plan, operation="noop") if agents[plan.agent_id].review_paused else plan
                     for plan in plans]
        checks = diagnose(effective, observations)
        blocked = {check.agent_id for check in checks if check.immediate and check.status == "violated"}
        by_id = {response["id"]: response for response in responses}
        for agent in agents:
            agent.coordination_blocked = agent.id in blocked
            agent.constraint_checks = [asdict(check) for check in checks if check.agent_id == agent.id]
            if agent.id in by_id:
                by_id[agent.id]["blocked"] = agent.coordination_blocked
                by_id[agent.id]["constraints"] = agent.constraint_checks
                by_id[agent.id]["deferred"] = agent.review_paused
        return responses

    def _pause(self, agents, plans, observations):
        request = self.pending
        targets = request.get("targets", [])
        if request.get("kind") == "progress" and not self.progress_pause:
            request["paused"] = set()
        elif self.pause_mode == "global":
            request["paused"] = {agent.id for agent in agents}
        elif self.pause_mode == "local" and request.get("kind") == "progress":
            # The agents the listed options would change or lock.
            request["paused"] = set(request["scope"])
        elif self.pause_mode == "local":
            request["paused"] |= self.adapter.dependencies(plans, observations, request["proposal"], targets)
            request["paused"].update(request["scope"])
        else:
            request["paused"] = set()
        for agent in agents:
            agent.review_paused = agent.id in request["paused"]

    def _commit(self, agents, observations, responses, plans, targets, edits, domains,
                source, request_id=None, already_verified=False, accept_unknown=False):
        """Check and apply one joint edit.

        ``accept_unknown``: for a review asked because facts were missing, a reviewer's
        edit whose check is still ``unknown`` is applied on the reviewer's judgement
        (logged as ``human_vouched``); an edit known to fail is still rejected.
        """
        step = agents[0].step
        started = time.perf_counter()
        expected = {target.target_id for target in targets}
        unchecked = self.revalidation == 'none' and source == 'human'
        if unchecked:
            # Ablation: commit the reviewed edit as is. The check below is only
            # logged, so the table can count stale edits that went through.
            shadow = self.adapter.verify(plans, observations, targets, edits, domains)
            would_fail = shadow.status != "repairable" or not expected <= set(shadow.covered_targets)
            self.validation_counts['unchecked'] += 1
            self.validation_counts['unchecked_would_fail'] += int(would_fail)
        elif already_verified:
            verification = self.pending['certificate']
            self.validation_counts['reused'] += 1
        elif self.revalidation == 'incremental':
            related = self.adapter.dependencies(plans, observations, edits, targets)
            verification = self.adapter.verify(plans, observations, targets, edits, domains, protected_agents=related)
            self.validation_counts['related_checks'] += 1
        else:
            verification = self.adapter.verify(plans, observations, targets, edits, domains)
            self.validation_counts['full_checks'] += 1
        vouched = False
        if not unchecked:
            self.validation_seconds += time.perf_counter() - started
            if accept_unknown and source == "human" and verification.status == "unknown":
                baseline = self.adapter.verify(plans, observations, targets, [], domains)
                vouched = may_vouch(baseline, verification)
            if not vouched and (verification.status != "repairable"
                                or not expected <= set(verification.covered_targets)):
                self.event("commit_rejected", step, request_id=request_id, source=source,
                           reasons=verification.reasons, target_results=verification.target_results)
                return False
            if vouched:
                self.validation_counts['human_vouched'] += 1
        self.commit_count += 1
        commit_id = f"commit_{self.commit_count:04d}"
        by_id = {response["id"]: response for response in responses}
        for edit in edits:
            response = by_id[edit.agent_id]
            response["plan_override"] = asdict(edit)
            response["intervention"] = {"commit_id": commit_id, "request_id": request_id, "source": source}
            response["thoughts"] = response["thoughts"] or agents[edit.agent_id].pending_plan_response
        self.commits.append({"commit_id": commit_id, "request_id": request_id,
                             "step": step, "source": source, "targets": targets})
        original = {plan.agent_id: plan for plan in plans}
        owners = {target.agent_id for target in targets}
        # Helpers whose own non-idle plan the repair replaced.
        interrupted = sorted({edit.agent_id for edit in edits if edit.agent_id not in owners
                              and original[edit.agent_id].operation != "noop"
                              and edit != original[edit.agent_id]})
        self.event("plan_applied", step, commit_id=commit_id, request_id=request_id, source=source,
                   edits=[asdict(edit) for edit in edits], targets=[asdict(target) for target in targets],
                   edited_agents=sorted({edit.agent_id for edit in edits}), interrupted_agents=interrupted,
                   certificate_unchanged=already_verified, unchecked=unchecked, human_vouched=vouched,
                   **({"would_fail_check": would_fail} if unchecked else {}))
        return True

    def _auto(self, agents, observations, responses, plans, analysis=None):
        if analysis is None:
            analysis = self.adapter.generate(plans, observations, max_scope_agents=self.search_agents,
                                             selection=self.selection,
                                             editable_agents={agent.id for agent in agents
                                                              if not getattr(agent, "is_human", False)
                                                              and agent.id not in self.tasks})
        selected = analysis["selected"]
        if selected is None:
            return False
        targets = [RepairTarget(**target) for target in analysis["targets"]
                   if target["target_id"] in selected["covered_targets"]]
        domains = {int(key): tuple(Plan(**plan) for plan in values) for key, values in analysis["domains"].items()}
        return self._commit(agents, observations, responses, plans, targets,
                            [Plan(**edit) for edit in selected["edits"]], domains, "auto")

    def _review_view(self, analysis, step):
        """The part of a scope analysis worth a person's attention now.

        Keeps the critical problems that were not asked about within the cooldown,
        the candidates that repair at least one of them, which signals made each
        problem critical, and the agents a verified-scope review should open.
        Returns the view and ``None``, or ``None`` and the reason no review is due.
        """
        signals = {target["target_id"]: self._signals(target, analysis, step) for target in analysis["targets"]}
        targets = [target for target in analysis["targets"] if signals[target["target_id"]]]
        if not targets:
            return None, "not_critical"
        targets = [target for target in targets
                   if step - self.last_asked.get(target["target_id"], -self.review_cooldown) >= self.review_cooldown]
        if not targets:
            return None, "cooldown"
        wanted = {target["target_id"] for target in targets}
        candidates = [candidate for candidate in analysis["candidates"]
                      if wanted & set(candidate["covered_targets"])]

        def preference(candidate):
            # Same preference as the search: repair more of these problems, then edit fewer agents.
            return (-len(wanted & set(candidate["covered_targets"])), -len(candidate["covered_targets"]),
                    len(candidate["edited_agents"]), tuple(candidate["edited_agents"]))

        selected = analysis["selected"]
        if selected is None or not wanted & set(selected["covered_targets"]):
            selected = min(candidates, key=preference) if candidates else None
        review_agents = None
        if selected is not None:
            # Open the selected repair plus verified alternatives that fit the review cap,
            # so a reviewer can choose whom to divert.
            review_agents = set(selected["edited_agents"])
            for candidate in sorted(candidates, key=preference):
                widened = review_agents | set(candidate["edited_agents"])
                if len(widened) <= self.capacity:
                    review_agents = widened
        elif any("unknown" in signals[target["target_id"]] for target in targets):
            # Nothing verified because facts are missing: open the agents involved in these
            # problems (their evidence groups), within the review cap, so a person can use
            # facts the search lacked. A problem known to be unrepairable opens nothing.
            review_agents = set()
            for target in targets:
                widened = review_agents | set(target["evidence_agents"])
                if len(widened) <= self.capacity:
                    review_agents = widened
            review_agents = review_agents or None
        return {**analysis, "targets": targets, "candidates": candidates, "selected": selected,
                "review_agents": sorted(review_agents) if review_agents else None,
                "signals": {target_id: signals[target_id] for target_id in sorted(wanted)}}, None

    def _signals(self, target, analysis, step):
        """Why ``target`` deserves a person now; an empty list means it does not."""
        if self.review_trigger == "all":
            return ["all"]
        if self.review_trigger == "stage":
            return ["stage"] if self.adapter.critical(target) else []
        found = []
        covering = [candidate for candidate in analysis["candidates"]
                    if target["target_id"] in candidate["covered_targets"]]
        if len({frozenset(candidate["edited_agents"]) for candidate in covering}) >= 2:
            found.append("choice")
        if not covering and any(scope["status"] == "unknown" for scope in analysis.get("scope_results", [])):
            found.append("unknown")
        if step - self.last_progress_step >= self.stall_steps:
            found.append("stalled")
        return found

    def _request(self, agents, observations, responses, plans, analysis):
        """Issue one review request; return None, or the reason no request was sent."""
        started = time.perf_counter()
        scope = choose_review_scope(analysis, self.strategy, self.capacity)
        if not scope:
            selected = analysis["selected"]
            if selected and len(selected["edited_agents"]) > self.capacity:
                return "over_review_capacity"
            return "no_new_review_scope"
        candidates = [candidate for candidate in analysis["candidates"]
                      if set(candidate["edited_agents"]) <= set(scope)]
        selected = analysis["selected"]
        if selected is not None and set(selected["edited_agents"]) <= set(scope):
            proposal = selected
        else:
            proposal = min(candidates, key=lambda candidate: (-len(candidate["covered_targets"]),
                                                            len(candidate["edited_agents"]))) if candidates else None
        goals = proposal["covered_targets"] if proposal else [target["target_id"]
                 for target in analysis["targets"] if target["agent_id"] in scope
                 or set(target["evidence_agents"]) <= set(scope)]
        # The request is about the problems in this view; others are repaired automatically.
        in_view = {target["target_id"] for target in analysis["targets"]}
        goals = [goal for goal in goals if goal in in_view]
        targets = [RepairTarget(**target) for target in analysis["targets"] if target["target_id"] in goals]
        edits = [Plan(**edit) for edit in proposal["edits"]] if proposal else []
        domains = {int(key): tuple(Plan(**plan) for plan in values) for key, values in analysis["domains"].items()}
        if self.strategy == "verified" and candidates:
            # Fields any verified repair inside the scope changes, so alternatives stay selectable.
            changed = {agent_id: set() for agent_id in scope}
            for candidate in candidates:
                for agent_id, fields in candidate["changed_fields"].items():
                    changed[int(agent_id)].update(fields)
            allowed_fields = {agent_id: [field for field in PLAN_FIELDS if field in changed[agent_id]]
                              or list(PLAN_FIELDS) for agent_id in scope}
        else:
            allowed_fields = {agent_id: list(PLAN_FIELDS) for agent_id in scope}
        original = {plan.agent_id: plan for plan in plans}
        domains = {agent_id: tuple(option for option in options
                                  if all(getattr(option, field) == getattr(original[agent_id], field)
                                         for field in PLAN_FIELDS if field not in allowed_fields.get(agent_id, ())))
                   for agent_id, options in domains.items() if agent_id in scope}
        evidence = self.adapter.evidence(plans, observations, targets, edits, domains,
                                        goals, self.evidence_mode)
        signature = json.dumps({"goals": goals, "facts": evidence["facts"]}, sort_keys=True)
        if signature == self.last_request_key:
            return "no_new_review_scope"
        self.last_request_key = signature
        accept_unknown = not edits and any("unknown" in analysis.get("signals", {}).get(target.target_id, [])
                                           for target in targets)
        self.issued += 1
        self.requests_by_kind["legal"] += 1
        request_id = f"review_{self.issued:04d}"
        step = agents[0].step
        document = {
            "request_id": request_id, "step": step, "strategy": self.strategy,
            "environment": self.adapter.name,
            "edit_agents": scope, "allowed_fields": allowed_fields,
            "plans": [asdict(plan) for plan in plans], "targets": [asdict(target) for target in targets],
            "proposal": [asdict(edit) for edit in edits],
            "domains": {agent_id: [asdict(option) for option in options] for agent_id, options in domains.items()},
            "evidence": evidence, "snapshot": self.adapter.snapshot(observations),
            "remaining_requests": None if self.request_budget is None else self.request_budget - self.issued,
            "timeout_seconds": self.timeout,
            # Wall-clock issue time, so the review page can show the time left.
            "issued_at": time.time(),
            # Why each problem was worth asking about: choice, unknown, stalled (or stage, all).
            "trigger_signals": analysis.get("signals", {}),
            # Facts are missing and nothing could be verified: the reviewer's judgement may
            # stand in for the check (an edit known to fail is still rejected).
            "accept_unknown": accept_unknown,
        }
        self.pending = {"id": request_id, "step": step, "scope": scope, "plans": plans,
                        "observations": snapshot_copy(observations), "targets": targets,
                        "domains": domains, "allowed_fields": allowed_fields,
                        "proposal": edits, "evidence": evidence, "started": self.clock(),
                        "paused": set(), "goals": set(goals), "accept_unknown": accept_unknown,
                        "certificate": self.adapter.verify(plans, observations, targets, edits, domains)}
        for target in targets:
            self.last_asked[target.target_id] = step
        self._pause(agents, plans, observations)
        document["pause_agents"] = sorted(self.pending["paused"])
        write_json(self.review_dir / f"{request_id}.json", document)
        self.review_preparation_seconds += time.perf_counter() - started
        self.event("review_requested", step, request_id=request_id, scope=scope,
                   pause_agents=document["pause_agents"], remaining_requests=document["remaining_requests"],
                   evidence_groups=evidence["fact_groups_after"], trigger_signals=document["trigger_signals"])
        return None

    def _resolve(self, agents, observations, responses, plans):
        if self.pending.get("kind") == "progress":
            return self._resolve_progress(agents, observations, responses, plans)
        pending = self.pending
        path = self.review_dir / f"{pending['id']}.response.json"
        timed_out = self.clock() - pending["started"] >= self.timeout
        if not path.exists() and not timed_out:
            self._pause(agents, plans, observations)
            return False
        response = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"decision": "timeout"}
        decision = response["decision"]
        self.human_seconds += response.get("read_seconds", 0)
        current_moves = self.adapter.moves(plans, observations)
        current_facts = self.adapter.facts(plans, observations, pending["evidence"]["agents"], current_moves)
        changes = [key for key, value in pending["evidence"]["facts"].items()
                   if current_facts[key] != value]
        for agent in agents:
            agent.review_paused = False
            if agent.pending_plan_response is not None:
                next(reply for reply in responses if reply["id"] == agent.id)["thoughts"] = agent.pending_plan_response
        accepted = False
        status = decision
        if decision in ("approve", "edit"):
            edits = [Plan(**edit) for edit in response.get("edits", [asdict(edit) for edit in pending["proposal"]])]
            original = {plan.agent_id: plan for plan in pending["plans"]}
            current = {plan.agent_id: plan for plan in plans}
            permission = all(edit.agent_id in pending["scope"] and all(
                getattr(edit, field) == getattr(original[edit.agent_id], field)
                for field in PLAN_FIELDS if field not in pending["allowed_fields"][edit.agent_id]
            ) for edit in edits)
            # The 'none' ablation skips every check of the current state, including this one.
            versions_match = self.revalidation == 'none' or all(
                current[agent_id] == original[agent_id] for agent_id in pending["scope"])
            if permission and versions_match:
                related = self.adapter.dependencies(plans, observations, edits, pending['targets'])
                reusable = (self.revalidation == 'incremental' and not changes
                            and edits == pending['proposal'] and pending['evidence']['sufficient']
                            and related <= set(pending['evidence']['agents']))
                accepted = self._commit(agents, observations, responses, plans, pending["targets"], edits,
                                        pending["domains"], "human", pending["id"], already_verified=reusable,
                                        accept_unknown=pending.get("accept_unknown", False))
                status = "accepted" if accepted else "invalid"
            else:
                status = "permission" if not permission else "stale_plan"
        self.event("review_resolved", agents[0].step, request_id=pending["id"], status=status,
                   decision=decision, changed_fact_groups=changes, read_seconds=response.get("read_seconds", 0),
                   waiting_seconds=self.clock() - pending["started"], provider=response.get("provider", "file"))
        write_json(self.review_dir / f"{pending['id']}.status.json", {"status": status})
        self.pending = None
        if not accepted:
            self._auto(agents, observations, responses, plans)
        return True

    def process(self, agents, observations, responses, analysis):
        for response in responses:
            if response.get("error"):
                # The agent keeps its previous plan this step; counted as a technical failure.
                self.event("planning_failed", agents[0].step, agent=response["id"], error=response["error"])
        plans = response_plans(agents, responses)
        if self.pending is None:
            targets = self.adapter.targets(plans, observations)
            if targets:
                directory = self.output_dir / 'failures'
                directory.mkdir(exist_ok=True)
                write_json(directory / f'step_{agents[0].step:05d}.json', {
                    'step': agents[0].step, 'environment': self.adapter.name,
                    'plans': [asdict(plan) for plan in plans],
                    'targets': [asdict(target) for target in targets],
                    'snapshot': self.adapter.snapshot(observations, full=True),
                })
        if self.pending is not None:
            resolved = self._resolve(agents, observations, responses, plans)
            if not resolved and self._progress_review_open() and self.mode in ("auto", "human"):
                # The team keeps working while a progress review is open; its legality
                # problems get the automatic repair, as everything not sent to a person does.
                self._auto(agents, observations, responses, plans)
        elif self.mode == "auto":
            self._auto(agents, observations, responses, plans, analysis)
        elif self.mode == "human" and analysis is not None and analysis["targets"]:
            exhausted = self.request_budget is not None and self.issued >= self.request_budget
            view, skipped = (None, 'budget_exhausted') if exhausted else self._review_view(analysis, agents[0].step)
            if not skipped:
                skipped = self._request(agents, observations, responses, plans, view)
            if skipped:
                details = {}
                if skipped == 'over_review_capacity':
                    details = {'scope_agents': view['selected']['edited_agents'],
                               'review_capacity': self.capacity}
                self.event('review_skipped', agents[0].step, reason=skipped, **details)
                # Everything not sent to a person, critical or not, gets the automatic repair.
                self._auto(agents, observations, responses, plans, analysis)
        effective = effective_plans(responses, plans)
        if self.progress_mode != "off" and (self.pending is None or self._progress_review_open()):
            # Progress problems are checked on the plans left after legality repair. While a
            # progress review is open, the other milestones are repaired automatically.
            self._progress(agents, observations, responses, effective)
            effective = effective_plans(responses, plans)
        return self._guard(agents, observations, responses, effective)

    def _progress_review_open(self):
        """A progress review is waiting for its reply and the team keeps working."""
        return self.pending is not None and self.pending.get("kind") == "progress" and not self.progress_pause

    # Progress repair -------------------------------------------------------------

    def prepare(self, agents, observations):
        """Before planning: end finished locked tasks and set every locked agent's next plan.

        A locked agent skips its language-model call (``Agent.progress_task`` is set).
        An agent waiting for a review keeps its task but gets no plan this step.
        """
        if not self.tasks:
            return
        step = agents[0].step
        active = {agent_id: task for agent_id, task in self.tasks.items() if not agents[agent_id].review_paused}
        for agent_id, (plan, reason) in self.adapter.task_plans(active, observations, step).items():
            if reason is not None:
                self._release(agents, agent_id, reason, step)
            else:
                self.adapter.apply_plan(agents[agent_id], plan)
                agents[agent_id].progress_task = self.tasks[agent_id]["option_id"]

    def _release(self, agents, agent_id, reason, step):
        task = self.tasks.pop(agent_id)
        agent = agents[agent_id]
        agent.progress_task = None
        self.adapter.release_agent(agent)
        self.task_releases[reason] += 1
        if reason in ("milestone_reached", "completed"):
            self.progress_counts["tasks_completed"] += 1
        notes = getattr(agent, "intervention_notes", None)
        if notes is not None:
            notes.append(f"Team assignment for {task['milestone']} ended: {reason}. Plan your next action yourself.")
        self.event("progress_task_released", step, agent=agent_id, option_id=task["option_id"],
                   milestone=task["milestone"], reason=reason, steps=step - task["started"])

    def _free_agents(self, agents, responses):
        """Agents a progress option may change: not locked, not waiting, not just edited."""
        replies = {response["id"]: response for response in responses}
        return {agent.id for agent in agents
                if agent.id in replies and not getattr(agent, "is_human", False) and not agent.review_paused
                and agent.id not in self.tasks and "plan_override" not in replies[agent.id]}

    def _progress_analysis(self, agents, observations, responses, plans, **options):
        started = time.perf_counter()
        step = agents[0].step
        handled = {task["milestone"] for task in self.tasks.values()}
        handled |= {milestone for milestone, until in self.progress_hold.items() if step < until}
        if self.pending is not None and self.pending.get("kind") == "progress":
            handled.add(self.pending["milestone"])  # waits for the reviewer
        analysis = self.adapter.progress_analysis(
            plans, observations, self.adapter.goals(agents, responses), self._free_agents(agents, responses),
            self.slow_margin, stalled=step - self.last_progress_step >= self.stall_steps, skip=handled, **options)
        self.progress_seconds += time.perf_counter() - started
        return analysis

    def _progress(self, agents, observations, responses, plans):
        step = agents[0].step
        problems = self._progress_analysis(agents, observations, responses, plans)["problems"]
        if problems:
            self.progress_counts["target_steps"] += 1
            for problem in problems:
                self.progress_counts[f"targets_{problem['type']}"] += 1
            self.event("progress_problems", step, problems=[{
                "milestone": problem["milestone"], "type": problem["type"],
                "now_estimate": problem["now_estimate"], "now_agents": problem["now_agents"],
                "options": [{key: option[key] for key in ("option_id", "estimate", "exact", "agents", "saves")}
                            for option in problem["options"]],
            } for problem in problems])
        if self.progress_mode == "log":
            return
        used = set()
        for problem in problems:
            options = [option for option in problem["options"] if not option_agents(option) & used]
            if not options:
                continue
            if self.progress_mode == "human" and self.pending is None:
                skipped = self._progress_request(agents, observations, plans, problem, options)
                if skipped is None:
                    break  # a review is open; the rest waits for the next step
                self.event("review_skipped", step, reason=skipped, review_kind="progress", milestone=problem["milestone"])
            self._apply_progress(agents, observations, responses, plans, options[0], "auto")
            used |= option_agents(options[0])

    def _progress_signals(self, problem, options, step):
        if self.review_trigger == "all":
            return ["all"]
        if self.review_trigger == "stage":
            return ["stage"] if self.adapter.critical(self.adapter.milestone_target(problem["milestone"])) else []
        found = []
        if len(options) >= 2:
            # Only a close call: the two best options divert different agents and the
            # ranking (exact first, then cost) barely separates them.
            best, second = options[0], options[1]
            if (best["exact"] == second["exact"] and second["cost"] - best["cost"] < self.slow_margin
                    and set(best["agents"]) != set(second["agents"])):
                found.append("choice")
        if step - self.last_progress_step >= self.stall_steps:
            found.append("stalled")
        return found

    def _progress_request(self, agents, observations, plans, problem, options):
        """Ask a person to choose among the options; return None, or why nobody was asked."""
        step = agents[0].step
        if self.request_budget is not None and self.issued >= self.request_budget:
            return "budget_exhausted"
        signals = self._progress_signals(problem, options, step)
        if not signals:
            return "not_critical"
        if step - self.last_asked.get(problem["problem_id"], -self.review_cooldown) < self.review_cooldown:
            return "cooldown"
        scope = sorted(set().union(*(option_agents(option) for option in options)))
        self.issued += 1
        self.requests_by_kind["progress"] += 1
        request_id = f"review_{self.issued:04d}"
        document = {
            "request_id": request_id, "kind": "progress", "step": step, "environment": self.adapter.name,
            "milestone": problem["milestone"], "problem_id": problem["problem_id"],
            "problem_type": problem["type"], "now_estimate": problem["now_estimate"],
            "now_agents": problem["now_agents"], "options": options, "recommended": options[0]["option_id"],
            "edit_agents": scope, "trigger_signals": {problem["problem_id"]: signals},
            "remaining_requests": None if self.request_budget is None else self.request_budget - self.issued,
            "timeout_seconds": self.timeout, "issued_at": time.time(),
        }
        self.pending = {"kind": "progress", "id": request_id, "step": step, "scope": scope, "paused": set(),
                        "started": self.clock(), "milestone": problem["milestone"],
                        "problem_id": problem["problem_id"], "recommended": options[0]["option_id"]}
        self.last_asked[problem["problem_id"]] = step
        self._pause(agents, plans, observations)
        document["pause_agents"] = sorted(self.pending["paused"])
        write_json(self.review_dir / f"{request_id}.json", document)
        self.event("review_requested", step, request_id=request_id, review_kind="progress", scope=scope,
                   pause_agents=document["pause_agents"], remaining_requests=document["remaining_requests"],
                   trigger_signals=document["trigger_signals"], milestone=problem["milestone"],
                   options=len(options))
        return None

    def _resolve_progress(self, agents, observations, responses, plans):
        pending = self.pending
        path = self.review_dir / f"{pending['id']}.response.json"
        timed_out = self.clock() - pending["started"] >= self.timeout
        # The team kept working and may have reached the milestone on its own.
        reached = self.adapter.milestone_reached(pending["milestone"], observations)
        if not path.exists() and not timed_out and not reached:
            self._pause(agents, plans, observations)
            return False
        if path.exists():
            response = json.loads(path.read_text(encoding="utf-8"))
        else:
            response = {"decision": "reached" if reached else "timeout"}
        decision = response["decision"]
        self.human_seconds += response.get("read_seconds", 0)
        replies = {reply["id"]: reply for reply in responses}
        for agent in agents:
            agent.review_paused = False
            if agent.pending_plan_response is not None and agent.id in replies:
                replies[agent.id]["thoughts"] = agent.pending_plan_response
        self.pending = None
        step, milestone = agents[0].step, pending["milestone"]
        # Legality problems of this step were not analysed while the review was open.
        if self.mode in ("auto", "human"):
            self._auto(agents, observations, responses, plans)
        current = effective_plans(responses, plans)
        option_id = pending["recommended"] if decision == "approve" else response.get("option_id")
        if reached:
            status, reply = "already_reached", "reached"
        elif decision in ("approve", "choose"):
            analysis = self._progress_analysis(agents, observations, responses, current, only=[milestone], force=True)
            match = [option for problem in analysis["problems"] for option in problem["options"]
                     if option["option_id"] == option_id]
            if match:
                self._apply_progress(agents, observations, responses, current, match[0], "human", pending["id"])
                status, reply = "accepted", "recommended" if option_id == pending["recommended"] else "other"
            else:
                status, reply = "stale_option", "stale"
        elif decision == "keep":
            self.progress_hold[milestone] = step + self.review_cooldown
            status, reply = "kept", "keep"
        else:
            status, reply = ("timeout", "timeout") if decision == "timeout" else ("reject", "skip")
        if status in ("stale_option", "timeout", "reject"):
            # As in condition C: the best option now, if the milestone is still clearly slow.
            for problem in self._progress_analysis(agents, observations, responses, current,
                                                   only=[milestone])["problems"]:
                if problem["options"]:
                    self._apply_progress(agents, observations, responses, current, problem["options"][0], "auto")
        self.progress_replies[reply] += 1
        self.event("review_resolved", step, request_id=pending["id"], review_kind="progress", status=status,
                   decision=decision, option_id=option_id, read_seconds=response.get("read_seconds", 0),
                   waiting_seconds=self.clock() - pending["started"], provider=response.get("provider", "file"))
        write_json(self.review_dir / f"{pending['id']}.status.json", {"status": status})
        return True

    @staticmethod
    def _override(replies, agents, agent_id, plan, source, request_id):
        reply = replies.get(agent_id)
        if reply is None:
            return
        reply["plan_override"] = asdict(plan)
        reply["intervention"] = {"commit_id": None, "request_id": request_id, "source": source}
        reply["thoughts"] = reply["thoughts"] or agents[agent_id].pending_plan_response

    def _apply_progress(self, agents, observations, responses, plans, option, source, request_id=None):
        """Apply one progress option: exact first-step edits, or locked tasks for multi-step work."""
        step = agents[0].step
        replies = {reply["id"]: reply for reply in responses}
        label = f"progress-{source}"
        locked = bool(self.progress_commit and option.get("tasks"))
        self.progress_counts["commits"] += 1
        self.progress_templates[option["template"]] += 1
        self.progress_sources[source] += 1
        if locked:
            budget = math.ceil(option["estimate"] * TASK_BUDGET_FACTOR) + TASK_BUDGET_EXTRA
            new = {}
            for key, macros in option["tasks"].items():
                agent_id = int(key)
                self.tasks[agent_id] = new[agent_id] = {
                    "agent_id": agent_id, "option_id": option["option_id"], "milestone": option["milestone"],
                    "macros": [list(macro) for macro in macros], "index": 0, "state": {}, "started": step,
                    "budget": budget, "estimate": option["estimate"], "source": source, "request_id": request_id}
                self.progress_counts["tasks"] += 1
            for agent_id, (plan, reason) in self.adapter.task_plans(new, observations, step).items():
                if reason is not None:
                    self._release(agents, agent_id, reason, step)
                else:
                    self._override(replies, agents, agent_id, plan, label, request_id)
                    agents[agent_id].progress_task = option["option_id"]
        elif option.get("first_plans"):
            for key, data in option["first_plans"].items():
                self._override(replies, agents, int(key), Plan(**data), label, request_id)
        # Without locking (ablation), a multi-step assignment reaches the agents only as a note.
        note = f"Team assignment ({label}): {option_en(option)}."
        for agent_id in sorted(option_agents(option)):
            notes = getattr(agents[agent_id], "intervention_notes", None)
            if notes is not None:
                notes.append(note)
        if option["exact"]:
            self.progress_counts["exact_commits"] += 1
            self.exact_checks.append(option)
        self.progress_open[option["milestone"]] = {"option_id": option["option_id"], "step": step,
                                                   "estimate": option["estimate"], "exact": option["exact"],
                                                   "source": source}
        self.event("progress_applied", step, request_id=request_id, source=source, locked=locked,
                   option={key: option.get(key) for key in (
                       "option_id", "template", "milestone", "executor", "agents", "interrupted", "estimate",
                       "exact", "saves", "first_plans", "tasks")})

    def before_actions(self, agents, observations):
        if self.pending is None:
            return []
        responses = [{"id": agent.id, "thoughts": None, "error": None}
                     for agent in agents if not getattr(agent, "is_human", False)]
        plans = [Plan.from_agent(agent) for agent in agents]
        self._resolve(agents, observations, responses, plans)
        replacements = {reply["id"]: Plan(**reply["plan_override"]) for reply in responses if "plan_override" in reply}
        effective = [replacements.get(plan.agent_id, plan) for plan in plans]
        return self._guard(agents, observations, responses, effective)

    def observe(self, before, after, agents, step):
        progress = self.adapter.progress(after)
        if self.progress is not None and progress > self.progress:
            self.last_progress_step = step
        self.progress = progress if self.progress is None else max(self.progress, progress)
        if step - self.last_progress_step >= self.stall_steps:
            self.stall_steps_total += 1
        self.committed_agent_steps += len(self.tasks)
        paused = [agent.id for agent in agents if agent.review_paused]
        self.paused_agent_steps += len(paused)
        self.event("environment_transition", step, actions=[agent.action for agent in agents], paused_agents=paused)
        if self.progress_mode != "off":
            for option in self.exact_checks:
                success = self.adapter.exact_effect(option, before, after)
                self.progress_counts["exact_success"] += int(success)
                self.event("progress_exact_outcome", step, option_id=option["option_id"], success=success)
            self.exact_checks.clear()
            for agent_id, task in list(self.tasks.items()):
                if self.adapter.milestone_reached(task["milestone"], after):
                    self._release(agents, agent_id, "milestone_reached", step)
            for milestone, record in list(self.progress_open.items()):
                if self.adapter.milestone_reached(milestone, after):
                    actual = step - record["step"]
                    self.estimate_errors.append(actual - record["estimate"])
                    self.event("progress_outcome", step, milestone=milestone, option_id=record["option_id"],
                               source=record["source"], estimate=record["estimate"], actual_steps=actual)
                    del self.progress_open[milestone]
        if before is None:
            return
        for commit in self.commits:
            events = {}
            for target in commit["targets"]:
                events[target.target_id] = self.adapter.target_met(target, before, after)
            outcome = {"commit_id": commit["commit_id"], "source": commit["source"],
                       "target_events": events, "recovered": all(events.values())}
            self.outcomes.append(outcome)
            self.event("repair_outcome", step, **outcome)
        self.commits.clear()

    def finish(self, info, step, reason):
        if self.pending is not None:
            write_json(self.review_dir / f"{self.pending['id']}.status.json", {"status": "episode_ended"})
            self.event("review_resolved", step, request_id=self.pending["id"], status="episode_ended")
            self.pending = None
        summary = {"total_steps": step, "requests_issued": self.issued, "request_budget": self.request_budget,
                   "human_seconds": self.human_seconds, "commits": self.commit_count,
                   "outcomes": self.outcomes, "paused_agent_steps": self.paused_agent_steps,
                   "task_success": self.adapter.success(info),
                   "termination_reason": reason, "mode": self.mode, "strategy": self.strategy,
                   "pause_mode": self.pause_mode, "evidence_mode": self.evidence_mode,
                   "revalidation": self.revalidation, "environment": self.adapter.name,
                   "review_trigger": self.review_trigger, "review_cooldown": self.review_cooldown,
                   "stall_steps": self.stall_steps, "stall_fraction": self.stall_fraction,
                   "step_limit": self.step_limit}
        summary['validation_counts'] = dict(self.validation_counts)
        summary['validation_seconds'] = self.validation_seconds
        summary['review_preparation_seconds'] = self.review_preparation_seconds
        if self.tasks:
            # Tasks still running when the episode ends.
            self.task_releases["episode_ended"] += len(self.tasks)
        counts = self.progress_counts
        summary.update({
            "requests_legal": self.requests_by_kind["legal"], "requests_progress": self.requests_by_kind["progress"],
            "progress_mode": self.progress_mode, "progress_commit": self.progress_commit,
            "progress_pause": self.progress_pause, "slow_margin": self.slow_margin,
            "progress_target_steps": counts["target_steps"],
            "progress_targets_by_type": {key[len("targets_"):]: value for key, value in counts.items()
                                         if key.startswith("targets_")},
            "progress_commits": counts["commits"], "progress_commits_by_template": dict(self.progress_templates),
            "progress_commits_by_source": dict(self.progress_sources),
            "progress_exact_commits": counts["exact_commits"], "progress_exact_success": counts["exact_success"],
            "progress_tasks": counts["tasks"], "progress_tasks_completed": counts["tasks_completed"],
            "progress_task_releases": dict(self.task_releases),
            "progress_estimate_errors": self.estimate_errors, "progress_replies": dict(self.progress_replies),
            "stall_steps_total": self.stall_steps_total, "committed_agent_steps": self.committed_agent_steps,
            "progress_seconds": self.progress_seconds,
        })
        write_json(self.output_dir / "intervention_summary.json", summary)
        self.event("episode_finished", step, **summary)
        return summary
