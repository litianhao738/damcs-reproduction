"""Budgeted plan review, joint commits, and observed execution feedback."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from .coordination import Plan, diagnose
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
        # The search has no cap; a minimal repair larger than the review cap is not sent.
        return selected["edited_agents"] if selected and len(selected["edited_agents"]) <= capacity else []
    if strategy == "owner":
        counts = Counter(target["agent_id"] for target in analysis["targets"])
        return sorted(counts, key=lambda agent_id: (-counts[agent_id], agent_id))[:capacity]
    raise ValueError(f"Unknown review strategy '{strategy}'; use 'verified' or 'owner'.")


class InterventionController:
    def __init__(self, output_dir, mode="suggest", strategy="verified", capacity=2,
                 request_budget=3, timeout=120, pause_mode="global", evidence_mode="compact",
                 clock=time.monotonic, revalidation="incremental", environment="crafter",
                 search_agents=None, selection="minimal"):
        """``capacity``: most agents one review may open. ``search_agents``: cap on the
        automatic repair search (None searches the smallest sufficient scope, uncapped).
        ``selection``: ``minimal`` or the ``any`` ablation. ``revalidation``:
        ``incremental``, ``full``, or the ``none`` ablation, which commits a reviewed
        edit without rechecking the current state (the check still runs, only to log
        whether it would have failed)."""
        if revalidation not in ("incremental", "full", "none"):
            raise ValueError("revalidation must be 'incremental', 'full' or 'none'")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir = self.output_dir / "reviews"
        self.review_dir.mkdir(exist_ok=True)
        self.mode, self.strategy, self.capacity = mode, strategy, capacity
        self.search_agents, self.selection = search_agents, selection
        self.request_budget, self.timeout = request_budget, timeout
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
        targets = request["targets"]
        if self.pause_mode == "global":
            request["paused"] = {agent.id for agent in agents}
        elif self.pause_mode == "local":
            request["paused"] |= self.adapter.dependencies(plans, observations, request["proposal"], targets)
            request["paused"].update(request["scope"])
        else:
            request["paused"] = set()
        for agent in agents:
            agent.review_paused = agent.id in request["paused"]

    def _commit(self, agents, observations, responses, plans, targets, edits, domains,
                source, request_id=None, already_verified=False):
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
        if not unchecked:
            self.validation_seconds += time.perf_counter() - started
            if verification.status != "repairable" or not expected <= set(verification.covered_targets):
                self.event("commit_rejected", step, request_id=request_id, source=source,
                           reasons=verification.reasons, target_results=verification.target_results)
                return False
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
                   certificate_unchanged=already_verified, unchecked=unchecked,
                   **({"would_fail_check": would_fail} if unchecked else {}))
        return True

    def _auto(self, agents, observations, responses, plans, analysis=None):
        if analysis is None:
            analysis = self.adapter.generate(plans, observations, max_scope_agents=self.search_agents,
                                                selection=self.selection,
                                                editable_agents={agent.id for agent in agents if not getattr(agent, "is_human", False)})
        selected = analysis["selected"]
        if selected is None:
            return False
        targets = [RepairTarget(**target) for target in analysis["targets"]
                   if target["target_id"] in selected["covered_targets"]]
        domains = {int(key): tuple(Plan(**plan) for plan in values) for key, values in analysis["domains"].items()}
        return self._commit(agents, observations, responses, plans, targets,
                            [Plan(**edit) for edit in selected["edits"]], domains, "auto")

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
        proposal = min(candidates, key=lambda candidate: (-len(candidate["covered_targets"]),
                                                        len(candidate["edited_agents"]))) if candidates else None
        goals = proposal["covered_targets"] if proposal else [target["target_id"]
                 for target in analysis["targets"] if target["agent_id"] in scope
                 or set(target["evidence_agents"]) <= set(scope)]
        targets = [RepairTarget(**target) for target in analysis["targets"] if target["target_id"] in goals]
        edits = [Plan(**edit) for edit in proposal["edits"]] if proposal else []
        domains = {int(key): tuple(Plan(**plan) for plan in values) for key, values in analysis["domains"].items()}
        allowed_fields = {agent_id: list(proposal["changed_fields"][agent_id])
                          if self.strategy == "verified" and proposal else list(PLAN_FIELDS) for agent_id in scope}
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
        self.issued += 1
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
            "remaining_requests": self.request_budget - self.issued, "timeout_seconds": self.timeout,
            # Wall-clock issue time, so the review page can show the time left.
            "issued_at": time.time(),
        }
        self.pending = {"id": request_id, "step": step, "scope": scope, "plans": plans,
                        "observations": snapshot_copy(observations), "targets": targets,
                        "domains": domains, "allowed_fields": allowed_fields,
                        "proposal": edits, "evidence": evidence, "started": self.clock(),
                        "paused": set(), "goals": set(goals),
                        "certificate": self.adapter.verify(plans, observations, targets, edits, domains)}
        self._pause(agents, plans, observations)
        document["pause_agents"] = sorted(self.pending["paused"])
        write_json(self.review_dir / f"{request_id}.json", document)
        self.review_preparation_seconds += time.perf_counter() - started
        self.event("review_requested", step, request_id=request_id, scope=scope,
                   pause_agents=document["pause_agents"], remaining_requests=document["remaining_requests"],
                   evidence_groups=evidence["fact_groups_after"])
        return None

    def _resolve(self, agents, observations, responses, plans):
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
                                        pending["domains"], "human", pending["id"], already_verified=reusable)
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
            self._resolve(agents, observations, responses, plans)
        elif self.mode == "auto":
            self._auto(agents, observations, responses, plans, analysis)
        elif self.mode == "human" and analysis is not None and analysis["targets"]:
            exhausted = self.issued >= self.request_budget
            skipped = 'budget_exhausted' if exhausted else \
                self._request(agents, observations, responses, plans, analysis)
            if skipped:
                details = {}
                if skipped == 'over_review_capacity':
                    details = {'scope_agents': analysis['selected']['edited_agents'],
                               'review_capacity': self.capacity}
                self.event('review_skipped', agents[0].step, reason=skipped, **details)
                self._auto(agents, observations, responses, plans, analysis)
        replacements = {response["id"]: Plan(**response["plan_override"]) for response in responses
                        if "plan_override" in response}
        effective = [replacements.get(plan.agent_id, plan) for plan in plans]
        return self._guard(agents, observations, responses, effective)

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
        paused = [agent.id for agent in agents if agent.review_paused]
        self.paused_agent_steps += len(paused)
        self.event("environment_transition", step, actions=[agent.action for agent in agents], paused_agents=paused)
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
                   "revalidation": self.revalidation, "environment": self.adapter.name}
        summary['validation_counts'] = dict(self.validation_counts)
        summary['validation_seconds'] = self.validation_seconds
        summary['review_preparation_seconds'] = self.review_preparation_seconds
        write_json(self.output_dir / "intervention_summary.json", summary)
        self.event("episode_finished", step, **summary)
        return summary
