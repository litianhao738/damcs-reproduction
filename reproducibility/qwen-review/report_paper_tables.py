"""Export MA-Crafter tables 1 and 2 from the formal campaign's episodes.json files."""

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from statistics import mean, stdev


MAIN = {
    "A": "DAMCS baseline",
    "B": "A + legality repair",
    "C": "B + automatic progress repair",
    "H": "C + selective Codex review",
}
ABLATION = {
    "C": "Full automatic system",
    "C-no-legality": "Without legality repair",
    "B": "Without progress repair",
    "C-no-llm": "Standalone rule-based planner (LLM off)",
}
MODES = {
    "A": ("diagnose", 0, "off", "log"),
    "B": ("repair", 2, "auto", "log"),
    "C": ("repair", 2, "auto", "auto"),
    "H": ("repair", 2, "human", "human"),
    "C-no-legality": ("diagnose", 0, "off", "auto"),
    "C-no-llm": ("repair", 2, "auto", "auto"),
}
COMPARISONS = (("H", "C"), ("B", "A"), ("C", "B"),
               ("C", "C-no-legality"), ("C", "C-no-llm"))
COMMON = (
    "Valid runs", "Diamond steps", "Success", "Milestones",
    "Stalled steps", "API on-demand equivalent (CNY)",
)
HUMAN = ("Repair success", "Review requests / run", "Review seconds / run")
HUMAN_DELTAS = {
    "diamond_step_or_limit": "Diamond steps",
    "task_success": "Task success (0/1)",
    "process_seconds": "Wall-clock seconds",
    "episode_cost": "API on-demand equivalent (CNY)",
    "paused_agent_steps": "Paused agent-steps",
}
DIAGNOSTICS = (
    "task_success", "diamond_step", "diamond_step_or_limit", "termination_reason", "target_snapshots",
    "selected_snapshots", "illegal_plan_steps", "illegal_plans_first", "illegal_plans_after_replan",
    "commits", "recovery_attempts", "recoveries", "repair_attempts", "repair_successes",
    "progress_target_steps", "progress_commits", "progress_exact_success_rate",
    "progress_task_completion_rate", "committed_agent_steps", "total_steps",
    "requests_legal", "requests_progress", "requests_choice", "requests_stalled", "requests_unknown",
    "timeouts", "real_human_replies", "paused_agent_steps", "planning_failures", "chat_json_retries",
    "evidence_groups_before", "evidence_groups_after", "chat_requests", "embedding_requests",
    "chat_input_tokens", "chat_output_tokens", "embedding_tokens", "process_seconds", "human_seconds",
    "validation_seconds", "validation_unchecked_would_fail", "progress_reply_recommended",
    "progress_reply_other", "progress_reply_keep", "progress_reply_skip", "progress_reply_stale",
    "reply_stale_plan", "chat_model", "embedding_model",
)


def episode_key(row):
    return row["variant"], row["seed"], row["repeat_id"]


def config_signature(row):
    return tuple(row.get(key) for key in (
        "git_head", "source_sha256", "model_profile", "chat_model", "embedding_model", "base_url",
        "generate_kwargs", "input_mode", "agent_num", "step_limit", "slow_margin", "stall_steps", "stall_fraction",
        "review_timeout", "review_step_seconds",
        "review_trigger", "review_cooldown", "review_max_agents", "pause_mode",
        "evidence_mode", "revalidation", "human_requests", "scope_max_agents", "scope_selection",
        "currency", "input_price_per_million_tokens", "output_price_per_million_tokens",
        "embedding_price_per_million_tokens",
    ))


def load_rows(root, keys, reviewer_id="codex"):
    observed, valid, signature = {}, {}, None
    for path in sorted(root.rglob("episodes.json")):
        for row in json.loads(path.read_text(encoding="utf-8-sig")):
            key = episode_key(row)
            if key not in keys:
                raise ValueError(f"{path}: unexpected episode {key}; use the formal campaign directory")
            if key in observed:
                raise ValueError(f"{path}: duplicate episode {key}")
            observed[key] = row
            if row.get("exit_code") != 0 or not row.get("completed"):
                continue
            tag = row["variant"]
            expected_llm = "off" if tag == "C-no-llm" else "on"
            if (row.get("environment") != "crafter" or row.get("step_limit") != 200
                    or row.get("agent_num") != 6
                    or row.get("model_profile") != "glm_bailian" or row.get("agent_llm") != expected_llm
                    or row.get("chat_model") != "glm-5.3" or row.get("input_mode") != "text"
                    or not row.get("git_head") or not row.get("source_sha256")):
                raise ValueError(f"{path}: unexpected experiment configuration for {key}")
            modes = tuple(row.get(name) for name in (
                "coordination_mode", "repair_rounds", "scope_mode", "config_progress_mode",
            ))
            if (modes != MODES[tag] or row.get("config_agent_llm") != expected_llm
                    or row.get("config_progress_commit") != "on" or row.get("execution_guard") is not True
                    or row.get("review_strategy") != "verified"):
                raise ValueError(f"{path}: actual episode settings do not match variant {key}")
            if expected_llm == "on" and (row.get("config_chat_model") != row.get("chat_model")
                                         or row.get("config_input_mode") != row.get("input_mode")
                                         or row.get("config_model_profile") != "glm_bailian"):
                raise ValueError(f"{path}: actual chat profile does not match the matrix for {key}")
            if tag == "H" and (row.get("reviewer") != "ai" or row.get("reviewer_id") != reviewer_id
                               or row.get("scripted_replies") or row.get("real_human_replies")):
                raise ValueError(f"{path}: H requires external reviewer {reviewer_id} for {key}")
            if tag == "C-no-llm" and any(row.get(name) != 0 for name in (
                    "chat_requests", "embedding_requests", "chat_input_tokens",
                    "chat_output_tokens", "embedding_tokens")):
                raise ValueError(f"{path}: the rule-only episode {key} recorded model usage")
            steps = row.get("diamond_step_or_limit")
            success = row.get("task_success")
            if (type(success) is not bool or steps is None or not 0 <= steps <= 200
                    or (success and steps != row.get("diamond_step"))
                    or (not success and (steps != 200 or row.get("diamond_step") is not None))):
                raise ValueError(f"{path}: missing or inconsistent diamond outcome for {key}")
            current = config_signature(row)
            if signature is not None and current != signature:
                raise ValueError(f"{path}: mixed commits or settings in the formal results at {key}")
            signature = current
            valid[key] = row
    return observed, valid


def format_stats(values, digits=1):
    if not values or any(value is None for value in values):
        return "—"
    avg = f"{mean(values):.{digits}f}"
    return f"{avg} ± {stdev(values):.{digits}f}" if len(values) > 1 else avg


def make_row(tag, label, rows, expected, with_human):
    successes = sum(row["task_success"] for row in rows)
    cells = [
        f"{tag}: {label}", f"{len(rows)}/{expected}",
        format_stats([row["diamond_step_or_limit"] for row in rows]),
        f"{successes}/{len(rows)} ({successes / len(rows):.0%})" if rows else "—",
        format_stats([row.get("milestones_reached") for row in rows]),
        format_stats([row.get("stall_steps_total") for row in rows]),
        format_stats([row.get("episode_cost") for row in rows], 4),
    ]
    if with_human:
        attempts = sum(row["repair_attempts"] for row in rows)
        recovered = sum(row["repair_successes"] for row in rows)
        cells += [
            f"{recovered}/{attempts} ({recovered / attempts:.0%})" if attempts else "—",
            format_stats([row.get("requests_issued") for row in rows]) if tag == "H" else "—",
            format_stats([row.get("human_seconds") for row in rows]) if tag == "H" else "—",
        ]
    return cells


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def markdown(header, rows):
    return "\n".join(
        "| " + " | ".join(map(str, row)) + " |"
        for row in (header, ("---",) * len(header), *rows)
    )


def tex_cell(value):
    replacements = {"%": r"\%", "±": r"\(\pm\)", "—": "--"}
    return "".join(replacements.get(char, char) for char in value)


def write_tex(path, header, rows):
    lines = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\hline"]
    for index, row in enumerate((header, *rows)):
        lines.append(" & ".join(tex_cell(value) for value in row) + r" \\")
        if index == 0:
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def human_comparison(valid, slots, seeds, repeats):
    pairs = [(seed, repeat, valid[("H", seed, repeat)], valid[("C", seed, repeat)])
             for seed, repeat in slots if ("H", seed, repeat) in valid and ("C", seed, repeat) in valid]
    detail = []
    summary = []
    for metric, label in HUMAN_DELTAS.items():
        per_seed = {seed: [] for seed in seeds}
        for seed, repeat, human, automatic in pairs:
            h, c = human.get(metric), automatic.get(metric)
            if h is None or c is None:
                continue
            delta = float(h) - float(c)
            detail.append((seed, repeat, human["reviewer_id"], metric, h, c, delta))
            per_seed[seed].append(delta)
        full_seeds = [mean(deltas) for deltas in per_seed.values() if len(deltas) == repeats]
        summary.append((label, f"{sum(map(len, per_seed.values()))}/{len(slots)}",
                        f"{len(full_seeds)}/{len(seeds)}",
                        format_stats(full_seeds, 4 if metric == "episode_cost" else 2)))
    return detail, summary


def rule_episode_rows(valid, slots):
    for seed, repeat in slots:
        row = valid.get(("C-no-llm", seed, repeat))
        if row is None:
            continue
        total_steps = row["total_steps"]
        locked = row["committed_agent_steps"]
        yield (seed, repeat, row["diamond_step"], row["diamond_step_or_limit"],
               row["milestones_reached"], row.get("step_make_iron_pickaxe"),
               row["termination_reason"], total_steps, locked,
               locked / (row["agent_num"] * total_steps) if total_steps else None,
               row["progress_tasks_completed"], row["progress_tasks"],
               row["chat_requests"], row["embedding_requests"], row["episode_cost"])


def report(root, output, seeds, repeats, reviewer_id="codex"):
    slots = tuple(product(seeds, range(repeats)))
    keys = tuple((tag, seed, repeat) for tag in dict.fromkeys((*MAIN, *ABLATION)) for seed, repeat in slots)
    observed, valid = load_rows(root, set(keys), reviewer_id)
    output.mkdir(parents=True, exist_ok=True)

    successful = len(valid)
    received = len(observed)
    expected = len(keys)
    notes = [
        f"Formal 200-step episodes completed: {successful}/{expected}. "
        f"Technical failures or incomplete: {received - successful}. Missing: {expected - received}.",
        "Metrics are mean ± sample SD unless shown as a count or a rate. "
        "Unsuccessful task runs contribute 200 diamond steps. Technical failures are excluded from outcome "
        "denominators and listed in episode_diagnostics.csv. Rates with no attempts are undefined.",
        "Repair success pools successful/submitted legality repairs, exact progress repairs and locked progress "
        "tasks. Review seconds sum elapsed time from receipt of a request to submission. "
        "API cost is the on-demand equivalent of GLM and embedding usage. "
        "GLM uses Token Plan credits and embedding uses the on-demand endpoint. "
        "Missing price or usage is shown as unavailable.",
        f"The campaign has {len(seeds)} map seeds. "
        "Repeat-level differences are summarized by seed before interpreting consistency.",
    ]
    if valid:
        first = next(iter(valid.values()))
        notes.append(f"Git revision: {first['git_head']}. Source SHA256: {first['source_sha256']}.")

    reviewer_label = "Qwen4-Max-Preview" if reviewer_id == "qwen4-max-preview" else "Codex"
    review_variant = "H-Q" if reviewer_id == "qwen4-max-preview" else "H"
    methods_with_review = {**MAIN, "H": f"C + selective {reviewer_label} review ({review_variant})"}
    if review_variant == "H-Q":
        notes.append("H-Q uses Qwen4-Max-Preview external AI review. The H-Q minus C contrast measures "
                     "the end-to-end effect of model-assisted external review. A human-review effect "
                     "would require episodes reviewed by people. The GLM and embedding CNY column uses "
                     "on-demand equivalents. Qwen4 reviewer usage is reported separately in Token Plan tokens.")
        usage_path = root / "table1_hq" / "qwen_reviewer_usage.csv"
        if usage_path.exists():
            with usage_path.open(encoding="utf-8", newline="") as stream:
                usage_rows = list(csv.DictReader(stream))
                usage_header = tuple(usage_rows[0])
            write_csv(output / "qwen_reviewer_usage.csv", usage_header,
                      [tuple(row[name] for name in usage_header) for row in usage_rows])
            total = sum(int(row["reviewer_total_tokens"]) for row in usage_rows)
            usage_observed = sum(int(row["api_attempts_with_usage"]) for row in usage_rows)
            attempts = sum(int(row["api_attempts"]) for row in usage_rows)
            notes.append(f"Qwen4 reviewer API returned usage for {usage_observed}/{attempts} attempts, "
                         f"totaling {total:,} tokens with recorded usage.")
    for index, methods in enumerate((methods_with_review, ABLATION), start=1):
        header = ("Condition", *COMMON, *(HUMAN if index == 1 else ()))
        rows = [
            make_row(tag, label, [valid[key] for key in keys if key[0] == tag and key in valid],
                     len(slots), index == 1)
            for tag, label in methods.items()
        ]
        notes.extend((f"## Table {index}", markdown(header, rows)))
        write_csv(output / f"table{index}.csv", header, rows)
        write_tex(output / f"table{index}.tex", header, rows)

    paired = []
    contrasts = []
    for candidate, reference in COMPARISONS:
        per_seed = {seed: [] for seed in seeds}
        for seed, repeat in slots:
            left = valid.get((candidate, seed, repeat))
            right = valid.get((reference, seed, repeat))
            if left is not None and right is not None:
                delta = left["diamond_step_or_limit"] - right["diamond_step_or_limit"]
                per_seed[seed].append(delta)
                paired.append((candidate, reference, seed, repeat, delta,
                               int(left["task_success"]) - int(right["task_success"]),
                               left.get("process_seconds"), right.get("process_seconds")))
        full_seeds = [mean(values) for values in per_seed.values() if len(values) == repeats]
        contrasts.append((f"{review_variant if candidate == 'H' else candidate} minus {reference}", f"{sum(map(len, per_seed.values()))}/{len(slots)}",
                          f"{len(full_seeds)}/{len(seeds)}", format_stats(full_seeds),
                          sum(delta < 0 for delta in full_seeds)))
    notes.extend(("## Paired diamond-step contrasts",
                  "Negative values favor the first condition. Seed-level summaries use complete pairs across repeats.",
                  markdown(("Contrast", "Paired runs", "Complete seeds", "Seed-level mean ± SD", "Seeds faster"),
                           contrasts)))
    write_csv(output / "paired_deltas.csv", ("candidate", "reference", "seed", "repeat_id", "diamond_step_delta",
              "success_delta", "candidate_wall_seconds", "reference_wall_seconds"), paired)

    human_detail, human_summary = human_comparison(valid, slots, seeds, repeats)
    human_header = (f"Metric ({review_variant} minus C)", "Pairs", "Complete maps", "Map-level mean ± SD")
    notes.extend((f"## Net effect of {reviewer_label} external review",
                  f"All {review_variant} episodes enter the comparison, including cases with no review request, "
                  "timeouts, and automatic fallback. Task outcomes, API cost, and review time retain "
                  "their own units.",
                  markdown(human_header, human_summary)))
    write_csv(output / "human_net_effect.csv", human_header, human_summary)
    write_tex(output / "human_net_effect.tex", human_header, human_summary)
    write_csv(output / "human_paired_metrics.csv",
              ("seed", "repeat_id", "reviewer_id", "metric", "human", "automatic", "delta"),
              human_detail)

    human = [row for key, row in valid.items() if key[0] == "H"]
    rules = [row for key, row in valid.items() if key[0] == "C-no-llm"]
    notes.extend(("## Experiment diagnostics", markdown(("Condition", "All-dead episodes", "Wall seconds / run"), [
        (tag, sum(row.get("termination_reason") == "all_dead" for row in rows) if rows else "—",
         format_stats([row.get("process_seconds") for row in rows]))
        for tag in dict.fromkeys((*MAIN, *ABLATION))
        for rows in [[row for key, row in valid.items() if key[0] == tag]]
    ])))
    if human:
        requests = sum(row["requests_issued"] for row in human)
        timeouts = sum(row["timeouts"] for row in human)
        notes.append(f"{review_variant}, totals over {len(human)} valid episodes: "
                     f"{requests} requests, "
                     f"{sum(row['ai_external_replies'] for row in human)} external reviewer replies, "
                     f"{timeouts} timeouts, "
                     f"{sum(row.get('reply_stale_plan') or 0 for row in human)} stale legality replies, "
                     f"{sum(row.get('progress_reply_stale') or 0 for row in human)} stale progress replies, "
                     f"{sum(row['paused_agent_steps'] for row in human)} paused agent-steps.")
    if rules:
        locked = sum(row['committed_agent_steps'] for row in rules)
        capacity = sum(row['total_steps'] for row in rules) * 6
        tasks = sum(row['progress_tasks'] for row in rules)
        finished = sum(row['progress_tasks_completed'] for row in rules)
        if capacity:
            notes.append(f"Rule-based planner: {locked}/{capacity} agent-steps locked ({locked / capacity:.1%}); "
                         f"{finished}/{tasks} locked tasks completed.")

    write_csv(output / "rule_planner_episodes.csv",
              ("seed", "repeat_id", "diamond_step", "diamond_step_or_limit", "milestones_reached",
               "iron_pickaxe_step", "termination_reason", "total_steps", "committed_agent_steps",
               "committed_agent_step_share", "progress_tasks_completed", "progress_tasks",
               "chat_requests", "embedding_requests", "api_cost_cny"),
              rule_episode_rows(valid, slots))

    diagnostic_rows = []
    for key in keys:
        row = observed.get(key, {})
        status = "complete" if key in valid else "technical_or_incomplete" if row else "missing"
        steps = row.get("total_steps")
        locked = row.get("committed_agent_steps")
        locked_share = locked / (6 * steps) if key in valid and steps and locked is not None else None
        diagnostic_rows.append((*key, status, row.get("reviewer_id"), row.get("exit_code"),
                                row.get("run_id"), locked_share, *(row.get(name) for name in DIAGNOSTICS)))
    write_csv(output / "episode_diagnostics.csv",
              ("variant", "seed", "repeat_id", "status", "reviewer_id", "exit_code", "run_id",
               "committed_agent_step_share", *DIAGNOSTICS), diagnostic_rows)
    (output / "paper_tables.md").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    return successful, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing formal batches only.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reviewer-id", choices=("codex", "qwen4-max-preview"), default="codex")
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Specify positive repeats and distinct map seeds.")
    try:
        completed, expected = report(args.input, args.output, args.seeds, args.repeats, args.reviewer_id)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Exported paper tables with {completed}/{expected} formal episodes.")


if __name__ == "__main__":
    main()
