"""Verify Qwen review replies and write episode-level reviewer usage."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from review_with_qwen import MODEL, REQUEST_NAME, REVIEWER_ID


def read_json(path):
    return json.loads(path.read_bytes())


def verify(repo, batch, mode):
    digest = hashlib.sha256((repo / "docs/codex_h_review_system_prompt.md").read_bytes()).hexdigest()
    episode_rows = read_json(batch / "episodes.json")
    expected = {(7, 0)} if mode == "pilot" else {(6, 0), (6, 1)}
    if (len(episode_rows) != len(expected)
            or {(row["seed"], row["repeat_id"]) for row in episode_rows} != expected
            or any(row["reviewer_id"] != REVIEWER_ID or not row["completed"] or row["exit_code"] != 0
                   for row in episode_rows)):
        raise ValueError("missing or failed Qwen episodes")
    by_run = {}
    calls = set()
    for line in (batch / "qwen_reviewer.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        for attempt in event.get("attempts", []):
            request_id = attempt.get("api_request_id")
            if request_id and request_id in calls:
                continue
            if request_id:
                calls.add(request_id)
            usage = attempt.get("usage") or {}
            run = str(Path(event["request"]).parent.parent)
            by_run.setdefault(run, []).append(usage)
    summaries = []
    for row in episode_rows:
        folder = Path(row["run_dir"])
        if not folder.resolve().is_relative_to(batch.resolve()):
            raise ValueError("episode belongs to a different batch")
        replies, committed = 0, 0
        for request_path in (folder / "reviews").glob("review_*.json"):
            if not REQUEST_NAME.fullmatch(request_path.name):
                continue
            response_path = request_path.with_suffix(".response.json")
            if not response_path.exists():
                continue
            request = read_json(request_path)
            response = read_json(response_path)
            if (response.get("reviewer_id") != REVIEWER_ID or response.get("provider") != "ai_external"
                    or response.get("reviewer_prompt_sha256") != digest
                    or response.get("reviewer_model_requested") != MODEL
                    or response.get("reviewer_model_returned") != "qwen4.0-max-preview"
                    or response.get("request_id") != request["request_id"]
                    or not request["issued_at"] <= response["submitted_at"] <= request["issued_at"] + request["timeout_seconds"]):
                raise ValueError(f"invalid review provenance in {response_path}")
            replies += 1
            status_path = request_path.with_suffix(".status.json")
            if status_path.exists() and read_json(status_path)["status"] in ("accepted", "kept"):
                committed += 1
        usage = by_run.get(str(folder), [])
        if replies != row["ai_external_replies"]:
            raise ValueError(f"{folder}: reply count differs from episode summary")
        summaries.append({
            "seed": row["seed"], "repeat_id": row["repeat_id"], "reviewer_id": REVIEWER_ID,
            "review_requests": row["requests_issued"], "reviewer_replies": replies,
            "accepted_or_kept": committed, "review_timeouts": row["timeouts"],
            "reviewer_wall_seconds": row["human_seconds"], "api_attempts": len(usage),
            "api_attempts_with_usage": sum(bool(item) for item in usage),
            "reviewer_input_tokens": sum(item.get("prompt_tokens", 0) for item in usage),
            "reviewer_output_tokens": sum(item.get("completion_tokens", 0) for item in usage),
            "reviewer_total_tokens": sum(item.get("total_tokens", 0) for item in usage),
        })
    output = batch / "qwen_reviewer_usage.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if mode == "pilot":
        if not summaries[0]["accepted_or_kept"]:
            raise ValueError("pilot needs an accepted or kept Qwen reply")
        proof = batch / "pilot_validated.json"
        proof.write_text(json.dumps({**summaries[0], "prompt_sha256": digest}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    args = parser.parse_args()
    verify(args.repo, args.batch, args.mode)
