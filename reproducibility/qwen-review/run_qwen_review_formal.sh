#!/usr/bin/env bash
set -euo pipefail
umask 077

# Keep this driver outside the runtime checkout whose source hash is recorded.
mode=${1:?Use pilot or formal}
repo=${2:?Specify the frozen experiment checkout}
campaign=${3:?Specify the campaign directory}
suite_dir=$(cd "$(dirname "$0")" && pwd)
python_bin=${DAMCS_PYTHON:-python3}
reviewer=qwen4-max-preview

cd "$repo"
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
case "$mode" in
    pilot)
        batch="$campaign/pilot/qwen4_review"
        settings=(--step_num 20 --seeds 7 --repeats 1 --review_trigger all)
        ;;
    formal)
        batch="$campaign/formal/table1_hq"
        settings=(--step_num 200 --seeds 4 5 6 --repeats 2 --review_trigger signals)
        ;;
    *)
        echo 'Use pilot or formal.' >&2
        exit 2
        ;;
esac
mkdir -p "$batch"
exec 8>"$batch/runner.lock"
flock -n 8
if [[ "$mode" == formal ]]; then
    auto_lock=${DAMCS_AUTO_LOCK:?Specify the automatic campaign lock file}
    exec 9>"$auto_lock"
    flock 9
    "$python_bin" - "$campaign" <<'PY'
from itertools import product
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "formal"
folders = {
    "table1_auto_a_r1": (("A",), (4, 5, 6), (1,)),
    "table1_auto_a_r0": (("A",), (5, 6), (0,)),
    "table1_auto_a_seed4_r0": (("A",), (4,), (0,)),
    "table1_auto_bc": (("B", "C"), (4, 5, 6), (0, 1)),
    "table2_legality": (("C-no-legality",), (4, 5, 6), (0, 1)),
    "table2_rule": (("C-no-llm",), (4, 5, 6), (0, 1)),
}
signature = None
for folder, (conditions, seeds, repeats) in folders.items():
    rows = json.loads((root / folder / "episodes.json").read_text())
    expected = set(product(conditions, seeds, repeats))
    observed = {(row["variant"], row["seed"], row["repeat_id"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        sys.exit(f"{folder}: automatic paired inputs are incomplete")
    for row in rows:
        if row.get("exit_code") != 0 or not row.get("completed") or row.get("step_limit") != 200:
            sys.exit(f"{folder}: automatic run failed")
        current = (row["git_head"], row["source_sha256"], row["chat_model"], row["embedding_model"])
        if signature and current != signature:
            sys.exit(f"{folder}: automatic batches differ in source or models")
        signature = current
PY
    pilot_check="$campaign/pilot/qwen4_review/pilot_validated.json"
    for ((interval=0; interval<60; interval++)); do
        [[ -f "$pilot_check" ]] && break
        sleep 5
    done
    test -f "$pilot_check"
fi

"$python_bin" -B "$suite_dir/review_with_qwen.py" --repo "$repo" --watch "$batch" >> "$batch/qwen_reviewer.jsonl" 2>&1 &
review_pid=$!
batch_pid=
stop() {
    kill "$review_pid" 2>/dev/null || true
    if [[ -n "$batch_pid" ]]; then
        kill -TERM -- "-$batch_pid" 2>/dev/null || true
    fi
}
trap stop EXIT
trap 'exit 130' INT TERM

# A dedicated process group lets the supervisor stop an episode if its watcher exits.
setsid "$python_bin" -B scripts/run_experiments.py --environment crafter --model_profiles glm_bailian \
    --agent_num 6 --conditions H --reviewer ai --reviewer_id "$reviewer" \
    --slow_margin 10 --stall_steps 35 --review_cooldown 10 --review_max_agents 2 \
    --pause_mode local --revalidation incremental --evidence_mode compact --review_timeout 120 \
    --review_step_seconds 1 --checkpoint_every 5 --resume --results_dir "$batch" "${settings[@]}" &
batch_pid=$!
while kill -0 "$batch_pid" 2>/dev/null; do
    if ! kill -0 "$review_pid" 2>/dev/null; then
        echo 'Qwen review watcher exited during the experiment.' >&2
        exit 1
    fi
    sleep 3
done
wait "$batch_pid"
batch_pid=
"$python_bin" -B "$suite_dir/verify_qwen_reviews.py" --repo "$repo" --batch "$batch" --mode "$mode"
if [[ "$mode" == formal ]]; then
    "$python_bin" -B "$suite_dir/report_paper_tables.py" --input "$campaign/formal" --output "$campaign/paper_qwen" --reviewer-id "$reviewer"
fi
