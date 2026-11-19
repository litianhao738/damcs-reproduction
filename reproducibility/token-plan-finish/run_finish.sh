#!/usr/bin/env bash
set -euo pipefail
umask 077
work=$(cd "$(dirname "$0")" && pwd)
setup=$(dirname "$work")
repo=/root/autodl-tmp/damcs-paper-d20cb582
original="$repo/results/paper_t1_t2_step200"
batch="$repo/results/paper_t1_t2_step200_token_plan_finish/table1_hq"
python_bin=$(cat "$setup/mirror-python.ready")
mkdir -p "$batch"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
cd "$repo"
source /etc/network_turbo >/dev/null
exec 8>"$work/finish.lock"
flock -n 8
(cd "$original/formal" && sha256sum --quiet -c "$work/original_episode_summaries.sha256")

"$python_bin" -B "$work/review_with_qwen.py" --repo "$repo" --watch "$batch" \
    </dev/null >> "$batch/qwen_reviewer.jsonl" 2>&1 &
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
setsid "$python_bin" -B scripts/run_experiments.py \
    --environment crafter --model_profiles glm_bailian \
    --agent_num 6 --conditions H --reviewer ai --reviewer_id qwen4-max-preview \
    --slow_margin 10 --stall_steps 35 --review_cooldown 10 --review_max_agents 2 \
    --pause_mode local --revalidation incremental --evidence_mode compact --review_timeout 120 \
    --review_step_seconds 1 --checkpoint_every 5 --resume --results_dir "$batch" \
    --step_num 200 --seeds 6 --repeats 2 --review_trigger signals &
batch_pid=$!
while kill -0 "$batch_pid" 2>/dev/null; do
    if ! kill -0 "$review_pid" 2>/dev/null; then
        echo 'Qwen preview reviewer exited during the experiment.' >&2
        exit 1
    fi
    "$python_bin" -B "$work/check_service_errors.py" "$batch"
    sleep 5
done
wait "$batch_pid"
batch_pid=
"$python_bin" -B "$work/check_service_errors.py" "$batch"
"$python_bin" -B "$work/verify_qwen_reviews.py" --repo "$repo" --batch "$batch" --mode formal
"$python_bin" -B "$work/report_and_bundle.py"
