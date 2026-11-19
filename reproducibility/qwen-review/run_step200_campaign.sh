#!/usr/bin/env bash
set -euo pipefail
umask 077
work=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$(dirname "$work")")
repo="$root/damcs-paper-d20cb582"
campaign="$repo/results/paper_t1_t2_step200"
export DAMCS_PYTHON=$(cat "$root/damcs-paper-setup/mirror-python.ready")
export DAMCS_AUTO_LOCK="$work/auto.lock"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
cd "$repo"
source /etc/network_turbo
exec 8>"$work/campaign.lock"
flock -n 8
exec 9>"$DAMCS_AUTO_LOCK"
flock -n 9
common=(--environment crafter --model_profiles glm_bailian --agent_num 6
        --slow_margin 10 --stall_steps 35 --review_trigger signals
        --review_cooldown 10 --review_max_agents 2 --pause_mode local
        --revalidation incremental --evidence_mode compact --review_timeout 120
        --review_step_seconds 1 --checkpoint_every 5 --resume --step_num 200)
run_batch() {
    local group=$1
    shift
    "$DAMCS_PYTHON" -B scripts/run_experiments.py "${common[@]}" \
        --results_dir "$campaign/formal/$group" "$@"
}
run_batch table1_auto_a_r1 --conditions A --seeds 4 5 6 --repeat_offset 1 --repeats 1
run_batch table1_auto_a_r0 --conditions A --seeds 5 6 --repeat_offset 0 --repeats 1
run_batch table1_auto_bc --conditions B C --seeds 4 5 6 --repeats 2
run_batch table2_legality --suite ablation --variants C-no-legality --seeds 4 5 6 --repeats 2
flock -u 9
exec 9>&-
bash "$work/run_qwen_review.sh" formal "$repo" "$campaign"
"$DAMCS_PYTHON" "$work/package_step200_delivery.py"
