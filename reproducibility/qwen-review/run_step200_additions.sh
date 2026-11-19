#!/usr/bin/env bash
set -euo pipefail
umask 077
work=$(cd "$(dirname "$0")" && pwd)
repo=$1
campaign=$2
phase=$3
cd "$repo"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
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
case "$phase" in
    rule)
        exec 8>"$work/rule.lock"
        flock 8
        run_batch table2_rule --suite ablation --variants C-no-llm --seeds 4 5 6 --repeat_offset 0 --repeats 2
        ;;
    baseline)
        run_batch table1_auto_a_seed4_r0 --conditions A --seeds 4 --repeat_offset 0 --repeats 1
        ;;
    *) exit 2 ;;
esac
