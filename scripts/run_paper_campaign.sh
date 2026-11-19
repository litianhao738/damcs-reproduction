#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin=${DAMCS_PYTHON:-python}
campaign=${DAMCS_CAMPAIGN:-results/paper_t1_t2_codex_token_plan}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
common=(--environment crafter --model_profiles glm_bailian --agent_num 6
        --slow_margin 10 --stall_steps 35 --review_trigger signals
        --review_cooldown 10 --review_max_agents 2 --pause_mode local
        --revalidation incremental --evidence_mode compact --review_timeout 120
        --review_step_seconds 1 --checkpoint_every 5 --resume)
formal=(--step_num 350 --seeds 4 5 6 --repeats 2)

run_batch() {
    "$python_bin" -B scripts/run_experiments.py "${common[@]}" "$@"
}

case "${1:-}" in
    pilot-rule)
        run_batch --suite ablation --variants C-no-llm --step_num 150 --seeds 7 8 --repeats 1 \
            --results_dir "$campaign/pilot/rule"
        ;;
    formal-rule)
        run_batch "${formal[@]}" --suite ablation --variants C-no-llm \
            --results_dir "$campaign/formal/table2_rule"
        ;;
    pilot-online)
        run_batch --step_num 150 --seeds 7 8 --repeats 1 --conditions B C \
            --results_dir "$campaign/pilot/bc"
        ;;
    pilot-format)
        run_batch --step_num 3 --seeds 7 --repeats 1 --conditions C \
            --results_dir "$campaign/pilot/format"
        ;;
    formal-online)
        run_batch "${formal[@]}" --conditions A B C --results_dir "$campaign/formal/table1_auto"
        run_batch "${formal[@]}" --suite ablation --variants C-no-legality \
            --results_dir "$campaign/formal/table2_legality"
        ;;
    pilot-human)
        run_batch --step_num 150 --seeds 7 --repeats 1 --conditions H --reviewer human \
            --reviewer_id "${2:?Specify pilot reviewer ID}" --results_dir "$campaign/pilot/human"
        ;;
    pilot-ai)
        run_batch --step_num 20 --seeds 7 --repeats 1 --conditions H --reviewer ai \
            --reviewer_id codex --review_trigger all --results_dir "$campaign/pilot/ai_external"
        ;;
    formal-h)
        seed=${2:?Specify seed 4, 5, or 6}
        repeat=${3:?Specify repeat 0 or 1}
        if [[ ! "$seed" =~ ^[456]$ || ! "$repeat" =~ ^[01]$ ]]; then
            printf '%s\n' 'Invalid seed or repeat.' >&2
            exit 2
        fi
        run_batch --step_num 350 --seeds "$seed" --repeats 1 --repeat_offset "$repeat" \
            --conditions H --reviewer ai --reviewer_id codex \
            --results_dir "$campaign/formal/table1_h_seed${seed}_r${repeat}"
        ;;
    formal-human)
        seed=${2:?Specify seed 4, 5, or 6}
        repeat=${3:?Specify repeat 0 or 1}
        reviewer=${4:?Specify reviewer ID}
        if [[ ! "$seed" =~ ^[456]$ || ! "$repeat" =~ ^[01]$ || ! "$reviewer" =~ ^[A-Za-z0-9_-]+$ ]]; then
            printf '%s\n' 'Invalid seed, repeat or reviewer ID.' >&2
            exit 2
        fi
        run_batch --step_num 350 --seeds "$seed" --repeats 1 --repeat_offset "$repeat" \
            --conditions H --reviewer human --reviewer_id "$reviewer" \
            --results_dir "$campaign/formal/table1_human_seed${seed}_r${repeat}"
        ;;
    cases)
        "$python_bin" -B scripts/evaluate_scope_examples.py --results_dir "$campaign/appendix/cases"
        ;;
    replay)
        mkdir -p "$campaign/appendix"
        "$python_bin" -B scripts/evaluate_frozen_scopes.py --input "$campaign/formal/table1_auto" \
            --capacities 2 --output "$campaign/appendix/scope_replay.json"
        ;;
    export)
        "$python_bin" -B scripts/report_paper_tables.py --input "$campaign/formal" --output "$campaign/paper"
        ;;
    *)
        printf '%s\n' 'Usage: run_paper_campaign.sh {pilot-rule|formal-rule|pilot-format|pilot-online|formal-online|pilot-human REVIEWER|pilot-ai|formal-h SEED REPEAT|formal-human SEED REPEAT REVIEWER|cases|replay|export}' >&2
        exit 2
        ;;
esac
