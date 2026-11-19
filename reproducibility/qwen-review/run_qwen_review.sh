#!/usr/bin/env bash
set -euo pipefail
work=$(cd "$(dirname "$0")" && pwd)
if [[ "$1" == formal ]]; then
    bash "$work/run_step200_additions.sh" "$2" "$3" baseline
    bash "$work/run_step200_additions.sh" "$2" "$3" rule
fi
exec bash "$work/run_qwen_review_formal.sh" "$@"
