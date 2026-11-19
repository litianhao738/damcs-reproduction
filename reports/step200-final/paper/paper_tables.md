Formal 200-step episodes completed: 36/36. Technical failures or incomplete: 0. Missing: 0.

Metrics are mean ± sample SD unless shown as a count or a rate. Unsuccessful task runs contribute 200 diamond steps. Technical failures are excluded from outcome denominators and listed in episode_diagnostics.csv. Rates with no attempts are undefined.

Repair success pools successful/submitted legality repairs, exact progress repairs and locked progress tasks. Review seconds sum elapsed time from receipt of a request to submission. API cost is the on-demand equivalent of GLM and embedding usage. GLM uses Token Plan credits and embedding uses the on-demand endpoint. Missing price or usage is shown as unavailable.

The campaign has 3 map seeds. Repeat-level differences are summarized by seed before interpreting consistency.

Git revision: d20cb582865a8e911a5b65076975a6f1e59874fd. Source SHA256: b0db93675d4f28d8865671f6b907c684c52adfea08f8dc8d6f78c741c7b2aa10.

H-Q uses Qwen4-Max-Preview external AI review. The H-Q minus C contrast measures the end-to-end effect of model-assisted external review. A human-review effect would require episodes reviewed by people. The GLM and embedding CNY column uses on-demand equivalents. Qwen4 reviewer usage is reported separately in Token Plan tokens.

Qwen4 reviewer API returned usage for 40/40 attempts, totaling 66,912 tokens with recorded usage.

## Table 1

| Condition | Valid runs | Diamond steps | Success | Milestones | Stalled steps | API on-demand equivalent (CNY) | Repair success | Review requests / run | Review seconds / run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: DAMCS baseline | 6/6 | 128.8 ± 64.0 | 4/6 (67%) | 9.7 ± 0.5 | 36.0 ± 44.6 | 21.3769 ± 13.0289 | — | — | — |
| B: A + legality repair | 6/6 | 143.0 ± 63.0 | 3/6 (50%) | 9.5 ± 0.5 | 33.7 ± 39.2 | 25.3496 ± 11.9300 | — | — | — |
| C: B + automatic progress repair | 6/6 | 104.8 ± 74.5 | 4/6 (67%) | 9.7 ± 0.5 | 40.5 ± 62.7 | 19.7978 ± 17.3476 | 54/56 (96%) | — | — |
| H: C + selective Qwen4-Max-Preview review (H-Q) | 6/6 | 116.8 ± 67.7 | 4/6 (67%) | 9.7 ± 0.5 | 34.0 ± 52.7 | 18.5532 ± 15.9251 | 62/62 (100%) | 4.8 ± 1.7 | 36.0 ± 22.5 |

## Table 2

| Condition | Valid runs | Diamond steps | Success | Milestones | Stalled steps | API on-demand equivalent (CNY) |
| --- | --- | --- | --- | --- | --- | --- |
| C: Full automatic system | 6/6 | 104.8 ± 74.5 | 4/6 (67%) | 9.7 ± 0.5 | 40.5 ± 62.7 | 19.7978 ± 17.3476 |
| C-no-legality: Without legality repair | 6/6 | 106.3 ± 73.7 | 4/6 (67%) | 9.7 ± 0.5 | 37.0 ± 57.3 | 15.7529 ± 13.4919 |
| B: Without progress repair | 6/6 | 143.0 ± 63.0 | 3/6 (50%) | 9.5 ± 0.5 | 33.7 ± 39.2 | 25.3496 ± 11.9300 |
| C-no-llm: Standalone rule-based planner (LLM off) | 6/6 | 136.7 ± 49.3 | 4/6 (67%) | 9.7 ± 0.5 | 29.7 ± 44.4 | 0.0000 ± 0.0000 |

## Paired diamond-step contrasts

Negative values favor the first condition. Seed-level summaries use complete pairs across repeats.

| Contrast | Paired runs | Complete seeds | Seed-level mean ± SD | Seeds faster |
| --- | --- | --- | --- | --- |
| H-Q minus C | 6/6 | 3/3 | 12.0 ± 20.8 | 0 |
| B minus A | 6/6 | 3/3 | 14.2 ± 12.3 | 0 |
| C minus B | 6/6 | 3/3 | -38.2 ± 43.2 | 2 |
| C minus C-no-legality | 6/6 | 3/3 | -1.5 ± 8.4 | 1 |
| C minus C-no-llm | 6/6 | 3/3 | -31.8 ± 29.4 | 2 |

## Net effect of Qwen4-Max-Preview external review

All H-Q episodes enter the comparison, including cases with no review request, timeouts, and automatic fallback. Task outcomes, API cost, and review time retain their own units.

| Metric (H-Q minus C) | Pairs | Complete maps | Map-level mean ± SD |
| --- | --- | --- | --- |
| Diamond steps | 6/6 | 3/3 | 12.00 ± 20.78 |
| Task success (0/1) | 6/6 | 3/3 | 0.00 ± 0.00 |
| Wall-clock seconds | 6/6 | 3/3 | 180.38 ± 403.45 |
| API on-demand equivalent (CNY) | 6/6 | 3/3 | -1.2446 ± 2.0996 |
| Paused agent-steps | 6/6 | 3/3 | 95.83 ± 32.11 |

## Experiment diagnostics

| Condition | All-dead episodes | Wall seconds / run |
| --- | --- | --- |
| A | 0 | 2109.6 ± 1175.0 |
| B | 0 | 2450.6 ± 1048.2 |
| C | 0 | 1851.4 ± 1537.4 |
| H | 0 | 2031.8 ± 1357.2 |
| C-no-legality | 0 | 1675.6 ± 1208.7 |
| C-no-llm | 0 | 61.1 ± 5.7 |

H-Q, totals over 6 valid episodes: 29 requests, 28 external reviewer replies, 1 timeouts, 0 stale legality replies, 2 stale progress replies, 575 paused agent-steps.

Rule-based planner: 582/4944 agent-steps locked (11.8%); 64/64 locked tasks completed.


The last two H-Q runs were rerun after replenishing Token Plan credits, using the same GLM-5.3 and Qwen4-Max-Preview configuration. The interrupted run and short Qwen4-Max endpoint test are excluded from the formal matrix.
