# DAMCS formal 200-step campaign

The formal matrix includes A, B, C, H-Q, C-no-legality, and C-no-llm.
Run each variant on map seeds 4, 5, 6 with repeat IDs 0 and 1. Each run uses
a 200-step limit. Earlier 350-step episodes are reference material and
are excluded from the 200-step formal tables. Reported failures use 200 as
the capped steps-to-diamond outcome. H-Q compares AI review against C
with matching seeds, repeats, models, and step budgets.

On AutoDL, run run_step200_campaign.sh from the setup directory outside the
simulation checkout to preserve the recorded simulation source fingerprint.
The two previously completed groups are regenerated under the new budget.
The output archive contains the full Git bundle, raw logs, and Markdown, CSV
and LaTeX tables. The live campaign can resume from its own checkpoints.
