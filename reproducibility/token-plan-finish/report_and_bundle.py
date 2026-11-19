"""Validate and package six matched Qwen4-Max-Preview review episodes."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import bundle_helpers as bundle

WORK = Path(__file__).resolve().parent
SETUP = WORK.parent
REPO = bundle.REPO
ORIGINAL = bundle.ORIGINAL
NEW = REPO / "results/paper_t1_t2_step200_token_plan_finish"
DELIVERY = REPO.parent / "damcs-paper-step200-token-plan-final"
REPORT = DELIVERY / "paper"
REVIEWER = "qwen4-max-preview"
MISSING = {("H", 6, 0), ("H", 6, 1)}


def collect():
    bundle.WORK = WORK
    bundle.SUPPLEMENT = NEW
    bundle.PAID = NEW / "table1_hq"
    bundle.DELIVERY = DELIVERY
    bundle.PAPER = REPORT
    bundle.unchanged_original()
    _, old = bundle.report.load_rows(ORIGINAL / "formal", bundle.SLOTS, REVIEWER)
    _, new = bundle.report.load_rows(NEW, MISSING, REVIEWER)
    if set(old) != bundle.SLOTS - MISSING or set(new) != MISSING:
        raise ValueError("Expected 34 original episodes and both seed-6 H-Q repeats")
    reference = next(iter(old.values()))
    if any(bundle.report.config_signature(row) != bundle.report.config_signature(reference) for row in new.values()):
        raise ValueError("Supplemental episodes differ from the original configuration")
    return old, new


def create_review_inputs(stage, rows):
    review_dir = stage / "table1_hq"
    review_dir.mkdir()
    review_rows = []
    mappings = {}
    for row in rows.values():
        if row["variant"] != "H":
            continue
        source = Path(row["run_dir"])
        target = review_dir / source.name
        target.mkdir()
        (target / "reviews").symlink_to(source / "reviews", target_is_directory=True)
        review_rows.append({**row, "run_dir": str(target)})
        mappings[source] = target
    (review_dir / "episodes.json").write_text(json.dumps(review_rows), encoding="utf-8")
    with (review_dir / "qwen_reviewer.jsonl").open("w", encoding="utf-8") as dst:
        for logfile in (ORIGINAL / "formal/table1_hq/qwen_reviewer.jsonl", NEW / "table1_hq/qwen_reviewer.jsonl"):
            for line in logfile.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                request = event.get("request")
                if not request:
                    if event.get("event") == "watching":
                        dst.write(line + "\n")
                    continue
                path = Path(request)
                if path.parent.parent in mappings:
                    target = mappings[path.parent.parent] / "reviews" / path.name
                    dst.write(json.dumps({**event, "request": str(target)}, ensure_ascii=False) + "\n")
    return review_dir


def export(old, new):
    import verify_qwen_reviews_all

    rows = {**old, **new}
    REPORT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DELIVERY) as folder:
        stage = Path(folder)
        base = stage / "automatic"
        base.mkdir()
        (base / "episodes.json").write_text(
            json.dumps([r for r in rows.values() if r["variant"] != "H"]), encoding="utf-8"
        )
        reviews = create_review_inputs(stage, rows)
        verify_qwen_reviews_all.verify(REPO, reviews, "formal")
        assert bundle.report.report(stage, REPORT, list(bundle.SEEDS), 2, REVIEWER) == (36, 36)
    path = REPORT / "paper_tables.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n\n"
                    "The last two H-Q runs were rerun after replenishing Token Plan credits, using the same "
                    "GLM-5.3 and Qwen4-Max-Preview configuration. The interrupted run and short "
                    "Qwen4-Max endpoint test are excluded from the formal matrix.\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    old, new = collect()
    bundle.package(old, new, export(old, new))
    bundle.unchanged_original()
