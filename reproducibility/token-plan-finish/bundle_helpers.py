"""Validate the 34 original episodes and 2 paid episodes and export both cohorts."""

import csv
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
import re
import sys
import zipfile

WORK = Path(__file__).resolve().parent
REPO = Path("/root/autodl-tmp/damcs-paper-d20cb582")
ORIGINAL = REPO / "results/paper_t1_t2_step200"
SUPPLEMENT = REPO / "results/paper_t1_t2_step200_paid_hq"
DELIVERY = Path("/root/autodl-tmp/damcs-paper-step200-paid-delivery")
PAID = SUPPLEMENT / "table1_hq"
PAPER = DELIVERY / "paper"
STEP_CAP = 200
SEEDS = (4, 5, 6)
REPEATS = (0, 1)
TAGS = ("A", "B", "C", "H", "C-no-legality", "C-no-llm")
SLOTS = set(product(TAGS, SEEDS, REPEATS))
PAID_SLOTS = {("H", 6, 0), ("H", 6, 1)}
PREVIEW = "qwen4-max-preview"
REVIEWER = "qwen4-max"
CREDENTIAL = re.compile(rb"sk-(?:sp-|ws-)?[A-Za-z0-9_.-]{23,}")
TEXT = {".log", ".json", ".jsonl", ".csv", ".md", ".sh", ".py", ".txt", ".tex", ".html"}

sys.path.insert(0, str(WORK.parent / "step200"))
import report_paper_tables as report


def unchanged_original():
    for line in (WORK / "original_episode_summaries.sha256").read_text().splitlines():
        expected, name = line.split(None, 1)
        actual = hashlib.sha256((ORIGINAL / "formal" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Original summary changed: {name}")


def archive_rows(bundle, rows, root, label):
    for row in rows.values():
        folder = Path(row["run_dir"]).resolve()
        if not folder.is_relative_to(root.resolve()) or not folder.is_dir():
            raise ValueError(f"Unrecognized run directory: {folder}")
        for f in sorted(folder.rglob("*")):
            add(bundle, f, f"raw/{label}/{f.relative_to(root).as_posix()}")
    for folder in sorted({Path(r["run_dir"]).parent for r in rows.values()}):
        for name in ("matrix.json", "episodes.json", "episodes.csv"):
            add(bundle, folder / name, f"raw/{label}/{folder.relative_to(root).as_posix()}/{name}")
    if label == "token_plan_supplement":
        add(bundle, PAID / "qwen_reviewer.jsonl", "raw/token_plan_supplement/qwen_reviewer.jsonl")
    if label == "token_plan":
        add(bundle, ORIGINAL / "formal/table1_hq/qwen_reviewer.jsonl", "raw/token_plan/table1_hq/qwen_reviewer.jsonl")


def add(bundle, path, name):
    if path.is_symlink() or not path.is_file() or path.suffix in {".pkl", ".tmp"}:
        return
    if path.suffix in TEXT and CREDENTIAL.search(path.read_bytes()):
        raise ValueError(f"Potential credential in {name}")
    bundle.write(path, name)


def package(original, paid, all_rows):
    manifest = {
        "complete": True,
        "formal_expected": len(SLOTS),
        "formal_valid": len(all_rows),
        "formal_step_limit": STEP_CAP,
        "formal_valid_by_variant": dict(Counter(k[0] for k in all_rows)),
        "cohorts": {"original": len(original), "token_plan_supplement": len(paid)},
        "reviewer_plan": {"qwen4-max-preview": 6},
        "source_sha256": next(iter(original.values()))["source_sha256"],
        "delivery_commit": (DELIVERY / "delivery_commit.txt").read_text().strip(),
    }
    for name, include_raw in (("damcs_table1_table2_step200_final", True), ("damcs_step200_final_tables", False)):
        target = DELIVERY / f"{name}.zip"
        temporary = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
            bundle.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
            for p in sorted(PAPER.rglob("*")):
                add(bundle, p, f"paper/{p.relative_to(PAPER).as_posix()}")
            if include_raw:
                for f in sorted(DELIVERY.glob("*.bundle*")):
                    add(bundle, f, f"source/{f.name}")
                archive_rows(bundle, original, ORIGINAL / "formal", "token_plan")
                archive_rows(bundle, paid, SUPPLEMENT, "token_plan_supplement")
                for f in sorted(WORK.glob("*.py")) + [WORK / "run_finish.sh"]:
                    add(bundle, f, f"reproducibility/{f.name}")
                add(bundle, WORK / "original_episode_summaries.sha256", "reproducibility/original_episode_summaries.sha256")
                for name in ("paper_experiment_protocol.md", "codex_h_review_system_prompt.md"):
                    add(bundle, REPO / "docs" / name, "reproducibility/" + name)
                add(bundle, REPO / "config/model_profiles.json", "reproducibility/model_profiles.json")
                add(bundle, WORK.parent / "pip-freeze-mirror.txt", "reproducibility/pip-freeze.txt")
                add(bundle, ORIGINAL / "reference_350/episodes.json", "reference_350/episodes.json")
        with zipfile.ZipFile(temporary) as bundle:
            failed = bundle.testzip()
            if failed:
                raise ValueError(f"Corrupt member: {failed}")
        temporary.replace(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
        print(f"{target} {target.stat().st_size} bytes {digest}", flush=True)


