import argparse,hashlib,json,os,re,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path("/root/autodl-tmp/damcs-paper-d20cb582")
SETUP=ROOT.parent/"damcs-paper-setup"
DELIVERY=ROOT.parent/"damcs-paper-step200-delivery"
CAMPAIGN=ROOT/"results/paper_t1_t2_step200"
BRANCH="deliverable/damcs-table1-table2-step200"
TAGS=("A","B","C","H","C-no-legality","C-no-llm")
expected={(variant,seed,repeat) for variant in TAGS for seed in (4,5,6) for repeat in (0,1)}
parser=argparse.ArgumentParser()
parser.add_argument("--partial",action="store_true")
args=parser.parse_args()
digest=(DELIVERY/"source-sha256.txt").read_text().strip()
records={}
summary_files=sorted((CAMPAIGN/"formal").glob("*/episodes.json"))
for source in summary_files:
    for row in json.loads(source.read_text()):
        key=(row["variant"],row["seed"],row["repeat_id"])
        if key not in expected or key in records:
            raise SystemExit("Missing or duplicate formal run identity: "+str(key))
        records[key]=row
valid={k:r for k,r in records.items() if r.get("completed") and r.get("exit_code")==0}
for key,row in valid.items():
    if row.get("step_limit") != 200:
        raise SystemExit("Unexpected step limit: "+str(key))
    if row.get("source_sha256")!=digest:
        raise SystemExit("Formal run source differs from delivery branch: "+str(key))
if not args.partial and valid.keys()!=expected:
    raise SystemExit(f"Full delivery waits for {len(expected)-len(valid)} formal episodes.")
env=dict(os.environ,DAMCS_PYTHON=sys.executable,DAMCS_CAMPAIGN=str(CAMPAIGN))
review_code = DELIVERY/"verified-worktree"/"reproducibility/qwen-review"
subprocess.run([sys.executable, str(review_code/"report_paper_tables.py"), "--input", str(CAMPAIGN/"formal"), "--output", str(CAMPAIGN/"paper_qwen"), "--reviewer-id", "qwen4-max-preview"], check=True)
expected_counts={tag:6 for tag in TAGS}
finished_counts={tag:sum(key[0]==tag for key in valid) for tag in TAGS}
manifest={
    "complete":valid.keys()==expected,
    "snapshot_utc":datetime.now(timezone.utc).isoformat(),
    "formal_step_limit":200,
    "formal_expected":len(expected),
    "formal_valid":len(valid),
    "formal_valid_by_variant":finished_counts,
    "table1_review_condition":"H-Q",
    "reviewer_id":"qwen4-max-preview",
    "reviewer_api_model":"qwen4.0-max-preview",
    "formal_expected_by_variant":expected_counts,
    "missing_or_incomplete":[list(item) for item in sorted(expected-valid.keys())],
    "runtime_git_base":(DELIVERY/"runtime-base-commit.txt").read_text().strip(),
    "delivery_branch":BRANCH,
    "delivery_branch_commit":(DELIVERY/"delivery-branch-commit.txt").read_text().strip(),
    "runtime_source_sha256":digest,
}
readme=f"""# DAMCS Table 1 and Table 2: 200-step formal campaign

Formal coverage at export: **{len(valid)}/{len(expected)} episodes**.
Status: {"Complete formal experiment." if manifest["complete"] else "Interim snapshot. Rows without completed data remain blank."}

From an extracted folder, clone the complete branch with:

```bash
git clone --branch {BRANCH} source/damcs-table1-table2-step200.bundle damcs-reproduction
```

The bundle contains the full branch history and the final source tree. The bundle
commit records the working-tree changes that were present in the live experiment.
Run metadata records the original base commit and the SHA-256 of that same source.
The runtime model configuration and reviewer instructions are in the branch.

`paper/` contains Markdown, LaTeX and CSV tables, paired H-Q versus C differences, run
diagnostics and rule-planner statistics. `raw/` contains formal matrices,
aggregate records and completed runs with their original logs and review files.
The 350-step reference cohort is stored separately under reference_350/.\nThe 200-step main study runs the complete 36-episode matrix.\nActive runs remain in the live campaign directory. Temporary restart
checkpoints remain on the server. Account credentials stay local to the host.

The automatic agents use GLM-5.3 via Token Plan and text-embedding-v4 via the
on-demand endpoint. The six H-Q runs use Qwen4-Max-Preview review through\nToken Plan. The paper tables identify these as external model review. The protocol and
dependency freeze are included under `reproducibility/`.
"""
tmp=DELIVERY/"damcs_table1_table2_step200.zip.tmp"
dest=DELIVERY/"damcs_table1_table2_step200.zip"
written=set()
with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    def add(path, arc):
        if arc in written or not path.is_file() or path.is_symlink() or path.name.endswith((".tmp",".pkl")):
            return
        if path.suffix in {".txt",".log",".json",".jsonl",".csv",".md",".yaml",".tex",".html"}:
            if re.search(rb"sk-(?:sp-|ws-)?[A-Za-z0-9_.-]{23,}",path.read_bytes()):
                raise SystemExit("Potential credential found in deliverable file: "+arc)
        z.write(path,arc)
        written.add(arc)
    z.writestr("README.md",readme)
    z.writestr("MANIFEST.json",json.dumps(manifest,indent=2)+"\n")
    for f in sorted(DELIVERY.glob("*")):
        if f.name.endswith((".bundle",".bundle.sha256")) or f.name in {"source-sha256.txt","runtime-base-commit.txt","delivery-branch-commit.txt"}:
            add(f,"source/"+f.name)
    for f in (CAMPAIGN/"paper_qwen").glob("*"):
        add(f,"paper/"+f.name)
    for f in sorted((CAMPAIGN/"formal").glob("*/*")):
        if f.name in {"matrix.json","episodes.json","episodes.csv","qwen_reviewer_usage.csv","qwen_reviewer.jsonl"}:
            add(f,"raw/"+f.relative_to(CAMPAIGN/"formal").as_posix())
    for key,row in sorted(valid.items()):
        folder=Path(row["run_dir"]).resolve()
        if not folder.is_relative_to((CAMPAIGN/"formal").resolve()) or not folder.is_dir():
            raise SystemExit("Formal run folder unavailable for "+str(key))
        for f in sorted(folder.rglob("*")):
            add(f,"raw/"+f.relative_to(CAMPAIGN/"formal").as_posix())
    for name in ("paper_experiment_protocol.md","codex_h_review_system_prompt.md"):
        add(ROOT/"docs"/name,"reproducibility/"+name)
    add(ROOT/"config"/"model_profiles.json","reproducibility/model_profiles.json")
    add(SETUP/"pip-freeze-mirror.txt","reproducibility/pip-freeze.txt")
    for f in sorted(review_code.glob("*")):
        add(f,"reproducibility/qwen-review/"+f.name)
    add(CAMPAIGN/"reference_350/episodes.json","reference_350/episodes.json")
    pilot = CAMPAIGN/"pilot/qwen4_review"
    if (pilot/"pilot_validated.json").exists():
        for f in (pilot/"pilot_validated.json", pilot/"qwen_reviewer_usage.csv", pilot/"qwen_reviewer.jsonl"):
            add(f,"pilot/"+f.name)
        for f in sorted(pilot.glob("*/reviews/review_*.json")):
            add(f,"pilot/"+f.relative_to(pilot).as_posix())
with zipfile.ZipFile(tmp) as check:
    bad=check.testzip()
    if bad: raise SystemExit("Invalid ZIP entry: "+bad)
os.replace(tmp,dest)
with dest.open("rb") as stream:
    checksum=hashlib.file_digest(stream,"sha256").hexdigest()
(DELIVERY/"damcs_table1_table2_step200.zip.sha256").write_text(f"{checksum}  {dest.name}\n")
print("Delivery",dest.name,"size",dest.stat().st_size,"valid",len(valid),"sha256",checksum)
