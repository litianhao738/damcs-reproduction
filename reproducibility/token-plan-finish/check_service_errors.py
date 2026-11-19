"""Signal credential or quota failures in the new batch before export."""

import json
import sys
from pathlib import Path

batch = Path(sys.argv[1])
fatal = (b"insufficient_quota", b"arrearage", b"accessdenied", b"billingnotenabled",
         b"invalidapikey", b"invalid_api_key", b"payment_required", b"quotaexhausted")
for path in batch.glob("*/events.jsonl"):
    for line in path.open("rb"):
        if any(flag in line.lower() for flag in fatal):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("event") == "planning_failed":
                print(f"GLM service error at {path.parent.name} step {obj.get('step')}", flush=True)
                sys.exit(1)
