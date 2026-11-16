"""Open the local review form for a running experiment's review directory."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_system.review_server import ReviewServer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ReviewServer(Path(args.results_dir) / "reviews", args.port)
    print(server.start(), flush=True)
    try:
        while True:
            time.sleep(1)
    finally:
        server.close()
