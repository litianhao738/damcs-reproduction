import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_system.llm_api import describe_llm_configuration


def parse_args():
    parser = argparse.ArgumentParser(description="Check LLM provider configuration.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when the selected provider is not fully configured.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = describe_llm_configuration()
    print(json.dumps(report, indent=2))
    if args.strict and not report["configured"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
