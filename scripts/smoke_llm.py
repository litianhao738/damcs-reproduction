import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_system.llm_api import (
    describe_llm_configuration,
    get_completion,
    get_client,
    get_llm_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal LLM smoke test.")
    parser.add_argument(
        "--mode",
        choices=["embedding", "chat", "both"],
        default="embedding",
        help="Which LLM API path to test.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 instead of skipping when LLM config is missing.",
    )
    return parser.parse_args()


def smoke_embedding():
    config = get_llm_config()
    response = get_client().embeddings.create(
        model=config.embedding_model,
        input=["Mcrafter LLM agent smoke test"],
    )
    return {"embedding_dimensions": len(response.data[0].embedding)}


def smoke_chat():
    messages = [
        {
            "role": "system",
            "content": "You generate concise, valid structured planning records.",
        },
        {
            "role": "user",
            "content": (
                "Create a minimal valid planning record for agent 0 at episode 1, "
                "timestep 0. The agent sees grass, has no inventory, did noop, "
                "has ultimate_goal collect_diamond, long_term_goal place_table, "
                "current_goal collect_wood, and plans to navigate to a tree. "
                "Use only valid enum values."
            ),
        },
    ]
    parsed = get_completion(messages)
    return {
        "episode": parsed.epsiode_number,
        "timestep": parsed.timestep,
        "current_goal": parsed.goal.current_goal.value,
        "final_next_action": parsed.action.final_next_action.value,
    }


def main():
    args = parse_args()
    config_report = describe_llm_configuration()
    if not config_report["configured"]:
        print(json.dumps({"skipped": True, "config": config_report}, indent=2))
        return 1 if args.strict else 0

    result = {"skipped": False, "config": config_report}
    if args.mode in {"embedding", "both"}:
        result["embedding"] = smoke_embedding()
    if args.mode in {"chat", "both"}:
        result["chat"] = smoke_chat()

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
