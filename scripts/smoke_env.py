import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import crafter
from crafter import constants as const


def parse_args():
    parser = argparse.ArgumentParser(description="Run a no-LLM Crafter environment smoke test.")
    parser.add_argument("--agent_num", type=int, default=2, help="Number of players to initialize.")
    parser.add_argument("--step_num", type=int, default=5, help="Number of noop environment steps to run.")
    parser.add_argument("--seed", type=int, default=4, help="World seed.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for the smoke report.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.agent_num < 1:
        raise ValueError("--agent_num must be at least 1")
    if args.step_num < 1:
        raise ValueError("--step_num must be at least 1")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = crafter.Env(length=args.step_num, n_players=args.agent_num, seed=args.seed)
    obs = env.reset()
    if args.agent_num == 1:
        obs_list = [obs]
    else:
        obs_list = list(obs)

    noop_action = const.actions.index("noop")
    actions = [noop_action] * args.agent_num
    rewards = None
    done = False
    info = None

    for _ in range(args.step_num):
        obs, rewards, done, info = env.step(actions if args.agent_num > 1 else actions[0])
        if done:
            break

    report = {
        "agent_num": args.agent_num,
        "requested_steps": args.step_num,
        "completed": bool(done),
        "initial_observation_count": len(obs_list),
        "initial_observation_shape": list(obs_list[0].shape),
        "last_rewards": rewards,
        "last_info_count": len(info) if isinstance(info, list) else 1,
    }

    report_path = results_dir / "smoke_env_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Smoke report written to {os.fspath(report_path)}")


if __name__ == "__main__":
    main()
