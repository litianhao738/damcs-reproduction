# 模块说明：执行不调用大模型的环境冒烟测试。
import argparse
import json
import os
import sys
from pathlib import Path

# 变量说明（PROJECT_ROOT）：项目根目录的绝对路径，用于让独立脚本导入仓库模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import crafter
from crafter import constants as const


def parse_args():
    # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
    parser = argparse.ArgumentParser(description="Run a no-LLM Crafter environment smoke test.")
    parser.add_argument("--agent_num", type=int, default=2, help="Number of players to initialize.")
    parser.add_argument("--step_num", type=int, default=5, help="Number of noop environment steps to run.")
    parser.add_argument("--seed", type=int, default=4, help="World seed.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for the smoke report.")
    return parser.parse_args()


def main():
    # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
    args = parse_args()
    if args.agent_num < 1:
        raise ValueError("--agent_num must be at least 1")
    if args.step_num < 1:
        raise ValueError("--step_num must be at least 1")

    # 变量说明（results_dir）：保存统计表、观察帧和其他实验产物的目录。
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 变量说明（env）：当前 Crafter 仿真环境实例。
    env = crafter.Env(length=args.step_num, n_players=args.agent_num, seed=args.seed)
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    obs = env.reset()
    if args.agent_num == 1:
        # 变量说明（obs_list）：将单玩家观察规范化为列表后的观察集合。
        obs_list = [obs]
    else:
        obs_list = list(obs)

    # 变量说明（noop_action）：`main` 中当前处理的动作。
    noop_action = const.actions.index("noop")
    # 变量说明（actions）：按智能体编号排列、将提交给环境的动作列表。
    actions = [noop_action] * args.agent_num
    # 变量说明（rewards）：本时间步每个智能体获得的奖励值。
    rewards = None
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    done = False
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    info = None

    for _ in range(args.step_num):
        obs, rewards, done, info = env.step(actions if args.agent_num > 1 else actions[0])
        if done:
            break

    # 变量说明（report）：准备输出到终端或返回给调用方的检查结果字典。
    report = {
        "agent_num": args.agent_num,
        "requested_steps": args.step_num,
        "completed": bool(done),
        "initial_observation_count": len(obs_list),
        "initial_observation_shape": list(obs_list[0].shape),
        "last_rewards": rewards,
        "last_info_count": len(info) if isinstance(info, list) else 1,
    }

    # 变量说明（report_path）：`main` 读取或写入当前文件时使用的完整路径。
    report_path = results_dir / "smoke_env_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Smoke report written to {os.fspath(report_path)}")


if __name__ == "__main__":
    main()
