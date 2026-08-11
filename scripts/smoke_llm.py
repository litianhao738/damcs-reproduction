# 模块说明：执行聊天模型和向量模型的连通性测试。
import argparse
import json
import sys
from pathlib import Path

# 变量说明（PROJECT_ROOT）：项目根目录的绝对路径，用于让独立脚本导入仓库模块。
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
    # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
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
    # 变量说明（config）：根据环境变量解析得到的大模型供应商配置。
    config = get_llm_config()
    # 变量说明（response）：当前 API 调用、任务或模型推理产生的响应对象。
    response = get_client().embeddings.create(
        model=config.embedding_model,
        input=["Mcrafter LLM agent smoke test"],
    )
    return {"embedding_dimensions": len(response.data[0].embedding)}


def smoke_chat():
    # 变量说明（messages）：按角色排列、准备提交给聊天模型的消息列表。
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
    # 变量说明（parsed）：经过 JSON 解析和 Pydantic 校验后的结构化模型响应。
    parsed = get_completion(messages)
    return {
        "episode": parsed.epsiode_number,
        "timestep": parsed.timestep,
        "current_goal": parsed.goal.current_goal.value,
        "final_next_action": parsed.action.final_next_action.value,
    }


def main():
    # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
    args = parse_args()
    # 变量说明（config_report）：`main` 中当前处理的配置。
    config_report = describe_llm_configuration()
    if not config_report["configured"]:
        print(json.dumps({"skipped": True, "config": config_report}, indent=2))
        return 1 if args.strict else 0

    # 变量说明（result）：`main` 中当前处理的结果。
    result = {"skipped": False, "config": config_report}
    if args.mode in {"embedding", "both"}:
        result["embedding"] = smoke_embedding()
    if args.mode in {"chat", "both"}:
        result["chat"] = smoke_chat()

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
