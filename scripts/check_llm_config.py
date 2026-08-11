# 模块说明：检查当前大模型供应商配置是否完整。
import argparse
import json
import sys
from pathlib import Path

# 变量说明（PROJECT_ROOT）：项目根目录的绝对路径，用于让独立脚本导入仓库模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_system.llm_api import describe_llm_configuration


def parse_args():
    # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
    parser = argparse.ArgumentParser(description="Check LLM provider configuration.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when the selected provider is not fully configured.",
    )
    return parser.parse_args()


def main():
    # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
    args = parse_args()
    # 变量说明（report）：准备输出到终端或返回给调用方的检查结果字典。
    report = describe_llm_configuration()
    print(json.dumps(report, indent=2))
    if args.strict and not report["configured"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
