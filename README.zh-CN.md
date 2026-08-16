# DAMCS 复现：基于 LLM Agent 的多智能体 Crafter

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库是 DAMCS 项目的可执行复现与工程化改造：
[LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning](https://happyeureka.github.io/damcs/)。

多个由大语言模型驱动的 Agent 在共享的二维 Crafter 世界中，通过资源采集、
工具制造、分层记忆和结构化协作，最终共同获得钻石。

> 本仓库基于原项目
> [13RENDA/Mcrafter_LLM_Agent](https://github.com/13RENDA/Mcrafter_LLM_Agent)。
> 使用代码时请引用 DAMCS 论文，并遵守上游项目及依赖库的许可证。

## 已复现的能力

- **Multi-Agent Crafter（MAC）**：支持配置 Agent 数量的共享、部分可观测环境。
- **LLM Agent**：根据局部 RGB 观察、环境文本和记忆生成结构化决策。
- **自适应知识图谱记忆**：包含情景、语义、程序、工作和长期记忆。
- **结构化协作**：按照 Agent 角色完成资源采集、工具制造和资源共享。
- **LangGraph 编排**：使用状态图组织环境步骤和 Agent 规划子图。
- **并发规划**：通过 `Send` fan-out/fan-in 并发执行多个 Agent 的 LLM 请求。
- **中断恢复**：每个完整环境步后原子保存环境、Agent、图状态和随机状态。
- **实验工具**：输出 CSV、观察帧、知识图谱、统计图，并支持视频生成。

默认任务是采集钻石，通常需要完成以下依赖链：木材、工作台、木镐、石头、
石镐、煤和铁、熔炉、铁镐，最后采集钻石。

## 项目结构

```text
.
|-- crafter/                   # 多智能体 Crafter 环境与贴图渲染
|-- memory_system/             # Agent、LLM、记忆、知识图谱和规划逻辑
|-- GUI/                       # 原有 HumanAgent GUI 变体
|-- scripts/                   # 环境、LLM 和 Qwen 辅助检查脚本
|-- descriptions/              # 提示词和环境描述
|-- icons/                     # 图表与 GUI 素材
|-- run.py                     # 无界面实验主入口
|-- langgraph_simulation.py    # LangGraph 父图和 Agent 规划子图
|-- checkpointing.py           # SQLite checkpoint 与 LLM 响应缓存
|-- runtime_snapshot.py        # 环境和 Agent 的结构化版本快照
|-- experiment_outputs.py      # 隔离实验目录与 manifest 管理
|-- run_gui_multi_view.py      # 多视图交互 GUI
|-- plot.py                    # 实验统计绘图
|-- make_video_from_results.py # 从观察帧生成视频
|-- requirements.txt           # Windows 友好的 Python 依赖
|-- environment-win.yml        # 推荐的最小 Conda 环境
`-- environment.yml            # 完整导出的 Conda 环境
```

## 环境要求

- Windows、Linux 或 macOS；当前主要测试流程使用 Windows。
- Conda 和 Python 3.10。
- Azure OpenAI、OpenAI 或 Qwen/DashScope 账号。
- GUI 模式需要可用显示环境和 `pygame`。

## 安装

推荐使用较小、跨平台兼容性更好的环境文件：

```powershell
conda env create -f environment-win.yml
conda activate mcrafter
```

也可以在平台兼容时使用完整环境：

```powershell
conda env create -f environment.yml
conda activate mcrafter
```

## 配置 LLM

复制配置模板并填写一个供应商的凭据。不要提交 `.env` 或 API Key 文件。

```powershell
Copy-Item .env.example .env
```

Qwen/DashScope 示例：

```dotenv
LLM_PROVIDER=qwen
QWEN_API_KEY=your-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.7-plus
QWEN_EMBEDDING_MODEL=text-embedding-v4
```

Azure OpenAI 和 OpenAI 示例见 [.env.example](.env.example)。运行完整实验前先检查：

```powershell
python scripts\check_llm_config.py
python scripts\smoke_llm.py --mode both
```

`smoke_llm.py` 会发起真实 API 请求，可能产生费用。

## 启动实验

先执行小规模无绘图测试：

```powershell
python run.py --agent_num 1 --step_num 10 --results_dir results\smoke --no_plot
```

执行六 Agent 实验：

```powershell
python run.py --run_id damcs-01 --agent_num 6 --step_num 350 --max_concurrency 3
```

PowerShell 多行命令使用反引号，不能使用 CMD 的 `^`：

```powershell
python run.py `
  --run_id damcs-01 `
  --agent_num 6 `
  --step_num 350 `
  --max_concurrency 3
```

默认使用 LangGraph。要和原始 Python `while` 循环比较，可选择 legacy：

```powershell
python run.py --orchestrator legacy --agent_num 1 --step_num 10 --no_plot
```

## LangGraph 执行流程

父图执行顺序：

```text
initialize
  -> prepare_actions
  -> step_environment
  -> update_agent_states
  -> Send(plan_agent) × N
  -> reducer
  -> apply_agent_responses
  -> complete_step
  -> 下一步或 END
```

每个 AI Agent 使用独立规划子图：

```text
START -> call_model -> validate_response -> END
```

`Send(plan_agent)` 分支由 LangGraph 内部并发调度。`graph.invoke()` 对调用方
是同步的：系统等待全部 Agent 规划完成并汇合后，才统一更新 Agent 和进入下一
环境步。

例如：

```powershell
python run.py --agent_num 6 --max_concurrency 3
```

最多同时执行 3 个 Agent 规划分支，6 个 Agent 通常分两批完成。实际是否调用
LLM 还取决于 Agent 是否需要重新规划、是否正在睡眠以及缓存是否命中。

## 中断与恢复

LangGraph 模式在每个完整环境步结束后提交一次 SQLite checkpoint。快照包含：

- Crafter 材质地图、世界对象、玩家和聊天记忆；
- 所有 Agent 的动作状态和分层记忆；
- LangGraph 父图状态；
- Python、NumPy 和环境内部随机数状态；
- 原实验产物目录。

启动一个带稳定 run_id 的实验：

```powershell
python run.py --run_id resume-test-01 --agent_num 2 --step_num 30 --max_concurrency 2 --checkpoint_db checkpoints\resume-test.sqlite --no_plot
```

至少完成一个步骤后可按 `Ctrl+C`。使用相同的 `run_id`、`agent_num`、
`step_num` 和 checkpoint 数据库恢复：

```powershell
python run.py --run_id resume-test-01 --agent_num 2 --step_num 30 --max_concurrency 2 --checkpoint_db checkpoints\resume-test.sqlite --no_plot --resume
```

成功时会看到：

```text
Experiment output: ...\20260816T064736962414Z_resume-test-01
Resuming run 'resume-test-01' from completed step 3.
Progress: 10% ... 3/30
```

恢复会复用原实验目录。已经成功缓存的 Agent LLM 响应不会重复请求；中断时仍在
进行、尚未写入缓存的请求可能重新调用。

限制：

- `--resume` 目前只支持 `--orchestrator langgraph`；
- 恢复只支持 `--num_of_turn 1`；
- `agent_num` 和 `step_num` 必须与 checkpoint 一致；
- 第一个完整环境步提交前中断时还没有可恢复 checkpoint；
- 旧版未记录 `output_dir` 的 checkpoint 会被安全拒绝。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--agent_num` | `6` | 共享环境中的 Agent 数量。 |
| `--step_num` | `350` | 每轮最大环境步数。 |
| `--num_of_turn` | `1` | 重复实验轮数。 |
| `--results_dir` | `results` | 自动创建 UTC/run-id 实验目录的根目录。 |
| `--no_plot` | 关闭 | 不显示和生成最终统计图。 |
| `--orchestrator` | `langgraph` | 使用 `langgraph` 或 `legacy` 编排。 |
| `--run_id` | 自动生成 | checkpoint 使用的稳定运行标识。 |
| `--resume` | 关闭 | 从指定 run_id 的最新完整环境步恢复。 |
| `--checkpoint_db` | `checkpoints/damcs.sqlite` | checkpoint 和 LLM 缓存数据库。 |
| `--max_concurrency` | `6` | LangGraph 同时运行的最大 Agent 规划分支数。 |
| `--human_agents` | 空 | 排除这些 ID 的 LLM 规划；headless 人工输入目前尚未接通。 |

## 实验产物

每次新实验创建 Windows 安全的独立目录，例如：

```text
results/20260816T031522123456Z_damcs-01/
  manifest.json
  all_rounds_stats.csv
  result_fig.png                 # 使用 --no_plot 时没有该文件
  round_001/
    stats.csv
    agent_0/
      0.png                      # RGB 局部观察
      0_stats.png                # 状态图
      knowledge_graph_1.html
    agent_1/
  round_002/                     # num_of_turn > 1 时存在
```

`manifest.json` 保存实验参数、运行状态、轮次信息和恢复次数。独立实验不会再把
统计结果追加到同一个 `all_rounds_stats.csv`，因此不会相互覆盖或混合。

## GUI 与人工控制

启动多视图 GUI：

```powershell
python run_gui_multi_view.py --agent_num 4 --step_num 150 --fps 5
```

启用指定人工 Agent：

```powershell
python run_gui_multi_view.py --human_agents 0,2 --agent_num 4 --step_num 150
```

按键：

- `v`：切换单 Agent、多 Agent 和全局视图；
- `l`：切换水平、垂直和网格布局；
- `0`–`9`：选择单视图跟随的 Agent；
- `h`：进入 HumanAgent 动作输入；
- `p`：暂停或继续；
- `Esc`：退出。

当前边界：`run.py --human_agents` 只把指定 Agent 排除在 LLM 规划之外，并不会
调用终端 `input()`。人工操作请使用 GUI。LangGraph 原生的持久化人工审批
（`interrupt/Command(resume)`）尚未实现。

## 不调用 LLM 的环境检查

```powershell
python scripts\smoke_env.py --agent_num 2 --step_num 5 --results_dir results\env_smoke
```

该命令用于检查 Crafter、渲染依赖和多人环境推进，不发送模型请求。

## 图表和视频

从某轮保存的 Agent 帧生成视频：

```powershell
python make_video_from_results.py --results_dir results\20260816T031522123456Z_damcs-01\round_001 --output_dir videos --multi_panel --fps 5
```

`plot.py` 提供统计绘图函数。主运行器默认生成 `result_fig.png`，使用
`--no_plot` 可以跳过。

## 测试

在 `mcrafter` 环境中运行：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：

- 父图与 legacy 单步业务顺序一致；
- Agent 规划子图真实并发；
- 环境终止路由；
- LLM 响应缓存；
- 结构化环境/Agent 快照；
- 模拟崩溃后从上一完整步骤恢复；
- UTC 实验目录和 manifest 生命周期。

## 常见问题

| 问题 | 处理方式 |
| --- | --- |
| `LLMConfigurationError` 或鉴权失败 | 检查 `.env` 和当前终端环境变量，然后运行 `scripts\check_llm_config.py`。 |
| Qwen 返回格式校验失败 | 程序会自动发送纠正请求；持续失败时检查模型能力和响应日志。 |
| GUI 无法打开 | 激活 `mcrafter`，安装 GUI 依赖，并确认当前会话有显示环境。 |
| 实验速度慢或费用高 | 先使用 `--agent_num 1 --step_num 10 --no_plot`。 |
| 找不到实验结果 | 查看启动时打印的 `Experiment output:`，或在 `results_dir` 下搜索 `manifest.json`。 |
| 恢复提示没有 checkpoint | 确认已至少完成一步，并使用相同 `run_id` 和 `checkpoint_db`。 |

## 与原 DAMCS 的关系

本仓库是复现与工程改造，不是 DAMCS 官方代码发布。整体遵循论文中的多智能体
Crafter、分层知识图谱记忆和结构化协作思路，但加入了本地运行配置、Qwen 支持、
LangGraph 编排、结构化 checkpoint、LLM 缓存和实验产物隔离。精确复现论文指标
仍需要匹配论文的模型、随机种子、提示词、实验轮数和评估协议。

## 引用

```bibtex
@misc{yang2025damcs,
  title={LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning},
  author={Hanqing Yang and Jingdi Chen and Marie Siew and Tania Lorido-Botran and Carlee Joe-Wong},
  year={2025},
  eprint={2502.05453},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2502.05453}
}
```

原方法和评估请参考 [DAMCS 项目主页](https://happyeureka.github.io/damcs/)
以及 [arXiv 论文](https://arxiv.org/abs/2502.05453)。
