# DAMCS 表 1、表 2 实验规程

本规程以 `feat/agentscope-coordination-repair` 的 `d20cb582865a8e911a5b65076975a6f1e59874fd` 为基础。执行程序记录 Git HEAD 和实际源码哈希，所有正式组共用同一份工作树、依赖和模型配置。

## 分支与实验条件

| 分支 | 提交 | 与实验有关的实现 |
| --- | --- | --- |
| `main` | `600caed` | 共享的 MA-Crafter 环境、DAMCS 记忆、分工和技能执行 |
| `dev` | `20fc884` | LangGraph 调度、SQLite 断点和实验输出管理 |
| `feat/agentscope-coordination-repair` | `d20cb582` | AgentScope、多 Agent 联合修复、进度规划、人审、CUBE 适配和实验导出 |

正式组从实验分支自身的 A 开始，以保持环境、模型接入和动作保护一致。最新提交把最多两轮重规划与联合范围 S 归入合法性修复 B。自动进度修复是 C，H 是 C 加关键时刻的 Codex 外部审阅。早期实验表中的 D 对应当前 C。S 会继续记录触发、可行性和提交次数，联合修复的能力用受控用例和正式快照重放验证。

| 表 | 条件 | 要回答的问题 |
| --- | --- | --- |
| 1 | A、B、C、H 各 6 局 | H 相对 C 的 Codex 外部审阅净收益 |
| 2 | C、C-no-legality、B、C-no-llm 各 6 局 | 合法性及进度模块的增量，以及规则规划器本身的能力 |

表 2 复用表 1 的 B 和 C。整批共 36 个独立运行，其中 6 局由 Codex 值守。每个条件在种子 4、5、6 上各跑两次，重复编号为 0、1。每局 6 个 Agent，最多 350 步。

## 表 1 估计量

预先指定 H−C 为主要对比。用同一地图种子与重复编号配对，报告每对差值、每张图的两次平均差值，以及 3 张图的描述性均值和样本标准差。未采钻的有效局记 350 步，同时给出实际成功局数。回退、超时、回复失效和没有审阅请求的值守局都进入 H。执行故障、网络中断或审阅者缺席单独登记，按原矩阵补齐。

净收益同时报告采钻步数、成功局数、实际运行秒数、GLM 和向量嵌入的 API 花费、暂停 Agent 步数和请求从打开到提交的时长。报告请求、回复、超时、失效和合法性/进度请求分布，各量保留各自单位。每局记录审阅者编号、开始和结束时间、是否持续值守、意外中断、处理顺序。审阅权限与证据遵循 `codex_h_review_system_prompt.md`。H 的身份和暴露条件在论文方法中披露。脚本审阅进入机制附表。

## 表 2 规则能力与机制证据

`C-no-llm` 将每个 Agent 的 LLM 初始提案设为 noop，保留环境状态、语义地图、配方、候选分工和锁定执行，由规则进度规划器推进任务。这一行独立报告成功局数、采钻步数、阶段达成、锁定 Agent 步占比、锁定任务完成数与 API 使用量。对话、Embedding 的请求数及 token 都必须为 0。C−C-no-llm 描述相同机制上叠加 LLM 后的变化。

| 机制主张 | 同一输入下的对照 | 记录 |
| --- | --- | --- |
| 合适的修改对象可能是队友 | `provider_only` 与同一正式快照下的当事人范围和联合验证范围 | 编辑对象、字段、恢复目标实际达成 |
| 单独可行的修改合并后可能冲突 | `shared_unit` 与多个目标各有单独候选的正式日志 | 共用资源冲突、联合覆盖率及分母 |
| 约束裁剪可减少呈现内容 | 同一合法性修复的精简与完整证据 | 事实组数、修复验证结果 |
| 局部等待与增量复核的表现 | `Hscript`、`Hscript-global-wait`、`Hscript-no-recheck` | 暂停 Agent 步数、失效处理、复核时间 |

报告实证和受控案例各自的观察范围。真实快照没有目标时，报告分母为 0。仅凭事实组数可报告展示量，真人阅读耗时与判断准确度需要有独立对照数据。

## 启动顺序

1. 在 AutoDL 检出固定提交，准备独立 Python 环境，保存依赖清单。运行仓库现有的实验、进度、修复范围、审阅和断点检查。
2. 跑规则规划器的表外种子 7、8、受控案例，以及 B/C 的 4 局表外在线试跑。H 用表外种子值守练习一次，核对请求频率和局部等待。拟冻结参数为 `slow_margin=10`、`stall_steps=35`、审阅冷却 10 步、超时 120 秒、开放容量 2、局部等待、增量复核和精简证据。若依据试跑改动参数，更新整个正式批次。
3. 在正式种子启动前记录模型 ID、版本、实际生成参数和有效价格。在线 Agent 使用百炼 Token Plan 的 `glm-5.3`，统一采用 `reasoning_effort=low` 与相同的文本观测、状态和环境历史。表外试跑观察到计划输出截断后，采用该推理参数重测。向量嵌入统一使用按量端点的 `text-embedding-v4`。表中费用为按量等价估算，每百万 token 分别为对话输入 8 元、输出 28 元、Embedding 输入 0.5 元。GLM 实际扣减的套餐额度以百炼账单为准。
4. 运行正式规则组、24 局在线自动组，以及 6 局 Codex 外部审阅。每局 H 仅在审阅者在线连接、可以实时读取审阅请求时，通过 `formal-h SEED REPEAT` 独立启动。所有分组持续写入 `episodes.json` 和 `episodes.csv`。完成后导出论文表与配对数据。

在服务器项目根目录运行下列命令。凭据通过项目忽略的 `local_config` 或服务器环境变量读取。

```bash
export DAMCS_PYTHON=/root/autodl-tmp/damcs-paper-py312-mirror/bin/python
bash scripts/run_paper_campaign.sh pilot-rule
bash scripts/run_paper_campaign.sh cases
bash scripts/run_paper_campaign.sh pilot-format
bash scripts/run_paper_campaign.sh pilot-online
bash scripts/run_paper_campaign.sh pilot-ai
bash scripts/run_paper_campaign.sh formal-rule
bash scripts/run_paper_campaign.sh formal-online
bash scripts/run_paper_campaign.sh formal-h 4 0
bash scripts/run_paper_campaign.sh replay
bash scripts/run_paper_campaign.sh export
```

Codex 逐条读取运行目录中与审阅界面对应的请求字段，回复注明 `ai_external`，执行器在接收后按原规则复核。进程中断时用相同启动参数恢复。更改源码、模型或流程后新开正式批次。

表外审阅演练仍写入试跑目录。

## 论文输出

`results/paper_t1_t2_codex_token_plan/paper/` 导出 Markdown、CSV、LaTeX、H−C 完整配对和代价、逐局规则能力及诊断。表中同时给出有效数与应跑数，技术故障、缺失、有效失败分别登记。每个种子只有两次重复，地图级估计只有 3 个，本批次以描述性统计为主。
