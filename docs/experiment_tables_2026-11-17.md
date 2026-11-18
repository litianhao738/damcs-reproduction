# 实验表设计

日期：2026-11-17。本文定义要跑的三张表、统一设置、指标和运行方法。表中空格在实验完成后填写；未填的格子不得用估计值代替。

> **暂缓开跑（2026-11-18）**：C、H1、H2 将扩展为同时修复"合法但慢"的计划（进度问题），表 2 将增加 `C-legal-only`、`C-no-commit` 两行，详见 [进度问题修复设计](progress_repair_design_2026-11-18.md)。这项改动实现并通过测试之前，表 1–3 都不要开跑；现有代码跑出的 C、H1、H2 结果不能与改动后的结果放进同一张表。

## 统一设置

主环境是 MA-Crafter（挖钻石），三张表都在这里完整运行。CUBE 只在表 1 中跑 B、C、H2 三个条件，用来检验机制换到另一个环境是否仍然有效；表 2、表 3 不跑 CUBE。

| 项目 | MA-Crafter（主环境） | CUBE（仅表 1） |
| --- | --- | --- |
| Agent 数 | 6 | 3 |
| 步数上限 | 350 | 30 |
| 地图与任务 | DAMCS 原版，团队目标是采到钻石 | 15×15，重量 2、3 的箱子各 1 个 |
| 种子 | 4、5、6 | 4、5、6 |
| 每种子重复 | 表 1、表 2 为 2 次；表 3 为 1 次 | 2 次 |

其余设置：

- 重规划最多 2 轮。
- 人审不设次数上限，只在**关键时刻**问人，其余问题一律自动修复。关键时刻用与环境无关的三个信号判断（`--review_trigger signals`，默认），满足任意一个即可：
  - **要做取舍**（`choice`）：能修好这个问题的方案不止一个，而且改的是不同的 Agent。机器只会按"少改人"机械地选，该打断谁需要人判断。
  - **拿不准**（`unknown`）：信息不全，无法确定能不能修好。
  - **团队卡住**（`stalled`）：团队进度连续一段时间没有增长，此时存在的问题都算。这段时间按每局步数的比例计算（`--stall_fraction 0.1`），Crafter 350 步对应 35 步，CUBE 30 步对应 3 步；`--stall_steps` 可改为固定步数。进度由环境提供：Crafter 是全队解锁的不同成就数，CUBE 是团队得分。
  - 同一个问题问过之后 10 步内不再重复问（`--review_cooldown 10`），避免一个卡住的问题每步都打扰审阅者。
  - 一次人审最多开放 2 个 Agent（`--review_max_agents 2`）。H2 会同时开放能修好问题的其他已验证方案涉及的 Agent（不超过 2 个），让人可以选择改谁；"拿不准"时开放问题涉及的 Agent。
  - 每局问人的次数因此不固定，要作为结果报告，并按信号分别统计（`requests_choice`、`requests_unknown`、`requests_stalled`）。
  - 旧的按阶段固定名单（Crafter：铁镐、采铁、采钻石；CUBE：每次推箱失败）保留为 `--review_trigger stage`，可作对照；`--review_trigger all` 对所有问题都问人，只用于练习。
- 人审默认使用局部等待、增量复核、精简证据，与表 2 的完整系统一致。
- 修复搜索不设人数上限，按"先修得全、再改得少"选择。
- 对话模型通过 `config/model_profiles.json` 中的命名配置选择。表 1、表 2 固定用 `qwen`；表 3 只改变对话模型。
- Embedding 在所有表中固定为配置文件里的同一个模型，与对话模型分开调用。
- 所有模型都使用同一种输出方式：要求模型返回 JSON，本地校验，出错时带着错误信息重试，重试次数记入 `chat_json_retries`。
- 所有条件（包括 A）都使用 DAMCS 底座：分层记忆、固定分工提示、技能执行器和动作合法性检查。各条件的差别只在其上叠加的机制。
- 统计以完整一局为单位；失败局与技术错误局都保留，并单独标出技术错误（`exit_code` 非 0 或 `completed` 为假）。

## 指标定义

Crafter 的主要指标是"采到钻石用了多少步"，不是成功率。已有记录中，qwen3.7-plus 在 6 人、150 步的设置下分别在第 91 步和第 134 步采到钻石；在 350 步上限下，各条件的成功率很可能都接近 100%，无法区分条件差异。

所有指标都由 `scripts/run_experiments.py` 汇总到每批结果的 `episodes.csv`。

| 指标 | 含义 | 汇总字段 |
| --- | --- | --- |
| 钻石步数 | Crafter：团队第一次采到钻石的步数。未采到时按步数上限计，并标记为截断 | `diamond_step`、`diamond_step_or_limit`、`diamond_censored` |
| 阶段步数 | Crafter：首次完成各阶段（木镐、石镐、铁镐等）的步数 | `step_collect_wood` … `step_collect_diamond` |
| 采到钻石比例 | Crafter：350 步内采到钻石的局数比例 | `task_success` |
| 里程碑数 | Crafter：10 个阶段里完成了几个 | `milestones_reached` |
| CUBE 成功率、得分 | 全部箱子交付的比例；团队得分 | `task_success`、`team_score` |
| 问题数/局 | 出现需要修复的协调问题的规划步数 | `target_snapshots` |
| 有方案比例 | 出现问题的步中，搜索找到可修复方案的比例 | `selected_snapshots / target_snapshots` |
| 修复生效率 | 提交的修复在实际执行中达成目标事件的比例 | `recoveries / recovery_attempts` |
| 平均改动人数 | 每次提交的修复改了几个 Agent 的计划 | `mean_edited_agents` |
| 被打断的原计划数 | 修复把本来在做别的事的帮手改去帮忙的次数 | `interrupted_plans` |
| 超出人审上限 | 最小修复超过 2 人、改走自动修复的次数 | `over_review_capacity` |
| 人审次数、阅读时间 | 发出的请求数、真人阅读与修改用时 | `requests_issued`、`human_seconds` |
| 问人的原因 | 因"要做取舍""拿不准""团队卡住"而发出的请求数（一个请求可能同时满足多个） | `requests_choice`、`requests_unknown`、`requests_stalled` |
| 未问人的原因 | 不是关键时刻、同一问题在冷却期内、超出人审上限等各自的次数 | `not_critical_skips`、`over_review_capacity`；其余见 `events.jsonl` 的 `review_skipped` |
| 技术失败 | 模型调用失败、该 Agent 沿用上一步计划的次数；JSON 格式错误后重试的次数 | `planning_failures`、`chat_json_retries` |
| 人审回复结果 | 各种回复状态的次数（采纳、过期、无效、超时等） | `reply_<状态>`，如 `reply_stale_plan` |
| 过期修复被提交数 | 关闭提交前复核时，本该被拦下却被提交的修复数 | `validation_unchecked_would_fail` |
| 复核耗时与次数 | 提交前复核的总时间和次数 | `validation_seconds`、`validation_*` |
| 被暂停的 Agent 步数 | 等待审阅期间停下的 Agent 步数之和 | `paused_agent_steps` |
| 证据事实组数 | 精简前后展示给人的事实组数 | `evidence_groups_before`、`evidence_groups_after` |
| 搜索耗时 | 修复范围搜索的总时间 | `search_seconds` |
| 模型调用与 token | 对话及 Embedding 请求次数和 token | `chat_requests`、`chat_input_tokens`、`chat_output_tokens`、`embedding_tokens` |
| 每局花费 | token × 配置文件中登记的单价；单价未填时为空 | `episode_cost` |

## 表 1：主实验——人审闭环是否有增量价值

固定 Qwen。每格写均值，括号内写标准差或局数。CUBE 只跑 B、C、H2，其余条件的 CUBE 列填"—"。

| 条件 | 含义 | Crafter 钻石步数 | Crafter 采到钻石比例 | Crafter 里程碑数 | CUBE 成功率 | CUBE 得分 | 修复生效率 | 人审次数/局 | 阅读时间/局 | 每局花费 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | DAMCS 底座：各自规划 + 动作合法性检查 | | | | — | — | — | — | — | |
| B | A + 最多 2 轮重规划 | | | | | | — | — | — | |
| C | B + 自动执行修复方案（最强自动对照） | | | | | | | — | — | |
| H1 | 人审，开放出问题的 Agent | | | | — | — | | | | |
| H2 | 人审，开放可修复范围 S（主方法） | | | | | | | | | |

- **人审只保留 H1 和 H2 两种**：
  - H1 是最直接的做法：谁出了问题，就把谁的计划开放给人。
  - H2 是本项目的方法：只开放能修好问题的最少几个人，以及修复实际要改的字段。
  - 编号变更（2026-11-17）：旧方案中开放整个相关小组全部字段的条件（当时叫 H1）已删除，因为它允许人在较大的范围内自由修改，结果难以量化和比较；原来的 H0 改名为 H1。旧文档中的 H0 即现在的 H1。
- **规模**：
  - Crafter：5 个条件 × 3 个种子 × 2 次重复，共 30 局，其中 H1、H2 的 12 局需要真人审阅。
  - CUBE：3 个条件 × 3 个种子 × 2 次重复，共 18 局，其中 H2 的 6 局需要真人审阅。
  - 合计 48 局，其中真人审阅 18 局。每局用 `--reviewer_id` 记录审阅者编号，审阅顺序另行登记。
  - 审阅者的操作步骤、判断标准、条件顺序轮换和登记表见 [审阅规程](review_protocol_2026-11-18.md)。
- **主要比较**：
  - A→B→C 逐级给出重规划（B − A）和修复搜索（C − B）的作用。
  - H2 对 C，看人审有没有超过自动修复。
  - H2 对 H1，看"开放能修好问题的人"是否优于"开放出问题的人"。
  - CUBE 列看 B、C、H2 的相对关系在另一个环境中是否一致。
- **脚本审阅**：只能检查流程，不填入本表。

## 表 2：部件消融——每个部件是否真的起作用

固定 Qwen，只跑 Crafter。在完整系统上每次只去掉或替换一个部件，其余设置不变。每个部件都有一个它直接负责的"对应指标"，判断依据是这个指标是否变差；钻石步数作为次要结果一起报告。

- **自动修复部分**：以表 1 的 C 为完整系统，数值直接引用表 1，不在本表重复列出。
  - 表 1 的 A→B→C 已经逐级给出重规划和修复搜索的作用。
  - 本表只列表 1 看不到的部件。
  - 本表的"− 定向重规划"是在保留修复搜索的前提下去掉重规划，回答"有了修复搜索，重规划是否还必要"，与表 1 的 B − A 是不同的问题。
- **人审部分**：以 H2 为完整系统，用脚本审阅（固定延迟 2 步、总是采用建议），避免真人差异混入。所以这里测的是机制本身，不是人的价值。表中写作"H2（脚本）"，与表 1 的真人 H2 区分。

| 组 | 版本 | 去掉或替换了什么 | 对应指标 | 对应指标结果 | 修复生效率 | 钻石步数 | 运行名（`variant`） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 自动 | − 定向重规划 | 保留修复搜索，重规划轮数 2 → 0 | 进入修复搜索的问题数/局 | | | | `C-no-replan` |
| 自动 | − 联合检查 | 每个问题单独找方案再合并 | 合并后仍成立的比例、冲突提交数 | | | | 离线：`evaluate_frozen_scopes.py` |
| 自动 | − 最小范围 | 在可行方案中不按"先修得全、再改得少"选择，改为任取一个（按状态固定选取，可复现） | 平均改动人数、被打断的原计划数 | | | | `C-any-scope` |
| 人审 | H2（脚本）完整 | 局部等待 + 增量复核 + 精简证据 | — | | | | `H2script` |
| 人审 | 局部等待 → 全局等待 | 审阅期间所有 Agent 都停下 | 被暂停的 Agent 步数 | | | | `H2script-global-wait` |
| 人审 | 局部等待 → 不等待 | 审阅期间照常执行，回复后再复核 | 过期回复数、被拒提交数 | | | | `H2script-no-wait` |
| 人审 | 增量复核 → 完整复核 | 每次都复核全队 | 复核耗时、复核结论是否一致 | | | | `H2script-full-recheck` |
| 人审 | − 提交前复核 | 回复后不再检查当前状态，直接提交 | 过期修复被提交数、修复生效率 | | | | `H2script-no-recheck` |
| 人审 | 精简证据 → 完整证据 | 展示全部相关事实 | 展示的事实组数 | | — | — | 离线：同一局同时记录前后组数 |

- **规模**：在线 7 个版本 × 3 个种子 × 2 次重复，共 42 局。两行离线计算不调用模型。预算紧时重复次数降为 1，共 21 局。
- **证据部件**：脚本审阅只能证明展示的内容变少了。证据是否缩短阅读时间、是否影响判断正确率，需要真人小试，结果只在小试完成后报告。
- **表 1 的分工**：表 1 已经覆盖"开放哪些 Agent 给人审"（H1 对 H2）这一项，本表不重复。
- **动作合法性保护**：这是所有条件共用的公平性设置，不作为贡献，不做消融。
- **如实报告**：一步修复在一局里出现的次数有限，3 个种子下钻石步数的差异可能不显著。部件是否起作用以对应指标为主要依据；若对应指标也没有变化，要写明该部件在当前设置下没有表现出作用。

## 表 3：跨模型对比——机制的作用是否依赖模型强弱

只跑 Crafter，只改对话模型，其余设置同表 1。主要模型是 Qwen、DeepSeek、Kimi、GLM，另加 GPT 作为国外参照。

基线是条件 A（DAMCS 底座），不是不带 DAMCS 记忆和分工的纯 API。本表要回答的是"本项目加在 DAMCS 之上的机制，包括自动修复 C 和主方法 H2，其作用是否随模型强弱变化"；若以纯 API 为基线，差值中会混入 DAMCS 本身的作用。

每格只填钻石步数的均值，括号内为标准差。

| 模型 | 档位 | A | B | C | H2 | 每局花费 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen | | | | | | |
| DeepSeek | | | | | | |
| Kimi | | | | | | |
| GLM | | | | | | |
| GPT | | | | | | |

- **表注**：未采到钻石的局按步数上限 350 计，并注明每个格子中这类局的数量。
- **怎么读**：
  - C 对 B 看可修复范围 S 单独的作用（核心创新）。
  - H2 对 C 看真人审阅相对自动修复的增量（主方法）。
  - A 本身的数值反映模型原本的强弱。
  - 差值由读者从各列直接得到，表中不另列。
- **正文中要补的一句话**：差值的方向是否在各模型上一致。按种子配对计算（同一种子下 C 减 B、H2 减 C），写明每个模型 3 个种子里有几个变好，例如"5 个模型中 4 个的 C 少于 B，其中 3 个模型 3/3 个种子都变好"。这一信息从均值中看不出来。
- **放到附表的指标**：问题数/局、修复生效率、人审次数和阅读时间、里程碑数、JSON 重试与报错中断（技术失败率）。弱模型若普遍采不到钻石，在附表中用里程碑数和做出铁镐的步数比较。汇总脚本都会导出这些字段。
- **H2 用真人审阅**，设置与表 1 的 H2 相同（局部等待、增量复核、精简证据，只在关键时刻问、不设次数上限，一次最多开放 2 个 Agent）。脚本审阅不能代替，因为它总是采纳建议，结果接近 C。
- **规模**：5 个模型 × 4 个条件 × 3 个种子 × 1 次，共 60 局，其中 H2 的 15 局需要真人审阅。Qwen 一行直接复用表 1 中重复编号为 0 的 A、B、C、H2 局，所以需要新跑 48 局，其中真人审阅 12 局。
- **审阅者分配**：审阅者不能与模型绑定，否则"模型差异"和"审阅者差异"无法区分。按种子分配审阅者，同一个种子的所有模型由同一位审阅者审阅；审阅时尽量不告诉审阅者当前是哪个模型；每位审阅者的模型顺序要轮换，见 [审阅规程](review_protocol_2026-11-18.md)。每局用 `--reviewer_id` 记录审阅者编号。
- **要检验的猜想**：
  - 弱模型出现问题更多，C − A 和 C − B 的改善应更大。
  - 人审的增量 H2 − C 也可能随模型变化：弱模型留给人修的问题更多；强模型的自动修复已经足够时，人审的价值变小。
  - 强模型问题少，机制的作用也小。
  - 所有结果都要如实报告。
- **措辞**：这是同一套任务上的跨模型对比，不另称新 benchmark。

## 运行方法

所有命令都在项目根目录、`mcrafter` 环境里执行。每批结果目录下会生成 `matrix.json`（运行清单）、每局一个子目录，以及汇总表 `episodes.csv`。先加 `--dry_run` 看一遍要跑的命令，确认后去掉它再正式运行；中断后可用 `--summarize_only` 重新汇总已有结果。

### 密钥

密钥只放在本机，不进 git：

- Qwen：沿用 `local_config/qwen_api_key.csv`。
- 其他厂商：新建 `local_config/api_keys.json`，例如：

  ```json
  {"DEEPSEEK_API_KEY": "...", "MOONSHOT_API_KEY": "...", "ZHIPU_API_KEY": "...", "OPENAI_API_KEY": "..."}
  ```

- 也可以直接设置同名环境变量，环境变量优先。

检查某个配置是否完整（不调用模型，不显示密钥）：

```powershell
python scripts\run_with_model.py --profile qwen
```

### 表 1

```powershell
# Crafter 自动条件
python scripts\run_experiments.py --model_profiles qwen --conditions A B C --seeds 4 5 6 --repeats 2 --results_dir results\table1_crafter_auto

# Crafter 真人审阅：每位审阅者按分到的种子运行，操作和判断标准见 docs\review_protocol_2026-11-18.md
# 单号审阅者用 --conditions H1 H2，双号审阅者用 --conditions H2 H1
python scripts\run_experiments.py --model_profiles qwen --conditions H1 H2 --seeds 4 --repeats 2 --reviewer human --reviewer_id R1 --results_dir results\table1_crafter_human_R1

# CUBE
python scripts\run_experiments.py --environment cube --agent_num 3 --step_num 30 --model_profiles qwen --conditions B C --seeds 4 5 6 --repeats 2 --results_dir results\table1_cube_auto
python scripts\run_experiments.py --environment cube --agent_num 3 --step_num 30 --model_profiles qwen --conditions H2 --seeds 4 --repeats 2 --reviewer human --reviewer_id R1 --results_dir results\table1_cube_human_R1
```

### 表 2

```powershell
python scripts\run_experiments.py --suite ablation --model_profiles qwen --seeds 4 5 6 --repeats 2 --results_dir results\table2

# 离线的"联合检查"一行：读取表 1 自动条件运行中保存的问题快照
python scripts\evaluate_frozen_scopes.py --input results\table1_crafter_auto --capacities 2 --output results\table2_joint_check.json
```

### 表 3

```powershell
# 自动条件
python scripts\run_experiments.py --model_profiles deepseek kimi glm gpt --conditions A B C --seeds 4 5 6 --repeats 1 --results_dir results\table3_auto

# 真人审阅：每位审阅者负责一个种子，审阅该种子下所有模型，审阅页面为 http://127.0.0.1:8765
python scripts\run_experiments.py --model_profiles deepseek kimi glm gpt --conditions H2 --seeds 4 --repeats 1 --reviewer human --reviewer_id R1 --results_dir results\table3_human_R1
```

### 单局试跑

```powershell
python scripts\run_with_model.py --profile kimi run.py --condition A --agent_num 6 --step_num 20 --seed 4 --results_dir results\trial_kimi --no_plot
```

### 中断后继续（checkpoint）

Crafter 每局默认每 5 步把完整运行状态存到该局目录下的 `checkpoint.pkl`（`--checkpoint_every 5`，0 为关闭），包括环境、各 Agent 的记忆、人审控制器、计数器和随机数状态。人审请求处理期间不存档；一局正常结束后存档自动删除。

断电、报错或手动中断后，用**原来的命令加 `--resume`** 重新运行即可：

```powershell
python scripts\run_experiments.py --model_profiles qwen --conditions A B C --seeds 4 5 6 --repeats 2 --results_dir results\table1_crafter_auto --resume
```

- 已经跑完的局直接跳过。
- 有存档的 Crafter 局从存档那一步接着跑，存档之后被中断的那几步会重跑一次，日志里中断那次的残留会先删掉。
- 没有存档的局（包括 CUBE，每局只有 30 步，不存档）从头重跑，原来的失败记录改名保留为 `<目录>.failedN`。
- 存档与代码版本绑定：代码改过之后不能接着跑，会报错提示，需要从头重跑这一局。
- 每局 `config.json` 的 `resumes` 记录了每次从第几步恢复；`process.json` 的 `attempts` 是运行次数。
- 中断那次在存档之后的模型调用不计入 `model_usage.json`，所以花费会略低于实际支出。

## 开跑前要填写和确认的内容

代码部分已经补齐。下面这些需要课题组决定或填写，不能由代码代替：

1. **型号**：在 `config/model_profiles.json` 中填写 DeepSeek、Kimi、GLM、GPT 的具体型号名，必须是能接收图片的版本，并填写档位（`tier`）。未填型号的配置会在开跑前直接报错。
   - DeepSeek 据第三方资料只有 Flash 档支持图片，单张图片可能有约 384 token 的上限，以官方文档为准。
   - 各家的接口地址已按公开的默认地址填写，使用代理或其他地址时自行修改。
2. **单价**：填写每个模型的输入、输出单价和 Embedding 单价（每百万 token，币种见 `currency`）。未填时 `episode_cost` 为空，其余结果不受影响。
3. **采样参数和思考模式**：
   - 是否统一设定温度：写在 `shared_generate_kwargs`，例如 `{"temperature": 0.7}`。
   - 关闭思考模式的方式各家不同，写在各自的 `generate_kwargs`。
   - 二者都会记录在每局的 `config.json` 中。
4. **试跑**：每个模型先用上面的单局试跑命令跑 20 步，确认三件事：能看清画面、JSON 重试次数不过多、花费可接受。之后再全量运行。
5. **代码版本**：同一张表的所有局必须用同一个代码版本跑，开跑后不再修改代码。每局 `config.json` 中的 `git_head` 和 `source_sha256` 用来核对。
