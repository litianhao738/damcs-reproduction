# 实验表设计

日期：2026-11-17。本文定义要跑的三张表、统一设置、指标和运行方法。表中空格在实验完成后填写；未填的格子不得用估计值代替。

> **条件为 A、B、C、H，代码已同步（2026-11-18）**：
>
> - **A**：DAMCS 底座。
> - **B**：A + 合法性修复。最多 2 轮重规划；重规划后仍不合法时，由可修复范围 S 兜底。
> - **C**：B + 自动处理"合法但慢"的计划（进度问题）。
> - **H**：C 的人审版，主方法。
>
> 框架共 3 个模块：合法性修复、进度修复、关键时刻人审。S 是合法性修复内部的兜底，不单独算模块。
>
> 当天条件改过两次：
>
> - 第一次：原 H1、H2 合并为 H；新增 D，即自动进度修复。
> - 第二次：原 B（只重规划）和原 C（重规划 + S）合并为现在的 B，配置沿用原 C；原 D 改名为 C。理由见表 1 下的"条件变更"。
>
> 旧编号对照：旧结果里的 C 是现在的 B，D 是现在的 C，H2 是没有进度修复的 H；旧 B（只重规划）已不是表格条件。每局 `config.json` 都记录了五项条件设置，可以据此核对。旧代码或旧编号跑出的结果不能和新结果放进同一张表。
>
> 设计见 [进度问题修复设计](progress_repair_design_2026-11-18.md)，实现和验证见 [实现说明](implementation_2026-11-16.md) 第 4.1 节。开跑前还要完成本文末尾"开跑前要填写和确认的内容"中的第 0 项（调参试跑与两项决定）。

## 统一设置

三张表都只在 MA-Crafter（挖钻石）上运行。

CUBE 的适配代码保留（`run_cube.py`，A、B、C、H 四个条件都能跑），但 2026-11-18 起不进三张表。因此论文中不能声称机制在另一个环境中同样有效；需要时可作为附录补跑。

| 项目 | 设置 |
| --- | --- |
| Agent 数 | 6 |
| 步数上限 | 350 |
| 地图与任务 | DAMCS 原版，团队目标是采到钻石 |
| 种子 | 4、5、6 |
| 每种子重复 | 表 1、表 2 为 2 次；表 3 为 1 次 |

其余设置：

- 重规划最多 2 轮。
- 要修的问题分两类：
  - **不合法问题**：某个 Agent 这一步做不成。B、C、H 先重规划，仍不合法时由可修复范围 S 兜底；A 只记录。
  - **进度问题**：计划都合法，但团队下一个里程碑明显来得太慢。只有 C、H 处理；A、B 只记录，不处理。
  - 定义见 [进度问题修复设计](progress_repair_design_2026-11-18.md)。
- 人审（条件 H）不设次数上限，只在**关键时刻**问人，其余问题一律按 C 的方式自动处理。两类问题共用同一套关键时刻判断：用与环境无关的三个信号（`--review_trigger signals`，默认），满足任意一个即可：
  - **要做取舍**（`choice`）：能修好这个问题的方案不止一个，而且改的是不同的 Agent。机器只会按"少改人"机械地选，该打断谁需要人判断。
    - 进度问题的条件更严：排前两名的方案改的 Agent 不同，而且排序几乎分不开，即两个都是精确方案或都是估计方案，代价相差不到 Δ（相当于少于一个人的分量）。排第一的方案明显更好时直接采用，不问人。
    - 这样收紧的原因是 2026-11-18 的 H 预试（200 步）：按原条件，5 个进度问题全都问了人；按新条件只会问 1 次（采煤）。
  - **拿不准**（`unknown`）：信息不全，无法确定能不能修好。
  - **团队卡住**（`stalled`）：团队进度连续一段时间没有增长，此时存在的问题都算。这段时间按每局步数的比例计算（`--stall_fraction 0.1`），350 步对应 35 步；`--stall_steps` 可改为固定步数。进度是全队达成的主线里程碑数（下文"阶段步数"的 10 个阶段；喝水、吃牛等不算）。
  - 同一个问题问过之后 10 步内不再重复问（`--review_cooldown 10`），避免一个卡住的问题每步都打扰审阅者。
  - 不合法问题：一次人审最多开放 2 个 Agent（`--review_max_agents 2`）。H 会同时开放能修好问题的其他已验证方案涉及的 Agent（不超过 2 个），让人可以选择改谁；"拿不准"时开放问题涉及的 Agent。
  - 进度问题：把排好序的候选方案列给人，人只能选其中一个、选"不改"或跳过（跳过即按排第一的方案自动处理），不能自由修改。
    - **审阅期间不让任何人等。** 团队照常行动：不合法问题和其他里程碑的进度问题都按 C 的方式自动处理，只有被问的这个里程碑等人决定。
    - **回复时按当时的状态重新计算所选方案。** 已经失效时按 C 的方式处理；团队在回复前自己完成了这个里程碑，请求自动结束（`already_reached`）。
    - 原因：H 预试中，每次进度审阅都停下 6 个 Agent 里的 5 个，5 次一共停了 247 个"Agent×步"，相当于全队停了约 41 步，钻石步数比 C 多 40 步（109 对 69）；两次真人回答选的都是推荐方案。
    - 旧的做法（暂停方案涉及的 Agent）保留为消融 `--progress_pause on`，见附表。
  - 每局问人的次数因此不固定，要作为结果报告，并按信号（`requests_choice`、`requests_unknown`、`requests_stalled`）和问题类型（`requests_legal`、`requests_progress`）分别统计。
  - 旧的按阶段固定名单（铁镐、采铁、采钻石）保留为 `--review_trigger stage`，可作对照；`--review_trigger all` 对所有问题都问人，只用于练习。
- 人审默认使用局部等待（只用于不合法问题）、增量复核、精简证据。
- 可修复范围 S 的搜索不设人数上限，按"先修得全、再改得少"选择。
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
| 非法计划数 | 计划里有当步做不成的动作的规划步数；首轮写出的非法计划数；重规划之后、S 之前仍不合法的计划数。所有条件都记录 | `illegal_plan_steps`、`illegal_plans_first`、`illegal_plans_after_replan` |
| 问题数/局 | 重规划后仍不合法、交给可修复范围 S 搜索的规划步数 | `target_snapshots` |
| 有方案比例 | 出现问题的步中，S 找到可修复方案的比例 | `selected_snapshots / target_snapshots` |
| S 修复次数 | S 实际提交的不合法修复次数（含人审采纳的） | `commits` |
| 修复生效率 | 提交的修复在实际执行中达成目标的比例。分子、分母都是合法性修复（S）与进度修复（精确方案、锁定任务）之和；两类分开的数字也导出 | `repair_success_rate`；分开：`recoveries / recovery_attempts`、`progress_exact_success_rate`、`progress_task_completion_rate` |
| 进度问题数/局 | 检查出进度问题的规划步数，所有条件都记录；另按类型（exact、slow、unknown）分开 | `progress_target_steps`、`progress_targets_<类型>` |
| 进度修复次数、完成率 | C、H 提交的进度修复次数（另按模板、来源分开）；精确方案的当步生效率；锁定任务的完成率 | `progress_commits`、`progress_commits_<模板>`、`progress_exact_success_rate`、`progress_task_completion_rate` |
| 估计误差 | 进度修复的实际步数减估计步数 | `progress_estimate_error_mean`、`progress_estimate_error_median` |
| 卡住总步数 | 主线里程碑超过卡住窗口没有增加的步数之和 | `stall_steps_total` |
| 锁定步数 | 锁定执行、没有调用 LLM 的 Agent 步数；除以"Agent 数 × 步数"即机制接管的比例 | `committed_agent_steps` |
| 平均改动人数 | 每次提交的修复改了几个 Agent 的计划 | `mean_edited_agents` |
| 被打断的原计划数 | 修复把本来在做别的事的帮手改去帮忙的次数 | `interrupted_plans` |
| 超出人审上限 | S 的最小修复超过 2 人、改走自动修复的次数 | `over_review_capacity` |
| 人审次数、阅读时间 | 发出的请求数、真人阅读与修改用时 | `requests_issued`、`human_seconds` |
| 问人的原因 | 因"要做取舍""拿不准""团队卡住"而发出的请求数（一个请求可能同时满足多个）；按问题类型分开的请求数 | `requests_choice`、`requests_unknown`、`requests_stalled`；`requests_legal`、`requests_progress` |
| 进度问题的人审选择 | 人选了推荐方案、其他方案、"不改"、跳过、超时、所选方案失效、回复前团队已自己完成的次数 | `progress_reply_recommended`、`_other`、`_keep`、`_skip`、`_timeout`、`_stale`、`_reached` |
| 未问人的原因 | 不是关键时刻、同一问题在冷却期内、超出人审上限等各自的次数 | `not_critical_skips`、`over_review_capacity`；其余见 `events.jsonl` 的 `review_skipped` |
| 技术失败 | 模型调用失败、该 Agent 沿用上一步计划的次数；JSON 格式错误后重试的次数 | `planning_failures`、`chat_json_retries` |
| 人审回复结果 | 各种回复状态的次数（采纳、过期、无效、超时等） | `reply_<状态>`，如 `reply_stale_plan` |
| 过期修复被提交数 | 关闭提交前复核时，本该被拦下却被提交的修复数 | `validation_unchecked_would_fail` |
| 复核耗时与次数 | 提交前复核的总时间和次数 | `validation_seconds`、`validation_*` |
| 被暂停的 Agent 步数 | 等待审阅期间停下的 Agent 步数之和 | `paused_agent_steps` |
| 证据事实组数 | 精简前后展示给人的事实组数 | `evidence_groups_before`、`evidence_groups_after` |
| 搜索耗时 | 可修复范围 S 搜索的总时间 | `search_seconds` |
| 模型调用与 token | 对话及 Embedding 请求次数和 token | `chat_requests`、`chat_input_tokens`、`chat_output_tokens`、`embedding_tokens` |
| 每局花费 | token × 配置文件中登记的单价；单价未填时为空 | `episode_cost` |

## 表 1：主实验——人审闭环是否有增量价值

固定 Qwen，只跑 Crafter。每格写均值，括号内写标准差或局数。

| 条件 | 含义 | 钻石步数 | 采到钻石比例 | 里程碑数 | 卡住总步数 | 修复生效率 | 人审次数/局 | 阅读时间/局 | 每局花费 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | DAMCS 底座：各自规划 + 动作合法性检查 | | | | | — | — | — | |
| B | A + 合法性修复：最多 2 轮重规划，仍不合法时由可修复范围 S 兜底 | | | | | | — | — | |
| C | B + 自动处理进度问题，采用排第一的方案（最强自动对照） | | | | | | — | — | |
| H | C 的人审版：关键时刻由人决定，其余同 C（主方法） | | | | | | | | |

- **主要比较**：相邻两个条件只差一个模块。
  - B − A：合法性修复的作用。
  - C − B：进度修复的作用。
  - H − C：人审的作用。
- **规模**：
  - 4 个条件 × 3 个种子 × 2 次重复，共 24 局，其中 H 的 6 局需要真人审阅。
  - 每局用 `--reviewer_id` 记录审阅者编号，审阅顺序另行登记。
  - 审阅者的操作步骤、判断标准和登记表见 [审阅规程](review_protocol_2026-11-18.md)。
- **"修复生效率"**：B 这一格只有 S 提交过修复时才有数；S 一次都没提交时写"—"。重规划的效果在正文用非法计划数说明，见指标定义。
- **脚本审阅**：只能检查流程，不填入本表。
- **条件变更（2026-11-18，第二次）**：原 B（只重规划）和原 C（重规划 + S）合并为现在的 B，原 D 改名为 C。依据是几局预试（6 个 Agent、150 步、种子 4，不进表格）：

  | 预试 | 首轮出现非法计划的步数 | 重规划后仍不合法 | S 实际修复 |
  | --- | --- | --- | --- |
  | 原 A，qwen3.7-plus（两局共 260 步） | 12 | 12（A 不修） | — |
  | 原 B，qwen3.7-plus（102 步） | 6 | 0 | — |
  | 原 C，qwen3.7-plus（149 步） | 4 | 0 | 0（149 次分析都没有可修目标） |
  | 原 H2，qwen3.8-max 关思考（149 步） | 32 | 2 | 0（145 次没有目标，4 次无解） |

  - 被提醒之后，模型基本都能改出合法计划，S 一次修复都没做过，所以原 B 和原 C 的行为相同，分成两行只是把同一设置多跑一遍。
  - Crafter 里常见的非法计划是"送出自己没有的东西"和"缺工具"，S 的一步联合模型修不了，所以它总是没有目标或无解。
  - S 留在代码里，作为合法性修复内部的兜底。正式运行中 S 的修复次数照常记录（`commits`）；不为 0 时要在正文说明。
  - 单局波动很大：同为原 A 的两局，一局第 111 步采到钻石，一局没采到。原 B 第 102 步采到、原 C 没采到，这个差距同样来自随机性，不是 S 造成的。
- **条件变更（2026-11-18，第一次）**：
  - 原 H1（开放出问题的 Agent）和 H2（开放可修复范围 S）合并为 H，使用原 H2 的审阅界面，并扩展到进度问题。
  - 当时把"开放谁更好"（H1 对 H2）改为离线统计；第二次变更后 S 降为兜底，这项统计不再进入表格。
  - 新增自动进度修复的条件（当时叫 D，现在是 C），用来把"有了进度方案"和"由人来选"这两个作用分开。
  - 更早的变更：2026-11-17 删除了开放整个相关小组的条件，并把 H0 改名为 H1。对照旧文档时：H0 即后来的 H1，现已不再作为表格条件；H2 即没有进度修复的 H。

## 表 2：模块消融

固定 Qwen，只跑 Crafter。从完整系统中每次去掉一个模块，其余不变。每格写均值，括号内写标准差；↓ 表示越小越好，↑ 表示越大越好。

| 方法 | 钻石步数 ↓ | 采到钻石比例 ↑ | 里程碑数 ↑ | 卡住总步数 ↓ | 每局花费 ↓ |
| --- | --- | --- | --- | --- | --- |
| 完整系统（C） | | | | | |
| w/o 合法性修复 | | | | | |
| w/o 进度修复 | | | | | |
| w/o LLM（仅机制） | | | | | |

表注：

- 完整系统是不需要真人的 C。人审的作用见表 1 的 H 与 C。
- "w/o 进度修复"就是表 1 的 B，数值直接引用。
- "w/o 合法性修复"把重规划和 S 都关掉：不合法计划只记录，和 A 一样；进度修复照常运行。
- "w/o LLM"中所有 Agent 一律什么都不做，不调用任何模型。C 与它的差距就是 LLM 的贡献。离线检查中，这一行在第 100 步采到了钻石（种子 4，150 步）。
- S 不单列：预试中它一次修复都没做过，单列只会得到和完整系统一样的一行（见表 1 下的"条件变更"）。

运行与规模：

- 命令 `--suite ablation`，新跑 `C-no-legality`、`C-no-llm` 两个版本，3 个种子 × 2 次重复，共 12 局，都不需要真人。`C-no-llm` 不花 API 费用。
- 想让差别更容易看清时，可把这两个版本和表 1 的 C 的重复次数都提到 4 次。

正文要补充的数字：

| 版本 | 要补充的数字 | 字段 |
| --- | --- | --- |
| 完整系统 | S 实际修复的次数（预试中为 0） | `commits` |
| w/o 合法性修复 | 非法计划数，以及进度修复有没有顶替上来 | `illegal_plans_first`、`progress_commits` |
| w/o LLM | 锁定步数占全部 Agent 步数的比例 | `committed_agent_steps` |

这些字段都在 `episodes.csv` 里。某个模块去掉后差距很小时，要说明是不是被其他模块顶替了，不能直接写"没用"。

其他：

- 动作合法性保护属于 DAMCS 底座，所有条件共用，不做消融。
- 3 个种子 × 2 次重复下，钻石步数的差异可能不显著，要如实报告。

### 附表：设计选择分析（放附录）

每行只把一处设计换成更简单的做法，比较原设计与替换后的结果。

| 设计选择 | 替换为 | 指标 | 原设计 | 替换后 |
| --- | --- | --- | --- | --- |
| 锁定执行 | 只写进提示 | 多步完成率 ↑ | | |
| 进度审阅时不暂停 | 暂停方案涉及的 Agent | 被暂停的 Agent 步数 ↓、钻石步数 ↓ | | |
| 提交前复核 | 不复核 | 过期修复被提交数 ↓ | | |
| 精简证据 | 完整证据 | 展示的事实组数 ↓ | | |

表注：

- **在线三组（可选）：** 用 `--suite appendix` 运行 `C-no-commit`、`Hscript`、`Hscript-progress-pause`、`Hscript-no-recheck`，共 24 局，都不需要真人。
  - 锁定执行与 C 比；暂停和复核与 `Hscript` 比。
  - `Hscript-progress-pause` 在进度审阅期间暂停方案涉及的 Agent（`--progress_pause on`，局部等待），即 2026-11-19 之前的做法。脚本审阅只延迟 2 步，所以这一行只反映"暂停"本身的代价；真人回复慢得多，实际代价更大（见上面统一设置里的预试数字）。
  - `Hscript` 是用脚本审阅（固定延迟 2 步、总是采纳推荐）的 H，测的是机制本身，不是人的价值。
- **精简证据（离线）：** 读 `Hscript` 各局记录的 `evidence_groups_before` 和 `evidence_groups_after`。
- **只对不合法问题的人审起作用的两行：** 提交前复核和精简证据。进度问题的回复总会按当前状态重新计算，也不展示证据。预试中 H 从没为不合法问题问过人；正式运行中如果仍然没有，这两行写"未触发"，不要用别的数字代替。
- **随 S 降级移出附表的三项：** 联合检查、最小范围、只改出问题的 Agent。正式运行中如果重规划后仍有不合法问题（`target_snapshots` > 0），可以用 `scripts/evaluate_frozen_scopes.py` 离线统计这三项，作为补充材料。
- 证据精简是否真能缩短阅读时间、是否影响判断，需要真人小试，小试完成后再报告。

## 表 3：跨模型对比——机制的作用是否依赖模型强弱

只跑 Crafter，只改对话模型，其余设置同表 1。主要模型是 Qwen、DeepSeek、Kimi、GLM，另加 GPT 作为国外参照。

基线是条件 A（DAMCS 底座），不是不带 DAMCS 记忆和分工的纯 API。本表要回答的是"本项目加在 DAMCS 之上的机制，包括合法性修复 B、进度修复 C 和主方法 H，其作用是否随模型强弱变化"；若以纯 API 为基线，差值中会混入 DAMCS 本身的作用。

每格只填钻石步数的均值，括号内为标准差。

| 模型 | 档位 | A | B | C | H | 每局花费 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen | | | | | | |
| DeepSeek | | | | | | |
| Kimi | | | | | | |
| GLM | | | | | | |
| GPT | | | | | | |

- **表注**：未采到钻石的局按步数上限 350 计，并注明每个格子中这类局的数量。
- **怎么读**：
  - B 对 A 看合法性修复的作用。写出非法计划多的模型，这一项可能更明显。
  - C 对 B 看处理进度问题的作用。
  - H 对 C 看真人审阅相对自动处理的增量（主方法）。
  - A 本身的数值反映模型原本的强弱。
  - 差值由读者从各列直接得到，表中不另列。
- **正文中要补的一句话**：差值的方向是否在各模型上一致。按种子配对计算（同一种子下 B 减 A、C 减 B、H 减 C），写明每个模型 3 个种子里有几个变好，例如"5 个模型中 4 个的 C 少于 B，其中 3 个模型 3/3 个种子都变好"。这一信息从均值中看不出来。
- **放到附表的指标**：非法计划数、进度问题数/局、修复生效率、S 修复次数、卡住总步数、人审次数和阅读时间、里程碑数、JSON 重试与报错中断（技术失败率）。弱模型若普遍采不到钻石，在附表中用里程碑数和做出铁镐的步数比较。汇总脚本都会导出这些字段。
- **H 用真人审阅**，设置与表 1 的 H 相同（只在关键时刻问、不设次数上限；进度审阅不暂停任何人；不合法问题局部等待、一次最多开放 2 个 Agent；增量复核、精简证据）。脚本审阅不能代替，因为它总是采纳推荐方案，结果接近 C。
- **规模**：5 个模型 × 4 个条件 × 3 个种子 × 1 次，共 60 局，其中 H 的 15 局需要真人审阅。Qwen 一行直接复用表 1 中重复编号为 0 的 A、B、C、H 局，所以需要新跑 48 局，其中真人审阅 12 局。
- **审阅者分配**：审阅者不能与模型绑定，否则"模型差异"和"审阅者差异"无法区分。按种子分配审阅者，同一个种子的所有模型由同一位审阅者审阅；审阅时尽量不告诉审阅者当前是哪个模型；每位审阅者的模型顺序要轮换，见 [审阅规程](review_protocol_2026-11-18.md)。每局用 `--reviewer_id` 记录审阅者编号。
- **要检验的猜想**：
  - 弱模型写出的非法计划更多，B − A 的改善应更大。预试中，qwen3.8-max 关思考时首轮出现非法计划的步数（32）明显多于 qwen3.7-plus（4–6）。
  - 弱模型重规划后仍不合法的情况可能更多，这时 S 才有机会起作用；S 的修复次数要按模型报告。
  - 进度问题可能在各模型上都存在：即使强模型很少写出不合法计划，也可能出现"合法但慢"的分工，所以 C − B 未必随模型变强而变小。
  - 人审的增量 H − C 也可能随模型变化：弱模型留给人判断的问题更多；强模型的自动处理已经足够时，人审的价值变小。
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
python scripts\run_experiments.py --model_profiles qwen --conditions H --seeds 4 --repeats 2 --reviewer human --reviewer_id R1 --results_dir results\table1_crafter_human_R1
```

### 表 2

```powershell
# 表 2：C-no-legality、C-no-llm（C-no-llm 不调用任何模型）
python scripts\run_experiments.py --suite ablation --model_profiles qwen --seeds 4 5 6 --repeats 2 --results_dir results\table2

# 附表的在线三组（可选）：C-no-commit、Hscript、Hscript-progress-pause、Hscript-no-recheck
python scripts\run_experiments.py --suite appendix --model_profiles qwen --seeds 4 5 6 --repeats 2 --reviewer scripted --results_dir results\appendix

# S 的补充统计（可选，只在重规划后仍有不合法问题时有内容）：读取表 1 自动条件运行中保存的问题快照
python scripts\evaluate_frozen_scopes.py --input results\table1_crafter_auto --capacities 2 --output results\appendix_scope_check.json
```

"精简证据 → 完整证据"一行读 `Hscript` 各局汇总中的 `evidence_groups_before` 和 `evidence_groups_after`。S 的补充统计中，`independent` 对应"联合检查 → 各问题单独修再合并"，`strategies.owner` 对 `strategies.verified` 对应"只改出问题的 Agent → S"，两者各给出 `has_witness`（能否修好）和 `target_coverage`（修好几个问题）。

### 表 3

```powershell
# 自动条件
python scripts\run_experiments.py --model_profiles deepseek kimi glm gpt --conditions A B C --seeds 4 5 6 --repeats 1 --results_dir results\table3_auto

# 真人审阅：每位审阅者负责一个种子，审阅该种子下所有模型，审阅页面为 http://127.0.0.1:8765
python scripts\run_experiments.py --model_profiles deepseek kimi glm gpt --conditions H --seeds 4 --repeats 1 --reviewer human --reviewer_id R1 --results_dir results\table3_human_R1
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
- 没有存档的局从头重跑，原来的失败记录改名保留为 `<目录>.failedN`。
- 存档与代码版本绑定：代码改过之后不能接着跑，会报错提示，需要从头重跑这一局。
- 每局 `config.json` 的 `resumes` 记录了每次从第几步恢复；`process.json` 的 `attempts` 是运行次数。
- 中断那次在存档之后的模型调用不计入 `model_usage.json`，所以花费会略低于实际支出。

## 开跑前要填写和确认的内容

0. **进度问题修复的开跑前工作**（代码已实现，见 [实现说明](implementation_2026-11-16.md) 第 4.1 节）：
   - **调参试跑**：在表格以外的种子（例如 7、8）上各跑 150 步的 B 和 C，调整"明显慢"的阈值 `--slow_margin` 和锁定上限，并看 H 的问人频率。不要在 4、5、6 上调。
   - **"只靠机制"对照已加入表 2**（`C-no-llm`，开关 `--agent_llm off`，2026-11-18）：离线检查中，LLM 回复一律是 noop 时，C 仍在第 100 步采到钻石，所以必须报告这一行，才能区分"机制本身的作用"和"修复 LLM 计划的作用"。
   - **H 的问人方式已定（2026-11-19）**：进度问题只在排前两名的方案分不出明显好坏、或团队卡住时问人，审阅期间不暂停任何人（见"统一设置"）。调参试跑时仍要看每局问几次。

下面这些需要课题组决定或填写，不能由代码代替：

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
