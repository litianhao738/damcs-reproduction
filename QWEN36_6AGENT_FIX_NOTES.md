# Qwen 3.6 Flash 六智能体兼容性修复记录

日期：2026-06-22  
模型：`qwen3.6-flash-2026-04-16`  
实验配置：6 agents，Qwen OpenAI-compatible API

## 背景

将默认聊天模型从 `qwen3.7-plus-2026-05-26` 切换为
`qwen3.6-flash-2026-04-16` 后，6-agent 实验在初始步骤或第 3 至 4 步异常退出，
没有生成最终 `all_rounds_stats.csv`。

模型 API、Embedding API 和多模态图片输入本身均可正常调用。退出原因主要是
Qwen 返回的 JSON 使用了合理但不符合原 Pydantic 严格枚举的值，同时原并发处理会让
任意一个 agent 的异常终止整个实验。

## 已复现错误

### 1. 视觉对象 `cow` 不在枚举中

错误：

```text
reflection.vision.2
Input should be ...
input_value='cow'
```

`cow` 是 Crafter 环境中的合法对象，但原 `MaterialType` 没有包含它。

### 2. 状态同义词不符合 ResultType

错误示例：

```text
action.next_action_prerequisites_status
input_value='satisfied'
```

内部只接受：

```text
success, failure, in_progress
```

### 3. long_term_goal_progress 返回状态词

错误：

```text
goal.long_term_goal_progress
input_value='in_progress'
```

该字段要求具体任务，例如 `collect_wood`，但模型有时会返回进度状态。

### 4. ActionType 同义词不一致

错误：

```text
reflection.last_action
input_value='navigate'
```

内部动作枚举使用大小写敏感的 `Navigator`。

### 5. 单个 agent 异常终止整个并发实验

原代码直接对所有 future 调用 `future.result()`。任何一个 agent 抛出校验异常都会中断
整个 6-agent 线程池。增加异常隔离后，reporter 又因为访问空响应产生：

```text
AttributeError: 'NoneType' object has no attribute 'reflection'
```

## 修改内容

### `scripts/run_with_qwen.ps1`

- 默认聊天模型改为 `qwen3.6-flash-2026-04-16`。
- Embedding 模型保持 `text-embedding-v4`。

### `memory_system/llm_api.py`

#### 扩充视觉对象枚举

新增合法 Crafter 世界对象：

```text
path, player, cow, zombie, skeleton, arrow, plant
```

同时更新 Qwen JSON schema 提示中的 `MaterialType` 列表。

#### 规范化 ResultType 同义词

在 Pydantic 校验前执行规范化：

```text
successful/completed/complete/done/satisfied/ready/met/true/yes -> success
failed/unsuccessful/unsatisfied/not_satisfied/not_ready/unmet/blocked/false/no -> failure
pending/ongoing/incomplete/not_completed -> in_progress
not_applicable -> failure
```

未知值不会被静默接受，仍会进入原有校验和纠正重试流程。

#### 修复 long_term_goal_progress

如果 `long_term_goal_progress` 不是合法 `GoalType`：

1. 优先使用合法的 `current_goal`；
2. 其次使用合法的 `long_term_goal`；
3. 最后回退为 `collect_wood`。

#### 规范化 ActionType 同义词

```text
navigate/navigator/navigate_to -> Navigator
collect/gather/mine -> do
wait -> sleep
```

#### JSON 解析重试

Qwen 响应纠正重试现在同时处理：

```text
Pydantic ValidationError
JSONDecodeError
```

### `utils.py`

#### agent 级别异常隔离

- 每个 future 单独获取结果。
- 某个 agent 思考失败时记录明确错误。
- 该 agent 本步不更新决策，保持安全动作，并在下一步重新思考。
- 其它 agent 和环境循环继续运行。

#### reporter 空响应处理

- 为 agent 记录 `thinking_error`。
- 成功响应后清除错误。
- 本步失败时 reporter 不再访问空 `final_response`，只输出错误摘要和下步重试说明。

## 验证

已完成以下检查：

```text
Python py_compile：通过
ResultType 规范化单元检查：通过
Goal progress 回退检查：通过
ActionType 规范化检查：通过
Qwen embedding smoke test：通过，1024 dimensions
Qwen text chat smoke test：通过
Qwen 双图片多模态 1-agent/1-step：通过
修复后 6-agent/1-step：通过
最终 6-agent/5-step 回归：通过并生成 CSV
```

最终 5-step 回归结果：

```csv
Collect wood,Place table,Make wood pickaxe,Collect stone,Make stone pickaxe,Collect iron,Collect coal,Place furnace,Make iron pickaxe,Collect diamond
2.0,,,,,,,,,
```

## 正式重跑命令

```cmd
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_with_qwen.ps1 -LlmTimeoutSeconds 90 -LlmMaxRetries 2 -LlmMaxTokens 1024 run.py --agent_num 6 --step_num 150 --num_of_turn 1 --results_dir results\qwen36_flash_20260416_agent_6a_150s_retry1 --no_plot
```

运行中出现黄色 `Qwen response validation failed, retrying` 表示自动纠正，不等于实验
失败。某个 agent 出现红色思考错误时，该 agent 会在下一步重试，不再导致整轮退出。

## 算法影响

这些改动属于 Qwen 输出兼容、输入校验和并发容错层修复，没有改变 Crafter 任务规则、
分层知识图谱、记忆模块、协作目标生成或环境动作逻辑。
