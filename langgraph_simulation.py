"""现有多智能体仿真流水线的 LangGraph 编排实现。

该模块保留原环境、Agent、提示词和单步执行顺序，只将外层 ``while``
循环迁移到状态图，以便把编排变化与智能体行为变化分开验证。
"""

from typing import Annotated, Any, TypedDict
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from tqdm import tqdm

from checkpointing import SimulationCheckpointStore

from utils import (
    AgentActionProcessor,
    AgentStateManager,
    AgentThinkingProcessor,
    EnvironmentManager,
    SimulationContextManager,
    SimulationLogger,
)


def merge_agent_responses(left, right):
    """合并并发 Agent 分支的响应，由后返回的同 ID 结果覆盖旧值。"""
    return {**left, **right}


class SimulationGraphState(TypedDict):
    """一轮仿真中在父图各节点之间传递的状态。"""

    # 已完成并持久化的环境步数。
    total_step: int
    # 当前环境/经验时间步编号。
    current_step: int
    # 当前 episode 编号，兼容原有经验记忆字段。
    episode_number: int
    # 多个 Agent 的 RGB 局部观察。
    obs: Any
    # 当前步各 Agent 的奖励。
    rewards: Any
    # 环境是否触发终止条件。
    done: bool
    # 背包、成就、语义地图和生存状态等环境附加信息。
    info: Any
    # 准备提交给环境的离散动作编号。
    agents_actions: list[int]
    # 标记本步哪些 Agent 需要产生新思考。
    agents_with_new_thought: list[bool]
    # 按 Agent ID 保存的个性化提示上下文。
    agents_contexts: dict[int, str]
    # 并发分支结果通过 merge_agent_responses 自动汇合。
    agents_responses: Annotated[dict[int, dict[str, Any]], merge_agent_responses]
    # 恢复运行时跳过 env.reset() 的一次性标志。
    resuming: bool


class AgentPlannerState(TypedDict):
    """单个 Agent 规划子图使用的最小状态。"""

    agent_id: int
    agents_contexts: dict[int, str]
    info: Any
    step: int
    response: dict[str, Any]


class LangGraphMultiAgentSimulation:
    """以行为兼容为优先目标的 LangGraph 多智能体仿真控制器。"""

    def __init__(
        self,
        agents,
        env,
        max_steps,
        n_players,
        checkpoint_db=None,
        run_id=None,
        resume=False,
        max_concurrency=None,
        output_dir=None,
    ):
        """保存运行参数、初始化处理器并编译父图与 Agent 子图。"""
        self.agents = agents
        self.env = env
        self.max_steps = max_steps
        self.n_players = n_players
        self.run_id = run_id
        self.resume = resume
        self.output_dir = output_dir
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        self.max_concurrency = max_concurrency or n_players
        self.checkpoint_store = (
            SimulationCheckpointStore(checkpoint_db) if checkpoint_db else None
        )
        if self.checkpoint_store is not None and not self.run_id:
            raise ValueError("checkpoint_db requires a non-empty run_id.")
        if self.resume and (self.checkpoint_store is None or not self.run_id):
            raise ValueError("Resume requires both checkpoint_db and run_id.")
        if (
            self.checkpoint_store is not None
            and not self.resume
            and self.checkpoint_store.exists(self.run_id)
        ):
            raise ValueError(
                f"Checkpoint run_id={self.run_id!r} already exists. "
                "Choose another --run_id or pass --resume."
            )

        # 复用 legacy 控制器的处理器，避免迁移时改变业务行为。
        self.action_processor = AgentActionProcessor()
        self.env_manager = EnvironmentManager()
        self.reporter = SimulationLogger()
        self.agent_state_manager = AgentStateManager()
        self.simulation_context_manager = SimulationContextManager(self.n_players)
        self.agent_thinking_processor = AgentThinkingProcessor()

        self._progress_bar = None
        self.agent_planner_graph = self._build_agent_planner_graph()
        self.graph = self._build_graph()

    def _build_agent_planner_graph(self):
        """构建“调用模型→校验响应”的单 Agent 规划子图。"""
        builder = StateGraph(AgentPlannerState)
        builder.add_node("call_model", self._planner_call_model)
        builder.add_node("validate_response", self._planner_validate_response)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", "validate_response")
        builder.add_edge("validate_response", END)
        return builder.compile()

    def _build_graph(self):
        """构建单步仿真节点、并发规划分支以及循环终止路由。"""
        builder = StateGraph(SimulationGraphState)
        builder.add_node("initialize", self._initialize)
        builder.add_node("prepare_actions", self._prepare_actions)
        builder.add_node("step_environment", self._step_environment)
        builder.add_node("update_agent_states", self._update_agent_states)
        builder.add_node("plan_agent", self._plan_agent)
        builder.add_node("apply_agent_responses", self._apply_agent_responses)
        builder.add_node("complete_step", self._complete_step)

        builder.add_edge(START, "initialize")
        builder.add_conditional_edges(
            "initialize",
            self._next_after_initialize,
            {"continue": "prepare_actions", "end": END},
        )
        builder.add_edge("prepare_actions", "step_environment")
        builder.add_edge("step_environment", "update_agent_states")
        builder.add_conditional_edges(
            "update_agent_states", self._route_agent_planners
        )
        builder.add_edge("plan_agent", "apply_agent_responses")
        builder.add_edge("apply_agent_responses", "complete_step")
        builder.add_conditional_edges(
            "complete_step",
            self._next_after_step,
            {"continue": "prepare_actions", "end": END},
        )
        return builder.compile()

    def _initialize(self, state: SimulationGraphState):
        """新实验重置环境；恢复实验则保留 checkpoint 中的状态。"""
        if state["resuming"]:
            return {"resuming": False}
        return {
            "total_step": 0,
            "current_step": 0,
            "episode_number": 0,
            "obs": self.env.reset(),
            "rewards": [0] * self.n_players,
            "done": False,
            "info": [{} for _ in range(self.n_players)],
            "agents_actions": [0] * self.n_players,
            "agents_with_new_thought": [False] * self.n_players,
            "agents_contexts": {},
            "agents_responses": {},
            "resuming": False,
        }

    def _next_after_initialize(self, state: SimulationGraphState):
        """初始化后判断是否已经满足结束条件。"""
        if state["done"] or state["total_step"] >= self.max_steps:
            return "end"
        return "continue"

    def _prepare_actions(self, state: SimulationGraphState):
        """执行高层技能并收集本步要提交给环境的离散动作。"""
        self.reporter.print_step_header(
            state["total_step"], state["current_step"]
        )
        self.action_processor.process_all_agent_actions(
            self.agents, self.env, self.n_players
        )
        return {
            "agents_actions": self.action_processor.collect_agent_actions(
                self.agents, self.n_players
            )
        }

    def _step_environment(self, state: SimulationGraphState):
        """同步推进一次共享环境，并更新全局合成站状态。"""
        obs, rewards, done, info = self.env_manager.step_environment(
            self.env, state["agents_actions"]
        )
        self.env_manager.update_crafting_stations(self.agents, self.env)

        # legacy 循环也会计算该信息；即使当前未消费，仍保持调用顺序兼容。
        self.env_manager.get_tool_availability_info(self.env, self.n_players)
        return {"obs": obs, "rewards": rewards, "done": done, "info": info}

    def _update_agent_states(self, state: SimulationGraphState):
        """把新观察写入 Agent，并为下一次规划生成个性化上下文。"""
        self.agent_state_manager.update_all_agent_states(
            self.agents,
            state["obs"],
            state["current_step"],
            self.env,
            state["info"],
            state["episode_number"],
        )
        agents_with_new_thought = (
            self.agent_state_manager.identify_agents_needing_thought(
                self.agents, state["info"]
            )
        )
        agents_contexts = self.simulation_context_manager.create_agent_contexts(
            self.agents, state["info"]
        )
        return {
            "agents_with_new_thought": agents_with_new_thought,
            "agents_contexts": agents_contexts,
        }

    def _route_agent_planners(self, state: SimulationGraphState):
        """为每个非人工 Agent 返回一个可并发调度的 Send 分支。"""
        ai_agent_ids = [
            agent.id
            for agent in self.agents
            if not (hasattr(agent, "is_human") and agent.is_human)
        ]
        if not ai_agent_ids:
            return "apply_agent_responses"
        # LangGraph 在同一个 superstep 中并行执行这些 plan_agent 分支。
        return [
            Send(
                "plan_agent",
                {
                    "agent_id": agent_id,
                    "agents_contexts": state["agents_contexts"],
                    "info": state["info"],
                    "step": state["current_step"],
                    "response": {},
                },
            )
            for agent_id in ai_agent_ids
        ]

    def _planner_call_model(self, state: AgentPlannerState):
        """优先读取 LLM 缓存，未命中时调用对应 Agent 的规划逻辑。"""
        agent_id = state["agent_id"]
        agent = self.agents[agent_id]
        request_hash = None
        # 请求摘要包含上下文和本 Agent 环境信息，避免错误复用响应。
        if self.checkpoint_store is not None:
            request_hash = self.checkpoint_store.llm_request_hash(
                state["agents_contexts"].get(agent_id), state["info"][agent_id]
            )
            cached = self.checkpoint_store.get_llm_response(
                run_id=self.run_id,
                step=state["step"],
                agent_id=agent_id,
                request_hash=request_hash,
            )
            if cached is not None:
                return {"response": cached}

        # 单个 Agent 失败不会打断其他并发分支，而是返回结构化错误。
        try:
            response = type(agent).process_agent(
                agent, state["agents_contexts"], state["info"]
            )
            response["error"] = None
            if self.checkpoint_store is not None:
                self.checkpoint_store.save_llm_response(
                    run_id=self.run_id,
                    step=state["step"],
                    agent_id=agent_id,
                    request_hash=request_hash,
                    response=response,
                )
            return {"response": response}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return {
                "response": {
                    "id": agent_id,
                    "thoughts": None,
                    "error": error,
                }
            }

    @staticmethod
    def _planner_validate_response(state: AgentPlannerState):
        """确保规划分支至少返回正确的 Agent ID 和 thoughts 字段。"""
        response = state.get("response")
        agent_id = state["agent_id"]
        if not isinstance(response, dict):
            response = {
                "id": agent_id,
                "thoughts": None,
                "error": "Planner returned a non-dictionary response.",
            }
        elif response.get("id") != agent_id or "thoughts" not in response:
            response = {
                "id": agent_id,
                "thoughts": None,
                "error": "Planner response failed id/thoughts validation.",
            }
        return {"response": response}

    def _plan_agent(self, state: AgentPlannerState):
        """执行单 Agent 子图，并把结果转换为父图 reducer 所需映射。"""
        result = self.agent_planner_graph.invoke(state)
        return {"agents_responses": {state["agent_id"]: result["response"]}}

    def _apply_agent_responses(self, state: SimulationGraphState):
        """等待并发分支汇合后，按 ID 顺序统一更新 Agent 技能状态。"""
        agents_responses = [
            state["agents_responses"][agent_id]
            for agent_id in sorted(state["agents_responses"])
        ]
        self.agent_thinking_processor.update_agents_from_responses(
            self.agents, agents_responses
        )
        return {}

    def _complete_step(self, state: SimulationGraphState):
        """完成日志、步数递增，并原子提交本环境步 checkpoint。"""
        self.reporter.show_step_report(
            self.agents, state["agents_with_new_thought"]
        )
        if self._progress_bar is not None:
            self._progress_bar.update(1)
        # 只有环境、Agent 和响应全部处理完成后才把步数视为已提交。
        committed_state = dict(state)
        committed_state.update({
            "total_step": state["total_step"] + 1,
            "current_step": state["current_step"] + 1,
            "resuming": False,
        })
        if self.checkpoint_store is not None:
            self.checkpoint_store.save(
                run_id=self.run_id,
                state=committed_state,
                env=self.env,
                agents=self.agents,
                max_steps=self.max_steps,
                n_players=self.n_players,
                output_dir=self.output_dir,
            )
        return committed_state

    def _next_after_step(self, state: SimulationGraphState):
        """环境终止或达到最大步数时结束，否则进入下一环境步。"""
        if state["done"] or state["total_step"] >= self.max_steps:
            return "end"
        return "continue"

    def run_simulation(self):
        """运行或恢复整张状态图，并返回与 legacy 控制器一致的结果。"""

        # 新运行的默认状态；恢复运行会在下方整体替换为 checkpoint 状态。
        initial_state: SimulationGraphState = {
            "total_step": 0,
            "current_step": 0,
            "episode_number": 0,
            "obs": None,
            "rewards": [0] * self.n_players,
            "done": False,
            "info": [{} for _ in range(self.n_players)],
            "agents_actions": [0] * self.n_players,
            "agents_with_new_thought": [False] * self.n_players,
            "agents_contexts": {},
            "agents_responses": {},
            "resuming": False,
        }

        # 恢复环境、Agent、随机状态以及图状态，并校验产物目录一致性。
        if self.resume:
            restored = self.checkpoint_store.load(
                run_id=self.run_id,
                max_steps=self.max_steps,
                n_players=self.n_players,
            )
            self.env = restored.env
            self.agents = restored.agents
            if self.output_dir and restored.output_dir:
                requested = str(Path(self.output_dir).resolve())
                if requested != restored.output_dir:
                    raise ValueError(
                        "Checkpoint output directory does not match the requested "
                        f"directory: {restored.output_dir!r} != {requested!r}."
                    )
            initial_state = restored.state
            initial_state["resuming"] = True
            print(
                f"Resuming run {self.run_id!r} from completed step "
                f"{restored.total_step}."
            )

        # 父图每个环境步约经过 6 个 superstep，预留额外递归空间。
        recursion_limit = max(30, self.max_steps * 7 + 10)
        completed_steps = initial_state["total_step"]
        with tqdm(
            total=self.max_steps,
            initial=completed_steps,
            desc="Progress",
        ) as bar:
            self._progress_bar = bar
            try:
                # invoke 对调用者同步，但 Send 分支由 LangGraph 内部并发调度。
                final_state = self.graph.invoke(
                    initial_state,
                    config={
                        "recursion_limit": recursion_limit,
                        "max_concurrency": self.max_concurrency,
                    },
                )
            finally:
                self._progress_bar = None

        self.reporter.print_simulation_complete()
        stats_df = self.reporter.log_stats(self.agents)
        return {
            "total_steps": final_state["total_step"],
            "completed": final_state["done"],
            "final_obs": final_state["obs"],
            "final_rewards": final_state["rewards"],
            "final_info": final_state["info"],
            "stats_df": stats_df,
        }

    def get_agent_summary(self):
        """返回便于调试或界面展示的 Agent 当前技能摘要。"""
        return [
            {
                "id": agent.id,
                "op": agent.op,
                "action": agent.action,
                "action_status": agent.action_status,
                "rss_to_collect": getattr(agent, "rss_to_collect", "N/A"),
                "rss_to_share": getattr(agent, "rss_to_share", "N/A"),
                "target_agent_id": getattr(agent, "target_agent_id", "N/A"),
            }
            for agent in self.agents
        ]
