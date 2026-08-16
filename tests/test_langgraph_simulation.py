"""LangGraph 编排、并发、缓存和结构化恢复的回归测试。"""

import unittest
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import crafter

from langgraph_simulation import LangGraphMultiAgentSimulation
from checkpointing import SimulationCheckpointStore
from memory_system.agent import HumanAgent
from runtime_snapshot import create_runtime_snapshot, restore_runtime_snapshot
from utils import AgentThinkingProcessor


class FakeAgent:
    """仅记录规划事件的轻量 Agent 测试替身。"""
    def __init__(self, agent_id, events=None):
        self.id = agent_id
        self.events = events
        self.op = "noop"
        self.action = "noop"
        self.action_status = "done"

    def process_agent(agent, contexts, info):
        if agent.events is not None:
            agent.events.append("plan_agent")
        return {"id": agent.id, "thoughts": None}


class FakeEnvironment:
    """使用计数器模拟确定性环境转移。"""
    def __init__(self, events):
        self.events = events
        self.step_count = 0

    def reset(self):
        self.events.append("reset")
        return ["initial-observation"]


class FakeActionProcessor:
    """记录动作处理顺序的测试替身。"""
    def __init__(self, events):
        self.events = events

    def process_all_agent_actions(self, agents, env, n_players):
        self.events.append("process_actions")

    def collect_agent_actions(self, agents, n_players):
        self.events.append("collect_actions")
        return [0] * n_players


class FakeEnvironmentManager:
    """生成可预测观察和奖励的环境管理器替身。"""
    def __init__(self, events):
        self.events = events
        self.steps = 0

    def step_environment(self, env, actions):
        env.step_count += 1
        self.steps = env.step_count
        self.events.append("env_step")
        count = len(actions)
        return (
            [f"observation-{self.steps}"] * count,
            [self.steps] * count,
            False,
            [{"sleeping": False}] * count,
        )

    def update_crafting_stations(self, agents, env):
        self.events.append("crafting_stations")

    def get_tool_availability_info(self, env, n_players):
        self.events.append("tool_availability")
        return ""


class FakeAgentStateManager:
    """记录 Agent 状态同步阶段是否执行。"""
    def __init__(self, events):
        self.events = events

    def update_all_agent_states(self, agents, obs, step, env, info, episode_number):
        self.events.append("update_agent_states")

    def identify_agents_needing_thought(self, agents, info):
        self.events.append("identify_thought")
        return [True] * len(agents)


class FakeContextManager:
    """为所有 Agent 返回固定规划上下文。"""
    def __init__(self, events):
        self.events = events

    def create_agent_contexts(self, agents, info):
        self.events.append("create_contexts")
        return {agent.id: "context" for agent in agents}


class FakeThinkingProcessor:
    """记录并发响应合并后的统一更新阶段。"""
    def __init__(self, events):
        self.events = events

    def process_agent_thinking_parallel(self, agents, contexts, info, **kwargs):
        self.events.append("think_parallel")
        return [{"id": 0, "thoughts": "noop"}]

    def update_agents_from_responses(self, agents, responses):
        self.events.append("apply_responses")


class FakeReporter:
    """以事件记录代替终端输出和真实统计计算。"""
    def __init__(self, events):
        self.events = events

    def print_step_header(self, total_step, current_step):
        self.events.append("step_header")

    def show_step_report(self, agents, new_thought):
        self.events.append("step_report")

    def print_simulation_complete(self):
        self.events.append("complete")

    def log_stats(self, agents):
        self.events.append("stats")
        return pd.DataFrame([{"agent_id": 0}])


class LangGraphSimulationTests(unittest.TestCase):
    """验证迁移后时序、并发和恢复语义与预期一致。"""

    def test_structured_snapshot_preserves_next_environment_transition(self):
        """恢复后的下一次环境转移应与原环境完全一致。"""
        env = crafter.Env(length=5, n_players=1, seed=4)
        env.reset()
        agent = HumanAgent(0)
        obs, reward, done, info = env.step([0])
        agent.update_state(obs, 0, env, info, 0, 0)
        state = {
            "total_step": 1,
            "current_step": 1,
            "episode_number": 0,
            "obs": [obs],
            "rewards": [reward],
            "done": done,
            "info": [info],
            "agents_actions": [0],
            "agents_with_new_thought": [False],
            "agents_contexts": {},
            "agents_responses": {},
            "resuming": False,
        }

        snapshot = create_runtime_snapshot(state, env, [agent])
        restored_state, restored_env, restored_agents = restore_runtime_snapshot(snapshot)

        original_next = env.step([0])
        restored_next = restored_env.step([0])
        self.assertEqual(restored_state["total_step"], 1)
        self.assertEqual(restored_agents[0].action, agent.action)
        self.assertEqual(original_next[1:3], restored_next[1:3])
        self.assertEqual(original_next[3]["inventory"], restored_next[3]["inventory"])
        self.assertTrue(np.array_equal(env._world._mat_map, restored_env._world._mat_map))
        self.assertTrue(
            np.array_equal(env._players[0].pos, restored_env._players[0].pos)
        )

    def test_agent_planner_subgraphs_run_in_parallel(self):
        """两个 Send 规划分支应在不同线程中同时到达屏障。"""
        barrier = threading.Barrier(2)
        thread_ids = set()
        lock = threading.Lock()

        class ParallelAgent(FakeAgent):
            def process_agent(agent, contexts, info):
                with lock:
                    thread_ids.add(threading.get_ident())
                barrier.wait(timeout=2)
                return {"id": agent.id, "thoughts": None}

        events = []
        simulation = LangGraphMultiAgentSimulation(
            [ParallelAgent(0, events), ParallelAgent(1, events)],
            FakeEnvironment(events),
            max_steps=1,
            n_players=2,
        )
        simulation.action_processor = FakeActionProcessor(events)
        simulation.env_manager = FakeEnvironmentManager(events)
        simulation.agent_state_manager = FakeAgentStateManager(events)
        simulation.simulation_context_manager = FakeContextManager(events)
        simulation.agent_thinking_processor = FakeThinkingProcessor(events)
        simulation.reporter = FakeReporter(events)

        result = simulation.run_simulation()

        planner_nodes = simulation.agent_planner_graph.get_graph().nodes
        self.assertIn("call_model", planner_nodes)
        self.assertIn("validate_response", planner_nodes)
        self.assertEqual(result["total_steps"], 1)
        self.assertEqual(len(thread_ids), 2)

    def test_llm_response_cache_avoids_duplicate_call(self):
        """同一步相同请求应只调用一次模型，随后直接命中缓存。"""
        with tempfile.TemporaryDirectory() as directory:
            cache = SimulationCheckpointStore(Path(directory) / "cache.sqlite")
            agent = FakeAgent(0)
            info = [{"sleeping": False, "inventory": {"wood": 0}}]
            contexts = {0: "same prompt"}

            with patch(
                "utils.Agent.process_agent",
                return_value={"id": 0, "thoughts": "cached decision"},
            ) as process_agent:
                first = AgentThinkingProcessor.process_agent_thinking_parallel(
                    [agent], contexts, info, cache, "cache-run", 4
                )
                second = AgentThinkingProcessor.process_agent_thinking_parallel(
                    [agent], contexts, info, cache, "cache-run", 4
                )

            self.assertEqual(first, second)
            self.assertEqual(process_agent.call_count, 1)

    def test_preserves_legacy_step_order(self):
        """父图每个环境步应保持 legacy 流水线的业务顺序。"""
        events = []
        simulation = LangGraphMultiAgentSimulation(
            [FakeAgent(0, events)], FakeEnvironment(events), max_steps=2, n_players=1
        )
        simulation.action_processor = FakeActionProcessor(events)
        simulation.env_manager = FakeEnvironmentManager(events)
        simulation.agent_state_manager = FakeAgentStateManager(events)
        simulation.simulation_context_manager = FakeContextManager(events)
        simulation.agent_thinking_processor = FakeThinkingProcessor(events)
        simulation.reporter = FakeReporter(events)

        result = simulation.run_simulation()

        per_step = [
            "step_header",
            "process_actions",
            "collect_actions",
            "env_step",
            "crafting_stations",
            "tool_availability",
            "update_agent_states",
            "identify_thought",
            "create_contexts",
            "plan_agent",
            "apply_responses",
            "step_report",
        ]
        self.assertEqual(events, ["reset"] + per_step * 2 + ["complete", "stats"])
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["final_obs"], ["observation-2"])
        self.assertEqual(result["final_rewards"], [2])

    def test_stops_when_environment_is_done(self):
        """环境返回 done 后不得继续推进后续步骤。"""
        events = []
        simulation = LangGraphMultiAgentSimulation(
            [FakeAgent(0, events)], FakeEnvironment(events), max_steps=10, n_players=1
        )
        simulation.action_processor = FakeActionProcessor(events)
        environment_manager = FakeEnvironmentManager(events)

        def done_after_first_step(env, actions):
            environment_manager.steps += 1
            events.append("env_step")
            return ["final"], [1], True, [{"sleeping": False}]

        environment_manager.step_environment = done_after_first_step
        simulation.env_manager = environment_manager
        simulation.agent_state_manager = FakeAgentStateManager(events)
        simulation.simulation_context_manager = FakeContextManager(events)
        simulation.agent_thinking_processor = FakeThinkingProcessor(events)
        simulation.reporter = FakeReporter(events)

        result = simulation.run_simulation()

        self.assertEqual(result["total_steps"], 1)
        self.assertTrue(result["completed"])
        self.assertEqual(events.count("env_step"), 1)

    def test_resumes_from_last_committed_step(self):
        """模拟崩溃后应从上一完整步恢复并复用原产物目录。"""
        class FailOnSecondPrepare(FakeActionProcessor):
            def __init__(self, events):
                super().__init__(events)
                self.calls = 0

            def process_all_agent_actions(self, agents, env, n_players):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated process crash")
                super().process_all_agent_actions(agents, env, n_players)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_db = Path(directory) / "checkpoints.sqlite"
            output_dir = Path(directory) / "experiment"
            first_events = []
            first = LangGraphMultiAgentSimulation(
                [FakeAgent(0, first_events)],
                FakeEnvironment(first_events),
                max_steps=3,
                n_players=1,
                checkpoint_db=checkpoint_db,
                run_id="resume-test",
                output_dir=output_dir,
            )
            first.action_processor = FailOnSecondPrepare(first_events)
            first.env_manager = FakeEnvironmentManager(first_events)
            first.agent_state_manager = FakeAgentStateManager(first_events)
            first.simulation_context_manager = FakeContextManager(first_events)
            first.agent_thinking_processor = FakeThinkingProcessor(first_events)
            first.reporter = FakeReporter(first_events)

            with self.assertRaisesRegex(RuntimeError, "simulated process crash"):
                first.run_simulation()

            second_events = []
            resumed = LangGraphMultiAgentSimulation(
                [FakeAgent(0, second_events)],
                FakeEnvironment(second_events),
                max_steps=3,
                n_players=1,
                checkpoint_db=checkpoint_db,
                run_id="resume-test",
                resume=True,
                output_dir=output_dir,
            )
            resumed.action_processor = FakeActionProcessor(second_events)
            resumed.env_manager = FakeEnvironmentManager(second_events)
            resumed.agent_state_manager = FakeAgentStateManager(second_events)
            resumed.simulation_context_manager = FakeContextManager(second_events)
            resumed.agent_thinking_processor = FakeThinkingProcessor(second_events)
            resumed.reporter = FakeReporter(second_events)

            result = resumed.run_simulation()

            self.assertEqual(result["total_steps"], 3)
            self.assertEqual(resumed.env.step_count, 3)
            self.assertEqual(second_events.count("env_step"), 2)
            self.assertEqual(
                SimulationCheckpointStore(checkpoint_db).get_output_dir("resume-test"),
                str(output_dir.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
