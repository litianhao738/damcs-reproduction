# 模块说明：多智能体仿真的行动处理、状态管理、协作上下文和日志工具。
import matplotlib.pyplot as plt
from memory_system.agent import ActionStatus, Agent
import pandas as pd
from crafter import constants as const
from memory_system.utils import go_and_find
from typing import List
from memory_system.utils import print_color
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import argparse
import os
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

class AgentActionProcessor:
    """Handles agent action processing"""
    
    @staticmethod
    def process_all_agent_actions(agents, env, n_players):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（n_players）：共享环境中的玩家总数。
        """Process actions for all agents with different operation types"""
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            AgentActionProcessor._process_single_agent_action(agent, agents, env, n_players)
    
    @staticmethod
    def _process_single_agent_action(agent, agents, env, n_players):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（n_players）：共享环境中的玩家总数。
        """Process action for a single agent based on operation type"""
        if agent.op == 'Navigator':
            AgentActionProcessor._process_navigator_action(agent)
        elif agent.op == 'share':
            AgentActionProcessor._process_share_action(agent, agents, env, n_players)
        else:
            AgentActionProcessor._process_general_action(agent)
    
    @staticmethod
    def _process_navigator_action(agent):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        """Process navigator operation for an agent"""
        agent.create_experience()  # Update working memory
        
        if agent.rss_to_collect != 'not_applicable':
            # 变量说明（action）：当前智能体准备执行的环境动作。
            # 变量说明（is_finished）：导航器是否已经到达目标并完成交互的布尔标记。
            action, is_finished = go_and_find(agent.info, agent.rss_to_collect)
            agent.update_action(const.actions[action])
            
            if agent.action_status == ActionStatus.IN_PROGRESS and is_finished:
                agent.update_action_status(ActionStatus.ALMOST_DONE)
            elif agent.action_status == ActionStatus.ALMOST_DONE and is_finished:
                agent.update_action_status(ActionStatus.DONE)
        else:
            agent.update_action('noop')
            agent.update_action_status(ActionStatus.DONE)
    
    @staticmethod
    def _process_share_action(agent, agents, env, n_players):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（n_players）：共享环境中的玩家总数。
        """Process share operation for an agent"""
        if agent.rss_to_share != 'not_applicable':
            env.switch_player(agent.id)
            env.exchange_item(agent.target_agent_id, agent.rss_to_share)
        
        agent.update_action('noop')
        agent.update_action_status(ActionStatus.DONE)
        
        # Interrupt all agents after the target agent
        AgentActionProcessor._interrupt_agents_after_target(agents, agent.target_agent_id, n_players)
    
    @staticmethod
    def _process_general_action(agent):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        """Process general operation for an agent"""
        agent.update_action(agent.op)
        agent.update_action_status(ActionStatus.DONE)
    
    @staticmethod
    def _interrupt_agents_after_target(agents, target_agent_id, n_players):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（target_agent_id）：资源共享操作的目标智能体编号；-1 表示无目标。
        # 变量说明（n_players）：共享环境中的玩家总数。
        """Interrupt all agents after the target agent"""
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        for i in range(target_agent_id, n_players):
            agents[i].update_action_status(ActionStatus.INTERRUPTED)
    
    @staticmethod
    def collect_agent_actions(agents, n_players):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（n_players）：共享环境中的玩家总数。
        """Collect all agents' actions for environment stepping"""
        # 变量说明（agents_actions）：所有智能体在当前时间步提交给环境的动作编号列表。
        agents_actions = [0] * n_players      
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            agents_actions[agent.id] = const.actions.index(agent.action)
        return agents_actions


class EnvironmentManager:
    """Manages environment state and updates"""
    
    @staticmethod
    def step_environment(env, agents_actions):
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（agents_actions）：所有智能体在当前时间步提交给环境的动作编号列表。
        """Step the environment with collected actions"""
        # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        # 变量说明（rewards）：本时间步每个智能体获得的奖励值。
        obs, rewards, done, info = env.step(agents_actions)
        if getattr(env, 'n_players', 1) == 1:
            return [obs], [rewards], done, [info]
        return obs, rewards, done, info
    
    @staticmethod
    def update_crafting_stations(agents, env):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        """Update crafting station status for all agents"""
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            if env._unlocked and "place_table" in env._unlocked:
                agent.update_crafting_station_status("table")
            if env._unlocked and 'place_furnace' in env._unlocked:
                agent.update_crafting_station_status("furnace")
    
    @staticmethod
    def get_tool_availability_info(env, n_players):
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（n_players）：共享环境中的玩家总数。
        """Get information about tool availability across all agents"""
        # 变量说明（tool_availability_info）：`get_tool_availability_info` 中当前处理的工具、可用性、状态信息。
        tool_availability_info = ""
        # 变量说明（tool）：`get_tool_availability_info` 中当前处理的工具。
        for tool in ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe']:
            # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
            for agent_id in range(n_players):
                if env._players[agent_id].inventory[tool] > 0:
                    tool_availability_info += f"Agent {agent_id} has {env._players[agent_id].inventory[tool]} {tool}\n"
        return tool_availability_info

class AgentStateManager:
    """Manages agent state updates and reporting"""
    
    @staticmethod
    def update_all_agent_states(agents, obs, step, env, info, episode_number):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        # 变量说明（step）：当前经验或环境时间步编号。
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        # 变量说明（episode_number）：当前回合编号。
        """Update states for all agents"""
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            agent.update_state(obs[agent.id], step, env, info[agent.id], episode_number, episode_timestep=step)
            print_color(f"Player {agent.id} is doing {agent.op, agent.rss_to_collect, agent.rss_to_share, agent.target_agent_id} with action {agent.action}; current status: {agent.action_status}", color='green')
    
    @staticmethod
    def identify_agents_needing_thought(agents, info):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        """Identify which agents need new thoughts"""
        # 变量说明（agents_with_new_thought）：按编号标记本步哪些智能体完成了重新规划。
        agents_with_new_thought = [False] * len(agents)
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            if agent.action_status != ActionStatus.IN_PROGRESS and info[agent.id]['sleeping'] == False:
                agents_with_new_thought[agent.id] = True
        return agents_with_new_thought


class WorkingMemoryParser:
    """Handles parsing of agent working memory"""
    
    @staticmethod
    def parse_agent_working_memory(agent):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        """Parse agent's working memory content"""
        if agent.wm_content == "":
            return None, None
        
        try:
            # 变量说明（wm_parts）：按标题切分后的工作记忆文本片段列表。
            wm_parts = agent.wm_content.split("### ")[1].split("- For ")
            
            if len(wm_parts) >= 3:
                # 变量说明（long_term_goal_req）：`parse_agent_working_memory` 中当前处理的目标。
                long_term_goal_req = WorkingMemoryParser._extract_goal_requirement(wm_parts[1])
                # 变量说明（current_goal_req）：`parse_agent_working_memory` 中当前处理的目标。
                current_goal_req = WorkingMemoryParser._extract_goal_requirement(wm_parts[2])
                return long_term_goal_req, current_goal_req
        except (IndexError, AttributeError):
            return None, None
        
        return None, None
    
    @staticmethod
    def _extract_goal_requirement(goal_text):
        # 变量说明（goal_text）：`_extract_goal_requirement` 中当前处理的目标、文本。
        """Extract and clean goal requirement text"""
        # 变量说明（goal_req）：`_extract_goal_requirement` 中当前处理的目标。
        goal_req = goal_text.split(": ")[1:]
        goal_req = ', '.join(goal_req)
        
        if "* " in goal_req:
            goal_req = goal_req.split("* ")[1]
        
        return goal_req

class GoalGenerator:
    """Generates goals based on agent status and collaboration needs"""
    
    @staticmethod
    def generate_goal_from_agent_status(previous_agent, long_term_goal, current_goal):
        # 变量说明（previous_agent）：`generate_goal_from_agent_status` 中当前处理的智能体。
        # 变量说明（long_term_goal）：`generate_goal_from_agent_status` 中当前处理的目标。
        # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
        """Generate goal text based on agent's status"""
        if not long_term_goal or not current_goal:
            return "Agent status unclear.\n"
        
        if "help_agent" in long_term_goal:
            return GoalGenerator._generate_help_based_goal(previous_agent, current_goal)
        else:
            return GoalGenerator._generate_work_based_goal(previous_agent, long_term_goal, current_goal)
    
    @staticmethod
    def _generate_help_based_goal(previous_agent, current_goal):
        # 变量说明（previous_agent）：`_generate_help_based_goal` 中当前处理的智能体。
        # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
        """Generate goal when previous agent is in help mode"""
        if 'ready' in current_goal or 'Ready' in current_goal:
            return f"Agent {previous_agent.id} is completing the task.\n"
        else:
            return (
                f"Agent {previous_agent.id} is working on {current_goal}"
                f"If he needs any tool, work on the tool and share with him.\n"
                f"If he needs anything else besides what he is working on, you should navigate to the other material and ask the next agent to share the appropriate tool with you.\n"
                f"Else, work on advancing your tools.\n"
            )
    
    @staticmethod
    def _generate_work_based_goal(previous_agent, long_term_goal, current_goal):
        # 变量说明（previous_agent）：`_generate_work_based_goal` 中当前处理的智能体。
        # 变量说明（long_term_goal）：`_generate_work_based_goal` 中当前处理的目标。
        # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
        """Generate goal when previous agent is in work mode"""
        if 'ready' in long_term_goal or 'Ready' in long_term_goal:
            return f"Agent {previous_agent.id} is completing its task. You should focus on advancing your tools.\n"
        else:
            if "work on long-term goal" in current_goal:
                return (
                    f"Agent {previous_agent.id} is completing {long_term_goal}.\n"
                    f"If he needs any tool, work on the tool and share with him.\n"
                    f"If he needs anything else besides what he is working on, you should navigate to the other material and ask the next agent to share the appropriate tool with you.\n"
                    f"Else, work on advancing your tools.\n"
                )
            else:
                return (
                    f"Agent {previous_agent.id} is working on {long_term_goal} He is focusing on {current_goal} "
                    f"If he needs any tool, work on the tool and share with him.\n"
                    f"If he needs anything else besides what he is working on, you should navigate to the other material and ask the next agent to share the appropriate tool with you.\n"
                    f"Else, work on advancing your tools.\n"
                )

class CollaborationContextGenerator:
    """Generates collaboration context for different agent types"""
    
    def __init__(self, n_players):
        # 变量说明（n_players）：共享环境中的玩家总数。
        # 变量说明（self.n_players）：共享环境中的玩家总数。
        self.n_players = n_players
        # 变量说明（self.starting_furnace）：协作流程是否已经进入收集石头并准备建造熔炉的阶段标记。
        self.starting_furnace = False
    
    def generate_context_for_agent(self, agent, agents):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        """Generate collaboration context for a specific agent"""
        if agent.id == 0:
            return self._generate_agent_0_context()
        elif agent.id == self.n_players - 1:
            return self._generate_last_agent_context(agent)
        else:
            return self._generate_middle_agent_context(agent, agents)
    
    def _generate_agent_0_context(self):
        """Generate context for agent 0 (main agent)"""
        return (
            "\n### Collaboration: Need Your Help \n"
            f"You are agent 0. Focus on your tasks while other agents share resources with you. "
            f"Once you craft an iron_pickaxe, you should share it to agent {self.n_players-1}\n"
        )
    
    def _generate_last_agent_context(self, agent):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        """Generate context for the last agent (diamond collector)"""
        # 变量说明（goal）：`_generate_last_agent_context` 中当前处理的目标。
        goal = "While other agents are working on make_iron_pickaxe, your long-term goal and current goal is to !!collect diamond!! You should navigate to a diamond, and collect the diamond. Other agent will share the tool with you.\n"
        
        return (
            "\n### Collaboration: Need Your Help!!!\n"
            f"{goal}"
            "\n### Collaboration Policy \n"
            f"You are agent {agent.id}. Your goal is to collect diamond. You should navigate to a diamond and collect the diamond. Other agents will share the tool with you.\n"
        )
    
    def _generate_middle_agent_context(self, agent, agents):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        """Generate context for middle agents"""
        # 变量说明（previous_agent）：`_generate_middle_agent_context` 中当前处理的智能体。
        previous_agent = agents[agent.id-1]
        # 变量说明（goal）：`_generate_middle_agent_context` 中当前处理的目标。
        goal = self._parse_previous_agent_goal(previous_agent)
        
        if agent.id == 1:
            return self._generate_agent_1_context(agent, goal)
        else:
            return self._generate_other_middle_agent_context(agent, goal)
    
    def _parse_previous_agent_goal(self, previous_agent):
        # 变量说明（previous_agent）：`_parse_previous_agent_goal` 中当前处理的智能体。
        """Parse the previous agent's working memory to determine current goal"""
        # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
        # 变量说明（long_term_goal）：`_parse_previous_agent_goal` 中当前处理的目标。
        long_term_goal, current_goal = WorkingMemoryParser.parse_agent_working_memory(previous_agent)
        return GoalGenerator.generate_goal_from_agent_status(previous_agent, long_term_goal, current_goal)
    
    def _generate_agent_1_context(self, agent, goal):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（goal）：`_generate_agent_1_context` 中当前处理的目标。
        """Generate context specifically for agent 1"""
        if self.starting_furnace or ('make_stone_pickaxe' in goal):
            self.starting_furnace = True
            return (
                "\n### Collaboration: Need Your Help!!!\n"
                f"While other agents working towards make_iron_pickaxe, your long-term goal and current goal should be !!share stone to agent 0!! You should focus on !!collect stone!! and share stone immediately.\n"
                "\n### Collaboration Policy \n"
                f"You are agent {agent.id}. Your ultimate goal is help_agent 0.\n"
                f"You long-term goal should be help_agent 0. Your current goal should be collect and share stone.\n"
            )
        else:
            return (
                "\n### Collaboration: Need Your Help!!!\n"
                f"{goal}"
                "\n### Collaboration Policy \n"
                f"You are agent {agent.id}. Your ultimate goal is help_agent 0. "
                f"You long-term goal should be 'help_agent'. You can do so by sharing resources/tools as soon as you gain access to the required items.\n"
                f"!!!Determine your current goal based on agents who need help. If they don't need help, you should focus on advancing your tools.\n"
            )
    
    def _generate_other_middle_agent_context(self, agent, goal):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（goal）：`_generate_other_middle_agent_context` 中当前处理的目标。
        """Generate context for other middle agents"""
        return (
            "\n### Collaboration: Need Your Help!!!\n"
            f"{goal}"
            "\n### Collaboration Policy \n"
            f"You are agent {agent.id}. Your ultimate goal is help_agent 0 and help_agent {agent.id-1}. You should prioritize helping agent 0.\n"
            f"You long-term goal should be 'help_agent'. You can do so by sharing resources/tools as soon as you gain access to the required items.\n"
            f"Do not share resources/tools that are not needed by the agents.\n"
            f"Try to help the agents. If they don't need help, you should focus on advancing your tools.\n"
            f"If you are not sure what to do, share stone to agent 0.\n"
            f"Reconsider how you can help the previous agent.\n"
            f"!!!Determine your current goal based on agents who need help. If they don't need help, you should focus on advancing your tools.\n"
        )

class AgentThinkingProcessor:
    """Handles parallel processing of agent thinking"""
    
    @staticmethod
    def process_agent_thinking_parallel(agents, agents_contexts, info):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（agents_contexts）：以智能体编号为键的个性化决策上下文字典。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        """Process agent thinking in parallel using ThreadPoolExecutor, but skip HumanAgents"""
        # 变量说明（agents_responses）：多个智能体并行思考后返回的结构化结果列表。
        agents_responses = []
        

        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（ai_agents）：需要调用大模型进行决策的非人工智能体列表。
        ai_agents = [agent for agent in agents if not (hasattr(agent, 'is_human') and agent.is_human)]
        
        if ai_agents:
            # 变量说明（executor）：用于并行执行多个智能体思考任务的线程池。
            with ThreadPoolExecutor() as executor:
                # 变量说明（futures）：线程池任务句柄到对应智能体对象的映射。
                futures = {
                    executor.submit(Agent.process_agent, agent, agents_contexts, info): agent
                    for agent in ai_agents
                }
                # 变量说明（future）：线程池中某个智能体尚未完成或已经完成的思考任务句柄。
                for future, agent in futures.items():
                    try:
                        # 变量说明（response）：当前 API 调用、任务或模型推理产生的响应对象。
                        response = future.result()
                        response["error"] = None
                        agents_responses.append(response)
                    # 变量说明（exc）：当前捕获到的异常对象。
                    except Exception as exc:
                        # 变量说明（error）：当前操作失败时记录的异常说明。
                        error = f"{type(exc).__name__}: {exc}"
                        print_color(
                            f"Agent {agent.id} thinking failed for this step: "
                            f"{error}",
                            color='red',
                        )
                        agents_responses.append(
                            {"id": agent.id, "thoughts": None, "error": error}
                        )
        
        return agents_responses
    
    @staticmethod
    def update_agents_from_responses(agents, agents_responses):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（agents_responses）：多个智能体并行思考后返回的结构化结果列表。
        """Update agents based on their thinking responses"""
        # 变量说明（response）：当前 API 调用、任务或模型推理产生的响应对象。
        for response in agents_responses:
            # 变量说明（agent）：当前正在处理的单个智能体对象。
            agent = agents[response['id']]
            if response['thoughts'] is not None:
                agent.thinking_error = None
                # 变量说明（op）：LLM 选定的高层操作名称，例如导航、制造或共享。
                # 变量说明（rss_to_collect）：导航操作要寻找并采集的目标资源名称。
                # 变量说明（rss_to_share）：共享操作准备转交给其他智能体的资源名称。
                # 变量说明（target_agent_id）：资源共享操作的目标智能体编号；-1 表示无目标。
                op, rss_to_collect, rss_to_share, target_agent_id = agent.consolidate_experience(response['thoughts'])
                agent.update_current_skill(op, rss_to_collect, rss_to_share, target_agent_id)
                agent.update_action_status(ActionStatus.IN_PROGRESS)
            elif response.get('error'):
                agent.thinking_error = response['error']


class SimulationLogger:
    """Handles simulation reporting and logging"""
    def __init__(self):
        # Task mapping: easy to modify and extend
        # 变量说明（self.tasks）：`__init__` 中待批量处理的任务集合。
        self.tasks = {
            "Collect wood": "collect_wood",
            "Place table": "place_table",
            "Make wood pickaxe": "make_wood_pickaxe",
            "Collect stone": "collect_stone",
            "Make stone pickaxe": "make_stone_pickaxe",
            "Collect iron": "collect_iron",
            "Collect coal": "collect_coal",
            "Place furnace": "place_furnace",
            "Make iron pickaxe": "make_iron_pickaxe",
            "Collect diamond": "collect_diamond",
        }

    @staticmethod
    def print_step_header(total_step, current_step):
        # 变量说明（total_step）：本轮已经执行的环境总步数。
        # 变量说明（current_step）：`print_step_header` 中当前处理的时间步。
        """Print step header information"""
        print_color("="*50, "total step: ", total_step, "current step: ", current_step, "="*50, color='green')
    
    @staticmethod
    def print_simulation_complete():
        """Print simulation completion message"""
        print_color("Simulation completed!", color='green')
    
    @staticmethod
    def show_step_report(agents, agents_with_new_thought):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（agents_with_new_thought）：按编号标记本步哪些智能体完成了重新规划。
        # 变量说明（report）：准备输出到终端或返回给调用方的检查结果字典。
        report = []
        # fig, ax = plt.subplots(1, len(agents), figsize=(2 * len(agents), 2))
        # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
        # 变量说明（has_new_thought）：表示 `has_new_thought` 所描述条件是否成立的布尔标记。
        for agent_id, has_new_thought in enumerate(agents_with_new_thought):
            # 变量说明（agent）：当前正在处理的单个智能体对象。
            agent = agents[agent_id]
            # figures
            # ax[agent.id].imshow(agent.obs)
            # ax[agent.id].set_axis_off()
            # ax[agent.id].set_title(f"Player {agent.id}")
            # info
            if has_new_thought:
                # Handle HumanAgent differently
                if hasattr(agent, 'is_human') and agent.is_human:
                    # 变量说明（context）：发送给大模型的当前决策上下文文本。
                    context = "Human controlled agent"
                    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
                    info = (
                            f"long_term_goal: Human controlled\n\n"
                            f"current_goal: Human controlled\n\n"
                            f"op: {agent.op}\n\n"
                            f"navigate to: {agent.rss_to_collect}\n\n"
                            f"share: {agent.rss_to_share}\n\n"
                            f"target_agent_id: {agent.target_agent_id}\n"
                    )
                    # 变量说明（summary）：对当前经验、目标或统计结果的精简文本摘要。
                    summary = f"Human agent {agent.id} chose: {agent.op}"
                else:
                    context = agent.think_context
                    # 变量说明（thinking_error）：智能体最近一次调用模型失败时记录的异常文本。
                    thinking_error = getattr(agent, 'thinking_error', None)
                    if thinking_error:
                        report.append({
                            "agent_id": agent.id,
                            "context": context,
                            "info": thinking_error,
                            "summary": "Thinking failed for this step; retrying next step.",
                        })
                        continue
                    # 变量说明（final_response）：本次思考最终采用的结构化 LLM 输出。
                    final_response = agent.experience.episodic_memory.final_response
                    # 变量说明（action）：当前智能体准备执行的环境动作。
                    # 变量说明（goal）：`show_step_report` 中当前处理的目标。
                    # 变量说明（reflection）：LLM 对上一动作、结果和重复行为的结构化反思字段。
                    reflection, goal, action = final_response.reflection, final_response.goal, final_response.action
                    info = (
                            f"long_term_goal: {goal.long_term_goal.value}\n\n"
                            f"current_goal: {goal.current_goal.value}\n\n"
                            f"op: {agent.op.value}\n\n"
                            f"navigate to: {agent.rss_to_collect.value}\n\n"
                            f"share: {agent.rss_to_share.value}\n\n"
                            f"target_agent_id: {agent.target_agent_id}\n"
                    )
                    summary = final_response.summary
            
                report.append({"agent_id": agent.id, 
                            "context": context,
                            "info": info,
                            "summary": summary})
            else:
                report.append({"agent_id": agent.id,
                            "context": "in progress",
                            "info": "in progress",
                            "summary": "in progress"})
                            
        # 变量说明（entry）：准备写入知识图谱的单步经验字段集合。
        for entry in report:
            # 变量说明（agent_type）：`show_step_report` 中当前处理的智能体。
            agent_type = "HumanAgent" if hasattr(agents[entry['agent_id']], 'is_human') and agents[entry['agent_id']].is_human else "Agent"
            print(f"\n[{agent_type} {entry['agent_id']}]")
            print("Context (Thinking):\n", entry['context'])
            # print("Info:\n", entry['info'])
            # print("Summary:\n", entry['summary'])
    
    def log_stats(self, agents):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（min_completion_steps）：`log_stats` 中待批量处理的时间步集合。
        min_completion_steps = []
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            # 变量说明（df）：保存实验统计数据的 Pandas DataFrame。
            df = pd.DataFrame(agent.replay)
            # Find the first step for each task completion
            # 变量说明（completion_steps）：`log_stats` 中待批量处理的时间步集合。
            # 变量说明（task_col）：`log_stats` 中当前处理的任务。
            # 变量说明（task_name）：`log_stats` 中当前处理的任务。
            completion_steps = {task_name: df[df[task_col] == 1]['step'].min() for task_name, task_col in self.tasks.items()}
            min_completion_steps.append(completion_steps)
        df = pd.DataFrame(min_completion_steps)
        print("Minimum step for each agent to complete each task:")
        print(df)
        print("\nEarliest step (across all agents) for each task:")
        # 变量说明（earliest_steps）：`log_stats` 中待批量处理的时间步集合。
        earliest_steps = df.min()
        print(earliest_steps)
        # 变量说明（earliest_df）：按步骤排序后用于确定首次完成时间的实验统计表。
        earliest_df = pd.DataFrame([earliest_steps])
        return earliest_df
    

class SimulationContextManager:
    """Manages the creation of simulation contexts for agents"""
    
    def __init__(self, n_players):
        # 变量说明（n_players）：共享环境中的玩家总数。
        # 变量说明（self.context_generator）：`__init__` 中当前处理的上下文。
        self.context_generator = CollaborationContextGenerator(n_players)
    
    def create_agent_contexts(self, agents, info):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        """Create contexts for all agents that need to think"""
        # 变量说明（agents_contexts）：以智能体编号为键的个性化决策上下文字典。
        agents_contexts = {}
        # agents_with_new_thought = [False] * len(agents)
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            # 变量说明（context）：发送给大模型的当前决策上下文文本。
            context = agent.create_experience()
            print(f"agent_{agent.id}", agent.wm_content)
            
            if agent.action_status != ActionStatus.IN_PROGRESS and info[agent.id]['sleeping'] == False:
                # agents_with_new_thought[agent.id] = True
                # 变量说明（collaboration_context）：`create_agent_contexts` 中当前处理的上下文。
                collaboration_context = self.context_generator.generate_context_for_agent(agent, agents)
                context += collaboration_context
                agents_contexts[agent.id] = context
        
        return agents_contexts

def get_arg_parser():
    # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
    parser = argparse.ArgumentParser(description='Run multi-agent simulation with optional human agents.')
    parser.add_argument('--human_agents', type=str, default='',
                        help='Comma-separated list of agent ids to be human-controlled, e.g. "0,2". Default: all AI agents.')
    parser.add_argument('--memory', action='store_true', default=False,
                        help='Whether agents have memory (default: False).')
    parser.add_argument('--communication', action='store_true', default=False,
                        help='Whether agents have communication (default: False).')
    parser.add_argument('--agent_num', type=int, default=6,
                        help='Number of agents (default: 6).')
    parser.add_argument('--step_num', type=int, default=350,
                        help='Number of steps (default: 350).')
    parser.add_argument('--num_of_turn', type=int, default=1,
                        help='Number of simulation turns (default: 1).')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for simulation outputs (default: results).')
    parser.add_argument('--no_plot', action='store_true', default=False,
                        help='Skip plotting after simulation.')
    # Add more custom arguments here as needed
    return parser
