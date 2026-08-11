# 模块说明：无界面多智能体实验的主入口，负责初始化环境、运行仿真并保存统计结果。
from tqdm import tqdm
from utils import (
    AgentActionProcessor, 
    EnvironmentManager, 
    SimulationLogger,
    AgentStateManager,
    SimulationContextManager,
    AgentThinkingProcessor,
    get_arg_parser
)
from memory_system.agent import ActionStatus
import crafter
from memory_system.agent import Agent, HumanAgent
from plot import Plotter
import pandas as pd
import os

# 变量说明（MAX_STEPS）：允许本轮仿真执行的最大步数。
MAX_STEPS = 350
# 变量说明（N_PLAYERS）：共享环境中的玩家总数。
N_PLAYERS = 6

class CompleteMulitAgentSimulation:
    """Complete simulation that handles the full agent processing pipeline"""
    
    def __init__(self, agents, env, max_steps, n_players):
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（max_steps）：允许本轮仿真执行的最大步数。
        # 变量说明（n_players）：共享环境中的玩家总数。
        # 变量说明（self.agents）：参与当前仿真的全部智能体对象列表。
        self.agents = agents
        # 变量说明（self.env）：当前 Crafter 仿真环境实例。
        self.env = env
        # 变量说明（self.max_steps）：允许本轮仿真执行的最大步数。
        self.max_steps = max_steps
        # 变量说明（self.n_players）：共享环境中的玩家总数。
        self.n_players = n_players
        
        # Initialize processors
        # 变量说明（self.action_processor）：把 LLM 高层决策转换成环境动作的处理器。
        self.action_processor = AgentActionProcessor()
        # 变量说明（self.env_manager）：统一推进环境并同步工作台、熔炉等共享状态的管理器。
        self.env_manager = EnvironmentManager()
        # 变量说明（self.reporter）：负责终端步骤报告和实验统计输出的日志器。
        self.reporter = SimulationLogger()
        # 变量说明（self.agent_state_manager）：把环境观察和状态同步到各智能体的管理器。
        self.agent_state_manager = AgentStateManager()
        # 变量说明（self.simulation_context_manager）：为每个智能体组装个性化协作提示的管理器。
        self.simulation_context_manager = SimulationContextManager(self.n_players)
        # 变量说明（self.agent_thinking_processor）：并行调用各 AI 智能体思考流程的处理器。
        self.agent_thinking_processor = AgentThinkingProcessor()

    def run_simulation(self):
        """Run the complete simulation with progress tracking"""
        # 变量说明（total_step）：本轮已经执行的环境总步数。
        total_step = 0
        # 变量说明（_step）：当前经验或环境时间步编号。
        _step = 0
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        obs = self.env.reset()
        # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
        done = False
        # 变量说明（episode_number）：当前回合编号。
        episode_number = 0
        
        # 变量说明（bar）：显示仿真进度的 tqdm 进度条。
        with tqdm(total=self.max_steps, desc="Progress") as bar:
            while total_step < self.max_steps and not done:
                self.reporter.print_step_header(total_step, _step)
                
                # Process each agent's actions with specific logic
                self.action_processor.process_all_agent_actions(
                    self.agents, self.env, self.n_players
                )
                
                # Collect actions and step environment
                # 变量说明（agents_actions）：所有智能体在当前时间步提交给环境的动作编号列表。
                agents_actions = self.action_processor.collect_agent_actions(
                    self.agents, self.n_players
                )
                
                # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
                # 变量说明（rewards）：本时间步每个智能体获得的奖励值。
                obs, rewards, done, info = self.env_manager.step_environment(
                    self.env, agents_actions
                )
                self.env_manager.update_crafting_stations(self.agents, self.env)
                # 变量说明（tool_avaliablity_info）：列出各智能体当前持有镐类工具的协作状态文本。
                tool_avaliablity_info = self.env_manager.get_tool_availability_info(self.env, self.n_players)

                self.agent_state_manager.update_all_agent_states(
                    self.agents, obs, _step, self.env, info, episode_number
                )
                # 变量说明（agents_with_new_thought）：按编号标记本步哪些智能体完成了重新规划。
                agents_with_new_thought = self.agent_state_manager.identify_agents_needing_thought(self.agents, info)
                # 变量说明（agents_contexts）：以智能体编号为键的个性化决策上下文字典。
                agents_contexts = self.simulation_context_manager.create_agent_contexts(self.agents, info)
                # 变量说明（agents_responses）：多个智能体并行思考后返回的结构化结果列表。
                agents_responses = self.agent_thinking_processor.process_agent_thinking_parallel(
                    self.agents, agents_contexts, info
                )
                self.agent_thinking_processor.update_agents_from_responses(
                    self.agents, agents_responses
                )
                self.reporter.show_step_report(self.agents, agents_with_new_thought)
                # Update counters and progress
                total_step += 1
                _step += 1
                bar.update(1)
        
        self.reporter.print_simulation_complete()
        # self.reporter.print_stats(self.agents)
        # 变量说明（df）：保存实验统计数据的 Pandas DataFrame。
        df = self.reporter.log_stats(self.agents)


        return {
            'total_steps': total_step,
            'completed': done,
            'final_obs': obs,
            'final_rewards': rewards,
            'final_info': info,
            'stats_df': df
        }
    
    def get_agent_summary(self):
        """Get a summary of current agent states"""
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        return [
            {
                'id': agent.id,
                'op': agent.op,
                'action': agent.action,
                'action_status': agent.action_status,
                'rss_to_collect': getattr(agent, 'rss_to_collect', 'N/A'),
                'rss_to_share': getattr(agent, 'rss_to_share', 'N/A'),
                'target_agent_id': getattr(agent, 'target_agent_id', 'N/A')
            }
            for agent in self.agents
        ]

def initialize_agents(human_agent_ids=None, n_players=None):
    # 变量说明（human_agent_ids）：由用户手动控制的智能体编号集合。
    # 变量说明（n_players）：共享环境中的玩家总数。
    if human_agent_ids is None:
        human_agent_ids = []
    elif isinstance(human_agent_ids, str) and human_agent_ids.strip():
        # Parse comma-separated string of agent IDs
        # 变量说明（x）：世界网格或图像中的横向坐标。
        human_agent_ids = [int(x.strip()) for x in human_agent_ids.split(',') if x.strip().isdigit()]
    elif not isinstance(human_agent_ids, list):
        human_agent_ids = []
    
    # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
    agents = []
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    for i in range(n_players):
        if i in human_agent_ids:
            agents.append(HumanAgent(id=i, kg_update_freq=10))
        else:
            agents.append(Agent(id=i, kg_update_freq=10))
    return agents

def initialize_environment(num_steps=None, n_players=None):
    # 变量说明（num_steps）：本次运行配置的环境步数上限。
    # 变量说明（n_players）：共享环境中的玩家总数。
    return crafter.Env(length=num_steps, n_players=n_players, seed=4)


def main():
    # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
    parser = get_arg_parser()
    # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
    args = parser.parse_args()
    # 变量说明（num_rounds）：需要重复执行的实验轮数。
    num_rounds = args.num_of_turn if hasattr(args, 'num_of_turn') else 1
    # 变量说明（num_steps）：本次运行配置的环境步数上限。
    num_steps = args.step_num if hasattr(args, 'step_num') else 350
    # 变量说明（human_agent_ids）：由用户手动控制的智能体编号集合。
    human_agent_ids = args.human_agents if hasattr(args, 'human_agents') else None
    # 变量说明（agent_num）：本轮实验创建的智能体数量。
    agent_num = args.agent_num if hasattr(args, 'agent_num') else 1
    # 变量说明（results_dir）：保存统计表、观察帧和其他实验产物的目录。
    results_dir = args.results_dir if hasattr(args, 'results_dir') else 'results'
    os.makedirs(results_dir, exist_ok=True)
    # 变量说明（all_stats）：等待合并的各轮实验统计表列表。
    all_stats = []
    # 变量说明（round_idx）：当前实验轮次的零基索引。
    for round_idx in range(num_rounds):
        print(f"=== Simulation Round {round_idx+1} ===")
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        agents = initialize_agents(human_agent_ids=human_agent_ids, n_players=agent_num)
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        env = initialize_environment(num_steps=num_steps, n_players=agent_num)
        # 变量说明（simulation）：组织智能体、环境和完整时间步循环的仿真控制器。
        simulation = CompleteMulitAgentSimulation(agents, env, num_steps, agent_num)
        # 变量说明（results_df）：当前一轮实验产生的统计 DataFrame。
        results_df = simulation.run_simulation()['stats_df']
        # Save stats, file name contains round number and agent number
        # 变量说明（save_path）：当前产物写入磁盘时使用的完整路径。
        save_path = os.path.join(results_dir, f"stats_round_{round_idx+1}_agent_{agent_num}.csv")
        results_df.to_csv(save_path, index=False)
        all_stats.append(results_df)
    # combine all rounds results
    # 变量说明（final_df）：合并所有实验轮次后的统计 DataFrame。
    final_df = pd.concat(all_stats, ignore_index=True)
    # 变量说明（all_rounds_path）：`main` 读取或写入当前文件时使用的完整路径。
    all_rounds_path = os.path.join(results_dir, "all_rounds_stats.csv")

    # if file exists, append to it, otherwise create a new file
    final_df.to_csv(
        all_rounds_path,
        mode='a',
        header=not os.path.exists(all_rounds_path),
        index=False
    )
    print("Appended results to all_rounds_stats.csv")
    if not getattr(args, 'no_plot', False):
        # 变量说明（plotter）：负责读取统计 CSV 并生成实验对比图的绘图器。
        plotter = Plotter()
        plotter.plot([all_rounds_path], legend_labels=['Six Agents with Communication'])


if __name__ == "__main__":
    main()
