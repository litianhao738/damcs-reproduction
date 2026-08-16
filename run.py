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
from langgraph_simulation import LangGraphMultiAgentSimulation
from checkpointing import SimulationCheckpointStore
from experiment_outputs import (
    create_experiment_directory,
    read_manifest,
    reopen_experiment_directory,
    update_manifest,
)
import pandas as pd
import uuid
from pathlib import Path

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

def initialize_agents(human_agent_ids=None, n_players=None, output_dir=None):
    """按玩家编号创建 Agent/HumanAgent，并绑定本轮独立产物目录。"""
    # 变量说明（human_agent_ids）：由用户手动控制的智能体编号集合。
    # 变量说明（n_players）：共享环境中的玩家总数。
    if human_agent_ids is None:
        human_agent_ids = []
    elif isinstance(human_agent_ids, str) and human_agent_ids.strip():
        # 将命令行逗号分隔的人工 Agent 编号转换为整数列表。
        # 变量说明（x）：世界网格或图像中的横向坐标。
        human_agent_ids = [int(x.strip()) for x in human_agent_ids.split(',') if x.strip().isdigit()]
    elif not isinstance(human_agent_ids, list):
        human_agent_ids = []
    
    # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
    agents = []
    output_dir = Path(output_dir or "results")
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    for i in range(n_players):
        if i in human_agent_ids:
            agents.append(HumanAgent(id=i, kg_update_freq=10, output_dir=output_dir / f"agent_{i}"))
        else:
            agents.append(Agent(id=i, kg_update_freq=10, output_dir=output_dir / f"agent_{i}"))
    return agents

def initialize_environment(num_steps=None, n_players=None):
    """使用固定随机种子创建可复现的多人 Crafter 环境。"""
    # 变量说明（num_steps）：本次运行配置的环境步数上限。
    # 变量说明（n_players）：共享环境中的玩家总数。
    return crafter.Env(length=num_steps, n_players=n_players, seed=4)


def main():
    """解析命令行参数，运行全部轮次并保存隔离的实验产物。"""
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
    # 恢复依赖 LangGraph 的结构化 checkpoint，且当前只支持单轮实验。
    if args.resume and args.orchestrator != 'langgraph':
        parser.error('--resume is only supported by --orchestrator langgraph')
    if args.resume and not args.run_id:
        parser.error('--resume requires --run_id')
    if args.resume and num_rounds != 1:
        parser.error('--resume requires --num_of_turn 1')
    # 未显式指定时生成随机 run_id；同一多轮实验共享此前缀。
    base_run_id = args.run_id or f"damcs-{uuid.uuid4().hex[:12]}"
    # results_root 是实验根目录，实际产物还会进入 UTC_run-id 子目录。
    results_root = args.results_dir if hasattr(args, 'results_dir') else 'results'
    # manifest 保存解析后的完整参数，便于复现实验。
    config = vars(args).copy()
    config["run_id"] = base_run_id

    # 恢复时以 checkpoint 记录为准，忽略新传入的 results_dir 位置。
    if args.resume:
        checkpoint_store = SimulationCheckpointStore(args.checkpoint_db)
        try:
            stored_output_dir = checkpoint_store.get_output_dir(base_run_id)
        except ValueError as error:
            parser.error(str(error))
        if not stored_output_dir:
            parser.error(
                "This checkpoint predates isolated experiment directories and "
                "cannot be resumed safely. Start a new run_id."
            )
        experiment_dir = reopen_experiment_directory(stored_output_dir, base_run_id)
    else:
        # 新实验先检查 run_id 冲突，避免创建无用目录后才失败。
        if args.orchestrator == 'langgraph':
            checkpoint_store = SimulationCheckpointStore(args.checkpoint_db)
            round_run_ids = [
                base_run_id if num_rounds == 1
                else f"{base_run_id}-round-{round_index + 1}"
                for round_index in range(num_rounds)
            ]
            duplicates = [
                run_id for run_id in round_run_ids
                if checkpoint_store.exists(run_id)
            ]
            if duplicates:
                parser.error(
                    "Checkpoint run id already exists: "
                    + ", ".join(repr(run_id) for run_id in duplicates)
                    + ". Choose another --run_id or pass --resume."
                )
        experiment_dir = create_experiment_directory(
            results_root, base_run_id, config
        )

    print(f"Experiment output: {experiment_dir}")
    # 收集各轮统计表，最后一次性写入实验级汇总 CSV。
    all_stats = []
    try:
        # 变量说明（round_idx）：当前实验轮次的零基索引。
        for round_idx in range(num_rounds):
            print(f"=== Simulation Round {round_idx+1} ===")
            # 多轮实验为每一轮使用独立 checkpoint run_id。
            round_run_id = (
                base_run_id
                if num_rounds == 1
                else f"{base_run_id}-round-{round_idx + 1}"
            )
            # 每轮的观察帧、知识图谱和 stats.csv 都写入自己的目录。
            round_dir = experiment_dir / f"round_{round_idx + 1:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
            agents = initialize_agents(
                human_agent_ids=human_agent_ids,
                n_players=agent_num,
                output_dir=round_dir,
            )
            # 变量说明（env）：当前 Crafter 仿真环境实例。
            env = initialize_environment(num_steps=num_steps, n_players=agent_num)
            # 变量说明（simulation）：组织智能体、环境和完整时间步循环的仿真控制器。
            if args.orchestrator == 'langgraph':
                print(f"Checkpoint run id: {round_run_id}")
                simulation = LangGraphMultiAgentSimulation(
                    agents,
                    env,
                    num_steps,
                    agent_num,
                    checkpoint_db=args.checkpoint_db,
                    run_id=round_run_id,
                    resume=args.resume,
                    max_concurrency=args.max_concurrency,
                    output_dir=experiment_dir,
                )
            else:
                simulation = CompleteMulitAgentSimulation(
                    agents, env, num_steps, agent_num
                )
            # LangGraph 和 legacy 控制器返回相同形状的结果字典。
            simulation_result = simulation.run_simulation()
            results_df = simulation_result['stats_df']
            save_path = round_dir / "stats.csv"
            results_df.to_csv(save_path, index=False)
            all_stats.append(results_df)

            # 用当前轮结果替换同索引旧记录，使恢复后重复写入仍然幂等。
            manifest = read_manifest(experiment_dir)
            rounds = [
                item for item in manifest.get("rounds", [])
                if item.get("index") != round_idx + 1
            ]
            rounds.append({
                "index": round_idx + 1,
                "run_id": round_run_id,
                "directory": round_dir.name,
                "total_steps": int(simulation_result['total_steps']),
                "completed": bool(simulation_result['completed']),
            })
            update_manifest(experiment_dir, rounds=sorted(rounds, key=lambda item: item["index"]))

        # 汇总文件直接重写，不再把独立实验追加到同一个 CSV。
        final_df = pd.concat(all_stats, ignore_index=True)
        all_rounds_path = experiment_dir / "all_rounds_stats.csv"
        final_df.to_csv(all_rounds_path, index=False)
        print(f"Saved combined results: {all_rounds_path}")
        if not getattr(args, 'no_plot', False):
            plotter = Plotter()
            plotter.plot(
                [str(all_rounds_path)],
                legend_labels=['Six Agents with Communication'],
                save_path=str(experiment_dir / "result_fig.png"),
            )
        update_manifest(experiment_dir, status="completed")
    except Exception as error:
        # 普通异常写入清单后继续抛出；Ctrl+C 保持 running 以表示可恢复。
        update_manifest(experiment_dir, status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
