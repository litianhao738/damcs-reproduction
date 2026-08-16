# 模块说明：定义 AI 智能体和人工智能体，并串联感知、记忆、思考与行动。
from .memory_system import MemorySystem, Experience
# import enum
from enum import Enum
from .knowledge_graph import LongTermMemory, SemanticMemory, WorkingMemory
from .backbone_models.vision_model import VisionModel
from .llm_api import TextModel, get_completion
import torch
from .utils import generate_inquiry, safe_extract_op
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
# vision_model = VisionModel()
# vision_model.load_state_dict(torch.load("memory_system/backbone_models/vision_model.pth"))
# text_model = TextModel()

class ActionStatus(Enum):
    # 变量说明（IN_PROGRESS）：动作仍在执行、暂时不需要重新规划的枚举值。
    IN_PROGRESS = 0
    # 变量说明（ALMOST_DONE）：导航已到达目标、等待最后一次交互确认的枚举值。
    ALMOST_DONE = 1
    # 变量说明（DONE）：动作已经完成、智能体可以开始下一次规划的枚举值。
    DONE = 2
    # 变量说明（INTERRUPTED）：动作因其他智能体共享资源等事件被打断的枚举值。
    INTERRUPTED = 3
    
class Agent:
    def __init__(self, id, kg_update_freq=10, output_dir=None):
        # 变量说明（id）：当前智能体、玩家或记录对象的唯一编号。
        # 变量说明（kg_update_freq）：长期记忆原计划每隔多少条经验更新知识图谱的频率参数。
        # 变量说明（self.memory_system）：保存该智能体全部经验和近期动作历史的记忆容器。
        self.memory_system = MemorySystem(num_history_actions=7)
        # 变量说明（self.id）：当前智能体、玩家或记录对象的唯一编号。
        self.id = id
        # 每个 Agent 只向自己的实验子目录写入观察帧和知识图谱。
        self.output_dir = Path(output_dir or Path("results") / f"agent_{id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 变量说明（self.action_status）：高层动作当前所处的执行状态。
        self.action_status = ActionStatus.DONE
        # 变量说明（self.op）：LLM 选定的高层操作名称，例如导航、制造或共享。
        self.op = 'noop'
        # 变量说明（self.action）：当前智能体准备执行的环境动作。
        self.action = 'noop'
        # 变量说明（self.rss_to_collect）：导航操作要寻找并采集的目标资源名称。
        self.rss_to_collect = ''
        # 变量说明（self.rss_to_share）：共享操作准备转交给其他智能体的资源名称。
        self.rss_to_share = ''
        # 变量说明（self.summary）：对当前经验、目标或统计结果的精简文本摘要。
        self.summary = ""
        # 变量说明（self.target_agent_id）：资源共享操作的目标智能体编号；-1 表示无目标。
        self.target_agent_id = -1
        # 变量说明（self.kg_update_freq）：长期记忆原计划每隔多少条经验更新知识图谱的频率参数。
        self.kg_update_freq = kg_update_freq
        # 变量说明（self.long_term_mem）：负责语义规则和层次化知识图谱的长期记忆对象。
        self.long_term_mem = LongTermMemory(path=str(self.output_dir), recent_experience_num=kg_update_freq)
        # 变量说明（self.replay）：按时间步保存成就和背包快照、用于最终统计的回放记录。
        self.replay = []
        # 变量说明（self.text_model）：调用嵌入模型生成文本向量的包装器。
        self.text_model = TextModel()
        
        # 变量说明（self.wm_content）：将知识图谱摘要整理后注入提示词的工作记忆文本。
        self.wm_content = ""
        # 变量说明（self.think_context）：当前智能体最近一次调用大模型时使用的完整上下文。
        self.think_context = ""
        # global communication
        # 变量说明（self.table_avaliable）：世界中是否已经放置工作台的共享状态标记。
        self.table_avaliable = False
        # 变量说明（self.furnace_avaliable）：世界中是否已经放置熔炉的共享状态标记。
        self.furnace_avaliable = False
    
    def update_crafting_station_status(self, station_name):
        # 变量说明（station_name）：需要更新可用状态的合成站名称，例如 table 或 furnace。
        if station_name == 'table':
            self.table_avaliable = True
        elif station_name == 'furnace':
            self.furnace_avaliable = True
            
    def __len__(self):
        return len(self.memory_system)
    
    def update_action(self, action):
        # 变量说明（action）：当前智能体准备执行的环境动作。
        self.action = action
    
    def update_action_status(self, status):
        # 变量说明（status）：调用方准备写入对象的最新执行状态。
        self.action_status = status
    
    def update_current_skill(self, op, rss_to_collect, rss_to_share, target_agent_id):
        # 变量说明（op）：LLM 选定的高层操作名称，例如导航、制造或共享。
        # 变量说明（rss_to_collect）：导航操作要寻找并采集的目标资源名称。
        # 变量说明（rss_to_share）：共享操作准备转交给其他智能体的资源名称。
        # 变量说明（target_agent_id）：资源共享操作的目标智能体编号；-1 表示无目标。
        self.op = op
        self.rss_to_collect = rss_to_collect
        self.rss_to_share = rss_to_share
        self.target_agent_id = target_agent_id
    
    def update_state(self, obs, step, env, info, episode_number, episode_timestep):
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        # 变量说明（step）：当前经验或环境时间步编号。
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        # 变量说明（episode_number）：当前回合编号。
        # 变量说明（episode_timestep）：当前回合内的时间步编号。
        # 变量说明（self.obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        self.obs = obs
        # 变量说明（self.step）：当前经验或环境时间步编号。
        self.step = step
        # 变量说明（self.env）：当前 Crafter 仿真环境实例。
        self.env = env
        # 变量说明（self.info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        self.info = info
        self.replay.append({'step': step} | info['achievements'] | info['inventory'])
        # 变量说明（self.episode_number）：当前回合编号。
        self.episode_number = episode_number
        # 变量说明（self.episode_timestep）：当前回合内的时间步编号。
        self.episode_timestep = episode_timestep
    
    def create_experience(self):
        # pre experience
        # 变量说明（env）：当前 Crafter 仿真环境实例。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        # 变量说明（step）：当前经验或环境时间步编号。
        obs, step, env, info = self.obs, self.step, self.env, self.info
        # 变量说明（self.experience）：由当前观察、情景状态和动作历史组成的一条经验。
        self.experience = Experience(None, self.text_model, path=str(self.output_dir))
        self.experience.sensory_memory.set_memory(vision=obs, step=step, show=False)
        self.experience.procedural_memory.set_memory(self.memory_system.get_history_actions())
        self.experience.episodic_memory.set_memory(player=env._players[self.id],
                                            temporal_info=step,
                                            scene_semantic=info['semantic'],
                                            episode_number=self.episode_number,
                                            episode_timestep=self.episode_timestep)
        # 变量说明（context）：发送给大模型的当前决策上下文文本。
        context = self.experience.generate_context_from_episodic_and_procedural_memory()
        # 当前版本尚未启用基于向量的相似经验检索，因此关闭 Embedding API 调用。
        # 情景文本、程序记忆、知识图谱和工作记忆仍会照常参与 Agent 决策。
        # self.experience.generate_embedding()
        #relevant_experiences = self.memory_system.find_relevant_experiences(self.experience, k=0)
        # 变量说明（wm_content）：将知识图谱摘要整理后注入提示词的工作记忆文本。
        wm_content = ""
        self.wm_content = ""
        if len(self.memory_system) > 0:
            # 变量说明（long_term_mem）：负责语义规则和层次化知识图谱的长期记忆对象。
            long_term_mem = LongTermMemory(path=str(self.output_dir), recent_experience_num=len(self.memory_system))
            if self.table_avaliable:
                long_term_mem.semantic_memory.update_crafting_station_status("table")
            if self.furnace_avaliable:
                long_term_mem.semantic_memory.update_crafting_station_status("furnace")
                
            long_term_mem.update_memory(self.memory_system.experiences)
            long_term_mem.update_knowledge_graph()
            # 变量说明（working_memory）：从长期知识图谱即时提取决策信息的工作记忆对象。
            working_memory = WorkingMemory(long_term_mem.G)
            # 变量说明（summaries）：从多个目标节点提取出的工作记忆摘要映射。
            summaries = working_memory.generate_working_memory_summaries()
            long_term_mem.plot_fancy_knowledge_graph(summaries)
            
            # 变量说明（wm）：最近一个当前目标对应的工作记忆摘要。
            wm = summaries[list(summaries)[-1]]
            
            # 变量说明（craft_station_summary）：`create_experience` 中当前处理的摘要。
            craft_station_summary = (
                f"\n### Crafting Station Availability:\n"
                f"  - Table: {'placed alerady' if self.table_avaliable else 'still needs to be placed'}.\n"
                f"  - Furnace: {'placed alerady' if self.furnace_avaliable else 'still needs to be placed'}.\n"
            )
            
            # 变量说明（CG_preq）：语义记忆对当前目标前置条件的检查结果文本。
            # 变量说明（LTG_preq）：语义记忆对长期目标前置条件的检查结果文本。
            LTG_preq, CG_preq = long_term_mem.semantic_memory.check_goal(
                                            wm['long_term_goal'], 
                                            wm['current_goal'], 
                                            self.experience.episodic_memory.inventory, 
                                            {'facing': self.experience.episodic_memory.describe_facing_object()})
            # 变量说明（prerequisites_summary）：`create_experience` 中当前处理的摘要。
            prerequisites_summary = (
                f"\n### Prerequisites Check:\n"
                f"  - For long-term goal: {LTG_preq}\n"
                f"  - For immediate goal: {'should work on long-term goal now.' if ('Ready' in LTG_preq or 'everything is ready' in LTG_preq) else CG_preq}\n"
            )
            
            # 变量说明（efforts_summary）：`create_experience` 中当前处理的摘要。
            efforts_summary = (
                f"- To satisify the prerequiste, you made the following efforts recently towards {wm['current_goal']}: \n" 
                f"  *{'; in '.join(wm['efforts_so_far'])}*.\n" 
                f"- Past Accomplishments: You have completed the following tasks: {', '.join(wm['past_events'])}.\n"
            )
            
            # might be confusing. quick fix: self.wm_content only wants prerequisites_summary
            wm_content = craft_station_summary + prerequisites_summary + efforts_summary # past_summary + 
            self.wm_content = prerequisites_summary
            
        # 变量说明（previous_timestamps_summary）：`create_experience` 中当前处理的摘要。
        previous_timestamps_summary = self.memory_system.generate_previous_timestamps_summary(k=1)
        # 变量说明（self.context）：发送给大模型的当前决策上下文文本。
        self.context = previous_timestamps_summary + context + wm_content
        self.context = self.context.replace("next", "").replace("Next", "")
        return self.context

    def think(self, context):
        # 变量说明（context）：发送给大模型的当前决策上下文文本。
        self.think_context = context
        # 变量说明（msg）：准备发送给大模型或其他组件的消息内容。
        # 发送给多模态模型的图片与本 Agent 本步保存的观察严格对应。
        msg = generate_inquiry(vision=str(self.output_dir / f'{self.step}.png'),
                                     description=context)
        # 变量说明（self.thoughts）：大模型返回并通过结构校验的决策对象。
        self.thoughts = get_completion(msg)
        return self.thoughts
    

    def consolidate_experience(self, final_response):
        # 变量说明（final_response）：本次思考最终采用的结构化 LLM 输出。
        self.experience.episodic_memory.final_response = final_response#.summary
        self.summary = self.experience.episodic_memory.generate_summary()
        self.memory_system.add_experience(self.experience)
        # 变量说明（extract_history）：从 LLM 行动结果整理出的、准备写入程序记忆的动作记录。
        # 变量说明（op）：LLM 选定的高层操作名称，例如导航、制造或共享。
        # 变量说明（rss_to_collect）：导航操作要寻找并采集的目标资源名称。
        # 变量说明（rss_to_share）：共享操作准备转交给其他智能体的资源名称。
        # 变量说明（target_agent_id）：资源共享操作的目标智能体编号；-1 表示无目标。
        op, rss_to_collect, rss_to_share, target_agent_id, extract_history = safe_extract_op(final_response)
        if extract_history:
            self.memory_system.history_actions.append(extract_history)
        return op, rss_to_collect, rss_to_share, target_agent_id

    # @staticmethod
    def process_agent(agent, agents_contexts, info):
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（agents_contexts）：以智能体编号为键的个性化决策上下文字典。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        if agent.action_status != ActionStatus.IN_PROGRESS and not info[agent.id]['sleeping']:
            # Skip HumanAgent processing - they are removed from pipeline
            if hasattr(agent, 'is_human') and agent.is_human:
                # HumanAgent is disabled - return None thoughts
                return {"id": agent.id, "thoughts": None}
            else:
                # Regular AI agent processing
                return {"id": agent.id, "thoughts": agent.think(agents_contexts[agent.id])}
        return {"id": agent.id, "thoughts": None} # In case the agent does not think    

class HumanAgent(Agent):
    """
    A subclass of Agent that allows a human user to manually select operations and participate in communication.
    """
    def __init__(self, id, kg_update_freq=10, output_dir=None):
        # 变量说明（id）：当前智能体、玩家或记录对象的唯一编号。
        # 变量说明（kg_update_freq）：长期记忆原计划每隔多少条经验更新知识图谱的频率参数。
        super().__init__(id, kg_update_freq, output_dir=output_dir)
        # 变量说明（self.is_human）：表示 `is_human` 所描述条件是否成立的布尔标记。
        self.is_human = True

    def choose_action(self):
        print(f"\n[HumanAgent {self.id}] Please select your operation:")
        print("Available operations: 'Navigator', 'share', 'noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep'")
        print("Available resources: 'wood', 'stone', 'coal', 'iron', 'diamond', 'not_applicable'")
        
        # 变量说明（op）：LLM 选定的高层操作名称，例如导航、制造或共享。
        op = input("Enter operation type: ").strip()
        if not op:
            op = 'noop'
            
        # 变量说明（rss_to_collect）：导航操作要寻找并采集的目标资源名称。
        rss_to_collect = input("Enter resource to collect (or 'not_applicable'): ").strip()
        if not rss_to_collect:
            rss_to_collect = 'not_applicable'
            
        # 变量说明（rss_to_share）：共享操作准备转交给其他智能体的资源名称。
        rss_to_share = input("Enter resource to share (or 'not_applicable'): ").strip()
        if not rss_to_share:
            rss_to_share = 'not_applicable'
            
        try:
            # 变量说明（target_agent_id）：资源共享操作的目标智能体编号；-1 表示无目标。
            target_agent_id = int(input("Enter target agent id (or -1 if not applicable): "))
        except ValueError:
            target_agent_id = -1
            
        self.update_current_skill(op, rss_to_collect, rss_to_share, target_agent_id)
        print(f"[HumanAgent {self.id}] Action set: op={op}, collect={rss_to_collect}, share={rss_to_share}, target_agent_id={target_agent_id}")

    def communicate(self, message, agents):
        # 变量说明（message）：准备发送、显示或写入通信记忆的文本消息。
        # 变量说明（agents）：参与当前仿真的全部智能体对象列表。
        """
        Send a message to other agents (simple broadcast for demonstration).
        """
        print(f"[HumanAgent {self.id}] Sending message: {message}")
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        for agent in agents:
            if agent.id != self.id:
                if hasattr(agent, 'receive_message'):
                    agent.receive_message(message, self.id)

    def receive_message(self, message, from_id):
        # 变量说明（message）：准备发送、显示或写入通信记忆的文本消息。
        # 变量说明（from_id）：`receive_message` 中当前对象的唯一编号或序列索引。
        print(f"[HumanAgent {self.id}] Received message from Agent {from_id}: {message}")    
