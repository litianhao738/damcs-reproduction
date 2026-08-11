# 模块说明：实现语义规则、长期知识图谱和工作记忆摘要。
from memory_system.llm_api import get_completion
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyvis.network import Network
import os
import memory_system.utils as utils
from concurrent.futures import ThreadPoolExecutor
from memory_system.llm_api import get_completion
import memory_system.utils as utils
import networkx as nx
from typing import List, Dict, Any, Tuple
from memory_system.utils import print_color

class SemanticMemory:
    def __init__(self):
        # 变量说明（self.crafting_recipes）：目标名称到所需工作站、工具和材料的规则表。
        self.crafting_recipes = {
            "collect_cow": {"facing": "cow"},
            "collect_drink": {"facing": "water"},
            "collect_wood": {"facing": "tree"},
            "collect_stone": {"facing": "stone", "wood_pickaxe": 1},
            "collect_coal": {"facing": "coal", "wood_pickaxe": 1},
            "collect_iron": {"facing": "iron", "stone_pickaxe": 1},
            "collect_diamond": {"facing": "diamond", "iron_pickaxe": 1},
            "obtain_diamond": {"facing": "diamond", "iron_pickaxe": 1},
            "place_table": {"facing": "grass", "wood": 2},
            "place_furnace": {"facing": "grass", "stone": 4},
            "make_wood_pickaxe": {"facing": "table", "wood": 1},
            "make_stone_pickaxe": {"facing": "table", "stone": 1, "wood": 1},
            "make_iron_pickaxe": {"facing": "furnace", "iron": 1, "coal": 1, "wood": 1},
        }
        # 变量说明（self.placed_a_table）：语义记忆记录的工作台是否已放置标记。
        self.placed_a_table = False
        # 变量说明（self.placed_a_furnace）：语义记忆记录的熔炉是否已放置标记。
        self.placed_a_furnace = False
        
    def update_crafting_station_status(self, crafting_station_name):
        # 变量说明（crafting_station_name）：需要标记为已放置的合成站名称。
        if crafting_station_name == "table":
            self.placed_a_table = True
        elif crafting_station_name == "furnace":
            self.placed_a_furnace = True
        
    def _check_goal(self, goal, inventory):
        # Get the required materials for the given goal
        # 变量说明（goal）：`_check_goal` 中当前处理的目标。
        # 变量说明（inventory）：当前智能体背包内各类物品及数量。
        # 变量说明（required_materials）：完成当前目标所需物品及最小数量。
        required_materials = self.crafting_recipes.get(goal, None)
        
        if goal == 'help_agent':
            return f"Goal 'help_agent' requires you to help the agent immediately if you do have the material in your inventory."
        if goal == "share":
            return f"Goal 'share' will work if and only if you have the material in your inventory. Double check!"
        
        if goal == "make_wood_pickaxe":
            if inventory.get('wood_pickaxe', 0) > 0:
                return f"Wood pickaxe is already made."
        if goal == "make_stone_pickaxe":
            if inventory.get('stone_pickaxe', 0) > 0:
                return f"Stone pickaxe is already made."
        if goal == "make_iron_pickaxe":
            if inventory.get('iron_pickaxe', 0) > 0:
                return f"Iron pickaxe is already made."
        
        if required_materials is None:
            return f"Goal '{goal}' is not recognized."
        
        # 变量说明（missing_items）：当前背包或朝向尚未满足的前置条件。
        missing_items = {}
        # 变量说明（is_facing_the_only_missing_item）：表示 `is_facing_the_only_missing_item` 所描述条件是否成立的布尔标记。
        is_facing_the_only_missing_item = True
        # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
        # 变量说明（requirement）：当前目标对某一材料、工具或朝向的具体要求。
        for item, requirement in required_materials.items():
            if item == "facing" and requirement != inventory['facing']:
                missing_items[item] = f"everything is ready, should navigate to {requirement}; "
                if requirement == 'table' and not self.placed_a_table:
                    missing_items[item] += f" need to place a {requirement} first, and Missing {2 - inventory.get('wood', 0)} wood, and navigate to grass to place it;"
                elif requirement == 'furnace' and not self.placed_a_furnace:
                    missing_items[item] += f" need to place a {requirement} first, and Missing {4 - inventory.get('stone', 0)} stone, and navigate to grass to place it;"
                
            # Check if the item exists in the inventory and if there is enough of it
            
            elif inventory.get(item, 0) < requirement:
                is_facing_the_only_missing_item = False
                missing_items[item] = f"Missing {requirement - inventory.get(item, 0)} {item}; "
        
        if not missing_items:
            return f"*Ready* '{goal}'"
        else:
            if is_facing_the_only_missing_item:
                # 变量说明（missing）：当前配置、背包或目标前置条件中缺失的字段或物品。
                missing = " ".join(missing_items.values())
            else:
                if 'facing' in missing_items:
                    del missing_items['facing']
                missing = " ".join(missing_items.values())
            return f"*Cannot complete* '{goal}': {missing}"
    
    def check_goal(self, LTG, CG, inventory, facing):
        # 变量说明（LTG）：Long-Term Goal 的缩写，表示当前长期目标。
        # 变量说明（CG）：Current Goal 的缩写，表示智能体当前立即执行的目标。
        # 变量说明（inventory）：当前智能体背包内各类物品及数量。
        # 变量说明（facing）：玩家当前面向的方向或面前物体名称。
        if type(inventory) != dict:
            # 变量说明（inventory_items）：从背包描述文本切分出的“物品名: 数量”字段列表。
            inventory_items = (inventory.split(', '))[1:]
            # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
            inventory = {
                item.split(': ')[0]: int(item.split(': ')[1])
                for item in inventory_items if int(item.split(': ')[1]) > 0
            }
        inventory |= facing # merge dict
        return self._check_goal(LTG, inventory), self._check_goal(CG, inventory)

class LongTermMemory:
    def __init__(self, path="", recent_experience_num=10):
        # 变量说明（path）：当前读取或写入对象对应的文件系统路径。
        # 变量说明（recent_experience_num）：每次构建知识图谱时最多纳入的最近经验条数。
        # 变量说明（self.semantic_memory）：保存合成配方与目标前置条件的规则型语义记忆。
        self.semantic_memory = SemanticMemory()
        # 变量说明（self.recent_episoidic_buffer）：预留的近期情景记忆缓冲区；当前建图流程会先清空。
        self.recent_episoidic_buffer = []
        # 变量说明（self.recent_experience_num）：每次构建知识图谱时最多纳入的最近经验条数。
        self.recent_experience_num = recent_experience_num
        # 变量说明（self.G）：以长期目标、当前目标和步骤为节点的 NetworkX 有向知识图谱。
        self.G = None
        # 变量说明（self.current_goal）：智能体目前正在直接推进的短期目标。
        self.current_goal = None
        # 变量说明（self.current_task）：长期记忆当前跟踪的任务；该字段目前为预留状态。
        self.current_task = None
        # 变量说明（self.event_nodes）：知识图谱中记录环境事件的节点列表；当前流程尚未填充。
        self.event_nodes = []
        # 变量说明（self.event_counter）：为经验事件生成唯一标识时使用的递增计数器。
        self.event_counter = 1
        # 变量说明（self.kg_count）：当前长期记忆已经生成知识图谱的次数。
        self.kg_count = 0 # knowledge graph counter
        # 变量说明（self.path）：当前读取或写入对象对应的文件系统路径。
        self.path = path
        # check if save path exists and create if not
        if self.path != "":
            if not os.path.exists(self.path):
                os.makedirs(self.path)
    
    def update_memory(self, experiences):
        # 变量说明（experiences）：该智能体按时间顺序积累的经验列表。
        # 变量说明（self.experiences）：该智能体按时间顺序积累的经验列表。
        self.experiences = experiences
    
    def update_knowledge_graph(self):
        self.kg_count += 1
        # get tqdm progress for each step
        self.recent_episoidic_buffer = []
        # exececute functions in a list in order
        # 变量说明（functions）：需要按顺序调用的知识图谱更新函数列表。
        functions = [
            self.extract_goal_and_relationships, 
            self.generate_knowledge_graph, 
            # self.generate_event_level_summaries, 
            # self.generate_community_level_summaries, 
            # self.plot_fancy_knowledge_graph
            ]
        
        # 变量说明（func）：当前准备调用的处理函数对象。
        for func in functions:
            func()
    
    def extract_goal_and_relationships(self):
        # 变量说明（self.processed_summaries）：从经验中抽取、已标准化为图节点字段的步骤数据列表。
        self.processed_summaries = []
        # 变量说明（experience）：由当前观察、情景状态和动作历史组成的一条经验。
        for experience in self.experiences[-self.recent_experience_num:]:
            # 变量说明（final_response）：本次思考最终采用的结构化 LLM 输出。
            final_response = experience.episodic_memory.final_response
            # 变量说明（action）：当前智能体准备执行的环境动作。
            # 变量说明（goal）：`extract_goal_and_relationships` 中当前处理的目标。
            # 变量说明（reflection）：LLM 对上一动作、结果和重复行为的结构化反思字段。
            reflection, goal, action = final_response.reflection, final_response.goal, final_response.action
            # Extract necessary information
            # 变量说明（step_data）：从单条经验提取、准备写入步骤节点的全部结构化字段。
            step_data = {
                'episode': final_response.epsiode_number,
                'timestep': final_response.timestep,
                'inventory': experience.episodic_memory.inventory,
                'past_events': final_response.past_events,
                'summary': final_response.summary,
                'current_facing_direction': final_response.current_facing_direction,
                # 'vision_matrix': experience.sensory_memory.vision,
                # 'vision': reflection.vision,
                'last_action': reflection.last_action,
                'last_action_result': reflection.last_action_result,
                'long_term_goal': goal.long_term_goal,
                'long_term_goal_progress': goal.long_term_goal_progress,
                'current_goal_reason': goal.current_goal_reason,
                'current_goal': goal.current_goal.value,
                'current_goal_status': goal.current_goal_status,
                'action': action.final_next_action.value,
                'action_destination': action.final_target_material_to_collect.value,
                'action_share_destination': action.final_target_material_to_share.value,
                'action_target_agent_id': action.final_target_agent_id,
                'action_reason': action.next_action_reason,
            }
            self.processed_summaries.append(step_data.copy())
            self.event_counter += 1
        return self.processed_summaries
    
    def generate_knowledge_graph(self):
        if self.G is None:
            self.G = nx.DiGraph()
            self.event_counter = 1
            # Initialize previous variables
            # 变量说明（self.previous_step_node）：知识图谱中上一时间步节点的名称，用于连接连续步骤。
            self.previous_step_node = None
            # 变量说明（self.previous_goal_node）：知识图谱中上一个当前目标节点的名称。
            self.previous_goal_node = None
            # 变量说明（self.previous_goal）：上一条经验声明的当前目标，用于检测目标变化。
            self.previous_goal = None
            # 变量说明（self.previous_long_term_goal_node）：知识图谱中上一个长期目标节点的名称。
            self.previous_long_term_goal_node = None
            # 变量说明（self.previous_long_term_goal）：上一条经验声明的长期目标，用于检测长期规划变化。
            self.previous_long_term_goal = None
            # Counters for unique identification
            # 变量说明（self.goal_instance_counter）：当前回合内为短期目标节点生成唯一名称的计数器。
            self.goal_instance_counter = 0
            # 变量说明（self.long_term_goal_instance_counter）：当前回合内为长期目标节点生成唯一名称的计数器。
            self.long_term_goal_instance_counter = 0
            # 变量说明（self.previous_inventory）：上一条经验的背包快照，用于判断状态是否变化。
            self.previous_inventory = None
            # 变量说明（self.previous_episode）：上一条建图经验所属的回合编号，用于识别回合切换。
            self.previous_episode = None
            # Initialize current long-term goal node
            # 变量说明（self.current_long_term_goal_node）：当前步骤应当归属的长期目标节点名称。
            self.current_long_term_goal_node = None

        # 变量说明（entry）：准备写入知识图谱的单步经验字段集合。
        for entry in self.processed_summaries:
            # 变量说明（episode）：当前回合编号或回合相关数据。
            episode = entry['episode']
            if episode != self.previous_episode:
                self.previous_step_node = None
                self.previous_goal_node = None
                self.previous_goal = None
                self.previous_long_term_goal_node = None
                self.previous_long_term_goal = None
                self.previous_inventory = None
                # Update the previous_episode
                self.previous_episode = episode
                # Reset goal instance counters for the new episode
                self.goal_instance_counter = 0
                self.long_term_goal_instance_counter = 0
                # Reset current long-term goal node
                self.current_long_term_goal_node = None

            # 变量说明（step_num）：`generate_knowledge_graph` 中对应对象的数量或累计次数。
            step_num = entry['timestep']
            # 变量说明（step_node）：`generate_knowledge_graph` 中当前处理的时间步、节点。
            step_node = f"Episode {episode} Step {step_num}"
            # Normalize strings
            # 变量说明（long_term_goal）：`generate_knowledge_graph` 中当前处理的目标。
            long_term_goal = entry['long_term_goal'].strip().lower().rstrip('.')
            # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
            current_goal = entry['current_goal'].strip().lower().rstrip('.')

            # Process the current inventory
            # 变量说明（inventory_items）：从背包描述文本切分出的“物品名: 数量”字段列表。
            inventory_items = (entry['inventory'].split(', '))[1:]
            # 变量说明（current_inventory）：从经验文本解析出的当前非空背包物品映射。
            # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
            current_inventory = {
                item.split(': ')[0]: int(item.split(': ')[1])
                for item in inventory_items if int(item.split(': ')[1]) > 0
            }
            # Perform the semantic memory check
            # LTG and CG
            # 变量说明（semantic_memory_check_results）：`generate_knowledge_graph` 中待批量处理的记忆、结果集合。
            semantic_memory_check_results = self.semantic_memory.check_goal(
                long_term_goal,
                current_goal,
                current_inventory,
                {'facing': entry['current_facing_direction']}
            )
            # Add the semantic memory check to the entry data
            entry['semantic_memory_check_LTG'], entry['semantic_memory_check_CG'] = semantic_memory_check_results

            # Check if the long-term goal has changed
            if long_term_goal != self.previous_long_term_goal:
                # Create a new long-term goal node
                self.long_term_goal_instance_counter += 1
                # 变量说明（long_term_goal_node）：`generate_knowledge_graph` 中当前处理的目标、节点。
                long_term_goal_node = f"Episode {episode} Long Term Goal {self.long_term_goal_instance_counter}: {long_term_goal}"
                self.G.add_node(long_term_goal_node, type='long_term_goal')
                # Connect previous long-term goal node to current long-term goal node
                if self.previous_long_term_goal_node:
                    self.G.add_edge(self.previous_long_term_goal_node, long_term_goal_node)
                # Update current long-term goal node
                self.current_long_term_goal_node = long_term_goal_node
            else:
                # Use the same long-term goal node
                long_term_goal_node = self.current_long_term_goal_node

            # Check if the current goal has changed
            if self.previous_goal != current_goal or current_inventory != self.previous_inventory:
                # Increment goal instance counter
                self.goal_instance_counter += 1

                # Create a unique goal node label
                # 变量说明（current_goal_node）：`generate_knowledge_graph` 中当前处理的目标、节点。
                current_goal_node = f"Episode {episode} Goal {self.goal_instance_counter}: {current_goal}"
                self.G.add_node(current_goal_node, type='current_goal')
                # Connect current goal node to current long-term goal node
                self.G.add_edge(long_term_goal_node, current_goal_node)
                # Connect previous goal node to current goal node
                if self.previous_goal_node:
                    self.G.add_edge(self.previous_goal_node, current_goal_node)
            else:
                # If the goal hasn't changed, use the same goal node
                current_goal_node = self.previous_goal_node

            # Create step node with semantic memory check included
            self.G.add_node(step_node, type='step', **entry)
            # Connect step node to current goal node
            self.G.add_edge(current_goal_node, step_node)

            # Connect steps sequentially if under the same current goal
            if self.previous_step_node and self.previous_goal == current_goal and current_inventory == self.previous_inventory:
                self.G.add_edge(self.previous_step_node, step_node)

            # Update previous variables
            self.previous_step_node = step_node
            self.previous_goal_node = current_goal_node
            self.previous_goal = current_goal
            self.previous_long_term_goal_node = long_term_goal_node
            self.previous_long_term_goal = long_term_goal
            self.previous_inventory = current_inventory

        return self.G

    #
    def plot_fancy_knowledge_graph(self, summaries=None):
        # 变量说明（summaries）：从多个目标节点提取出的工作记忆摘要映射。
        # 变量说明（goal_node）：`plot_fancy_knowledge_graph` 中当前处理的目标、节点。
        # 变量说明（summary）：对当前经验、目标或统计结果的精简文本摘要。
        for goal_node, summary in summaries.items():
            # Update the node data with the working memory summary
            if self.G.has_node(goal_node):
                self.G.nodes[goal_node]['working_memory'] = summary

        # 变量说明（G_loaded）：已经构建完成、准备交给 PyVis 渲染的知识图谱。
        G_loaded = self.G
        if G_loaded is None or len(G_loaded) == 0:
            print("Knowledge graph is empty. Cannot plot.")
            return


        # Initialize pyvis network for visualization
        # 变量说明（net）：用于生成可交互 HTML 的 PyVis Network 对象。
        net = Network(notebook=True, cdn_resources='remote', height='800px', width='100%')

        # Add all nodes with appropriate styles and tooltips
        # 变量说明（data）：与当前节点、文件或处理步骤关联的结构化数据。
        # 变量说明（node）：当前遍历或处理的知识图谱节点标识。
        for node, data in G_loaded.nodes(data=True):
            # 变量说明（node_type）：`plot_fancy_knowledge_graph` 中当前处理的节点。
            node_type = data.get('type', 'unknown')
            if node_type == 'step':
                # Existing code for step nodes remains the same
                # 变量说明（long_term_goal）：`plot_fancy_knowledge_graph` 中当前处理的目标。
                long_term_goal = data.get('long_term_goal', '')
                # 变量说明（long_term_goal_progress）：`plot_fancy_knowledge_graph` 中待批量处理的目标集合。
                long_term_goal_progress = data.get('long_term_goal_progress', '')
                # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
                current_goal = data.get('current_goal', '')
                # 变量说明（action）：当前智能体准备执行的环境动作。
                action = data.get('action', '')
                # 变量说明（action_destination）：`plot_fancy_knowledge_graph` 中当前处理的动作。
                action_destination = data.get('action_destination', '')
                # 变量说明（action_share_destination）：`plot_fancy_knowledge_graph` 中当前处理的动作。
                action_share_destination = data.get('action_share_destination', '')
                # 变量说明（inventory）：当前智能体背包内各类物品及数量。
                inventory = data.get('inventory', '')
                # 变量说明（current_facing_direction）：`plot_fancy_knowledge_graph` 中当前处理的方向。
                current_facing_direction = data.get('current_facing_direction', '')
                # Process inventory for display
                # 变量说明（inventory_items）：从背包描述文本切分出的“物品名: 数量”字段列表。
                inventory_items = (inventory.split(', '))[1:]
                # 变量说明（inventory_dict）：`plot_fancy_knowledge_graph` 中待批量处理的背包集合。
                # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
                inventory_dict = {
                    item.split(': ')[0]: int(item.split(': ')[1])
                    for item in inventory_items if int(item.split(': ')[1]) > 0
                }
                # 变量说明（action_reason）：`plot_fancy_knowledge_graph` 中当前处理的动作。
                action_reason = data.get('action_reason', '')
                # Retrieve semantic memory check
                # 变量说明（semantic_memory_check_LTG）：`plot_fancy_knowledge_graph` 中当前处理的记忆。
                semantic_memory_check_LTG = data.get('semantic_memory_check_LTG', '')
                # 变量说明（semantic_memory_check_CG）：`plot_fancy_knowledge_graph` 中当前处理的记忆。
                semantic_memory_check_CG = data.get('semantic_memory_check_CG', '')
                # Create tooltip
                # 变量说明（title）：当前窗口、图表或输出区域显示的标题文本。
                title = (
                    f"\nLong-Term Goal: {long_term_goal}"
                    f"\nLTG Progress: {long_term_goal_progress}"
                    f"\nCurrent Goal: {current_goal}"
                    f"\nAction: {action}"
                    f"\nDestination: {action_destination}"
                    f"\nShare Destination: {action_share_destination}"
                    f"\nReason: {action_reason}"
                    f"\nInventory: {inventory_dict}"
                    f"\nSemantic Memory Check LTG: {semantic_memory_check_LTG}"
                    f"\nSemantic Memory Check CG: {semantic_memory_check_CG}"
                )
                net.add_node(
                    node,
                    label=node,
                    color='skyblue',
                    title=title,
                    shape='box',
                    size=15
                )
            elif node_type == 'current_goal':
                # Extract data from the current goal node
                # 变量说明（current_goal_description）：`plot_fancy_knowledge_graph` 中当前处理的目标。
                current_goal_description = node.split(': ', 1)[-1]
                # Retrieve the working memory summary
                # 变量说明（working_memory）：从长期知识图谱即时提取决策信息的工作记忆对象。
                working_memory = data.get('working_memory', {})
                # Build the tooltip using the working memory summary
                title = f"Current Goal: {current_goal_description}"
                if working_memory:
                    # Include working memory details in the tooltip
                    title += (
                        f"\nReason: {working_memory.get('current_goal_reason', '')}"
                        f"\n### Long-Term Goal: {working_memory.get('long_term_goal', '')}"
                        f"\n### LTG Progress: {working_memory.get('long_term_goal_progress', '')}"
                        f"\n### Current Progress and Efforts So Far:"
                    )
                    # Include efforts
                    # 变量说明（efforts）：当前目标下已经尝试过的动作文本列表。
                    efforts = working_memory.get('efforts_so_far', [])
                    # 变量说明（effort）：知识图谱中某个时间步采取行动的简短文本描述。
                    for effort in efforts:
                        title += f"\n- {effort}"
                    # Include past events
                    # 变量说明（past_events）：`plot_fancy_knowledge_graph` 中待批量处理的事件集合。
                    past_events = working_memory.get('past_events', [])
                    if past_events:
                        title += f"\n### Completed Events:"
                        # 变量说明（event）：`plot_fancy_knowledge_graph` 中当前处理的事件。
                        for event in past_events:
                            title += f"\n- {event}"
                    # Include the semantic memory check
                    title += f"\n### Semantic Memory Check:\n{working_memory.get('semantic_memory_check', '')}"
                else:
                    # Fallback if no working memory is available
                    # 变量说明（current_goal_reason）：`plot_fancy_knowledge_graph` 中当前处理的目标。
                    current_goal_reason = data.get('current_goal_reason', '')
                    title += f"Reason: {current_goal_reason}"
                net.add_node(
                    node,
                    label=node,
                    color='lightgreen',
                    title=title,
                    shape='ellipse',
                    size=15
                )
            elif node_type == 'long_term_goal':
                # Existing code for long-term goal nodes remains the same
                # 变量说明（long_term_goal_description）：`plot_fancy_knowledge_graph` 中当前处理的目标。
                long_term_goal_description = node.split(': ', 1)[-1]
                long_term_goal_progress = data.get('long_term_goal_progress', '')
                # 变量说明（long_term_goal_status）：`plot_fancy_knowledge_graph` 中当前对象或操作所处的状态。
                long_term_goal_status = data.get('long_term_goal_status', '')
                # Create tooltip
                title = (
                    f"<b>Long-Term Goal:</b> {long_term_goal_description}<br>"
                    f"<b>Progress:</b> {long_term_goal_progress}<br>"
                    f"<b>Status:</b> {long_term_goal_status}"
                )
                net.add_node(
                    node,
                    label=node,
                    color='lightcoral',
                    title=title,
                    shape='ellipse',
                    size=20
                )
            else:
                # For any other node types
                net.add_node(
                    node,
                    label=node,
                    title='',
                    shape='ellipse',
                    size=15
                )

        # Add all edges directly from the graph
        # 变量说明（u）：当前图边的起点节点，或数组计算中的第一个临时值。
        # 变量说明（v）：当前图边的终点节点，或数组计算中的第二个临时值。
        for u, v in G_loaded.edges():
            net.add_edge(u, v)

        # Configure physics for a smoother layout
        net.set_options("""
        var options = {
        "nodes": {
            "borderWidth": 2,
            "shadow": {
            "enabled": true
            },
            "font": {
            "size": 12,
            "face": "Arial"
            }
        },
        "edges": {
            "color": {
            "inherit": true
            },
            "smooth": {
            "type": "continuous"
            },
            "arrows": {
            "to": {
                "enabled": true,
                "scaleFactor": 0.5
            }
            }
        },
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -20000,
            "springLength": 150
            },
            "minVelocity": 0.75
        }
        }
        """)

        # Show or save the interactive network
        if self.path != "":
            net.save_graph(f"{self.path}/knowledge_graph_{self.kg_count}.html")
            print_color(f"Knowledge graph saved to {self.path}/knowledge_graph_{self.kg_count}.html", color="yellow")
        else:
            net.show("fancy_knowledge_graph.html")


# Define the WorkingMemory class
class WorkingMemory:
    def __init__(self, knowledge_graph: nx.DiGraph):
        # 变量说明（knowledge_graph）：`__init__` 中当前处理的图结构。
        self.G = knowledge_graph

    def retrieve_working_memory(self):
        # get most recent goal node
        # 变量说明（data）：与当前节点、文件或处理步骤关联的结构化数据。
        # 变量说明（node）：当前遍历或处理的知识图谱节点标识。
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'current_goal':
                # 变量说明（current_goal_node）：`retrieve_working_memory` 中当前处理的目标、节点。
                current_goal_node = node
                break
        # get working memory summary
        # 变量说明（working_memory_summary）：`retrieve_working_memory` 中当前处理的记忆、摘要。
        working_memory_summary = self.generate_working_memory_summaries()
        return working_memory_summary
    
    def generate_working_memory_summaries(self) -> Dict[str, Any]:
        """
        Generates a working memory summary for each current goal node in the knowledge graph.
        Returns a dictionary where keys are current goal node identifiers and values are the working memory summaries.
        """
        # 变量说明（working_memory_summaries）：`generate_working_memory_summaries` 中待批量处理的记忆集合。
        working_memory_summaries = {}
        # Iterate over all current goal nodes in the knowledge graph
        # 变量说明（data）：与当前节点、文件或处理步骤关联的结构化数据。
        # 变量说明（node）：当前遍历或处理的知识图谱节点标识。
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'current_goal':
                # The current goal node
                # 变量说明（current_goal_node）：`generate_working_memory_summaries` 中当前处理的目标、节点。
                current_goal_node = node

                # Extract the episode number from the current goal node
                # 变量说明（episode）：当前回合编号或回合相关数据。
                episode = self._extract_episode_number(current_goal_node)

                # Extract the long-term goal and related information
                # 变量说明（long_term_goal_info）：`generate_working_memory_summaries` 中当前处理的目标、状态信息。
                long_term_goal_info = self._get_long_term_goal_info(current_goal_node)

                # Get efforts (summary of steps) done so far under this current goal
                # 变量说明（efforts）：当前目标下已经尝试过的动作文本列表。
                efforts = self._get_efforts_under_goal(current_goal_node)

                # Get past events that have occurred before this current goal
                # 变量说明（past_events）：`generate_working_memory_summaries` 中待批量处理的事件集合。
                past_events = self._get_past_events(current_goal_node, episode)

                # Extract current goal description from node label if not in data
                # 变量说明（current_goal_description）：`generate_working_memory_summaries` 中当前处理的目标。
                current_goal_description = data.get('current_goal_description', '')
                if not current_goal_description:
                    # Assume node label is like "Episode X Goal Y: goal_description"
                    if ': ' in current_goal_node:
                        current_goal_description = current_goal_node.split(': ', 1)[-1]
                    else:
                        current_goal_description = current_goal_node

                # Get semantic memory check from the most recent step
                # 变量说明（steps）：`generate_working_memory_summaries` 中待批量处理的时间步集合。
                steps = self._get_steps_under_goal(current_goal_node)
                if steps:
                    # 变量说明（most_recent_step）：`generate_working_memory_summaries` 中当前处理的时间步。
                    most_recent_step = steps[-1]
                    # 变量说明（most_recent_step_data）：`generate_working_memory_summaries` 中当前处理的时间步。
                    most_recent_step_data = self.G.nodes[most_recent_step]
                    # 变量说明（semantic_memory_check_LTG）：`generate_working_memory_summaries` 中当前处理的记忆。
                    semantic_memory_check_LTG = most_recent_step_data.get('semantic_memory_check_LTG', '')
                    # 变量说明（semantic_memory_check_CG）：`generate_working_memory_summaries` 中当前处理的记忆。
                    semantic_memory_check_CG = most_recent_step_data.get('semantic_memory_check_CG', '')
                else:
                    semantic_memory_check_LTG = ''
                    semantic_memory_check_CG = ''

                # Construct the working memory summary
                # 变量说明（working_memory_summary）：`generate_working_memory_summaries` 中当前处理的记忆、摘要。
                working_memory_summary = {
                    'long_term_goal': long_term_goal_info['long_term_goal'],
                    'long_term_goal_progress': long_term_goal_info['long_term_goal_progress'],
                    'long_term_goal_reason': long_term_goal_info['long_term_goal_reason'],
                    'current_goal': current_goal_description,
                    'current_goal_reason': long_term_goal_info['current_goal_reason'],
                    'efforts_so_far': efforts,
                    'past_events': past_events,
                    'semantic_memory_check_LTG': semantic_memory_check_LTG,  # Include semantic memory check
                    'semantic_memory_check_CG': semantic_memory_check_CG,  # Include semantic memory check
                }

                # Add the summary to the dictionary
                working_memory_summaries[current_goal_node] = working_memory_summary

        return working_memory_summaries

    def _extract_episode_number(self, node_label: str) -> int:
        # 变量说明（node_label）：`_extract_episode_number` 中当前处理的节点、标签。
        """
        Extracts the episode number from a node label.
        Assumes node labels are in the format "Episode X ..."
        """
        try:
            if node_label.startswith('Episode '):
                # 变量说明（parts）：当前名称、路径或文本按分隔符拆分得到的片段列表。
                parts = node_label.split(' ', 2)
                # 变量说明（episode_number）：当前回合编号。
                episode_number = int(parts[1])
                return episode_number
        except (IndexError, ValueError):
            pass
        # Default to episode 1 if unable to extract
        return 1

    def _get_long_term_goal_info(self, current_goal_node: str) -> Dict[str, str]:
        # 变量说明（current_goal_node）：`_get_long_term_goal_info` 中当前处理的目标、节点。
        """
        Retrieves the long-term goal and related information associated with the current goal node.
        """
        # 变量说明（long_term_goal_info）：`_get_long_term_goal_info` 中当前处理的目标、状态信息。
        long_term_goal_info = {
            'long_term_goal': '',
            'long_term_goal_progress': '',
            'long_term_goal_reason': '',
            'current_goal_reason': ''
        }

        # Find the long-term goal node connected to this current goal node
        # 变量说明（pred）：当前知识图谱节点的直接前驱节点。
        for pred in self.G.predecessors(current_goal_node):
            # 变量说明（pred_data）：当前前驱节点保存的属性字典。
            pred_data = self.G.nodes[pred]
            if pred_data.get('type') == 'long_term_goal':
                # Extract long-term goal description from node label
                if ': ' in pred:
                    # 变量说明（long_term_goal_description）：`_get_long_term_goal_info` 中当前处理的目标。
                    long_term_goal_description = pred.split(': ', 1)[-1]
                else:
                    long_term_goal_description = pred
                long_term_goal_info['long_term_goal'] = long_term_goal_description
                # Since we might not have 'long_term_goal_reason' directly, we'll extract it from the steps
                break

        # Get the first step under this current goal to extract additional information
        # 变量说明（steps）：`_get_long_term_goal_info` 中待批量处理的时间步集合。
        steps = self._get_steps_under_goal(current_goal_node)
        if steps:
            # 变量说明（first_step_data）：`_get_long_term_goal_info` 中当前处理的时间步。
            first_step_data = self.G.nodes[steps[0]]
            long_term_goal_info['long_term_goal_progress'] = first_step_data.get('long_term_goal_progress', '')
            long_term_goal_info['current_goal_reason'] = first_step_data.get('current_goal_reason', '')
            # Extract 'long_term_goal_reason' from the step data if available
            long_term_goal_info['long_term_goal_reason'] = first_step_data.get('long_term_goal_reason', '')

        return long_term_goal_info

    def _get_efforts_under_goal(self, current_goal_node: str) -> List[str]:
        # 变量说明（current_goal_node）：`_get_efforts_under_goal` 中当前处理的目标、节点。
        """
        Collects a summary of steps (efforts) done so far under the given current goal node.
        """
        # 变量说明（efforts）：当前目标下已经尝试过的动作文本列表。
        efforts = []
        # 变量说明（steps）：`_get_efforts_under_goal` 中待批量处理的时间步集合。
        steps = self._get_steps_under_goal(current_goal_node)
        # 变量说明（step）：当前经验或环境时间步编号。
        for step in steps:
            # 变量说明（step_data）：从单条经验提取、准备写入步骤节点的全部结构化字段。
            step_data = self.G.nodes[step]
            # 变量说明（action）：当前智能体准备执行的环境动作。
            action = step_data.get('action', '')
            # 变量说明（action_destination）：`_get_efforts_under_goal` 中当前处理的动作。
            action_destination = step_data.get('action_destination', '')
            # 变量说明（action_share_destination）：`_get_efforts_under_goal` 中当前处理的动作。
            action_share_destination = step_data.get('action_share_destination', '')
            # 变量说明（last_action_result）：`_get_efforts_under_goal` 中当前处理的动作、结果。
            last_action_result = step_data.get('last_action_result', '')
            # 变量说明（timestep）：当前经验在所属回合内的时间步编号。
            timestep = step_data.get('timestep', '')
            # Construct a summary of the step
            if action == 'share':
                # 变量说明（effort_summary）：`_get_efforts_under_goal` 中当前处理的摘要。
                effort_summary = (
                    f"In step {timestep}: {action} {action_share_destination} to target agent. "
                )
            elif action == 'Navigator':
                effort_summary = (
                    f"In step {timestep}: {action} to {action_destination}. "
                )
            else:
                effort_summary = (
                    f"In step {timestep}: {action}. "
                )
            efforts.append(effort_summary)
        return efforts

    def _get_steps_under_goal(self, current_goal_node: str) -> List[str]:
        # 变量说明（current_goal_node）：`_get_steps_under_goal` 中当前处理的目标、节点。
        """
        Retrieves all step nodes under the given current goal node, sorted by timestep.
        """
        # 变量说明（steps）：`_get_steps_under_goal` 中待批量处理的时间步集合。
        steps = []
        # 变量说明（succ）：当前知识图谱节点的直接后继节点。
        for succ in self.G.successors(current_goal_node):
            # 变量说明（succ_data）：当前后继节点保存的属性字典。
            succ_data = self.G.nodes[succ]
            if succ_data.get('type') == 'step':
                steps.append(succ)
        # Sort the steps by timestep
        # 变量说明（x）：世界网格或图像中的横向坐标。
        steps = sorted(steps, key=lambda x: self.G.nodes[x]['timestep'])
        return steps

    def _get_past_events(self, current_goal_node: str, episode: int) -> List[str]:
        # 变量说明（current_goal_node）：`_get_past_events` 中当前处理的目标、节点。
        # 变量说明（episode）：当前回合编号或回合相关数据。
        """
        Retrieves past events (past goal descriptions) that occurred before the current goal,
        only if they belong to the same episode.
        """
        # 变量说明（past_events）：`_get_past_events` 中待批量处理的事件集合。
        past_events = []
        # Collect past goal descriptions
        # 变量说明（visited_goals）：`_get_past_events` 中待批量处理的目标集合。
        visited_goals = set()
        self._collect_past_goal_descriptions(current_goal_node, past_events, visited_goals, episode)
        # Reverse to get the correct chronological order
        past_events = past_events[::-1]
        return past_events

    def _collect_past_goal_descriptions(self, goal_node: str, past_events: List[str], visited_goals: set, episode: int):
        # 变量说明（goal_node）：`_collect_past_goal_descriptions` 中当前处理的目标、节点。
        # 变量说明（past_events）：`_collect_past_goal_descriptions` 中待批量处理的事件集合。
        # 变量说明（visited_goals）：`_collect_past_goal_descriptions` 中待批量处理的目标集合。
        # 变量说明（episode）：当前回合编号或回合相关数据。
        """
        Recursively collects past goal descriptions, only if they belong to the same episode.
        """
        if goal_node in visited_goals:
            return
        visited_goals.add(goal_node)
        # 变量说明（pred）：当前知识图谱节点的直接前驱节点。
        for pred in self.G.predecessors(goal_node):
            # 变量说明（pred_data）：当前前驱节点保存的属性字典。
            pred_data = self.G.nodes[pred]
            if pred_data.get('type') == 'current_goal':
                # Extract the episode number of the predecessor goal node
                # 变量说明（pred_episode）：`_collect_past_goal_descriptions` 中当前处理的回合。
                pred_episode = self._extract_episode_number(pred)
                if pred_episode == episode:
                    # Extract the goal description
                    if ': ' in pred:
                        # 变量说明（goal_description）：`_collect_past_goal_descriptions` 中当前处理的目标。
                        goal_description = pred.split(': ', 1)[-1]
                    else:
                        goal_description = pred
                    past_events.append(goal_description)
                    # Recursively collect past goals
                    self._collect_past_goal_descriptions(pred, past_events, visited_goals, episode)
            elif pred_data.get('type') == 'long_term_goal':
                # Continue traversing if needed
                continue