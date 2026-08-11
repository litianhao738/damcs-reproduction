# 模块说明：实现感知记忆、情景记忆和程序记忆三类记忆模块。
import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_player_vitals_and_inventory
from .constants import WORLD_SIZE, world_ids_item
# from .llm_api import get_completion
import os

class SensoryMemory():
    def __init__(self, vision_model=None, text_model=None, path=None):
        # 变量说明（vision_model）：把观察图像编码为特征的视觉模型；当前实现中为预留接口。
        # 变量说明（text_model）：调用嵌入模型生成文本向量的包装器。
        # 变量说明（path）：当前读取或写入对象对应的文件系统路径。
        # 变量说明（self.vision_model）：把观察图像编码为特征的视觉模型；当前实现中为预留接口。
        self.vision_model = vision_model
        # 变量说明（self.text_model）：调用嵌入模型生成文本向量的包装器。
        self.text_model = text_model
        # 变量说明（self.obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        self.obs = None
        # 变量说明（self.vision）：从完整观察中切出的当前游戏画面像素数组。
        self.vision = None
        # 变量说明（self.stats_bar）：从观察底部切出的生命状态与背包栏图像。
        self.stats_bar = None
        # 变量说明（self.vision_embedding）：`__init__` 中对应文本或图像的特征向量。
        self.vision_embedding = None
        # 变量说明（self.vision_reconstruction）：视觉自编码器根据潜在特征还原出的图像。
        self.vision_reconstruction = None
        # 变量说明（self.conversation）：感知记忆预留的智能体通信文本；当前流程尚未写入。
        self.conversation = None # WAIT FOR MULTIAGENT
        # 变量说明（self.conversation_embedding）：`__init__` 中对应文本或图像的特征向量。
        self.conversation_embedding = None # WAIT FOR MULTIAGENT
        # 变量说明（self.path）：当前读取或写入对象对应的文件系统路径。
        self.path = path
        # if path does not exist, create it
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        
        
    def set_memory(self, vision, step, show=False):
        # 变量说明（vision）：从完整观察中切出的当前游戏画面像素数组。
        # 变量说明（step）：当前经验或环境时间步编号。
        # 变量说明（show）：是否立即显示生成图像或图表的布尔参数。
        self.obs = vision
        self.set_vision(vision, step, show)
        
    def set_vision(self, obs, step, show=False):
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        # 变量说明（step）：当前经验或环境时间步编号。
        # 变量说明（show）：是否立即显示生成图像或图表的布尔参数。
        self.vision = obs[:48, :, :].astype('float32') / 255.0
        self.stats_bar = obs[49:, :, :].astype('float32') / 255.0
        
        plt.imsave(f'{self.path}/{step}.png', self.vision)
        plt.imsave(f'{self.path}/{step}_stats.png', self.stats_bar)
        if show:
            #fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            #ax[0].imshow(self.vision)
            #ax[1].imshow(self.stats_bar)
            #ax[0].set_axis_off()
            #ax[1].set_axis_off()
            plt.figure(figsize=(5, 3))
            plt.imshow(obs.astype('float32') / 255.0)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    """Didn't use, may implement in the future"""    
    def get_vision_embedding(self, reconstruct=False, show_reconstruction=False):
        
        # if self.vision_embedding is None:
        #     self.vision_embedding = torch.tensor(self.vision, dtype=torch.float32).unsqueeze(0)
        #     self.vision_embedding = self.vision_embedding.permute(0, 3, 1, 2)
        #     self.vision_embedding = self.vision_model.encoder(self.vision_embedding)
        # if reconstruct:
        #     self.render_reconstruction(show_reconstruction)
        # return self.vision_embedding
        # 变量说明（reconstruct）：是否要求视觉模型同时重建输入图像的布尔参数。
        # 变量说明（show_reconstruction）：是否弹窗显示视觉模型重建结果的布尔参数。
        return torch.tensor([0])
    
    """Didn't use, may implement in the future
    def render_reconstruction(self, show_reconstruction):
        if self.vision_reconstruction is None:
            self.vision_reconstruction = self.vision_model.decoder(self.vision_embedding)
            self.vision_reconstruction = self.vision_reconstruction.squeeze(0).permute(1, 2, 0).detach().numpy()
        # draw original and reconstructed images
        if show_reconstruction:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.vision)
            ax[1].imshow(self.vision_reconstruction)
            ax[2].imshow(self.stats_bar)
            plt.show()
    """ 
    def get_conversation_embedding(self):
        return None
        #if self.conversation_embedding is None:
        #    self.conversation_embedding = self.text_model(self.conversation)
        #return self.conversation_embedding
    
    def generate_embedding(self):
        return {"vision_embedding": self.get_vision_embedding(), "conversation_embedding": self.get_conversation_embedding()}
"""
json_output_example = \

Use double quotation marks for string values. Do not use single quotation marks.
example 1:
{
    "step": 1, 
    "goal": "Collect Wood", 
    "current_progress": "No progress",
    "partially_done": False, 
    "partially_done_task": "null", 
    "goal_completion": False, 
    "final_action": "Navigate to tree",
    "summary": "Initial step to collect wood, navigating to tree. Partial goal is not done yet. [AND EPISODE SUMMARY: what happened, what failed, why failed, what learned]"
}
example 2:
{
    "step": 8, 
    "goal": "Collect Wood", 
    "current_progress": "Collected  1 wood, facing grass",
    "partially_done": True, 
    "partially_done_task": "Collect 1 wood.", 
    "goal_completion": False, 
    "final_action": "do",
    "summary": "performed do to collect tree. Partial goal is done. [AND EPISODE SUMMARY: what happened, what failed, why failed, what learned]"
}

"""
# with open('descriptions/env_description.txt', 'r') as file:
#     env_description = file.read()
class EpisodicMemory():
    def __init__(self, text_model=None):
        # 变量说明（text_model）：调用嵌入模型生成文本向量的包装器。
        self.text_model = text_model
        
        # 变量说明（self.vitals）：玩家当前生命、食物、饮水和能量状态的文本或数值集合。
        self.vitals = None
        # 变量说明（self.inventory）：当前智能体背包内各类物品及数量。
        self.inventory = None
        # 变量说明（self.temporal_info）：`__init__` 中当前处理的状态信息。
        self.temporal_info = None
        # 变量说明（self.spatial_info）：`__init__` 中当前处理的状态信息。
        self.spatial_info = None
        # 变量说明（self.facing_direction）：`__init__` 中当前处理的方向。
        self.facing_direction = None
        # 变量说明（self.facing_object）：玩家当前朝向的相邻网格物体名称缓存。
        self.facing_object = None
        
        # 变量说明（self.scene_semantic）：用物品整数编号表示的当前局部场景语义矩阵。
        self.scene_semantic = None
        # 变量说明（self.final_response）：本次思考最终采用的结构化 LLM 输出。
        self.final_response = None
        
        # 变量说明（self.description）：当前场景、经验、参数或节点的人类可读说明文本。
        self.description = None
        # 变量说明（self.description_embedding）：`__init__` 中对应文本或图像的特征向量。
        self.description_embedding = None
        # 变量说明（self.summary）：对当前经验、目标或统计结果的精简文本摘要。
        self.summary = None
        # 变量说明（self.summary_embedding）：`__init__` 中对应文本或图像的特征向量。
        self.summary_embedding = None
        
        # 变量说明（self.episode_ct）：`__init__` 中当前处理的回合。
        self.episode_ct = 1
    
    def set_memory(self, player, temporal_info, scene_semantic, episode_number, episode_timestep):
        # 变量说明（player）：当前处理的 Crafter 玩家对象。
        # 变量说明（temporal_info）：`set_memory` 中当前处理的状态信息。
        # 变量说明（scene_semantic）：用物品整数编号表示的当前局部场景语义矩阵。
        # 变量说明（episode_number）：当前回合编号。
        # 变量说明（episode_timestep）：当前回合内的时间步编号。
        self.summary = None
        self.vitals, self.inventory = get_player_vitals_and_inventory(player)
        self.temporal_info = temporal_info
        self.spatial_info = player.pos
        self.scene_semantic = scene_semantic
        # 变量说明（self.scene_semantic_embedding）：`set_memory` 中对应文本或图像的特征向量。
        self.scene_semantic_embedding = None
        self.facing_direction = player.facing
        # 变量说明（self.episode_number）：当前回合编号。
        self.episode_number = episode_number
        # 变量说明（self.episode_timestep）：当前回合内的时间步编号。
        self.episode_timestep = episode_timestep
        
    def describe_facing_object(self):
        if self.facing_object is None:
            # 变量说明（x）：世界网格或图像中的横向坐标。
            # 变量说明（y）：世界网格或图像中的纵向坐标。
            x, y = self.spatial_info[0] + self.facing_direction[0], self.spatial_info[1] + self.facing_direction[1]
            # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
            obj = self.scene_semantic[x, y]
            self.facing_object = world_ids_item[obj]
        return self.facing_object
    
    def generate_description(self):
        self.description = f"### Current Experience\n" + \
                           f"- This is episoide {self.episode_number}. It is currently {self.episode_timestep} steps into the episoide. You are at loaction {self.spatial_info}." + \
                           f"\n- Facing Direction: " + \
                           f"You are facing a {self.describe_facing_object()}." + \
                           f"\n- Current Health Stats: " + \
                           f"{self.vitals}." + \
                           f"\n- Current Inventory: " + \
                           f"{self.inventory}.\n"
        return self.description
    
    def generate_description_embedding(self):
        if self.description_embedding is None:
            self.description_embedding = self.text_model(self.description)
        return self.description_embedding
    
    def generate_scene_semantic_embedding(self):
        # 变量说明（counts）：`generate_scene_semantic_embedding` 中待批量处理的数量集合。
        # 变量说明（unique）：语义矩阵中实际出现过的不同物品整数编号。
        unique, counts = np.unique(self.scene_semantic, return_counts=True)
        self.scene_semantic_embedding = np.zeros(19)
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        # 变量说明（u）：当前图边的起点节点，或数组计算中的第一个临时值。
        for i, u in enumerate(unique):
            self.scene_semantic_embedding[u] = counts[i]
        return self.scene_semantic_embedding
    
    # to conclude what happened
    def generate_summary(self):
        # assert self.final_response is not None, "Error genereate summary: Final response is not obtained."
        if self.final_response is None:
            self.summary = None
            return self.summary
        self.summary = self.final_response.summary
        return self.summary
        # if self.summary is None:
            # input = [   
            #             {
            #                 "role": "system", 
            #                 "content": "You are a helpful assistant." 
            #             },
            #             {
            #                 "role": "user",
            #                 "content": "Description of the environment: \n" + env_description
            #             },
            #             {
            #                 "role": "user", 
            #                 "content": "Please summarize the following content in a paragraph, clearly, comprehensively, and concisely. " + \
            #                             "1. What is the time in the episode? What happened? " + \
            #                             "2. What is the goal? Find the closest description from enviornment task list. Use the exact task name. " + \
            #                             "3. What is the current progress? "+ \
            #                             "4. Is the goal partially done? If so, what have been done? " + \
            #                             "5. If the goal is completed, did you success? Why or why not? " + \
            #                             "6. Did you learn anything new?"
            #                             "7. What is the final action taken? Why?"+ \
            #                             "Please answer in JSON format clearly with keys: step, goal, current_progress, partially_done, partially_done_task, goal_completion, final_action, and summary. The summary part should be less than 200 words." + \
            #                             "Do not add unncessary formating characters. The output should be directly convertable to json. Do not add '''json to the begining." + \
            #                             json_output_example + \
            #                             f"Content: \n{self.description + self.final_response}"
            #             }
            #         ]
            # self.summary = get_completion(input)
            # self.summary = self.summary.replace('\t', '').replace('\n', '')
            # self.summary = self.summary.replace('false', 'False').replace('true', 'True').replace('```json', '').replace('```', '').replace('json', '')
            # self.summary = self.summary.replace('null', 'None')
            # self.summary = eval(self.summary)
        return self.summary
    
    def generate_summary_embedding(self):
        if self.summary is None:
            if self.generate_summary() is None:
                self.summary_embedding = None
                return self.summary_embedding
        self.summary_embedding = self.text_model(self.summary)
        return self.summary_embedding
        # return None
    
    
    def generate_embedding(self):
        return {"description_embedding": self.generate_description_embedding(),
                "temporal_info_embedding": self.temporal_info,
                "spatial_info_embedding": self.spatial_info,
                "scene_semantic_embedding": self.generate_scene_semantic_embedding(),
                "summary_embedding": self.generate_summary_embedding(),
                } # can be None when set as target for searching as the llm have not yet generate action for current scene

# add current action
class ProceduralMemory():
    def __init__(self, text_model=None, num_past_actions=None):
        # 变量说明（text_model）：调用嵌入模型生成文本向量的包装器。
        # 变量说明（num_past_actions）：`__init__` 中对应对象的数量或累计次数。
        # 变量说明（self.actions）：按智能体编号排列、将提交给环境的动作列表。
        self.actions = None
        self.description = None
        self.description_embedding = None
        self.text_model = text_model
        # 变量说明（self.num_past_actions）：`__init__` 中对应对象的数量或累计次数。
        self.num_past_actions = num_past_actions or 7
        
    def set_memory(self, actions):
        # 变量说明（actions）：按智能体编号排列、将提交给环境的动作列表。
        self.actions = actions
        
    def generate_description(self):
        self.description = f"\n### Past Actions\n" + \
                           f"The last {self.num_past_actions} actions are: "
        # 变量说明（act）：当前准备执行或写入动作历史的动作值。
        for act in self.actions:
            self.description += act['content']
        self.description += "\n"
        return self.description
        
    def generate_embedding(self):
        if self.description is None:
            self.generate_description()
        if self.description_embedding is None:
            self.description_embedding = self.text_model(self.description)
        return {"actions_description_embedding": self.description_embedding}