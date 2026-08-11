# 模块说明：定义经验容器以及智能体的基础记忆管理结构。
from .memory_modules import SensoryMemory, EpisodicMemory, ProceduralMemory
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from .utils import *

class MemorySystem():
    def __init__(self, num_history_actions=7):
        # 变量说明（num_history_actions）：程序记忆最多向提示词提供的近期动作数量。
        # 变量说明（self.experiences）：该智能体按时间顺序积累的经验列表。
        self.experiences = []
        # 变量说明（self.goal）：`__init__` 中当前处理的目标。
        self.goal = ""
        # self.vision_embeddings = []
        # 变量说明（self.history_actions）：最近若干次高层动作及其结果的文本记录。
        self.history_actions = []
        # 变量说明（self.num_history_actions）：程序记忆最多向提示词提供的近期动作数量。
        self.num_history_actions = num_history_actions
        # 变量说明（self.current_experience）：`__init__` 中当前处理的经验。
        self.current_experience = None
    
    def add_experience(self, experience):
        # embeddings = experience.generate_embedding()
        # self.vision_embeddings.append(embeddings['vision_embedding'].squeeze(0).detach().cpu().numpy())
        # 变量说明（experience）：由当前观察、情景状态和动作历史组成的一条经验。
        self.experiences.append(experience)
        
    def generate_previous_timestamps_summary(self, k=1):
        # 变量说明（k）：限制检索、回看或邻居搜索数量的整数参数。
        # 变量说明（previous_timestamps_summary）：`generate_previous_timestamps_summary` 中当前处理的摘要。
        previous_timestamps_summary = ""#f"\nShowing the recent {k} experiences in the memory system:\n"
        # 变量说明（experience）：由当前观察、情景状态和动作历史组成的一条经验。
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        for i, experience in enumerate(self.experiences[-k:]):
            previous_timestamps_summary += f"\n### Last Experience Summary \n"
            previous_timestamps_summary += experience.episodic_memory.summary#['summary']
            previous_timestamps_summary += f"\n"
        previous_timestamps_summary += f"\n"
        return previous_timestamps_summary

    # Not using may implement in the future
    # def find_relevant_experiences(self, experience, k=3):
    #     target_experience = experience.generate_embedding()
    #     if k==0 or len(self.experiences) == 0:
    #         return "No experiences found in the memory system."

    #     exp_encoding = target_experience['vision_embedding'].squeeze(0).detach().cpu().numpy()
    #     vision_embeddings = np.array(self.vision_embeddings)
    #     cosine_sim = cosine_similarity([exp_encoding], vision_embeddings)
    #     most_similar_indices = np.argsort(cosine_sim)[0][-k:]
    #     content = f"I have found the following {k} experiences that are similar to the current situation:"
    #     for i in most_similar_indices:
    #         content += f"Experience {i}:\n" + self.experiences[i].episodic_memory.summary + "\n"
    #     return content

    def get_history_actions(self):
        return self.history_actions[-self.num_history_actions:]
    
    def __len__(self):
        return len(self.experiences)
    
class Experience:
    def __init__(self, vision_model=None, text_model=None, show=False, path=None):
        # 变量说明（vision_model）：把观察图像编码为特征的视觉模型；当前实现中为预留接口。
        # 变量说明（text_model）：调用嵌入模型生成文本向量的包装器。
        # 变量说明（show）：是否立即显示生成图像或图表的布尔参数。
        # 变量说明（path）：当前读取或写入对象对应的文件系统路径。
        # 变量说明（self.path）：当前读取或写入对象对应的文件系统路径。
        self.path = path
        # 变量说明（self.sensory_memory）：`__init__` 中当前处理的记忆。
        self.sensory_memory = SensoryMemory(vision_model, text_model, self.path)
        # 变量说明（self.episodic_memory）：`__init__` 中当前处理的记忆。
        self.episodic_memory = EpisodicMemory(text_model)
        # 变量说明（self.procedural_memory）：`__init__` 中当前处理的记忆。
        self.procedural_memory = ProceduralMemory(text_model)
        # self.sensory_embedding = None
        # 变量说明（self.episodic_embedding）：`__init__` 中对应文本或图像的特征向量。
        self.episodic_embedding = None
        # 变量说明（self.procedural_embedding）：`__init__` 中对应文本或图像的特征向量。
        self.procedural_embedding = None
        # 变量说明（self.context_description）：`__init__` 中当前处理的上下文。
        self.context_description = None
    
    def generate_embedding(self):
        # self.sensory_embedding = self.sensory_memory.generate_embedding()
        self.episodic_embedding = self.episodic_memory.generate_embedding()
        self.procedural_embedding = self.procedural_memory.generate_embedding()
        # return {**self.sensory_embedding, **self.episodic_embedding, **self.procedural_embedding}
        return {**self.episodic_embedding, **self.procedural_embedding}
    
    def generate_context_from_episodic_and_procedural_memory(self):
        # 变量说明（episodic_description）：由当前位置、背包和生命状态生成的情景记忆文本。
        episodic_description = self.episodic_memory.generate_description()
        # 变量说明（procedural_description）：由最近若干动作生成的程序记忆文本。
        procedural_description = self.procedural_memory.generate_description()
        self.context_description = episodic_description + procedural_description
        return self.context_description
    
    # Not using
    # def save_experience(self, msg, relevant_experiences):
    #     step = self.episodic_memory.temporal_info
    #     response_content = self.episodic_memory.final_response
    #     save_txt(f'results/images3/{step}.txt', msg, self.context_description, relevant_experiences, response_content)
