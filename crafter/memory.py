# 模块说明：提供简易环形帧缓冲区。
class ChatMemory:
    def __init__(self, player_id):
        # 变量说明（player_id）：当前查看或操作的玩家编号。
        # 变量说明（self.player_id）：当前查看或操作的玩家编号。
        self.player_id = player_id
        # 变量说明（self.memory）：`__init__` 中当前处理的记忆。
        self.memory = []

    def store(self, msg):
        # 变量说明（msg）：准备发送给大模型或其他组件的消息内容。
        self.memory.append(msg)
        
    def __str__(self):
        return f"This is the memory of agent {self.player_id}. \nThe latest 5 conversations are: ###{self.memory[-5:]}.###"