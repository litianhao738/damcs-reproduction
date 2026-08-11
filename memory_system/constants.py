# 模块说明：保存记忆系统使用的世界尺寸、物品映射和动作常量。
import crafter
import crafter.constants as const

# 变量说明（env）：当前 Crafter 仿真环境实例。
env = crafter.Env()
# 变量说明（world_item_ids）：模块 中待批量处理的世界、物品集合。
world_item_ids = env._world._mat_ids
world_item_ids.update({'player': len(world_item_ids),
                        'cow': len(world_item_ids) + 1, 
                        'zombie': len(world_item_ids) + 2,
                        'skeleton': len(world_item_ids) + 3,
                        'arrow': len(world_item_ids) + 4,
                        'plant': len(world_item_ids) + 5, 
                    })
# reverse the dictionary
# 变量说明（k）：限制检索、回看或邻居搜索数量的整数参数。
# 变量说明（v）：当前图边的终点节点，或数组计算中的第二个临时值。
# 变量说明（world_ids_item）：模块 中当前处理的世界、物品。
world_ids_item = {v: k for k, v in world_item_ids.items()}
# 变量说明（w）：当前区域、图像或纹理的像素宽度。
# 变量说明（walkables）：寻路过程中多个候选可通行网格的列表。
walkables = {world_item_ids[w] for w in const.walkable}
del env

# 变量说明（ACTIONS）：按智能体编号排列、将提交给环境的动作列表。
ACTIONS = ["noop", "move_left", "move_right",
                "move_up","move_down","do",
                "sleep","place_stone", "place_table",
                "place_furnace","place_plant","make_wood_pickaxe",
                "make_stone_pickaxe","make_iron_pickaxe","make_wood_sword",
                "make_stone_sword","make_iron_sword", "Navigator"]
# 变量说明（MATERIALS）：Crafter 支持的全部地形和资源材质名称列表。
MATERIALS = ["water", "grass", "stone", 
                "path", "sand", "tree", 
                "lava", "coal", "iron", 
                "diamond", "table", "furnace", 'cow']
# 变量说明（VITALS）：玩家当前生命、食物、饮水和能量状态的文本或数值集合。
VITALS = ["health", "food", "drink", "energy"]
# 变量说明（file）：模块 中当前处理的文件。
with open("descriptions/general_description.txt", "r") as file:
    # 变量说明（GENERAL_DESCRIPTION）：从说明文件读取的通用智能体行为约束文本。
    GENERAL_DESCRIPTION = file.read()
with open("descriptions/env_description.txt", "r") as file:
    # 变量说明（ENV_DESCRIPTION）：从说明文件读取的 Crafter 环境规则文本。
    ENV_DESCRIPTION = file.read()
with open("descriptions/response_description.txt", "r") as file:
    # 变量说明（RESPONSE_DESCRIPTION）：模块 中当前处理的响应。
    RESPONSE_DESCRIPTION = file.read()
# 变量说明（WORLD_SIZE）：模块 中当前处理的世界。
WORLD_SIZE = [64, 64]