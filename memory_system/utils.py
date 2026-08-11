# 模块说明：提供提示词构造、动作解析、寻路和终端输出等记忆系统辅助函数。
import base64
from mimetypes import guess_type
from pathlib import Path
from PIL import Image
import io
from .constants import world_item_ids, walkables, MATERIALS, ACTIONS, GENERAL_DESCRIPTION, ENV_DESCRIPTION, RESPONSE_DESCRIPTION, VITALS
from .pathfinding import PathFinding
import crafter.constants as const
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 变量说明（pathfinding_model）：模块 中当前处理的模型。
pathfinding_model = PathFinding(walkables)

def local_image_to_data_url(image_path):
    # 变量说明（image_path）：`local_image_to_data_url` 读取或写入当前文件时使用的完整路径。
    image_path = str(image_path)
    # 变量说明（mime_type）：当前图像或消息附件的 MIME 媒体类型。
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    # Read and encode the image file
    # 变量说明（image_file）：`local_image_to_data_url` 中当前处理的图像、文件。
    with open(image_path, "rb") as image_file:
        # 变量说明（base64_encoded_data）：图像二进制内容经过 Base64 编码后的 ASCII 字符串。
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


# Reference texture sheet shown to the LLM alongside each agent view.
# 变量说明（TEXTURE_REFERENCE_PATH）：模块 读取或写入当前文件时使用的完整路径。
TEXTURE_REFERENCE_PATH = Path(__file__).resolve().with_name("textures.png")
# 变量说明（textures）：从素材目录加载并按名称索引的纹理图像集合。
textures = local_image_to_data_url(TEXTURE_REFERENCE_PATH)

#load the text from the file
def generate_inquiry(vision, description=""):
    # 变量说明（vision）：从完整观察中切出的当前游戏画面像素数组。
    # 变量说明（description）：当前场景、经验、参数或节点的人类可读说明文本。
    # 变量说明（data_url）：把图像字节编码为可嵌入消息的 data URL。
    data_url = local_image_to_data_url(vision)
    # 变量说明（msg）：准备发送给大模型或其他组件的消息内容。
    msg = [ 
            {
                "role": "system", 
                "content": "You are a helpful assistant." 
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": GENERAL_DESCRIPTION + ENV_DESCRIPTION + "The image of all materials in the game and your current view of the envionrment."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": textures,
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    },
                    {
                        "type": "text",
                        "text": description + """\n
### Instructions for response
- Please answer each question accurately and faithfully relying on current inventory, view, and situation (health stats, inventory items, and recent actions) and what you see.
- consider the past experiences, goals, and relevant expriences. What happened, what did you do, did you succeed, and can you do better?
- consider action prerequisites when make plans.\n
""",
                    },
                ]
            }
        ]
    return msg

# Not Using
# def find_most_similar_term(term, references):
#     vectorizer = TfidfVectorizer()
#     max = 0
#     most_similar = ''
#     for ref in references:
#         vectors = vectorizer.fit_transform([term, ref])
#         similarity = cosine_similarity(vectors)
#         similarity_score = similarity[0][1]
#         if similarity_score > max:
#             max = similarity_score
#             most_similar = ref
#     return most_similar, max

def print_color(*args, **kwargs):
    # get color from kwargs
    # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
    # 变量说明（kwargs）：调用方通过关键字传入、继续转发给底层对象的参数字典。
    # 变量说明（color）：`print_color` 中当前处理的颜色。
    color = kwargs['color']
    # get color code
    # 变量说明（color_code）：`print_color` 中当前处理的颜色。
    color_code = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'black': '\033[30m',
    }.get(color, '\033[30m')
    print(color_code, end='')
    print(*args, end='')
    print('\033[0m')

# def safe_extract_op(response_content):
#     history = ({
#             "role": "user", 
#             "content": f"\"Could not recognize action\". " 
#     })
    
#     try:
#         op = response_content.split('#***#')
#         if len(op) >= 2:
#             op = op[-2]
#             op = op[-50:] # only take the last 50 characters
#         else:
#             print_color(f"ExtractError0: could not extract from op. {response_content} is splitted into {len(op)} pieces, setting op to noop.", color='red')
#             return 'noop', None, history
#     except:
#         print_color(f"ExtractError1: could not extract from {response_content}, setting op to noop.", color='red')
#         return 'noop', None, history
    
#     most_likely_op, _ = find_most_similar_term(op, references=ACTIONS)
#     if most_likely_op == '':
#         print_color(f"ExtractError2: could not find most similar term for {op}, setting op to noop.", color='red')
#         return 'noop', None, history
#     else:
#         print_color(f"Skill: {most_likely_op}", color='cyan')
#         if most_likely_op == 'Navigator':
#             try:
#                 rss = op.split(';')[1]
#             except:
#                 print_color(f"ExtractError3: could not extract rss from {op}, setting rss to grass.", color='red')
#                 return 'noop', None, history
            
#             most_likely_rss, _ = find_most_similar_term(rss, references=MATERIALS)
#             if most_likely_rss == '':
#                 print_color(f"ExtractError4: could not find most similar term for {rss}, setting rss to grass.", color='red')
#                 return 'noop', None, history
#             else:
#                 print_color(f"Resource: {most_likely_rss}", color='cyan')
#                 history = {
#                     "role": "user", 
#                     "content": f"Attempted \"navigate to a {most_likely_rss}\". " 
#                 }
#                 return most_likely_op, most_likely_rss, history # no history added as the action is recognized
#         else:
#             history = {
#                     "role": "user", 
#                     "content": f"Attempted \"{most_likely_op}\". " 
#                 }
#             return most_likely_op, None, history # no history added as the action is recognized

def safe_extract_op(response_content):
    # 变量说明（response_content）：`safe_extract_op` 中当前处理的响应。
    # 变量说明（op）：LLM 选定的高层操作名称，例如导航、制造或共享。
    op = response_content.action.final_next_action
    # 变量说明（rss_to_collect）：导航操作要寻找并采集的目标资源名称。
    rss_to_collect = response_content.action.final_target_material_to_collect
    # 变量说明（rss_to_share）：共享操作准备转交给其他智能体的资源名称。
    rss_to_share = response_content.action.final_target_material_to_share 
    # 变量说明（target_id）：`safe_extract_op` 中当前对象的唯一编号或序列索引。
    target_id = response_content.action.final_target_agent_id
    if op == "Navigator":
        # 变量说明（history）：按时间顺序保存的既往动作、状态或统计记录。
        history = ({
                "role": "user", 
                "content": f"Attempted \"navigate to a {rss_to_collect}\". " 
        })
    elif op == "share":
        history = ({
                "role": "user", 
                "content": f"Attempted \"share {rss_to_share} with agent {target_id}\". " 
        })
    else:
        history = ({
                "role": "user", 
                "content": f"Attempted \"{op}\". " 
        })
    return op, rss_to_collect, rss_to_share, target_id, history


def go_and_find(info, target_rss):
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    # 变量说明（target_rss）：自动导航器正在寻找的目标资源名称。
    # 变量说明（rss）：当前要采集、检查或共享的资源名称。
    rss = world_item_ids[target_rss]
    # 变量说明（_pos）：对象在世界网格中的二维坐标。
    # 变量说明（grid）：按世界坐标存储材质或寻路状态的二维网格。
    grid, _pos = info['semantic'].T, info['player_pos']
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    pos = (_pos[1], _pos[0])  # _pos is in (x, y), but grid is in (r, c)
    # 变量说明（actions）：按智能体编号排列、将提交给环境的动作列表。
    actions = pathfinding_model.find(grid, pos, rss)
    if actions:
        # 变量说明（is_finished）：导航器是否已经到达目标并完成交互的布尔标记。
        is_finished = True if len(actions) == 1 else False # 2 means one more step; 0 means not finished
        return actions[0], is_finished
    else:
        return 0, True # 0 is action for noop; 1 means finished


def go_to_something(env, rss):
    # 变量说明（env）：当前 Crafter 仿真环境实例。
    # 变量说明（rss）：当前要采集、检查或共享的资源名称。
    # 变量说明（is_finished）：导航器是否已经到达目标并完成交互的布尔标记。
    is_finished = 0 # 0 means not finished; 1 means finished; 2 means one more step
    # 变量说明（action）：当前智能体准备执行的环境动作。
    action = 0
    # 变量说明（num_step）：`go_to_something` 中对应对象的数量或累计次数。
    num_step = 0
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    done = False
    # 变量说明（history）：按时间顺序保存的既往动作、状态或统计记录。
    history = ({
        "role": "user", 
        "content": f"\"Failed navigate to a {rss} \". " 
    })
    while not done:
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        obs, _, done, info = env.step(action)
        action, is_finished = go_and_find(info, rss)
        num_step += 1
        if is_finished:
            if action == -1:
                return obs, num_step, done, info, history
            history = ({
                "role": "user", 
                "content": f"\"Navigated to a {rss}\". "
            })
            if action != 0:
                obs, _, done, info = env.step(action) # step one more time to face the target rss
                num_step += 1
            return obs, num_step, done, info, history
    
    return obs, num_step, done, info, history
    

def safe_execute_skill(env, op, rss, step):
    # 变量说明（env）：当前 Crafter 仿真环境实例。
    # 变量说明（op）：LLM 选定的高层操作名称，例如导航、制造或共享。
    # 变量说明（rss）：当前要采集、检查或共享的资源名称。
    # 变量说明（step）：当前经验或环境时间步编号。
    if op == 'Navigator':
        # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
        # 变量说明（history）：按时间顺序保存的既往动作、状态或统计记录。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        # 变量说明（step_）：`safe_execute_skill` 中当前处理的时间步。
        obs, step_, done, info, history = go_to_something(env, rss) # return history if the navigator worked
        step += step_
    else:
        # 变量说明（action）：当前智能体准备执行的环境动作。
        action = const.actions.index(op)
        obs, _, done, info = env.step(action)
        step += 1
        if action == 0:
            history = None
        else:
            history = {
                "role": "user", 
                "content": f"Attempted \"{op}\". " 
            }
    return obs, step, done, info, history

# def generate_image(obs, path):
#     img = Image.fromarray(obs[:48,:, :], 'RGB') # only image part
#     img.save(f'{path}.jpg')
#     stats_image = Image.fromarray(obs[48:,:, :], 'RGB') # only image part
#     stats_image.save(f'{path}_stats.jpg')
#     return img, stats_image

def get_player_vitals_and_inventory(player):
    # 变量说明（player）：当前处理的 Crafter 玩家对象。
    # 变量说明（vitals_str）：玩家生命、食物、饮水和能量状态的格式化文本。
    vitals_str = ""
    # 变量说明（inventory_str）：`get_player_vitals_and_inventory` 中当前处理的背包。
    inventory_str = ""
    # 变量说明（key）：当前字典条目的键。
    # 变量说明（value）：当前配置项、字段或计算结果对应的值。
    for key, value in player.inventory.items():
        if key in VITALS:
            vitals_str += f"{key}: {value}, "
        else:
            inventory_str += f"{key}: {value}, "
    return vitals_str[:-2], inventory_str[:-2]

# save jsonl file
# def save_jsonl(path, msg):
#     with open(path, 'w') as f:
#         for entry in msg:
#             f.write(json.dumps(entry) + '\n')
#         f.write(episodic_description+procedural_summary+relevant_experiences)
#         f.write(json.dumps({"Response": f"{response_content}"}) + '\n')

# Not Using
# def save_txt(path, msg, context, relevant_experiences, response_content):
#     with open(path, 'w') as f:
#         for entry in msg:
#             if type(entry['content']) == list:
#                 for c in entry['content']:
#                     if 'text' in c.keys():
#                         f.write(c['text'])
#                         f.write("\n")
#                     else:
#                         f.write(c['image_url']['url'])
#                         f.write("\n")
#             else:
#                 f.write(entry['content'])
#                 f.write("\n")
#         f.write("\n===================== context ===================\n")
#         f.write(context)
#         f.write("\n===================== relevant_experiences ===================\n")
#         f.write(relevant_experiences)
#         f.write("\n===================== response_content ===================\n")
#         f.write(response_content)

if __name__ == "__main__":
    # 变量说明（data_url）：把图像字节编码为可嵌入消息的 data URL。
    data_url = local_image_to_data_url("images/0.jpg")
    print(data_url)
