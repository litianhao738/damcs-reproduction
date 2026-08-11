# 模块说明：实现支持多个玩家的 Crafter 环境、奖励、观察和物资交换。
import collections

import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen

from . import memory

# 变量说明（NONE_EXCHANGE_ITEMS）：禁止在玩家之间转移的生命值、能量等状态字段。
NONE_EXCHANGE_ITEMS = ['health', 'energy']
# Gym is an optional dependency.
try:
  import gym
  # 变量说明（DiscreteSpace）：离散动作空间类型；无 Gym 时使用兼容命名元组。
  DiscreteSpace = gym.spaces.Discrete
  # 变量说明（BoxSpace）：连续张量空间类型；无 Gym 时使用兼容命名元组。
  BoxSpace = gym.spaces.Box
  # 变量说明（DictSpace）：字典组合空间类型；无 Gym 时使用兼容命名元组。
  DictSpace = gym.spaces.Dict
  # 变量说明（BaseClass）：有 Gym 时指向 gym.Env、否则回退到 object 的环境基类。
  BaseClass = gym.Env
except ImportError:
  DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
  BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
  DictSpace = collections.namedtuple('DictSpace', 'spaces')
  BaseClass = object

class Env(BaseClass):
  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=10000, n_players=1, seed=None):
    # 变量说明（area）：生成世界时使用的二维地图尺寸。
    # 变量说明（view）：单个玩家局部观察窗口的行列尺寸。
    # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    # 变量说明（length）：一个环境回合最多允许推进的时间步数。
    # 变量说明（n_players）：共享环境中的玩家总数。
    # 变量说明（seed）：控制世界生成和随机事件、保证结果可复现的随机种子。
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
    # 变量说明（self._area）：生成世界时使用的二维地图尺寸。
    self._area = area
    # 变量说明（self._view）：单个玩家局部观察窗口的行列尺寸。
    self._view = view
    # 变量说明（self._size）：当前图像、窗口、网格或集合的尺寸。
    self._size = size
    # 变量说明（self._reward）：当前时间步或当前玩家获得的标量奖励。
    self._reward = reward
    # 变量说明（self._length）：一个环境回合最多允许推进的时间步数。
    self._length = length
    # 变量说明（self._seed）：控制世界生成和随机事件、保证结果可复现的随机种子。
    self._seed = seed
    # 变量说明（self._episode）：当前回合编号或回合相关数据。
    self._episode = 0
    # 变量说明（self._world）：包含材质网格和世界对象的 Crafter 世界实例。
    self._world = engine.World(area, constants.materials, (12, 12))
    # 变量说明（self._textures）：从素材目录加载并按名称索引的纹理图像集合。
    self._textures = engine.Textures(constants.root / 'assets')
    # 变量说明（item_rows）：观察画面底部显示背包物品所需的像素行数。
    item_rows = int(np.ceil(len(constants.items) / view[0]))
    # 变量说明（self._local_view）：把玩家附近世界区域渲染成图像的局部视图对象。
    self._local_view = engine.LocalView(
        self._world, self._textures, [view[0], view[1] - item_rows])
    # 变量说明（self._item_view）：把玩家生命状态和背包渲染到状态栏的视图对象。
    self._item_view = engine.ItemView(
        self._textures, [view[0], item_rows])
    # 变量说明（self._sem_view）：把局部世界对象转换成整数编号矩阵的语义视图对象。
    self._sem_view = engine.SemanticView(self._world, [
        objects.Player, objects.Cow, objects.Zombie,
        objects.Skeleton, objects.Arrow, objects.Plant])
    # 变量说明（self._step）：当前经验或环境时间步编号。
    self._step = None
    # 变量说明（self._unlocked）：本回合所有玩家已经首次完成的成就名称集合。
    self._unlocked = {}
    
    #self._player = None
    #self._last_health = None
    
    # multi player
    # 变量说明（self.player_id）：当前查看或操作的玩家编号。
    self.player_id = 0 # set the curr viewing player
    # 变量说明（self.n_players）：共享环境中的玩家总数。
    self.n_players = n_players
    # 变量说明（self._alive_players_id）：当前仍然存活的玩家编号集合。
    self._alive_players_id = set() # player ids
    # 变量说明（self.canvases）：每个玩家在当前时间步渲染出的观察画面列表。
    self.canvases = []
    # 变量说明（id_）：`__init__` 中当前处理的编号。
    # 变量说明（self.memories）：每个玩家用于保存通信消息的 ChatMemory 对象列表。
    self.memories = [memory.ChatMemory(id_) for id_ in range(self.n_players)]
    # 变量说明（self._players）：共享世界中的全部玩家对象。
    self._players = [None] * self.n_players
    # 变量说明（self._last_healths）：上一个时间步各玩家的生命值，用于计算受伤奖励。
    self._last_healths = [None] * self.n_players
    # 变量说明（self._last_inventory）：上一个时间步各玩家的背包快照，用于计算采集奖励。
    self._last_inventory = [None] * self.n_players
    
    # Some libraries expect these attributes to be set.
    # 变量说明（self.reward_range）：Gym 兼容字段；当前环境未显式限制奖励取值范围。
    self.reward_range = None
    # 变量说明（self.metadata）：Gym 兼容字段；用于声明环境渲染能力的元数据。
    self.metadata = None

  # multiagent gui
  def switch_player(self, player_id):
    # 变量说明（player_id）：当前查看或操作的玩家编号。
    self.player_id = player_id
    
  def exchange_item(self, target_player_id, item):
    # 变量说明（target_player_id）：`exchange_item` 中当前对象的唯一编号或序列索引。
    # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
    if item in NONE_EXCHANGE_ITEMS:
      print("Cannot exchange item: ", item)
      return
    if self.player_id == target_player_id:
      print("Cannot exchange item with self.")
      return
    # 变量说明（curr_player）：`exchange_item` 中当前处理的玩家。
    curr_player = self._players[self.player_id]
    # curr_player_item_before = curr_player.inventory[item]
    # target_player_item_before = self._players[target_player_id].inventory[item]
    if curr_player.inventory[item] > 0:
      curr_player.inventory[item] -= 1
      self._players[target_player_id].inventory[item] += 1

    # assert curr_player_item_before == curr_player.inventory[item] + 1
    # assert target_player_item_before == self._players[target_player_id].inventory[item] - 1

  def chat(self, msg, player_ids):
    # 变量说明（msg）：准备发送给大模型或其他组件的消息内容。
    # 变量说明（player_ids）：`chat` 中待批量处理的玩家集合。
    # 变量说明（player_id）：当前查看或操作的玩家编号。
    for player_id in player_ids:
      self.memories[player_id].store(msg)
  
  def show_history(self):
    print(self.memories[self.player_id])
      
  @property
  def observation_space(self):
    return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

  @property
  def action_space(self):
    return DiscreteSpace(len(constants.actions))

  @property
  def action_names(self):
    return constants.actions

  def reset(self):
    self._episode += 1
    
    self._step = 0
    # fixed world
    #self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._world.reset(seed=hash((self._seed, self._seed)) % (2 ** 31 - 1))
    self._update_time()
    
    # 变量说明（center）：当前对象、视图或地图区域的中心坐标。
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    # 变量说明（x）：世界网格或图像中的横向坐标。
    # 变量说明（y）：世界网格或图像中的纵向坐标。
    x, y = center
    
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    for i in range(self.n_players):
      self._alive_players_id.add(i)
      self._players[i] = objects.Player(self._world, (x+i - self.n_players // 2, y))
      self._last_healths[i] = self._players[i].health
      self._last_inventory[i] = self._players[i].inventory.copy()
      self._world.add(self._players[i])

    self._unlocked = set()
    worldgen.generate_world(self._world, self._players[self.n_players // 2])
    self.canvases = []
    self.render_all()
    if self.n_players == 1:
      return self._obs()[0]
    return self._obs()
  
  def step(self, actions):
    # 变量说明（actions）：按智能体编号排列、将提交给环境的动作列表。
    self._step += 1
    self._update_time()
    
    # 变量说明（player_rewards）：`step` 中待批量处理的玩家、奖励集合。
    player_rewards = {}
    # 变量说明（unlocked）：本回合所有玩家已经首次完成的成就名称集合。
    unlocked = set()
    # 变量说明（players_to_remove）：`step` 中当前处理的玩家。
    players_to_remove = []
    # 变量说明（player_id）：当前查看或操作的玩家编号。
    for player_id in self._alive_players_id:
      if self.n_players == 1:
        if type(actions) is list:
          # 变量说明（action）：当前智能体准备执行的环境动作。
          action = actions[0]
        else:
          action = actions
      else:
        action = actions[player_id]
      # 变量说明（curr_reward）：`step` 中当前处理的奖励。
      # 变量说明（is_alive）：表示 `is_alive` 所描述条件是否成立的布尔标记。
      curr_reward, is_alive = self.step_one_player(action, player_id)
      player_rewards[player_id] = curr_reward
      
      # unlocked.update(curr_unlocked)
      if is_alive == False:
        players_to_remove.append(player_id)
    
    for player_id in players_to_remove:
      self._alive_players_id.remove(player_id)
    
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    for obj in self._world.objects:
      obj.update()
    if self._step % 10 == 0:
      # 变量说明（chunk）：从完整数组或文件中切出的当前数据块。
      # 变量说明（objs）：当前网格区域内待处理的世界对象集合。
      for chunk, objs in self._world.chunks.items():
        self._balance_chunk(chunk, objs)
        
    self.render_all()
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    obs = self._obs()
    
    # 变量说明（all_dead）：是否所有玩家都已死亡、需要结束回合的布尔标记。
    all_dead = (len(self._alive_players_id) <= 0)
    # 变量说明（over）：环境时间步是否达到回合长度上限的布尔标记。
    over = self._length and self._step >= self._length
    
    # added for single task
    # 变量说明（is_collected_diamond）：表示 `is_collected_diamond` 所描述条件是否成立的布尔标记。
    # 变量说明（player）：当前处理的 Crafter 玩家对象。
    is_collected_diamond = [player.achievements['collect_diamond'] > 0 for player in self._players]
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    done = all_dead or over or any(is_collected_diamond)
    if done:
      print("!!Done!!")
    #reward = sum(reward)
    #if unlocked:
    #  before = len(self._unlocked)
    #  self._unlocked |= unlocked
    #  after = len(self._unlocked)
    #  reward += max(1.0, after - before)
      
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    info = []
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    reward = []
    # 变量说明（id_）：`step` 中当前处理的编号。
    # 变量说明（p）：当前循环正在处理的玩家、位置或概率值。
    for id_, p in enumerate(self._players):
      reward.append(player_rewards[id_] if id_ in player_rewards.keys() else 0)
      # 变量说明（info_p）：`step` 中当前处理的状态信息。
      info_p = {
        'id': id_,
        'inventory': p.inventory.copy(),
        'achievements': p.achievements.copy(),
        'sleeping': p.sleeping,
        'discount': 1 - float(p.health <= 0),
        'semantic': self._sem_view(),
        'player_pos': p.pos,
        'player_facing': p.facing,
        'reward': reward[-1],
        'dead': p.health <= 0,
        'unlocked': unlocked,
        'action': p.action,
        'view': self._view,
      }
      info.append(info_p)
    #if not self._reward:
    #  reward = 0.0
      
    if self.n_players == 1:
      return obs[0], reward[0], done, info[0]
    return obs, reward, done, info
    
  def step_one_player(self, action, player_id):
    # 变量说明（action）：当前智能体准备执行的环境动作。
    # 变量说明（player_id）：当前查看或操作的玩家编号。
    # 变量说明（curr_player）：`step_one_player` 中当前处理的玩家。
    curr_player = self._players[player_id]
    
    curr_player.action = constants.actions[action]
    #for obj in self._world.objects:
    #  if curr_player.distance(obj) < 2 * max(self._view):
    #    obj.update()
    
    #if self._step % 10 == 0:
    #  for chunk, objs in self._world.chunks.items():
    #    self._balance_chunk(chunk, objs)
    # updated
    # 变量说明（curr_player_reward）：`step_one_player` 中当前处理的玩家、奖励。
    curr_player_reward = (curr_player.health - self._last_healths[player_id]) # / 10
    self._last_healths[player_id] = curr_player.health
    
    # task_difficulties = {
    #     # first time complete the task
    #     'collect_wood': 2, 'collect_sapling': 2, 'place_plant': 5, 'eat_plant': 10, 'collect_stone': 5, 'make_wood_pickaxe': 2, 'make_wood_sword': 2,
    #     'collect_coal': 10, 'collect_iron': 10, 'make_stone_pickaxe': 10, 'make_stone_sword': 10, 'place_table': 20, 'place_furnace': 20, 'collect_drink': 2, 
    #     'wake_up': 5, 'make_iron_pickaxe': 7, 'make_iron_sword': 7, 'eat_cow': 5, 'collect_diamond': 10, 'place_stone': 1, 'defeat_zombie': 8, 'defeat_skeleton': 8,
    #     # subsequent tasks
    #     'wood': 2, 'stone': 4, 'coal': 6, 'iron': 6, 'diamond': 30,
    # }
    # task_difficulties = {
    #     # first time complete the task
    #     'collect_wood': 1, 'collect_sapling': 1, 'place_plant': 1, 'eat_plant': 1, 'collect_stone': 1, 'make_wood_pickaxe': 3, 'make_wood_sword': 3,
    #     'collect_coal': 3, 'collect_iron': 3, 'make_stone_pickaxe': 4, 'make_stone_sword': 4, 'place_table': 3, 'place_furnace': 3, 'collect_drink': 1, 
    #     'wake_up': 1, 'make_iron_pickaxe': 3, 'make_iron_sword': 3, 'eat_cow': 2, 'collect_diamond': 1, 'place_stone': 1, 'defeat_zombie': 5, 'defeat_skeleton': 5,
    #     # subsequent tasks
    #     'wood': 1, 'stone': 2, 'coal': 3, 'iron': 3, 'diamond': 10,
    # }
    
    # 变量说明（task_difficulties）：`step_one_player` 中待批量处理的任务集合。
    task_difficulties = {
        # first time complete the task
        'collect_wood': 1, 'collect_sapling': 0, 'place_plant': 0, 'eat_plant': 0, 'collect_stone': 4, 'make_wood_pickaxe': 3, 'make_wood_sword': 0,
        'collect_coal': 4, 'collect_iron': 6, 'make_stone_pickaxe': 5, 'make_stone_sword': 1, 'place_table': 2, 'place_furnace': 5, 'collect_drink': 1, 
        'wake_up': 1, 'make_iron_pickaxe': 7, 'make_iron_sword': 0, 'eat_cow': 1, 'collect_diamond': 10, 'place_stone': 1, 'defeat_zombie': 0, 'defeat_skeleton': 0,
        # subsequent tasks
        'wood': 1, 'stone': 1, 'coal': 1, 'iron': 1, 'diamond': 1,
    }

    # 变量说明（count）：`step_one_player` 中当前处理的数量。
    # 变量说明（name）：当前对象、指标或配置项的名称。
    # 变量说明（unlocked）：本回合所有玩家已经首次完成的成就名称集合。
    unlocked = {
      name for name, count in curr_player.achievements.items()
      if count > 0 and name not in self._unlocked}

    if action == 0: # if noop
     curr_player_reward -= 1
    elif action < 5:
     curr_player_reward += 0.1
      
    if unlocked:
      self._unlocked |= unlocked
      # print("Unlocked: ", unlocked)
      # 变量说明（t）：当前时间、纹理或元组元素。
      for t in unlocked:
        curr_player_reward += task_difficulties[t] * 2
     
    # find the difference between the two dictionaries
    # diff_inventory = {k: curr_player.inventory[k] - self._last_inventory[player_id][k] for k in curr_player.inventory if curr_player.inventory[k] != self._last_inventory[player_id][k]}
    # for k, v in diff_inventory.items():
    #   if k in task_difficulties.keys() and v > 0:
    #     curr_player_reward += task_difficulties[k]
    self._last_inventory[player_id] = curr_player.inventory.copy()
    
    # remove if health is 0
    # 变量说明（is_alive）：表示 `is_alive` 所描述条件是否成立的布尔标记。
    is_alive = True
    if curr_player.health <= 0:
      self._world.remove(curr_player)
      is_alive = False
    return curr_player_reward, is_alive
    #return curr_player_reward, curr_unlocked, is_alive

  def render_one_player(self, player_id, size=None):
    # 变量说明（player_id）：当前查看或操作的玩家编号。
    # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
    size = size or self._size
    # 变量说明（unit）：世界坐标映射到渲染画布时每个网格单元的像素尺寸。
    unit = size // self._view
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    
    # 变量说明（curr_player）：`render_one_player` 中当前处理的玩家。
    curr_player = self._players[player_id]
    # 变量说明（local_view）：把玩家附近世界区域渲染成图像的局部视图对象。
    local_view = self._local_view(curr_player, unit)
    # 变量说明（item_view）：把玩家生命状态和背包渲染到状态栏的视图对象。
    item_view = self._item_view(curr_player.inventory, unit)
    # 变量说明（view）：单个玩家局部观察窗口的行列尺寸。
    view = np.concatenate([local_view, item_view], 1)
    
    # 变量说明（border）：纹理切片或地图区域外围需要保留的边框宽度。
    border = (size - (size // self._view) * self._view) // 2
    # 变量说明（h）：当前区域、图像或纹理的像素高度。
    # 变量说明（w）：当前区域、图像或纹理的像素宽度。
    # 变量说明（x）：世界网格或图像中的横向坐标。
    # 变量说明（y）：世界网格或图像中的纵向坐标。
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))
  
  def render_all(self, size=None):
    # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
    self.canvases = []
    # 变量说明（player_id）：当前查看或操作的玩家编号。
    for player_id in range(len(self._players)):
      self.canvases.append(self.render_one_player(player_id, size))    
    return self.canvases
  
  def render(self, size=None):
    # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
    self.render_all(size)
    return self.canvases[self.player_id]

  def _obs(self):
    return self.canvases

  """def render(self, size=None):
    size = size or self._size
    unit = size // self._view
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    
    curr_player = self._players[self.player_id]
    local_view = self._local_view(curr_player, unit)
    item_view = self._item_view(curr_player.inventory, unit)
    view = np.concatenate([local_view, item_view], 1)
    
    border = (size - (size // self._view) * self._view) // 2
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))

  def _obs(self):
    return self.render()"""

  def _update_time(self):
    # https://www.desmos.com/calculator/grfbc6rs3h
    # 变量说明（progress）：当前目标、下载或实验任务的完成进度。
    progress = (self._step / 300) % 1 + 0.3
    # 变量说明（daylight）：当前时间步的环境光照强度，取值通常位于 0 到 1。
    daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
    self._world.daylight = daylight

  def _balance_chunk(self, chunk, objs):
    # 变量说明（chunk）：从完整数组或文件中切出的当前数据块。
    # 变量说明（objs）：当前网格区域内待处理的世界对象集合。
    # 变量说明（light）：当前区域或时间步计算出的环境亮度。
    light = self._world.daylight
    # for player_id in self._alive_players_id:
    #   curr_player = self._players[player_id]
    #   self._balance_object(
    #       chunk, objs, objects.Zombie, 'grass', 6, 0, 0.0, 0.4,
    #       lambda pos: objects.Zombie(self._world, pos, curr_player),
    #       lambda num, space: (
    #           0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
    #   self._balance_object(
    #       chunk, objs, objects.Skeleton, 'path', 7, 7, 0.0, 0.1,
    #       lambda pos: objects.Skeleton(self._world, pos, curr_player),
    #       lambda num, space: (0 if space < 6 else 1, 2))
    
    # no need go in forloop
    self._balance_object(
       chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
       # 变量说明（pos）：对象在世界网格中的二维坐标。
       lambda pos: objects.Cow(self._world, pos),
       # 变量说明（num）：平衡数据块时当前数值需要拆分或补齐的元素数量。
       # 变量说明（space）：当前数据块距离目标长度仍可容纳的元素数量。
       lambda num, space: (0 if space < 30 else 1, 1.5 + light))

  def _balance_object(
      self, chunk, objs, cls, material, span_dist, despan_dist,
      spawn_prob, despawn_prob, ctor, target_fn):
    
    # 变量说明（chunk）：从完整数组或文件中切出的当前数据块。
    # 变量说明（objs）：当前网格区域内待处理的世界对象集合。
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    # 变量说明（span_dist）：实体与玩家距离小于该阈值时允许生成的距离。
    # 变量说明（despan_dist）：实体距离玩家超过该阈值时允许移除的距离。
    # 变量说明（spawn_prob）：当前时间步生成某类生物或对象的概率。
    # 变量说明（despawn_prob）：当前时间步移除远处生物或对象的概率。
    # 变量说明（ctor）：当前语义编号对应的世界对象构造类型。
    # 变量说明（target_fn）：被记录器包装、实际执行环境操作的原始函数。
    # 变量说明（xmax）：当前矩形或地图区域包含的最大横坐标。
    # 变量说明（xmin）：当前矩形或地图区域包含的最小横坐标。
    # 变量说明（ymax）：当前矩形或地图区域包含的最大纵坐标。
    # 变量说明（ymin）：当前矩形或地图区域包含的最小纵坐标。
    xmin, xmax, ymin, ymax = chunk
    # 变量说明（random）：控制环境包装器是否使用随机动作的布尔参数。
    random = self._world.random
    # 变量说明（creatures）：世界生成或更新阶段待创建、遍历的生物对象列表。
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    creatures = [obj for obj in objs if isinstance(obj, cls)]
    # 变量说明（mask）：`_balance_object` 中当前处理的掩码。
    mask = self._world.mask(*chunk, material)
    # 变量说明（target_max）：当前数组归一化后的目标上界。
    # 变量说明（target_min）：当前数组归一化后的目标下界。
    target_min, target_max = target_fn(len(creatures), mask.sum())
    if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
      # 变量说明（xs）：当前区域包含的横坐标序列。
      xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
      # 变量说明（ys）：当前区域包含的纵坐标序列。
      ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
      xs, ys = xs[mask], ys[mask]
      # 变量说明（i）：当前序列或网格遍历使用的索引。
      i = random.randint(0, len(xs))
      # 变量说明（pos）：对象在世界网格中的二维坐标。
      pos = np.array((xs[i], ys[i]))
      # 变量说明（empty）：当前区域、集合或输入是否为空的布尔标记。
      empty = self._world[pos][1] is None
      # modified
      # 变量说明（away）：使生物远离危险或其他对象的方向向量。
      away = True
      # 变量说明（p）：当前循环正在处理的玩家、位置或概率值。
      for p in self._players:
        # 变量说明（_away）：使生物远离危险或其他对象的方向向量。
        _away = p.distance(pos) >= span_dist
        away = away and _away
      if empty and away:
        #print(ctor(pos))
        self._world.add(ctor(pos))
          
      #away = self._player.distance(pos) >= span_dist or self._player2.distance(pos) >= span_dist
      #if empty and away:
      #  self._world.add(ctor(pos))
    elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
      obj = creatures[random.randint(0, len(creatures))]
      #away = self._player.distance(obj.pos) >= despan_dist and self._player2.distance(obj.pos) >= despan_dist
      away = True
      for p in self._players:
        _away = p.distance(obj.pos) >= despan_dist
        away = away and _away
      if away:
        self._world.remove(obj)