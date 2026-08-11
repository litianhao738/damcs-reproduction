# 模块说明：定义玩家、生物、投射物和可交互世界对象的行为。
import numpy as np

from . import constants
from . import engine


class Object:

  def __init__(self, world, pos):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    # 变量说明（self.world）：包含材质网格和世界对象的 Crafter 世界实例。
    self.world = world
    # 变量说明（self.pos）：对象在世界网格中的二维坐标。
    self.pos = np.array(pos)
    # 变量说明（self.random）：控制环境包装器是否使用随机动作的布尔参数。
    self.random = world.random
    # 变量说明（self.inventory）：当前智能体背包内各类物品及数量。
    self.inventory = {'health': 0}
    # 变量说明（self.removed）：当前时间步从世界中移除的对象列表。
    self.removed = False

  @property
  def texture(self):
    raise 'unknown'

  @property
  def walkable(self):
    return constants.walkable

  @property
  def health(self):
    return self.inventory['health']

  @health.setter
  def health(self, value):
    # 变量说明（value）：当前配置项、字段或计算结果对应的值。
    self.inventory['health'] = max(0, value)

  @property
  def all_dirs(self):
    return ((-1, 0), (+1, 0), (0, -1), (0, +1))

  def move(self, direction):
    # 变量说明（direction）：对象当前朝向或移动方向向量。
    direction = np.array(direction)
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    target = self.pos + direction
    if self.is_free(target):
      self.world.move(self, target)
      return True
    return False

  def is_free(self, target, materials=None):
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    # 变量说明（materials）：Crafter 支持的全部地形和资源材质名称列表。
    materials = self.walkable if materials is None else materials
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    material, obj = self.world[target]
    return obj is None and material in materials

  def distance(self, target):
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    if hasattr(target, 'pos'):
      target = target.pos
    return np.abs(target - self.pos).sum()

  def toward(self, target, long_axis=True):
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    # 变量说明（long_axis）：当前矩形区域较长边对应的维度编号。
    if hasattr(target, 'pos'):
      target = target.pos
    # 变量说明（offset）：当前坐标、切片或布局相对起点的偏移量。
    offset = target - self.pos
    # 变量说明（dists）：候选位置到目标位置的距离数组。
    dists = np.abs(offset)
    if (dists[0] > dists[1] if long_axis else dists[0] <= dists[1]):
      return np.array((np.sign(offset[0]), 0))
    else:
      return np.array((0, np.sign(offset[1])))

  def random_dir(self):
    return self.all_dirs[self.random.randint(0, 4)]


class Player(Object):

  def __init__(self, world, pos):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    super().__init__(world, pos)
    # 变量说明（self.facing）：玩家当前面向的方向或面前物体名称。
    self.facing = (0, 1)
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    # 变量说明（name）：当前对象、指标或配置项的名称。
    self.inventory = {
        name: info['initial'] for name, info in constants.items.items()}
    # 变量说明（self.achievements）：各类游戏成就名称到累计完成次数的映射。
    self.achievements = {name: 0 for name in constants.achievements}
    # 变量说明（self.action）：当前智能体准备执行的环境动作。
    self.action = 'noop'
    # 变量说明（self.sleeping）：玩家当前是否处于睡眠状态的布尔标记。
    self.sleeping = False
    # 变量说明（self._last_health）：`__init__` 中当前处理的生命值。
    self._last_health = self.health
    # 变量说明（self._hunger）：玩家当前食物饱腹值。
    self._hunger = 0
    # 变量说明（self._thirst）：玩家当前饮水值。
    self._thirst = 0
    # 变量说明（self._fatigue）：玩家因移动或操作累计的能量消耗值。
    self._fatigue = 0
    # 变量说明（self._recover）：玩家当前时间步恢复的生命、食物或能量数量。
    self._recover = 0

  @property
  def texture(self):
    if self.sleeping:
      return 'player-sleep'
    return {
        (-1, 0): 'player-left',
        (+1, 0): 'player-right',
        (0, -1): 'player-up',
        (0, +1): 'player-down',
    }[tuple(self.facing)]

  @property
  def walkable(self):
    return constants.walkable + ['lava']

  def update(self):
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    target = (self.pos[0] + self.facing[0], self.pos[1] + self.facing[1])
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    material, obj = self.world[target]
    # 变量说明（action）：当前智能体准备执行的环境动作。
    action = self.action
    if self.sleeping:
      if self.inventory['energy'] < constants.items['energy']['max']:
        action = 'sleep'
      else:
        self.sleeping = False
        self.achievements['wake_up'] += 1
    if action == 'noop':
      pass
    elif action.startswith('move_'):
      self._move(action[len('move_'):])
    elif action == 'do' and obj:
      self._do_object(obj)
    elif action == 'do':
      self._do_material(target, material)
    elif action == 'sleep':
      if self.inventory['energy'] < constants.items['energy']['max']:
        self.sleeping = True
    elif action.startswith('place_'):
      self._place(action[len('place_'):], target, material)
    elif action.startswith('make_'):
      self._make(action[len('make_'):])
    self._update_life_stats()
    self._degen_or_regen_health()
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    # 变量说明（name）：当前对象、指标或配置项的名称。
    for name, amount in self.inventory.items():
      # 变量说明（maxmium）：原代码拼写保留；当前缩放或统计计算允许的最大值。
      maxmium = constants.items[name]['max']
      self.inventory[name] = max(0, min(amount, maxmium))
    # This needs to happen after the inventory states are clamped
    # because it involves the health water inventory count.
    self._wake_up_when_hurt()

  def _update_life_stats(self):
    self._hunger += 0.5 if self.sleeping else 1
    if self._hunger > 25:
      self._hunger = 0
      self.inventory['food'] -= 1
    self._thirst += 0.5 if self.sleeping else 1
    if self._thirst > 20:
      self._thirst = 0
      self.inventory['drink'] -= 1
    if self.sleeping:
      self._fatigue = min(self._fatigue - 1, 0)
    else:
      self._fatigue += 1
    if self._fatigue < -10:
      self._fatigue = 0
      self.inventory['energy'] += 1
    if self._fatigue > 30:
      self._fatigue = 0
      self.inventory['energy'] -= 1

  def _degen_or_regen_health(self):
    # 变量说明（necessities）：完成当前制作或放置动作所需的材料数量映射。
    necessities = (
        self.inventory['food'] > 0,
        self.inventory['drink'] > 0,
        self.inventory['energy'] > 0 or self.sleeping)
    if all(necessities):
      self._recover += 2 if self.sleeping else 1
    else:
      self._recover -= 0.5 if self.sleeping else 1
    if self._recover > 25:
      self._recover = 0
      # 变量说明（self.health）：`_degen_or_regen_health` 中当前处理的生命值。
      self.health += 1
    if self._recover < -15:
      self._recover = 0
      self.health -= 1

  def _wake_up_when_hurt(self):
    if self.health < self._last_health:
      self.sleeping = False
    self._last_health = self.health

  def _move(self, direction):
    # 变量说明（direction）：对象当前朝向或移动方向向量。
    # 变量说明（directions）：`_move` 中待批量处理的方向集合。
    directions = dict(left=(-1, 0), right=(+1, 0), up=(0, -1), down=(0, +1))
    self.facing = directions[direction]
    self.move(self.facing)
    if self.world[self.pos][0] == 'lava':
      self.health = 0

  def _do_object(self, obj):
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    # 变量说明（damage）：本次攻击或环境事件造成的生命值扣减量。
    damage = max([
        1,
        self.inventory['wood_sword'] and 2,
        self.inventory['stone_sword'] and 3,
        self.inventory['iron_sword'] and 5,
    ])
    if isinstance(obj, Plant):
      if obj.ripe:
        obj.grown = 0
        self.inventory['food'] += 4
        self.achievements['eat_plant'] += 1
    if isinstance(obj, Fence):
      self.world.remove(obj)
      self.inventory['fence'] += 1
      self.achievements['collect_fence'] += 1
    if isinstance(obj, Zombie):
      obj.health -= damage
      if obj.health <= 0:
        self.achievements['defeat_zombie'] += 1
    if isinstance(obj, Skeleton):
      obj.health -= damage
      if obj.health <= 0:
        self.achievements['defeat_skeleton'] += 1
    if isinstance(obj, Cow):
      obj.health -= damage
      if obj.health <= 0:
        self.inventory['food'] += 6
        self.achievements['eat_cow'] += 1
        # TODO: Keep track of previous inventory state to do this in a more
        # general way.
        self._hunger = 0

  def _do_material(self, target, material):
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    if material == 'water':
      # TODO: Keep track of previous inventory state to do this in a more
      # general way.
      self._thirst = 0
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    info = constants.collect.get(material)
    if not info:
      return
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    # 变量说明（name）：当前对象、指标或配置项的名称。
    for name, amount in info['require'].items():
      if self.inventory[name] < amount:
        return
    self.world[target] = info['leaves']
    if self.random.uniform() <= info.get('probability', 1):
      for name, amount in info['receive'].items():
        self.inventory[name] += amount
        self.achievements[f'collect_{name}'] += 1

  def _place(self, name, target, material):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    if self.world[target][1]:
      return
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    info = constants.place[name]
    if material not in info['where']:
      return
    # 变量说明（k）：限制检索、回看或邻居搜索数量的整数参数。
    # 变量说明（v）：当前图边的终点节点，或数组计算中的第二个临时值。
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    if info['type'] == 'material':
      self.world[target] = name
    elif info['type'] == 'object':
      cls = {
          'fence': Fence,
          'plant': Plant,
      }[name]
      self.world.add(cls(self.world, target))
    self.achievements[f'place_{name}'] += 1

  def _make(self, name):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    # 变量说明（nearby）：玩家局部范围内满足距离条件的世界对象集合。
    nearby, _ = self.world.nearby(self.pos, 1)
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    info = constants.make[name]
    # 变量说明（util）：当前合成配方执行后需要恢复或扣除的生命状态数量映射。
    if not all(util in nearby for util in info['nearby']):
      return
    # 变量说明（k）：限制检索、回看或邻居搜索数量的整数参数。
    # 变量说明（v）：当前图边的终点节点，或数组计算中的第二个临时值。
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    self.inventory[name] += info['gives']
    self.achievements[f'make_{name}'] += 1


class Cow(Object):

  def __init__(self, world, pos):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    super().__init__(world, pos)
    self.health = 3

  @property
  def texture(self):
    return 'cow'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    if self.random.uniform() < 0.5:
      # 变量说明（direction）：对象当前朝向或移动方向向量。
      direction = self.random_dir()
      self.move(direction)


class Zombie(Object):

  def __init__(self, world, pos, player):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    # 变量说明（player）：当前处理的 Crafter 玩家对象。
    super().__init__(world, pos)
    # 变量说明（self.player）：当前处理的 Crafter 玩家对象。
    self.player = player
    self.health = 5
    # 变量说明（self.cooldown）：对象再次攻击、移动或生成前需要等待的时间步数。
    self.cooldown = 0

  @property
  def texture(self):
    return 'zombie'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    # 变量说明（dist）：当前两个位置、对象或节点之间计算出的距离。
    dist = self.distance(self.player)
    if dist <= 8 and self.random.uniform() < 0.9:
      self.move(self.toward(self.player, self.random.uniform() < 0.8))
    else:
      self.move(self.random_dir())
    dist = self.distance(self.player)
    if dist <= 1:
      if self.cooldown:
        self.cooldown -= 1
      else:
        if self.player.sleeping:
          # 变量说明（damage）：本次攻击或环境事件造成的生命值扣减量。
          damage = 7
        else:
          damage = 2
        self.player.health -= damage
        self.cooldown = 5


class Skeleton(Object):

  def __init__(self, world, pos, player):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    # 变量说明（player）：当前处理的 Crafter 玩家对象。
    super().__init__(world, pos)
    self.player = player
    self.health = 3
    # 变量说明（self.reload）：投射物或生物距离下次攻击还需等待的时间步数。
    self.reload = 0

  @property
  def texture(self):
    return 'skeleton'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    self.reload = max(0, self.reload - 1)
    # 变量说明（dist）：当前两个位置、对象或节点之间计算出的距离。
    dist = self.distance(self.player.pos)
    if dist <= 3:
      # 变量说明（moved）：玩家在当前时间步是否成功改变位置的布尔标记。
      moved = self.move(-self.toward(self.player, self.random.uniform() < 0.6))
      if moved:
        return
    if dist <= 5 and self.random.uniform() < 0.5:
      self._shoot(self.toward(self.player))
    elif dist <= 8 and self.random.uniform() < 0.3:
      self.move(self.toward(self.player, self.random.uniform() < 0.6))
    elif self.random.uniform() < 0.2:
      self.move(self.random_dir())

  def _shoot(self, direction):
    # 变量说明（direction）：对象当前朝向或移动方向向量。
    if self.reload > 0:
      return
    if direction[0] == 0 and direction[1] == 0:
      return
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    pos = self.pos + direction
    if self.is_free(pos, Arrow.walkable):
      self.world.add(Arrow(self.world, pos, direction))
      self.reload = 4


class Arrow(Object):

  def __init__(self, world, pos, facing):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    # 变量说明（facing）：玩家当前面向的方向或面前物体名称。
    super().__init__(world, pos)
    self.facing = facing

  @property
  def texture(self):
    return {
        (-1, 0): 'arrow-left',
        (+1, 0): 'arrow-right',
        (0, -1): 'arrow-up',
        (0, +1): 'arrow-down',
    }[tuple(self.facing)]

  @engine.staticproperty
  def walkable():
    return constants.walkable + ['water', 'lava']

  def update(self):
    # 变量说明（target）：当前搜索、移动或计算操作的目标对象或坐标。
    target = self.pos + self.facing
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    material, obj = self.world[target]
    if obj:
      obj.health -= 2
      self.world.remove(self)
    elif material not in self.walkable:
      self.world.remove(self)
      if material in ['table', 'furnace']:
        self.world[target] = 'path'
    else:
      self.move(self.facing)


class Plant(Object):

  def __init__(self, world, pos):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    super().__init__(world, pos)
    self.health = 1
    # 变量说明（self.grown）：植物是否已经成熟、可以采集的布尔状态。
    self.grown = 0

  @property
  def texture(self):
    if self.ripe:
      return 'plant-ripe'
    else:
      return 'plant'

  @property
  def ripe(self):
    return self.grown > 300

  def update(self):
    self.grown += 1
    # 变量说明（dir_）：`update` 中当前处理的目录。
    # 变量说明（objs）：当前网格区域内待处理的世界对象集合。
    objs = [self.world[self.pos + dir_][1] for dir_ in self.all_dirs]
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    if any(isinstance(obj, (Zombie, Skeleton, Cow)) for obj in objs):
      self.health -= 1
    if self.health <= 0:
      self.world.remove(self)


class Fence(Object):
  def __init__(self, world, pos):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    super().__init__(world, pos)

  @property
  def texture(self):
    return 'fence'

  def update(self):
    pass