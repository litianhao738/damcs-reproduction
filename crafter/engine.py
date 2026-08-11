# 模块说明：实现 Crafter 世界底层网格、材质、对象管理与渲染。
import collections
import functools
import pathlib

import imageio.v3 as imageio
import numpy as np
from PIL import Image, ImageEnhance


class AttrDict(dict):

  # 变量说明（__getattr__）：包装器覆盖属性查找逻辑前保存的环境原始 __getattr__ 方法。
  __getattr__ = dict.__getitem__


class staticproperty:

  def __init__(self, function):
    # 变量说明（function）：命令行或运行器当前选择执行的函数名称。
    # 变量说明（self.function）：命令行或运行器当前选择执行的函数名称。
    self.function = function

  def __get__(self, instance, owner=None):
    # 变量说明（instance）：当前包装器或装饰器创建的目标类实例。
    # 变量说明（owner）：当前被记录器包装、实际持有原始方法的环境对象。
    return self.function()


class World:

  def __init__(self, area, materials, chunk_size):
    # 变量说明（area）：生成世界时使用的二维地图尺寸。
    # 变量说明（materials）：Crafter 支持的全部地形和资源材质名称列表。
    # 变量说明（chunk_size）：批量统计或数组处理时每个数据块包含的元素数量。
    # 变量说明（self.area）：生成世界时使用的二维地图尺寸。
    self.area = area
    # 变量说明（self._chunk_size）：批量统计或数组处理时每个数据块包含的元素数量。
    self._chunk_size = chunk_size
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    # 变量说明（self._mat_names）：与材质整数编号一一对应的材质名称列表。
    # 变量说明（x）：世界网格或图像中的横向坐标。
    self._mat_names = {i: x for i, x in enumerate([None] + materials)}
    # 变量说明（self._mat_ids）：当前地图区域内出现的材质整数编号数组。
    self._mat_ids = {x: i for i, x in enumerate([None] + materials)}
    self.reset()

  def reset(self, seed=None):
    # 变量说明（seed）：控制世界生成和随机事件、保证结果可复现的随机种子。
    # 变量说明（self.random）：控制环境包装器是否使用随机动作的布尔参数。
    self.random = np.random.RandomState(seed)
    # 变量说明（self.daylight）：当前时间步的环境光照强度，取值通常位于 0 到 1。
    self.daylight = 0.0
    # 变量说明（self._chunks）：按固定长度拆分后的数据块列表。
    self._chunks = collections.defaultdict(set)
    # 变量说明（self._objects）：当前世界、视图或查询范围内的对象集合。
    self._objects = [None]
    # 变量说明（self._mat_map）：`reset` 中当前处理的地图。
    self._mat_map = np.zeros(self.area, np.uint8)
    # 变量说明（self._obj_map）：`reset` 中当前处理的地图。
    self._obj_map = np.zeros(self.area, np.uint32)

  @property
  def objects(self):
    # Return a new list so the objects cannot change while being iterated over.
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    return [obj for obj in self._objects if obj]

  @property
  def chunks(self):
    return self._chunks.copy()

  def add(self, obj):
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    assert hasattr(obj, 'pos')
    obj.pos = np.array(obj.pos)
    assert self._obj_map[tuple(obj.pos)] == 0
    # 变量说明（index）：当前元素在序列中的零基位置。
    index = len(self._objects)
    self._objects.append(obj)
    self._obj_map[tuple(obj.pos)] = index
    self._chunks[self.chunk_key(obj.pos)].add(obj)

  def remove(self, obj):
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    if obj.removed:
      return
    self._objects[self._obj_map[tuple(obj.pos)]] = None
    self._obj_map[tuple(obj.pos)] = 0
    self._chunks[self.chunk_key(obj.pos)].remove(obj)
    obj.removed = True

  def move(self, obj, pos):
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    if obj.removed:
      return
    pos = np.array(pos)
    assert self._obj_map[tuple(pos)] == 0
    # 变量说明（index）：当前元素在序列中的零基位置。
    index = self._obj_map[tuple(obj.pos)]
    self._obj_map[tuple(pos)] = index
    self._obj_map[tuple(obj.pos)] = 0
    # 变量说明（old_chunk）：加入当前值之前保留的上一数据块。
    old_chunk = self.chunk_key(obj.pos)
    # 变量说明（new_chunk）：将当前值加入后形成的新数据块。
    new_chunk = self.chunk_key(pos)
    if old_chunk != new_chunk:
      self._chunks[old_chunk].remove(obj)
      self._chunks[new_chunk].add(obj)
    obj.pos = pos

  def __setitem__(self, pos, material):
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    if material not in self._mat_ids:
      # 变量说明（id_）：`__setitem__` 中当前处理的编号。
      id_ = len(self._mat_ids)
      self._mat_ids[material] = id_
    self._mat_map[tuple(pos)] = self._mat_ids[material]

  def __getitem__(self, pos):
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    if not _inside((0, 0), pos, self.area):
      return None, None
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    material = self._mat_names[self._mat_map[tuple(pos)]]
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    obj = self._objects[self._obj_map[tuple(pos)]]
    return material, obj

  def nearby(self, pos, distance):
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    # 变量说明（distance）：当前两个位置、对象或节点之间计算出的距离。
    # 变量说明（d）：当前循环使用的方向、距离或字典数据项。
    # 变量说明（x）：世界网格或图像中的横向坐标。
    # 变量说明（y）：世界网格或图像中的纵向坐标。
    (x, y), d = pos, distance
    # 变量说明（ids）：当前材质或对象集合对应的整数语义编号数组。
    ids = set(self._mat_map[
        x - d: x + d + 1, y - d: y + d + 1].flatten().tolist())
    # 变量说明（materials）：Crafter 支持的全部地形和资源材质名称列表。
    materials = tuple(self._mat_names[x] for x in ids)
    # 变量说明（indices）：当前筛选、排序或采样得到的元素索引列表。
    indices = self._obj_map[
        x - d: x + d + 1, y - d: y + d + 1].flatten().tolist()
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    # 变量说明（objs）：当前网格区域内待处理的世界对象集合。
    objs = {self._objects[i] for i in indices if i > 0}
    return materials, objs

  def mask(self, xmin, xmax, ymin, ymax, material):
    # 变量说明（xmin）：当前矩形或地图区域包含的最小横坐标。
    # 变量说明（xmax）：当前矩形或地图区域包含的最大横坐标。
    # 变量说明（ymin）：当前矩形或地图区域包含的最小纵坐标。
    # 变量说明（ymax）：当前矩形或地图区域包含的最大纵坐标。
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    # 变量说明（region）：当前读取、填充或渲染的二维矩形区域。
    region = self._mat_map[xmin: xmax, ymin: ymax]
    return (region == self._mat_ids[material])

  def count(self, material):
    # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
    return (self._mat_map == self._mat_ids[material]).sum()

  def chunk_key(self, pos):
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    # 变量说明（csx）：纹理图集中单个精灵切片的横向像素尺寸。
    # 变量说明（csy）：纹理图集中单个精灵切片的纵向像素尺寸。
    # 变量说明（x）：世界网格或图像中的横向坐标。
    # 变量说明（y）：世界网格或图像中的纵向坐标。
    (x, y), (csx, csy) = pos, self._chunk_size
    # 变量说明（xmin）：当前矩形或地图区域包含的最小横坐标。
    # 变量说明（ymin）：当前矩形或地图区域包含的最小纵坐标。
    xmin, ymin = (x // csx) * csx, (y // csy) * csy
    # 变量说明（xmax）：当前矩形或地图区域包含的最大横坐标。
    xmax = min(xmin + csx, self.area[0])
    # 变量说明（ymax）：当前矩形或地图区域包含的最大纵坐标。
    ymax = min(ymin + csy, self.area[1])
    return (xmin, xmax, ymin, ymax)


class Textures:

  def __init__(self, directory):
    # 变量说明（directory）：当前函数准备读取或写入文件的目录路径。
    # 变量说明（self._originals）：包装器替换方法前保存的环境原始方法字典。
    self._originals = {}
    # 变量说明（self._textures）：从素材目录加载并按名称索引的纹理图像集合。
    self._textures = {}
    # 变量说明（filename）：当前读取、保存或展示的文件名称。
    for filename in pathlib.Path(directory).glob('*.png'):
      # 变量说明（image）：当前读取、处理或渲染的图像数组。
      image = imageio.imread(filename.read_bytes())
      image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
      self._originals[filename.stem] = image
      self._textures[(filename.stem, image.shape[:2])] = image

  def get(self, name, size):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
    if name is None:
      name = 'unknown'
    size = int(size[0]), int(size[1])
    # 变量说明（key）：当前字典条目的键。
    key = name, size
    if key not in self._textures:
      # 变量说明（image）：当前读取、处理或渲染的图像数组。
      image = self._originals[name]
      image = Image.fromarray(image)
      image = image.resize(size[::-1], resample=Image.NEAREST)
      image = np.array(image)
      self._textures[key] = image
    return self._textures[key]


class GlobalView:

  pass


class UncoverView:

  pass


class LocalView:

  def __init__(self, world, textures, grid):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（textures）：从素材目录加载并按名称索引的纹理图像集合。
    # 变量说明（grid）：按世界坐标存储材质或寻路状态的二维网格。
    # 变量说明（self._world）：包含材质网格和世界对象的 Crafter 世界实例。
    self._world = world
    self._textures = textures
    # 变量说明（self._grid）：按世界坐标存储材质或寻路状态的二维网格。
    self._grid = np.array(grid)
    # 变量说明（self._offset）：当前坐标、切片或布局相对起点的偏移量。
    self._offset = self._grid // 2
    # 变量说明（self._area）：生成世界时使用的二维地图尺寸。
    self._area = np.array(self._world.area)
    # 变量说明（self._center）：当前对象、视图或地图区域的中心坐标。
    self._center = None

  def __call__(self, player, unit):
    # 变量说明（player）：当前处理的 Crafter 玩家对象。
    # 变量说明（unit）：世界坐标映射到渲染画布时每个网格单元的像素尺寸。
    # 变量说明（self._unit）：世界坐标映射到渲染画布时每个网格单元的像素尺寸。
    self._unit = np.array(unit)
    self._center = np.array(player.pos)
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    canvas = np.zeros(tuple(self._grid * unit) + (3,), np.uint8) + 127
    # 变量说明（x）：世界网格或图像中的横向坐标。
    for x in range(self._grid[0]):
      # 变量说明（y）：世界网格或图像中的纵向坐标。
      for y in range(self._grid[1]):
        # 变量说明（pos）：对象在世界网格中的二维坐标。
        pos = self._center + np.array([x, y]) - self._offset
        if not _inside((0, 0), pos, self._area):
          continue
        # 变量说明（texture）：`__call__` 中当前处理的纹理。
        texture = self._textures.get(self._world[pos][0], unit)
        _draw(canvas, np.array([x, y]) * unit, texture)
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    for obj in self._world.objects:
      pos = obj.pos - self._center + self._offset
      if not _inside((0, 0), pos, self._grid):
        continue
      texture = self._textures.get(obj.texture, unit)
      _draw_alpha(canvas, pos * unit, texture)
    canvas = self._light(canvas, self._world.daylight)
    if player.sleeping:
      canvas = self._sleep(canvas)
    # if player.health < 1:
    #   canvas = self._tint(canvas, (128, 0, 0), 0.6)
    return canvas

  def _light(self, canvas, daylight):
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    # 变量说明（daylight）：当前时间步的环境光照强度，取值通常位于 0 到 1。
    # 变量说明（night）：当前世界是否处于夜晚阶段的布尔标记。
    night = canvas
    if daylight < 0.5:
      night = self._noise(night, 2 * (0.5 - daylight), 0.5)
    night = np.array(ImageEnhance.Color(
        Image.fromarray(night.astype(np.uint8))).enhance(0.4))
    night = self._tint(night, (0, 16, 64), 0.5)
    return daylight * canvas + (1 - daylight) * night

  def _sleep(self, canvas):
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    canvas = np.array(ImageEnhance.Color(
        Image.fromarray(canvas.astype(np.uint8))).enhance(0.0))
    canvas = self._tint(canvas, (0, 0, 16), 0.5)
    return canvas

  def _tint(self, canvas, color, amount):
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    # 变量说明（color）：`_tint` 中当前处理的颜色。
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    color = np.array(color)
    return (1 - amount) * canvas + amount * color

  def _noise(self, canvas, amount, stddev):
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    # 变量说明（stddev）：当前数组或噪声分布使用的标准差。
    # 变量说明（noise）：用于生成地形、光照或扰动效果的随机噪声数组。
    noise = self._world.random.uniform(32, 127, canvas.shape[:2])[..., None]
    # 变量说明（mask）：`_noise` 中当前处理的掩码。
    mask = amount * self._vignette(canvas.shape, stddev)[..., None]
    return (1 - mask) * canvas + mask * noise

  @functools.lru_cache(10)
  def _vignette(self, shape, stddev):
    # 变量说明（shape）：当前数组、纹理或世界区域各维度的长度。
    # 变量说明（stddev）：当前数组或噪声分布使用的标准差。
    # 变量说明（xs）：当前区域包含的横坐标序列。
    # 变量说明（ys）：当前区域包含的纵坐标序列。
    xs, ys = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]))
    return 1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / (stddev ** 2)).T


class ItemView:

  def __init__(self, textures, grid):
    # 变量说明（textures）：从素材目录加载并按名称索引的纹理图像集合。
    # 变量说明（grid）：按世界坐标存储材质或寻路状态的二维网格。
    self._textures = textures
    self._grid = np.array(grid)

  def __call__(self, inventory, unit):
    # 变量说明（inventory）：当前智能体背包内各类物品及数量。
    # 变量说明（unit）：世界坐标映射到渲染画布时每个网格单元的像素尺寸。
    unit = np.array(unit)
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    canvas = np.zeros(tuple(self._grid * unit) + (3,), np.uint8)
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    # 变量说明（index）：当前元素在序列中的零基位置。
    # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
    for index, (item, amount) in enumerate(inventory.items()):
      if amount < 1:
        continue
      self._item(canvas, index, item, unit)
      self._amount(canvas, index, amount, unit)
    return canvas

  def _item(self, canvas, index, item, unit):
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    # 变量说明（index）：当前元素在序列中的零基位置。
    # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
    # 变量说明（unit）：世界坐标映射到渲染画布时每个网格单元的像素尺寸。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * unit + 0.1 * unit).astype(np.int32)
    # 变量说明（texture）：`_item` 中当前处理的纹理。
    texture = self._textures.get(item, 0.8 * unit)
    _draw_alpha(canvas, pos, texture)

  def _amount(self, canvas, index, amount, unit):
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    # 变量说明（index）：当前元素在序列中的零基位置。
    # 变量说明（amount）：本次增加、扣除、交换或绘制的数据数量。
    # 变量说明（unit）：世界坐标映射到渲染画布时每个网格单元的像素尺寸。
    # 变量说明（pos）：对象在世界网格中的二维坐标。
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * unit + 0.4 * unit).astype(np.int32)
    # 变量说明（text）：`_amount` 中当前处理的文本。
    text = str(amount) if amount in list(range(10)) else 'unknown'
    # 变量说明（texture）：`_amount` 中当前处理的纹理。
    texture = self._textures.get(text, 0.6 * unit)
    _draw_alpha(canvas, pos, texture)


class SemanticView:

  def __init__(self, world, obj_types):
    # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
    # 变量说明（obj_types）：语义视图需要识别的世界对象类型列表。
    self._world = world
    self._mat_ids = world._mat_ids.copy()
    # 变量说明（c）：当前循环正在处理的颜色通道、列或计数值。
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    # 变量说明（self._obj_ids）：语义视图内每种对象类型对应的整数编号映射。
    self._obj_ids = {
        c: len(self._mat_ids) + i
        for i, c in enumerate(obj_types)}

  def __call__(self):
    # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
    canvas = self._world._mat_map.copy()
    # 变量说明（obj）：当前遍历、渲染或交互的世界对象。
    for obj in self._world.objects:
      canvas[tuple(obj.pos)] = self._obj_ids[type(obj)]
    return canvas


def _inside(lhs, mid, rhs):
  # 变量说明（lhs）：图像混合或二元运算左侧的输入数组。
  # 变量说明（mid）：当前区间或区域的中点坐标。
  # 变量说明（rhs）：图像混合或二元运算右侧的输入数组。
  return (lhs[0] <= mid[0] < rhs[0]) and (lhs[1] <= mid[1] < rhs[1])

def _draw(canvas, pos, texture):
  # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
  # 变量说明（pos）：对象在世界网格中的二维坐标。
  # 变量说明（texture）：`_draw` 中当前处理的纹理。
  # 变量说明（h）：当前区域、图像或纹理的像素高度。
  # 变量说明（w）：当前区域、图像或纹理的像素宽度。
  # 变量说明（x）：世界网格或图像中的横向坐标。
  # 变量说明（y）：世界网格或图像中的纵向坐标。
  (x, y), (w, h) = pos, texture.shape[:2]
  if texture.shape[-1] == 4:
    texture = texture[..., :3]
  canvas[x: x + w, y: y + h] = texture

def _draw_alpha(canvas, pos, texture):
  # 变量说明（canvas）：用于绘制当前世界视图的像素画布。
  # 变量说明（pos）：对象在世界网格中的二维坐标。
  # 变量说明（texture）：`_draw_alpha` 中当前处理的纹理。
  # 变量说明（h）：当前区域、图像或纹理的像素高度。
  # 变量说明（w）：当前区域、图像或纹理的像素宽度。
  # 变量说明（x）：世界网格或图像中的横向坐标。
  # 变量说明（y）：世界网格或图像中的纵向坐标。
  (x, y), (w, h) = pos, texture.shape[:2]
  if texture.shape[-1] == 4:
    # 变量说明（alpha）：图像混合、颜色透明度或指数平滑使用的权重。
    alpha = texture[..., 3:].astype(np.float32) / 255
    texture = texture[..., :3].astype(np.float32) / 255
    # 变量说明（current）：迭代算法当前正在处理的节点、位置或状态。
    current = canvas[x: x + w, y: y + h].astype(np.float32) / 255
    # 变量说明（blended）：前景与背景按透明度混合后的图像数组。
    blended = alpha * texture + (1 - alpha) * current
    texture = (255 * blended).astype(np.uint8)
  canvas[x: x + w, y: y + h] = texture
