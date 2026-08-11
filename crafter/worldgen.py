# 模块说明：根据噪声和地形规则生成 Crafter 世界。
import functools

import numpy as np
import opensimplex

from . import constants
from . import objects


def generate_world(world, player):
  # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
  # 变量说明（player）：当前处理的 Crafter 玩家对象。
  # 变量说明（simplex）：用于生成连续自然地形的 OpenSimplex 噪声生成器。
  simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
  # 变量说明（tunnels）：世界生成器已挖掘的通道坐标集合。
  tunnels = np.zeros(world.area, bool)
  # 变量说明（x）：世界网格或图像中的横向坐标。
  for x in range(world.area[0]):
    # 变量说明（y）：世界网格或图像中的纵向坐标。
    for y in range(world.area[1]):
      _set_material(world, (x, y), player, tunnels, simplex)
  for x in range(world.area[0]):
    for y in range(world.area[1]):
      _set_object(world, (x, y), player, tunnels)


def _set_material(world, pos, player, tunnels, simplex):
  # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
  # 变量说明（pos）：对象在世界网格中的二维坐标。
  # 变量说明（player）：当前处理的 Crafter 玩家对象。
  # 变量说明（tunnels）：世界生成器已挖掘的通道坐标集合。
  # 变量说明（simplex）：用于生成连续自然地形的 OpenSimplex 噪声生成器。
  # 变量说明（x）：世界网格或图像中的横向坐标。
  # 变量说明（y）：世界网格或图像中的纵向坐标。
  x, y = pos
  simplex = functools.partial(_simplex, simplex)
  # 变量说明（uniform）：随机采样得到的均匀分布数值或噪声数组。
  uniform = world.random.uniform
  # 变量说明（start）：搜索、切片或计时过程使用的起始位置。
  start = 4 - np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
  start += 2 * simplex(x, y, 8, 3)
  start = 1 / (1 + np.exp(-start))
  # 变量说明（water）：表示水域材质或取水目标的枚举成员。
  water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
  water -= 2 * start
  # 变量说明（mountain）：世界生成过程中用于塑造山地的噪声高度数组。
  mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
  mountain -= 4 * start + 0.3 * water
  if start > 0.5:
    world[x, y] = 'grass'
  elif mountain > 0.15:
    if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3):  # cave
      world[x, y] = 'path'
    elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnle
      world[x, y] = 'path'
      tunnels[x, y] = True
    elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnle
      world[x, y] = 'path'
      tunnels[x, y] = True
    elif simplex(x, y, 1, 8) > 0 and uniform() > 0.85:
      world[x, y] = 'coal'
    elif simplex(x, y, 2, 6) > 0.4 and uniform() > 0.75:
      world[x, y] = 'iron'
    elif mountain > 0.18 and uniform() > 0.994:
      world[x, y] = 'diamond'
    elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.35:
      world[x, y] = 'lava'
    else:
      world[x, y] = 'stone'
  elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2:
    world[x, y] = 'sand'
  elif 0.3 < water:
    world[x, y] = 'water'
  else:  # grassland
    if simplex(x, y, 5, 7) > 0 and uniform() > 0.8:
      world[x, y] = 'tree'
    else:
      world[x, y] = 'grass'


def _set_object(world, pos, player, tunnels):
  # 变量说明（world）：包含材质网格和世界对象的 Crafter 世界实例。
  # 变量说明（pos）：对象在世界网格中的二维坐标。
  # 变量说明（player）：当前处理的 Crafter 玩家对象。
  # 变量说明（tunnels）：世界生成器已挖掘的通道坐标集合。
  # 变量说明（x）：世界网格或图像中的横向坐标。
  # 变量说明（y）：世界网格或图像中的纵向坐标。
  x, y = pos
  # 变量说明（uniform）：随机采样得到的均匀分布数值或噪声数组。
  uniform = world.random.uniform
  # 变量说明（dist）：当前两个位置、对象或节点之间计算出的距离。
  dist = np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
  # 变量说明（material）：当前网格单元使用的地形或资源材质名称。
  material, _ = world[x, y]
  if material not in constants.walkable:
    pass
  elif dist > 3 and material == 'grass' and uniform() > 0.985:
    world.add(objects.Cow(world, (x, y)))
  #elif dist > 10 and uniform() > 0.993:
  #  world.add(objects.Zombie(world, (x, y), player))
  #elif material == 'path' and tunnels[x, y] and uniform() > 0.95:
  #  world.add(objects.Skeleton(world, (x, y), player))


def _simplex(simplex, x, y, z, sizes, normalize=True):
  # 变量说明（simplex）：用于生成连续自然地形的 OpenSimplex 噪声生成器。
  # 变量说明（x）：世界网格或图像中的横向坐标。
  # 变量说明（y）：世界网格或图像中的纵向坐标。
  # 变量说明（z）：当前噪声高度、深度坐标或第三维数值。
  # 变量说明（sizes）：待布局、缩放或比较对象的尺寸列表。
  # 变量说明（normalize）：包装器覆盖前保存的环境原始规范化方法。
  if not isinstance(sizes, dict):
    sizes = {sizes: 1}
  # 变量说明（value）：当前配置项、字段或计算结果对应的值。
  value = 0
  # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
  # 变量说明（weight）：当前奖励、图边、采样或图像混合使用的权重。
  for size, weight in sizes.items():
    if hasattr(simplex, 'noise3d'):
      # 变量说明（noise）：用于生成地形、光照或扰动效果的随机噪声数组。
      noise = simplex.noise3d(x / size, y / size, z)
    else:
      noise = simplex.noise3(x / size, y / size, z)
    value += weight * noise
  if normalize:
    value /= sum(sizes.values())
  return value
