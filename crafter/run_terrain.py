# 模块说明：生成并展示 Crafter 地形。
import argparse

import imageio
import numpy as np

import crafter


def main():
  # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--amount', type=int, default=4)
  parser.add_argument('--cols', type=int, default=4)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--size', type=int, default=1024)
  parser.add_argument('--filename', type=str, default='terrain.png')
  # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
  args = parser.parse_args()

  # 变量说明（env）：当前 Crafter 仿真环境实例。
  env = crafter.Env(args.area, args.area, args.size, seed=args.seed)
  # 变量说明（images）：等待组合或处理的图像集合。
  images = []
  # 变量说明（index）：当前元素在序列中的零基位置。
  for index in range(args.amount):
    images.append(env.reset())
    # 变量说明（diamonds）：世界生成阶段选出的钻石矿点坐标集合。
    diamonds = env._world.count('diamond')
    print(f'Map: {index:>2}, diamonds: {diamonds:>2}')

  # 变量说明（rows）：`main` 中待批量处理的行数集合。
  rows = len(images) // args.cols
  # 变量说明（strips）：从纹理图集中切分得到的精灵条带列表。
  strips = []
  # 变量说明（row）：`main` 中当前处理的行数。
  for row in range(rows):
    # 变量说明（strip）：从纹理图集中切出的单行或单列精灵图像。
    strip = []
    # 变量说明（col）：当前处理的网格列索引。
    for col in range(args.cols):
      try:
        strip.append(images[row * args.cols + col])
      except IndexError:
        strip.append(np.zeros_like(strip[-1]))
    strips.append(np.concatenate(strip, 1))
  # 变量说明（grid）：按世界坐标存储材质或寻路状态的二维网格。
  grid = np.concatenate(strips, 0)

  imageio.imsave(args.filename, grid)
  print('Saved', args.filename)


if __name__ == '__main__':
  main()
