# 模块说明：启动单智能体 Crafter 图形界面。
import argparse

import numpy as np
try:
  import pygame
except ImportError:
  print('Please install the pygame package to use the GUI.')
  raise
from PIL import Image

import crafter


def main():
  # 变量说明（boolean）：命令行文本转换得到的布尔值。
  # 变量说明（x）：世界网格或图像中的横向坐标。
  boolean = lambda x: bool(['False', 'True'].index(x))
  # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
  parser.add_argument('--length', type=int, default=None)
  parser.add_argument('--health', type=int, default=9)
  parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
  parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
  parser.add_argument('--record', type=str, default=None)
  parser.add_argument('--fps', type=int, default=5)
  parser.add_argument('--wait', type=boolean, default=False)
  parser.add_argument('--death', type=str, default='reset', choices=[
      'continue', 'reset', 'quit'])
  # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
  args = parser.parse_args()

  # 变量说明（keymap）：键盘按键到 Crafter 动作编号的映射。
  keymap = {
      pygame.K_a: 'move_left',
      pygame.K_d: 'move_right',
      pygame.K_w: 'move_up',
      pygame.K_s: 'move_down',
      pygame.K_SPACE: 'do',
      pygame.K_TAB: 'sleep',

      pygame.K_r: 'place_stone',
      pygame.K_t: 'place_table',
      pygame.K_f: 'place_furnace',
      pygame.K_p: 'place_plant',

      pygame.K_1: 'make_wood_pickaxe',
      pygame.K_2: 'make_stone_pickaxe',
      pygame.K_3: 'make_iron_pickaxe',
      pygame.K_4: 'make_wood_sword',
      pygame.K_5: 'make_stone_sword',
      pygame.K_6: 'make_iron_sword',
  }
  print('Actions:')
  # 变量说明（action）：当前智能体准备执行的环境动作。
  # 变量说明（key）：当前字典条目的键。
  for key, action in keymap.items():
    print(f'  {pygame.key.name(key)}: {action}')

  crafter.constants.items['health']['max'] = args.health
  crafter.constants.items['health']['initial'] = args.health

  # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
  size = list(args.size)
  size[0] = size[0] or args.window[0]
  size[1] = size[1] or args.window[1]

  # 变量说明（env）：当前 Crafter 仿真环境实例。
  env = crafter.Env(
      area=args.area, view=args.view, length=args.length, seed=args.seed)
  env = crafter.Recorder(env, args.record)
  env.reset()
  # 变量说明（achievements）：各类游戏成就名称到累计完成次数的映射。
  achievements = set()
  # 变量说明（duration）：当前动作、动画或视频片段持续的时间或帧数。
  duration = 0
  # 变量说明（return_）：是否让当前辅助函数返回生成结果的布尔参数。
  return_ = 0
  # 变量说明（was_done）：执行当前包装器逻辑前，环境是否已经结束的状态快照。
  was_done = False
  print('Diamonds exist:', env._world.count('diamond'))

  pygame.init()
  # 变量说明（screen）：Pygame 创建的主显示窗口表面。
  screen = pygame.display.set_mode(args.window)
  # 变量说明（clock）：控制 GUI 刷新频率的 Pygame 时钟。
  clock = pygame.time.Clock()
  # 变量说明（running）：控制 GUI 主循环是否继续运行的布尔标记。
  running = True
  while running:

    # Rendering.
    # 变量说明（image）：当前读取、处理或渲染的图像数组。
    image = env.render(size)
    if size != args.window:
      image = Image.fromarray(image)
      image = image.resize(args.window, resample=Image.NEAREST)
      image = np.array(image)
    # 变量说明（surface）：由观察图像转换得到、准备绘制到窗口的 Pygame Surface。
    surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(args.fps)

    # Keyboard input.
    action = None
    pygame.event.pump()
    # 变量说明（event）：`main` 中当前处理的事件。
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False
      elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
        action = keymap[event.key]
    if action is None:
      # 变量说明（pressed）：当前键盘上处于按下状态的按键集合。
      pressed = pygame.key.get_pressed()
      for key, action in keymap.items():
        if pressed[key]:
          break
      else:
        if args.wait and not env._player.sleeping:
          continue
        else:
          action = 'noop'

    # Environment step.
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    _, reward, done, _ = env.step(env.action_names.index(action))
    duration += 1

    # Achievements.
    # 变量说明（count）：`main` 中当前处理的数量。
    # 变量说明（name）：当前对象、指标或配置项的名称。
    # 变量说明（unlocked）：本回合所有玩家已经首次完成的成就名称集合。
    unlocked = {
        name for name, count in env._player.achievements.items()
        if count > 0 and name not in achievements}
    for name in unlocked:
      achievements |= unlocked
      # 变量说明（total）：当前统计、距离或权重计算得到的累计总值。
      total = len(env._player.achievements.keys())
      print(f'Achievement ({len(achievements)}/{total}): {name}')
    if env._step > 0 and env._step % 100 == 0:
      print(f'Time step: {env._step}')
    if reward:
      print(f'Reward: {reward}')
      return_ += reward

    # Episode end.
    if done and not was_done:
      was_done = True
      print('Episode done!')
      print('Duration:', duration)
      print('Return:', return_)
      if args.death == 'quit':
        running = False
      if args.death == 'reset':
        print('\nStarting a new episode.')
        env.reset()
        achievements = set()
        was_done = False
        duration = 0
        return_ = 0
      if args.death == 'continue':
        pass

  pygame.quit()


if __name__ == '__main__':
  main()
