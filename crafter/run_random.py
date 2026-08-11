# 模块说明：使用随机动作运行 Crafter 环境，用于快速测试。
import argparse
import pathlib
import time

import numpy as np

import crafter


def main():
  # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--length', type=int, default=10000)
  parser.add_argument('--health', type=int, default=9)
  parser.add_argument('--record', type=pathlib.Path, default=None)
  parser.add_argument('--episodes', type=int, default=1)
  # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
  args = parser.parse_args()

  # 变量说明（random）：控制环境包装器是否使用随机动作的布尔参数。
  random = np.random.RandomState(args.seed)
  crafter.constants.items['health']['max'] = args.health
  crafter.constants.items['health']['initial'] = args.health
  # 变量说明（env）：当前 Crafter 仿真环境实例。
  env = crafter.Env(area=args.area, length=args.length, seed=args.seed)
  env = crafter.Recorder(env, args.record)

  for _ in range(args.episodes):

    # 变量说明（start）：搜索、切片或计时过程使用的起始位置。
    start = time.time()
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    obs = env.reset()
    print('')
    print(f'Reset time: {1000*(time.time()-start):.2f}ms')
    print('Coal exist:    ', env._world.count('coal'))
    print('Iron exist:    ', env._world.count('iron'))
    print('Diamonds exist:', env._world.count('diamond'))

    start = time.time()
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    done = False
    while not done:
      # 变量说明（action）：当前智能体准备执行的环境动作。
      action = random.randint(0, env.action_space.n)
      # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
      # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
      obs, reward, done, info = env.step(action)
    # 变量说明（duration）：当前动作、动画或视频片段持续的时间或帧数。
    duration = time.time() - start
    # 变量说明（step）：当前经验或环境时间步编号。
    step = env._step
    print(f'Step time: {1000*duration/step:.2f}ms ({int(step/duration)} FPS)')
    print('Episode length:', step)


if __name__ == '__main__':
  main()
