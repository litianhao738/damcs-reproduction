# 模块说明：启动多智能体 Crafter 图形界面。
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
  parser.add_argument('--n_players', type=int, default=3)
  
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
      
      # multi player
      pygame.K_LSHIFT: 'switch_player',
      pygame.K_SEMICOLON: 'pause',
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
      area=args.area, view=args.view, length=args.length, seed=args.seed, n_players=args.n_players)
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
  # 变量说明（pid）：当前进程、玩家或被检查对象的整数编号。
  pid=0
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
        # added for multi player
        if args.wait and not (env._player.sleeping or env._player2.sleeping):
          continue
        
        else:
          action = 'noop'

    print('Action:', action, env.player_id)
    if action == 'switch_player':
      pid += 1
      pid %= env.n_players
      env.switch_player(pid)
      continue
    elif action == 'pause':
      # 变量说明（auxiliary_action）：`main` 中当前处理的动作。
      auxiliary_action = ""
      while auxiliary_action not in ['e', 'c', 'exit']:
        auxiliary_action = input("Please select one of the following actions [exchange item (e), communicate (c)]: ")
      if auxiliary_action == 'e':
        # 变量说明（target_player_id）：`main` 中当前对象的唯一编号或序列索引。
        target_player_id = -1
        print(f"You are Player {env.player_id}.")
        print("Here is your inventory:", env._players[env.player_id].inventory)
        print(f"There are {env.n_players} players in total and their ids are 0~{env.n_players-1}. The alive players are {env._alive_players_id}.")
        target_player_id = int(input(f"Please select a player (id) to exchange item: "))
        # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
        item = input("Please select an item to exchange: ")
        env.exchange_item(target_player_id, item)
      elif auxiliary_action == 'c':
        def _change_target_chat_player_ids():
          print("Chatting mode is on. You can type 'exit' to exit chatting mode; 'history' to see past interactions; 'change' to change the target player ids.")
          print(f"There are {env.n_players} players in total and their ids are 0~{env.n_players-1}. The alive players are {env._alive_players_id}.")
          # 变量说明（target_player_ids）：`_change_target_chat_player_ids` 中待批量处理的玩家集合。
          target_player_ids = input(f"Please select the player ids you want to communicate with (separate by space) or all (a): ")
          if target_player_ids == 'a':
            target_player_ids = list(range(env.n_players))
          else:
            target_player_ids = target_player_ids.split()
            # 变量说明（id_）：`_change_target_chat_player_ids` 中当前处理的编号。
            target_player_ids = [int(id_) for id_ in target_player_ids]
          target_player_ids.append(env.player_id)
          return set(target_player_ids)
          
        # 变量说明（chat）：玩家之间按时间保存的通信消息列表。
        chat = ""
        # 变量说明（target_player_ids）：`main` 中待批量处理的玩家集合。
        target_player_ids = _change_target_chat_player_ids()
        while chat != 'exit':
          if chat == 'history':
            env.show_history()
            chat = ""
          elif chat == 'change':
            target_player_ids = _change_target_chat_player_ids()
          elif chat != "":
            # 变量说明（msg）：准备发送给大模型或其他组件的消息内容。
            msg = {'role': f"Player {env.player_id}", 'message': chat, "to": target_player_ids}
            env.chat(msg, target_player_ids)
          chat = input(f"Player {env.player_id}: ")
      elif auxiliary_action == 'exit':
        continue
      continue
      
    # Environment step.
    # 变量说明（actions）：按智能体编号排列、将提交给环境的动作列表。
    actions = ['noop'] * env.n_players
    actions[env.player_id] = action
    # 变量说明（actions_index_list）：`main` 中待批量处理的动作、索引集合。
    actions_index_list = [env.action_names.index(action) for action in actions]
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    _, reward, done, _ = env.step(actions_index_list)
    duration += 1

    # Achievements.
    # unlocked = {
    #     name for name, count in env._player.achievements.items()
    #     if count > 0 and name not in achievements}
    # for name in unlocked:
    #   achievements |= unlocked
    #   total = len(env._player.achievements.keys())
    #   print(f'Achievement ({len(achievements)}/{total}): {name}')
    # if env._step > 0 and env._step % 100 == 0:
    #   print(f'Time step: {env._step}')
    # if reward:
    #   print(f'Reward: {reward}')
    #   return_ += reward

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
