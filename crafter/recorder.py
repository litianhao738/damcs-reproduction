# 模块说明：封装环境运行记录、视频写入和统计保存。
import datetime
import json
import pathlib

import imageio
import numpy as np
import os

class Recorder:

  def __init__(
      self, env, directory, save_stats=True, save_video=True,
      save_episode=True, video_size=(512, 512)):
    # 变量说明（env）：当前 Crafter 仿真环境实例。
    # 变量说明（directory）：当前函数准备读取或写入文件的目录路径。
    # 变量说明（save_stats）：是否把当前回合统计数据写入磁盘的布尔参数。
    # 变量说明（save_video）：`__init__` 中当前处理的视频。
    # 变量说明（save_episode）：`__init__` 中当前处理的回合。
    # 变量说明（video_size）：`__init__` 中当前处理的视频。
    if directory and save_stats:
      env = StatsRecorder(env, directory)
    if directory and save_video:
      env = VideoRecorder(env, directory, video_size)
    if directory and save_episode:
      env = EpisodeRecorder(env, directory)
    # 变量说明（self._env）：当前 Crafter 仿真环境实例。
    self._env = env

  def __getattr__(self, name):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)


class StatsRecorder:

  def __init__(self, env, directory):
    # 变量说明（env）：当前 Crafter 仿真环境实例。
    # 变量说明（directory）：当前函数准备读取或写入文件的目录路径。
    self._env = env
    # 变量说明（self._directory）：当前函数准备读取或写入文件的目录路径。
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    # 变量说明（self._file）：`__init__` 中当前处理的文件。
    self._file = (self._directory / 'stats.jsonl').open('a')
    # 变量说明（self._length）：一个环境回合最多允许推进的时间步数。
    self._length = None
    # 变量说明（self._reward）：当前时间步或当前玩家获得的标量奖励。
    self._reward = None
    # 变量说明（self._unlocked）：本回合所有玩家已经首次完成的成就名称集合。
    self._unlocked = None
    # 变量说明（self._stats）：当前玩家状态、实验成就或绘图统计数据。
    self._stats = None

  def __getattr__(self, name):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    obs = self._env.reset()
    self._length = 0
    self._reward = 0
    self._unlocked = None
    self._stats = None
    return obs

  def step(self, action):
    # 变量说明（action）：当前智能体准备执行的环境动作。
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    obs, reward, done, info = self._env.step(action)
    # for single agent test
    #if type(info) is list:
    #  info = info[0] # to be considered?
    self._length += 1
    self._reward += info['reward']
    if done:
      self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
      # 变量说明（key）：当前字典条目的键。
      # 变量说明（value）：当前配置项、字段或计算结果对应的值。
      for key, value in info['achievements'].items():
        self._stats[f'achievement_{key}'] = value
      self._save()
    return obs, reward, done, info

  def _save(self):
    self._file.write(json.dumps(self._stats) + '\n')
    self._file.flush()


# modified for multiagent
class VideoRecorder:
  def __init__(self, env, directory, size=(512, 512)):
    # 变量说明（env）：当前 Crafter 仿真环境实例。
    # 变量说明（directory）：当前函数准备读取或写入文件的目录路径。
    # 变量说明（size）：当前图像、窗口、网格或集合的尺寸。
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    # 变量说明（self._size）：当前图像、窗口、网格或集合的尺寸。
    self._size = size
    # modified for multiagent
    # 变量说明（self._players_frames）：`__init__` 中待批量处理的玩家、图像帧集合。
    self._players_frames = [None] * env.n_players
    
    #self._frames = None

  def __getattr__(self, name):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    obs = self._env.reset()
    if self._env.n_players == 1:
      # 变量说明（obs_all）：当前时间步所有玩家的视觉观察列表。
      obs_all = [obs]
    else:
      obs_all = obs
    #obs_all = self._env.render_all(self._size)
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    for i in range(self._env.n_players):
      # make canvas grayscale
      #obs[i] = np.mean(obs[i], axis=-1)
      self._players_frames[i] = [obs_all[i]]
    return obs

  def step(self, action):
    # 变量说明（action）：当前智能体准备执行的环境动作。
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    obs, reward, done, info = self._env.step(action)
    if self._env.n_players == 1:
      # 变量说明（obs_all）：当前时间步所有玩家的视觉观察列表。
      obs_all = [obs]
    else:
      obs_all = obs
    #canvases = self._env.render_all(self._size)
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    for i in range(self._env.n_players):
      # make canvas grayscale
      #canvases[i] = np.mean(canvases[i], axis=-1)
      self._players_frames[i].append(obs_all[i])
    #self._frames.append(self._env.render(self._size))
    if done:
      self._save()
    return obs, reward, done, info

  def _save(self):
    # 变量说明（filepath）：当前准备读取或写入文件的完整路径。
    filepath = str(self._directory / self._env.episode_name)
    if not os.path.exists(filepath):
      os.makedirs(filepath)
    else:
      # clear the directory
      # 变量说明（f）：当前循环使用的文件、函数或 A* 估计总代价值。
      for f in os.listdir(filepath):
        os.remove(os.path.join(filepath, f))
        
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    for i in range(self._env.n_players):
      imageio.mimsave(filepath+f'/agent_{i}.mp4', self._players_frames[i])
    #imageio.mimsave(filename, self._frames)

class EpisodeRecorder:
  def __init__(self, env, directory):
    # 变量说明（env）：当前 Crafter 仿真环境实例。
    # 变量说明（directory）：当前函数准备读取或写入文件的目录路径。
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    # 变量说明（self._episode）：当前回合编号或回合相关数据。
    self._episode = None

  def __getattr__(self, name):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    obs = self._env.reset()
    self._episode = [{'image': obs}]
    return obs

  def step(self, action):
    # Transitions are defined from the environment perspective, meaning that a
    # transition contains the action and the resulting reward and next
    # observation produced by the environment in response to said action.
    # 变量说明（action）：当前智能体准备执行的环境动作。
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    obs, reward, done, info = self._env.step(action)
    #info = info[0] # to be considered?
    # 变量说明（transition）：纹理或动画状态之间使用的过渡比例。
    transition = {
        'action': action, 'image': obs, 'reward': reward, 'done': done,
    }
    # 变量说明（key）：当前字典条目的键。
    # 变量说明（value）：当前配置项、字段或计算结果对应的值。
    for key, value in info.items():
      if key in ('inventory', 'achievements'):
        continue
      transition[key] = value
    for key, value in info['achievements'].items():
      transition[f'achievement_{key}'] = value
    for key, value in info['inventory'].items():
      transition[f'ainventory_{key}'] = value
    self._episode.append(transition)
    if done:
      self._save()
    return obs, reward, done, info

  def _save(self):
    # 变量说明（filename）：当前读取、保存或展示的文件名称。
    filename = str(self._directory / (self._env.episode_name + '.npz'))
    # Fill in zeros for keys missing at the first time step.
    # 变量说明（key）：当前字典条目的键。
    # 变量说明（value）：当前配置项、字段或计算结果对应的值。
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    # 变量说明（episode）：当前回合编号或回合相关数据。
    # 变量说明（k）：限制检索、回看或邻居搜索数量的整数参数。
    episode = {
        # 变量说明（step）：当前经验或环境时间步编号。
        k: np.array([step[k] for step in self._episode])
        for k in self._episode[0]}
    np.savez_compressed(filename, **episode)


class EpisodeName:

  def __init__(self, env):
    # 变量说明（env）：当前 Crafter 仿真环境实例。
    self._env = env
    # 变量说明（self._timestamp）：当前帧、消息或统计记录对应的时间步标识。
    self._timestamp = None
    self._unlocked = None
    self._length = None

  def __getattr__(self, name):
    # 变量说明（name）：当前对象、指标或配置项的名称。
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    obs = self._env.reset()
    self._timestamp = None
    self._unlocked = None
    self._length = 0
    return obs

  def step(self, action):
    # 变量说明（action）：当前智能体准备执行的环境动作。
    # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
    # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
    # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
    # 变量说明（reward）：当前时间步或当前玩家获得的标量奖励。
    obs, reward, done, info = self._env.step(action)
    #if type(info) is list:
    #  info = info[0] # to be considered?
    self._length += 1
    if done:
      self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
      self._unlocked = 0#sum(int(v >= 1) for v in info['achievements'].values())
    return obs, reward, done, info

  @property
  def episode_name(self):
    return f'{self._timestamp}-ach{self._unlocked}-len{self._length}'