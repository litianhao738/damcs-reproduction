"""Crafter 环境与 DAMCS 智能体运行状态的显式版本化快照。"""

from __future__ import annotations

import base64
import importlib
import json
import random
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from crafter import Env, memory, objects
from memory_system.agent import Agent, HumanAgent


# 快照载荷版本；改变字段语义或编码方式时必须递增。
RUNTIME_SNAPSHOT_VERSION = 1
# 这些字段指向运行时对象，恢复时重新连接，不能直接递归序列化。
_SKIPPED_OBJECT_FIELDS = {"env", "world", "random", "text_model", "vision_model"}


def _class_path(value_or_class) -> str:
    """生成可跨进程重新导入的“模块:限定类名”标识。"""
    cls = value_or_class if isinstance(value_or_class, type) else type(value_or_class)
    return f"{cls.__module__}:{cls.__qualname__}"


def _import_class(path: str):
    """根据快照中的类路径动态导入对应 Python 类。"""
    module_name, qualname = path.split(":", 1)
    value = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def encode_value(value: Any):
    """把运行时值递归编码成可写入 JSON 的带类型结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return encode_value(value.item())
    # 数组按原始字节编码，同时保留 dtype 和 shape，避免精度损失。
    if isinstance(value, np.ndarray):
        return {
            "$type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    if isinstance(value, bytes):
        return {"$type": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return {"$type": "enum", "class": _class_path(value), "name": value.name}
    if isinstance(value, Path):
        return {"$type": "path", "value": str(value)}
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [encode_value(item) for item in value]}
    if isinstance(value, set):
        return {"$type": "set", "items": [encode_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        return {
            "$type": "dict",
            "items": [
                [encode_value(key), encode_value(item)] for key, item in value.items()
            ],
        }
    # NetworkX 图转换为 node-link 数据后继续使用统一编码器。
    if isinstance(value, nx.Graph):
        return {
            "$type": "networkx",
            "class": _class_path(value),
            "data": encode_value(nx.node_link_data(value)),
        }
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return {
                "$type": "torch_tensor",
                "data": encode_value(value.detach().cpu().numpy()),
            }
    except ImportError:
        pass
    if hasattr(value, "model_dump") and hasattr(type(value), "model_validate"):
        return {
            "$type": "pydantic",
            "class": _class_path(value),
            "data": encode_value(value.model_dump(mode="python")),
        }
    # 普通对象仅记录字段；被跳过的运行时引用由恢复流程重新挂接。
    if hasattr(value, "__dict__"):
        return {
            "$type": "object",
            "class": _class_path(value),
            "fields": {
                key: encode_value(item)
                for key, item in vars(value).items()
                if key not in _SKIPPED_OBJECT_FIELDS
            },
        }
    raise TypeError(f"Unsupported checkpoint value: {_class_path(value)}")


def decode_value(value: Any):
    """把 encode_value 生成的结构递归还原为 Python 运行时值。"""
    if not isinstance(value, (dict, list)):
        return value
    if isinstance(value, list):
        return [decode_value(item) for item in value]

    value_type = value.get("$type")
    if value_type == "ndarray":
        raw = base64.b64decode(value["data"])
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).copy().reshape(value["shape"])
    if value_type == "bytes":
        return base64.b64decode(value["data"])
    if value_type == "enum":
        return _import_class(value["class"])[value["name"]]
    if value_type == "path":
        return Path(value["value"])
    if value_type == "tuple":
        return tuple(decode_value(item) for item in value["items"])
    if value_type == "set":
        return {decode_value(item) for item in value["items"]}
    if value_type == "dict":
        return {
            decode_value(key): decode_value(item) for key, item in value["items"]
        }
    if value_type == "networkx":
        graph_data = decode_value(value["data"])
        return nx.node_link_graph(graph_data)
    if value_type == "torch_tensor":
        import torch

        return torch.from_numpy(decode_value(value["data"]))
    if value_type == "pydantic":
        return _import_class(value["class"]).model_validate(decode_value(value["data"]))
    if value_type == "object":
        cls = _import_class(value["class"])
        instance = cls.__new__(cls)
        for key, item in value["fields"].items():
            setattr(instance, key, decode_value(item))
        return instance
    return {key: decode_value(item) for key, item in value.items()}


def _snapshot_environment(env):
    """提取 Crafter 世界、玩家、对象、材质和随机数状态。"""
    # 玩家使用稳定的列表索引表示，避免把进程内存地址写入快照。
    player_ids = {id(player): index for index, player in enumerate(env._players)}
    world_object_ids = {id(obj) for obj in env._world.objects}

    def snapshot_object(obj):
        """序列化一个玩家或世界对象，并记录其关联玩家。"""
        fields = {
            key: encode_value(value)
            for key, value in vars(obj).items()
            if key not in {"world", "random", "pos", "removed", "player"}
        }
        record = {
            "class": _class_path(obj),
            "pos": encode_value(obj.pos),
            "fields": fields,
        }
        if isinstance(obj, objects.Player):
            record["player_id"] = player_ids[id(obj)]
            record["in_world"] = id(obj) in world_object_ids
        elif hasattr(obj, "player"):
            record["target_player_id"] = player_ids[id(obj.player)]
        return record

    # 玩家先于其他对象保存，恢复时才能正确重建对象对玩家的引用。
    object_records = [snapshot_object(player) for player in env._players]
    object_records.extend(
        snapshot_object(obj)
        for obj in env._world.objects
        if not isinstance(obj, objects.Player)
    )

    return {
        "config": {
            "area": encode_value(env._area),
            "view": encode_value(env._view),
            "size": encode_value(env._size),
            "reward": env._reward,
            "length": env._length,
            "n_players": env.n_players,
            "seed": env._seed,
        },
        "episode": env._episode,
        "step": env._step,
        "unlocked": encode_value(env._unlocked),
        "player_id": env.player_id,
        "alive_player_ids": encode_value(env._alive_players_id),
        "last_healths": encode_value(env._last_healths),
        "last_inventory": encode_value(env._last_inventory),
        "memories": [encode_value(vars(item)) for item in env.memories],
        "world": {
            "daylight": encode_value(env._world.daylight),
            "random_state": encode_value(env._world.random.get_state()),
            "mat_names": encode_value(env._world._mat_names),
            "mat_ids": encode_value(env._world._mat_ids),
            "mat_map": encode_value(env._world._mat_map),
            "objects": object_records,
        },
    }


def _restore_environment(snapshot):
    """根据结构化快照重新构造一个可继续 step 的 Crafter 环境。"""
    config = snapshot["config"]
    env = Env(
        area=decode_value(config["area"]),
        view=decode_value(config["view"]),
        size=decode_value(config["size"]),
        reward=config["reward"],
        length=config["length"],
        n_players=config["n_players"],
        seed=config["seed"],
    )
    world = env._world
    # 先清空构造函数生成的世界，再写回提交时的材质和对象。
    world.reset()
    world_snapshot = snapshot["world"]
    world._mat_names = decode_value(world_snapshot["mat_names"])
    world._mat_ids = decode_value(world_snapshot["mat_ids"])
    world._mat_map = decode_value(world_snapshot["mat_map"])
    world.daylight = decode_value(world_snapshot["daylight"])
    saved_world_random_state = decode_value(world_snapshot["random_state"])
    world.random.set_state(saved_world_random_state)

    env._players = [None] * env.n_players
    player_records = [
        record for record in world_snapshot["objects"] if "player_id" in record
    ]
    other_records = [
        record for record in world_snapshot["objects"] if "player_id" not in record
    ]

    def restore_object(record, add_to_world=True):
        """绕过构造函数恢复对象字段，并按需加入 World。"""
        cls = _import_class(record["class"])
        obj = cls.__new__(cls)
        obj.world = world
        obj.pos = decode_value(record["pos"])
        obj.random = world.random
        obj.removed = not add_to_world
        for key, item in record["fields"].items():
            setattr(obj, key, decode_value(item))
        if "target_player_id" in record:
            obj.player = env._players[record["target_player_id"]]
        if add_to_world:
            world.add(obj)
        return obj

    # 必须先恢复玩家，随后怪物/箭等对象才能重新绑定目标玩家。
    for record in sorted(player_records, key=lambda item: item["player_id"]):
        player = restore_object(record, add_to_world=record.get("in_world", True))
        env._players[record["player_id"]] = player
    for record in other_records:
        restore_object(record)

    env._episode = snapshot["episode"]
    env._step = snapshot["step"]
    env._unlocked = decode_value(snapshot["unlocked"])
    env.player_id = snapshot["player_id"]
    env._alive_players_id = decode_value(snapshot["alive_player_ids"])
    env._last_healths = decode_value(snapshot["last_healths"])
    env._last_inventory = decode_value(snapshot["last_inventory"])
    env.memories = []
    for item in snapshot["memories"]:
        fields = decode_value(item)
        chat_memory = memory.ChatMemory(fields["player_id"])
        chat_memory.memory = fields["memory"]
        env.memories.append(chat_memory)
    env.render_all()
    # 渲染会消耗世界随机数；渲染完成后写回已提交位置，保证后续转移一致。
    world.random.set_state(saved_world_random_state)
    return env


def _snapshot_agents(agents):
    """保存每个 Agent 的类型、标识、记忆和当前动作状态。"""
    return [
        {
            "class": _class_path(agent),
            "id": agent.id,
            "kg_update_freq": agent.kg_update_freq,
            "fields": {
                key: encode_value(value)
                for key, value in vars(agent).items()
                if key not in {"text_model", "env"}
            },
        }
        for agent in agents
    ]


def _relink_text_models(value, text_model, visited):
    """递归把恢复后的记忆模块重新连接到新建的 TextModel。"""
    if value is None or id(value) in visited:
        return
    if isinstance(value, (str, int, float, bool, bytes, np.ndarray)):
        return
    visited.add(id(value))
    if type(value).__name__ in {"SensoryMemory", "EpisodicMemory", "ProceduralMemory"}:
        value.text_model = text_model
    if isinstance(value, dict):
        children = list(value.keys()) + list(value.values())
    elif isinstance(value, (list, tuple, set)):
        children = list(value)
    elif hasattr(value, "__dict__"):
        children = list(vars(value).values())
    else:
        children = []
    for child in children:
        _relink_text_models(child, text_model, visited)


def _restore_agents(snapshots, env):
    """重建 Agent/HumanAgent，并绑定恢复后的同一个环境。"""
    agents = []
    for snapshot in snapshots:
        cls = _import_class(snapshot["class"])
        if not issubclass(cls, (Agent, HumanAgent)):
            raise ValueError(f"Unsupported agent class in checkpoint: {snapshot['class']}")
        # 调用构造函数获得不应序列化的模型客户端，再覆盖持久化字段。
        agent = cls(snapshot["id"], kg_update_freq=snapshot["kg_update_freq"])
        text_model = agent.text_model
        for key, item in snapshot["fields"].items():
            setattr(agent, key, decode_value(item))
        agent.text_model = text_model
        agent.env = env
        _relink_text_models(agent, text_model, set())
        agents.append(agent)
    return agents


def create_runtime_snapshot(state, env, agents) -> bytes:
    """创建包含图状态、环境、Agent 和随机数状态的 JSON 字节快照。"""
    # 真实 DAMCS 对象使用专用格式；测试替身使用通用对象编码格式。
    if isinstance(env, Env) and all(isinstance(agent, Agent) for agent in agents):
        environment_snapshot = {"kind": "crafter", "data": _snapshot_environment(env)}
        agents_snapshot = {"kind": "damcs", "data": _snapshot_agents(agents)}
    else:
        environment_snapshot = {"kind": "generic", "data": encode_value(env)}
        agents_snapshot = {"kind": "generic", "data": encode_value(agents)}
    payload = {
        "version": RUNTIME_SNAPSHOT_VERSION,
        "state": encode_value(state),
        "environment": environment_snapshot,
        "agents": agents_snapshot,
        "python_random_state": encode_value(random.getstate()),
        "numpy_random_state": encode_value(np.random.get_state()),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def restore_runtime_snapshot(snapshot: bytes):
    """校验版本并恢复快照，返回“图状态、环境、Agent 列表”。"""
    payload = json.loads(snapshot.decode("utf-8"))
    if payload.get("version") != RUNTIME_SNAPSHOT_VERSION:
        raise ValueError(
            f"Runtime snapshot version {payload.get('version')} is incompatible with "
            f"version {RUNTIME_SNAPSHOT_VERSION}."
        )
    if payload["environment"]["kind"] == "crafter":
        env = _restore_environment(payload["environment"]["data"])
    else:
        env = decode_value(payload["environment"]["data"])
    if payload["agents"]["kind"] == "damcs":
        agents = _restore_agents(payload["agents"]["data"], env)
    else:
        agents = decode_value(payload["agents"]["data"])
    # 最后恢复全局随机数，避免对象重建过程改变下一次随机结果。
    random.setstate(decode_value(payload["python_random_state"]))
    np.random.set_state(decode_value(payload["numpy_random_state"]))
    return decode_value(payload["state"]), env, agents
