# 模块说明：封装不同大模型供应商、结构化输出以及文本向量接口。
from typing import Any
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from openai import AzureOpenAI, OpenAI
import re
import tiktoken
from .utils import print_color
from pydantic import BaseModel, Field, ValidationError
from typing import Union, Literal
import time
from openai import APITimeoutError

from enum import Enum


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is not configured for runtime calls."""


@dataclass(frozen=True)
class LLMConfig:
    # 变量说明（provider）：当前选择的大模型服务商名称。
    provider: str
    # 变量说明（chat_model）：用于智能体决策的聊天模型或部署名称。
    chat_model: str
    # 变量说明（embedding_model）：用于生成记忆文本向量的模型或部署名称。
    embedding_model: str
    # 变量说明（azure_endpoint）：Azure OpenAI 资源的 HTTPS 服务端点。
    azure_endpoint: str | None = None
    # 变量说明（azure_api_key）：访问 Azure OpenAI 资源所需的 API 密钥。
    azure_api_key: str | None = None
    # 变量说明（azure_api_version）：Azure OpenAI 请求使用的接口版本字符串。
    azure_api_version: str | None = None
    # 变量说明（openai_api_key）：访问 OpenAI 或兼容服务所需的 API 密钥。
    openai_api_key: str | None = None
    # 变量说明（openai_base_url）：OpenAI 兼容接口的基础 URL；官方服务可为空。
    openai_base_url: str | None = None


def _env(name, default=None):
    # 变量说明（name）：需要读取的环境变量名称。
    # 变量说明（default）：对应环境变量未配置时返回的默认值。
    # 变量说明（value）：从操作系统环境中读取并去除首尾空白后的配置值。
    value = os.getenv(name, default)
    if isinstance(value, str):
        value = value.strip()
    return value or None


def _env_int(name, default):
    # 变量说明（name）：需要读取的环境变量名称。
    # 变量说明（default）：对应环境变量未配置时返回的默认值。
    # 变量说明（value）：从操作系统环境中读取并去除首尾空白后的配置值。
    value = _env(name)
    if value is None:
        return default
    return int(value)


def _env_float(name, default):
    # 变量说明（name）：需要读取的环境变量名称。
    # 变量说明（default）：对应环境变量未配置时返回的默认值。
    # 变量说明（value）：从操作系统环境中读取并去除首尾空白后的配置值。
    value = _env(name)
    if value is None:
        return default
    return float(value)


def get_llm_config() -> LLMConfig:
    # 变量说明（provider）：当前选择的大模型服务商名称。
    provider = (_env("LLM_PROVIDER", "azure") or "azure").lower()

    if provider == "azure":
        # 变量说明（endpoint）：从环境变量读取的 Azure OpenAI 服务端点。
        endpoint = _env("AZURE_OPENAI_ENDPOINT")
        # 变量说明（api_key）：访问当前大模型服务商所需的 API 密钥。
        api_key = _env("AZURE_OPENAI_API_KEY")
        # 变量说明（api_version）：Azure OpenAI 请求使用的接口版本字符串。
        api_version = _env("AZURE_OPENAI_API_VERSION")
        # 变量说明（missing）：当前配置、背包或目标前置条件中缺失的字段或物品。
        # 变量说明（name）：当前对象、指标或配置项的名称。
        # 变量说明（value）：当前配置项、字段或计算结果对应的值。
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": endpoint,
                "AZURE_OPENAI_API_KEY": api_key,
                "AZURE_OPENAI_API_VERSION": api_version,
            }.items()
            if not value
        ]
        if missing:
            raise LLMConfigurationError(
                "Azure OpenAI is selected but missing environment variables: "
                + ", ".join(missing)
            )
        return LLMConfig(
            provider=provider,
            chat_model=_env("AZURE_OPENAI_CHAT_DEPLOYMENT", _env("OPENAI_MODEL", "gpt-4o")),
            embedding_model=_env(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                _env("OPENAI_EMBEDDING_MODEL", "text-embedding"),
            ),
            azure_endpoint=endpoint,
            azure_api_key=api_key,
            azure_api_version=api_version,
        )

    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "OpenAI is selected but missing environment variable: OPENAI_API_KEY"
            )
        return LLMConfig(
            provider=provider,
            chat_model=_env("OPENAI_MODEL", "gpt-4o"),
            embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=api_key,
        )

    if provider == "qwen":
        api_key = _env("QWEN_API_KEY", _env("DASHSCOPE_API_KEY"))
        if not api_key:
            raise LLMConfigurationError(
                "Qwen is selected but missing environment variable: QWEN_API_KEY "
                "or DASHSCOPE_API_KEY"
            )
        return LLMConfig(
            provider=provider,
            chat_model=_env("QWEN_CHAT_MODEL", _env("OPENAI_MODEL", "qwen3.7-plus")),
            embedding_model=_env(
                "QWEN_EMBEDDING_MODEL",
                _env("OPENAI_EMBEDDING_MODEL", "text-embedding-v4"),
            ),
            openai_api_key=api_key,
            openai_base_url=_env(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )

    raise LLMConfigurationError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'azure', 'openai', or 'qwen'."
    )


def describe_llm_configuration() -> dict[str, Any]:
    # 变量说明（provider）：当前选择的大模型服务商名称。
    provider = (_env("LLM_PROVIDER", "azure") or "azure").lower()
    try:
        # 变量说明（config）：根据环境变量解析得到的大模型供应商配置。
        config = get_llm_config()
        return {
            "configured": True,
            "provider": config.provider,
            "chat_model": config.chat_model,
            "embedding_model": config.embedding_model,
            "missing": [],
        }
    # 变量说明（exc）：当前捕获到的异常对象。
    except LLMConfigurationError as exc:
        # 变量说明（missing_by_provider）：每种大模型供应商当前缺失的环境变量名称列表。
        missing_by_provider = {
            # 变量说明（name）：当前对象、指标或配置项的名称。
            "azure": [
                name
                for name in [
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_OPENAI_API_KEY",
                    "AZURE_OPENAI_API_VERSION",
                ]
                if not _env(name)
            ],
            "openai": ["OPENAI_API_KEY"] if not _env("OPENAI_API_KEY") else [],
            "qwen": []
            if _env("QWEN_API_KEY", _env("DASHSCOPE_API_KEY"))
            else ["QWEN_API_KEY or DASHSCOPE_API_KEY"],
        }
        return {
            "configured": False,
            "provider": provider,
            "chat_model": _env("AZURE_OPENAI_CHAT_DEPLOYMENT", _env("OPENAI_MODEL", "gpt-4o"))
            if provider == "azure"
            else _env("QWEN_CHAT_MODEL", _env("OPENAI_MODEL", "qwen3.7-plus"))
            if provider == "qwen"
            else _env("OPENAI_MODEL", "gpt-4o"),
            "embedding_model": _env(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                _env("OPENAI_EMBEDDING_MODEL", "text-embedding"),
            )
            if provider == "azure"
            else _env(
                "QWEN_EMBEDDING_MODEL",
                _env("OPENAI_EMBEDDING_MODEL", "text-embedding-v4"),
            )
            if provider == "qwen"
            else _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "missing": missing_by_provider.get(provider, []),
            "error": str(exc),
        }


@lru_cache(maxsize=1)
def get_client():
    # 变量说明（config）：根据环境变量解析得到的大模型供应商配置。
    config = get_llm_config()
    if config.provider == "azure":
        return AzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            timeout=_env_float("LLM_TIMEOUT_SECONDS", 120.0),
        )
    if config.openai_base_url:
        return OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=_env_float("LLM_TIMEOUT_SECONDS", 120.0),
        )
    return OpenAI(api_key=config.openai_api_key, timeout=_env_float("LLM_TIMEOUT_SECONDS", 120.0))

class ResultType(str, Enum):
    # 变量说明（SUCCESS）：表示操作成功的结果状态枚举成员。
    SUCCESS = "success"
    # 变量说明（FAILURE）：表示操作失败的结果状态枚举成员。
    FAILURE = "failure"
    # 变量说明（IN_PROGRESS）：动作仍在执行、暂时不需要重新规划的枚举值。
    IN_PROGRESS = "in_progress"
    
class ActionType(str, Enum):
    # 变量说明（noop）：表示本时间步不执行操作的动作枚举成员。
    noop = "noop"
    # 变量说明（move_left）：表示向左移动的动作枚举成员。
    move_left = "move_left"
    # 变量说明（move_right）：表示向右移动的动作枚举成员。
    move_right = "move_right"
    # 变量说明（move_up）：表示向上移动的动作枚举成员。
    move_up = "move_up"
    # 变量说明（move_down）：表示向下移动的动作枚举成员。
    move_down = "move_down"
    # 变量说明（do）：表示与面前对象交互的基础动作枚举成员。
    do = "do"
    # 变量说明（sleep）：表示让玩家睡眠恢复能量的动作枚举成员。
    sleep = "sleep"
    # 变量说明（place_stone）：表示放置石块的环境动作枚举成员。
    place_stone = "place_stone"
    # 变量说明（place_table）：表示放置工作台动作或目标的枚举成员。
    place_table = "place_table"
    # 变量说明（place_furnace）：表示放置熔炉动作或目标的枚举成员。
    place_furnace = "place_furnace"
    # 变量说明（place_plant）：表示种植植物的环境动作枚举成员。
    place_plant = "place_plant"
    # 变量说明（make_wood_pickaxe）：表示制造木镐动作或目标的枚举成员。
    make_wood_pickaxe = "make_wood_pickaxe"
    # 变量说明（make_stone_pickaxe）：表示制造石镐动作或目标的枚举成员。
    make_stone_pickaxe = "make_stone_pickaxe"
    # 变量说明（make_iron_pickaxe）：表示制造铁镐动作或目标的枚举成员。
    make_iron_pickaxe = "make_iron_pickaxe"
    # 变量说明（Navigator）：表示调用自动寻路器的高层动作枚举成员。
    Navigator = "Navigator"
    # 变量说明（share）：表示资源共享动作或目标的枚举成员。
    share = "share"
    
class GoalType(str, Enum):
    # 变量说明（COLLECT_WOOD）：表示采集木材任务的枚举成员。
    COLLECT_WOOD = "collect_wood"
    # 变量说明（MAKE_WOOD_PICKAXE）：表示制造木镐动作或目标的枚举成员。
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    # 变量说明（COLLECT_STONE）：表示采集石头任务的枚举成员。
    COLLECT_STONE = "collect_stone"
    # 变量说明（MAKE_STONE_PICKAXE）：表示制造石镐动作或目标的枚举成员。
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    # 变量说明（COLLECT_IRON）：表示采集铁矿任务的枚举成员。
    COLLECT_IRON = "collect_iron"
    # 变量说明（MAKE_IRON_PICKAXE）：表示制造铁镐动作或目标的枚举成员。
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    # 变量说明（COLLECT_DIAMOND）：表示采集钻石任务的枚举成员。
    COLLECT_DIAMOND = "collect_diamond"
    
    # 变量说明（PLACE_TABLE）：表示放置工作台动作或目标的枚举成员。
    PLACE_TABLE = "place_table"
    # 变量说明（PLACE_FURNACE）：表示放置熔炉动作或目标的枚举成员。
    PLACE_FURNACE = "place_furnace"
    # 变量说明（COLLECT_COAL）：表示采集煤炭任务的枚举成员。
    COLLECT_COAL = "collect_coal"
    # 变量说明（SHARE）：表示资源共享动作或目标的枚举成员。
    SHARE = "share"

class LongTermGoalType(str, Enum):
    # 变量说明（MAKE_WOOD_PICKAXE）：表示制造木镐动作或目标的枚举成员。
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    # 变量说明（MAKE_STONE_PICKAXE）：表示制造石镐动作或目标的枚举成员。
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    # 变量说明（MAKE_IRON_PICKAXE）：表示制造铁镐动作或目标的枚举成员。
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    # 变量说明（PLACE_TABLE）：表示放置工作台动作或目标的枚举成员。
    PLACE_TABLE = "place_table"
    # 变量说明（PLACE_FURNACE）：表示放置熔炉动作或目标的枚举成员。
    PLACE_FURNACE = "place_furnace"
    # 变量说明（COLLECT_DIAMOND）：表示采集钻石任务的枚举成员。
    COLLECT_DIAMOND = "collect_diamond"
    # 变量说明（HELP_AGENT）：`LongTermGoalType` 中当前处理的智能体。
    HELP_AGENT = "help_agent"
    
class MaterialType(str, Enum):
    # 变量说明（TABLE）：表示工作台物品、材质或导航目标的枚举成员。
    TABLE = "table"
    # 变量说明（FURNACE）：表示熔炉物品、材质或导航目标的枚举成员。
    FURNACE = "furnace"
    # 变量说明（GRASS）：表示草地材质的枚举成员。
    GRASS = "grass"
    # 变量说明（PATH）：当前读取或写入对象对应的文件系统路径。
    PATH = "path"
    # 变量说明（SAND）：表示沙地材质的枚举成员。
    SAND = "sand"
    # 变量说明（LAVA）：表示岩浆材质的枚举成员。
    LAVA = "lava"
    # 变量说明（TREE）：表示树木材质或采集木材目标的枚举成员。
    TREE = "tree"
    # 变量说明（WATER）：表示水域材质或取水目标的枚举成员。
    WATER = "water"
    # 变量说明（STONE）：表示石头资源或石头材质的枚举成员。
    STONE = "stone"
    # 变量说明（COAL）：表示煤炭资源或煤炭材质的枚举成员。
    COAL = "coal"
    # 变量说明（IRON）：表示铁矿资源或铁质材料的枚举成员。
    IRON = "iron"
    # 变量说明（DIAMOND）：表示钻石资源或采集钻石目标的枚举成员。
    DIAMOND = "diamond"
    # 变量说明（PLAYER）：当前处理的 Crafter 玩家对象。
    PLAYER = "player"
    # 变量说明（COW）：表示牛这一世界生物类型的枚举成员。
    COW = "cow"
    # 变量说明（ZOMBIE）：表示僵尸这一敌对生物类型的枚举成员。
    ZOMBIE = "zombie"
    # 变量说明（SKELETON）：表示骷髅这一敌对生物类型的枚举成员。
    SKELETON = "skeleton"
    # 变量说明（ARROW）：表示箭矢投射物类型或材质的枚举成员。
    ARROW = "arrow"
    # 变量说明（PLANT）：表示可种植植物类型或材质的枚举成员。
    PLANT = "plant"
    
class Reflection(BaseModel):
    # 变量说明（vision）：从完整观察中切出的当前游戏画面像素数组。
    vision: list[MaterialType] = Field(description="List of materials you see around you.")
    # 变量说明（last_action）：`Reflection` 中当前处理的动作。
    last_action: ActionType #= Field(description="Your last action.")
    # 变量说明（last_action_result）：`Reflection` 中当前处理的动作、结果。
    last_action_result: ResultType
    # 变量说明（last_action_result_reflection）：`Reflection` 中当前处理的动作、结果。
    last_action_result_reflection: str #= Field(description="Reflection on the last action.")
    # 变量说明（last_action_repeated_reflection）：`Reflection` 中当前处理的动作。
    last_action_repeated_reflection: str = Field(description="Did you repeat the last action? If so, why?")

class Goal(BaseModel):
    # 变量说明（ultimate_goal）：`Goal` 中当前处理的目标。
    ultimate_goal: LongTermGoalType = Field(description="What is your ultimate goal?")
    
    # 变量说明（long_term_goal）：`Goal` 中当前处理的目标。
    long_term_goal: LongTermGoalType = Field(description="Working towards the ultimate goal, what should be your next goal?")
    # 变量说明（long_term_goal_subgoals）：`Goal` 中待批量处理的目标集合。
    long_term_goal_subgoals: str = Field(Description="What are the subgoals to complete the long term goal?")
    # 变量说明（long_term_goal_progress）：`Goal` 中待批量处理的目标集合。
    long_term_goal_progress: GoalType = Field(Description="What is the progress of the long term goal?")
    # 变量说明（long_term_goal_status）：`Goal` 中当前对象或操作所处的状态。
    long_term_goal_status: ResultType
    
    # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
    current_goal: GoalType = Field(description="The current goal that you are working on.")
    # 变量说明（current_goal_reason）：`Goal` 中当前处理的目标。
    current_goal_reason: str #= Field(description="Why do you choose the current goal?")
    # 变量说明（current_goal_status）：`Goal` 中当前对象或操作所处的状态。
    current_goal_status: ResultType
    
class NavigationDestinationItems(str, Enum):
    # 变量说明（TREE）：表示树木材质或采集木材目标的枚举成员。
    TREE = "tree"
    # 变量说明（WATER）：表示水域材质或取水目标的枚举成员。
    WATER = "water"
    # 变量说明（STONE）：表示石头资源或石头材质的枚举成员。
    STONE = "stone"
    # 变量说明（IRON）：表示铁矿资源或铁质材料的枚举成员。
    IRON = "iron"
    # 变量说明（DIAMOND）：表示钻石资源或采集钻石目标的枚举成员。
    DIAMOND = "diamond"
    # 变量说明（COAL）：表示煤炭资源或煤炭材质的枚举成员。
    COAL = "coal"
    # 变量说明（GRASS）：表示草地材质的枚举成员。
    GRASS = "grass"
    # COW = 'cow'
    # 变量说明（TABLE）：表示工作台物品、材质或导航目标的枚举成员。
    TABLE = "table"
    # 变量说明（FURNACE）：表示熔炉物品、材质或导航目标的枚举成员。
    FURNACE = "furnace"
    # 变量说明（NOT_APPICABLE）：拼写沿用原代码、表示当前字段不适用的枚举成员。
    NOT_APPICABLE = "not_applicable"
    
class ShareableItems(str, Enum):
    # 变量说明（WOOD）：表示木材资源的枚举成员。
    WOOD = "wood"
    # 变量说明（STONE）：表示石头资源或石头材质的枚举成员。
    STONE = "stone"
    # 变量说明（COAL）：表示煤炭资源或煤炭材质的枚举成员。
    COAL = "coal"
    # 变量说明（IRON）：表示铁矿资源或铁质材料的枚举成员。
    IRON = "iron"
    # 变量说明（DIAMOND）：表示钻石资源或采集钻石目标的枚举成员。
    DIAMOND = "diamond"
    # 变量说明（WOOD_PICKAXE）：表示木镐物品的枚举成员。
    WOOD_PICKAXE = "wood_pickaxe"
    # 变量说明（STONE_PICKAXE）：表示石镐物品的枚举成员。
    STONE_PICKAXE = "stone_pickaxe"
    # 变量说明（IRON_PICKAXE）：表示铁镐物品的枚举成员。
    IRON_PICKAXE = "iron_pickaxe"
    # food
    # water
    # 变量说明（NOT_APPLICABLE）：表示当前采集、共享或导航字段不适用的枚举成员。
    NOT_APPLICABLE = "not_applicable"

class InventoryItems(str, Enum):
    # 变量说明（WOOD）：表示木材资源的枚举成员。
    WOOD = "wood"
    # 变量说明（STONE）：表示石头资源或石头材质的枚举成员。
    STONE = "stone"
    # 变量说明（COAL）：表示煤炭资源或煤炭材质的枚举成员。
    COAL = "coal"
    # 变量说明（IRON）：表示铁矿资源或铁质材料的枚举成员。
    IRON = "iron"
    # 变量说明（DIAMOND）：表示钻石资源或采集钻石目标的枚举成员。
    DIAMOND = "diamond"
    # 变量说明（WOOD_PICKAXE）：表示木镐物品的枚举成员。
    WOOD_PICKAXE = "wood_pickaxe"
    # 变量说明（STONE_PICKAXE）：表示石镐物品的枚举成员。
    STONE_PICKAXE = "stone_pickaxe"
    # 变量说明（IRON_PICKAXE）：表示铁镐物品的枚举成员。
    IRON_PICKAXE = "iron_pickaxe"
 
class InventoryItemsCount(BaseModel):
    # 变量说明（item）：当前遍历、采集、交换或渲染的物品。
    item: InventoryItems
    # 变量说明（count）：`InventoryItemsCount` 中当前处理的数量。
    count: int

# Model for planning the next action
class NextAction(BaseModel):
    # 变量说明（next_action）：`NextAction` 中当前处理的动作。
    next_action: ActionType = Field(description="What is the next action you plan to take?")
    # 变量说明（next_action_reason）：`NextAction` 中当前处理的动作。
    next_action_reason: str # = Field(description="Why do you think this shoud be the next action?")
    # 变量说明（next_action_prerequisites_status）：`NextAction` 中当前对象或操作所处的状态。
    next_action_prerequisites_status: ResultType = Field(description="Are the prerequisites met?")
    # 变量说明（next_action_prerequisites）：`NextAction` 中待批量处理的动作集合。
    next_action_prerequisites: str = Field(description="What prerequisites are not met?")
    # 变量说明（final_next_action）：`NextAction` 中当前处理的动作。
    final_next_action: ActionType = Field(description="What is your final decision on next action.")
    # 变量说明（final_next_action_reason）：`NextAction` 中当前处理的动作。
    final_next_action_reason: str # = Field(description="Reason for choosing this action.")
    # 变量说明（final_next_action_status）：`NextAction` 中当前对象或操作所处的状态。
    final_next_action_status: ResultType # = Field(description="What is the status of the final decision?")
    # 变量说明（final_target_material_to_collect）：LLM 最终决定导航并采集的资源名称。
    final_target_material_to_collect: NavigationDestinationItems = Field(description="Navigate to where?")
    # 变量说明（final_target_material_to_share）：LLM 最终决定转交给其他智能体的物品名称。
    final_target_material_to_share: ShareableItems = Field(description="Share what?")
    # 变量说明（final_target_agent_id）：`NextAction` 中当前对象的唯一编号或序列索引。
    final_target_agent_id: int = Field(description="Which agent to share with, if applicable, or return -1.")
    
class Collaboration(BaseModel):
    # 变量说明（target_agent_to_help）：`Collaboration` 中当前处理的智能体。
    target_agent_to_help: int = Field(description="Which agent should you help, if applicable?")
    # 变量说明（target_agent_need）：`Collaboration` 中当前处理的智能体。
    target_agent_need: ShareableItems = Field(description="What does the target agent need, if applicable?")
    # 变量说明（help_method）：提示词中建议当前智能体帮助队友的方式文本。
    help_method: str = Field(description="What can you do to help the agent, if applicable?")
    # 变量说明（can_help_now）：表示 `can_help_now` 所描述条件是否成立的布尔标记。
    can_help_now: ResultType = Field(description="Can you help the agent now? Do you have the resources in inventory?")
    # 变量说明（being_helped_by_agent）：`Collaboration` 中当前处理的智能体。
    being_helped_by_agent: int = Field(description="Which agent is helping you, if applicable?")
    # 变量说明（help_method_by_agent）：`Collaboration` 中当前处理的智能体。
    help_method_by_agent: str = Field(description="What is the agent doing to help you, if applicable?")
    # 变量说明（change_in_plan）：LLM 是否因环境变化而调整原计划的结构化字段。
    change_in_plan: str = Field(description="How does the help from the agent change your plan, if applicable?")
    
class ResponseEvent(BaseModel):
    # 变量说明（epsiode_number）：LLM 结构化输出中的回合编号；字段名沿用原拼写。
    epsiode_number: int = Field(Description="What is the current episode?")
    # 变量说明（timestep）：当前经验在所属回合内的时间步编号。
    timestep: int = Field(Description="What is the current timestep in the episode?")
    # 变量说明（past_events）：`ResponseEvent` 中待批量处理的事件集合。
    past_events: str = Field(Description="Briefly describe the past events in the episode.")
    # 变量说明（current_facing_direction）：`ResponseEvent` 中当前处理的方向。
    current_facing_direction: MaterialType
    # 变量说明（current_inventory）：从经验文本解析出的当前非空背包物品映射。
    current_inventory: list[InventoryItemsCount] = Field(Description="What is in your current inventory? Only list items with item count greater than 0.")
    # 变量说明（collaboration）：LLM 输出中描述能否帮助其他智能体的结构化协作字段。
    collaboration: Collaboration
    # 变量说明（reflection）：LLM 对上一动作、结果和重复行为的结构化反思字段。
    reflection: Reflection
    # 变量说明（goal）：`ResponseEvent` 中当前处理的目标。
    goal: Goal
    # 变量说明（action）：当前智能体准备执行的环境动作。
    action: NextAction
    # 变量说明（summary）：对当前经验、目标或统计结果的精简文本摘要。
    summary: str = Field(Description=(
                                "Summarize the episode, including the timestep, long-term goal, progress, significant events, and plan. "
                                "Explain your actions, the rationale behind your decisions. Treat as if you have done the next actions aleardy. Explain your intended support for other agents (if applicable). What should come next?"
                                "Keep the summary concise and focused on key information, using *past tense* for everything as it serves as a note for future reference. Use clear and plain language. "
                                "Use PAST TENSE!!!\n"
"""
Template:
This is agent [...]. In Episode: [...] ; Timestep: [...]. My inventory contained [...]. In the past, I successfully [...]; I failed to [...]. On collaboration, [...].
I [current action: made/placed/navigated/ate/shared] a [current object] because [...]. This action succeeded/failed/was in progress, becuase it does not exist in my inventory. I planned to work towards [goal] because [...].
"""
                           )
                        )
    #                         "Ssummrize the episode,"
    #                         "including timestep, long term goal, your progress, what happened in the past, and plan. "
    #                         "Include how you plan to help the other agent, if applicable."
    #                         "What action do you decide to do, why (I decided to xxx in this step because xxx)."
    #                         "What should be the next goal and why?"
    #                         "Be clear and concise and include the most important information."
    #                         "The whole summary act as a note for the next episode, so use past tense for everything; Avoid using 'next' to describe action because it would be done by next time step.")
    # )


def _json_schema_instruction() -> str:
    return (
        "Return only one compact valid JSON object. Do not use markdown fences. "
        "Use short strings. Do not add extra keys. "
        "Valid enum values: "
        "ResultType=[success,failure,in_progress]; "
        "ActionType=[noop,move_left,move_right,move_up,move_down,do,sleep,place_stone,place_table,place_furnace,place_plant,make_wood_pickaxe,make_stone_pickaxe,make_iron_pickaxe,Navigator,share]; "
        "MaterialType=[table,furnace,grass,path,sand,lava,tree,water,stone,coal,iron,diamond,player,cow,zombie,skeleton,arrow,plant]; "
        "GoalType=[collect_wood,make_wood_pickaxe,collect_stone,make_stone_pickaxe,collect_iron,make_iron_pickaxe,collect_diamond,place_table,place_furnace,collect_coal,share]; "
        "LongTermGoalType=[make_wood_pickaxe,make_stone_pickaxe,make_iron_pickaxe,place_table,place_furnace,collect_diamond,help_agent]; "
        "NavigationDestinationItems=[tree,water,stone,iron,diamond,coal,grass,table,furnace,not_applicable]; "
        "ShareableItems=[wood,stone,coal,iron,diamond,wood_pickaxe,stone_pickaxe,iron_pickaxe,not_applicable]; "
        "InventoryItems=[wood,stone,coal,iron,diamond,wood_pickaxe,stone_pickaxe,iron_pickaxe]. "
        "Required JSON shape: {"
        "\"epsiode_number\":0,"
        "\"timestep\":0,"
        "\"past_events\":\"short text\","
        "\"current_facing_direction\":\"grass\","
        "\"current_inventory\":[{\"item\":\"wood\",\"count\":1}],"
        "\"collaboration\":{\"target_agent_to_help\":-1,\"target_agent_need\":\"not_applicable\",\"help_method\":\"none\",\"can_help_now\":\"failure\",\"being_helped_by_agent\":-1,\"help_method_by_agent\":\"none\",\"change_in_plan\":\"none\"},"
        "\"reflection\":{\"vision\":[\"grass\"],\"last_action\":\"noop\",\"last_action_result\":\"failure\",\"last_action_result_reflection\":\"short text\",\"last_action_repeated_reflection\":\"short text\"},"
        "\"goal\":{\"ultimate_goal\":\"collect_diamond\",\"long_term_goal\":\"place_table\",\"long_term_goal_subgoals\":\"short text\",\"long_term_goal_progress\":\"collect_wood\",\"long_term_goal_status\":\"in_progress\",\"current_goal\":\"collect_wood\",\"current_goal_reason\":\"short text\",\"current_goal_status\":\"in_progress\"},"
        "\"action\":{\"next_action\":\"Navigator\",\"next_action_reason\":\"short text\",\"next_action_prerequisites_status\":\"in_progress\",\"next_action_prerequisites\":\"short text\",\"final_next_action\":\"Navigator\",\"final_next_action_reason\":\"short text\",\"final_next_action_status\":\"in_progress\",\"final_target_material_to_collect\":\"tree\",\"final_target_material_to_share\":\"not_applicable\",\"final_target_agent_id\":-1},"
        "\"summary\":\"one short sentence\""
        "}."
    )


def _with_json_schema_instruction(messages):
    # 变量说明（messages）：按角色排列、准备提交给聊天模型的消息列表。
    # 变量说明（instruction）：追加到模型消息中的 JSON 输出格式约束指令。
    instruction = {"role": "system", "content": _json_schema_instruction()}
    return [instruction, *messages]


def _normalize_result_status(value):
    # 变量说明（value）：当前配置项、字段或计算结果对应的值。
    if not isinstance(value, str):
        return value
    # 变量说明（normalized）：去除大小写、空格等格式差异后的规范化值。
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    # 变量说明（aliases）：非标准模型输出到项目标准枚举值的别名映射。
    aliases = {
        "successful": "success",
        "completed": "success",
        "complete": "success",
        "done": "success",
        "satisfied": "success",
        "ready": "success",
        "met": "success",
        "true": "success",
        "yes": "success",
        "failed": "failure",
        "unsuccessful": "failure",
        "unsatisfied": "failure",
        "not_satisfied": "failure",
        "not_ready": "failure",
        "unmet": "failure",
        "blocked": "failure",
        "false": "failure",
        "no": "failure",
        "not_applicable": "failure",
        "pending": "in_progress",
        "ongoing": "in_progress",
        "incomplete": "in_progress",
        "not_completed": "in_progress",
    }
    return aliases.get(normalized, normalized)


def _normalize_action(value):
    # 变量说明（value）：当前配置项、字段或计算结果对应的值。
    if not isinstance(value, str):
        return value
    # 变量说明（normalized）：去除大小写、空格等格式差异后的规范化值。
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    # 变量说明（aliases）：非标准模型输出到项目标准枚举值的别名映射。
    aliases = {
        "navigate": "Navigator",
        "navigator": "Navigator",
        "navigate_to": "Navigator",
        "collect": "do",
        "gather": "do",
        "mine": "do",
        "wait": "sleep",
    }
    return aliases.get(normalized, normalized)


def _normalize_response_event_data(data):
    # 变量说明（data）：与当前节点、文件或处理步骤关联的结构化数据。
    # 变量说明（result_status_paths）：`_normalize_response_event_data` 中待批量处理的结果、路径集合。
    result_status_paths = (
        ("collaboration", "can_help_now"),
        ("reflection", "last_action_result"),
        ("goal", "long_term_goal_status"),
        ("goal", "current_goal_status"),
        ("action", "next_action_prerequisites_status"),
        ("action", "final_next_action_status"),
    )
    # 变量说明（field）：当前处理、校验或收集的结构化字段名称。
    # 变量说明（section）：当前正在规范化的结构化模型输出分区名称。
    for section, field in result_status_paths:
        # 变量说明（section_data）：结构化模型输出中当前分区的字段字典。
        section_data = data.get(section)
        if isinstance(section_data, dict) and field in section_data:
            section_data[field] = _normalize_result_status(section_data[field])

    # 变量说明（reflection_data）：待规范化的 LLM 反思字段字典。
    reflection_data = data.get("reflection")
    if isinstance(reflection_data, dict) and "last_action" in reflection_data:
        reflection_data["last_action"] = _normalize_action(reflection_data["last_action"])

    # 变量说明（action_data）：`_normalize_response_event_data` 中当前处理的动作。
    action_data = data.get("action")
    if isinstance(action_data, dict):
        for field in ("next_action", "final_next_action"):
            if field in action_data:
                action_data[field] = _normalize_action(action_data[field])

    # 变量说明（goal_data）：`_normalize_response_event_data` 中当前处理的目标。
    goal_data = data.get("goal")
    if isinstance(goal_data, dict):
        # 变量说明（goal）：`_normalize_response_event_data` 中当前处理的目标。
        # 变量说明（valid_goals）：`_normalize_response_event_data` 中待批量处理的目标集合。
        valid_goals = {goal.value for goal in GoalType}
        # 变量说明（progress）：当前目标、下载或实验任务的完成进度。
        progress = goal_data.get("long_term_goal_progress")
        if isinstance(progress, str):
            progress = progress.strip().lower().replace("-", "_").replace(" ", "_")
        if progress not in valid_goals:
            # 变量说明（current_goal）：智能体目前正在直接推进的短期目标。
            current_goal = goal_data.get("current_goal")
            # 变量说明（long_term_goal）：`_normalize_response_event_data` 中当前处理的目标。
            long_term_goal = goal_data.get("long_term_goal")
            if current_goal in valid_goals:
                progress = current_goal
            elif long_term_goal in valid_goals:
                progress = long_term_goal
            else:
                progress = GoalType.COLLECT_WOOD.value
        goal_data["long_term_goal_progress"] = progress
    return data


def _parse_response_event_content(content: str) -> ResponseEvent:
    # 变量说明（content）：当前消息、文件或模型响应的主体文本。
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    # 变量说明（data）：与当前节点、文件或处理步骤关联的结构化数据。
    data = json.loads(content)
    if "epsiode_number" not in data and "episode_number" in data:
        data["epsiode_number"] = data.pop("episode_number")
    data = _normalize_response_event_data(data)
    return ResponseEvent.model_validate(data)
    
def get_completion(messages, model=None):
    # 变量说明（messages）：按角色排列、准备提交给聊天模型的消息列表。
    # 变量说明（model）：`get_completion` 中当前处理的模型。
    # 变量说明（max_retries）：模型请求失败或结构校验失败时的最大重试次数。
    max_retries = _env_int("LLM_MAX_RETRIES", 3)
    # 变量说明（current_messages）：当前重试轮次准备提交给聊天模型的完整消息列表。
    current_messages = list(messages)
    # 变量说明（attempt）：当前模型请求重试次数的零基索引。
    for attempt in range(max_retries):
        try:
            # 变量说明（config）：根据环境变量解析得到的大模型供应商配置。
            config = get_llm_config()
            if config.provider == "qwen":
                # 变量说明（response）：当前 API 调用、任务或模型推理产生的响应对象。
                response = get_client().chat.completions.create(
                    model=model or config.chat_model,
                    messages=_with_json_schema_instruction(current_messages),
                    response_format={"type": "json_object"},
                    max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
                )
                # 变量说明（content）：当前消息、文件或模型响应的主体文本。
                content = response.choices[0].message.content
                try:
                    return _parse_response_event_content(content)
                # 变量说明（e）：当前循环正在处理的事件、边或异常元素。
                except (ValidationError, json.JSONDecodeError) as e:
                    if attempt < max_retries - 1:
                        print_color(
                            f"Qwen response validation failed, retrying with correction... (attempt {attempt+1})",
                            color="yellow",
                        )
                        current_messages = [
                            *current_messages,
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    "The previous JSON did not match the required schema. "
                                    f"Validation error: {e}. Return corrected JSON only."
                                ),
                            },
                        ]
                        continue
                    raise
            response = get_client().beta.chat.completions.parse(
                model=model or config.chat_model,
                messages=current_messages,
                response_format=ResponseEvent,
            )
            return response.choices[0].message.parsed
        except (APITimeoutError, TimeoutError) as e:
            if attempt < max_retries - 1:
                print_color(f"Timeout occurred in get_completion, retrying in 10 seconds... (attempt {attempt+1})", color="yellow")
                time.sleep(10)
            else:
                print_color("Timeout occurred in get_completion, max retries reached.", color="red")
                raise

class TextModel:
    def __init__(self):
        # 变量说明（self.tokenizer）：用于估算提示词 token 数量的 tiktoken 编码器。
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # self.tokenizer = None
    
    def __call__(self, text) -> Any:
        # 变量说明（text）：`__call__` 中当前处理的文本。
        # 变量说明（normalized_text）：`__call__` 中当前处理的文本。
        normalized_text = self.nomralize(text)
        # 变量说明（token_len）：当前文本经分词器编码后的 token 数量。
        token_len = self.check_token_len(text)
        if token_len > 8192:
            print_color(f"TextEmbbeding Warning: The token length of the text is {token_len}, it may cause the model to fail to generate the response.", color="red")
        return self.ada_embedding(normalized_text)
    
    def ada_embedding(self, text):
        # 变量说明（text）：`ada_embedding` 中当前处理的文本。
        # 变量说明（config）：根据环境变量解析得到的大模型供应商配置。
        config = get_llm_config()
        # 变量说明（response）：当前 API 调用、任务或模型推理产生的响应对象。
        response = get_client().embeddings.create(input=[text], model=config.embedding_model)
        return response.data[0].embedding
        # return ''
    def check_token_len(self, text):
        # 变量说明（text）：`check_token_len` 中当前处理的文本。
        return len(self.tokenizer.encode(text))
        # return 0
        
    @staticmethod
    def nomralize(s) -> Any:
        # 变量说明（s）：当前处理的字符串、尺寸或序列元素。
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,","",s)
        # remove all instances of multiple spaces
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()
        return s
