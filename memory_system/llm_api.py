from typing import Any
from dataclasses import dataclass
from functools import lru_cache
import asyncio
import json
import os
import re
import threading
from collections import Counter
from time import perf_counter
import tiktoken
from .utils import print_color
from pydantic import BaseModel, Field, ValidationError
from typing import Union, Literal
from openai import NOT_GIVEN, APITimeoutError, AsyncAzureOpenAI

from agentscope.embedding import OpenAITextEmbedding
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from enum import Enum


MODEL_USAGE = Counter()


def get_model_usage():
    return dict(MODEL_USAGE)


async def measured_model_call(kind, model, *args, **kwargs):
    MODEL_USAGE[kind + '_requests'] += 1
    started = perf_counter()
    try:
        response = await model(*args, **kwargs)
    finally:
        MODEL_USAGE['model_seconds'] += perf_counter() - started
    if response.usage is not None:
        if kind == 'chat':
            MODEL_USAGE['chat_input_tokens'] += response.usage.input_tokens
            MODEL_USAGE['chat_output_tokens'] += response.usage.output_tokens
        else:
            MODEL_USAGE['embedding_tokens'] += response.usage.tokens or 0
    return response


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is not configured for runtime calls."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    chat_model: str
    embedding_model: str
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None


def _env(name, default=None):
    value = os.getenv(name, default)
    if isinstance(value, str):
        value = value.strip()
    return value or None


def _env_int(name, default):
    value = _env(name)
    if value is None:
        return default
    return int(value)


def _env_float(name, default):
    value = _env(name)
    if value is None:
        return default
    return float(value)


def get_llm_config() -> LLMConfig:
    provider = (_env("LLM_PROVIDER", "azure") or "azure").lower()

    if provider == "azure":
        endpoint = _env("AZURE_OPENAI_ENDPOINT")
        api_key = _env("AZURE_OPENAI_API_KEY")
        api_version = _env("AZURE_OPENAI_API_VERSION")
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

    if provider == "compatible":
        # Any OpenAI-compatible chat endpoint (DeepSeek, Kimi, GLM, ...).
        missing = [name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_CHAT_MODEL") if not _env(name)]
        if missing:
            raise LLMConfigurationError(
                "An OpenAI-compatible provider is selected but missing environment variables: "
                + ", ".join(missing)
            )
        return LLMConfig(
            provider=provider,
            chat_model=_env("LLM_CHAT_MODEL"),
            embedding_model=_env("EMBEDDING_MODEL", ""),
            openai_api_key=_env("LLM_API_KEY"),
            openai_base_url=_env("LLM_BASE_URL"),
        )

    raise LLMConfigurationError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'azure', 'openai', 'qwen', or 'compatible'."
    )


EMBEDDING_OVERRIDE_VARIABLES = ("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL")


def get_embedding_override() -> LLMConfig | None:
    """Embedding endpoint configured separately from the chat provider.

    Experiments compare chat models while keeping one embedding model, so any
    ``EMBEDDING_*`` variable switches embedding calls to that endpoint.
    """
    values = {name: _env(name) for name in EMBEDDING_OVERRIDE_VARIABLES}
    if not any(values.values()):
        return None
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise LLMConfigurationError(
            "A separate embedding endpoint is partly configured; missing: " + ", ".join(missing)
        )
    return LLMConfig(
        provider="embedding",
        chat_model="",
        embedding_model=values["EMBEDDING_MODEL"],
        openai_api_key=values["EMBEDDING_API_KEY"],
        openai_base_url=values["EMBEDDING_BASE_URL"],
    )


def json_mode(config: LLMConfig) -> str:
    """``json_object``: ask for JSON, validate locally and retry with the error.
    ``structured``: provider-enforced structured output (OpenAI and Azure)."""
    default = "structured" if config.provider in ("openai", "azure") else "json_object"
    mode = (_env("LLM_JSON_MODE", default) or default).lower()
    if mode not in ("json_object", "structured"):
        raise LLMConfigurationError("LLM_JSON_MODE must be 'json_object' or 'structured'.")
    return mode


def generate_kwargs() -> dict[str, Any]:
    """Extra chat request fields shared by every call, e.g. temperature or
    a provider switch that turns thinking off, given as a JSON object."""
    value = _env("LLM_GENERATE_KWARGS")
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LLMConfigurationError(f"LLM_GENERATE_KWARGS is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMConfigurationError("LLM_GENERATE_KWARGS must be a JSON object.")
    return parsed


def describe_llm_configuration() -> dict[str, Any]:
    provider = (_env("LLM_PROVIDER", "azure") or "azure").lower()
    try:
        config = get_llm_config()
        embedding = get_embedding_override()
        return {
            "configured": True,
            "provider": config.provider,
            "chat_model": config.chat_model,
            "embedding_model": embedding.embedding_model if embedding else config.embedding_model,
            "embedding_endpoint": "separate" if embedding else "chat provider",
            "json_mode": json_mode(config),
            "missing": [],
        }
    except LLMConfigurationError as exc:
        missing_by_provider = {
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
            "compatible": [name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_CHAT_MODEL")
                           if not _env(name)],
        }
        return {
            "configured": False,
            "provider": provider,
            "chat_model": _env("AZURE_OPENAI_CHAT_DEPLOYMENT", _env("OPENAI_MODEL", "gpt-4o"))
            if provider == "azure"
            else _env("QWEN_CHAT_MODEL", _env("OPENAI_MODEL", "qwen3.7-plus"))
            if provider == "qwen"
            else _env("LLM_CHAT_MODEL")
            if provider == "compatible"
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


_loop = None
_loop_lock = threading.Lock()


def run_async(coro):
    """Run an AgentScope coroutine from synchronous code.

    AgentScope models and agents are async, while the simulation loop and the
    pygame GUI are synchronous. All coroutines share one background event loop
    so the cached async clients stay bound to a single loop.
    """
    global _loop
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if _loop is not None and running_loop is _loop:
        coro.close()
        raise RuntimeError(
            "run_async cannot block the AgentScope event loop; await the coroutine instead."
        )
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever, name="agentscope-loop", daemon=True
            ).start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def _client_kwargs(config: LLMConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": _env_float("LLM_TIMEOUT_SECONDS", 120.0)}
    if config.provider == "azure":
        kwargs["azure_endpoint"] = config.azure_endpoint
        kwargs["api_version"] = config.azure_api_version
    elif config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    return kwargs


@lru_cache(maxsize=1)
def get_chat_model() -> OpenAIChatModel:
    """AgentScope chat model for the configured provider.

    Azure, OpenAI and Qwen (DashScope OpenAI-compatible endpoint) all go
    through AgentScope's OpenAIChatModel.
    """
    config = get_llm_config()
    is_azure = config.provider == "azure"
    return OpenAIChatModel(
        model_name=config.chat_model,
        api_key=config.azure_api_key if is_azure else config.openai_api_key,
        stream=False,
        client_type="azure" if is_azure else "openai",
        client_kwargs=_client_kwargs(config),
        generate_kwargs=generate_kwargs(),
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAITextEmbedding:
    """AgentScope text embedding model: the separate ``EMBEDDING_*`` endpoint
    when configured, otherwise the chat provider's own embedding model."""
    config = get_embedding_override()
    if config is None:
        config = get_llm_config()
        if config.provider == "compatible":
            raise LLMConfigurationError(
                "The OpenAI-compatible chat provider has no embedding endpoint here; set "
                + ", ".join(EMBEDDING_OVERRIDE_VARIABLES) + "."
            )
    is_azure = config.provider == "azure"
    api_key = config.azure_api_key if is_azure else config.openai_api_key
    dimensions = _env_int("LLM_EMBEDDING_DIMENSIONS", None)
    model = OpenAITextEmbedding(
        api_key=api_key,
        model_name=config.embedding_model,
        # AgentScope always forwards `dimensions`; NOT_GIVEN makes the OpenAI
        # SDK omit it, so each model keeps its native size (and models that
        # reject the parameter, e.g. text-embedding-ada-002, keep working).
        dimensions=NOT_GIVEN if dimensions is None else dimensions,
        **({} if is_azure else _client_kwargs(config)),
    )
    if is_azure:
        # OpenAITextEmbedding only builds a plain OpenAI client.
        model.client = AsyncAzureOpenAI(api_key=api_key, **_client_kwargs(config))
    return model


class ResultType(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    
class ActionType(str, Enum):
    noop = "noop"
    move_left = "move_left"
    move_right = "move_right"
    move_up = "move_up"
    move_down = "move_down"
    do = "do"
    sleep = "sleep"
    place_stone = "place_stone"
    place_table = "place_table"
    place_furnace = "place_furnace"
    place_plant = "place_plant"
    make_wood_pickaxe = "make_wood_pickaxe"
    make_stone_pickaxe = "make_stone_pickaxe"
    make_iron_pickaxe = "make_iron_pickaxe"
    Navigator = "Navigator"
    share = "share"
    
class GoalType(str, Enum):
    COLLECT_WOOD = "collect_wood"
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    COLLECT_STONE = "collect_stone"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    COLLECT_IRON = "collect_iron"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    COLLECT_DIAMOND = "collect_diamond"
    
    PLACE_TABLE = "place_table"
    PLACE_FURNACE = "place_furnace"
    COLLECT_COAL = "collect_coal"
    SHARE = "share"

class LongTermGoalType(str, Enum):
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    PLACE_TABLE = "place_table"
    PLACE_FURNACE = "place_furnace"
    COLLECT_DIAMOND = "collect_diamond"
    HELP_AGENT = "help_agent"
    
class MaterialType(str, Enum):
    TABLE = "table"
    FURNACE = "furnace"
    GRASS = "grass"
    PATH = "path"
    SAND = "sand"
    LAVA = "lava"
    TREE = "tree"
    WATER = "water"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    PLAYER = "player"
    COW = "cow"
    ZOMBIE = "zombie"
    SKELETON = "skeleton"
    ARROW = "arrow"
    PLANT = "plant"
    
class Reflection(BaseModel):
    vision: list[MaterialType] = Field(description="List of materials you see around you.")
    last_action: ActionType #= Field(description="Your last action.")
    last_action_result: ResultType
    last_action_result_reflection: str #= Field(description="Reflection on the last action.")
    last_action_repeated_reflection: str = Field(description="Did you repeat the last action? If so, why?")

class Goal(BaseModel):
    ultimate_goal: LongTermGoalType = Field(description="What is your ultimate goal?")
    
    long_term_goal: LongTermGoalType = Field(description="Working towards the ultimate goal, what should be your next goal?")
    long_term_goal_subgoals: str = Field(description="What are the subgoals to complete the long term goal?")
    long_term_goal_progress: GoalType = Field(description="What is the progress of the long term goal?")
    long_term_goal_status: ResultType
    
    current_goal: GoalType = Field(description="The current goal that you are working on.")
    current_goal_reason: str #= Field(description="Why do you choose the current goal?")
    current_goal_status: ResultType
    
class NavigationDestinationItems(str, Enum):
    TREE = "tree"
    WATER = "water"
    STONE = "stone"
    IRON = "iron"
    DIAMOND = "diamond"
    COAL = "coal"
    GRASS = "grass"
    # COW = 'cow'
    TABLE = "table"
    FURNACE = "furnace"
    NOT_APPICABLE = "not_applicable"
    
class ShareableItems(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    WOOD_PICKAXE = "wood_pickaxe"
    STONE_PICKAXE = "stone_pickaxe"
    IRON_PICKAXE = "iron_pickaxe"
    # food
    # water
    NOT_APPLICABLE = "not_applicable"

class InventoryItems(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    COAL = "coal"
    IRON = "iron"
    DIAMOND = "diamond"
    WOOD_PICKAXE = "wood_pickaxe"
    STONE_PICKAXE = "stone_pickaxe"
    IRON_PICKAXE = "iron_pickaxe"
 
class InventoryItemsCount(BaseModel):
    item: InventoryItems
    count: int

# Model for planning the next action
class NextAction(BaseModel):
    next_action: ActionType = Field(description="What is the next action you plan to take?")
    next_action_reason: str # = Field(description="Why do you think this shoud be the next action?")
    next_action_prerequisites_status: ResultType = Field(description="Are the prerequisites met?")
    next_action_prerequisites: str = Field(description="What prerequisites are not met?")
    final_next_action: ActionType = Field(description="What is your final decision on next action.")
    final_next_action_reason: str # = Field(description="Reason for choosing this action.")
    final_next_action_status: ResultType # = Field(description="What is the status of the final decision?")
    final_target_material_to_collect: NavigationDestinationItems = Field(description="Navigate to where?")
    final_target_material_to_share: ShareableItems = Field(description="Share what?")
    final_target_agent_id: int = Field(description="Which agent to share with, if applicable, or return -1.")
    
class Collaboration(BaseModel):
    target_agent_to_help: int = Field(description="Which agent should you help, if applicable?")
    target_agent_need: ShareableItems = Field(description="What does the target agent need, if applicable?")
    help_method: str = Field(description="What can you do to help the agent, if applicable?")
    can_help_now: ResultType = Field(description="Can you help the agent now? Do you have the resources in inventory?")
    being_helped_by_agent: int = Field(description="Which agent is helping you, if applicable?")
    help_method_by_agent: str = Field(description="What is the agent doing to help you, if applicable?")
    change_in_plan: str = Field(description="How does the help from the agent change your plan, if applicable?")
    
class ResponseEvent(BaseModel):
    epsiode_number: int = Field(description="What is the current episode?")
    timestep: int = Field(description="What is the current timestep in the episode?")
    past_events: str = Field(description="Briefly describe the past events in the episode.")
    current_facing_direction: MaterialType
    current_inventory: list[InventoryItemsCount] = Field(description="What is in your current inventory? Only list items with item count greater than 0.")
    collaboration: Collaboration
    reflection: Reflection
    goal: Goal
    action: NextAction
    summary: str = Field(description=(
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
    instruction = {"role": "system", "content": _json_schema_instruction()}
    return [instruction, *messages]


def _normalize_result_status(value):
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
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
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
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
    result_status_paths = (
        ("collaboration", "can_help_now"),
        ("reflection", "last_action_result"),
        ("goal", "long_term_goal_status"),
        ("goal", "current_goal_status"),
        ("action", "next_action_prerequisites_status"),
        ("action", "final_next_action_status"),
    )
    for section, field in result_status_paths:
        section_data = data.get(section)
        if isinstance(section_data, dict) and field in section_data:
            section_data[field] = _normalize_result_status(section_data[field])

    reflection_data = data.get("reflection")
    if isinstance(reflection_data, dict) and "last_action" in reflection_data:
        reflection_data["last_action"] = _normalize_action(reflection_data["last_action"])

    action_data = data.get("action")
    if isinstance(action_data, dict):
        for field in ("next_action", "final_next_action"):
            if field in action_data:
                action_data[field] = _normalize_action(action_data[field])

    goal_data = data.get("goal")
    if isinstance(goal_data, dict):
        valid_goals = {goal.value for goal in GoalType}
        progress = goal_data.get("long_term_goal_progress")
        if isinstance(progress, str):
            progress = progress.strip().lower().replace("-", "_").replace(" ", "_")
        if progress not in valid_goals:
            current_goal = goal_data.get("current_goal")
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
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    data = json.loads(content)
    if "epsiode_number" not in data and "episode_number" in data:
        data["epsiode_number"] = data.pop("episode_number")
    data = _normalize_response_event_data(data)
    return ResponseEvent.model_validate(data)
    
async def _to_openai_messages(messages) -> list[dict]:
    """Accept AgentScope Msg objects or already formatted OpenAI dicts."""
    messages = list(messages)
    if messages and isinstance(messages[0], Msg):
        return await OpenAIChatFormatter().format(messages)
    return messages


def _response_text(response) -> str:
    return "".join(
        block["text"] for block in response.content if block["type"] == "text"
    )


async def aget_completion(messages, model=None) -> ResponseEvent:
    """Ask the AgentScope chat model for one structured ResponseEvent."""
    max_retries = _env_int("LLM_MAX_RETRIES", 3)
    if max_retries < 1:
        raise LLMConfigurationError("LLM_MAX_RETRIES must be at least 1.")
    current_messages = await _to_openai_messages(messages)
    for attempt in range(max_retries):
        try:
            config = get_llm_config()
            chat_model = get_chat_model()
            overrides = {"model": model} if model else {}
            if json_mode(config) == "json_object":
                response = await measured_model_call('chat', chat_model,
                    _with_json_schema_instruction(current_messages),
                    response_format={"type": "json_object"},
                    max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
                    **overrides,
                )
                content = _response_text(response)
                try:
                    return _parse_response_event_content(content)
                except (ValidationError, json.JSONDecodeError) as e:
                    if attempt < max_retries - 1:
                        MODEL_USAGE['chat_json_retries'] += 1
                        print_color(
                            f"Response validation failed, retrying with correction... (attempt {attempt+1})",
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
            response = await measured_model_call('chat', chat_model,
                current_messages,
                structured_model=ResponseEvent,
                **overrides,
            )
            return ResponseEvent.model_validate(response.metadata)
        except (APITimeoutError, TimeoutError) as e:
            if attempt < max_retries - 1:
                print_color(f"Timeout occurred in get_completion, retrying in 10 seconds... (attempt {attempt+1})", color="yellow")
                await asyncio.sleep(10)
            else:
                print_color("Timeout occurred in get_completion, max retries reached.", color="red")
                raise


def get_completion(messages, model=None) -> ResponseEvent:
    """Synchronous wrapper around aget_completion."""
    return run_async(aget_completion(messages, model))


class TextModel:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # self.tokenizer = None
    
    def __call__(self, text) -> Any:
        normalized_text = self.nomralize(text)
        token_len = self.check_token_len(text)
        if token_len > 8192:
            print_color(f"TextEmbbeding Warning: The token length of the text is {token_len}, it may cause the model to fail to generate the response.", color="red")
        return self.ada_embedding(normalized_text)
    
    def ada_embedding(self, text):
        response = run_async(measured_model_call('embedding', get_embedding_model(), [text]))
        return response.embeddings[0]
        # return ''
    def check_token_len(self, text):
        return len(self.tokenizer.encode(text))
        # return 0
        
    @staticmethod
    def nomralize(s) -> Any:
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,","",s)
        # remove all instances of multiple spaces
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()
        return s
