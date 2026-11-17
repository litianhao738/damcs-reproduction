"""Named chat-model settings shared by the experiment tables.

``config/model_profiles.json`` (committed) holds model names, endpoints and
prices. Keys stay local: an environment variable, ``local_config/api_keys.json``
(``{"DEEPSEEK_API_KEY": "..."}``), or, for Qwen, the existing
``local_config/qwen_api_key.csv``. Every profile uses the same embedding
endpoint and the same JSON-and-validate output path, so profiles differ only
in the chat model.
"""

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "config" / "model_profiles.json"
KEYS_PATH = ROOT / "local_config" / "api_keys.json"
QWEN_CSV_PATH = ROOT / "local_config" / "qwen_api_key.csv"


class ProfileError(ValueError):
    """A profile is unknown, incomplete, or has no key available."""


def load_profiles(path=PROFILES_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _local_keys():
    if KEYS_PATH.exists():
        return json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    return {}


def _qwen_csv_key():
    if not QWEN_CSV_PATH.exists():
        return None
    with QWEN_CSV_PATH.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row.get("id") == "apiKey":
            values = [value for key, value in row.items() if key != "id"]
            return values[0].strip() if values and values[0] else None
    return None


def resolve_key(variable):
    key = os.environ.get(variable) or _local_keys().get(variable)
    if not key and variable in ("QWEN_API_KEY", "DASHSCOPE_API_KEY"):
        key = _qwen_csv_key()
    if not key:
        raise ProfileError(
            f"No API key for {variable}: set the environment variable or add it to {KEYS_PATH}."
        )
    return key


def profile_env(name, settings=None):
    """Environment variables that select profile ``name`` for one run."""
    settings = settings or load_profiles()
    profiles = settings["profiles"]
    if name not in profiles:
        raise ProfileError(f"Unknown model profile '{name}'. Known: {', '.join(profiles)}")
    profile = profiles[name]
    if not profile.get("model"):
        raise ProfileError(f"Model profile '{name}' has no model name; fill it in {PROFILES_PATH}.")
    key = resolve_key(profile["api_key_env"])
    if profile["provider"] == "qwen":
        env = {"LLM_PROVIDER": "qwen", "QWEN_API_KEY": key, "QWEN_BASE_URL": profile["base_url"],
               "QWEN_CHAT_MODEL": profile["model"]}
    elif profile["provider"] == "compatible":
        env = {"LLM_PROVIDER": "compatible", "LLM_API_KEY": key, "LLM_BASE_URL": profile["base_url"],
               "LLM_CHAT_MODEL": profile["model"]}
    else:
        raise ProfileError(f"Profile '{name}' uses unsupported provider '{profile['provider']}'.")
    embedding = settings["embedding"]
    env.update({
        "EMBEDDING_API_KEY": resolve_key(embedding["api_key_env"]),
        "EMBEDDING_BASE_URL": embedding["base_url"],
        "EMBEDDING_MODEL": embedding["model"],
        "LLM_JSON_MODE": "json_object",
        "LLM_GENERATE_KWARGS": json.dumps({**settings.get("shared_generate_kwargs", {}),
                                           **profile.get("generate_kwargs", {})}),
        "MODEL_PROFILE": name,
    })
    env.update({key: str(value) for key, value in settings.get("runtime", {}).items()})
    return env


def profile_record(name, settings=None):
    """Non-secret facts about a profile, stored with every run."""
    settings = settings or load_profiles()
    profile = settings["profiles"][name]
    embedding = settings["embedding"]
    return {
        "model_profile": name, "chat_model": profile["model"], "model_tier": profile.get("tier", ""),
        "base_url": profile["base_url"], "currency": settings.get("currency"),
        "input_price_per_million_tokens": profile.get("input_price_per_million_tokens"),
        "output_price_per_million_tokens": profile.get("output_price_per_million_tokens"),
        "embedding_model": embedding["model"],
        "embedding_price_per_million_tokens": embedding.get("price_per_million_tokens"),
        "generate_kwargs": {**settings.get("shared_generate_kwargs", {}), **profile.get("generate_kwargs", {})},
    }


def episode_cost(usage, record):
    """Cost from token counts and registered prices; None while a price is missing."""
    prices = (record.get("input_price_per_million_tokens"), record.get("output_price_per_million_tokens"),
              record.get("embedding_price_per_million_tokens"))
    if any(price is None for price in prices):
        return None
    tokens = (usage.get("chat_input_tokens", 0), usage.get("chat_output_tokens", 0),
              usage.get("embedding_tokens", 0))
    return sum(count * price / 1_000_000 for count, price in zip(tokens, prices))
