"""Check model profiles: environment selection, key lookup, records and cost, offline."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_system import model_profiles
from memory_system.model_profiles import (
    ProfileError, episode_cost, load_profiles, profile_env, profile_record,
)

SETTINGS = {
    "currency": "CNY",
    "runtime": {"LLM_MAX_RETRIES": 2},
    "shared_generate_kwargs": {"temperature": 0.5},
    "embedding": {"api_key_env": "QWEN_API_KEY", "base_url": "https://embed.invalid/v1",
                  "model": "shared-embedding", "price_per_million_tokens": 0.5},
    "profiles": {
        "qwen": {"provider": "qwen", "model": "qwen-vision", "base_url": "https://qwen.invalid/v1",
                 "api_key_env": "QWEN_API_KEY", "input_price_per_million_tokens": 2,
                 "output_price_per_million_tokens": 8, "generate_kwargs": {}},
        "deepseek": {"provider": "compatible", "model": "deepseek-vision", "base_url": "https://ds.invalid/v1",
                     "api_key_env": "DEEPSEEK_API_KEY", "tier": "flash",
                     "input_price_per_million_tokens": 1, "output_price_per_million_tokens": 2,
                     "generate_kwargs": {"temperature": 0.2}},
        "empty": {"provider": "compatible", "model": "", "base_url": "https://x.invalid/v1",
                  "api_key_env": "EMPTY_API_KEY"},
    },
}


class ModelProfileChecks(unittest.TestCase):
    def setUp(self):
        self.local = tempfile.TemporaryDirectory(prefix="model_profiles_")
        self.addCleanup(self.local.cleanup)
        directory = Path(self.local.name)
        self.keys = directory / "api_keys.json"
        self.csv = directory / "qwen_api_key.csv"
        for name, value in (("KEYS_PATH", self.keys), ("QWEN_CSV_PATH", self.csv)):
            patcher = patch.object(model_profiles, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = patch.dict(os.environ, {"QWEN_API_KEY": "", "DEEPSEEK_API_KEY": "", "EMPTY_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)
        for name in ("QWEN_API_KEY", "DEEPSEEK_API_KEY", "EMPTY_API_KEY"):
            os.environ.pop(name)

    def test_compatible_profile_sets_chat_shared_embedding_and_json_mode(self):
        self.keys.write_text(json.dumps({"DEEPSEEK_API_KEY": "ds-key", "QWEN_API_KEY": "qwen-key"}))
        env = profile_env("deepseek", SETTINGS)
        self.assertEqual((env["LLM_PROVIDER"], env["LLM_CHAT_MODEL"], env["LLM_API_KEY"]),
                         ("compatible", "deepseek-vision", "ds-key"))
        self.assertEqual((env["EMBEDDING_MODEL"], env["EMBEDDING_API_KEY"]), ("shared-embedding", "qwen-key"))
        self.assertEqual(env["LLM_JSON_MODE"], "json_object")
        # Profile settings override the shared ones.
        self.assertEqual(json.loads(env["LLM_GENERATE_KWARGS"]), {"temperature": 0.2})
        self.assertEqual((env["MODEL_PROFILE"], env["LLM_MAX_RETRIES"]), ("deepseek", "2"))

    def test_qwen_key_falls_back_to_the_existing_csv_and_env_wins(self):
        self.csv.write_text("id,value\napiKey,csv-key\nopenAiCompatible,https://q.invalid\n", encoding="utf-8")
        env = profile_env("qwen", SETTINGS)
        self.assertEqual((env["LLM_PROVIDER"], env["QWEN_API_KEY"], env["QWEN_CHAT_MODEL"]),
                         ("qwen", "csv-key", "qwen-vision"))
        os.environ["QWEN_API_KEY"] = "env-key"
        self.assertEqual(profile_env("qwen", SETTINGS)["QWEN_API_KEY"], "env-key")

    def test_missing_model_key_or_profile_fails_before_running(self):
        with self.assertRaisesRegex(ProfileError, "no model name"):
            profile_env("empty", SETTINGS)
        with self.assertRaisesRegex(ProfileError, "DEEPSEEK_API_KEY"):
            profile_env("deepseek", SETTINGS)
        with self.assertRaisesRegex(ProfileError, "Unknown model profile"):
            profile_env("nobody", SETTINGS)

    def test_record_has_no_secret_and_cost_needs_every_price(self):
        record = profile_record("deepseek", SETTINGS)
        self.assertEqual((record["chat_model"], record["model_tier"]), ("deepseek-vision", "flash"))
        self.assertNotIn("key", json.dumps(record).lower().replace("api_key_env", ""))
        usage = {"chat_input_tokens": 1_000_000, "chat_output_tokens": 500_000, "embedding_tokens": 2_000_000}
        self.assertAlmostEqual(episode_cost(usage, record), 1 + 1 + 1)
        self.assertIsNone(episode_cost(usage, {**record, "output_price_per_million_tokens": None}))

    def test_committed_profiles_cover_the_five_table_three_vendors(self):
        settings = load_profiles()
        self.assertEqual(set(settings["profiles"]), {"qwen", "deepseek", "kimi", "glm", "gpt"})
        for name, profile in settings["profiles"].items():
            with self.subTest(profile=name):
                self.assertIn(profile["provider"], ("qwen", "compatible"))
                self.assertTrue(profile["base_url"].startswith("https://"))
                self.assertNotIn("api_key", {key for key in profile if key != "api_key_env"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
