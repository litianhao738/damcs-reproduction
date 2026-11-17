"""Check AgentScope integration with local HTTP responses and no LLM requests."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from PIL import Image
from agentscope.agent import AgentBase
from agentscope.message import Msg

from memory_system import llm_api
from memory_system.agent import ActionStatus, Agent, HumanAgent
from utils import AgentThinkingProcessor


class AgentScopeIntegrationChecks(unittest.TestCase):
    def setUp(self):
        env_patch = patch.dict(os.environ, {
            "LLM_PROVIDER": "qwen",
            "OPENAI_API_KEY": "offline-test-key",
            "OPENAI_BASE_URL": "https://offline.invalid/v1",
            "QWEN_API_KEY": "offline-test-key",
            "QWEN_BASE_URL": "https://offline.invalid/v1",
            "QWEN_CHAT_MODEL": "offline-chat",
            "AZURE_OPENAI_API_KEY": "offline-test-key",
            "AZURE_OPENAI_ENDPOINT": "https://offline.invalid",
            "AZURE_OPENAI_API_VERSION": "2024-10-21",
            "LLM_MAX_RETRIES": "1",
            "LLM_EMBEDDING_DIMENSIONS": "",
            # Start each check from the chat provider's own embedding and defaults.
            "EMBEDDING_API_KEY": "", "EMBEDDING_BASE_URL": "", "EMBEDDING_MODEL": "",
            "LLM_JSON_MODE": "", "LLM_GENERATE_KWARGS": "",
        })
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.clear_models()
        self.addCleanup(self.clear_models)
        self.plan = json.loads(
            (Path(__file__).parent / "fixtures" / "agentscope_plan.json").read_text(encoding="utf-8")
        )
        self.requests = []
        self.active_requests = 0
        self.max_active_requests = 0
        self.invalid_chat_responses = 0
        self.http_client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))
        self.addCleanup(lambda: llm_api.run_async(self.http_client.aclose()))
        client_kwargs = llm_api._client_kwargs

        def offline_kwargs(config):
            return {**client_kwargs(config), "http_client": self.http_client, "max_retries": 0}

        client_patch = patch.object(llm_api, "_client_kwargs", side_effect=offline_kwargs)
        client_patch.start()
        self.addCleanup(client_patch.stop)
        self.output = tempfile.TemporaryDirectory(prefix="mcrafter-agentscope-")
        self.addCleanup(self.output.cleanup)

    @staticmethod
    def clear_models():
        llm_api.get_chat_model.cache_clear()
        llm_api.get_embedding_model.cache_clear()

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append({"url": request.url, "body": body})
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            await asyncio.sleep(0.01)
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0,
                              "embedding": [0.25] * body.get("dimensions", 3)}],
                    "model": body["model"],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                })
            if not request.url.path.endswith("/chat/completions"):
                raise AssertionError(f"Unexpected request: {request.url}")
            if "fail this agent" in json.dumps(body["messages"]):
                return httpx.Response(400, json={"error": {"message": "offline failure"}})
            content = self.plan
            if self.invalid_chat_responses:
                self.invalid_chat_responses -= 1
                content = {}
            return httpx.Response(200, json={
                "id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(content),
                                         "refusal": None}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            })
        finally:
            self.active_requests -= 1

    def make_agent(self, agent_id=0, human=False):
        # Planning here uses supplied context; memory tokenization is not needed.
        with patch("memory_system.agent.TextModel"):
            agent = (HumanAgent if human else Agent)(agent_id, results_dir=self.output.name)
        agent.step = 0
        Image.new("RGB", (8, 8), "green").save(agent.agent_dir / "0.png")
        return agent

    def check_provider(self, provider):
        os.environ["LLM_PROVIDER"] = provider
        plan = llm_api.get_completion([Msg("environment", "Collect wood.", "user")])
        self.assertEqual(plan.action.final_next_action.value, "Navigator")
        chat_request = self.requests[-1]
        expected_format = "json_object" if provider == "qwen" else "json_schema"
        self.assertEqual(chat_request["body"]["response_format"]["type"], expected_format)
        response = llm_api.run_async(llm_api.get_embedding_model()(["offline embedding"]))
        self.assertEqual(len(response.embeddings[0]), 3)
        self.assertNotIn("dimensions", self.requests[-1]["body"])
        if provider == "azure":
            self.assertIn("/openai/deployments/", str(chat_request["url"]))
            self.assertEqual(chat_request["url"].params["api-version"], "2024-10-21")
            self.assertIn("/openai/deployments/", str(self.requests[-1]["url"]))

    def test_openai_chat_and_embedding(self):
        self.check_provider("openai")

    def test_azure_chat_and_embedding(self):
        self.check_provider("azure")

    def test_qwen_chat_and_embedding(self):
        self.check_provider("qwen")

    def use_compatible(self, separate_embedding=True):
        os.environ.update({"LLM_PROVIDER": "compatible", "LLM_API_KEY": "offline-test-key",
                           "LLM_BASE_URL": "https://compatible.invalid/v1", "LLM_CHAT_MODEL": "vendor-vision"})
        if separate_embedding:
            os.environ.update({"EMBEDDING_API_KEY": "offline-embedding-key",
                               "EMBEDDING_BASE_URL": "https://embedding.invalid/v1",
                               "EMBEDDING_MODEL": "shared-embedding"})

    def test_compatible_chat_uses_json_mode_and_separate_embedding(self):
        self.use_compatible()
        os.environ["LLM_GENERATE_KWARGS"] = '{"temperature": 0.3}'
        plan = llm_api.get_completion([Msg("environment", "Collect wood.", "user")])
        self.assertEqual(plan.action.final_next_action.value, "Navigator")
        chat = self.requests[-1]
        self.assertEqual(chat["url"].host, "compatible.invalid")
        self.assertEqual(chat["body"]["model"], "vendor-vision")
        self.assertEqual(chat["body"]["response_format"]["type"], "json_object")
        self.assertEqual(chat["body"]["temperature"], 0.3)
        llm_api.run_async(llm_api.get_embedding_model()(["offline embedding"]))
        embedding = self.requests[-1]
        self.assertEqual((embedding["url"].host, embedding["body"]["model"]),
                         ("embedding.invalid", "shared-embedding"))
        report = llm_api.describe_llm_configuration()
        self.assertEqual((report["embedding_endpoint"], report["json_mode"]), ("separate", "json_object"))
        self.assertNotIn("offline-test-key", json.dumps(report))

    def test_compatible_chat_without_embedding_endpoint_fails_clearly(self):
        self.use_compatible(separate_embedding=False)
        with self.assertRaisesRegex(llm_api.LLMConfigurationError, "EMBEDDING_API_KEY"):
            llm_api.get_embedding_model()
        os.environ["EMBEDDING_MODEL"] = "shared-embedding"
        with self.assertRaisesRegex(llm_api.LLMConfigurationError, "partly configured"):
            llm_api.get_embedding_override()

    def test_json_mode_retry_is_counted(self):
        self.use_compatible()
        os.environ["LLM_MAX_RETRIES"] = "2"
        self.invalid_chat_responses = 1
        before = llm_api.get_model_usage().get("chat_json_retries", 0)
        with redirect_stdout(io.StringIO()):
            llm_api.get_completion([Msg("environment", "Collect wood.", "user")])
        self.assertEqual(llm_api.get_model_usage()["chat_json_retries"], before + 1)

    def test_embedding_dimensions_override(self):
        os.environ["LLM_EMBEDDING_DIMENSIONS"] = "2"
        result = llm_api.run_async(llm_api.get_embedding_model()(["dimensions"]))
        self.assertEqual(len(result.embeddings[0]), 2)
        self.assertEqual(self.requests[-1]["body"]["dimensions"], 2)

    def test_agent_multimodal_reply(self):
        agent = self.make_agent()
        self.assertIsInstance(agent, AgentBase)
        response = llm_api.run_async(agent(Msg("environment", "Collect wood.", "user")))
        self.assertIsInstance(response, Msg)
        self.assertEqual(llm_api.ResponseEvent.model_validate(response.metadata).summary, self.plan["summary"])
        json.dumps(response.metadata)
        images = [block for message in self.requests[-1]["body"]["messages"]
                  if isinstance(message["content"], list) for block in message["content"]
                  if block["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertTrue(all(block["image_url"]["url"].startswith("data:image/") for block in images))

    def test_agent_consumes_post_reply_hook_metadata(self):
        agent = self.make_agent()

        def update_summary(agent, kwargs, response):
            metadata = {**response.metadata, "summary": "Plan updated by AgentScope hook."}
            return Msg(agent.name, metadata["summary"], "assistant", metadata=metadata)

        agent.register_instance_hook("post_reply", "update_summary", update_summary)
        self.assertEqual(agent.think("Collect wood.").summary, "Plan updated by AgentScope hook.")
        responses = AgentThinkingProcessor.process_agent_thinking_parallel(
            [agent], {0: "Collect wood."}, [{"sleeping": False}],
        )
        self.assertEqual(responses[0]["thoughts"].summary, "Plan updated by AgentScope hook.")

    def test_parallel_planning_skips_and_isolates_failures(self):
        agents = [self.make_agent(i, human=(i == 4)) for i in range(6)]
        agents[3].action_status = ActionStatus.IN_PROGRESS
        info = [{"sleeping": i == 2} for i in range(6)]
        with redirect_stdout(io.StringIO()):
            responses = AgentThinkingProcessor.process_agent_thinking_parallel(
                agents, {0: "Collect wood.", 1: "Collect wood.", 5: "fail this agent"}, info,
            )
        by_id = {response["id"]: response for response in responses}
        self.assertEqual(set(by_id), {0, 1, 2, 3, 5})
        self.assertIsNotNone(by_id[0]["thoughts"])
        self.assertIsNotNone(by_id[1]["thoughts"])
        self.assertIsNone(by_id[2]["thoughts"])
        self.assertIsNone(by_id[3]["thoughts"])
        self.assertIn("offline failure", by_id[5]["error"])
        self.assertGreaterEqual(self.max_active_requests, 2)

    def test_qwen_invalid_json_is_corrected(self):
        os.environ["LLM_MAX_RETRIES"] = "2"
        self.invalid_chat_responses = 1
        with redirect_stdout(io.StringIO()):
            result = llm_api.get_completion([Msg("environment", "Collect wood.", "user")])
        self.assertEqual(result.summary, self.plan["summary"])
        self.assertEqual(len(self.requests), 2)
        self.assertIn("Validation error:", self.requests[-1]["body"]["messages"][-1]["content"])

    def test_zero_retries_fails_before_request(self):
        os.environ["LLM_MAX_RETRIES"] = "0"
        with self.assertRaisesRegex(llm_api.LLMConfigurationError, "at least 1"):
            llm_api.get_completion([])
        self.assertEqual(self.requests, [])

    def test_nested_sync_call_fails_without_deadlock(self):
        async def nested():
            llm_api.run_async(asyncio.sleep(0))

        with self.assertRaisesRegex(RuntimeError, "await the coroutine"):
            llm_api.run_async(nested())
        self.assertEqual(llm_api.run_async(asyncio.sleep(0, result="loop still healthy")), "loop still healthy")


if __name__ == "__main__":
    print(json.dumps({name: version(name) for name in ("agentscope", "openai", "mcp")}), flush=True)
    unittest.main(verbosity=2)
