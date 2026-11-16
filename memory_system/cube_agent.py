"""Small AgentScope planner for the fully observed upstream CUBE task."""

from dataclasses import asdict
import json
from typing import Literal

from agentscope.agent import AgentBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from pydantic import BaseModel

from .coordination import Plan
from .cube_adapter import OPERATIONS
from .llm_api import get_chat_model, measured_model_call, _response_text


class CubeDecision(BaseModel):
    action: Literal[0, 1, 2, 3, 4]
    reason: str


class CubeAgent(AgentBase):
    def __init__(self, agent_id):
        super().__init__()
        self.id, self.name = agent_id, f'agent_{agent_id}'
        self.op = self.action = 'noop'
        self.rss_to_collect = self.rss_to_share = 'not_applicable'
        self.target_agent_id = -1
        self.review_paused = self.coordination_blocked = False
        self.pending_plan_response = self.thoughts = None
        self.constraint_checks = []
        self.step = 0
        self.history = []

    async def reply(self, msg):
        messages = await OpenAIChatFormatter().format([
            Msg('system', 'You control one agent in cooperative block pushing. Coordinates are (row,column). '
                'Actions: 0 stay, 1 up, 2 down, 3 left, 4 right. A square block of weight W needs W '
                'agents simultaneously pushing the same face. Push blocks to the rightmost goal column. '
                'Choose only your own next action, coordinating from the shared plans and recent results. '
                'Return JSON only: {"action":0,"reason":"short reason"}.', 'system'),
            msg,
        ])
        response = await measured_model_call('chat', get_chat_model(), messages,
                                             response_format={'type': 'json_object'}, max_tokens=512)
        decision = CubeDecision.model_validate_json(_response_text(response))
        plan = Plan(self.id, OPERATIONS[decision.action], 'not_applicable', 'not_applicable', -1)
        return Msg(self.name, decision.reason, 'assistant', metadata=asdict(plan))

    def accept(self, response):
        data = response.get('plan_override', response.get('plan'))
        if data is not None:
            self.op = data['operation']
            if not response.get('deferred', False):
                self.history.append({'step': self.step, 'plan': data,
                                     'source': response.get('intervention', {}).get('source', 'planner')})

    def context(self, observations, plans, feedback=None):
        return Msg('environment', json.dumps({
            'your_id': self.id, 'step': self.step, 'observation': observations[self.id]['cube'],
            'team_plans': [asdict(plan) for plan in plans], 'recent_accepted_plans': self.history[-3:],
            'repair_feedback': feedback,
        }), 'user')
