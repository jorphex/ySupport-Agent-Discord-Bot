import tests as _test_environment  # noqa: F401

import json
import unittest

from ticket_investigation.json_endpoint import (
    ConditionalTicketExecutionJsonEndpoint,
)
from ticket_investigation.transport import TicketExecutionTransportRequest


class _StaticEndpoint:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.calls = 0

    async def execute_json_turn(self, request_json: str, hooks=None) -> str:
        self.calls += 1
        return json.dumps(
            {
                "flow_outcome": {
                    "raw_final_reply": self.marker,
                    "conversation_history": [],
                    "completed_agent_key": None,
                    "requires_human_handoff": False,
                },
                "updated_job": {
                    "channel_id": 1,
                    "requested_intent": "docs",
                    "mode": "waiting_for_user",
                    "current_specialty": None,
                    "last_specialty": None,
                    "evidence": {
                        "wallet": None,
                        "chain": None,
                        "tx_hashes": [],
                        "withdrawal_target_chain": None,
                        "withdrawal_target_vault": None,
                    },
                },
            }
        )


def _request(*, channel_id: int, requested_intent: str) -> TicketExecutionTransportRequest:
    return TicketExecutionTransportRequest(
        aggregated_text="help",
        input_list=[],
        current_history=[],
        run_context={
            "channel_id": channel_id,
            "project_context": "yearn",
            "repo_last_search_artifact_refs": [],
        },
        investigation_job={
            "channel_id": channel_id,
            "requested_intent": requested_intent,
            "mode": "idle",
            "evidence": {
                "wallet": None,
                "chain": None,
                "tx_hashes": [],
                "withdrawal_target_chain": None,
                "withdrawal_target_vault": None,
            },
        },
        workflow_name="tests.canary",
    )


class TicketExecutionCanaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_conditional_endpoint_routes_matching_channel_to_canary(self) -> None:
        default = _StaticEndpoint("default")
        canary = _StaticEndpoint("canary")
        endpoint = ConditionalTicketExecutionJsonEndpoint(
            default,
            canary,
            channel_ids={"42"},
            intents=set(),
        )

        response_json = await endpoint.execute_json_turn(
            _request(channel_id=42, requested_intent="docs").to_json()
        )
        self.assertEqual(json.loads(response_json)["flow_outcome"]["raw_final_reply"], "canary")
        self.assertEqual(default.calls, 0)
        self.assertEqual(canary.calls, 1)

    async def test_conditional_endpoint_routes_matching_intent_to_canary(self) -> None:
        default = _StaticEndpoint("default")
        canary = _StaticEndpoint("canary")
        endpoint = ConditionalTicketExecutionJsonEndpoint(
            default,
            canary,
            channel_ids=set(),
            intents={"investigate_issue"},
        )

        response_json = await endpoint.execute_json_turn(
            _request(channel_id=1, requested_intent="investigate_issue").to_json()
        )
        self.assertEqual(json.loads(response_json)["flow_outcome"]["raw_final_reply"], "canary")
        self.assertEqual(default.calls, 0)
        self.assertEqual(canary.calls, 1)

    async def test_conditional_endpoint_uses_default_when_no_selector_matches(self) -> None:
        default = _StaticEndpoint("default")
        canary = _StaticEndpoint("canary")
        endpoint = ConditionalTicketExecutionJsonEndpoint(
            default,
            canary,
            channel_ids={"42"},
            intents={"investigate_issue"},
        )

        response_json = await endpoint.execute_json_turn(
            _request(channel_id=1, requested_intent="docs").to_json()
        )
        self.assertEqual(json.loads(response_json)["flow_outcome"]["raw_final_reply"], "default")
        self.assertEqual(default.calls, 1)
        self.assertEqual(canary.calls, 0)
