import json
import unittest

import config
from ticket_investigation_json_endpoint import (
    ConditionalTicketExecutionJsonEndpoint,
    build_ticket_execution_json_endpoint,
)
from ticket_investigation_transport import TicketExecutionTransportRequest


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


class _ExecutorStub:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("canary tests should not execute the local delegate")


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
        wants_bug_review_status=False,
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

    def test_endpoint_factory_wraps_canary_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_canary = config.TICKET_EXECUTION_CANARY_ENDPOINT
        original_canary_channels = config.TICKET_EXECUTION_CANARY_CHANNEL_IDS
        original_canary_intents = config.TICKET_EXECUTION_CANARY_INTENTS
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "local"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CANARY_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_CANARY_CHANNEL_IDS = {"42"}
            config.TICKET_EXECUTION_CANARY_INTENTS = set()
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            endpoint = build_ticket_execution_json_endpoint(_ExecutorStub())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CANARY_ENDPOINT = original_canary
            config.TICKET_EXECUTION_CANARY_CHANNEL_IDS = original_canary_channels
            config.TICKET_EXECUTION_CANARY_INTENTS = original_canary_intents
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIsInstance(endpoint, ConditionalTicketExecutionJsonEndpoint)
