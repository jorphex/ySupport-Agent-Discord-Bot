import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import config
from ticket_investigation_json_endpoint import (
    ShadowTicketExecutionJsonEndpoint,
    build_ticket_execution_json_endpoint,
)
from ticket_investigation_transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)


class _StaticEndpoint:
    def __init__(self, response_json: str) -> None:
        self.response_json = response_json
        self.calls = 0

    async def execute_json_turn(self, request_json: str, hooks=None) -> str:
        self.calls += 1
        return self.response_json


class _FailingEndpoint:
    async def execute_json_turn(self, request_json: str, hooks=None) -> str:
        raise RuntimeError("shadow failed")


class _ExecutorStub:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("shadow tests should not execute the local delegate")


def _transport_result_json(reply: str) -> str:
    return json.dumps(
        {
            "flow_outcome": {
                "raw_final_reply": reply,
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


class ShadowTicketExecutionEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_endpoint_returns_primary_and_records_shadow(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 1,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 1,
                "requested_intent": "docs",
                "mode": "idle",
                "evidence": {
                    "wallet": None,
                    "chain": None,
                    "tx_hashes": [],
                    "withdrawal_target_chain": None,
                    "withdrawal_target_vault": None,
                },
            },
            workflow_name="tests.shadow",
            wants_bug_review_status=False,
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            primary = _StaticEndpoint(_transport_result_json("primary"))
            shadow = _StaticEndpoint(_transport_result_json("shadow"))
            endpoint = ShadowTicketExecutionJsonEndpoint(
                primary,
                shadow,
                primary_mode="local",
                shadow_mode="codex_support_exec",
                artifact_dir=artifact_dir,
            )
            response_json = await endpoint.execute_json_turn(request.to_json())
            await asyncio.sleep(0.05)

            self.assertEqual(
                TicketExecutionTransportResult.from_json(response_json).flow_outcome[
                    "raw_final_reply"
                ],
                "primary",
            )
            records = list(Path(artifact_dir).glob("*.json"))
            self.assertEqual(len(records), 1)
            payload = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual(
                payload["primary_result"]["flow_outcome"]["raw_final_reply"],
                "primary",
            )
            self.assertEqual(
                payload["shadow_result"]["flow_outcome"]["raw_final_reply"],
                "shadow",
            )

    async def test_shadow_endpoint_records_shadow_error_without_affecting_primary(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 1,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 1,
                "requested_intent": "docs",
                "mode": "idle",
                "evidence": {
                    "wallet": None,
                    "chain": None,
                    "tx_hashes": [],
                    "withdrawal_target_chain": None,
                    "withdrawal_target_vault": None,
                },
            },
            workflow_name="tests.shadow",
            wants_bug_review_status=False,
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            endpoint = ShadowTicketExecutionJsonEndpoint(
                _StaticEndpoint(_transport_result_json("primary")),
                _FailingEndpoint(),
                primary_mode="local",
                shadow_mode="codex_support_exec",
                artifact_dir=artifact_dir,
            )
            response_json = await endpoint.execute_json_turn(request.to_json())
            await asyncio.sleep(0.05)

            self.assertEqual(
                TicketExecutionTransportResult.from_json(response_json).flow_outcome[
                    "raw_final_reply"
                ],
                "primary",
            )
            records = list(Path(artifact_dir).glob("*.json"))
            self.assertEqual(len(records), 1)
            payload = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertIn("shadow_error", payload)

    def test_endpoint_factory_wraps_shadow_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_shadow = config.TICKET_EXECUTION_SHADOW_ENDPOINT
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_shadow_artifact_dir = config.TICKET_EXECUTION_SHADOW_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "local"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SHADOW_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            config.TICKET_EXECUTION_SHADOW_ARTIFACT_DIR = "/tmp/ticket-shadow"
            endpoint = build_ticket_execution_json_endpoint(_ExecutorStub())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_SHADOW_ENDPOINT = original_shadow
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
            config.TICKET_EXECUTION_SHADOW_ARTIFACT_DIR = original_shadow_artifact_dir

        self.assertIsInstance(endpoint, ShadowTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.shadow_mode, "subprocess")
        self.assertEqual(endpoint.artifact_dir, "/tmp/ticket-shadow")
