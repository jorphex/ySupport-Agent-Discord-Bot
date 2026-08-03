import tests as _test_environment  # noqa: F401

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from codex_support_contract import (
    SupportTurnResult,
)
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ticket_investigation.codex_support_subprocess import (
    CodexSupportExecutionOutput,
    parse_codex_support_execution_output,
)
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)


class CodexSupportEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_support_json_endpoint_uses_codex_support_smoke_reply(
        self,
    ) -> None:
        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=[sys.executable, "-c", "print('should-not-run')"],
            allowed_command_prefixes=[
                [sys.executable, "-c", "print('should-not-run')"]
            ],
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="smoke",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 110,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 110,
                "requested_intent": "smoke_probe",
                "mode": "idle",
                "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_smoke",
            smoke_mode="ping",
        )

        response_json = await endpoint.execute_json_turn(request.to_json())
        transport_result = TicketExecutionTransportResult.from_json(response_json)
        flow_outcome = transport_result.flow_outcome
        updated_job = transport_result.updated_job

        self.assertEqual(
            flow_outcome["raw_final_reply"],
            "ticket_execution_smoke_ok:codex_support_exec",
        )
        self.assertEqual(updated_job["channel_id"], 110)

    async def test_codex_support_endpoint_serializes_same_conversation(self) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()
        call_count = 0

        async def fake_run_streaming_subprocess(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_started.set()
                await release_first.wait()
                return CodexSupportExecutionOutput(final_response_text=response)
            second_started.set()
            return CodexSupportExecutionOutput(final_response_text=response)

        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=["codex", "exec"],
            allowed_command_prefixes=[["codex", "exec"]],
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="investigate support",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 109,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 109,
                "requested_intent": "investigate_issue",
                "mode": "collecting",
                "evidence": {"wallet": None, "chain": "base", "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_exec",
        )

        with mock.patch(
            "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
            side_effect=fake_run_streaming_subprocess,
        ):
            task_one = asyncio.create_task(
                endpoint.execute_json_turn(request.to_json())
            )
            await first_started.wait()
            task_two = asyncio.create_task(
                endpoint.execute_json_turn(request.to_json())
            )
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())
            release_first.set()
            await asyncio.gather(task_one, task_two)

        self.assertTrue(second_started.is_set())

    async def test_codex_support_endpoint_does_not_retry_auth_error(self) -> None:
        call_count = 0

        async def fake_run_streaming_subprocess(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("refresh_token_reused token_expired")

        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=["codex", "exec"],
            allowed_command_prefixes=[["codex", "exec"]],
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="investigate support",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 111,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 111,
                "requested_intent": "docs_qa",
                "mode": "waiting_for_user",
                "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_exec",
        )

        with mock.patch(
            "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
            side_effect=fake_run_streaming_subprocess,
        ):
            with self.assertRaisesRegex(RuntimeError, "refresh_token_reused"):
                await endpoint.execute_json_turn(request.to_json())

        self.assertEqual(call_count, 1)

    async def test_codex_support_endpoint_uses_service_auth_link_in_place(
        self,
    ) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()

        with tempfile.TemporaryDirectory() as temp_dir:
            auth_source = Path(temp_dir) / "auth-source.json"
            auth_source.write_text('{"auth_mode":"source-old"}', encoding="utf-8")
            canonical_source = Path(temp_dir) / "canonical-auth.json"
            canonical_source.write_text(
                '{"auth_mode":"canonical-old"}', encoding="utf-8"
            )
            os.utime(auth_source, (1, 1))
            os.utime(canonical_source, (1, 1))
            bot_home = Path(temp_dir) / "bot-home"
            bot_home.mkdir(parents=True, exist_ok=True)

            async def fake_run_streaming_subprocess(**kwargs):
                (bot_home / "auth.json").write_text(
                    '{"auth_mode":"bot-refreshed"}',
                    encoding="utf-8",
                )
                return CodexSupportExecutionOutput(final_response_text=response)

            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                codex_home=bot_home,
                codex_auth_link_source=canonical_source,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="investigate support",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 112,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 112,
                    "requested_intent": "docs_qa",
                    "mode": "waiting_for_user",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_support_exec",
            )

            with mock.patch(
                "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

            transport_result = TicketExecutionTransportResult.from_json(response_json)
            self.assertEqual(
                transport_result.flow_outcome["raw_final_reply"],
                "support-ok",
            )
            self.assertEqual(
                auth_source.read_text(encoding="utf-8"),
                '{"auth_mode":"source-old"}',
            )
            self.assertEqual(
                canonical_source.read_text(encoding="utf-8"),
                '{"auth_mode":"bot-refreshed"}',
            )

    async def test_codex_support_endpoint_replaces_stale_home_auth_with_service_link(
        self,
    ) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()

        with tempfile.TemporaryDirectory() as temp_dir:
            auth_source = Path(temp_dir) / "auth-source.json"
            auth_source.write_text('{"auth_mode":"source-old"}', encoding="utf-8")
            canonical_source = Path(temp_dir) / "canonical-auth.json"
            canonical_source.write_text(
                '{"auth_mode":"canonical-fresh"}',
                encoding="utf-8",
            )
            os.utime(auth_source, (4, 4))
            os.utime(canonical_source, (1, 1))
            bot_home = Path(temp_dir) / "bot-home"
            bot_home.mkdir(parents=True, exist_ok=True)
            (bot_home / "auth.json").write_text(
                '{"auth_mode":"bot-stale"}',
                encoding="utf-8",
            )
            os.utime(bot_home / "auth.json", (5, 5))

            seen_home_auth = {}

            async def fake_run_streaming_subprocess(**kwargs):
                seen_home_auth["text"] = (bot_home / "auth.json").read_text(
                    encoding="utf-8"
                )
                return CodexSupportExecutionOutput(final_response_text=response)

            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                codex_home=bot_home,
                codex_auth_link_source=canonical_source,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="investigate support",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 113,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 113,
                    "requested_intent": "docs_qa",
                    "mode": "waiting_for_user",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_support_exec",
            )

            with mock.patch(
                "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

            transport_result = TicketExecutionTransportResult.from_json(response_json)
            self.assertEqual(
                transport_result.flow_outcome["raw_final_reply"],
                "support-ok",
            )
            self.assertEqual(
                seen_home_auth["text"],
                '{"auth_mode":"canonical-fresh"}',
            )
            self.assertEqual(
                auth_source.read_text(encoding="utf-8"),
                '{"auth_mode":"source-old"}',
            )
            self.assertEqual(
                canonical_source.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-fresh"}',
            )

    async def test_codex_support_endpoint_links_live_auth_before_first_attempt(
        self,
    ) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()

        with tempfile.TemporaryDirectory() as temp_dir:
            live_auth = Path(temp_dir) / "live-home" / "auth.json"
            live_auth.parent.mkdir(parents=True, exist_ok=True)
            live_auth.write_text('{"auth_mode":"live"}', encoding="utf-8")
            bot_home = Path(temp_dir) / "bot-home"
            bot_home.mkdir(parents=True, exist_ok=True)
            seen_home_auth = {}

            async def fake_run_streaming_subprocess(**kwargs):
                bot_auth_path = bot_home / "auth.json"
                seen_home_auth["is_symlink"] = bot_auth_path.is_symlink()
                seen_home_auth["resolved"] = bot_auth_path.resolve(strict=False)
                seen_home_auth["text"] = bot_auth_path.read_text(encoding="utf-8")
                return CodexSupportExecutionOutput(final_response_text=response)

            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                codex_home=bot_home,
                codex_auth_link_source=live_auth,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="investigate support",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 114,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 114,
                    "requested_intent": "docs_qa",
                    "mode": "waiting_for_user",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_support_exec",
            )

            with mock.patch(
                "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

            transport_result = TicketExecutionTransportResult.from_json(response_json)
            self.assertEqual(
                transport_result.flow_outcome["raw_final_reply"],
                "support-ok",
            )
            self.assertTrue(seen_home_auth["is_symlink"])
            self.assertEqual(
                seen_home_auth["resolved"],
                live_auth.resolve(strict=False),
            )
            self.assertEqual(
                seen_home_auth["text"],
                '{"auth_mode":"live"}',
            )

    async def test_codex_support_endpoint_serializes_codex_runs_across_conversations(
        self,
    ) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        in_flight = 0
        max_in_flight = 0

        async def fake_run_streaming_subprocess(**kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if not first_entered.is_set():
                first_entered.set()
                await release_first.wait()
            else:
                second_entered.set()
            in_flight -= 1
            return CodexSupportExecutionOutput(final_response_text=response)

        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=["codex", "exec"],
            allowed_command_prefixes=[["codex", "exec"]],
        )
        request_one = TicketExecutionTransportRequest(
            aggregated_text="first",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 201,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 201,
                "requested_intent": "docs_qa",
                "mode": "waiting_for_user",
                "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_exec",
        )
        request_two = TicketExecutionTransportRequest(
            aggregated_text="second",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 202,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 202,
                "requested_intent": "docs_qa",
                "mode": "waiting_for_user",
                "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_exec",
        )

        with mock.patch(
            "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
            side_effect=fake_run_streaming_subprocess,
        ):
            first_task = asyncio.create_task(
                endpoint.execute_json_turn(request_one.to_json())
            )
            await first_entered.wait()
            second_task = asyncio.create_task(
                endpoint.execute_json_turn(request_two.to_json())
            )
            await asyncio.sleep(0.05)
            self.assertFalse(
                second_entered.is_set(),
                "second Codex run should not enter while the first run is active",
            )
            release_first.set()
            first_response, second_response = await asyncio.gather(
                first_task,
                second_task,
            )

        self.assertEqual(max_in_flight, 1)
        self.assertTrue(second_entered.is_set())
        self.assertEqual(
            TicketExecutionTransportResult.from_json(first_response).flow_outcome[
                "raw_final_reply"
            ],
            "support-ok",
        )
        self.assertEqual(
            TicketExecutionTransportResult.from_json(second_response).flow_outcome[
                "raw_final_reply"
            ],
            "support-ok",
        )

    def test_parse_codex_support_execution_output_uses_only_final_answer(self) -> None:
        stdout_text = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "mcp_tool_call",
                            "tool": "search_documentation",
                            "result": {
                                "structured_content": {
                                    "result": "Docs answer\n\n- bullet"
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": '{"answer":"final","evidence_summary":"x","handoff_reason":null,"requires_human_handoff":false,"used_tools":["ysupport_mcp.search_documentation"]}',
                        },
                    }
                ),
            ]
        )
        execution_output = parse_codex_support_execution_output(stdout_text)
        self.assertIn('"answer":"final"', execution_output.final_response_text)
