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
    SignedTransactionSafetyViolation,
    SupportTurnRequest,
    SupportTurnResult,
)
from codex_support_sessions import CodexSupportSessionManager
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ticket_investigation.codex_support_subprocess import (
    CodexSupportExecutionOutput,
)
from ticket_investigation.executor import TicketExecutionHooks
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)


class _FakeExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Factory test should not execute the delegate.")


EXAMPLE_YSUPPORT_MCP_URL = "http://ysupport-mcp.example.test/mcp"
_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION = "0xf8cb" + ("ab" * 203)
_SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION = "0x02f8" + ("ab" * 120)
_SYNTHETIC_HIGH_TYPE_RAW_SIGNED_TRANSACTION = "0x7afa" + ("ab" * 120)
_SHORT_LEGACY_RAW_SIGNED_TRANSACTION = (
    "0xf85f8001825208940000000000000000000000000000000000000000808025"
    "a05a420b0a542873e0f1a0a6bcf149ab3d26204c0fe61ebcb30dad82a8e7e9a370"
    "a057d3f4e87c966ab79a22aa4fbfcffd48f586a65190125aac497aacafab2a7a6f"
)


def _transaction_safety_support_request() -> SupportTurnRequest:
    return SupportTurnRequest(
        current_user_message="toujours pas",
        recent_transcript=[],
        channel_type="ticket",
        channel_id=1,
        project_context="yearn",
        workflow_name="tests.verify",
        initial_button_intent="investigate_issue",
        requested_intent="investigate_issue",
        evidence={},
        support_state={},
        constraints={"allowed_tools": ["shell"]},
    )


def _transaction_safety_transport_request(
    *,
    history: list[dict[str, str]] | None = None,
) -> TicketExecutionTransportRequest:
    return TicketExecutionTransportRequest(
        aggregated_text="toujours pas",
        input_list=[],
        current_history=history or [],
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
            "evidence": {"wallet": None, "chain": "katana", "tx_hashes": []},
        },
        workflow_name="tests.endpoint.codex_support_exec",
        wants_bug_review_status=False,
    )


class CodexSupportEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_support_json_endpoint_round_trips_response_and_writes_bundle(
        self,
    ) -> None:
        fake_codex = (
            "import json,os,pathlib,sys; "
            "prompt=sys.stdin.read(); "
            "cwd=pathlib.Path(os.getcwd()); "
            "request=json.loads((cwd/'support_request.json').read_text()); "
            "response={"
            "'answer':'support-ok:{}:{}:{}:{}:{}:{}:{}:{}'.format("
            "request['channel_type'],"
            "request['constraints']['no_file_writes'],"
            "'-m' in sys.argv,"
            "'gpt-5.6-sol' in sys.argv,"
            "'model_reasoning_effort=\"medium\"' in ' '.join(sys.argv),"
            "'--json' in sys.argv,"
            "'--output-schema' in sys.argv,"
            "(cwd/'support_response_schema.json').exists(),"
            "'Read the support turn request from ' in prompt and "
            "str(cwd/'support_request.json') in prompt"
            "),"
            "'requires_human_handoff':False,"
            "'handoff_reason':None,"
            "'evidence_summary':'checked',"
            "'used_tools':['shell','ysupport_mcp']"
            "}; "
            "sys.stdout.write(json.dumps({'type':'thread.started','thread_id':'019dade1-5acf-70e2-9c61-f5ba37862a78'}) + '\\n'); "
            "sys.stdout.write(json.dumps({'type':'item.started','item':{'id':'item_1','type':'mcp_tool_call','tool_name':'support_dashboard_harvests'}}) + '\\n'); "
            "sys.stdout.write(json.dumps({'type':'item.completed','item':{'id':'item_2','type':'agent_message','text':json.dumps(response)}}))"
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            with tempfile.TemporaryDirectory() as codex_home_dir:
                auth_source = Path(codex_home_dir) / "source-auth.json"
                auth_source.write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")
                bot_home = Path(codex_home_dir) / "bot-home"
                session_dir = Path(codex_home_dir) / "sessions"
                progress_updates: list[str] = []

                async def record_progress(text: str) -> None:
                    progress_updates.append(text)

                with mock.patch(
                    "codex_support_home._choose_ysupport_stdio_launcher",
                    return_value={
                        "command": "python3",
                        "args": ["mcp_server.py"],
                        "cwd": str(Path(__file__).resolve().parents[1]),
                        "env_vars": ["OPENAI_API_KEY"],
                        "env": {
                            "MCP_TRANSPORT": "stdio",
                            "MCP_SERVER_API_KEY": "secret-key",
                        },
                    },
                ):
                    endpoint = CodexSupportTicketExecutionJsonEndpoint(
                        codex_command=[sys.executable, "-c", fake_codex],
                        model="gpt-5.6-sol",
                        reasoning_effort="medium",
                        repo_root=Path(__file__).resolve().parents[1],
                        codex_home=bot_home,
                        codex_auth_source=auth_source,
                        session_dir=session_dir,
                        ysupport_mcp_url="http://127.0.0.1:8000/mcp",
                        ysupport_mcp_container="ysupport-mcp",
                        mcp_server_api_key="secret-key",
                        allowed_command_prefixes=[[sys.executable, "-c", fake_codex]],
                        artifact_dir=artifact_dir,
                    )
                    request = TicketExecutionTransportRequest(
                        aggregated_text="investigate support",
                        input_list=[],
                        current_history=[{"role": "user", "content": "earlier"}],
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
                            "evidence": {
                                "wallet": None,
                                "chain": "base",
                                "tx_hashes": [],
                            },
                        },
                        workflow_name="tests.endpoint.codex_support_exec",
                        wants_bug_review_status=False,
                    )

                    response_json = await endpoint.execute_json_turn(
                        request.to_json(),
                        hooks=TicketExecutionHooks(
                            send_progress_update=record_progress,
                        ),
                    )

                    artifact_entries = os.listdir(artifact_dir)
                    self.assertEqual(len(artifact_entries), 1)
                    run_dir = os.path.join(artifact_dir, artifact_entries[0])
                    self.assertTrue(
                        os.path.exists(os.path.join(run_dir, "support_request.json"))
                    )
                    self.assertTrue(
                        os.path.exists(
                            os.path.join(run_dir, "support_response_schema.json")
                        )
                    )
                    self.assertTrue(
                        os.path.exists(
                            os.path.join(run_dir, "codex_support_prompt.txt")
                        )
                    )
                    self.assertTrue(os.path.exists(os.path.join(run_dir, "stdout.txt")))
                    self.assertTrue(os.path.exists(os.path.join(run_dir, "stderr.txt")))
                    support_request_payload = json.loads(
                        Path(run_dir, "support_request.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(support_request_payload["channel_id"], 109)
                    self.assertEqual(
                        support_request_payload["initial_button_intent"],
                        "investigate_issue",
                    )
                    self.assertEqual(
                        support_request_payload["support_state"]["known_targets"][
                            "chain"
                        ],
                        "base",
                    )
                    self.assertTrue((bot_home / "config.toml").exists())
                    self.assertTrue((bot_home / "auth.json").exists())
                    self.assertTrue((bot_home / "ysupport_instructions.md").exists())
                    self.assertIn(
                        "[mcp_servers.ysupport]",
                        (bot_home / "config.toml").read_text(encoding="utf-8"),
                    )
                    self.assertIn(
                        'command = "python3"',
                        (bot_home / "config.toml").read_text(encoding="utf-8"),
                    )
                    self.assertIn(
                        "model_instructions_file",
                        (bot_home / "config.toml").read_text(encoding="utf-8"),
                    )
                    session_record = CodexSupportSessionManager(session_dir).load(
                        "ticket:109"
                    )
                    self.assertIsNotNone(session_record)
                    assert session_record is not None
                    self.assertEqual(
                        session_record.session_id,
                        "019dade1-5acf-70e2-9c61-f5ba37862a78",
                    )

        transport_result = TicketExecutionTransportResult.from_json(response_json)
        flow_outcome = transport_result.flow_outcome
        updated_job = transport_result.updated_job
        self.assertEqual(
            flow_outcome["raw_final_reply"],
            "support-ok:ticket:True:True:True:True:True:True:True",
        )
        self.assertEqual(updated_job["mode"], "waiting_for_user")
        self.assertIsNone(updated_job["current_specialty"])
        self.assertIn("Checking recent harvests", progress_updates)

    async def test_codex_support_endpoint_rewrites_unsafe_signed_transaction_output(
        self,
    ) -> None:
        unsafe_response = SupportTurnResult(
            answer=(
                "Use the Katana public broadcaster and paste this raw signed "
                f"transaction: `{_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION}`"
            ),
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Found the signed replacement transaction in the pending pool.",
            used_tools=["shell"],
        ).to_json()
        tx_hash = "0x" + ("12" * 32)
        safe_response = SupportTurnResult(
            answer=(
                f"Transaction {tx_hash} is still pending. In Rabby, clear the pending "
                "queue and retry through the official Yearn interface."
            ),
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the transaction status by hash.",
            used_tools=["shell"],
        ).to_json()
        session_id = "019dade1-5acf-70e2-9c61-f5ba37862a78"
        calls: list[dict[str, object]] = []
        unsafe_stdout_removed = False

        async def fake_run_streaming_subprocess(**kwargs):
            nonlocal unsafe_stdout_removed
            calls.append(kwargs)
            if len(calls) == 1:
                run_dir = kwargs["artifact_run_dir"]
                Path(run_dir, "stdout.txt").write_text(
                    json.dumps(
                        {
                            "type": "thread.started",
                            "thread_id": session_id,
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                first_run_dir = calls[0]["artifact_run_dir"]
                unsafe_stdout_removed = not Path(
                    first_run_dir,
                    "stdout.txt",
                ).exists()
            response = unsafe_response if len(calls) == 1 else safe_response
            return CodexSupportExecutionOutput(final_response_text=response)

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "sessions"
            artifact_dir = Path(temp_dir) / "artifacts"
            session_manager = CodexSupportSessionManager(session_dir)
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                session_dir=session_dir,
                allowed_command_prefixes=[["codex", "exec"]],
                artifact_dir=str(artifact_dir),
            )
            request = _transaction_safety_transport_request(
                history=[
                    {
                        "role": "assistant",
                        "content": "Please clear Rabby's pending queue and retry.",
                    }
                ]
            )

            with (
                mock.patch.object(
                    endpoint,
                    "_delete_codex_session",
                    new=mock.AsyncMock(return_value=True),
                ) as mock_delete,
                mock.patch(
                    "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                    side_effect=fake_run_streaming_subprocess,
                ),
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

            record = session_manager.load("ticket:109")

        transport_result = TicketExecutionTransportResult.from_json(response_json)
        self.assertEqual(
            transport_result.flow_outcome["raw_final_reply"],
            SupportTurnResult.from_json(safe_response).answer,
        )
        self.assertEqual(len(calls), 2)
        mock_delete.assert_awaited_once_with(session_id)
        self.assertIn(session_id, calls[1]["command"])
        self.assertIn(
            "Rewrite the response using only safe, read-only transaction troubleshooting.",
            calls[1]["stdin_text"],
        )
        self.assertNotIn(_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION, response_json)
        self.assertTrue(unsafe_stdout_removed)
        self.assertIsNone(record)
        self.assertFalse(artifact_dir.exists())

    async def test_contaminated_session_deletion_survives_repeated_cancellation(
        self,
    ) -> None:
        unsafe_response = SupportTurnResult(
            answer=f"Broadcast `{_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION}`.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Found a raw signed transaction.",
            used_tools=["shell"],
        ).to_json()
        session_id = "019dade1-5acf-70e2-9c61-f5ba37862a78"
        rewrite_started = asyncio.Event()
        deletion_started = asyncio.Event()
        allow_deletion = asyncio.Event()
        deletion_completed = asyncio.Event()
        run_count = 0

        async def fake_run_streaming_subprocess(**kwargs):
            nonlocal run_count
            del kwargs
            run_count += 1
            if run_count == 1:
                return CodexSupportExecutionOutput(final_response_text=unsafe_response)
            rewrite_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def fake_delete(_session_id: str) -> bool:
            self.assertEqual(_session_id, session_id)
            deletion_started.set()
            await allow_deletion.wait()
            deletion_completed.set()
            return True

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "sessions"
            session_manager = CodexSupportSessionManager(session_dir)
            session_manager.record_success(
                conversation_key="ticket:109",
                session_id=session_id,
            )
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                session_dir=session_dir,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = _transaction_safety_transport_request()

            with (
                mock.patch.object(
                    endpoint,
                    "_delete_codex_session",
                    side_effect=fake_delete,
                ),
                mock.patch(
                    "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                    side_effect=fake_run_streaming_subprocess,
                ),
            ):
                turn_task = asyncio.create_task(
                    endpoint.execute_json_turn(request.to_json())
                )
                await asyncio.wait_for(rewrite_started.wait(), timeout=1)
                turn_task.cancel()
                await asyncio.wait_for(deletion_started.wait(), timeout=1)

                turn_task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(turn_task.done())

                allow_deletion.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(turn_task, timeout=1)

            record = session_manager.load("ticket:109")

        self.assertTrue(deletion_completed.is_set())
        self.assertIsNone(record)

    async def test_codex_support_endpoint_limits_transaction_safety_rewrite_to_once(
        self,
    ) -> None:
        unsafe_response = SupportTurnResult(
            answer=f"Broadcast `{_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION}`.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Found a raw signed transaction.",
            used_tools=["shell"],
        ).to_json()
        session_id = "019dade1-5acf-70e2-9c61-f5ba37862a78"

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "sessions"
            session_manager = CodexSupportSessionManager(session_dir)
            session_manager.record_success(
                conversation_key="ticket:109",
                session_id=session_id,
            )
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                session_dir=session_dir,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = _transaction_safety_transport_request()

            with (
                mock.patch.object(
                    endpoint,
                    "_delete_codex_session",
                    new=mock.AsyncMock(return_value=True),
                ) as mock_delete,
                mock.patch(
                    "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                    return_value=CodexSupportExecutionOutput(
                        final_response_text=unsafe_response
                    ),
                ) as mock_run,
            ):
                with self.assertRaises(SignedTransactionSafetyViolation):
                    await endpoint.execute_json_turn(request.to_json())

            failed_record = session_manager.load("ticket:109")

        self.assertEqual(mock_run.call_count, 2)
        mock_delete.assert_awaited_once_with(session_id)
        self.assertIsNone(failed_record)

    async def test_codex_support_endpoint_detaches_session_when_deletion_fails(
        self,
    ) -> None:
        unsafe_response = SupportTurnResult(
            answer=f"Broadcast `{_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION}`.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Found a raw signed transaction.",
            used_tools=["shell"],
        ).to_json()
        safe_response = SupportTurnResult(
            answer="Use the wallet's built-in cancel flow and share only the public hash.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Kept troubleshooting read-only.",
            used_tools=["shell"],
        ).to_json()
        session_id = "019dade1-5acf-70e2-9c61-f5ba37862a78"
        calls: list[dict[str, object]] = []

        async def fake_run_streaming_subprocess(**kwargs):
            calls.append(kwargs)
            response = unsafe_response if len(calls) == 1 else safe_response
            return CodexSupportExecutionOutput(final_response_text=response)

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "sessions"
            session_manager = CodexSupportSessionManager(session_dir)
            session_manager.record_success(
                conversation_key="ticket:109",
                session_id=session_id,
            )
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                session_dir=session_dir,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = _transaction_safety_transport_request()

            with (
                mock.patch.object(
                    endpoint,
                    "_delete_codex_session",
                    new=mock.AsyncMock(return_value=False),
                ) as mock_delete,
                mock.patch(
                    "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                    side_effect=fake_run_streaming_subprocess,
                ),
                self.assertLogs(level="WARNING") as captured_logs,
            ):
                first_response = await endpoint.execute_json_turn(request.to_json())
                second_response = await endpoint.execute_json_turn(request.to_json())

            record = session_manager.load("ticket:109")

        self.assertEqual(
            TicketExecutionTransportResult.from_json(first_response).flow_outcome[
                "raw_final_reply"
            ],
            SupportTurnResult.from_json(safe_response).answer,
        )
        self.assertEqual(
            TicketExecutionTransportResult.from_json(second_response).flow_outcome[
                "raw_final_reply"
            ],
            SupportTurnResult.from_json(safe_response).answer,
        )
        mock_delete.assert_awaited_once_with(session_id)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("resume", calls[2]["command"])
        self.assertIsNone(record)
        self.assertTrue(
            any(
                "Detached contaminated Codex session" in line
                for line in captured_logs.output
            )
        )

    async def test_codex_support_endpoint_records_verification_failure_not_success(
        self,
    ) -> None:
        invalid_response = SupportTurnResult(
            answer="unsupported",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="unchecked",
            used_tools=["not_allowed"],
        ).to_json()
        session_id = "019dade1-5acf-70e2-9c61-f5ba37862a78"

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "sessions"
            session_manager = CodexSupportSessionManager(session_dir)
            session_manager.record_success(
                conversation_key="ticket:109",
                session_id=session_id,
            )
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                session_dir=session_dir,
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
                wants_bug_review_status=False,
            )

            with mock.patch(
                "ticket_investigation.codex_support_endpoint.run_codex_support_json_subprocess",
                return_value=CodexSupportExecutionOutput(
                    final_response_text=invalid_response
                ),
            ):
                with self.assertRaisesRegex(ValueError, "not allowed"):
                    await endpoint.execute_json_turn(request.to_json())

            failed_record = session_manager.load("ticket:109")
            self.assertIsNotNone(failed_record)
            assert failed_record is not None
            self.assertEqual(failed_record.run_count, 1)
            self.assertEqual(failed_record.consecutive_failures, 1)
