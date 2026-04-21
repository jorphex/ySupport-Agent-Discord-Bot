from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from codex_support_sessions import CodexSupportSessionManager
from ticket_investigation.codex_bundle import DEFAULT_CODEX_EXEC_COMMAND
from ticket_investigation.codex_support_endpoint import (
    _build_codex_support_command,
)
from ticket_investigation.transport import TicketExecutionTransportRequest


class CodexSupportSessionManagerTests(unittest.TestCase):
    def test_record_success_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                artifact_dir="/tmp/run-1",
            )

            record = manager.load("ticket:123")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.session_id, "019dade1-5acf-70e2-9c61-f5ba37862a78")
        self.assertEqual(record.run_count, 1)
        self.assertEqual(record.last_artifact_dir, "/tmp/run-1")

    def test_extract_session_id_from_stderr(self) -> None:
        manager = CodexSupportSessionManager("/tmp", max_age_hours=168)
        stderr_text = (
            "OpenAI Codex v0.117.0\n"
            "session id: 019dade1-5acf-70e2-9c61-f5ba37862a78\n"
        )
        self.assertEqual(
            manager.extract_session_id(stderr_text),
            "019dade1-5acf-70e2-9c61-f5ba37862a78",
        )

    def test_conversation_key_for_ticket_request(self) -> None:
        manager = CodexSupportSessionManager("/tmp", max_age_hours=168)
        request = TicketExecutionTransportRequest(
            aggregated_text="hello",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 123,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
            },
            investigation_job={
                "channel_id": 123,
                "requested_intent": "investigate_issue",
                "mode": "collecting",
                "evidence": {},
            },
            workflow_name="tests.session_key",
            wants_bug_review_status=False,
        )

        self.assertEqual(
            manager.conversation_key_for_request(request),
            "ticket:123",
        )

    def test_conversation_key_for_public_request_prefers_owner_id(self) -> None:
        manager = CodexSupportSessionManager("/tmp", max_age_hours=168)
        request = TicketExecutionTransportRequest(
            aggregated_text="hello",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 123,
                "is_public_trigger": True,
                "conversation_owner_id": 777,
                "project_context": "yearn",
                "initial_button_intent": None,
            },
            investigation_job={
                "channel_id": 123,
                "requested_intent": None,
                "mode": "idle",
                "evidence": {},
            },
            workflow_name="tests.public_session_key",
            wants_bug_review_status=False,
        )

        self.assertEqual(
            manager.conversation_key_for_request(request),
            "public_user:777",
        )

    def test_summary_reports_active_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                artifact_dir="/tmp/run-1",
                requested_intent="investigate_issue",
                guardrail_profile="ticket_support",
            )

            summary = manager.summary()

        self.assertEqual(summary["root_dir"], temp_dir)
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(len(summary["records"]), 1)
        self.assertEqual(
            summary["records"][0]["conversation_key"],
            "ticket:123",
        )

    def test_load_for_turn_resets_after_repeated_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                requested_intent="investigate_issue",
                guardrail_profile="ticket_support",
            )
            manager.record_failure(
                conversation_key="ticket:123",
                error_text="first failure",
            )
            manager.record_failure(
                conversation_key="ticket:123",
                error_text="second failure",
            )

            record = manager.load_for_turn(
                conversation_key="ticket:123",
                requested_intent="investigate_issue",
                guardrail_profile="ticket_support",
                human_handoff_active=False,
            )

        self.assertIsNone(record)
        self.assertFalse((Path(temp_dir) / "ticket_123.json").exists())

    def test_load_for_turn_resets_when_guardrail_profile_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                requested_intent="investigate_issue",
                guardrail_profile="ticket_support",
            )

            record = manager.load_for_turn(
                conversation_key="ticket:123",
                requested_intent="investigate_issue",
                guardrail_profile="public_support",
                human_handoff_active=False,
            )

        self.assertIsNone(record)
        self.assertFalse((Path(temp_dir) / "ticket_123.json").exists())

    def test_load_for_turn_resets_when_handoff_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                requested_intent="investigate_issue",
                guardrail_profile="ticket_support",
            )

            record = manager.load_for_turn(
                conversation_key="ticket:123",
                requested_intent="investigate_issue",
                guardrail_profile="ticket_support",
                human_handoff_active=True,
            )

        self.assertIsNone(record)
        self.assertFalse((Path(temp_dir) / "ticket_123.json").exists())

    def test_prune_expired_removes_old_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=1)
            record_path = Path(temp_dir) / "ticket_123.json"
            record_path.write_text(
                json.dumps(
                    {
                        "conversation_key": "ticket:123",
                        "session_id": "019dade1-5acf-70e2-9c61-f5ba37862a78",
                        "created_at_utc": "2026-04-19T00:00:00+00:00",
                        "updated_at_utc": "2026-04-19T00:00:00+00:00",
                        "run_count": 1,
                        "last_artifact_dir": "/tmp/run-1",
                    }
                ),
                encoding="utf-8",
            )

            removed = manager.prune_expired()

        self.assertEqual(removed, 1)
        self.assertFalse(record_path.exists())


class CodexSupportCommandBuilderTests(unittest.TestCase):
    def test_fresh_command_drops_ephemeral_and_keeps_schema(self) -> None:
        command = _build_codex_support_command(
            codex_command=DEFAULT_CODEX_EXEC_COMMAND,
            model="gpt-5.4",
            reasoning_effort="medium",
            response_schema_path=Path("/tmp/schema.json"),
            run_dir_path=Path("/tmp/run"),
            resume_session_id=None,
        )

        self.assertNotIn("--ephemeral", command)
        self.assertIn("--output-schema", command)
        self.assertIn("-C", command)

    def test_resume_command_uses_exec_resume_without_schema(self) -> None:
        command = _build_codex_support_command(
            codex_command=DEFAULT_CODEX_EXEC_COMMAND,
            model="gpt-5.4",
            reasoning_effort="medium",
            response_schema_path=Path("/tmp/schema.json"),
            run_dir_path=Path("/tmp/run"),
            resume_session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
        )

        self.assertEqual(command[:4], ["codex", "exec", "resume", "019dade1-5acf-70e2-9c61-f5ba37862a78"])
        self.assertNotIn("--ephemeral", command)
        self.assertNotIn("--output-schema", command)
        self.assertEqual(command[-1], "-")
