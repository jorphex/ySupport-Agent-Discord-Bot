from __future__ import annotations
import tests as _test_environment  # noqa: F401

import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from codex_support_sessions import CodexSupportSessionManager
from ticket_investigation.codex_support_endpoint import (
    DEFAULT_CODEX_EXEC_COMMAND,
    CodexSupportTicketExecutionJsonEndpoint,
    _build_codex_support_command,
    _build_codex_delete_command,
    _expired_unreferenced_rollout_ids,
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
            record_path = Path(temp_dir) / "ticket_123.json"
            record_mode = record_path.stat().st_mode & 0o777

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.session_id, "019dade1-5acf-70e2-9c61-f5ba37862a78")
        self.assertEqual(record.run_count, 1)
        self.assertEqual(record.last_artifact_dir, "/tmp/run-1")
        self.assertEqual(record_mode, 0o600)

    def test_failed_atomic_write_preserves_previous_session_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
            )

            with patch(
                "codex_support_sessions.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    manager.record_success(
                        conversation_key="ticket:123",
                        session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                    )

            record = manager.load("ticket:123")
            temporary_files = list(Path(temp_dir).glob("*.tmp"))

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.run_count, 1)
        self.assertEqual(temporary_files, [])

    def test_invalid_session_record_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record_path = Path(temp_dir) / "ticket_123.json"
            record_path.write_text(
                json.dumps(
                    {
                        "conversation_key": "ticket:123",
                        "session_id": "------------------------------------",
                        "created_at_utc": "not-a-timestamp",
                        "updated_at_utc": 123,
                        "run_count": "one",
                    }
                ),
                encoding="utf-8",
            )
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)

            record = manager.load("ticket:123")

        self.assertIsNone(record)

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

    def test_extract_session_id_from_thread_started_jsonl(self) -> None:
        manager = CodexSupportSessionManager("/tmp", max_age_hours=168)
        stdout_text = "\n".join(
            [
                "not-json",
                json.dumps({"type": "item.started", "item": {}}),
                json.dumps(
                    {
                        "type": "thread.started",
                        "thread_id": "019dade1-5acf-70e2-9c61-f5ba37862a78",
                    }
                ),
            ]
        )

        self.assertEqual(
            manager.extract_session_id_from_jsonl(stdout_text),
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

    def test_conversation_key_is_absent_without_owner_or_channel(self) -> None:
        manager = CodexSupportSessionManager("/tmp", max_age_hours=168)
        request = TicketExecutionTransportRequest(
            aggregated_text="hello",
            input_list=[],
            current_history=[],
            run_context={"is_public_trigger": True},
            investigation_job={"evidence": {}},
            workflow_name="tests.unkeyed",
            wants_bug_review_status=False,
        )

        self.assertIsNone(manager.conversation_key_for_request(request))

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
    def test_delete_command_uses_supported_force_delete(self) -> None:
        command = _build_codex_delete_command(
            codex_command=DEFAULT_CODEX_EXEC_COMMAND,
            session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
        )

        self.assertEqual(
            command,
            [
                "codex",
                "delete",
                "--force",
                "019dade1-5acf-70e2-9c61-f5ba37862a78",
            ],
        )

    def test_expired_rollout_scan_preserves_active_and_recent_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir) / "sessions" / "2026" / "07" / "01"
            sessions_dir.mkdir(parents=True)
            expired_id = "019dade1-5acf-70e2-9c61-f5ba37862a78"
            active_id = "019dade1-5acf-70e2-9c61-f5ba37862a79"
            recent_id = "019dade1-5acf-70e2-9c61-f5ba37862a80"
            old_timestamp = (
                datetime.now(timezone.utc) - timedelta(days=8)
            ).timestamp()
            for session_id in (expired_id, active_id):
                path = sessions_dir / f"rollout-2026-07-01T00-00-00-{session_id}.jsonl"
                path.write_text("{}\n", encoding="utf-8")
                os.utime(path, (old_timestamp, old_timestamp))
            recent_path = (
                sessions_dir
                / f"rollout-2026-07-01T00-00-00-{recent_id}.jsonl"
            )
            recent_path.write_text("{}\n", encoding="utf-8")

            session_ids = _expired_unreferenced_rollout_ids(
                Path(temp_dir),
                active_session_ids={active_id},
                cutoff=datetime.now(timezone.utc) - timedelta(days=7),
            )

        self.assertEqual(session_ids, [expired_id])


class CodexSupportSessionCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_deletes_only_expired_unreferenced_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = Path(temp_dir) / "home"
            session_dir = Path(temp_dir) / "active"
            rollout_dir = codex_home / "sessions" / "2026" / "07" / "01"
            rollout_dir.mkdir(parents=True)
            active_id = "019dade1-5acf-70e2-9c61-f5ba37862a79"
            expired_id = "019dade1-5acf-70e2-9c61-f5ba37862a78"
            old_timestamp = (
                datetime.now(timezone.utc) - timedelta(days=8)
            ).timestamp()
            for session_id in (active_id, expired_id):
                path = rollout_dir / f"rollout-2026-07-01T00-00-00-{session_id}.jsonl"
                path.write_text("{}\n", encoding="utf-8")
                os.utime(path, (old_timestamp, old_timestamp))
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=DEFAULT_CODEX_EXEC_COMMAND,
                codex_home=codex_home,
                session_dir=session_dir,
                session_max_age_hours=168,
            )
            endpoint.session_manager.record_success(
                conversation_key="ticket:123",
                session_id=active_id,
            )

            with patch.object(
                endpoint,
                "_delete_codex_session",
                AsyncMock(return_value=True),
            ) as delete_session:
                removed = await endpoint.prune_expired_sessions()

        self.assertEqual(removed, 1)
        delete_session.assert_awaited_once_with(expired_id)

    def test_fresh_command_drops_ephemeral_and_keeps_schema(self) -> None:
        command = _build_codex_support_command(
            codex_command=DEFAULT_CODEX_EXEC_COMMAND,
            model="gpt-5.4",
            reasoning_effort="medium",
            response_schema_path=Path("/tmp/schema.json"),
            run_dir_path=Path("/tmp/run"),
            image_paths=[],
            resume_session_id=None,
        )

        self.assertNotIn("--ephemeral", command)
        self.assertIn("--output-schema", command)
        self.assertIn("-C", command)

    def test_resume_command_uses_exec_resume_with_schema(self) -> None:
        command = _build_codex_support_command(
            codex_command=DEFAULT_CODEX_EXEC_COMMAND,
            model="gpt-5.4",
            reasoning_effort="medium",
            response_schema_path=Path("/tmp/schema.json"),
            run_dir_path=Path("/tmp/run"),
            image_paths=[],
            resume_session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
        )

        self.assertEqual(command[:4], ["codex", "exec", "resume", "019dade1-5acf-70e2-9c61-f5ba37862a78"])
        self.assertNotIn("--ephemeral", command)
        self.assertIn("--output-schema", command)
        self.assertEqual(command[command.index("--output-schema") + 1], "/tmp/schema.json")
        self.assertEqual(command[-1], "-")

    def test_command_passes_each_image_to_codex(self) -> None:
        command = _build_codex_support_command(
            codex_command=DEFAULT_CODEX_EXEC_COMMAND,
            model="gpt-5.4",
            reasoning_effort="medium",
            response_schema_path=Path("/tmp/schema.json"),
            run_dir_path=Path("/tmp/run"),
            image_paths=[Path("/tmp/first.png"), Path("/tmp/second.jpg")],
            resume_session_id=None,
        )

        image_arguments = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "-i"
        ]
        self.assertEqual(image_arguments, ["/tmp/first.png", "/tmp/second.jpg"])
