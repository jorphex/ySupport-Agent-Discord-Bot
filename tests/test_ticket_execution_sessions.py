from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from codex_support_sessions import CodexSupportSessionManager
from ticket_execution_sessions import main as ticket_execution_sessions_main


class TicketExecutionSessionsCliTests(unittest.TestCase):
    def test_summary_prints_active_session_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                artifact_dir="/tmp/run-1",
            )
            captured = io.StringIO()
            with patch(
                "ticket_execution_sessions.build_session_manager",
                return_value=manager,
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_sessions_main(["summary"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(captured.getvalue())
        self.assertEqual(payload["active_sessions"], 1)

    def test_reset_deletes_one_session_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=168)
            manager.record_success(
                conversation_key="ticket:123",
                session_id="019dade1-5acf-70e2-9c61-f5ba37862a78",
                artifact_dir="/tmp/run-1",
            )
            captured = io.StringIO()
            with patch(
                "ticket_execution_sessions.build_session_manager",
                return_value=manager,
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_sessions_main(["reset", "ticket:123"])

            remaining = manager.load("ticket:123")

        self.assertEqual(exit_code, 0)
        self.assertIsNone(remaining)
        payload = json.loads(captured.getvalue())
        self.assertTrue(payload["reset"])

    def test_prune_reports_removed_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=1)
            manager._record_path("ticket:123").write_text(
                json.dumps(
                    {
                        "conversation_key": "ticket:123",
                        "session_id": "019dade1-5acf-70e2-9c61-f5ba37862a78",
                        "created_at_utc": "2026-04-19T00:00:00+00:00",
                        "updated_at_utc": "2026-04-19T00:00:00+00:00",
                        "run_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            captured = io.StringIO()
            with patch(
                "ticket_execution_sessions.build_session_manager",
                return_value=manager,
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_sessions_main(["prune"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(captured.getvalue())
        self.assertEqual(payload["removed"], 1)
