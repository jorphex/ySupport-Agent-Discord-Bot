from __future__ import annotations
import tests as _test_environment  # noqa: F401

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ysupport.execution.codex_support_sessions import CodexSupportSessionManager
from ysupport.execution.sessions import main as ticket_execution_sessions_main


class TicketExecutionSessionsCliTests(unittest.TestCase):
    def test_summary_does_not_prune_expired_session_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CodexSupportSessionManager(temp_dir, max_age_hours=1)
            record_path = manager._record_path("ticket:expired")
            record_path.write_text(
                json.dumps(
                    {
                        "conversation_key": "ticket:expired",
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
                'ysupport.execution.sessions.build_session_manager',
                return_value=manager,
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_sessions_main(["summary"])

            self.assertTrue(record_path.exists())

        self.assertEqual(exit_code, 0)
        payload = json.loads(captured.getvalue())
        self.assertEqual(payload["active_sessions"], 0)
