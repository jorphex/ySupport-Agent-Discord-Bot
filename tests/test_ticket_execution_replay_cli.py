import tests as _test_environment  # noqa: F401

import tempfile
import unittest
from unittest.mock import patch

from ysupport import config
from ysupport.execution.replay_cli import (
    execute_request_json,
)


class TicketExecutionReplayCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_request_json_disables_production_sessions_by_default(
        self,
    ) -> None:
        original_session_dir = config.TICKET_EXECUTION_CODEX_SESSION_DIR
        observed = {}

        class _FakeEndpoint:
            async def execute_json_turn(self, request_json: str) -> str:
                observed["session_dir"] = config.TICKET_EXECUTION_CODEX_SESSION_DIR
                return '{"ok":true}'

        with patch(
            'ysupport.execution.replay_cli.build_ticket_execution_json_endpoint',
            return_value=_FakeEndpoint(),
        ):
            response_json = await execute_request_json(
                '{"request":"demo"}',
                mode="codex_support_exec",
                fallback_mode="",
            )

        self.assertEqual(response_json, '{"ok":true}')
        self.assertIsNone(observed["session_dir"])
        self.assertEqual(
            config.TICKET_EXECUTION_CODEX_SESSION_DIR,
            original_session_dir,
        )

    async def test_fresh_session_requires_explicit_isolated_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit replay session_dir"):
            await execute_request_json(
                '{"request":"demo"}',
                fresh_session=True,
            )

    async def test_replay_rejects_configured_live_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as session_dir:
            original_session_dir = config.TICKET_EXECUTION_CODEX_SESSION_DIR
            config.TICKET_EXECUTION_CODEX_SESSION_DIR = session_dir
            try:
                with self.assertRaisesRegex(ValueError, "configured live session"):
                    await execute_request_json(
                        '{"request":"demo"}',
                        session_dir=session_dir,
                    )
            finally:
                config.TICKET_EXECUTION_CODEX_SESSION_DIR = original_session_dir
