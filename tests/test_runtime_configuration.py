import tests as _test_environment  # noqa: F401

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


import config
import docs_repo_tools
from ticket_execution.status import (
    _build_codex_session_summary,
    build_ticket_execution_status,
    main as ticket_execution_status_main,
)


class ConfigSummaryTests(unittest.TestCase):
    def test_invalid_boolean_env_keeps_safe_default_and_reports_error(self) -> None:
        config._INVALID_ENV_ERRORS.pop("TEST_BOOLEAN_SETTING", None)
        try:
            with patch.dict(
                os.environ,
                {"TEST_BOOLEAN_SETTING": "treu"},
            ):
                value = config._env_bool("TEST_BOOLEAN_SETTING", default=True)

            self.assertTrue(value)
            self.assertEqual(
                config._INVALID_ENV_ERRORS["TEST_BOOLEAN_SETTING"],
                "TEST_BOOLEAN_SETTING must be a boolean",
            )
        finally:
            config._INVALID_ENV_ERRORS.pop("TEST_BOOLEAN_SETTING", None)

    def test_invalid_integer_env_keeps_default_and_reports_error(self) -> None:
        config._INVALID_ENV_ERRORS.pop("TEST_INTEGER_SETTING", None)
        try:
            with patch.dict(
                os.environ,
                {"TEST_INTEGER_SETTING": "sixty"},
            ):
                value = config._env_int("TEST_INTEGER_SETTING", 60)

            self.assertEqual(value, 60)
            self.assertEqual(
                config._INVALID_ENV_ERRORS["TEST_INTEGER_SETTING"],
                "TEST_INTEGER_SETTING must be an integer",
            )
        finally:
            config._INVALID_ENV_ERRORS.pop("TEST_INTEGER_SETTING", None)

    def test_ticket_execution_runtime_validation_covers_active_shadow_codex(
        self,
    ) -> None:
        with patch.multiple(
            config,
            TICKET_EXECUTION_ENDPOINT="local",
            TICKET_EXECUTION_FALLBACK_ENDPOINT="",
            TICKET_EXECUTION_SHADOW_ENDPOINT="codex_support_exec",
            TICKET_EXECUTION_CANARY_ENDPOINT="",
            TICKET_EXECUTION_CODEX_HOME=None,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires TICKET_EXECUTION_CODEX_HOME",
            ):
                config.validate_ticket_execution_runtime_config()

    def test_ticket_execution_runtime_validation_requires_explicit_mcp_url(
        self,
    ) -> None:
        with patch.multiple(
            config,
            TICKET_EXECUTION_ENDPOINT="codex_support_exec",
            TICKET_EXECUTION_FALLBACK_ENDPOINT="",
            TICKET_EXECUTION_SHADOW_ENDPOINT="",
            TICKET_EXECUTION_CANARY_ENDPOINT="",
            TICKET_EXECUTION_CODEX_HOME="/tmp/codex-home",
            MCP_SERVER_API_KEY="mcp-key",
            TICKET_EXECUTION_CODEX_YSUPPORT_MCP_URL="",
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires TICKET_EXECUTION_CODEX_YSUPPORT_MCP_URL",
            ):
                config.validate_ticket_execution_runtime_config()

    def test_ticket_execution_runtime_validation_covers_active_canary_codex(
        self,
    ) -> None:
        with patch.multiple(
            config,
            TICKET_EXECUTION_ENDPOINT="local",
            TICKET_EXECUTION_FALLBACK_ENDPOINT="",
            TICKET_EXECUTION_SHADOW_ENDPOINT="",
            TICKET_EXECUTION_CANARY_ENDPOINT="codex_support_exec",
            TICKET_EXECUTION_CANARY_CHANNEL_IDS={"42"},
            TICKET_EXECUTION_CANARY_INTENTS=set(),
            TICKET_EXECUTION_CODEX_HOME=None,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires TICKET_EXECUTION_CODEX_HOME",
            ):
                config.validate_ticket_execution_runtime_config()

    def test_unselected_canary_is_not_an_active_execution_mode(self) -> None:
        with patch.multiple(
            config,
            TICKET_EXECUTION_ENDPOINT="local",
            TICKET_EXECUTION_FALLBACK_ENDPOINT="",
            TICKET_EXECUTION_SHADOW_ENDPOINT="",
            TICKET_EXECUTION_CANARY_ENDPOINT="codex_support_exec",
            TICKET_EXECUTION_CANARY_CHANNEL_IDS=set(),
            TICKET_EXECUTION_CANARY_INTENTS=set(),
            TICKET_EXECUTION_CODEX_HOME=None,
        ):
            self.assertEqual(config.ticket_execution_endpoint_modes(), ("local",))
            config.validate_ticket_execution_runtime_config()

    def test_runtime_environment_validation_requires_core_bot_settings(self) -> None:
        original_openai = config.OPENAI_API_KEY
        original_token = config.DISCORD_BOT_TOKEN
        original_category = config.YEARN_TICKET_CATEGORY_ID
        original_trigger = config.YEARN_PUBLIC_TRIGGER_CHAR
        original_pr_channel = config.PR_MARKETING_CHANNEL_ID
        try:
            config.OPENAI_API_KEY = None
            config.DISCORD_BOT_TOKEN = None
            config.YEARN_TICKET_CATEGORY_ID = None
            config.YEARN_PUBLIC_TRIGGER_CHAR = None
            config.PR_MARKETING_CHANNEL_ID = None
            with self.assertRaises(ValueError) as exc:
                config.validate_runtime_environment_config()
        finally:
            config.OPENAI_API_KEY = original_openai
            config.DISCORD_BOT_TOKEN = original_token
            config.YEARN_TICKET_CATEGORY_ID = original_category
            config.YEARN_PUBLIC_TRIGGER_CHAR = original_trigger
            config.PR_MARKETING_CHANNEL_ID = original_pr_channel

        self.assertIn("OPENAI_API_KEY is required", str(exc.exception))
        self.assertIn("DISCORD_BOT_TOKEN is required", str(exc.exception))


class ReportArtifactFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_report_artifact_reads_public_gist_via_github_api(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "files": {
                        "report.md": {
                            "content": "Issue: unstake resets stream time via block.timestamp",
                        }
                    }
                }

        with patch.object(
            docs_repo_tools.requests, "get", return_value=FakeResponse()
        ) as mock_get:
            result = await docs_repo_tools.core_fetch_report_artifact(
                "https://gist.github.com/example/abcdef1234567890"
            )

        self.assertIn("Fetched public report artifact", result)
        self.assertIn("unstake resets stream time", result)
        self.assertIn("abcdef1234567890", result)
        mock_get.assert_called_once()


    async def test_fetch_report_artifact_rejects_unsupported_hosts(self) -> None:
        result = await docs_repo_tools.core_fetch_report_artifact(
            "https://example.com/report.txt"
        )
        self.assertIn("Unsupported report URL", result)


class TicketExecutionStatusTests(unittest.TestCase):
    def test_missing_session_directory_is_reported_without_creation(self) -> None:
        original_session_dir = config.TICKET_EXECUTION_CODEX_SESSION_DIR
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                missing_path = Path(temp_dir) / "missing-sessions"
                config.TICKET_EXECUTION_CODEX_SESSION_DIR = str(missing_path)

                summary = _build_codex_session_summary()

                self.assertEqual(summary["active_sessions"], 0)
                self.assertFalse(missing_path.exists())
        finally:
            config.TICKET_EXECUTION_CODEX_SESSION_DIR = original_session_dir

    def test_ticket_execution_status_main_returns_nonzero_without_codex_home(
        self,
    ) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_home = config.TICKET_EXECUTION_CODEX_HOME
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_HOME = None
            captured = io.StringIO()
            with patch(
                "ticket_execution.status.get_repo_context_status",
                return_value={"state": "disabled", "fresh": False},
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_status_main([])
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_HOME = original_home

        self.assertEqual(exit_code, 1)
        payload = json.loads(captured.getvalue())
        self.assertFalse(payload["ticket_execution"]["validation_ok"])
        self.assertIn(
            "requires TICKET_EXECUTION_CODEX_HOME",
            payload["ticket_execution"]["validation_error"],
        )

    def test_ticket_execution_status_main_returns_nonzero_for_invalid_runtime_env(
        self,
    ) -> None:
        original_token = config.DISCORD_BOT_TOKEN
        original_category = config.YEARN_TICKET_CATEGORY_ID
        try:
            config.DISCORD_BOT_TOKEN = None
            config.YEARN_TICKET_CATEGORY_ID = None
            captured = io.StringIO()
            with patch(
                "ticket_execution.status.get_repo_context_status",
                return_value={"state": "disabled", "fresh": False},
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_status_main([])
        finally:
            config.DISCORD_BOT_TOKEN = original_token
            config.YEARN_TICKET_CATEGORY_ID = original_category

        self.assertEqual(exit_code, 1)
        payload = json.loads(captured.getvalue())
        self.assertFalse(payload["runtime_environment"]["validation_ok"])
        self.assertIn(
            "DISCORD_BOT_TOKEN is required",
            payload["runtime_environment"]["validation_error"],
        )

    def test_build_ticket_execution_status_resolves_relative_command_against_configured_cwd(
        self,
    ) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_cwd = config.TICKET_EXECUTION_SUBPROCESS_CWD
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                worker_path = os.path.join(temp_dir, "worker.sh")
                with open(worker_path, "w", encoding="utf-8") as worker_file:
                    worker_file.write("#!/bin/sh\nexit 0\n")
                os.chmod(worker_path, 0o755)
                config.TICKET_EXECUTION_ENDPOINT = "subprocess"
                config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
                config.TICKET_EXECUTION_SUBPROCESS_COMMAND = ["./worker.sh", "--json"]
                config.TICKET_EXECUTION_SUBPROCESS_CWD = temp_dir
                with patch(
                    "ticket_execution.status.get_repo_context_status",
                    return_value={"state": "ready", "fresh": True},
                ):
                    status = build_ticket_execution_status()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_SUBPROCESS_CWD = original_cwd

        primary_probe = status["ticket_execution"]["primary_command_probe"]
        self.assertIsNotNone(primary_probe)
        assert primary_probe is not None
        self.assertTrue(primary_probe["available"])
        self.assertEqual(primary_probe["resolved_path"], os.path.realpath(worker_path))
