from tests import TEST_STATE_ROOT

import io
import json
import os
from pathlib import Path
import sys
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
    def test_telegram_summary_uses_the_5_6_cost_efficient_role(self) -> None:
        self.assertEqual(config.TELEGRAM_HANDOFF_SUMMARY_MODEL, "gpt-5.6-terra")
        self.assertEqual(config.TELEGRAM_HANDOFF_SUMMARY_REASONING_EFFORT, "low")

    def test_mcp_docker_context_is_an_explicit_tracked_allowlist(self) -> None:
        dockerignore = Path("Dockerfile.mcp.dockerignore").read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(dockerignore[0], "**")
        self.assertIn("!Dockerfile.mcp", dockerignore)
        self.assertIn("!mcp_server.py", dockerignore)
        self.assertIn("!yearn_rag/repo_sources.json", dockerignore)
        self.assertNotIn("!.env", dockerignore)

    def test_host_launcher_requires_the_project_python_and_explicit_auth(self) -> None:
        launcher = Path("scripts/run_ysupport_host.sh").read_text(encoding="utf-8")

        self.assertIn('PYTHON="${REPO_ROOT}/.venv/bin/python"', launcher)
        self.assertIn("TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE:?", launcher)
        self.assertNotIn("exec python3", launcher)

    def test_support_dashboard_tls_verification_defaults_on(self) -> None:
        self.assertTrue(config.SUPPORT_DASHBOARD_VERIFY_SSL)

    def test_mcp_compose_keeps_dashboard_tls_verification_on_by_default(
        self,
    ) -> None:
        compose = Path("compose.mcp.yaml").read_text(encoding="utf-8")

        self.assertIn(
            'SUPPORT_DASHBOARD_VERIFY_SSL: "${SUPPORT_DASHBOARD_VERIFY_SSL:-true}"',
            compose,
        )

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

    def test_ticket_execution_test_state_keeps_runs_ephemeral(self) -> None:
        self.assertEqual(
            config.TICKET_EXECUTION_STATE_ROOT,
            Path(TEST_STATE_ROOT),
        )
        self.assertEqual(
            config.TICKET_EXECUTION_CODEX_HOME,
            f"{TEST_STATE_ROOT}/home",
        )
        self.assertEqual(
            config.TICKET_EXECUTION_ARTIFACT_DIR,
            "",
        )
        self.assertEqual(
            config.TICKET_EXECUTION_SHADOW_ARTIFACT_DIR,
            "",
        )
        self.assertEqual(
            config.TICKET_EXECUTION_CODEX_SESSION_DIR,
            f"{TEST_STATE_ROOT}/sessions",
        )

    def test_build_rpc_urls_prefers_explicit_per_chain_env(self) -> None:
        rpc_urls = config.build_rpc_urls(
            {
                "ETHEREUM_RPC_URL": "https://ethereum.example",
                "BASE_RPC_URL": "https://base.example",
                "ARBITRUM_RPC_URL": "https://arbitrum.example",
                "OPTIMISM_RPC_URL": "https://optimism.example",
                "POLYGON_RPC_URL": "https://polygon.example",
                "SONIC_RPC_URL": "https://sonic.example",
                "KATANA_RPC_URL": "https://katana.example",
            }
        )

        self.assertEqual(rpc_urls["ethereum"], "https://ethereum.example")
        self.assertEqual(rpc_urls["base"], "https://base.example")
        self.assertEqual(rpc_urls["arbitrum"], "https://arbitrum.example")
        self.assertEqual(rpc_urls["optimism"], "https://optimism.example")
        self.assertEqual(rpc_urls["polygon"], "https://polygon.example")
        self.assertEqual(rpc_urls["sonic"], "https://sonic.example")
        self.assertEqual(rpc_urls["katana"], "https://katana.example")

    def test_build_rpc_urls_falls_back_to_alchemy_and_default_katana(self) -> None:
        original_alchemy_key = config.ALCHEMY_KEY
        try:
            config.ALCHEMY_KEY = "alchemy-test-key"
            rpc_urls = config.build_rpc_urls({})
        finally:
            config.ALCHEMY_KEY = original_alchemy_key

        self.assertEqual(
            rpc_urls["ethereum"],
            "https://eth-mainnet.g.alchemy.com/v2/alchemy-test-key",
        )
        self.assertEqual(
            rpc_urls["base"],
            "https://base-mainnet.g.alchemy.com/v2/alchemy-test-key",
        )
        self.assertEqual(
            rpc_urls["arbitrum"],
            "https://arb-mainnet.g.alchemy.com/v2/alchemy-test-key",
        )
        self.assertEqual(
            rpc_urls["optimism"],
            "https://opt-mainnet.g.alchemy.com/v2/alchemy-test-key",
        )
        self.assertEqual(
            rpc_urls["polygon"],
            "https://polygon-mainnet.g.alchemy.com/v2/alchemy-test-key",
        )
        self.assertEqual(
            rpc_urls["sonic"],
            "https://sonic-mainnet.g.alchemy.com/v2/alchemy-test-key",
        )
        self.assertEqual(
            rpc_urls["katana"],
            "https://katana-mainnet.g.alchemy.com/v2/alchemy-test-key",
        )

    def test_ticket_execution_runtime_summary_includes_fallback_and_codex_details(
        self,
    ) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_model = config.TICKET_EXECUTION_CODEX_MODEL
        original_reasoning = config.TICKET_EXECUTION_CODEX_REASONING_EFFORT
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_CODEX_MODEL = "gpt-5.6-sol"
            config.TICKET_EXECUTION_CODEX_REASONING_EFFORT = "medium"
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"

            summary = config.ticket_execution_runtime_summary()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_MODEL = original_model
            config.TICKET_EXECUTION_CODEX_REASONING_EFFORT = original_reasoning
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIn("primary=codex_support_exec", summary)
        self.assertIn("fallback=local", summary)
        self.assertIn("codex_model=gpt-5.6-sol", summary)
        self.assertIn("codex_reasoning=medium", summary)
        self.assertIn(f"state_root={TEST_STATE_ROOT}", summary)
        self.assertIn(f"codex_session_dir={TEST_STATE_ROOT}/sessions", summary)
        self.assertIn("artifact_dir=/tmp/ticket-artifacts", summary)

    def test_ticket_execution_runtime_warnings_flag_primary_codex_without_fallback(
        self,
    ) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            warnings = config.ticket_execution_runtime_warnings()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback

        self.assertIn(
            "primary codex_support_exec is enabled without a fallback endpoint",
            warnings,
        )

    def test_ticket_execution_runtime_validation_allows_ephemeral_codex_runs(
        self,
    ) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_ARTIFACT_DIR = ""
            config.validate_ticket_execution_runtime_config()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

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

    async def test_pretriage_repo_claim_combines_repo_search_artifacts_and_docs(
        self,
    ) -> None:
        async def fake_search_repo_context(
            query: str,
            limit=None,
            include_legacy: bool = False,
        ) -> str:
            self.assertIn("unstake", query)
            return "Top repo hits:\n- segment:11518\n- segment:11540"

        async def fake_fetch_repo_artifacts(artifact_refs_text: str) -> str:
            self.assertEqual(artifact_refs_text, "segment:11518, segment:11540")
            return "Fetched repo artifacts:\nLiquidLockerDepositor.vy excerpts"

        async def fake_answer_from_docs(user_query: str) -> str:
            self.assertIn("unstake", user_query)
            return "Official docs say unstaking starts a 14-day linear cooldown."

        with (
            patch.object(
                docs_repo_tools,
                "core_search_repo_context",
                new=fake_search_repo_context,
            ),
            patch.object(
                docs_repo_tools,
                "core_fetch_repo_artifacts",
                new=fake_fetch_repo_artifacts,
            ),
            patch.object(
                docs_repo_tools, "core_answer_from_docs", new=fake_answer_from_docs
            ),
        ):
            result = await docs_repo_tools.core_pretriage_repo_claim(
                "unstake resets the stream start time to block.timestamp"
            )

        self.assertIn("Repo search:", result)
        self.assertIn("segment:11518", result)
        self.assertIn("LiquidLockerDepositor.vy", result)
        self.assertIn("Docs context:", result)


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

    def test_build_ticket_execution_status_reports_repo_context_and_valid_config(
        self,
    ) -> None:
        with (
            patch(
                "ticket_execution.status.get_repo_context_status",
                return_value={"state": "ready", "fresh": True},
            ),
            patch(
                "ticket_execution.status._build_codex_session_summary",
                return_value={"root_dir": "/tmp/sessions", "active_sessions": 2},
            ),
        ):
            status = build_ticket_execution_status()

        self.assertIn("ticket_execution", status)
        self.assertIn("runtime_environment", status)
        self.assertIn("repo_context", status)
        self.assertTrue(status["ticket_execution"]["validation_ok"])
        self.assertTrue(status["runtime_environment"]["validation_ok"])
        self.assertNotIn("endpoint_build_ok", status["ticket_execution"])
        self.assertEqual(
            status["ticket_execution"]["sandbox_policy"]["workspace_mode"],
            "temporary_per_turn",
        )
        self.assertEqual(
            status["ticket_execution"]["sandbox_policy"]["export_mode"],
            "ephemeral_only",
        )
        self.assertIsNone(status["ticket_execution"]["artifact_dir"])
        self.assertEqual(
            status["ticket_execution"]["codex_session_summary"],
            {"root_dir": "/tmp/sessions", "active_sessions": 2},
        )
        self.assertEqual(status["repo_context"]["state"], "ready")

    def test_build_ticket_execution_status_smoke_probe_reports_success_for_local_endpoint(
        self,
    ) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        try:
            config.TICKET_EXECUTION_ENDPOINT = "local"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            with patch(
                "ticket_execution.status.get_repo_context_status",
                return_value={"state": "ready", "fresh": True},
            ):
                status = build_ticket_execution_status(include_smoke_probe=True)
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback

        smoke_probe = status["ticket_execution"]["smoke_probe"]
        self.assertTrue(smoke_probe["ok"])
        self.assertEqual(
            smoke_probe["raw_final_reply"], "ticket_execution_smoke_ok:local"
        )

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

    def test_build_ticket_execution_status_reports_command_probe_for_codex(
        self,
    ) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_CODEX_COMMAND = [
                sys.executable,
                "-c",
                "print('ok')",
            ]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            with patch(
                "ticket_execution.status.get_repo_context_status",
                return_value={"state": "ready", "fresh": True},
            ):
                status = build_ticket_execution_status()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertTrue(status["ticket_execution"]["validation_ok"])
        primary_probe = status["ticket_execution"]["primary_command_probe"]
        self.assertIsNotNone(primary_probe)
        assert primary_probe is not None
        self.assertTrue(primary_probe["available"])
        self.assertEqual(primary_probe["command"][:2], [sys.executable, "-c"])

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
