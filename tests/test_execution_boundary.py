import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import config
from state import BotRunContext, TicketInvestigationJob
from ticket_execution_status import (
    build_ticket_execution_status,
    main as ticket_execution_status_main,
)
from ticket_investigation_codex_bundle import (
    DEFAULT_CODEX_EXEC_COMMAND,
    build_codex_ticket_execution_bundle,
)
from ticket_investigation_codex_endpoint import CodexExecTicketExecutionJsonEndpoint
from ticket_investigation_json_endpoint import (
    ExecutorBackedTicketExecutionJsonEndpoint,
    FailoverTicketExecutionJsonEndpoint,
    build_ticket_execution_json_endpoint,
)
from ticket_investigation_runtime import TicketAgentFlowOutcome, TicketTurnRequest
from ticket_investigation_subprocess_endpoint import SubprocessTicketExecutionJsonEndpoint
from ticket_investigation_transport import (
    TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA,
    TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA,
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)
import tools_lib


class ConfigSummaryTests(unittest.TestCase):
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
        self.assertEqual(rpc_urls["katana"], "https://rpc.katana.network")

    def test_ticket_execution_runtime_summary_includes_fallback_and_codex_details(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_model = config.TICKET_EXECUTION_CODEX_MODEL
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_CODEX_MODEL = "gpt-5.4"
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"

            summary = config.ticket_execution_runtime_summary()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_MODEL = original_model
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIn("primary=codex_exec", summary)
        self.assertIn("fallback=local", summary)
        self.assertIn("codex_model=gpt-5.4", summary)
        self.assertIn("artifact_dir=/tmp/ticket-artifacts", summary)

    def test_ticket_execution_runtime_warnings_flag_primary_codex_without_fallback(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            warnings = config.ticket_execution_runtime_warnings()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback

        self.assertIn(
            "primary codex_exec is enabled without a fallback endpoint",
            warnings,
        )

    def test_ticket_execution_runtime_validation_requires_workspace_for_codex(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_run_dir_root = config.TICKET_EXECUTION_RUN_DIR_ROOT
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_ARTIFACT_DIR = ""
            config.TICKET_EXECUTION_RUN_DIR_ROOT = ""
            with self.assertRaises(ValueError):
                config.validate_ticket_execution_runtime_config()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
            config.TICKET_EXECUTION_RUN_DIR_ROOT = original_run_dir_root

    def test_runtime_environment_validation_requires_core_bot_settings(self) -> None:
        original_openai = config.OPENAI_API_KEY
        original_token = config.DISCORD_BOT_TOKEN
        original_category = config.YEARN_TICKET_CATEGORY_ID
        original_trigger = config.YEARN_PUBLIC_TRIGGER_CHAR
        original_pr_channel = config.PR_MARKETING_CHANNEL_ID
        original_handoff = config.HUMAN_HANDOFF_TARGET_USER_ID
        try:
            config.OPENAI_API_KEY = None
            config.DISCORD_BOT_TOKEN = None
            config.YEARN_TICKET_CATEGORY_ID = None
            config.YEARN_PUBLIC_TRIGGER_CHAR = None
            config.PR_MARKETING_CHANNEL_ID = None
            config.HUMAN_HANDOFF_TARGET_USER_ID = None
            with self.assertRaises(ValueError) as exc:
                config.validate_runtime_environment_config()
        finally:
            config.OPENAI_API_KEY = original_openai
            config.DISCORD_BOT_TOKEN = original_token
            config.YEARN_TICKET_CATEGORY_ID = original_category
            config.YEARN_PUBLIC_TRIGGER_CHAR = original_trigger
            config.PR_MARKETING_CHANNEL_ID = original_pr_channel
            config.HUMAN_HANDOFF_TARGET_USER_ID = original_handoff

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

        with patch("tools_lib.requests.get", return_value=FakeResponse()) as mock_get:
            result = await tools_lib.core_fetch_report_artifact(
                "https://gist.github.com/example/abcdef1234567890"
            )

        self.assertIn("Fetched public report artifact", result)
        self.assertIn("unstake resets stream time", result)
        self.assertIn("abcdef1234567890", result)
        mock_get.assert_called_once()

    async def test_fetch_report_artifact_rejects_unsupported_hosts(self) -> None:
        result = await tools_lib.core_fetch_report_artifact("https://example.com/report.txt")
        self.assertIn("Unsupported report URL", result)

    async def test_pretriage_repo_claim_combines_repo_search_artifacts_and_docs(self) -> None:
        async def fake_search_repo_context(
            query: str,
            limit=None,
            include_legacy: bool = False,
            include_ui: bool = False,
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
            patch("tools_lib.core_search_repo_context", new=fake_search_repo_context),
            patch("tools_lib.core_fetch_repo_artifacts", new=fake_fetch_repo_artifacts),
            patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs),
        ):
            result = await tools_lib.core_pretriage_repo_claim(
                "unstake resets the stream start time to block.timestamp"
            )

        self.assertIn("Repo search:", result)
        self.assertIn("segment:11518", result)
        self.assertIn("LiquidLockerDepositor.vy", result)
        self.assertIn("Docs context:", result)


class TicketExecutionStatusTests(unittest.TestCase):
    def test_build_ticket_execution_status_reports_repo_context_and_valid_config(self) -> None:
        with patch(
            "ticket_execution_status.get_repo_context_status",
            return_value={"state": "ready", "fresh": True},
        ):
            status = build_ticket_execution_status()

        self.assertIn("ticket_execution", status)
        self.assertIn("runtime_environment", status)
        self.assertIn("repo_context", status)
        self.assertTrue(status["ticket_execution"]["validation_ok"])
        self.assertTrue(status["runtime_environment"]["validation_ok"])
        self.assertTrue(status["ticket_execution"]["endpoint_build_ok"])
        self.assertEqual(
            status["ticket_execution"]["sandbox_policy"]["workspace_mode"],
            "temporary_per_turn",
        )
        self.assertEqual(status["repo_context"]["state"], "ready")

    def test_build_ticket_execution_status_smoke_probe_reports_success_for_local_endpoint(self) -> None:
        with patch(
            "ticket_execution_status.get_repo_context_status",
            return_value={"state": "ready", "fresh": True},
        ):
            status = build_ticket_execution_status(include_smoke_probe=True)

        smoke_probe = status["ticket_execution"]["smoke_probe"]
        self.assertTrue(smoke_probe["ok"])
        self.assertEqual(smoke_probe["raw_final_reply"], "ticket_execution_smoke_ok:local")

    def test_ticket_execution_status_main_returns_nonzero_for_invalid_codex_policy(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_run_dir_root = config.TICKET_EXECUTION_RUN_DIR_ROOT
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_ARTIFACT_DIR = ""
            config.TICKET_EXECUTION_RUN_DIR_ROOT = ""
            captured = io.StringIO()
            with patch(
                "ticket_execution_status.get_repo_context_status",
                return_value={"state": "disabled", "fresh": False},
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_status_main()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
            config.TICKET_EXECUTION_RUN_DIR_ROOT = original_run_dir_root

        self.assertEqual(exit_code, 1)
        payload = json.loads(captured.getvalue())
        self.assertFalse(payload["ticket_execution"]["validation_ok"])
        self.assertFalse(payload["ticket_execution"]["endpoint_build_ok"])
        self.assertIn(
            "requires TICKET_EXECUTION_ARTIFACT_DIR",
            payload["ticket_execution"]["validation_error"],
        )

    def test_ticket_execution_status_main_returns_nonzero_for_invalid_runtime_env(self) -> None:
        original_token = config.DISCORD_BOT_TOKEN
        original_category = config.YEARN_TICKET_CATEGORY_ID
        try:
            config.DISCORD_BOT_TOKEN = None
            config.YEARN_TICKET_CATEGORY_ID = None
            captured = io.StringIO()
            with patch(
                "ticket_execution_status.get_repo_context_status",
                return_value={"state": "disabled", "fresh": False},
            ):
                with redirect_stdout(captured):
                    exit_code = ticket_execution_status_main()
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

    def test_build_ticket_execution_status_reports_command_probe_for_codex(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_CODEX_COMMAND = [sys.executable, "-c", "print('ok')"]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            with patch(
                "ticket_execution_status.get_repo_context_status",
                return_value={"state": "ready", "fresh": True},
            ):
                status = build_ticket_execution_status()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertTrue(status["ticket_execution"]["validation_ok"])
        self.assertTrue(status["ticket_execution"]["endpoint_build_ok"])
        self.assertEqual(
            status["ticket_execution"]["endpoint_class"],
            "FailoverTicketExecutionJsonEndpoint",
        )
        primary_probe = status["ticket_execution"]["primary_command_probe"]
        self.assertIsNotNone(primary_probe)
        assert primary_probe is not None
        self.assertTrue(primary_probe["available"])
        self.assertEqual(primary_probe["command"][:2], [sys.executable, "-c"])

    def test_build_ticket_execution_status_resolves_relative_command_against_configured_cwd(self) -> None:
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
                    "ticket_execution_status.get_repo_context_status",
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


class _FakeExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Factory tests should not execute the delegate.")


class TicketExecutionEndpointFactoryTests(unittest.TestCase):
    def test_build_endpoint_returns_local_endpoint_by_default(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        try:
            config.TICKET_EXECUTION_ENDPOINT = "local"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command

        self.assertIsInstance(endpoint, ExecutorBackedTicketExecutionJsonEndpoint)

    def test_build_endpoint_returns_subprocess_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = []
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

        self.assertIsInstance(endpoint, SubprocessTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.command[1:], ["-m", "ticket_investigation_worker_cli"])

    def test_build_endpoint_allows_configured_subprocess_prefix(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

        self.assertIsInstance(endpoint, SubprocessTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.command[:2], ["codex", "exec"])

    def test_build_endpoint_rejects_unknown_mode(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_codex_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "invalid"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            config.TICKET_EXECUTION_CODEX_COMMAND = []
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = []
            with self.assertRaises(ValueError):
                build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_CODEX_COMMAND = original_codex_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

    def test_build_endpoint_returns_codex_exec_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_model = config.TICKET_EXECUTION_CODEX_MODEL
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_CODEX_MODEL = "gpt-5.4"
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_CODEX_MODEL = original_model
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIsInstance(endpoint, CodexExecTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.codex_command[:2], ["codex", "exec"])
        self.assertEqual(endpoint.model, "gpt-5.4")

    def test_build_endpoint_wraps_primary_with_fallback_when_configured(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_CODEX_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIsInstance(endpoint, FailoverTicketExecutionJsonEndpoint)
        self.assertIsInstance(endpoint.primary, CodexExecTicketExecutionJsonEndpoint)
        self.assertIsInstance(endpoint.fallback, ExecutorBackedTicketExecutionJsonEndpoint)


class TicketTransportTests(unittest.TestCase):
    def test_transport_schemas_cover_required_top_level_fields(self) -> None:
        self.assertEqual(
            TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA["required"],
            [
                "aggregated_text",
                "input_list",
                "current_history",
                "run_context",
                "investigation_job",
                "workflow_name",
                "wants_bug_review_status",
            ],
        )
        self.assertEqual(
            TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA["required"],
            ["flow_outcome", "updated_job"],
        )

    def test_transport_request_round_trip_preserves_job_and_context(self) -> None:
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[{"role": "assistant", "content": "context"}],
            run_context=BotRunContext(
                channel_id=94,
                category_id=12,
                project_context="yearn",
                initial_button_intent="investigate_issue",
            ),
            investigation_job=TicketInvestigationJob(channel_id=94),
            workflow_name="tests.transport",
        )
        request.investigation_job.begin_collecting("investigate_issue")
        request.investigation_job.remember_chain("katana")
        request.investigation_job.remember_tx_hash(
            "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        )

        transport = TicketExecutionTransportRequest.from_turn_request(
            request,
            wants_bug_review_status=True,
        )
        hydrated = transport.to_turn_request()

        self.assertTrue(transport.wants_bug_review_status)
        self.assertEqual(hydrated.run_context.channel_id, 94)
        self.assertEqual(hydrated.run_context.initial_button_intent, "investigate_issue")
        self.assertEqual(hydrated.investigation_job.mode, "collecting")
        self.assertEqual(hydrated.investigation_job.evidence.chain, "katana")
        self.assertEqual(
            hydrated.investigation_job.evidence.tx_hashes,
            ["0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"],
        )

    def test_transport_request_json_round_trip_preserves_job_and_context(self) -> None:
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[{"role": "assistant", "content": "context"}],
            run_context=BotRunContext(
                channel_id=97,
                category_id=12,
                project_context="yearn",
                initial_button_intent="investigate_issue",
            ),
            investigation_job=TicketInvestigationJob(channel_id=97),
            workflow_name="tests.transport.json",
        )
        request.investigation_job.begin_collecting("investigate_issue")
        request.investigation_job.remember_chain("katana")
        transport = TicketExecutionTransportRequest.from_turn_request(
            request,
            wants_bug_review_status=True,
        )

        hydrated = TicketExecutionTransportRequest.from_json(
            transport.to_json()
        ).to_turn_request()

        self.assertEqual(hydrated.run_context.channel_id, 97)
        self.assertEqual(hydrated.investigation_job.mode, "collecting")
        self.assertEqual(hydrated.investigation_job.evidence.chain, "katana")
        self.assertIsNone(
            TicketExecutionTransportRequest.from_json(transport.to_json()).smoke_mode
        )

    def test_transport_request_json_round_trip_preserves_smoke_mode(self) -> None:
        transport = TicketExecutionTransportRequest(
            aggregated_text="smoke",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 99,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 99,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.transport.smoke",
            wants_bug_review_status=False,
            smoke_mode="ping",
        )

        hydrated = TicketExecutionTransportRequest.from_json(transport.to_json())

        self.assertEqual(hydrated.smoke_mode, "ping")

    def test_transport_result_round_trip_preserves_flow_and_job(self) -> None:
        job = TicketInvestigationJob(channel_id=95)
        job.begin_investigating()
        job.complete_specialist_turn("bug")
        result = TicketExecutionTransportResult.from_execution_parts(
            TicketAgentFlowOutcome(
                raw_final_reply="ok",
                conversation_history=[{"role": "assistant", "content": "ok"}],
                completed_agent_key="bug",
                requires_human_handoff=False,
            ),
            job,
        )

        flow_outcome, updated_job = result.to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "ok")
        self.assertEqual(flow_outcome.completed_agent_key, "bug")
        self.assertEqual(updated_job.mode, "investigating")
        self.assertEqual(updated_job.current_specialty, "bug")


class CodexBundleTests(unittest.TestCase):
    def test_build_codex_ticket_execution_bundle_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = build_codex_ticket_execution_bundle(
                request_json='{"example":"request"}',
                run_dir=temp_dir,
                repo_root="/root/bots/discord/ysupport",
            )

            self.assertEqual(
                bundle.command[: len(DEFAULT_CODEX_EXEC_COMMAND)],
                DEFAULT_CODEX_EXEC_COMMAND,
            )
            self.assertTrue(bundle.request_path.exists())
            self.assertTrue(bundle.response_schema_path.exists())
            self.assertTrue(bundle.prompt_path.exists())
            self.assertIn("request.json", bundle.prompt_text)
            self.assertIn("response_schema.json", bundle.prompt_text)
            self.assertIsNone(bundle.expected_response_path)
            self.assertIn("--output-schema", bundle.command)
            self.assertIn("--add-dir", bundle.command)

    def test_build_codex_ticket_execution_bundle_appends_model_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = build_codex_ticket_execution_bundle(
                request_json='{"example":"request"}',
                run_dir=temp_dir,
                repo_root="/root/bots/discord/ysupport",
                model="gpt-5.4",
            )

            self.assertIn("-m", bundle.command)
            self.assertIn("gpt-5.4", bundle.command)

    def test_build_codex_ticket_execution_bundle_writes_expected_response_for_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = build_codex_ticket_execution_bundle(
                request_json='{"example":"request"}',
                run_dir=temp_dir,
                repo_root="/root/bots/discord/ysupport",
                response_json_override=(
                    '{"flow_outcome":{"raw_final_reply":"ticket_execution_smoke_ok:codex_exec",'
                    '"conversation_history":[],"completed_agent_key":null,'
                    '"requires_human_handoff":false},"updated_job":{"channel_id":0,'
                    '"mode":"idle","evidence":{"tx_hashes":[]}}}'
                ),
            )

            self.assertIsNotNone(bundle.expected_response_path)
            assert bundle.expected_response_path is not None
            self.assertTrue(bundle.expected_response_path.exists())
            self.assertIn("expected_response.json", bundle.prompt_text)

    def test_transport_result_json_round_trip_preserves_flow_and_job(self) -> None:
        job = TicketInvestigationJob(channel_id=98)
        job.begin_investigating()
        job.complete_specialist_turn("data")
        transport_result = TicketExecutionTransportResult.from_execution_parts(
            TicketAgentFlowOutcome(
                raw_final_reply="answer",
                conversation_history=[{"role": "assistant", "content": "answer"}],
                completed_agent_key="data",
                requires_human_handoff=False,
            ),
            job,
        )

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            transport_result.to_json()
        ).to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "answer")
        self.assertEqual(flow_outcome.completed_agent_key, "data")
        self.assertEqual(updated_job.mode, "investigating")
        self.assertEqual(updated_job.current_specialty, "data")


class OnchainInspectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_tx_summary_profiles_contracts_and_formats_transfers(self) -> None:
        class _HexLike:
            def __init__(self, value: str) -> None:
                self._value = value

            def hex(self) -> str:
                return self._value

        class _FakeEth:
            def get_transaction_receipt(self, tx_hash: str) -> dict:
                return {
                    "status": 1,
                    "blockNumber": 27331428,
                    "gasUsed": 771090,
                    "logs": [
                        {
                            "address": "0x2222222222222222222222222222222222222222",
                            "logIndex": 0,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        }
                    ],
                }

            def get_transaction(self, tx_hash: str) -> dict:
                return {
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x3333333333333333333333333333333333333333",
                }

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        async def fake_inspect_contract_profile(web3_instance, contract_address: str, *, block_identifier=None):
            self.assertEqual(block_identifier, 27331428)
            return {
                "address": contract_address,
                "symbol": "yvWBUSDT",
                "name": "Yearn Katana Vault",
                "decimals": 18,
                "asset": "0x4444444444444444444444444444444444444444",
                "asset_symbol": "USDT",
                "asset_decimals": 6,
                "kind": "erc4626_like",
                "has_code": True,
            }

        with patch.dict(tools_lib.WEB3_INSTANCES, {"katana": _FakeWeb3()}, clear=True):
            with patch(
                "tools_lib._decode_logs_with_abis",
                return_value=[
                    {
                        "event": "Transfer",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 0,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "from": "0x1111111111111111111111111111111111111111",
                            "to": "0x5555555555555555555555555555555555555555",
                            "value": 650914700000000000000,
                        },
                    }
                ],
            ):
                with patch(
                    "tools_lib._inspect_contract_profile",
                    new=fake_inspect_contract_profile,
                ):
                    summary = await tools_lib.core_inspect_onchain(
                        chain="katana",
                        mode="tx_summary",
                        tx_hash="0x" + "a" * 64,
                    )

        self.assertIn("Transaction summary", summary)
        self.assertIn("contracts_profiled", summary)
        self.assertIn("yvWBUSDT", summary)
        self.assertIn("650.9147", summary)
        self.assertIn("erc4626_like", summary)

    async def test_tx_investigate_summarizes_transfer_and_approval_activity(self) -> None:
        class _HexLike:
            def __init__(self, value: str) -> None:
                self._value = value

            def hex(self) -> str:
                return self._value

        class _FakeEth:
            def get_transaction_receipt(self, tx_hash: str) -> dict:
                return {
                    "status": 1,
                    "blockNumber": 27331428,
                    "gasUsed": 771090,
                    "logs": [
                        {
                            "address": "0x2222222222222222222222222222222222222222",
                            "logIndex": 0,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        },
                        {
                            "address": "0x3333333333333333333333333333333333333333",
                            "logIndex": 1,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        },
                    ],
                }

            def get_transaction(self, tx_hash: str) -> dict:
                return {
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x4444444444444444444444444444444444444444",
                }

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        async def fake_inspect_contract_profile(web3_instance, contract_address: str, *, block_identifier=None):
            self.assertEqual(block_identifier, 27331428)
            if contract_address.lower() == "0x2222222222222222222222222222222222222222":
                return {
                    "address": contract_address,
                    "symbol": "frxUSD",
                    "name": "Frax USD",
                    "decimals": 18,
                    "asset": None,
                    "asset_symbol": None,
                    "asset_decimals": None,
                    "kind": "erc20_like",
                    "has_code": True,
                }
            return {
                "address": contract_address,
                "symbol": "yvWBUSDT",
                "name": "Yearn Katana Vault",
                "decimals": 18,
                "asset": "0x5555555555555555555555555555555555555555",
                "asset_symbol": "USDT",
                "asset_decimals": 6,
                "kind": "erc4626_like",
                "has_code": True,
            }

        with patch.dict(tools_lib.WEB3_INSTANCES, {"katana": _FakeWeb3()}, clear=True):
            with patch(
                "tools_lib._decode_logs_with_abis",
                return_value=[
                    {
                        "event": "Transfer",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 0,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "from": "0x1111111111111111111111111111111111111111",
                            "to": "0x3333333333333333333333333333333333333333",
                            "value": 1990379783000000000000,
                        },
                    },
                    {
                        "event": "Approval",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 1,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "owner": "0x1111111111111111111111111111111111111111",
                            "spender": "0x3333333333333333333333333333333333333333",
                            "value": 1990379783000000000000,
                        },
                    },
                ],
            ):
                with patch(
                    "tools_lib._inspect_contract_profile",
                    new=fake_inspect_contract_profile,
                ):
                    investigation = await tools_lib.core_inspect_onchain(
                        chain="katana",
                        mode="tx_investigate",
                        tx_hash="0x" + "a" * 64,
                    )

        self.assertIn("Transaction investigation", investigation)
        self.assertIn("user_transfers_out", investigation)
        self.assertIn("approvals", investigation)
        self.assertIn("Observed 1 transfer(s) out from the tx sender.", investigation)
        self.assertIn("Decoded 1 approval event(s).", investigation)


class SearchVaultsTests(unittest.IsolatedAsyncioTestCase):
    async def test_core_search_vaults_supports_all_query(self) -> None:
        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self) -> None:
                return None

            async def json(self):
                return [
                    {
                        "chainID": 1,
                        "address": "0x1111111111111111111111111111111111111111",
                        "name": "Yearn USDC",
                        "symbol": "yvUSDC",
                        "token": {
                            "name": "USD Coin",
                            "symbol": "USDC",
                            "address": "0x2222222222222222222222222222222222222222",
                            "decimals": 6,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.042,
                            "points": {"weekAgo": 0.04, "monthAgo": 0.041, "inception": 0.05},
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 1000000},
                        "info": {"riskLevel": 1, "isRetired": False, "isBoosted": False, "isHighlighted": True},
                        "migration": {"available": False},
                        "strategies": [],
                    },
                    {
                        "chainID": 42161,
                        "address": "0x3333333333333333333333333333333333333333",
                        "name": "Yearn DAI",
                        "symbol": "yvDAI",
                        "token": {
                            "name": "Dai Stablecoin",
                            "symbol": "DAI",
                            "address": "0x4444444444444444444444444444444444444444",
                            "decimals": 18,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.031,
                            "points": {"weekAgo": 0.029, "monthAgo": 0.03, "inception": 0.032},
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 500000},
                        "info": {"riskLevel": 2, "isRetired": False, "isBoosted": False, "isHighlighted": False},
                        "migration": {"available": False},
                        "strategies": [],
                    },
                ]

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("tools_lib.aiohttp.ClientSession", return_value=FakeSession()):
            result = await tools_lib.core_search_vaults("all")

        self.assertIn("Found 2 Yearn vault(s) matching 'all'.", result)
        self.assertIn("Vault: Yearn USDC (yvUSDC)", result)
        self.assertIn("Vault: Yearn DAI (yvDAI)", result)

    async def test_core_search_vaults_recommended_only_filters_single_strategy_ys_vaults(self) -> None:
        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self) -> None:
                return None

            async def json(self):
                return [
                    {
                        "chainID": 1,
                        "address": "0x1111111111111111111111111111111111111111",
                        "name": "Yearn Saver Strategy",
                        "symbol": "ysUSDC",
                        "kind": "Single Strategy",
                        "featuringScore": 0.10,
                        "token": {
                            "name": "USD Coin",
                            "symbol": "USDC",
                            "address": "0x2222222222222222222222222222222222222222",
                            "decimals": 6,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.089,
                            "points": {"weekAgo": 0.08, "monthAgo": 0.085, "inception": 0.09},
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 300000},
                        "info": {"riskLevel": 3, "isRetired": False, "isBoosted": False, "isHighlighted": False},
                        "migration": {"available": False},
                        "strategies": [{"name": "Only Strategy"}],
                    },
                    {
                        "chainID": 1,
                        "address": "0x3333333333333333333333333333333333333333",
                        "name": "Yearn USDC",
                        "symbol": "yvUSDC",
                        "kind": "Multi Strategy",
                        "featuringScore": 0.95,
                        "token": {
                            "name": "USD Coin",
                            "symbol": "USDC",
                            "address": "0x4444444444444444444444444444444444444444",
                            "decimals": 6,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.042,
                            "points": {"weekAgo": 0.04, "monthAgo": 0.041, "inception": 0.05},
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 1000000},
                        "info": {"riskLevel": 1, "isRetired": False, "isBoosted": False, "isHighlighted": True},
                        "migration": {"available": False},
                        "strategies": [{"name": "Strat One"}, {"name": "Strat Two"}],
                    },
                ]

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("tools_lib.aiohttp.ClientSession", return_value=FakeSession()):
            result = await tools_lib.core_search_vaults("all", recommended_only=True)

        self.assertIn("Vault: Yearn USDC (yvUSDC)", result)
        self.assertNotIn("Vault: Yearn Saver Strategy (ysUSDC)", result)
