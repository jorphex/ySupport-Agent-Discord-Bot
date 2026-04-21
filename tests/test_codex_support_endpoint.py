import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import config
from codex_support_home import prepare_codex_support_home
from codex_support_contract import (
    SupportTurnRequest,
    SupportTurnResult,
    verify_support_turn_result,
)
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
    _codex_support_prompt,
)
from ticket_investigation.executor import TicketExecutionHooks
from ticket_investigation.json_endpoint import build_ticket_execution_json_endpoint
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)


class _FakeExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Factory test should not execute the delegate.")


EXAMPLE_YSUPPORT_MCP_URL = "http://ysupport-mcp.example.test/mcp"


class CodexSupportEndpointTests(unittest.IsolatedAsyncioTestCase):
    def test_prepare_codex_support_home_writes_config_and_copies_auth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_source = Path(temp_dir) / "source-auth.json"
            auth_source.write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")

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
                with mock.patch(
                    "codex_support_home._is_http_url_reachable",
                    return_value=False,
                ):
                    home = prepare_codex_support_home(
                        codex_home=Path(temp_dir) / "bot-home",
                        repo_root=Path(__file__).resolve().parents[1],
                        auth_source=auth_source,
                        ysupport_mcp_url="http://127.0.0.1:8000/mcp",
                        mcp_server_api_key="secret-key",
                        web_search_mode="live",
                    )

            self.assertTrue(home.config_path.exists())
            self.assertTrue(home.auth_path.exists())
            self.assertTrue(home.instructions_path.exists())
            self.assertTrue(home.ysupport_mcp_enabled)
            config_text = home.config_path.read_text(encoding="utf-8")
            self.assertIn('sandbox_mode = "danger-full-access"', config_text)
            self.assertIn('web_search = "live"', config_text)
            self.assertIn("model_instructions_file =", config_text)
            self.assertIn("[mcp_servers.ysupport]", config_text)
            self.assertIn('command = "python3"', config_text)
            self.assertIn('args = ["mcp_server.py"]', config_text)
            self.assertIn('[mcp_servers.ysupport.env]', config_text)
            self.assertNotIn("openai_docs", config_text)
            self.assertIn(
                "You are ySupport.",
                home.instructions_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                home.auth_path.read_text(encoding="utf-8"),
                '{"auth_mode":"chatgpt"}',
            )
            self.assertNotIn("\r", config_text)

    def test_read_codex_mcp_url_from_home_reads_ysupport_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = Path(temp_dir)
            (codex_home / "config.toml").write_text(
                '\n'.join(
                    [
                        '[mcp_servers.ysupport]',
                        'enabled = true',
                        f'url = "{EXAMPLE_YSUPPORT_MCP_URL}"',
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                config._read_codex_mcp_url_from_home(codex_home),
                EXAMPLE_YSUPPORT_MCP_URL,
            )

    def test_prepare_codex_support_home_disables_unreachable_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "codex_support_home._choose_ysupport_stdio_launcher",
                return_value=None,
            ):
                with mock.patch(
                    "codex_support_home._is_http_url_reachable",
                    return_value=False,
                ):
                    home = prepare_codex_support_home(
                        codex_home=Path(temp_dir) / "bot-home",
                        ysupport_mcp_url="http://127.0.0.1:8000/mcp",
                        mcp_server_api_key="secret-key",
                        web_search_mode="live",
                    )
            self.assertFalse(home.ysupport_mcp_enabled)
            config_text = home.config_path.read_text(encoding="utf-8")
            self.assertNotIn("[mcp_servers.ysupport]", config_text)

    def test_prepare_codex_support_home_prefers_reachable_http_mcp_over_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "codex_support_home._is_http_url_reachable",
                return_value=True,
            ):
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
                ) as choose_launcher:
                    home = prepare_codex_support_home(
                        codex_home=Path(temp_dir) / "bot-home",
                        ysupport_mcp_url=EXAMPLE_YSUPPORT_MCP_URL,
                        mcp_server_api_key="secret-key",
                        web_search_mode="live",
                    )
            choose_launcher.assert_not_called()
            config_text = home.config_path.read_text(encoding="utf-8")
            self.assertIn(f'url = "{EXAMPLE_YSUPPORT_MCP_URL}"', config_text)
            self.assertNotIn('command = "python3"', config_text)

    def test_support_turn_request_uses_recent_transcript_slice(self) -> None:
        current_history = [
            {"role": "user", "content": f"m{i}"}
            for i in range(15)
        ]
        request = TicketExecutionTransportRequest(
            aggregated_text="latest question",
            input_list=[],
            current_history=current_history,
            run_context={
                    "channel_id": 90,
                "is_public_trigger": True,
                "project_context": "yearn",
                "initial_button_intent": "docs_qa",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 90,
                "requested_intent": "investigate_issue",
                "mode": "idle",
                "evidence": {"tx_hashes": ["0xabc"]},
            },
            workflow_name="tests.support_request",
            wants_bug_review_status=False,
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        self.assertEqual(support_request.current_user_message, "latest question")
        self.assertEqual(support_request.channel_type, "public")
        self.assertEqual(support_request.channel_id, 90)
        self.assertEqual(support_request.initial_button_intent, "docs_qa")
        self.assertEqual(support_request.requested_intent, "investigate_issue")
        self.assertEqual(support_request.support_state["investigation_mode"], "idle")
        self.assertFalse(support_request.support_state["human_handoff_active"])
        self.assertEqual(
            support_request.support_state["known_targets"]["tx_hashes"],
            ["0xabc"],
        )
        self.assertEqual(
            support_request.support_state["repo_context"]["last_search_artifact_refs"],
            [],
        )
        self.assertEqual(
            support_request.support_state["workflow_context"]["guardrail_profile"],
            "public_support",
        )
        self.assertEqual(
            support_request.support_state["workflow_context"]["expected_first_actions"],
            ["Answer directly in-channel and keep public-channel replies concise."],
        )
        self.assertEqual(len(support_request.recent_transcript), 12)
        self.assertEqual(
            [item["content"] for item in support_request.recent_transcript],
            [f"m{i}" for i in range(3, 15)],
        )
        self.assertEqual(
            support_request.constraints["allowed_tools"],
            ["shell", "web_search", "ysupport_mcp"],
        )
        support_request_without_mcp = SupportTurnRequest.from_ticket_execution_request(
            request,
            ysupport_mcp_enabled=False,
        )
        self.assertEqual(
            support_request_without_mcp.constraints["allowed_tools"],
            ["shell", "web_search"],
        )

    def test_support_turn_request_includes_deposit_withdrawal_workflow_context(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="0xB8B9E3097c8b1DDdF9C5ea9d48A7eBeaF09D67d2",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 91,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "data_deposits_withdrawals_start",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 91,
                "requested_intent": "data_deposits_withdrawals_start",
                "mode": "waiting_for_user",
                "evidence": {"wallet": "0xB8B9E3097c8b1DDdF9C5ea9d48A7eBeaF09D67d2"},
            },
            workflow_name="tests.deposit_flow",
            wants_bug_review_status=False,
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        workflow_context = support_request.support_state["workflow_context"]
        self.assertEqual(
            workflow_context["guardrail_profile"],
            "ticket_deposits_withdrawals",
        )
        self.assertTrue(workflow_context["button_context_known"])
        self.assertIn(
            "If the user provides a wallet address, start with wallet position lookup before asking for more detail.",
            workflow_context["expected_first_actions"],
        )
        self.assertEqual(
            workflow_context["non_support_boundaries"],
            [
                "listing",
                "partnership",
                "marketing",
                "vendor_security",
                "job_inquiry",
            ],
        )

    def test_verify_support_turn_result_rejects_discord_redirects(self) -> None:
        request = SupportTurnRequest(
            current_user_message="help",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["shell", "ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Please open a Discord ticket and join discord.gg/example",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the available support facts.",
            used_tools=["shell"],
        )
        with self.assertRaises(ValueError):
            verify_support_turn_result(result, request)

    def test_verify_support_turn_result_rejects_unallowed_tools(self) -> None:
        request = SupportTurnRequest(
            current_user_message="help",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Here is the answer.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the docs.",
            used_tools=["shell", "ysupport_mcp"],
        )
        with self.assertRaises(ValueError):
            verify_support_turn_result(result, request)

    def test_verify_support_turn_result_requires_handoff_reason(self) -> None:
        request = SupportTurnRequest(
            current_user_message="help",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="A human should review this.",
            requires_human_handoff=True,
            handoff_reason=None,
            evidence_summary="Checked the docs.",
            used_tools=["ysupport_mcp"],
        )
        with self.assertRaises(ValueError):
            verify_support_turn_result(result, request)

    def test_verify_support_turn_result_normalizes_and_passes(self) -> None:
        request = SupportTurnRequest(
            current_user_message="Can a human review this too?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["shell", "ysupport_mcp"]},
        )
        raw_result = SupportTurnResult.from_json(
            json.dumps(
                {
                    "answer": "  Here is the answer.  ",
                    "requires_human_handoff": True,
                    "handoff_reason": "  needs strategist confirmation  ",
                    "evidence_summary": "  Checked the docs and repo. ",
                    "used_tools": [
                        "shell",
                        "ysupport_mcp.search_vaults",
                        "shell",
                        " ",
                    ],
                }
            )
        )
        verified = verify_support_turn_result(raw_result, request)
        self.assertEqual(verified.answer, "Here is the answer.")
        self.assertEqual(verified.handoff_reason, "needs strategist confirmation")
        self.assertEqual(verified.evidence_summary, "Checked the docs and repo.")
        self.assertEqual(verified.used_tools, ["shell", "ysupport_mcp.search_vaults"])

    def test_verify_support_turn_result_downgrades_optional_handoff_offer(self) -> None:
        request = SupportTurnRequest(
            current_user_message="vault hasn't harvested after 10 days",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "Confirmed: the vault has not reported since April 8. "
                "The dashboard looks fresh, so this does not look like stale UI data. "
                "I can hand this off for strategist review to check why keeper activity paused."
            ),
            requires_human_handoff=True,
            handoff_reason=(
                "Public evidence confirms the missing harvests, but the specific reason "
                "for no keeper calls needs human strategist review."
            ),
            evidence_summary="Checked vault harvest history.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertNotIn("hand this off", verified.answer.lower())
        self.assertIn("dashboard looks fresh", verified.answer.lower())

    def test_verify_support_turn_result_strips_human_ops_review_hint(self) -> None:
        request = SupportTurnRequest(
            current_user_message="vault hasn't harvested after 10 days",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "Confirmed: the vault has not reported since April 8. "
                "This looks like real report inactivity, not just stale frontend data, "
                "so this should get a human ops review."
            ),
            requires_human_handoff=True,
            handoff_reason=(
                "The inactivity is confirmed, but determining why the vault has not "
                "been reported and whether intervention is needed requires human operator review."
            ),
            evidence_summary="Checked vault harvest history.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertNotIn("human ops review", verified.answer.lower())
        self.assertNotIn("should get a human", verified.answer.lower())
        self.assertIn("real report inactivity", verified.answer.lower())

    def test_codex_support_prompt_requests_fuller_prose_for_investigations(self) -> None:
        prompt_text = _codex_support_prompt(
            support_request_path=Path("support_request.json"),
            response_schema_path=Path("support_response_schema.json"),
        )

        self.assertIn("Routine support: concise.", prompt_text)
        self.assertIn("Investigations and report triage: enough prose", prompt_text)
        self.assertIn("Do not mention handoff if public evidence already answers the main question.", prompt_text)

    def test_codex_support_runtime_validation_requires_dedicated_home(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_home = config.TICKET_EXECUTION_CODEX_HOME
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_run_dir_root = config.TICKET_EXECUTION_RUN_DIR_ROOT
        original_api_key = config.MCP_SERVER_API_KEY
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_HOME = None
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            config.TICKET_EXECUTION_RUN_DIR_ROOT = ""
            config.MCP_SERVER_API_KEY = "secret-key"
            with self.assertRaises(ValueError):
                config.validate_ticket_execution_runtime_config()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_HOME = original_home
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
            config.TICKET_EXECUTION_RUN_DIR_ROOT = original_run_dir_root
            config.MCP_SERVER_API_KEY = original_api_key

    def test_endpoint_factory_builds_codex_support_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_home = config.TICKET_EXECUTION_CODEX_HOME
        original_api_key = config.MCP_SERVER_API_KEY
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            config.TICKET_EXECUTION_CODEX_HOME = "/tmp/ysupport-codex-home"
            config.MCP_SERVER_API_KEY = "secret-key"
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
            config.TICKET_EXECUTION_CODEX_HOME = original_home
            config.MCP_SERVER_API_KEY = original_api_key

        self.assertIsInstance(endpoint, CodexSupportTicketExecutionJsonEndpoint)

    async def test_codex_support_json_endpoint_round_trips_response_and_writes_bundle(self) -> None:
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
            "'gpt-5.4' in sys.argv,"
            "'model_reasoning_effort=\"medium\"' in ' '.join(sys.argv),"
            "'--json' in sys.argv,"
            "'--output-schema' in sys.argv,"
            "(cwd/'support_response_schema.json').exists(),"
            "'Read the support turn request from support_request.json.' in prompt"
            "),"
            "'requires_human_handoff':False,"
            "'handoff_reason':None,"
            "'evidence_summary':'checked',"
            "'used_tools':['shell','ysupport_mcp']"
            "}; "
            "sys.stdout.write(json.dumps({'type':'thread.started','thread_id':'t1'}) + '\\n'); "
            "sys.stdout.write(json.dumps({'type':'item.started','item':{'id':'item_1','type':'mcp_tool_call','tool_name':'support_dashboard_harvests'}}) + '\\n'); "
            "sys.stdout.write(json.dumps({'type':'item.completed','item':{'id':'item_2','type':'agent_message','text':json.dumps(response)}}))"
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            with tempfile.TemporaryDirectory() as codex_home_dir:
                auth_source = Path(codex_home_dir) / "source-auth.json"
                auth_source.write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")
                bot_home = Path(codex_home_dir) / "bot-home"
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
                        model="gpt-5.4",
                        reasoning_effort="medium",
                        repo_root=Path(__file__).resolve().parents[1],
                        codex_home=bot_home,
                        codex_auth_source=auth_source,
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
                            "evidence": {"wallet": None, "chain": "base", "tx_hashes": []},
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
                    self.assertTrue(os.path.exists(os.path.join(run_dir, "support_request.json")))
                    self.assertTrue(os.path.exists(os.path.join(run_dir, "support_response_schema.json")))
                    self.assertTrue(os.path.exists(os.path.join(run_dir, "codex_support_prompt.txt")))
                    self.assertTrue(os.path.exists(os.path.join(run_dir, "stdout.txt")))
                    self.assertTrue(os.path.exists(os.path.join(run_dir, "stderr.txt")))
                    support_request_payload = json.loads(
                        Path(run_dir, "support_request.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(support_request_payload["channel_id"], 109)
                    self.assertEqual(
                        support_request_payload["initial_button_intent"],
                        "investigate_issue",
                    )
                    self.assertEqual(
                        support_request_payload["support_state"]["known_targets"]["chain"],
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

    async def test_codex_support_json_endpoint_uses_codex_support_smoke_reply(self) -> None:
        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=[sys.executable, "-c", "print('should-not-run')"],
            allowed_command_prefixes=[[sys.executable, "-c", "print('should-not-run')"]],
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
            wants_bug_review_status=False,
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
                return response
            second_started.set()
            return response

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
            wants_bug_review_status=False,
        )

        with mock.patch(
            "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
            side_effect=fake_run_streaming_subprocess,
        ):
            task_one = asyncio.create_task(endpoint.execute_json_turn(request.to_json()))
            await first_started.wait()
            task_two = asyncio.create_task(endpoint.execute_json_turn(request.to_json()))
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())
            release_first.set()
            await asyncio.gather(task_one, task_two)

        self.assertTrue(second_started.is_set())
