import tests as _test_environment  # noqa: F401

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import config
from codex_support_home import (
    prepare_codex_auth_link,
    prepare_codex_support_home,
    sync_codex_auth_state,
)
from codex_support_contract import (
    SignedTransactionSafetyViolation,
    SupportTurnRequest,
    SupportTurnResult,
    support_result_to_transport_result,
    verify_support_turn_result,
)
from codex_support_sessions import CodexSupportSessionManager
from ticket_investigation.codex_support_endpoint import (
    CodexSupportExecutionOutput,
    CodexSupportTicketExecutionJsonEndpoint,
    _download_attachment_image,
    _parse_codex_support_execution_output,
    _prepare_support_request_attachments,
    _read_attachment_image_body,
    _codex_support_prompt,
    _codex_support_transaction_safety_rewrite_prompt,
    _run_codex_support_json_subprocess,
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
_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION = "0xf8cb" + ("ab" * 203)
_SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION = "0x02f8" + ("ab" * 120)


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
    async def test_codex_session_deletion_handles_unavailable_executable(self) -> None:
        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=["missing-codex-for-test", "exec"],
            allowed_command_prefixes=[["missing-codex-for-test", "exec"]],
        )

        with self.assertLogs(level="WARNING") as captured_logs:
            deleted = await endpoint._delete_codex_session(
                "019dade1-5acf-70e2-9c61-f5ba37862a78"
            )

        self.assertFalse(deleted)
        self.assertTrue(
            any(
                "Failed to delete Codex session" in line
                for line in captured_logs.output
            )
        )

    async def test_cancelled_codex_execution_kills_spawned_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            child_pid_path = Path(temp_dir) / "child_pid.txt"
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib, subprocess, sys, time; "
                    "child = subprocess.Popen("
                    "[sys.executable, '-c', 'import time; time.sleep(60)']); "
                    f"pathlib.Path({str(child_pid_path)!r}).write_text("
                    "str(child.pid), encoding='utf-8'); "
                    "time.sleep(60)"
                ),
            ]
            task = asyncio.create_task(
                _run_codex_support_json_subprocess(
                    command=command,
                    stdin_text="",
                    cwd=None,
                    env=dict(os.environ),
                    timeout_seconds=60,
                    max_output_chars=1000,
                    max_error_chars=1000,
                    timeout_message="timed out",
                    empty_stdout_message="empty",
                    oversized_stdout_message="oversized",
                    metadata={},
                    artifact_run_dir=None,
                    progress_callback=None,
                )
            )

            deadline = asyncio.get_running_loop().time() + 5
            while (
                not child_pid_path.exists()
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.05)
            self.assertTrue(child_pid_path.exists())
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            while asyncio.get_running_loop().time() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.05)
            else:
                self.fail("Cancelled Codex execution left a child process running.")

    async def test_image_attachment_stream_enforces_size_without_content_length(
        self,
    ) -> None:
        class _FakeContent:
            async def iter_chunked(self, _chunk_size: int):
                yield b"123"
                yield b"45"

        response = SimpleNamespace(content_length=None, content=_FakeContent())
        with mock.patch(
            "ticket_investigation.codex_support_endpoint._MAX_ATTACHMENT_IMAGE_BYTES",
            4,
        ):
            with self.assertRaisesRegex(ValueError, "20 MiB"):
                await _read_attachment_image_body(response)

    async def test_image_attachment_download_rejects_non_discord_and_oversized_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            attachments_dir = Path(temp_dir) / "attachments"
            attachments_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "Discord CDN"):
                await _download_attachment_image(
                    attachment={
                        "filename": "image.png",
                        "url": "https://example.com/image.png",
                        "content_type": "image/png",
                    },
                    attachments_dir=attachments_dir,
                    index=1,
                )
            with self.assertRaisesRegex(ValueError, "20 MiB"):
                await _download_attachment_image(
                    attachment={
                        "filename": "image.png",
                        "url": "https://cdn.discordapp.com/attachments/1/2/image.png",
                        "content_type": "image/png",
                        "size": 20 * 1024 * 1024 + 1,
                    },
                    attachments_dir=attachments_dir,
                    index=1,
                )

    async def test_image_attachment_preparation_fails_closed_on_download_error(
        self,
    ) -> None:
        support_request = SupportTurnRequest(
            current_user_message="What does this screenshot show?",
            recent_transcript=[],
            attachments=[
                {
                    "filename": "evidence.png",
                    "url": "https://cdn.discordapp.com/attachments/1/2/evidence.png",
                    "content_type": "image/png",
                    "is_image": True,
                }
            ],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.image_failure",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={},
            constraints={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "ticket_investigation.codex_support_endpoint._download_attachment_image",
                side_effect=RuntimeError("expired URL"),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Could not prepare image attachment evidence.png: expired URL",
                ):
                    await _prepare_support_request_attachments(
                        support_request,
                        run_dir=temp_dir,
                    )

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
            self.assertIn("view_image = false", config_text)
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

    def test_prepare_codex_support_home_syncs_auth_source_from_canonical_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_source = Path(temp_dir) / "auth-source.json"
            auth_source.write_text('{"auth_mode":"stale"}', encoding="utf-8")
            canonical_source = Path(temp_dir) / "canonical-auth.json"
            canonical_source.write_text('{"auth_mode":"fresh"}', encoding="utf-8")

            with mock.patch(
                "codex_support_home._choose_ysupport_stdio_launcher",
                return_value=None,
            ), mock.patch(
                "codex_support_home._is_http_url_reachable",
                return_value=False,
            ):
                home = prepare_codex_support_home(
                    codex_home=Path(temp_dir) / "bot-home",
                    auth_source=auth_source,
                    auth_sync_source=canonical_source,
                    ysupport_mcp_url="http://127.0.0.1:8000/mcp",
                    mcp_server_api_key="secret-key",
                    web_search_mode="live",
                )

            self.assertEqual(
                auth_source.read_text(encoding="utf-8"),
                '{"auth_mode":"fresh"}',
            )
            self.assertEqual(
                home.auth_path.read_text(encoding="utf-8"),
                '{"auth_mode":"fresh"}',
            )

    def test_prepare_codex_auth_link_symlinks_bot_auth_to_live_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_auth = Path(temp_dir) / "bot-home" / "auth.json"
            live_auth = Path(temp_dir) / "live-home" / "auth.json"
            live_auth.parent.mkdir(parents=True, exist_ok=True)
            live_auth.write_text('{"auth_mode":"live"}', encoding="utf-8")

            linked = prepare_codex_auth_link(
                home_auth_path=home_auth,
                auth_link_source_path=live_auth,
            )

            self.assertEqual(linked, live_auth)
            self.assertTrue(home_auth.is_symlink())
            self.assertEqual(
                home_auth.resolve(strict=False),
                live_auth.resolve(strict=False),
            )
            self.assertEqual(
                home_auth.read_text(encoding="utf-8"),
                '{"auth_mode":"live"}',
            )

    def test_prepare_codex_support_home_can_link_auth_instead_of_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            live_auth = Path(temp_dir) / "live-home" / "auth.json"
            live_auth.parent.mkdir(parents=True, exist_ok=True)
            live_auth.write_text('{"auth_mode":"live"}', encoding="utf-8")

            with mock.patch(
                "codex_support_home._choose_ysupport_stdio_launcher",
                return_value=None,
            ), mock.patch(
                "codex_support_home._is_http_url_reachable",
                return_value=False,
            ):
                home = prepare_codex_support_home(
                    codex_home=Path(temp_dir) / "bot-home",
                    auth_link_source=live_auth,
                    ysupport_mcp_url="http://127.0.0.1:8000/mcp",
                    mcp_server_api_key="secret-key",
                    web_search_mode="live",
                )

            self.assertTrue(home.auth_path.is_symlink())
            self.assertEqual(
                home.auth_path.resolve(strict=False),
                live_auth.resolve(strict=False),
            )

    def test_sync_codex_auth_state_uses_freshest_existing_file_for_all_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_auth = Path(temp_dir) / "bot-home" / "auth.json"
            source_auth = Path(temp_dir) / "auth-source.json"
            canonical_auth = Path(temp_dir) / "canonical-auth.json"
            source_auth.write_text('{"auth_mode":"source-old"}', encoding="utf-8")
            canonical_auth.write_text('{"auth_mode":"canonical-new"}', encoding="utf-8")
            os.utime(source_auth, (1, 1))
            os.utime(canonical_auth, None)

            freshest = sync_codex_auth_state(
                home_auth_path=home_auth,
                auth_source_path=source_auth,
                auth_sync_source_path=canonical_auth,
            )

            self.assertEqual(freshest, canonical_auth)
            self.assertEqual(
                home_auth.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-new"}',
            )
            self.assertEqual(
                source_auth.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-new"}',
            )
            self.assertEqual(
                canonical_auth.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-new"}',
            )

    def test_sync_codex_auth_state_prefers_canonical_source_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_auth = Path(temp_dir) / "bot-home" / "auth.json"
            source_auth = Path(temp_dir) / "auth-source.json"
            canonical_auth = Path(temp_dir) / "canonical-auth.json"
            home_auth.parent.mkdir(parents=True, exist_ok=True)
            home_auth.write_text('{"auth_mode":"bot-stale"}', encoding="utf-8")
            source_auth.write_text('{"auth_mode":"source-stale"}', encoding="utf-8")
            canonical_auth.write_text('{"auth_mode":"canonical-good"}', encoding="utf-8")
            os.utime(home_auth, (5, 5))
            os.utime(source_auth, (4, 4))
            os.utime(canonical_auth, (1, 1))

            chosen = sync_codex_auth_state(
                home_auth_path=home_auth,
                auth_source_path=source_auth,
                auth_sync_source_path=canonical_auth,
                preferred_source_path=canonical_auth,
            )

            self.assertEqual(chosen, canonical_auth)
            self.assertEqual(
                home_auth.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-good"}',
            )
            self.assertEqual(
                source_auth.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-good"}',
            )

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

    def test_support_turn_request_preserves_internal_turn_context(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="thanks. we already have this queued pending sigs",
            input_list=[],
            current_history=[{"role": "user", "content": "please dump rewards"}],
            turn_source="internal_team",
            turn_instruction=(
                "This input is from the internal team, not from the user. "
                "Write the next Discord update for the user."
            ),
            run_context={
                "channel_id": 91,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 91,
                "requested_intent": "investigate_issue",
                "mode": "escalated_to_human",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.internal_team_request",
            wants_bug_review_status=False,
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        self.assertEqual(support_request.current_turn_source, "internal_team")
        self.assertIn("internal team", support_request.current_turn_instruction or "")
        self.assertEqual(
            support_request.current_user_message,
            "thanks. we already have this queued pending sigs",
        )

    def test_internal_team_result_does_not_append_team_reply_as_user_history(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="thanks. we already have this queued pending sigs",
            input_list=[],
            current_history=[{"role": "user", "content": "please dump rewards"}],
            turn_source="internal_team",
            turn_instruction="Write the next Discord update for the user.",
            run_context={
                "channel_id": 92,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 92,
                "requested_intent": "investigate_issue",
                "mode": "escalated_to_human",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.internal_team_history",
            wants_bug_review_status=False,
        )
        result = SupportTurnResult(
            answer="The swap has already been queued and is pending signatures.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="team update",
            used_tools=[],
        )

        transport_result = support_result_to_transport_result(result, request)
        conversation_history = transport_result.flow_outcome["conversation_history"]

        self.assertEqual(
            conversation_history,
            [
                {"role": "user", "content": "please dump rewards"},
                {
                    "role": "assistant",
                    "content": "The swap has already been queued and is pending signatures.",
                },
            ],
        )

    def test_support_turn_request_includes_deposit_withdrawal_workflow_context(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="0xB8B9E3097c8b1DDdF9C5ea9d48A7eBeaF09D67d2",
            input_list=[],
            current_history=[],
            attachments=[],
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

    def test_support_turn_request_preserves_image_attachments(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="Why do these numbers differ?",
            input_list=[],
            current_history=[],
            attachments=[
                {
                    "filename": "image.png",
                    "url": "https://cdn.example.test/image.png",
                    "content_type": "image/png",
                    "size": 1234,
                    "is_image": True,
                }
            ],
            run_context={
                "channel_id": 92,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 92,
                "requested_intent": "investigate_issue",
                "mode": "collecting",
                "evidence": {},
            },
            workflow_name="tests.image_support",
            wants_bug_review_status=False,
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        self.assertEqual(len(support_request.attachments), 1)
        self.assertEqual(
            support_request.constraints["allowed_tools"],
            ["shell", "web_search", "ysupport_mcp"],
        )

    def test_verify_support_turn_result_allows_view_image_for_image_backed_request(self) -> None:
        request = SupportTurnRequest(
            current_user_message="What do these screenshots show?",
            recent_transcript=[],
            attachments=[
                {
                    "filename": "image.png",
                    "url": "https://cdn.example.test/image.png",
                    "content_type": "image/png",
                    "size": 1234,
                    "is_image": True,
                }
            ],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="The screenshot shows a Yearn vault APY breakdown.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the screenshots and Yearn support data.",
            used_tools=["view_image", "ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertEqual(verified.used_tools, ["view_image", "ysupport_mcp"])

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

    def test_verify_support_turn_result_rejects_transaction_sized_hex_payloads(
        self,
    ) -> None:
        request = _transaction_safety_support_request()
        payloads = {
            "historical_legacy_shape": _SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION,
            "current_typed_shape": _SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION,
        }
        for payload_name, payload in payloads.items():
            for field_name in ("answer", "evidence_summary", "handoff_reason"):
                with self.subTest(payload=payload_name, field_name=field_name):
                    result = SupportTurnResult(
                        answer="Use only the transaction hash.",
                        requires_human_handoff=False,
                        handoff_reason=None,
                        evidence_summary="Checked the pending transaction.",
                        used_tools=["shell"],
                    )
                    setattr(
                        result,
                        field_name,
                        "Paste this signed transaction into a public broadcaster: "
                        f"`{payload}`",
                    )

                    with self.assertRaises(SignedTransactionSafetyViolation):
                        verify_support_turn_result(result, request)

    def test_verify_support_turn_result_allows_transaction_hashes_and_addresses(
        self,
    ) -> None:
        request = _transaction_safety_support_request()
        tx_hash = "0x" + ("12" * 32)
        address = "0x" + ("34" * 20)
        long_calldata = "0xdead" + ("56" * 120)
        result = SupportTurnResult(
            answer=(
                f"Transaction {tx_hash} from {address} is still pending. "
                f"Decoded call data: {long_calldata}."
            ),
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the transaction status by hash.",
            used_tools=["shell"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertEqual(verified.answer, result.answer)

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
                    "handoff_reason": "  needs private internal strategist confirmation  ",
                    "handoff_kind": "private_internal_fact",
                    "evidence_summary": "  Checked the docs and repo. ",
                    "used_tools": [
                        "shell",
                        "ysupport_mcp.search_vaults",
                        "functions.mcp__ysupport__search_documentation",
                        "shell",
                        " ",
                    ],
                }
            )
        )
        verified = verify_support_turn_result(raw_result, request)
        self.assertEqual(verified.answer, "Here is the answer.")
        self.assertEqual(
            verified.handoff_reason,
            "needs private internal strategist confirmation",
        )
        self.assertEqual(verified.handoff_kind, "private_internal_fact")
        self.assertEqual(verified.evidence_summary, "Checked the docs and repo.")
        self.assertEqual(
            verified.used_tools,
            ["shell", "ysupport_mcp.search_vaults", "mcp__ysupport__search_documentation"],
        )

    def test_verify_support_turn_result_downgrades_generic_human_request(self) -> None:
        request = SupportTurnRequest(
            current_user_message="I want a human to look at this.",
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
                "The transaction is still pending and has not reverted. "
                "A human can review it too."
            ),
            requires_human_handoff=True,
            handoff_reason="The user asked for human review.",
            evidence_summary="Checked the transaction status.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertEqual(
            verified.answer,
            "The transaction is still pending and has not reverted.",
        )

    def test_verify_support_turn_result_downgrades_generic_moderator_request(self) -> None:
        request = SupportTurnRequest(
            current_user_message="I need a moderator to review this.",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "Please finish the documented verification flow first. "
                "A moderator can review this afterward."
            ),
            requires_human_handoff=True,
            handoff_reason="The user asked for moderator review.",
            evidence_summary="Checked the documented verification process.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertNotIn("moderator", verified.answer.lower())

    def test_verify_support_turn_result_allows_concrete_moderator_access_action(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message=(
                "I completed verification and restarted Discord, "
                "but I still cannot see the general channel."
            ),
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "The documented verification and client refresh steps are complete. "
                "A moderator must now inspect the account's channel access."
            ),
            requires_human_handoff=True,
            handoff_reason=(
                "A moderator access change is required after the documented "
                "verification steps were exhausted."
            ),
            handoff_kind="access_or_permission_action",
            evidence_summary="Checked the documented verification process.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertTrue(verified.requires_human_handoff)
        self.assertIn("moderator access", verified.handoff_reason or "")

    def test_verify_support_turn_result_accepts_dot_prefixed_ysupport_mcp_tools(self) -> None:
        request = SupportTurnRequest(
            current_user_message="Why is TVL not updating after my deposit?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="data_deposits_withdrawals_start",
            requested_intent="data_deposits_withdrawals_start",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult.from_json(
            json.dumps(
                {
                    "answer": "TVL updates can lag slightly after deposit.",
                    "requires_human_handoff": False,
                    "handoff_reason": None,
                    "evidence_summary": "Checked vault metadata and docs.",
                    "used_tools": [
                        "mcp__ysupport.search_vaults",
                        "mcp__ysupport.support_dashboard_discover",
                        "mcp__ysupport.support_dashboard_token_venues",
                        "mcp__ysupport.search_documentation",
                        "mcp__ysupport.search_repo_context",
                    ],
                }
            )
        )
        verified = verify_support_turn_result(result, request)
        self.assertEqual(
            verified.used_tools,
            [
                "mcp__ysupport.search_vaults",
                "mcp__ysupport.support_dashboard_discover",
                "mcp__ysupport.support_dashboard_token_venues",
                "mcp__ysupport.search_documentation",
                "mcp__ysupport.search_repo_context",
            ],
        )

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

    def test_verify_support_turn_result_does_not_override_model_handoff_decision(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message="pls dump dola rewards for strategy 0x1111111111111111111111111111111111111111",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="The rewards are still sitting on the strategy and have not been swapped yet.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked current strategy state.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)

    def test_verify_support_turn_result_clears_reason_without_handoff(self) -> None:
        request = SupportTurnRequest(
            current_user_message="How do I withdraw?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="data_deposits_withdrawals_start",
            requested_intent="data_deposits_withdrawals_start",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Use the Withdraw action on the vault position.",
            requires_human_handoff=False,
            handoff_reason="No human action is actually required.",
            handoff_kind=None,
            evidence_summary="Checked the documented withdrawal flow.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertIsNone(verified.handoff_kind)

    def test_verify_support_turn_result_allows_semantic_manual_strategy_handoff(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message=(
                "Can the Yearn team dump the accumulated DOLA rewards for strategy "
                "0x1111111111111111111111111111111111111111?"
            ),
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="The rewards are still on the strategy, so the team must perform the requested action.",
            requires_human_handoff=True,
            handoff_reason=(
                "A manual strategy action is required to sell the accumulated rewards."
            ),
            handoff_kind="manual_strategy_action",
            evidence_summary="Checked current strategy state.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertTrue(verified.requires_human_handoff)
        self.assertEqual(verified.handoff_reason, result.handoff_reason)

    def test_verify_support_turn_result_does_not_keyword_route_user_vault_sale(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message="How do I sell or withdraw my Yearn vault shares?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="data_deposits_withdrawals_start",
            requested_intent="data_deposits_withdrawals_start",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Use the Withdraw action on the vault position.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the documented withdrawal flow.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)

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
        request_path = Path("support_request.json")
        schema_path = Path("support_response_schema.json")
        prompt_text = _codex_support_prompt(
            support_request_path=request_path,
            response_schema_path=schema_path,
        )

        self.assertIn(str(request_path.resolve()), prompt_text)
        self.assertIn(str(schema_path.resolve()), prompt_text)
        self.assertIn("Routine support: concise.", prompt_text)
        self.assertIn("Investigations and report triage: enough prose", prompt_text)
        self.assertIn("Do not mention handoff if public evidence already answers the main question.", prompt_text)
        self.assertIn(
            "exhaust the relevant available documentation, live-data, repository, web, and image evidence",
            prompt_text,
        )
        self.assertIn(
            "does not by itself justify handoff",
            prompt_text,
        )
        self.assertIn(
            "Never describe a required human or team action while returning requires_human_handoff=false.",
            prompt_text,
        )
        self.assertIn(
            "access_or_permission_action, fund_or_account_recovery, security_process, manual_strategy_action, private_internal_fact, or human_decision",
            prompt_text,
        )
        self.assertIn(
            "do not claim that you have escalated, handed off, or notified anyone",
            prompt_text,
        )
        self.assertIn("Use `current_turn_source`", prompt_text)
        self.assertIn("If `current_turn_source` is `internal_team`", prompt_text)
        self.assertIn(
            "synthesize a concise direct answer from the Yearn documentation excerpts",
            prompt_text,
        )
        self.assertIn(
            "Do not expose retrieval metadata",
            prompt_text,
        )
        self.assertIn(
            "Never ask for, retrieve, retain, reconstruct, quote, display, submit, broadcast, or recommend manually broadcasting a raw signed transaction.",
            prompt_text,
        )
        self.assertIn(
            "Reaching this safety boundary does not by itself justify human handoff.",
            prompt_text,
        )
        self.assertNotIn(
            "documentation tool already returns a complete answer",
            prompt_text,
        )

    def test_codex_support_prompt_leads_ambiguous_bug_intake_with_security_path(
        self,
    ) -> None:
        prompt_text = _codex_support_prompt(
            support_request_path=Path("support_request.json"),
            response_schema_path=Path("support_response_schema.json"),
        )

        security_path_index = prompt_text.index(
            "begin the reply with https://github.com/yearn/yearn-security/blob/master/SECURITY.md"
        )
        product_intake_index = prompt_text.index(
            "Only after that, offer to accept ordinary product-bug details."
        )
        self.assertLess(security_path_index, product_intake_index)
        self.assertIn(
            "Do not stop or request human handoff solely because the user used generic bug-report wording.",
            prompt_text,
        )

    def test_codex_support_prompt_requires_complete_gas_sufficiency_evidence(
        self,
    ) -> None:
        prompt_text = _codex_support_prompt(
            support_request_path=Path("support_request.json"),
            response_schema_path=Path("support_response_schema.json"),
        )
        rewrite_prompt_text = _codex_support_transaction_safety_rewrite_prompt(
            response_schema_path=Path("support_response_schema.json"),
        )

        for rendered_prompt in (prompt_text, rewrite_prompt_text):
            self.assertIn(
                "gas limit multiplied by maximum fee per gas, or legacy gas price",
                rendered_prompt,
            )
            self.assertIn("include a conservative buffer", rendered_prompt)
            self.assertIn(
                "pending or wallet-queued transactions that may reserve the balance",
                rendered_prompt,
            )
            self.assertIn(
                "Never claim the wallet definitely has enough gas from its current balance alone.",
                rendered_prompt,
            )
            self.assertIn(
                "state that sufficiency is conditional and name the missing check",
                rendered_prompt,
            )

    def test_codex_support_runtime_validation_requires_dedicated_home(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_home = config.TICKET_EXECUTION_CODEX_HOME
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_api_key = config.MCP_SERVER_API_KEY
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_HOME = None
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            config.MCP_SERVER_API_KEY = "secret-key"
            with self.assertRaises(ValueError):
                config.validate_ticket_execution_runtime_config()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_HOME = original_home
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
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

            with mock.patch.object(
                endpoint,
                "_delete_codex_session",
                new=mock.AsyncMock(return_value=True),
            ) as mock_delete, mock.patch(
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
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
                return CodexSupportExecutionOutput(
                    final_response_text=unsafe_response
                )
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

            with mock.patch.object(
                endpoint,
                "_delete_codex_session",
                side_effect=fake_delete,
            ), mock.patch(
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
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

            with mock.patch.object(
                endpoint,
                "_delete_codex_session",
                new=mock.AsyncMock(return_value=True),
            ) as mock_delete, mock.patch(
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
                return_value=CodexSupportExecutionOutput(
                    final_response_text=unsafe_response
                ),
            ) as mock_run:
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

            with mock.patch.object(
                endpoint,
                "_delete_codex_session",
                new=mock.AsyncMock(return_value=False),
            ) as mock_delete, mock.patch(
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
            ), self.assertLogs(level="WARNING") as captured_logs:
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

    async def test_codex_support_endpoint_records_verification_failure_not_success(self) -> None:
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
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
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
                return CodexSupportExecutionOutput(final_response_text=response)
            second_started.set()
            return CodexSupportExecutionOutput(final_response_text=response)

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

    async def test_codex_support_endpoint_retries_once_on_auth_error(self) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()
        call_count = 0

        async def fake_run_streaming_subprocess(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("refresh_token_reused token_expired")
            return CodexSupportExecutionOutput(final_response_text=response)

        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=["codex", "exec"],
            allowed_command_prefixes=[["codex", "exec"]],
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="investigate support",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 111,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 111,
                "requested_intent": "docs_qa",
                "mode": "waiting_for_user",
                "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_exec",
            wants_bug_review_status=False,
        )

        with mock.patch.object(
            endpoint,
            "_prepare_support_home",
            return_value=False,
        ) as mock_prepare, mock.patch(
            "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
            side_effect=fake_run_streaming_subprocess,
        ):
            response_json = await endpoint.execute_json_turn(request.to_json())

        transport_result = TicketExecutionTransportResult.from_json(response_json)
        self.assertEqual(
            transport_result.flow_outcome["raw_final_reply"],
            "support-ok",
        )
        self.assertEqual(call_count, 2)
        self.assertEqual(mock_prepare.call_count, 2)

    async def test_codex_support_endpoint_syncs_bot_auth_back_to_sources_after_run(self) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()

        with tempfile.TemporaryDirectory() as temp_dir:
            auth_source = Path(temp_dir) / "auth-source.json"
            auth_source.write_text('{"auth_mode":"source-old"}', encoding="utf-8")
            canonical_source = Path(temp_dir) / "canonical-auth.json"
            canonical_source.write_text('{"auth_mode":"canonical-old"}', encoding="utf-8")
            os.utime(auth_source, (1, 1))
            os.utime(canonical_source, (1, 1))
            bot_home = Path(temp_dir) / "bot-home"
            bot_home.mkdir(parents=True, exist_ok=True)

            async def fake_run_streaming_subprocess(**kwargs):
                (bot_home / "auth.json").write_text(
                    '{"auth_mode":"bot-refreshed"}',
                    encoding="utf-8",
                )
                return CodexSupportExecutionOutput(final_response_text=response)

            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                codex_home=bot_home,
                codex_auth_source=auth_source,
                codex_auth_sync_source=canonical_source,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="investigate support",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 112,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 112,
                    "requested_intent": "docs_qa",
                    "mode": "waiting_for_user",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_support_exec",
                wants_bug_review_status=False,
            )

            with mock.patch(
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

            transport_result = TicketExecutionTransportResult.from_json(response_json)
            self.assertEqual(
                transport_result.flow_outcome["raw_final_reply"],
                "support-ok",
            )
            self.assertEqual(
                auth_source.read_text(encoding="utf-8"),
                '{"auth_mode":"bot-refreshed"}',
            )
            self.assertEqual(
                canonical_source.read_text(encoding="utf-8"),
                '{"auth_mode":"bot-refreshed"}',
            )

    async def test_codex_support_endpoint_syncs_auth_before_first_attempt(self) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()

        with tempfile.TemporaryDirectory() as temp_dir:
            auth_source = Path(temp_dir) / "auth-source.json"
            auth_source.write_text('{"auth_mode":"source-old"}', encoding="utf-8")
            canonical_source = Path(temp_dir) / "canonical-auth.json"
            canonical_source.write_text(
                '{"auth_mode":"canonical-fresh"}',
                encoding="utf-8",
            )
            os.utime(auth_source, (4, 4))
            os.utime(canonical_source, (1, 1))
            bot_home = Path(temp_dir) / "bot-home"
            bot_home.mkdir(parents=True, exist_ok=True)
            (bot_home / "auth.json").write_text(
                '{"auth_mode":"bot-stale"}',
                encoding="utf-8",
            )
            os.utime(bot_home / "auth.json", (5, 5))

            seen_home_auth = {}

            async def fake_run_streaming_subprocess(**kwargs):
                seen_home_auth["text"] = (bot_home / "auth.json").read_text(
                    encoding="utf-8"
                )
                return CodexSupportExecutionOutput(final_response_text=response)

            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                codex_home=bot_home,
                codex_auth_source=auth_source,
                codex_auth_sync_source=canonical_source,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="investigate support",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 113,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 113,
                    "requested_intent": "docs_qa",
                    "mode": "waiting_for_user",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_support_exec",
                wants_bug_review_status=False,
            )

            with mock.patch(
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

            transport_result = TicketExecutionTransportResult.from_json(response_json)
            self.assertEqual(
                transport_result.flow_outcome["raw_final_reply"],
                "support-ok",
            )
            self.assertEqual(
                seen_home_auth["text"],
                '{"auth_mode":"canonical-fresh"}',
            )
            self.assertEqual(
                auth_source.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-fresh"}',
            )
            self.assertEqual(
                canonical_source.read_text(encoding="utf-8"),
                '{"auth_mode":"canonical-fresh"}',
            )

    async def test_codex_support_endpoint_links_live_auth_before_first_attempt(self) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()

        with tempfile.TemporaryDirectory() as temp_dir:
            live_auth = Path(temp_dir) / "live-home" / "auth.json"
            live_auth.parent.mkdir(parents=True, exist_ok=True)
            live_auth.write_text('{"auth_mode":"live"}', encoding="utf-8")
            bot_home = Path(temp_dir) / "bot-home"
            bot_home.mkdir(parents=True, exist_ok=True)
            seen_home_auth = {}

            async def fake_run_streaming_subprocess(**kwargs):
                bot_auth_path = bot_home / "auth.json"
                seen_home_auth["is_symlink"] = bot_auth_path.is_symlink()
                seen_home_auth["resolved"] = bot_auth_path.resolve(strict=False)
                seen_home_auth["text"] = bot_auth_path.read_text(encoding="utf-8")
                return CodexSupportExecutionOutput(final_response_text=response)

            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=["codex", "exec"],
                codex_home=bot_home,
                codex_auth_link_source=live_auth,
                allowed_command_prefixes=[["codex", "exec"]],
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="investigate support",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 114,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 114,
                    "requested_intent": "docs_qa",
                    "mode": "waiting_for_user",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_support_exec",
                wants_bug_review_status=False,
            )

            with mock.patch(
                "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
                side_effect=fake_run_streaming_subprocess,
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

            transport_result = TicketExecutionTransportResult.from_json(response_json)
            self.assertEqual(
                transport_result.flow_outcome["raw_final_reply"],
                "support-ok",
            )
            self.assertTrue(seen_home_auth["is_symlink"])
            self.assertEqual(
                seen_home_auth["resolved"],
                live_auth.resolve(strict=False),
            )
            self.assertEqual(
                seen_home_auth["text"],
                '{"auth_mode":"live"}',
            )

    async def test_codex_support_endpoint_serializes_codex_runs_across_conversations(
        self,
    ) -> None:
        response = SupportTurnResult(
            answer="support-ok",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="checked",
            used_tools=["shell"],
        ).to_json()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        in_flight = 0
        max_in_flight = 0

        async def fake_run_streaming_subprocess(**kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if not first_entered.is_set():
                first_entered.set()
                await release_first.wait()
            else:
                second_entered.set()
            in_flight -= 1
            return CodexSupportExecutionOutput(final_response_text=response)

        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=["codex", "exec"],
            allowed_command_prefixes=[["codex", "exec"]],
        )
        request_one = TicketExecutionTransportRequest(
            aggregated_text="first",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 201,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 201,
                "requested_intent": "docs_qa",
                "mode": "waiting_for_user",
                "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_exec",
            wants_bug_review_status=False,
        )
        request_two = TicketExecutionTransportRequest(
            aggregated_text="second",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 202,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 202,
                "requested_intent": "docs_qa",
                "mode": "waiting_for_user",
                "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_support_exec",
            wants_bug_review_status=False,
        )

        with mock.patch.object(
            endpoint,
            "_prepare_support_home",
            return_value=False,
        ), mock.patch(
            "ticket_investigation.codex_support_endpoint._run_codex_support_json_subprocess",
            side_effect=fake_run_streaming_subprocess,
        ):
            first_task = asyncio.create_task(
                endpoint.execute_json_turn(request_one.to_json())
            )
            await first_entered.wait()
            second_task = asyncio.create_task(
                endpoint.execute_json_turn(request_two.to_json())
            )
            await asyncio.sleep(0.05)
            self.assertFalse(
                second_entered.is_set(),
                "second Codex run should not enter while the first run is active",
            )
            release_first.set()
            first_response, second_response = await asyncio.gather(
                first_task,
                second_task,
            )

        self.assertEqual(max_in_flight, 1)
        self.assertTrue(second_entered.is_set())
        self.assertEqual(
            TicketExecutionTransportResult.from_json(first_response).flow_outcome[
                "raw_final_reply"
            ],
            "support-ok",
        )
        self.assertEqual(
            TicketExecutionTransportResult.from_json(second_response).flow_outcome[
                "raw_final_reply"
            ],
            "support-ok",
        )

    def test_parse_codex_support_execution_output_uses_only_final_answer(self) -> None:
        stdout_text = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "mcp_tool_call",
                            "tool": "search_documentation",
                            "result": {
                                "structured_content": {
                                    "result": "Docs answer\n\n- bullet"
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": '{"answer":"final","evidence_summary":"x","handoff_reason":null,"requires_human_handoff":false,"used_tools":["ysupport_mcp.search_documentation"]}',
                        },
                    }
                ),
            ]
        )
        execution_output = _parse_codex_support_execution_output(stdout_text)
        self.assertIn('"answer":"final"', execution_output.final_response_text)
