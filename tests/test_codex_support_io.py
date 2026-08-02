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
    SupportTurnRequest,
)
from ticket_investigation.codex_support_attachments import (
    _download_attachment_image,
    _read_attachment_image_body,
    prepare_support_request_attachments,
)
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ticket_investigation.codex_support_subprocess import (
    run_codex_support_json_subprocess,
)


from tests.codex_support_test_support import EXAMPLE_YSUPPORT_MCP_URL


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
                run_codex_support_json_subprocess(
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

    async def test_codex_support_stream_accepts_large_jsonl_tool_event(self) -> None:
        response = {
            "answer": "grounded answer",
            "requires_human_handoff": False,
            "handoff_reason": None,
            "evidence_summary": "checked",
            "used_tools": ["shell"],
        }
        command = [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "tool_event={'type':'item.completed','item':{"
                "'type':'command_execution','aggregated_output':'x'*300000}}; "
                f"response={response!r}; "
                "final_event={'type':'item.completed','item':{"
                "'type':'agent_message','text':json.dumps(response)}}; "
                "sys.stdout.write(json.dumps(tool_event)+'\\n'+json.dumps(final_event))"
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = await run_codex_support_json_subprocess(
                command=command,
                stdin_text="",
                cwd=None,
                env=dict(os.environ),
                timeout_seconds=5,
                max_output_chars=1000,
                max_error_chars=1000,
                timeout_message="timed out",
                empty_stdout_message="empty",
                oversized_stdout_message="oversized",
                metadata={},
                artifact_run_dir=Path(temp_dir),
                progress_callback=None,
            )

            self.assertEqual(output.final_response_text, json.dumps(response))
            self.assertGreater(
                (Path(temp_dir) / "stdout.txt").stat().st_size,
                300000,
            )

    async def test_image_attachment_stream_enforces_size_without_content_length(
        self,
    ) -> None:
        class _FakeContent:
            async def iter_chunked(self, _chunk_size: int):
                yield b"123"
                yield b"45"

        response = SimpleNamespace(content_length=None, content=_FakeContent())
        with mock.patch(
            "ticket_investigation.codex_support_attachments._MAX_ATTACHMENT_IMAGE_BYTES",
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
                "ticket_investigation.codex_support_attachments._download_attachment_image",
                side_effect=RuntimeError("expired URL"),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Could not prepare image attachment evidence.png: expired URL",
                ):
                    await prepare_support_request_attachments(
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
            self.assertIn("[mcp_servers.ysupport.env]", config_text)
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

    def test_prepare_codex_support_home_syncs_auth_source_from_canonical_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_source = Path(temp_dir) / "auth-source.json"
            auth_source.write_text('{"auth_mode":"stale"}', encoding="utf-8")
            canonical_source = Path(temp_dir) / "canonical-auth.json"
            canonical_source.write_text('{"auth_mode":"fresh"}', encoding="utf-8")

            with (
                mock.patch(
                    "codex_support_home._choose_ysupport_stdio_launcher",
                    return_value=None,
                ),
                mock.patch(
                    "codex_support_home._is_http_url_reachable",
                    return_value=False,
                ),
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

            with (
                mock.patch(
                    "codex_support_home._choose_ysupport_stdio_launcher",
                    return_value=None,
                ),
                mock.patch(
                    "codex_support_home._is_http_url_reachable",
                    return_value=False,
                ),
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

    def test_sync_codex_auth_state_uses_freshest_existing_file_for_all_targets(
        self,
    ) -> None:
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

    def test_sync_codex_auth_state_prefers_canonical_source_when_requested(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_auth = Path(temp_dir) / "bot-home" / "auth.json"
            source_auth = Path(temp_dir) / "auth-source.json"
            canonical_auth = Path(temp_dir) / "canonical-auth.json"
            home_auth.parent.mkdir(parents=True, exist_ok=True)
            home_auth.write_text('{"auth_mode":"bot-stale"}', encoding="utf-8")
            source_auth.write_text('{"auth_mode":"source-stale"}', encoding="utf-8")
            canonical_auth.write_text(
                '{"auth_mode":"canonical-good"}', encoding="utf-8"
            )
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
                "\n".join(
                    [
                        "[mcp_servers.ysupport]",
                        "enabled = true",
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

    def test_prepare_codex_support_home_prefers_reachable_http_mcp_over_stdio(
        self,
    ) -> None:
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
