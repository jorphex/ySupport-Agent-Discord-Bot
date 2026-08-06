import tests as _test_environment  # noqa: F401

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import config
from codex_support_home import (
    prepare_codex_auth_link,
    prepare_codex_support_home,
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
from ticket_investigation.json_endpoint import _codex_env
from ticket_investigation.codex_support_subprocess import (
    _usage_from_codex_event,
    run_codex_support_json_subprocess,
)


from tests.codex_support_test_support import EXAMPLE_YSUPPORT_MCP_URL


class CodexSupportEndpointTests(unittest.IsolatedAsyncioTestCase):
    def test_codex_environment_excludes_bot_and_provider_secrets(self) -> None:
        environment = {
            "HOME": "/tmp/service-home",
            "PATH": "/usr/bin",
            "LANG": "C.UTF-8",
            "OPENAI_API_KEY": "openai-secret",
            "PINECONE_API_KEY": "pinecone-secret",
            "ALCHEMY_KEY": "alchemy-secret",
            "MCP_SERVER_API_KEY": "mcp-secret",
            "GITHUB_TOKEN": "github-secret",
            "DISCORD_BOT_TOKEN": "discord-secret",
            "TELEGRAM_BOT_TOKEN": "telegram-secret",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            codex_env = _codex_env()

        self.assertEqual(codex_env["HOME"], "/tmp/service-home")
        self.assertEqual(codex_env["PATH"], "/usr/bin")
        self.assertEqual(codex_env["LANG"], "C.UTF-8")
        self.assertEqual(codex_env["CODEX_HOME"], config.TICKET_EXECUTION_CODEX_HOME)
        for secret_key in environment.keys() - {"HOME", "PATH", "LANG"}:
            self.assertNotIn(secret_key, codex_env)

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

    @unittest.skipIf(os.name == "nt", "POSIX process-group behavior")
    async def test_timeout_kills_descendant_after_process_leader_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            child_pid_path = Path(temp_dir) / "child_pid.txt"
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib, subprocess, sys; "
                    "child = subprocess.Popen("
                    "[sys.executable, '-c', 'import time; time.sleep(60)']); "
                    f"pathlib.Path({str(child_pid_path)!r}).write_text("
                    "str(child.pid), encoding='utf-8')"
                ),
            ]

            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await run_codex_support_json_subprocess(
                    command=command,
                    stdin_text="",
                    cwd=None,
                    env=dict(os.environ),
                    timeout_seconds=0.2,
                    max_output_chars=1000,
                    max_error_chars=1000,
                    timeout_message="timed out",
                    empty_stdout_message="empty",
                    oversized_stdout_message="oversized",
                    metadata={},
                    artifact_run_dir=None,
                    progress_callback=None,
                )

            self.assertTrue(child_pid_path.exists())
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.05)
            else:
                self.fail("Timed-out Codex execution left a descendant running.")

    async def test_timeout_is_one_deadline_for_streams_and_process_exit(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import os,time; time.sleep(0.4); "
                "os.close(1); os.close(2); time.sleep(60)"
            ),
        ]

        started_at = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            await run_codex_support_json_subprocess(
                command=command,
                stdin_text="",
                cwd=None,
                env=dict(os.environ),
                timeout_seconds=0.5,
                max_output_chars=1000,
                max_error_chars=1000,
                timeout_message="timed out",
                empty_stdout_message="empty",
                oversized_stdout_message="oversized",
                metadata={},
                artifact_run_dir=None,
                progress_callback=None,
            )
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 0.75)

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
            with mock.patch(
                "ticket_investigation.codex_support_subprocess._MAX_STDOUT_CAPTURE_CHARS",
                1024,
            ):
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
            self.assertLessEqual((Path(temp_dir) / "stdout.txt").stat().st_size, 1024)
            metadata = json.loads(
                (Path(temp_dir) / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metadata["stdout_truncated"])
            self.assertFalse(metadata["stderr_truncated"])

    async def test_codex_support_stream_records_usage_and_timing_metrics(self) -> None:
        response = {"answer": "grounded answer"}
        command = [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "events=["
                "{'type':'turn.started'},"
                "{'type':'item.started','item':{'type':'command_execution'}},"
                f"{{'type':'item.completed','item':{{'type':'agent_message','text':json.dumps({response!r})}}}},"
                "{'type':'turn.completed','usage':{"
                "'input_tokens':1200,'cached_input_tokens':900,"
                "'cache_write_input_tokens':0,'output_tokens':50,"
                "'reasoning_output_tokens':10}}]; "
                "sys.stdout.write('\\n'.join(json.dumps(event) for event in events))"
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertLogs(level="INFO") as captured_logs:
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
            self.assertTrue(
                any(
                    "input_tokens=1200" in line
                    and "cached_input_tokens=900" in line
                    and "cache_write_input_tokens=0" in line
                    for line in captured_logs.output
                )
            )
            metadata = json.loads(
                (Path(temp_dir) / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["input_tokens"], 1200)
            self.assertEqual(metadata["cached_input_tokens"], 900)
            self.assertEqual(metadata["cache_write_input_tokens"], 0)
            self.assertIsInstance(metadata["first_item_ms"], int)
            self.assertIsInstance(metadata["total_ms"], int)
            self.assertGreaterEqual(metadata["total_ms"], metadata["first_item_ms"])

    def test_codex_usage_parser_ignores_invalid_numeric_fields(self) -> None:
        usage = _usage_from_codex_event(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": True,
                    "cached_input_tokens": -1,
                    "output_tokens": "50",
                },
            }
        )

        self.assertIsNone(usage)

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

    def test_prepare_codex_support_home_writes_private_http_config_and_links_auth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            live_auth = Path(temp_dir) / "service-auth.json"
            live_auth.write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")
            home = prepare_codex_support_home(
                codex_home=Path(temp_dir) / "bot-home",
                auth_link_source=live_auth,
                ysupport_mcp_url=EXAMPLE_YSUPPORT_MCP_URL,
                mcp_server_api_key="secret-key",
                web_search_mode="live",
            )

            self.assertTrue(home.config_path.exists())
            self.assertTrue(home.auth_path.is_symlink())
            self.assertTrue(home.instructions_path.exists())
            self.assertTrue(home.ysupport_mcp_enabled)
            config_text = home.config_path.read_text(encoding="utf-8")
            self.assertIn('sandbox_mode = "danger-full-access"', config_text)
            self.assertIn('web_search = "live"', config_text)
            self.assertIn("model_instructions_file =", config_text)
            self.assertIn("[mcp_servers.ysupport]", config_text)
            self.assertIn(f'url = "{EXAMPLE_YSUPPORT_MCP_URL}"', config_text)
            self.assertIn('Authorization = "Bearer secret-key"', config_text)
            self.assertNotIn("command =", config_text)
            self.assertNotIn("[tools]", config_text)
            self.assertNotIn("view_image", config_text)
            self.assertNotIn("openai_docs", config_text)
            self.assertTrue(
                home.instructions_path.read_text(encoding="utf-8").startswith(
                    "You are ySupport,"
                )
            )
            self.assertEqual(
                home.auth_path.read_text(encoding="utf-8"),
                '{"auth_mode":"chatgpt"}',
            )
            self.assertEqual(home.config_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(home.instructions_path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("\r", config_text)

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

    def test_prepare_codex_auth_link_rejects_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "not a readable file"):
                prepare_codex_auth_link(
                    home_auth_path=Path(temp_dir) / "bot-home" / "auth.json",
                    auth_link_source_path=Path(temp_dir) / "missing-auth.json",
                )

    def test_prepare_codex_support_home_can_run_without_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = prepare_codex_support_home(
                codex_home=Path(temp_dir) / "bot-home",
                ysupport_mcp_url="",
                mcp_server_api_key="",
            )

            self.assertFalse(home.ysupport_mcp_enabled)
            self.assertNotIn(
                "[mcp_servers.ysupport]",
                home.config_path.read_text(encoding="utf-8"),
            )

    def test_prepare_codex_support_home_rejects_partial_or_invalid_http_mcp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for url, key in (
                (EXAMPLE_YSUPPORT_MCP_URL, ""),
                ("", "secret-key"),
                ("stdio://ysupport", "secret-key"),
                ("http://user:password@127.0.0.1/mcp", "secret-key"),
            ):
                with self.subTest(url=url, has_key=bool(key)):
                    with self.assertRaises(ValueError):
                        prepare_codex_support_home(
                            codex_home=Path(temp_dir) / "bot-home",
                            ysupport_mcp_url=url,
                            mcp_server_api_key=key,
                        )
