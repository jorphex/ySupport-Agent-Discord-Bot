import io
import tempfile
import unittest
from unittest.mock import patch

import config
from ticket_execution_replay_cli import (
    _reset_replay_session,
    build_arg_parser,
    execute_request_json,
    load_request_json,
)


class TicketExecutionReplayCliTests(unittest.IsolatedAsyncioTestCase):
    def test_load_request_json_reads_file(self) -> None:
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as handle:
            handle.write('{"request":"from-file"}')
            handle.flush()

            loaded = load_request_json(handle.name)

        self.assertEqual(loaded, '{"request":"from-file"}')

    def test_load_request_json_reads_stdin_when_path_missing(self) -> None:
        with patch("sys.stdin", io.StringIO('{"request":"from-stdin"}')):
            loaded = load_request_json(None)

        self.assertEqual(loaded, '{"request":"from-stdin"}')

    def test_arg_parser_supports_mode_no_fallback_and_fresh_session(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "sample-request.json",
                "--mode",
                "codex_support_exec",
                "--no-fallback",
                "--fresh-session",
            ]
        )

        self.assertEqual(args.request_path, "sample-request.json")
        self.assertEqual(args.mode, "codex_support_exec")
        self.assertTrue(args.no_fallback)
        self.assertTrue(args.fresh_session)

    def test_reset_replay_session_resets_conversation_key_for_request(self) -> None:
        request_json = (
            '{"aggregated_text":"help","input_list":[],"current_history":[],'
            '"run_context":{"channel_id":123,"category_id":null,"is_public_trigger":false,'
            '"conversation_owner_id":null,"project_context":"yearn","initial_button_intent":null,'
            '"repo_search_calls":0,"repo_fetch_calls":0,"repo_searches_without_fetch":0,'
            '"repo_last_search_query":null,"repo_last_search_artifact_refs":[]},'
            '"investigation_job":{"channel_id":123,"requested_intent":null,"mode":"idle",'
            '"current_specialty":null,"last_specialty":null,'
            '"evidence":{"wallet":null,"chain":null,"tx_hashes":[],'
            '"withdrawal_target_chain":null,"withdrawal_target_vault":null}},'
            '"workflow_name":"tests.replay","wants_bug_review_status":false,'
            '"precomputed_boundary":null,"smoke_mode":null}'
        )

        observed = {}

        class _FakeManager:
            def conversation_key_for_request(self, request):
                observed["channel_id"] = request.run_context["channel_id"]
                return "ticket:123"

            def reset(self, conversation_key: str) -> None:
                observed["conversation_key"] = conversation_key

        with patch(
            "ticket_execution_replay_cli.CodexSupportSessionManager",
            return_value=_FakeManager(),
        ):
            _reset_replay_session(request_json)

        self.assertEqual(observed["channel_id"], 123)
        self.assertEqual(observed["conversation_key"], "ticket:123")

    async def test_execute_request_json_applies_override_and_restores_config(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT

        observed = {}

        class _FakeEndpoint:
            async def execute_json_turn(self, request_json: str) -> str:
                observed["request_json"] = request_json
                observed["runtime_mode"] = config.TICKET_EXECUTION_ENDPOINT
                observed["runtime_fallback"] = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
                return '{"ok":true}'

        def _fake_build_endpoint(delegate):
            observed["delegate_type"] = type(delegate).__name__
            observed["build_mode"] = config.TICKET_EXECUTION_ENDPOINT
            observed["build_fallback"] = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
            return _FakeEndpoint()

        class _FakeExecutor:
            pass

        try:
            with patch(
                "ticket_execution_replay_cli.build_local_ticket_investigation_executor",
                return_value=_FakeExecutor(),
            ), patch(
                "ticket_execution_replay_cli.build_ticket_execution_json_endpoint",
                side_effect=_fake_build_endpoint,
            ):
                response_json = await execute_request_json(
                    '{"request":"demo"}',
                    mode="local",
                    fallback_mode="",
                    fresh_session=False,
                )
        finally:
            self.assertEqual(config.TICKET_EXECUTION_ENDPOINT, original_mode)
            self.assertEqual(
                config.TICKET_EXECUTION_FALLBACK_ENDPOINT,
                original_fallback,
            )

        self.assertEqual(response_json, '{"ok":true}')
        self.assertEqual(observed["request_json"], '{"request":"demo"}')
        self.assertEqual(observed["build_mode"], "local")
        self.assertEqual(observed["runtime_mode"], "local")
        self.assertEqual(observed["build_fallback"], "")
        self.assertEqual(observed["runtime_fallback"], "")
        self.assertEqual(observed["delegate_type"], "_FakeExecutor")

    async def test_execute_request_json_uses_noop_delegate_for_non_local_modes(self) -> None:
        observed = {}

        class _FakeEndpoint:
            async def execute_json_turn(self, request_json: str) -> str:
                observed["request_json"] = request_json
                return '{"ok":true}'

        def _fake_build_endpoint(delegate):
            observed["delegate_type"] = type(delegate).__name__
            return _FakeEndpoint()

        with patch(
            "ticket_execution_replay_cli.build_local_ticket_investigation_executor",
            side_effect=AssertionError("local executor should not be built"),
        ), patch(
            "ticket_execution_replay_cli.build_ticket_execution_json_endpoint",
            side_effect=_fake_build_endpoint,
        ):
            response_json = await execute_request_json(
                '{"request":"demo"}',
                mode="codex_support_exec",
                fallback_mode="",
                fresh_session=False,
            )

        self.assertEqual(response_json, '{"ok":true}')
        self.assertEqual(observed["delegate_type"], "_NoopExecutor")
