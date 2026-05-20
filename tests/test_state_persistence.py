from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import state
from state import (
    apply_initial_button_intent,
    PublicConversation,
    TeamHandoffNotice,
    TicketInvestigationJob,
    bug_report_debounce_channels,
    channel_intent_after_button,
    channels_awaiting_initial_button_press,
    clear_team_handoff_notice,
    clear_public_conversation,
    clear_ticket_channel_state,
    conversation_threads,
    hydrate_public_conversation,
    hydrate_ticket_state,
    last_bot_reply_ts_by_channel,
    last_wallet_by_channel,
    mark_ticket_channel_stopped,
    mark_ticket_awaiting_initial_button,
    pending_wallet_confirmation_by_channel,
    persist_public_conversation,
    persist_ticket_state,
    public_conversations,
    recover_ticket_channel_from_runtime_stop,
    remember_team_handoff_notice,
    reset_ticket_channel_for_terminal_reply,
    stop_reasons_by_channel,
    stopped_channels,
    stop_ticket_channel,
    team_handoff_notice_by_channel,
    ticket_owner_user_id_by_channel,
    ticket_investigation_jobs,
)


class TicketStatePersistenceTests(unittest.TestCase):
    def test_ticket_state_round_trip_persists_history_job_and_flags(self) -> None:
        channel_id = 101
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
            ):
                job = TicketInvestigationJob(channel_id=channel_id)
                job.record_requested_intent("investigate_issue")
                job.mark_waiting_for_user()
                job.remember_wallet("0xabc")
                conversation_threads[channel_id] = [{"role": "user", "content": "hi"}]
                ticket_investigation_jobs[channel_id] = job
                last_wallet_by_channel[channel_id] = "0xabc"
                pending_wallet_confirmation_by_channel[channel_id] = "0xdef"
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                ticket_owner_user_id_by_channel[channel_id] = 555
                stopped_channels.add(channel_id)
                stop_reasons_by_channel[channel_id] = "manual_stop"
                channels_awaiting_initial_button_press.add(channel_id)
                channel_intent_after_button[channel_id] = "investigate_issue"
                bug_report_debounce_channels.add(channel_id)

                persist_ticket_state(channel_id)

                conversation_threads.pop(channel_id, None)
                ticket_investigation_jobs.pop(channel_id, None)
                last_wallet_by_channel.pop(channel_id, None)
                pending_wallet_confirmation_by_channel.pop(channel_id, None)
                last_bot_reply_ts_by_channel.pop(channel_id, None)
                ticket_owner_user_id_by_channel.pop(channel_id, None)
                stopped_channels.discard(channel_id)
                channels_awaiting_initial_button_press.discard(channel_id)
                channel_intent_after_button.pop(channel_id, None)
                bug_report_debounce_channels.discard(channel_id)

                hydrate_ticket_state(channel_id)

                self.assertEqual(
                    conversation_threads[channel_id],
                    [{"role": "user", "content": "hi"}],
                )
                self.assertEqual(
                    ticket_investigation_jobs[channel_id].requested_intent,
                    "investigate_issue",
                )
                self.assertEqual(last_wallet_by_channel[channel_id], "0xabc")
                self.assertEqual(
                    pending_wallet_confirmation_by_channel[channel_id],
                    "0xdef",
                )
                self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 555)
                self.assertIn(channel_id, stopped_channels)
                self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
                self.assertIn(channel_id, channels_awaiting_initial_button_press)
                self.assertEqual(
                    channel_intent_after_button[channel_id],
                    "investigate_issue",
                )
                self.assertIn(channel_id, bug_report_debounce_channels)

    def test_clear_ticket_channel_state_keeps_stopped_marker_and_persists_minimal_state(self) -> None:
        channel_id = 102
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
                patch("state.reset_ticket_codex_session"),
            ):
                conversation_threads[channel_id] = [{"role": "user", "content": "hi"}]
                ticket_investigation_jobs[channel_id] = TicketInvestigationJob(channel_id=channel_id)
                last_wallet_by_channel[channel_id] = "0xabc"
                ticket_owner_user_id_by_channel[channel_id] = 777

                clear_ticket_channel_state(
                    channel_id,
                    keep_stopped=True,
                    delete_persisted=False,
                )

                self.assertNotIn(channel_id, conversation_threads)
                self.assertNotIn(channel_id, ticket_investigation_jobs)
                self.assertNotIn(channel_id, last_wallet_by_channel)
                self.assertIn(channel_id, stopped_channels)
                self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 777)
                self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
                payload = state._read_json(ticket_dir / f"{channel_id}.json")
                self.assertIsNotNone(payload)
                assert payload is not None
                self.assertTrue(payload["stopped"])
                self.assertEqual(payload["stop_reason"], "manual_stop")
                self.assertEqual(payload["history"], [])
                self.assertEqual(payload["ticket_owner_user_id"], 777)

    def test_apply_initial_button_intent_updates_waiting_flags_and_job(self) -> None:
        channel_id = 105
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
            ):
                mark_ticket_awaiting_initial_button(channel_id)

                job = apply_initial_button_intent(
                    channel_id,
                    "investigate_issue",
                    investigation_mode="collecting",
                    bug_report_debounce=True,
                )

                self.assertEqual(job.requested_intent, "investigate_issue")
                self.assertEqual(job.mode, "collecting")
                self.assertNotIn(channel_id, channels_awaiting_initial_button_press)
                self.assertEqual(
                    channel_intent_after_button[channel_id],
                    "investigate_issue",
                )
                self.assertIn(channel_id, bug_report_debounce_channels)

                payload = state._read_json(ticket_dir / f"{channel_id}.json")
                self.assertIsNotNone(payload)
                assert payload is not None
                self.assertEqual(payload["channel_intent_after_button"], "investigate_issue")
                self.assertFalse(payload["awaiting_initial_button_press"])
                self.assertTrue(payload["bug_report_debounce"])

    def test_reset_ticket_channel_for_terminal_reply_deletes_persisted_state(self) -> None:
        channel_id = 103
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
                patch("state.reset_ticket_codex_session"),
            ):
                conversation_threads[channel_id] = [{"role": "user", "content": "hi"}]
                ticket_investigation_jobs[channel_id] = TicketInvestigationJob(channel_id=channel_id)
                ticket_owner_user_id_by_channel[channel_id] = 999
                persist_ticket_state(channel_id)

                reset_ticket_channel_for_terminal_reply(channel_id)

                self.assertNotIn(channel_id, conversation_threads)
                self.assertNotIn(channel_id, ticket_investigation_jobs)
                self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 999)
                self.assertFalse((ticket_dir / f"{channel_id}.json").exists())

    def test_stop_ticket_channel_and_mark_ticket_channel_stopped_persist_stop_state(self) -> None:
        channel_id = 104
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
                patch("state.reset_ticket_codex_session"),
            ):
                conversation_threads[channel_id] = [{"role": "user", "content": "hi"}]

                stop_ticket_channel(channel_id)
                self.assertIn(channel_id, stopped_channels)
                self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")

                payload = state._read_json(ticket_dir / f"{channel_id}.json")
                self.assertIsNotNone(payload)
                assert payload is not None
                self.assertTrue(payload["stopped"])
                self.assertEqual(payload["stop_reason"], "manual_stop")

                stopped_channels.discard(channel_id)
                stop_reasons_by_channel.pop(channel_id, None)
                mark_ticket_channel_stopped(channel_id, reason="runtime_error")
                payload = state._read_json(ticket_dir / f"{channel_id}.json")
                self.assertIsNotNone(payload)
                assert payload is not None
                self.assertTrue(payload["stopped"])
                self.assertEqual(payload["stop_reason"], "runtime_error")

    def test_recover_ticket_channel_from_runtime_stop_preserves_context(self) -> None:
        channel_id = 106
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
                patch("state.reset_ticket_codex_session"),
            ):
                job = TicketInvestigationJob(channel_id=channel_id)
                job.record_requested_intent("data_deposits_withdrawals_start")
                job.mark_waiting_for_user()
                job.remember_wallet("0xabc")
                ticket_investigation_jobs[channel_id] = job
                last_wallet_by_channel[channel_id] = "0xabc"
                ticket_owner_user_id_by_channel[channel_id] = 888
                stopped_channels.add(channel_id)
                stop_reasons_by_channel[channel_id] = "runtime_error"

                persisted_before = state._read_json(ticket_dir / f"{channel_id}.json")
                if persisted_before is None:
                    persist_ticket_state(channel_id)

                recovered = recover_ticket_channel_from_runtime_stop(channel_id)

                self.assertTrue(recovered)
                self.assertNotIn(channel_id, stopped_channels)
                self.assertNotIn(channel_id, stop_reasons_by_channel)
                self.assertEqual(last_wallet_by_channel[channel_id], "0xabc")
                self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 888)
                self.assertIsNotNone(ticket_investigation_jobs[channel_id])
                payload = state._read_json(ticket_dir / f"{channel_id}.json")
                self.assertIsNotNone(payload)
                assert payload is not None
                self.assertFalse(payload["stopped"])
                self.assertIsNone(payload["stop_reason"])

    def test_team_handoff_notice_persists_and_round_trips(self) -> None:
        channel_id = 107
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
            ):
                remember_team_handoff_notice(
                    channel_id,
                    TeamHandoffNotice(
                        telegram_chat_id="123",
                        telegram_message_id=456,
                        target="support_manual",
                        reason="manual follow-up needed",
                        message_text="Ticket-0107: support_manual",
                        reply_consumed=True,
                        status="delivered_pending_close",
                        pending_reply_text="tell the user we queued the tx",
                        pending_reply_message_id=789,
                    ),
                )

                team_handoff_notice_by_channel.pop(channel_id, None)
                hydrate_ticket_state(channel_id)

                notice = team_handoff_notice_by_channel.get(channel_id)
                self.assertIsNotNone(notice)
                assert notice is not None
                self.assertEqual(notice.telegram_chat_id, "123")
                self.assertEqual(notice.telegram_message_id, 456)
                self.assertEqual(notice.target, "support_manual")
                self.assertEqual(notice.reason, "manual follow-up needed")
                self.assertEqual(notice.message_text, "Ticket-0107: support_manual")
                self.assertTrue(notice.reply_consumed)
                self.assertEqual(notice.status, "delivered_pending_close")
                self.assertEqual(notice.pending_reply_text, "tell the user we queued the tx")
                self.assertEqual(notice.pending_reply_message_id, 789)

                clear_team_handoff_notice(channel_id)
                self.assertNotIn(channel_id, team_handoff_notice_by_channel)

    def test_telegram_update_offset_persists_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            telegram_state_path = Path(temp_dir) / "telegram_state.json"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
                patch.object(state, "_TELEGRAM_STATE_PATH", telegram_state_path),
            ):
                state.persist_telegram_update_offset(12345)
                self.assertEqual(state.load_telegram_update_offset(), 12345)


class PublicStatePersistenceTests(unittest.TestCase):
    def test_public_state_round_trip_persists_history_and_investigation_job(self) -> None:
        author_id = 501
        with tempfile.TemporaryDirectory() as temp_dir:
            ticket_dir = Path(temp_dir) / "tickets"
            public_dir = Path(temp_dir) / "public"
            with (
                patch.object(state, "_TICKET_STATE_DIR", ticket_dir),
                patch.object(state, "_PUBLIC_STATE_DIR", public_dir),
            ):
                job = TicketInvestigationJob(channel_id=77)
                job.record_requested_intent("docs_qa")
                job.mark_waiting_for_user()
                public_conversations[author_id] = PublicConversation(
                    history=[{"role": "assistant", "content": "Earlier context"}],
                    last_interaction_time=datetime.now(timezone.utc),
                    investigation_job=job,
                )

                persist_public_conversation(author_id)
                public_conversations.pop(author_id, None)
                hydrate_public_conversation(author_id)

                conversation = public_conversations.get(author_id)
                self.assertIsNotNone(conversation)
                assert conversation is not None
                self.assertEqual(
                    conversation.history,
                    [{"role": "assistant", "content": "Earlier context"}],
                )
                self.assertIsNotNone(conversation.investigation_job)
                assert conversation.investigation_job is not None
                self.assertEqual(
                    conversation.investigation_job.requested_intent,
                    "docs_qa",
                )

    def test_clear_public_conversation_resets_public_codex_session(self) -> None:
        with patch("state.reset_public_codex_session") as mock_reset:
            public_conversations[501] = PublicConversation(
                history=[],
                last_interaction_time=datetime.now(timezone.utc),
            )
            clear_public_conversation(501)

        mock_reset.assert_called_once_with(501)
