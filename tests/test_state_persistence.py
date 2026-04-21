from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import state
from state import (
    PublicConversation,
    TicketInvestigationJob,
    bug_report_debounce_channels,
    channel_intent_after_button,
    channels_awaiting_initial_button_press,
    clear_public_conversation,
    clear_ticket_channel_state,
    conversation_threads,
    hydrate_public_conversation,
    hydrate_ticket_state,
    last_bot_reply_ts_by_channel,
    last_wallet_by_channel,
    pending_wallet_confirmation_by_channel,
    persist_public_conversation,
    persist_ticket_state,
    public_conversations,
    stopped_channels,
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
                stopped_channels.add(channel_id)
                channels_awaiting_initial_button_press.add(channel_id)
                channel_intent_after_button[channel_id] = "investigate_issue"
                bug_report_debounce_channels.add(channel_id)

                persist_ticket_state(channel_id)

                conversation_threads.pop(channel_id, None)
                ticket_investigation_jobs.pop(channel_id, None)
                last_wallet_by_channel.pop(channel_id, None)
                pending_wallet_confirmation_by_channel.pop(channel_id, None)
                last_bot_reply_ts_by_channel.pop(channel_id, None)
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
                self.assertIn(channel_id, stopped_channels)
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

                clear_ticket_channel_state(
                    channel_id,
                    keep_stopped=True,
                    delete_persisted=False,
                )

                self.assertNotIn(channel_id, conversation_threads)
                self.assertNotIn(channel_id, ticket_investigation_jobs)
                self.assertNotIn(channel_id, last_wallet_by_channel)
                self.assertIn(channel_id, stopped_channels)
                payload = state._read_json(ticket_dir / f"{channel_id}.json")
                self.assertIsNotNone(payload)
                assert payload is not None
                self.assertTrue(payload["stopped"])
                self.assertEqual(payload["history"], [])


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
