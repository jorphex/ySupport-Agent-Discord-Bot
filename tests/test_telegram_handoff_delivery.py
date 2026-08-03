import tests as _test_environment  # noqa: F401

import asyncio
from types import SimpleNamespace
from unittest.mock import patch


import discord

import config
import state
from discord_support_runtime import InternalInstructionTurnResult
from state import (
    TeamHandoffNotice,
    channel_intent_after_button,
    clear_team_handoff_notice,
    clear_ticket_channel_state,
    clear_ticket_investigation_job,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    last_bot_reply_ts_by_channel,
    pending_tasks,
    stopped_channels,
    team_handoff_notice_by_channel,
    ticket_investigation_jobs,
    ticket_owner_user_id_by_channel,
)
from handoff import (
    build_archived_handoff_notice,
    build_closed_handoff_notice,
    build_handoff_notice,
    build_pending_delivery_handoff_notice,
)
from ysupport import (
    TicketBot,
)
from tests.ticket_flow_test_support import TicketFlowTestCase
from tests.test_ticket_intake import (
    _FakeDiscordChannel,
)


class TicketFlowTests(TicketFlowTestCase):
    async def test_telegram_handoff_reply_rejects_vague_team_message(self) -> None:
        channel_id = 195
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel

        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
        )
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()

        update = {
            "update_id": 2,
            "message": {
                "message_id": 999,
                "chat": {"id": "123"},
                "text": "ok",
                "reply_to_message": {"message_id": 456},
            },
        }
        telegram_feedback: list[tuple[str | None, str, int | None]] = []

        async def fake_send_telegram_message(
            *,
            chat_id: str | None,
            message_text: str,
            reply_to_message_id: int | None = None,
        ):
            telegram_feedback.append((chat_id, message_text, reply_to_message_id))
            return None

        try:
            with patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"):
                with patch(
                    "telegram_handoff_controller.send_telegram_message",
                    new=fake_send_telegram_message,
                ):
                    await bot.telegram_handoffs._handle_telegram_handoff_update(update)
            self.assertEqual(fake_channel.sent_messages, [])
            self.assertEqual(
                telegram_feedback,
                [
                    (
                        "123",
                        "That reply is too vague to send to the user. Reply with the exact update I should give them. Only one clear reply will be used.",
                        999,
                    )
                ],
            )
            notice = team_handoff_notice_by_channel[channel_id]
            self.assertEqual(notice.status, "open")
            self.assertIsNone(notice.pending_reply_text)
        finally:
            clear_team_handoff_notice(channel_id)
            clear_ticket_investigation_job(channel_id)
            last_bot_reply_ts_by_channel.pop(channel_id, None)

    async def test_pending_telegram_handoff_reply_resumes_and_posts_update(
        self,
    ) -> None:
        channel_id = 96
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel

        conversation_threads[channel_id] = [
            {"role": "user", "content": "initial issue"}
        ]
        ticket_owner_user_id_by_channel[channel_id] = 777
        original_notice = build_handoff_notice(
            reason="manual follow-up needed",
            summary="initial issue",
            channel_id=channel_id,
            guild_id=2,
        )
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text=original_notice,
            status="pending_delivery",
            pending_reply_text="tell the user the tx is queued pending signatures",
        )
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        channel_intent_after_button[channel_id] = "investigate_issue"
        edited_messages: list[tuple[str, int, str]] = []

        async def fake_edit_handoff_notice(
            *, chat_id: str, message_id: int, message_text: str
        ) -> bool:
            edited_messages.append((chat_id, message_id, message_text))
            return True

        async def fake_internal_turn(**kwargs) -> InternalInstructionTurnResult:
            return InternalInstructionTurnResult(
                reply=(
                    "The transaction has been queued and is pending multisig "
                    "signatures."
                ),
                conversation_history=conversation_threads[channel_id]
                + [{"role": "assistant", "content": "Delivered update."}],
            )

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch(
                "telegram_handoff_controller.edit_handoff_notice",
                new=fake_edit_handoff_notice,
            ):
                with patch(
                    "telegram_handoff_controller._run_internal_instruction_turn",
                    new=fake_internal_turn,
                ):
                    with patch(
                        "telegram_handoff_controller.send_long_message",
                        new=fake_send_long_message,
                    ):
                        await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
            self.assertEqual(
                edited_messages,
                [
                    (
                        "123",
                        456,
                        build_pending_delivery_handoff_notice(original_notice),
                    ),
                    ("123", 456, build_closed_handoff_notice(original_notice)),
                ],
            )
            self.assertEqual(
                fake_channel.sent_messages,
                ["The transaction has been queued and is pending multisig signatures."],
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(
                ticket_investigation_jobs[channel_id].mode, "waiting_for_user"
            )
        finally:
            conversation_threads.pop(channel_id, None)
            ticket_owner_user_id_by_channel.pop(channel_id, None)
            channel_intent_after_button.pop(channel_id, None)
            clear_team_handoff_notice(channel_id)
            clear_ticket_investigation_job(channel_id)
            last_bot_reply_ts_by_channel.pop(channel_id, None)

    async def test_failed_discord_delivery_does_not_commit_team_reply(self) -> None:
        channel_id = 296
        channel = _FakeDiscordChannel(channel_id)
        channel.category = SimpleNamespace(id=1)
        previous_history = [{"role": "user", "content": "initial issue"}]
        conversation_threads[channel_id] = previous_history
        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            status="pending_delivery",
            pending_reply_text="tell the user it is queued",
        )
        team_handoff_notice_by_channel[channel_id] = notice
        get_or_create_ticket_investigation_job(
            channel_id
        ).mark_escalated_to_human()
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: channel

        async def fake_internal_turn(**kwargs) -> InternalInstructionTurnResult:
            return InternalInstructionTurnResult(
                reply="The transaction is queued.",
                conversation_history=previous_history
                + [
                    {
                        "role": "assistant",
                        "content": "The transaction is queued.",
                    }
                ],
            )

        try:
            with (
                patch(
                    "telegram_handoff_controller.edit_handoff_notice",
                    return_value=True,
                ),
                patch(
                    "telegram_handoff_controller._run_internal_instruction_turn",
                    new=fake_internal_turn,
                ),
                patch(
                    "telegram_handoff_controller.send_long_message",
                    side_effect=RuntimeError("Discord send failed"),
                ),
                patch(
                    "telegram_handoff_controller.reset_ticket_codex_session"
                ) as reset_session,
            ):
                delivered = await bot.telegram_handoffs._deliver_telegram_handoff_reply(
                    channel_id=channel_id,
                    notice=notice,
                    team_reply_text="tell the user it is queued",
                )

            self.assertFalse(delivered)
            self.assertEqual(conversation_threads[channel_id], previous_history)
            self.assertEqual(notice.status, "pending_delivery")
            reset_session.assert_called_once_with(channel_id)
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_failed_team_reply_synthesis_stays_pending_without_raw_delivery(
        self,
    ) -> None:
        channel_id = 197
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            status="pending_delivery",
            pending_reply_text="internal shorthand only",
        )
        team_handoff_notice_by_channel[channel_id] = notice

        async def fail_synthesis(**_kwargs) -> str:
            raise RuntimeError("attachment unavailable")

        try:
            with (
                patch(
                    "telegram_handoff_controller.edit_handoff_notice", return_value=True
                ),
                patch(
                    "telegram_handoff_controller._run_internal_instruction_turn",
                    new=fail_synthesis,
                ),
                patch("telegram_handoff_controller.send_long_message") as send_message,
            ):
                delivered = await bot.telegram_handoffs._deliver_telegram_handoff_reply(
                    channel_id=channel_id,
                    notice=notice,
                    team_reply_text="internal shorthand only",
                )

            self.assertFalse(delivered)
            send_message.assert_not_called()
            self.assertEqual(notice.status, "pending_delivery")
            self.assertEqual(notice.pending_reply_text, "internal shorthand only")
            self.assertIn(channel_id, team_handoff_notice_by_channel)
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_discord_stop_during_telegram_reply_prevents_late_delivery(
        self,
    ) -> None:
        channel_id = 299
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            status="pending_delivery",
            pending_reply_text="Tell the user this is resolved.",
        )
        team_handoff_notice_by_channel[channel_id] = notice
        get_or_create_ticket_investigation_job(channel_id).mark_escalated_to_human()
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel

        async def stop_during_synthesis(**kwargs) -> str:
            self.assertTrue(state.stop_ticket_channel(channel_id))
            return "This reply must not be posted."

        try:
            with (
                patch(
                    "telegram_handoff_controller.edit_handoff_notice", return_value=True
                ),
                patch(
                    "telegram_handoff_controller._run_internal_instruction_turn",
                    new=stop_during_synthesis,
                ),
                patch(
                    "telegram_handoff_controller.reset_ticket_codex_session",
                ) as reset_session,
                patch("telegram_handoff_controller.send_long_message") as send_message,
            ):
                delivered = await bot.telegram_handoffs._deliver_telegram_handoff_reply(
                    channel_id=channel_id,
                    notice=notice,
                    team_reply_text="Tell the user this is resolved.",
                )

            self.assertFalse(delivered)
            send_message.assert_not_awaited()
            reset_session.assert_called_once_with(channel_id)
            self.assertIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_deleted_ticket_archives_open_telegram_handoff_notice(self) -> None:
        channel_id = 188
        original_notice = build_handoff_notice(
            reason="manual follow-up needed",
            summary="need human",
            channel_id=channel_id,
            guild_id=734804446353031319,
        )
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text=original_notice,
        )

        bot = TicketBot(intents=discord.Intents.none())
        fake_channel = _FakeDiscordChannel(channel_id)
        edits: list[tuple[str, int, str]] = []

        async def active_turn() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(active_turn())
        pending_tasks[channel_id] = task

        async def fake_edit_handoff_notice(
            *, chat_id: str, message_id: int, message_text: str
        ) -> bool:
            edits.append((chat_id, message_id, message_text))
            return True

        try:
            with (
                patch(
                    "ticket_channel_lifecycle.discord.TextChannel", _FakeDiscordChannel
                ),
                patch(
                    "ticket_channel_lifecycle.edit_handoff_notice",
                    side_effect=fake_edit_handoff_notice,
                ),
                patch("state.reset_ticket_codex_session"),
            ):
                await bot.on_guild_channel_delete(fake_channel)
            self.assertEqual(
                edits,
                [("123", 456, build_archived_handoff_notice(original_notice))],
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertTrue(task.cancelled())
            self.assertNotIn(channel_id, pending_tasks)
        finally:
            pending_tasks.pop(channel_id, None)
            if not task.done():
                task.cancel()
            clear_team_handoff_notice(channel_id)

    async def test_delivered_pending_close_resumes_without_resending_discord_update(
        self,
    ) -> None:
        channel_id = 196
        original_notice = build_handoff_notice(
            reason="manual follow-up needed",
            summary="need human",
            channel_id=channel_id,
            guild_id=734804446353031319,
        )
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text=original_notice,
            status="delivered_pending_close",
            pending_reply_text="tell the user the tx is queued pending signatures",
        )

        bot = TicketBot(intents=discord.Intents.none())
        edits: list[tuple[str, int, str]] = []

        async def fake_edit_handoff_notice(
            *, chat_id: str, message_id: int, message_text: str
        ) -> bool:
            edits.append((chat_id, message_id, message_text))
            return True

        async def fake_internal_turn(**kwargs) -> str:
            raise AssertionError(
                "Delivered-pending-close should not synthesize a second Discord update."
            )

        async def fake_send_long_message(channel, message, **kwargs):
            raise AssertionError(
                "Delivered-pending-close should not resend to Discord."
            )

        try:
            with patch(
                "telegram_handoff_controller.edit_handoff_notice",
                new=fake_edit_handoff_notice,
            ):
                with patch(
                    "telegram_handoff_controller._run_internal_instruction_turn",
                    new=fake_internal_turn,
                ):
                    with patch(
                        "telegram_handoff_controller.send_long_message",
                        new=fake_send_long_message,
                    ):
                        await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
            self.assertEqual(
                edits,
                [("123", 456, build_closed_handoff_notice(original_notice))],
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
        finally:
            clear_team_handoff_notice(channel_id)
