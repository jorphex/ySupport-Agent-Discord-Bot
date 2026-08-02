import tests as _test_environment  # noqa: F401

import asyncio
from unittest.mock import patch


import discord

import config
import state
from state import (
    TeamHandoffNotice,
    clear_ticket_channel_state,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    pending_attachments_by_channel,
    pending_messages,
    pending_tasks,
    stopped_channels,
    stop_reasons_by_channel,
    team_handoff_notice_by_channel,
    ticket_investigation_jobs,
    ticket_owner_user_id_by_channel,
)
from handoff import (
    DISMISS_HANDOFF_CALLBACK_DATA,
    build_dismissed_handoff_notice,
    build_handoff_notice,
)
from ysupport import (
    TicketBot,
)

from tests.ticket_flow_test_support import TicketFlowTestCase


class TicketFlowTests(TicketFlowTestCase):
    async def test_telegram_handoff_dismissal_stops_and_clears_ticket_before_delete(
        self,
    ) -> None:
        channel_id = 295
        conversation_threads[channel_id] = [
            {"role": "user", "content": "initial issue"},
            {"role": "user", "content": "follow-up details"},
        ]
        pending_messages[channel_id] = "queued follow-up"
        pending_attachments_by_channel[channel_id] = [
            {"url": "https://cdn.example/queued.png"}
        ]
        ticket_owner_user_id_by_channel[channel_id] = 777
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            followup_attachments=[{"url": "https://cdn.example/parked.png"}],
        )
        state.persist_ticket_state(channel_id)

        task_started = asyncio.Event()
        task_cancelled = asyncio.Event()

        async def active_ticket_turn() -> None:
            task_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                task_cancelled.set()
                raise

        active_task = asyncio.create_task(active_ticket_turn())
        pending_tasks[channel_id] = active_task
        await task_started.wait()

        update = {
            "update_id": 3,
            "callback_query": {
                "id": "callback-1",
                "data": DISMISS_HANDOFF_CALLBACK_DATA,
                "from": {"id": 888},
                "message": {
                    "message_id": 456,
                    "chat": {"id": "123"},
                },
            },
        }
        callback_answers: list[tuple[str, str]] = []
        retired_messages: list[tuple[str, int, str]] = []

        async def fake_answer_callback(
            *,
            callback_query_id: str,
            message_text: str,
        ) -> bool:
            self.assertIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            await asyncio.sleep(0)
            self.assertTrue(task_cancelled.is_set())
            callback_answers.append((callback_query_id, message_text))
            return True

        async def fake_retire_notice(
            *,
            chat_id: str,
            message_id: int,
            fallback_message_text: str,
        ) -> bool:
            self.assertIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, conversation_threads)
            self.assertNotIn(channel_id, ticket_investigation_jobs)
            self.assertNotIn(channel_id, pending_messages)
            self.assertNotIn(channel_id, pending_attachments_by_channel)
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            persisted = state._read_json(state._TICKET_STATE_DIR / f"{channel_id}.json")
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertTrue(persisted["stopped"])
            self.assertEqual(persisted["stop_reason"], "manual_stop")
            self.assertIsNone(persisted["team_handoff_notice"])
            retired_messages.append((chat_id, message_id, fallback_message_text))
            return True

        bot = TicketBot(intents=discord.Intents.none())
        try:
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch(
                    "telegram_handoff_controller.answer_telegram_callback_query",
                    new=fake_answer_callback,
                ),
                patch(
                    "telegram_handoff_controller.retire_handoff_notice",
                    new=fake_retire_notice,
                ),
                patch("state.reset_ticket_codex_session") as reset_session,
                patch(
                    "telegram_handoff_controller._run_internal_instruction_turn"
                ) as internal_turn,
                patch("telegram_handoff_controller.send_long_message") as send_message,
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(update)

                stopped_channels.discard(channel_id)
                stop_reasons_by_channel.pop(channel_id, None)
                ticket_owner_user_id_by_channel.pop(channel_id, None)
                state.hydrate_ticket_state(channel_id)
                self.assertIn(channel_id, stopped_channels)
                self.assertEqual(
                    stop_reasons_by_channel[channel_id],
                    "manual_stop",
                )
                self.assertEqual(
                    ticket_owner_user_id_by_channel[channel_id],
                    777,
                )
                self.assertNotIn(channel_id, team_handoff_notice_by_channel)

                await bot.telegram_handoffs._handle_telegram_handoff_update(update)

            with self.assertRaises(asyncio.CancelledError):
                await active_task
            self.assertEqual(
                callback_answers,
                [
                    (
                        "callback-1",
                        "Handoff dismissed. Handle this ticket in Discord.",
                    ),
                    ("callback-1", "This handoff is already closed."),
                ],
            )
            self.assertEqual(
                retired_messages,
                [
                    (
                        "123",
                        456,
                        build_dismissed_handoff_notice("Dismissed. Handle in Discord."),
                    )
                ],
            )
            self.assertNotIn(channel_id, pending_tasks)
            self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 777)
            reset_session.assert_called_once_with(channel_id)
            internal_turn.assert_not_called()
            send_message.assert_not_called()

            payload = state._read_json(state._TICKET_STATE_DIR / f"{channel_id}.json")
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertTrue(payload["stopped"])
            self.assertEqual(payload["stop_reason"], "manual_stop")
            self.assertEqual(payload["ticket_owner_user_id"], 777)
            self.assertEqual(payload["history"], [])
            self.assertIsNone(payload["investigation_job"])
            self.assertIsNone(payload["team_handoff_notice"])
        finally:
            if not active_task.done():
                active_task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_telegram_handoff_dismissal_rejects_wrong_or_closed_notice(
        self,
    ) -> None:
        channel_id = 296
        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
        )
        team_handoff_notice_by_channel[channel_id] = notice
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        callback_answers: list[str] = []

        async def fake_answer_callback(
            *,
            callback_query_id: str,
            message_text: str,
        ) -> bool:
            callback_answers.append(message_text)
            return True

        def callback_update(*, data: str, chat_id: str, message_id: int):
            return {
                "callback_query": {
                    "id": "callback-2",
                    "data": data,
                    "message": {
                        "message_id": message_id,
                        "chat": {"id": chat_id},
                    },
                }
            }

        bot = TicketBot(intents=discord.Intents.none())
        try:
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch(
                    "telegram_handoff_controller.answer_telegram_callback_query",
                    new=fake_answer_callback,
                ),
                patch(
                    "telegram_handoff_controller.retire_handoff_notice"
                ) as retire_notice,
                patch("state.reset_ticket_codex_session") as reset_session,
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(
                    callback_update(
                        data="other_action",
                        chat_id="123",
                        message_id=456,
                    )
                )
                await bot.telegram_handoffs._handle_telegram_handoff_update(
                    callback_update(
                        data=DISMISS_HANDOFF_CALLBACK_DATA,
                        chat_id="999",
                        message_id=456,
                    )
                )
                await bot.telegram_handoffs._handle_telegram_handoff_update(
                    callback_update(
                        data=DISMISS_HANDOFF_CALLBACK_DATA,
                        chat_id="123",
                        message_id=999,
                    )
                )
                notice.status = "pending_delivery"
                await bot.telegram_handoffs._handle_telegram_handoff_update(
                    callback_update(
                        data=DISMISS_HANDOFF_CALLBACK_DATA,
                        chat_id="123",
                        message_id=456,
                    )
                )

            self.assertNotIn(channel_id, stopped_channels)
            self.assertIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(
                callback_answers,
                [
                    "This action is no longer available.",
                    "This action is no longer available.",
                    "This handoff is already closed.",
                    "This handoff is already closed.",
                ],
            )
            retire_notice.assert_not_called()
            reset_session.assert_not_called()
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_telegram_handoff_cleanup_failure_stays_retriable(
        self,
    ) -> None:
        channel_id = 297
        original_notice = build_handoff_notice(
            reason="manual follow-up needed",
            summary="initial issue",
            channel_id=channel_id,
            guild_id=2,
        )
        conversation_threads[channel_id] = [
            {"role": "user", "content": "initial issue"}
        ]
        get_or_create_ticket_investigation_job(channel_id).mark_escalated_to_human()
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text=original_notice,
        )
        update = {
            "callback_query": {
                "id": "callback-3",
                "data": DISMISS_HANDOFF_CALLBACK_DATA,
                "message": {
                    "message_id": 456,
                    "chat": {"id": "123"},
                },
            }
        }
        bot = TicketBot(intents=discord.Intents.none())
        try:
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch(
                    "telegram_handoff_controller.answer_telegram_callback_query",
                    return_value=True,
                ),
                patch(
                    "telegram_handoff_controller.retire_handoff_notice",
                    return_value=False,
                ) as retire_notice,
                patch("telegram_handoff_controller.logging.warning") as warning,
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(update)

            retire_notice.assert_awaited_once_with(
                chat_id="123",
                message_id=456,
                fallback_message_text=build_dismissed_handoff_notice(original_notice),
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
            self.assertIn(channel_id, team_handoff_notice_by_channel)
            self.assertNotIn(channel_id, conversation_threads)
            self.assertNotIn(channel_id, ticket_investigation_jobs)
            persisted = state._read_json(state._TICKET_STATE_DIR / f"{channel_id}.json")
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertTrue(persisted["stopped"])
            self.assertIsNotNone(persisted["team_handoff_notice"])
            warning.assert_called_once()

            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)
            team_handoff_notice_by_channel.pop(channel_id, None)
            state.hydrate_ticket_state(channel_id)
            self.assertIn(channel_id, stopped_channels)
            self.assertIn(channel_id, team_handoff_notice_by_channel)

            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch(
                    "telegram_handoff_controller.answer_telegram_callback_query",
                    return_value=True,
                ),
                patch(
                    "telegram_handoff_controller.retire_handoff_notice",
                    return_value=True,
                ) as retry_retire,
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(update)

            retry_retire.assert_awaited_once_with(
                chat_id="123",
                message_id=456,
                fallback_message_text=build_dismissed_handoff_notice(original_notice),
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_telegram_handoff_dismissal_can_retry_after_state_write_failure(
        self,
    ) -> None:
        channel_id = 298
        ticket_owner_user_id_by_channel[channel_id] = 777
        conversation_threads[channel_id] = [
            {"role": "user", "content": "initial issue"}
        ]
        get_or_create_ticket_investigation_job(channel_id).mark_escalated_to_human()
        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
        )
        team_handoff_notice_by_channel[channel_id] = notice
        update = {
            "callback_query": {
                "id": "callback-4",
                "data": DISMISS_HANDOFF_CALLBACK_DATA,
                "message": {
                    "message_id": 456,
                    "chat": {"id": "123"},
                },
            }
        }

        bot = TicketBot(intents=discord.Intents.none())
        try:
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch("state._write_json", return_value=False),
                patch(
                    "telegram_handoff_controller.answer_telegram_callback_query",
                    return_value=True,
                ) as answer_callback,
                patch(
                    "telegram_handoff_controller.retire_handoff_notice"
                ) as retire_notice,
                patch("telegram_handoff_controller.logging.error") as log_error,
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(update)

            answer_callback.assert_awaited_once_with(
                callback_query_id="callback-4",
                message_text=(
                    "The handoff could not be dismissed safely. "
                    "Please try again, or use Stop Bot in Discord."
                ),
            )
            retire_notice.assert_not_called()
            log_error.assert_called_once()
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 777)
            self.assertIs(team_handoff_notice_by_channel[channel_id], notice)

            reply_update = {
                "message": {
                    "message_id": 999,
                    "chat": {"id": "123"},
                    "text": "Tell the user this is resolved.",
                    "reply_to_message": {"message_id": 456},
                }
            }
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch.object(
                    bot.telegram_handoffs,
                    "_deliver_telegram_handoff_reply",
                ) as deliver_reply,
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(
                    reply_update
                )
            deliver_reply.assert_not_awaited()

            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch(
                    "telegram_handoff_controller.answer_telegram_callback_query",
                    return_value=True,
                ) as retry_answer,
                patch(
                    "telegram_handoff_controller.retire_handoff_notice",
                    return_value=True,
                ) as retry_retire,
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(update)

            retry_answer.assert_awaited_once_with(
                callback_query_id="callback-4",
                message_text="Handoff dismissed. Handle this ticket in Discord.",
            )
            retry_retire.assert_awaited_once_with(
                chat_id="123",
                message_id=456,
                fallback_message_text=build_dismissed_handoff_notice(
                    "Dismissed. Handle in Discord."
                ),
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            persisted = state._read_json(state._TICKET_STATE_DIR / f"{channel_id}.json")
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertTrue(persisted["stopped"])
            self.assertIsNone(persisted["team_handoff_notice"])
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)
