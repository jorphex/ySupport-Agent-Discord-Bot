import tests as _test_environment  # noqa: F401

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


import discord

import config
import handoff
from discord_support_runtime import InternalInstructionTurnResult
from state import (
    TeamHandoffNotice,
    channel_intent_after_button,
    clear_team_handoff_notice,
    clear_ticket_investigation_job,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    last_bot_reply_ts_by_channel,
    team_handoff_notice_by_channel,
    ticket_investigation_jobs,
    ticket_owner_user_id_by_channel,
)
from handoff import (
    DISMISS_HANDOFF_CALLBACK_DATA,
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
    async def test_handoff_summary_uses_configured_terra_low_role(self) -> None:
        create_response = AsyncMock(
            return_value=SimpleNamespace(
                output_text='{"summary":"A manual vault action remains."}'
            )
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=create_response),
        )

        with (
            patch.dict("os.environ", {"YSUPPORT_ALLOW_TEST_LLM": "1"}),
            patch.object(config, "OPENAI_API_KEY", "test-key"),
            patch.object(config, "TELEGRAM_HANDOFF_SUMMARY_MODEL", "gpt-5.6-terra"),
            patch.object(
                config,
                "TELEGRAM_HANDOFF_SUMMARY_REASONING_EFFORT",
                "low",
            ),
            patch("handoff._get_handoff_summary_async_client", return_value=client),
        ):
            summary = await handoff.summarize_handoff_summary(
                reason="manual strategy action",
                summary="The reward sale still needs an operator.",
            )

        self.assertEqual(summary, "A manual vault action remains.")
        request = create_response.await_args.kwargs
        self.assertEqual(request["model"], "gpt-5.6-terra")
        self.assertEqual(request["reasoning"], {"effort": "low"})

    async def test_telegram_handoff_reply_consumes_notice_and_posts_update(
        self,
    ) -> None:
        channel_id = 95
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
            followup_attachments=[
                {
                    "filename": "details.png",
                    "url": "https://cdn.example/details.png",
                    "content_type": "image/png",
                    "is_image": True,
                }
            ],
        )
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        channel_intent_after_button[channel_id] = "investigate_issue"
        edited_messages: list[tuple[str, int, str]] = []
        internal_turn_calls: list[dict[str, object]] = []

        update = {
            "update_id": 1,
            "message": {
                "message_id": 999,
                "chat": {"id": "123"},
                "text": "tell the user the tx is queued pending signatures",
                "reply_to_message": {"message_id": 456},
            },
        }

        async def fake_edit_handoff_notice(
            *, chat_id: str, message_id: int, message_text: str
        ) -> bool:
            edited_messages.append((chat_id, message_id, message_text))
            return True

        async def fake_internal_turn(**kwargs) -> InternalInstructionTurnResult:
            internal_turn_calls.append(
                {
                    "prompt_text": kwargs["prompt_text"],
                    "instruction_text": kwargs["instruction_text"],
                    "attachments": kwargs["attachments"],
                }
            )
            return InternalInstructionTurnResult(
                reply=(
                    "The transaction has been queued and is pending multisig "
                    "signatures."
                ),
                conversation_history=conversation_threads[channel_id]
                + [{"role": "assistant", "content": "Delivered update."}],
                input_history=conversation_threads[channel_id],
            )

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"):
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
                            await bot.telegram_handoffs._handle_telegram_handoff_update(
                                update
                            )
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
            self.assertEqual(len(internal_turn_calls), 1)
            self.assertEqual(
                internal_turn_calls[0]["prompt_text"],
                "tell the user the tx is queued pending signatures",
            )
            self.assertEqual(
                internal_turn_calls[0]["attachments"],
                [
                    {
                        "filename": "details.png",
                        "url": "https://cdn.example/details.png",
                        "content_type": "image/png",
                        "is_image": True,
                    }
                ],
            )
            self.assertIn(
                "This input is from the internal team",
                internal_turn_calls[0]["instruction_text"],
            )
            self.assertIn(
                "Write directly to the user, not back to the team",
                internal_turn_calls[0]["instruction_text"],
            )
            self.assertIn(
                "Expand shorthand like `pending sigs`",
                internal_turn_calls[0]["instruction_text"],
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

    async def test_handoff_send_and_poll_payloads_enable_ticket_dismissal_only(
        self,
    ) -> None:
        api_calls: list[tuple[str, dict[str, object]]] = []

        async def fake_telegram_api_call(method: str, payload: dict[str, object]):
            api_calls.append((method, payload))
            if method == "sendMessage":
                return {
                    "ok": True,
                    "result": {
                        "message_id": len(api_calls),
                        "chat": {"id": 123},
                    },
                }
            return {"ok": True, "result": []}

        with (
            patch.dict("os.environ", {"YSUPPORT_ALLOW_TEST_TELEGRAM": "1"}),
            patch.object(config, "TELEGRAM_BOT_TOKEN", "test-token"),
            patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
            patch("handoff._telegram_api_call", new=fake_telegram_api_call),
        ):
            await handoff.send_handoff_notice(
                "ticket handoff",
                dismiss_enabled=True,
            )
            await handoff.send_handoff_notice(
                "public alert",
                dismiss_enabled=False,
            )
            await handoff.fetch_telegram_updates(offset=42)
            await handoff.edit_handoff_notice(
                chat_id="123",
                message_id=7,
                message_text="closed",
            )
            await handoff.answer_telegram_callback_query(
                callback_query_id="callback-1",
                message_text="dismissed",
            )
            await handoff.delete_telegram_message(
                chat_id="123",
                message_id=7,
            )

        ticket_payload = api_calls[0][1]
        self.assertEqual(
            ticket_payload["reply_markup"],
            {
                "inline_keyboard": [
                    [
                        {
                            "text": "Dismiss and handle in Discord",
                            "callback_data": DISMISS_HANDOFF_CALLBACK_DATA,
                        }
                    ]
                ]
            },
        )
        self.assertNotIn("reply_markup", api_calls[1][1])
        self.assertEqual(api_calls[2][0], "getUpdates")
        self.assertEqual(
            api_calls[2][1]["allowed_updates"],
            ["message", "callback_query"],
        )
        self.assertEqual(api_calls[2][1]["offset"], 42)
        self.assertEqual(api_calls[3][0], "editMessageText")
        self.assertEqual(
            api_calls[3][1]["reply_markup"],
            {"inline_keyboard": []},
        )
        self.assertEqual(
            api_calls[4],
            (
                "answerCallbackQuery",
                {
                    "callback_query_id": "callback-1",
                    "text": "dismissed",
                },
            ),
        )
        self.assertEqual(
            api_calls[5],
            (
                "deleteMessage",
                {
                    "chat_id": "123",
                    "message_id": 7,
                },
            ),
        )

    async def test_retire_handoff_notice_falls_back_to_closed_status(self) -> None:
        with (
            patch(
                "handoff.delete_telegram_message",
                return_value=False,
            ) as delete_message,
            patch(
                "handoff.edit_handoff_notice",
                return_value=True,
            ) as edit_notice,
        ):
            retired = await handoff.retire_handoff_notice(
                chat_id="123",
                message_id=456,
                fallback_message_text="closed",
            )

        self.assertTrue(retired)
        delete_message.assert_awaited_once_with(
            chat_id="123",
            message_id=456,
        )
        edit_notice.assert_awaited_once_with(
            chat_id="123",
            message_id=456,
            message_text="closed",
        )
