import tests as _test_environment  # noqa: F401

import asyncio
import unittest
from dataclasses import dataclass
from unittest.mock import patch


import discord

import config
import handoff
from state import (
    TeamHandoffNotice,
    clear_ticket_channel_state,
    team_handoff_notice_by_channel,
)
from handoff import (
    build_failed_delivery_handoff_notice,
    build_handoff_notice,
    TelegramApiError,
)
from support_agents import (
    TicketTriageDecision,
)
from ysupport import (
    TicketBot,
)


@dataclass
class _FakeResult:
    final_output: str | None
    last_agent: object | None
    _history: list
    _decision: TicketTriageDecision | None = None

    def final_output_as(self, model_type):
        if self._decision is None:
            raise AssertionError("No structured decision configured.")
        return self._decision

    def to_input_list(self):
        return self._history


class _FakeRunner:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if not self._results:
            raise AssertionError("No fake result available for runner call.")
        return self._results.pop(0)


class _FakeInvestigationExecutor:
    def __init__(self, *, result=None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.calls = []

    async def execute_turn(self, request, hooks=None):
        self.calls.append({"request": request, "hooks": hooks})
        if self.exc is not None:
            raise self.exc
        return self.result


class TicketFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        boundary_patcher = patch(
            "ysupport._outer_support_boundary_result",
            return_value={
                "classification": "yearn_support",
                "tripwire_triggered": False,
            },
        )
        boundary_patcher.start()
        self.addCleanup(boundary_patcher.stop)
        runtime_boundary_patcher = patch(
            "ticket_investigation.runtime.evaluate_support_boundary",
            return_value={
                "classification": "yearn_support",
                "tripwire_triggered": False,
            },
        )
        runtime_boundary_patcher.start()
        self.addCleanup(runtime_boundary_patcher.stop)
        summary_patcher = patch(
            "discord_support_runtime.summarize_handoff_summary",
            return_value=None,
        )
        summary_patcher.start()
        self.addCleanup(summary_patcher.stop)

    async def test_telegram_update_offset_persists_only_after_update_handling(
        self,
    ) -> None:
        bot = TicketBot(intents=discord.Intents.none())
        bot.telegram_handoffs._telegram_update_offset = None
        handled_updates: list[dict[str, object]] = []
        persisted_offsets: list[int] = []
        sleep_calls: list[float] = []

        update = {"update_id": 41, "message": {"text": "noop"}}
        call_count = {"fetch": 0}

        async def fake_fetch_telegram_updates(offset):
            call_count["fetch"] += 1
            if call_count["fetch"] == 1:
                return [update]
            raise asyncio.CancelledError

        async def fake_handle_success(payload):
            handled_updates.append(payload)

        async def fake_handle_failure(_payload):
            raise RuntimeError("boom")

        def fake_persist_telegram_update_offset(offset: int) -> None:
            persisted_offsets.append(offset)

        try:
            with patch(
                "telegram_handoff_controller.fetch_telegram_updates",
                new=fake_fetch_telegram_updates,
            ):
                with patch(
                    "telegram_handoff_controller.persist_telegram_update_offset",
                    new=fake_persist_telegram_update_offset,
                ):
                    with patch.object(
                        bot.telegram_handoffs,
                        "_handle_telegram_handoff_update",
                        new=fake_handle_success,
                    ):
                        with self.assertRaises(asyncio.CancelledError):
                            await bot.telegram_handoffs._telegram_handoff_reply_loop()
            self.assertEqual(handled_updates, [update])
            self.assertEqual(persisted_offsets, [42])
            self.assertEqual(bot.telegram_handoffs._telegram_update_offset, 42)

            bot.telegram_handoffs._telegram_update_offset = None
            persisted_offsets.clear()
            call_count["fetch"] = 0

            async def fake_sleep(delay: float) -> None:
                sleep_calls.append(delay)
                raise asyncio.CancelledError

            with patch(
                "telegram_handoff_controller.fetch_telegram_updates",
                new=fake_fetch_telegram_updates,
            ):
                with patch(
                    "telegram_handoff_controller.persist_telegram_update_offset",
                    new=fake_persist_telegram_update_offset,
                ):
                    with patch.object(
                        bot.telegram_handoffs,
                        "_handle_telegram_handoff_update",
                        new=fake_handle_failure,
                    ):
                        with patch(
                            "telegram_handoff_controller.asyncio.sleep", new=fake_sleep
                        ):
                            with self.assertRaises(asyncio.CancelledError):
                                await (
                                    bot.telegram_handoffs._telegram_handoff_reply_loop()
                                )
            self.assertEqual(persisted_offsets, [])
            self.assertIsNone(bot.telegram_handoffs._telegram_update_offset)
            self.assertEqual(sleep_calls, [5])
        finally:
            await bot.close()

    async def test_telegram_loop_checks_for_new_pending_state_each_poll(
        self,
    ) -> None:
        bot = TicketBot(intents=discord.Intents.none())
        recovery_calls = 0
        fetch_calls = 0

        async def fake_recover_pending() -> None:
            nonlocal recovery_calls
            recovery_calls += 1

        async def fake_fetch_updates(_offset):
            nonlocal fetch_calls
            fetch_calls += 1
            if fetch_calls == 1:
                return []
            raise asyncio.CancelledError

        try:
            with (
                patch.object(
                    bot.telegram_handoffs,
                    "_resume_pending_telegram_handoff_replies",
                    new=fake_recover_pending,
                ),
                patch(
                    "telegram_handoff_controller.fetch_telegram_updates",
                    new=fake_fetch_updates,
                ),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await bot.telegram_handoffs._telegram_handoff_reply_loop()

            self.assertEqual(recovery_calls, 2)
            self.assertEqual(fetch_calls, 2)
        finally:
            await bot.close()

    async def test_pending_telegram_recovery_is_bounded_per_notice_state(
        self,
    ) -> None:
        channel_id = 298
        original_notice = build_handoff_notice(
            reason="manual follow-up needed",
            summary="initial issue",
            channel_id=channel_id,
            guild_id=2,
        )
        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text=original_notice,
        )
        team_handoff_notice_by_channel[channel_id] = notice
        bot = TicketBot(intents=discord.Intents.none())
        delivery_calls = 0
        failed_status_edits: list[tuple[str, int, str]] = []

        async def fail_delivery(**_kwargs) -> bool:
            nonlocal delivery_calls
            delivery_calls += 1
            return False

        async def edit_failed_status(
            *,
            chat_id: str,
            message_id: int,
            message_text: str,
        ) -> bool:
            failed_status_edits.append((chat_id, message_id, message_text))
            return True

        try:
            with (
                patch.object(
                    bot.telegram_handoffs,
                    "_deliver_telegram_handoff_reply",
                    new=fail_delivery,
                ),
                patch(
                    "telegram_handoff_controller.edit_handoff_notice",
                    new=edit_failed_status,
                ),
            ):
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                self.assertEqual(delivery_calls, 0)

                notice.status = "pending_delivery"
                notice.pending_reply_text = "Tell the user the transaction is queued."
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                self.assertEqual(delivery_calls, 1)

                notice.telegram_message_id = 457
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                self.assertEqual(delivery_calls, 2)
                self.assertEqual(
                    failed_status_edits,
                    [
                        (
                            "123",
                            456,
                            build_failed_delivery_handoff_notice(original_notice),
                        ),
                        (
                            "123",
                            457,
                            build_failed_delivery_handoff_notice(original_notice),
                        ),
                    ],
                )
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)
            await bot.close()

    async def test_telegram_api_transport_failure_raises_typed_error(self) -> None:
        with (
            patch.object(config, "TELEGRAM_BOT_TOKEN", "test-token"),
            patch(
                "handoff.request.urlopen",
                side_effect=TimeoutError("read timed out"),
            ),
        ):
            with self.assertRaisesRegex(
                TelegramApiError,
                "Telegram API call getUpdates failed: read timed out",
            ):
                await handoff._telegram_api_call("getUpdates", {"timeout": 25})

    async def test_telegram_send_failures_keep_delivery_contracts(self) -> None:
        failure = TelegramApiError("Telegram API call failed")
        with (
            patch.dict(
                "os.environ",
                {"YSUPPORT_ALLOW_TEST_TELEGRAM": "1"},
            ),
            patch.object(config, "TELEGRAM_BOT_TOKEN", "test-token"),
            patch("handoff._telegram_api_call", side_effect=failure),
            patch("handoff.logging.error") as mock_error,
            patch("handoff.logging.warning") as mock_warning,
        ):
            sent = await handoff.send_telegram_message(
                chat_id="123",
                message_text="handoff",
            )
            edited = await handoff.edit_handoff_notice(
                chat_id="123",
                message_id=456,
                message_text="updated",
            )
            callback_answered = await handoff.answer_telegram_callback_query(
                callback_query_id="callback-1",
                message_text="dismissed",
            )
            deleted = await handoff.delete_telegram_message(
                chat_id="123",
                message_id=456,
            )

        self.assertIsNone(sent)
        self.assertFalse(edited)
        self.assertFalse(callback_answered)
        self.assertFalse(deleted)
        self.assertEqual(mock_error.call_count, 2)
        self.assertEqual(mock_warning.call_count, 2)

    async def test_telegram_polling_failure_warns_and_backs_off(self) -> None:
        bot = TicketBot(intents=discord.Intents.none())
        sleep_calls: list[float] = []

        async def fake_fetch_telegram_updates(_offset):
            raise TelegramApiError(
                "Telegram API call getUpdates failed: read timed out"
            )

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            raise asyncio.CancelledError

        try:
            with (
                patch(
                    "telegram_handoff_controller.fetch_telegram_updates",
                    new=fake_fetch_telegram_updates,
                ),
                patch("ysupport.asyncio.sleep", new=fake_sleep),
                patch("ysupport.logging.warning") as mock_warning,
                patch("ysupport.logging.error") as mock_error,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await bot.telegram_handoffs._telegram_handoff_reply_loop()

            self.assertEqual(sleep_calls, [5])
            mock_warning.assert_called_once()
            self.assertIn(
                "Telegram polling temporarily unavailable",
                mock_warning.call_args.args[0],
            )
            mock_error.assert_not_called()
        finally:
            await bot.close()
