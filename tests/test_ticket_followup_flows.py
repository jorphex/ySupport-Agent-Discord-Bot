import tests as _test_environment  # noqa: F401

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch


import discord

import config
from state import (
    active_ticket_executor_tasks,
    active_ticket_payloads,
    BotRunContext,
    TicketInvestigationJob,
    TeamHandoffNotice,
    clear_ticket_channel_state,
    clear_ticket_investigation_job,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    is_ticket_waiting_for_team,
    last_bot_reply_ts_by_channel,
    pending_attachments_by_channel,
    pending_messages,
    pending_tasks,
    team_handoff_notice_by_channel,
    ticket_owner_user_id_by_channel,
)
from handoff import (
    TelegramSentMessage,
)
from ticket_investigation.runtime import (
    TicketAgentFlowOutcome,
    TicketTurnRequest,
)
from ysupport import (
    TicketBot,
)
from tests.ticket_flow_test_support import (
    FakeInvestigationExecutor as _FakeInvestigationExecutor,
    TicketFlowTestCase,
)
from tests.test_ticket_intake import (
    _FakeDiscordChannel,
)


class TicketFlowTests(TicketFlowTestCase):
    async def test_process_ticket_message_formats_handoff_reply_and_notifies_telegram(
        self,
    ) -> None:
        channel_id = 93
        fake_channel = _FakeDiscordChannel(channel_id)
        handoff_notices: list[str] = []

        updated_job = TicketInvestigationJob(channel_id=channel_id)
        fake_executor = _FakeInvestigationExecutor(
            result=type(
                "_Result",
                (),
                {
                    "flow_outcome": TicketAgentFlowOutcome(
                        raw_final_reply=(
                            "This likely needs manual review. "
                            f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
                        ),
                        conversation_history=[
                            {
                                "role": "user",
                                "content": "PPS is flat and rewards need to be dumped.",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "This likely needs manual review. "
                                    f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
                                ),
                            },
                        ],
                        completed_agent_key=None,
                        requires_human_handoff=True,
                        handoff_reason=(
                            "manual strategy action is required to process rewards"
                        ),
                    ),
                    "updated_job": updated_job,
                },
            )()
        )
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = fake_executor

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="investigate_issue",
            conversation_owner_id=777,
        )

        pending_messages[channel_id] = "PPS is flat and rewards need to be dumped."
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        async def fake_send_handoff_notice(
            message_text: str,
            *,
            dismiss_enabled: bool = False,
        ) -> TelegramSentMessage:
            self.assertTrue(dismiss_enabled)
            handoff_notices.append(message_text)
            return TelegramSentMessage(
                chat_id="123",
                message_id=456,
                message_text=message_text,
            )

        try:
            with patch("ysupport.send_long_message", new=fake_send_long_message):
                with patch(
                    "discord_support_runtime.send_handoff_notice",
                    new=fake_send_handoff_notice,
                ):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)
        self.assertEqual(len(fake_channel.sent_messages), 1)
        self.assertIn("This likely needs manual review.", fake_channel.sent_messages[0])
        self.assertIn("I've notified the support team", fake_channel.sent_messages[0])
        self.assertNotIn(
            config.HUMAN_HANDOFF_TAG_PLACEHOLDER, fake_channel.sent_messages[0]
        )
        self.assertEqual(len(handoff_notices), 1)
        self.assertIn(
            "<b>Reason</b>: manual strategy action is required to process rewards",
            handoff_notices[0],
        )

    async def test_failed_ticket_handoff_notification_does_not_park_or_claim_success(
        self,
    ) -> None:
        channel_id = 193
        fake_channel = _FakeDiscordChannel(channel_id)
        updated_job = TicketInvestigationJob(channel_id=channel_id)
        updated_job.mark_escalated_to_human()
        fake_executor = _FakeInvestigationExecutor(
            result=type(
                "_Result",
                (),
                {
                    "flow_outcome": TicketAgentFlowOutcome(
                        raw_final_reply=(
                            "This needs manual review. "
                            f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
                        ),
                        conversation_history=[],
                        completed_agent_key=None,
                        requires_human_handoff=True,
                    ),
                    "updated_job": updated_job,
                },
            )()
        )
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = fake_executor
        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="investigate_issue",
            conversation_owner_id=777,
        )
        pending_messages[channel_id] = "manual review needed"

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with (
                patch("ysupport.send_long_message", new=fake_send_long_message),
                patch("discord_support_runtime.send_handoff_notice", return_value=None),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
            ):
                await bot.process_ticket_message(channel_id, run_context)

            self.assertFalse(is_ticket_waiting_for_team(channel_id))
            self.assertEqual(
                get_or_create_ticket_investigation_job(channel_id).mode,
                "waiting_for_user",
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertIn(
                "couldn't send the internal team notification",
                fake_channel.sent_messages[0],
            )
            self.assertNotIn("I've notified", fake_channel.sent_messages[0])
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_waiting_for_team_ticket_stores_followup_and_sends_parked_ack(
        self,
    ) -> None:
        channel_id = 94
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)
        owner = SimpleNamespace(id=777, bot=False, name="owner")
        message = SimpleNamespace(
            id=654,
            author=owner,
            content="here are more details",
            channel=fake_channel,
            reference=None,
            created_at=datetime.now(timezone.utc),
            attachments=[
                SimpleNamespace(
                    id=321,
                    filename="details.png",
                    url="https://cdn.example/details.png",
                    content_type="image/png",
                    size=123,
                )
            ],
        )

        bot = TicketBot(intents=discord.Intents.none())
        conversation_threads[channel_id] = [
            {"role": "assistant", "content": "Team notified."}
        ]
        ticket_owner_user_id_by_channel[channel_id] = 777
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
        )
        last_bot_reply_ts_by_channel.pop(channel_id, None)

        try:
            with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                with patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}):
                    await bot.on_message(message)
            self.assertTrue(is_ticket_waiting_for_team(channel_id))
            self.assertEqual(
                conversation_threads[channel_id][-1]["content"],
                "here are more details",
            )
            self.assertEqual(len(fake_channel.sent_messages), 1)
            self.assertIn(
                "The team has already been notified",
                fake_channel.sent_messages[0],
            )
            self.assertEqual(
                team_handoff_notice_by_channel[channel_id].followup_attachments,
                [
                    {
                        "filename": "details.png",
                        "url": "https://cdn.example/details.png",
                        "content_type": "image/png",
                        "size": 123,
                        "is_image": True,
                        "attachment_id": 321,
                        "source_message_id": 654,
                    }
                ],
            )
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_debounce_followup_is_silently_added_without_losing_payload(
        self,
    ) -> None:
        channel_id = 294
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)
        owner = SimpleNamespace(id=777, bot=False, name="owner")
        captured_requests: list[TicketTurnRequest] = []

        class _Executor:
            async def execute_turn(self, request: TicketTurnRequest, hooks=None):
                captured_requests.append(request)
                return SimpleNamespace(
                    flow_outcome=TicketAgentFlowOutcome(
                        raw_final_reply="Combined answer.",
                        conversation_history=[],
                        completed_agent_key="docs",
                        requires_human_handoff=False,
                    ),
                    updated_job=request.investigation_job,
                )

        def ticket_message(
            message_id: int,
            content: str,
            attachment: SimpleNamespace,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                author=owner,
                content=content,
                channel=fake_channel,
                reference=None,
                created_at=datetime.now(timezone.utc),
                attachments=[attachment],
            )

        first = ticket_message(
            1001,
            "The withdrawal is stuck.",
            SimpleNamespace(
                id=501,
                filename="first.png",
                url="https://cdn.discordapp.com/attachments/1/2/first.png",
                content_type="image/png",
                size=100,
            ),
        )
        followup = ticket_message(
            1002,
            "It is on Base.",
            SimpleNamespace(
                id=502,
                filename="second.png",
                url="https://cdn.discordapp.com/attachments/1/2/second.png",
                content_type="image/png",
                size=200,
            ),
        )

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _Executor()
        ticket_owner_user_id_by_channel[channel_id] = owner.id

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch.object(config, "COOLDOWN_SECONDS", 0.01),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
                patch("ysupport.send_long_message", new=fake_send_long_message),
            ):
                await bot.on_message(first)
                await bot.on_message(followup)
                await asyncio.wait_for(pending_tasks[channel_id], timeout=1)

            self.assertEqual(len(captured_requests), 1)
            self.assertEqual(
                captured_requests[0].aggregated_text,
                "The withdrawal is stuck.\nIt is on Base.",
            )
            self.assertEqual(
                [item["filename"] for item in captured_requests[0].attachments],
                ["first.png", "second.png"],
            )
            self.assertEqual(fake_channel.sent_messages, ["Combined answer."])
            self.assertNotIn(channel_id, pending_messages)
            self.assertNotIn(channel_id, pending_attachments_by_channel)
            self.assertNotIn(channel_id, active_ticket_payloads)
            self.assertNotIn(channel_id, active_ticket_executor_tasks)
            self.assertNotIn(channel_id, pending_tasks)
        finally:
            task = pending_tasks.pop(channel_id, None)
            if task is not None:
                task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_pre_executor_followup_restarts_silently_with_complete_payload(
        self,
    ) -> None:
        channel_id = 395
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)
        owner = SimpleNamespace(id=777, bot=False, name="owner")
        first_boundary_started = asyncio.Event()
        first_boundary_cancelled = asyncio.Event()
        boundary_calls = 0
        captured_requests: list[TicketTurnRequest] = []

        async def fake_boundary(_text: str):
            nonlocal boundary_calls
            boundary_calls += 1
            if boundary_calls == 1:
                first_boundary_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_boundary_cancelled.set()
                    raise
            return {
                "classification": "yearn_support",
                "tripwire_triggered": False,
            }

        class _Executor:
            async def execute_turn(self, request: TicketTurnRequest, hooks=None):
                del hooks
                captured_requests.append(request)
                return SimpleNamespace(
                    flow_outcome=TicketAgentFlowOutcome(
                        raw_final_reply="Updated answer.",
                        conversation_history=[],
                        completed_agent_key="docs",
                        requires_human_handoff=False,
                    ),
                    updated_job=request.investigation_job,
                )

        def ticket_message(message_id: int, content: str) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                author=owner,
                content=content,
                channel=fake_channel,
                reference=None,
                created_at=datetime.now(timezone.utc),
                attachments=[],
            )

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _Executor()
        ticket_owner_user_id_by_channel[channel_id] = owner.id

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch.object(config, "COOLDOWN_SECONDS", 0),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
                patch("ysupport.send_long_message", new=fake_send_long_message),
                patch("ysupport._outer_support_boundary_result", new=fake_boundary),
            ):
                await bot.on_message(ticket_message(3001, "Initial vault issue."))
                await asyncio.wait_for(first_boundary_started.wait(), timeout=1)
                self.assertNotIn(channel_id, active_ticket_executor_tasks)

                await bot.on_message(ticket_message(3002, "The tx hash is 0xabc."))
                restarted_task = pending_tasks[channel_id]
                await asyncio.wait_for(first_boundary_cancelled.wait(), timeout=1)
                await asyncio.wait_for(restarted_task, timeout=1)

            self.assertEqual(boundary_calls, 2)
            self.assertEqual(len(captured_requests), 1)
            self.assertEqual(
                captured_requests[0].aggregated_text,
                "Initial vault issue.\nThe tx hash is 0xabc.",
            )
            self.assertEqual(fake_channel.sent_messages, ["Updated answer."])
            self.assertNotIn(channel_id, pending_messages)
            self.assertNotIn(channel_id, active_ticket_payloads)
            self.assertNotIn(channel_id, active_ticket_executor_tasks)
            self.assertNotIn(channel_id, pending_tasks)
        finally:
            task = pending_tasks.pop(channel_id, None)
            if task is not None:
                task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_active_followup_restarts_with_complete_payload(
        self,
    ) -> None:
        channel_id = 394
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)
        owner = SimpleNamespace(id=777, bot=False, name="owner")
        first_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        second_completed = asyncio.Event()
        captured_requests: list[TicketTurnRequest] = []

        class _Executor:
            async def execute_turn(
                _executor_self,
                request: TicketTurnRequest,
                hooks=None,
            ):
                del _executor_self, hooks
                captured_requests.append(request)
                self.assertIs(
                    active_ticket_executor_tasks.get(channel_id),
                    asyncio.current_task(),
                )
                if len(captured_requests) == 1:
                    first_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        first_cancelled.set()
                        raise
                second_completed.set()
                return SimpleNamespace(
                    flow_outcome=TicketAgentFlowOutcome(
                        raw_final_reply="Updated answer.",
                        conversation_history=[],
                        completed_agent_key="docs",
                        requires_human_handoff=False,
                    ),
                    updated_job=request.investigation_job,
                )

        def ticket_message(message_id: int, content: str) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                author=owner,
                content=content,
                channel=fake_channel,
                reference=None,
                created_at=datetime.now(timezone.utc),
                attachments=[],
            )

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _Executor()
        ticket_owner_user_id_by_channel[channel_id] = owner.id

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch.object(config, "COOLDOWN_SECONDS", 0),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
                patch("ysupport.send_long_message", new=fake_send_long_message),
            ):
                await bot.on_message(ticket_message(2001, "Initial vault issue."))
                await asyncio.wait_for(first_started.wait(), timeout=1)
                await bot.on_message(ticket_message(2002, "The tx hash is 0xabc."))
                await asyncio.wait_for(first_cancelled.wait(), timeout=1)
                await asyncio.wait_for(second_completed.wait(), timeout=1)
                while channel_id in pending_tasks:
                    await asyncio.sleep(0)

            self.assertEqual(len(captured_requests), 2)
            self.assertEqual(
                captured_requests[1].aggregated_text,
                "Initial vault issue.\nThe tx hash is 0xabc.",
            )
            self.assertEqual(
                fake_channel.sent_messages,
                [
                    "Got it. I’ve added your follow-up to your previous request for context, "
                    "and I’m continuing to work on it now. Please wait for my response. "
                    "There’s no need to resend anything.",
                    "Updated answer.",
                ],
            )
            self.assertNotIn(channel_id, pending_messages)
            self.assertNotIn(channel_id, active_ticket_payloads)
            self.assertNotIn(channel_id, active_ticket_executor_tasks)
            self.assertNotIn(channel_id, pending_tasks)
        finally:
            task = pending_tasks.pop(channel_id, None)
            if task is not None:
                task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_active_followups_cannot_race_while_acknowledgement_sends(
        self,
    ) -> None:
        channel_id = 396
        acknowledgement_started = asyncio.Event()
        release_acknowledgement = asyncio.Event()

        class _BlockingAcknowledgementChannel(_FakeDiscordChannel):
            async def send(self, message: str, *args, **kwargs):
                del args, kwargs
                self.sent_messages.append(message)
                if message.startswith("Got it."):
                    acknowledgement_started.set()
                    await release_acknowledgement.wait()

        channel = _BlockingAcknowledgementChannel(channel_id)
        channel.category = SimpleNamespace(id=1)
        channel.guild = SimpleNamespace(id=2)
        owner = SimpleNamespace(id=777, bot=False, name="owner")

        def ticket_message(
            message_id: int,
            content: str,
            filename: str,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                author=owner,
                content=content,
                channel=channel,
                reference=None,
                created_at=datetime.now(timezone.utc),
                attachments=[
                    SimpleNamespace(
                        id=message_id,
                        filename=filename,
                        url=(f"https://cdn.discordapp.com/attachments/1/2/{filename}"),
                        content_type="image/png",
                        size=100,
                    )
                ],
            )

        async def active_turn() -> None:
            await asyncio.Event().wait()

        active_task = asyncio.create_task(active_turn())
        pending_tasks[channel_id] = active_task
        active_ticket_executor_tasks[channel_id] = active_task
        active_ticket_payloads[channel_id] = (
            active_task,
            "Initial vault issue.",
            [
                {
                    "filename": "initial.png",
                    "url": ("https://cdn.discordapp.com/attachments/1/2/initial.png"),
                }
            ],
        )
        ticket_owner_user_id_by_channel[channel_id] = owner.id
        bot = TicketBot(intents=discord.Intents.none())

        first_handler: asyncio.Task | None = None
        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch.object(config, "COOLDOWN_SECONDS", 60),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
            ):
                first_handler = asyncio.create_task(
                    bot.on_message(ticket_message(4001, "It is on Base.", "base.png"))
                )
                await asyncio.wait_for(acknowledgement_started.wait(), timeout=1)
                first_replacement = pending_tasks[channel_id]

                await asyncio.wait_for(
                    bot.on_message(
                        ticket_message(
                            4002,
                            "The tx hash is 0xabc.",
                            "transaction.png",
                        )
                    ),
                    timeout=1,
                )
                final_replacement = pending_tasks[channel_id]

                self.assertIsNot(first_replacement, final_replacement)
                await asyncio.sleep(0)
                self.assertTrue(first_replacement.done())
                self.assertEqual(
                    pending_messages[channel_id],
                    "Initial vault issue.\nIt is on Base.\nThe tx hash is 0xabc.",
                )
                self.assertEqual(
                    [
                        attachment["filename"]
                        for attachment in pending_attachments_by_channel[channel_id]
                    ],
                    ["initial.png", "base.png", "transaction.png"],
                )
                self.assertEqual(
                    channel.sent_messages,
                    [
                        "Got it. I’ve added your follow-up to your previous request for context, "
                        "and I’m continuing to work on it now. Please wait for my response. "
                        "There’s no need to resend anything."
                    ],
                )

                release_acknowledgement.set()
                await asyncio.wait_for(first_handler, timeout=1)

            with self.assertRaises(asyncio.CancelledError):
                await active_task
        finally:
            release_acknowledgement.set()
            if first_handler is not None and not first_handler.done():
                first_handler.cancel()
            task = pending_tasks.pop(channel_id, None)
            if task is not None:
                task.cancel()
            if not active_task.done():
                active_task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)
