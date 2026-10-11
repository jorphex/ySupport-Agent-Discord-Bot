import tests as _test_environment  # noqa: F401

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch


import discord

import config
import state
from state import (
    active_ticket_executor_tasks,
    active_ticket_payloads,
    BotRunContext,
    TeamHandoffNotice,
    clear_ticket_channel_state,
    conversation_threads,
    pending_attachments_by_channel,
    pending_messages,
    pending_tasks,
    persist_ticket_state,
    stopped_channels,
    stop_reasons_by_channel,
    team_handoff_notice_by_channel,
    ticket_investigation_jobs,
    ticket_owner_user_id_by_channel,
)
from ticket_investigation.runtime import (
    TicketAgentFlowOutcome,
    TicketTurnRequest,
)
from views import StopBotView
from discord_support_runtime import (
    _build_staff_summon_history,
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
    async def test_staff_summon_preserves_handoff_history_and_concurrent_followups(
        self,
    ) -> None:
        channel_id = 219
        channel = _FakeDiscordChannel(channel_id)
        channel.category = SimpleNamespace(id=1)
        owner_id = 777
        ticket_owner_user_id_by_channel[channel_id] = owner_id
        message = SimpleNamespace(
            id=220,
            channel=channel,
            author=SimpleNamespace(
                id=888,
                bot=False,
                name="contributor",
                guild_permissions=SimpleNamespace(administrator=True),
            ),
            content="y: Tell the user we are reviewing this",
            reference=None,
            attachments=[
                SimpleNamespace(
                    filename="staff.png",
                    url="https://cdn.example/staff.png",
                    content_type="image/png",
                    size=123,
                )
            ],
        )
        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="Team review needed.",
        )
        team_handoff_notice_by_channel[channel_id] = notice
        state.mark_ticket_channel_stopped(channel_id, reason="staff_takeover")
        older_detail = {
            "role": "user",
            "content": "The original error was a reverted withdrawal.",
        }
        recent_history = [{"role": "user", "content": "Please check again."}]
        conversation_threads[channel_id] = [older_detail] + recent_history
        during_generation = "Correction: this happened on Arbitrum."
        during_delivery = "The same error affects my second wallet."
        staff_reply = "An operator is reviewing this."
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _: channel
        requests = []

        async def owner_followup(text):
            await bot.on_message(
                SimpleNamespace(
                    id=221,
                    channel=channel,
                    reference=None,
                    content=text,
                    author=SimpleNamespace(id=owner_id, bot=False, name="owner"),
                    attachments=[],
                )
            )

        class _Executor:
            async def execute_turn(self, request, hooks=None):
                requests.append(request)
                if len(requests) == 1:
                    await owner_followup(during_generation)
                return SimpleNamespace(
                    flow_outcome=TicketAgentFlowOutcome(
                        raw_final_reply=staff_reply,
                        conversation_history=request.current_history
                        + [{"role": "assistant", "content": staff_reply}],
                        completed_agent_key=None,
                        requires_human_handoff=False,
                    ),
                    updated_job=request.investigation_job,
                )

        async def send_staff_reply(target, text, **kwargs):
            await target.send(text, **kwargs)
            await owner_followup(during_delivery)

        bot.investigation_executor = _Executor()
        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
                patch("ysupport._build_staff_summon_history", return_value=recent_history),
                patch("ysupport.send_long_message", new=send_staff_reply),
            ):
                await bot.on_message(message)
                await pending_tasks[channel_id]
            history = (
                [older_detail]
                + recent_history
                + [
                    {"role": "user", "content": during_generation},
                    {"role": "user", "content": during_delivery},
                    {"role": "assistant", "content": staff_reply},
                ]
            )
            self.assertEqual(channel.sent_messages, [staff_reply])
            self.assertIs(team_handoff_notice_by_channel[channel_id], notice)
            self.assertEqual(conversation_threads[channel_id], history)
            payload = state._read_json(state._TICKET_STATE_DIR / f"{channel_id}.json")
            self.assertEqual(payload["history"], history)
            self.assertTrue(payload["stopped"])
            self.assertEqual(payload["stop_reason"], "staff_takeover")
            self.assertEqual(payload["team_handoff_notice"]["status"], "open")
            self.assertEqual(
                [item["url"] for item in notice.followup_attachments],
                ["https://cdn.example/staff.png"],
            )
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch("telegram_handoff_controller.edit_handoff_notice", return_value=True),
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(
                    {
                        "message": {
                            "message_id": 999,
                            "chat": {"id": "123"},
                            "text": "Tell the user an operator is reviewing this.",
                            "reply_to_message": {"message_id": 456},
                        }
                    }
                )
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[1].current_history, history)
            self.assertEqual(requests[1].attachments, notice.followup_attachments)
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "staff_takeover")
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_stopped_staff_summon_failure_stays_in_discord(self) -> None:
        channel_id = 207
        channel = _FakeDiscordChannel(channel_id)
        message = SimpleNamespace(
            id=208,
            channel=channel,
            author=SimpleNamespace(name="contributor"),
            attachments=[],
        )
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _FakeInvestigationExecutor(
            exc=RuntimeError("override failed")
        )
        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="other_free_form",
        )

        async def fake_send_long_message(target_channel, message_text, **kwargs):
            await target_channel.send(message_text, **kwargs)

        try:
            with (
                patch("ysupport.send_long_message", new=fake_send_long_message),
                patch("ysupport._notify_handoff") as mock_notify,
                patch(
                    "ysupport._build_staff_summon_history",
                    return_value=[],
                ),
            ):
                stopped_channels.add(channel_id)
                await bot._handle_ticket_staff_summon(
                    message,
                    run_context,
                    "check the ticket",
                    was_stopped=True,
                )

            mock_notify.assert_not_called()
            self.assertEqual(len(channel.sent_messages), 1)
            self.assertIn("couldn't complete", channel.sent_messages[0])
            self.assertIn(
                "remains under manual staff control", channel.sent_messages[0]
            )
            self.assertNotIn(channel_id, active_ticket_executor_tasks)
            self.assertNotIn("notified", channel.sent_messages[0])
        finally:
            clear_ticket_channel_state(
                channel_id,
                keep_stopped=False,
                delete_persisted=True,
            )

    async def test_staff_summon_history_preserves_ticket_roles(self) -> None:
        owner_id = 701
        bot_id = 702
        contributor_role = SimpleNamespace(id=config.TICKET_CONTRIBUTOR_ROLE_ID)

        def author(
            user_id: int,
            *,
            bot: bool = False,
            staff: bool = False,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                id=user_id,
                bot=bot,
                roles=[contributor_role] if staff else [],
                guild_permissions=SimpleNamespace(administrator=False),
            )

        def history_message(
            message_id: int,
            message_author: SimpleNamespace,
            content: str,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                author=message_author,
                content=content,
                attachments=[],
            )

        chronological_messages = [
            history_message(1, author(owner_id), "My transaction is pending."),
            history_message(2, author(bot_id, bot=True), "Share the public hash."),
            history_message(3, author(703, staff=True), "The wallet needs more gas."),
            history_message(4, author(704), "Unrelated spectator message."),
            history_message(5, author(705, bot=True), "Ticket Tool opener."),
            history_message(6, author(703, staff=True), "y: explain the gas issue"),
        ]

        class _HistoryChannel:
            async def history(self, *, limit: int):
                del limit
                for item in reversed(chronological_messages):
                    yield item

        history = await _build_staff_summon_history(
            _HistoryChannel(),
            exclude_message_id=6,
            ticket_owner_user_id=owner_id,
            bot_user_id=bot_id,
        )

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "My transaction is pending."},
                {"role": "assistant", "content": "Share the public hash."},
                {
                    "role": "system",
                    "content": (
                        "Internal support staff message: The wallet needs more gas."
                    ),
                },
            ],
        )

    async def test_staff_takeover_preserves_handoff_for_later_telegram_reply(self) -> None:
        channel_id = 209
        channel = _FakeDiscordChannel(channel_id)
        channel.category = SimpleNamespace(id=1)
        channel.guild = SimpleNamespace(id=2)
        owner_id = 777
        staff = SimpleNamespace(
            id=888,
            bot=False,
            name="contributor",
            roles=[SimpleNamespace(id=config.TICKET_CONTRIBUTOR_ROLE_ID)],
            guild_permissions=SimpleNamespace(administrator=False),
        )
        message = SimpleNamespace(
            id=210,
            author=staff,
            content="Thanks, will take a look.",
            channel=channel,
            reference=None,
            created_at=datetime.now(timezone.utc),
            attachments=[],
        )

        async def active_turn() -> None:
            await asyncio.Event().wait()

        active_task = asyncio.create_task(active_turn())
        pending_tasks[channel_id] = active_task
        active_ticket_executor_tasks[channel_id] = active_task
        pending_messages[channel_id] = "The queue also contains another detail."
        ticket_owner_user_id_by_channel[channel_id] = owner_id
        conversation_threads[channel_id] = [
            {"role": "user", "content": "My withdrawal failed."}
        ]
        job = state.get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        job.remember_chain("polygon")
        attachments = [
            {
                "filename": "error.png",
                "url": "https://cdn.example/error.png",
                "is_image": True,
            }
        ]
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text="Team review needed.",
            followup_attachments=list(attachments),
        )
        staff_image_message = SimpleNamespace(
            **{
                **vars(message),
                "id": 211,
                "content": "",
                "attachments": [
                    SimpleNamespace(
                        filename="staff-error.png",
                        url="https://cdn.example/staff-error.png",
                        content_type="image/png",
                        size=456,
                    )
                ],
            }
        )
        attachments.append(
            {
                "filename": "staff-error.png",
                "url": "https://cdn.example/staff-error.png",
                "content_type": "image/png",
                "size": 456,
                "is_image": True,
                "source_message_id": 211,
            }
        )
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: channel
        requests = []

        class _Executor:
            async def execute_turn(self, request, hooks=None):
                requests.append(request)
                return SimpleNamespace(
                    flow_outcome=TicketAgentFlowOutcome(
                        raw_final_reply="The withdrawal is queued for signatures.",
                        conversation_history=request.current_history
                        + [
                            {
                                "role": "assistant",
                                "content": "The withdrawal is queued for signatures.",
                            }
                        ],
                        completed_agent_key=None,
                        requires_human_handoff=False,
                    ),
                    updated_job=request.investigation_job,
                )

        bot.investigation_executor = _Executor()

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
                patch("views.retire_handoff_notice", return_value=True) as retire,
            ):
                await bot.on_message(message)
                await bot.on_message(staff_image_message)

            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "staff_takeover")
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], owner_id)
            self.assertNotIn(channel_id, pending_tasks)
            self.assertNotIn(channel_id, active_ticket_executor_tasks)
            self.assertEqual(team_handoff_notice_by_channel[channel_id].status, "open")
            self.assertIs(ticket_investigation_jobs[channel_id], job)
            self.assertEqual(channel.sent_messages, [])
            retire.assert_not_awaited()
            with self.assertRaises(asyncio.CancelledError):
                await active_task

            owner_message = SimpleNamespace(
                **{
                    **vars(message),
                    "author": SimpleNamespace(id=owner_id, bot=False, name="owner"),
                    "content": "The error persists today.",
                }
            )
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
            ):
                await bot.on_message(owner_message)
            self.assertEqual(channel.sent_messages, [])
            self.assertEqual(requests, [])

            # Simulate a process restart without overwriting the durable snapshot.
            with patch("state.persist_ticket_state", return_value=True):
                clear_ticket_channel_state(channel_id)
            state.hydrate_ticket_state(channel_id)
            self.assertEqual(stop_reasons_by_channel[channel_id], "staff_takeover")
            self.assertEqual(
                ticket_investigation_jobs[channel_id].evidence.chain, "polygon"
            )
            self.assertEqual(
                team_handoff_notice_by_channel[channel_id].followup_attachments, attachments
            )

            update = {
                "message": {
                    "message_id": 999,
                    "chat": {"id": "123"},
                    "text": "Tell the user the withdrawal is queued for signatures.",
                    "reply_to_message": {"message_id": 456},
                }
            }
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch("telegram_handoff_controller.edit_handoff_notice", return_value=True),
            ):
                await bot.telegram_handoffs._handle_telegram_handoff_update(update)
                await bot.telegram_handoffs._handle_telegram_handoff_update(update)

            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].turn_source, "internal_team")
            self.assertEqual(requests[0].attachments, attachments)
            self.assertIn(
                {"role": "user", "content": "The queue also contains another detail."},
                requests[0].current_history,
            )
            self.assertIn(
                {
                    "role": "system",
                    "content": "Internal support staff message: Thanks, will take a look.",
                },
                requests[0].current_history,
            )
            self.assertIn(
                {"role": "user", "content": "The error persists today."},
                requests[0].current_history,
            )
            self.assertEqual(
                channel.sent_messages, ["The withdrawal is queued for signatures."]
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            payload = state._read_json(state._TICKET_STATE_DIR / f"{channel_id}.json")
            self.assertTrue(payload["stopped"])
            self.assertEqual(payload["stop_reason"], "staff_takeover")
            self.assertIsNone(payload["team_handoff_notice"])
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
            ):
                await bot.on_message(owner_message)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(channel.sent_messages), 1)
        finally:
            pending_tasks.pop(channel_id, None)
            if not active_task.done():
                active_task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_empty_staff_summon_shows_usage_without_changing_state(self) -> None:
        channel_id = 216
        channel = _FakeDiscordChannel(channel_id)
        channel.category = SimpleNamespace(id=1)
        channel.guild = SimpleNamespace(id=2)
        staff = SimpleNamespace(
            id=888,
            bot=False,
            name="contributor",
            roles=[SimpleNamespace(id=config.TICKET_CONTRIBUTOR_ROLE_ID)],
            guild_permissions=SimpleNamespace(administrator=False),
        )
        message = SimpleNamespace(
            id=217,
            author=staff,
            content="Y:   ",
            channel=channel,
            reference=None,
            created_at=datetime.now(timezone.utc),
            attachments=[],
        )
        ticket_owner_user_id_by_channel[channel_id] = 777
        bot = TicketBot(intents=discord.Intents.none())

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
            ):
                await bot.on_message(message)

            self.assertNotIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, pending_tasks)
            self.assertEqual(
                channel.sent_messages,
                ["Add an instruction after `y:` to ask ySupport to reply."],
            )
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_active_admin_summon_cancels_older_run_and_stays_active(self) -> None:
        channel_id = 211
        owner_id = 777
        admin = SimpleNamespace(
            id=999,
            bot=False,
            name="administrator",
            roles=[],
            guild_permissions=SimpleNamespace(administrator=True),
        )
        owner = SimpleNamespace(
            id=owner_id,
            bot=False,
            name="owner",
            roles=[],
            guild_permissions=SimpleNamespace(administrator=False),
        )
        summon_attachment = SimpleNamespace(
            id=321,
            filename="staff-evidence.png",
            url="https://cdn.discordapp.com/attachments/1/2/staff-evidence.png",
            content_type="image/png",
            size=123,
        )

        class _HistoryChannel(_FakeDiscordChannel):
            def __init__(self) -> None:
                super().__init__(channel_id)
                self.category = SimpleNamespace(id=1)
                self.guild = SimpleNamespace(id=2)
                self.history_messages: list[SimpleNamespace] = []

            async def history(self, *, limit: int):
                del limit
                for item in reversed(self.history_messages):
                    yield item

        channel = _HistoryChannel()
        owner_message = SimpleNamespace(
            id=212,
            author=owner,
            content="Why is my transaction still pending?",
            attachments=[],
        )
        summon_message = SimpleNamespace(
            id=213,
            author=admin,
            content="Y: explain the gas requirement clearly",
            channel=channel,
            reference=None,
            created_at=datetime.now(timezone.utc),
            attachments=[summon_attachment],
        )
        channel.history_messages = [owner_message, summon_message]

        captured_requests: list[TicketTurnRequest] = []

        class _Executor:
            async def execute_turn(self, request: TicketTurnRequest, hooks=None):
                del hooks
                captured_requests.append(request)
                return SimpleNamespace(
                    flow_outcome=TicketAgentFlowOutcome(
                        raw_final_reply="Add enough ETH to cover the transaction fee.",
                        conversation_history=request.current_history
                        + [
                            {
                                "role": "assistant",
                                "content": (
                                    "Add enough ETH to cover the transaction fee."
                                ),
                            }
                        ],
                        completed_agent_key=None,
                        requires_human_handoff=False,
                    ),
                    updated_job=request.investigation_job,
                )

        async def older_turn() -> None:
            await asyncio.Event().wait()

        old_task = asyncio.create_task(older_turn())
        pending_tasks[channel_id] = old_task
        pending_messages[channel_id] = "Why is my transaction still pending?"
        pending_attachments_by_channel[channel_id] = [
            {"url": "https://cdn.discordapp.com/attachments/1/2/owner.png"}
        ]
        active_ticket_payloads[channel_id] = (
            old_task,
            "Why is my transaction still pending?",
            [{"url": "https://cdn.discordapp.com/attachments/1/2/owner.png"}],
        )
        ticket_owner_user_id_by_channel[channel_id] = owner_id
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _Executor()

        sent_views: list[object | None] = []

        async def fake_send_long_message(target, text, **kwargs):
            sent_views.append(kwargs.get("view"))
            await target.send(text, **kwargs)

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _HistoryChannel),
                patch("ysupport.send_long_message", new=fake_send_long_message),
                patch("ysupport._notify_handoff") as notify,
            ):
                await bot.on_message(summon_message)
                summon_task = pending_tasks[channel_id]
                await asyncio.wait_for(summon_task, timeout=1)

            with self.assertRaises(asyncio.CancelledError):
                await old_task
            self.assertEqual(len(captured_requests), 1)
            request = captured_requests[0]
            self.assertEqual(request.turn_source, "internal_team")
            self.assertEqual(
                request.aggregated_text,
                "explain the gas requirement clearly",
            )
            self.assertIn("authorized Yearn support staff", request.turn_instruction)
            self.assertNotIn("manual staff control", request.turn_instruction)
            self.assertEqual(
                request.current_history,
                [
                    {
                        "role": "user",
                        "content": "Why is my transaction still pending?",
                    }
                ],
            )
            self.assertEqual(request.attachments[0]["filename"], "staff-evidence.png")
            self.assertNotIn(channel_id, stopped_channels)
            self.assertIn(channel_id, conversation_threads)
            self.assertNotIn(channel_id, pending_messages)
            self.assertNotIn(channel_id, pending_attachments_by_channel)
            self.assertNotIn(channel_id, active_ticket_payloads)
            self.assertNotIn(channel_id, active_ticket_executor_tasks)
            self.assertNotIn(channel_id, pending_tasks)
            self.assertEqual(
                channel.sent_messages,
                ["Add enough ETH to cover the transaction fee."],
            )
            self.assertEqual(len(sent_views), 1)
            self.assertIsInstance(sent_views[0], StopBotView)
            notify.assert_not_called()
        finally:
            pending_tasks.pop(channel_id, None)
            if not old_task.done():
                old_task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_stopped_contributor_summon_is_one_shot_and_stays_stopped(
        self,
    ) -> None:
        channel_id = 214
        owner_id = 777
        contributor = SimpleNamespace(
            id=888,
            bot=False,
            name="contributor",
            roles=[SimpleNamespace(id=config.TICKET_CONTRIBUTOR_ROLE_ID)],
            guild_permissions=SimpleNamespace(administrator=False),
        )

        class _HistoryChannel(_FakeDiscordChannel):
            def __init__(self) -> None:
                super().__init__(channel_id)
                self.category = SimpleNamespace(id=1)
                self.guild = SimpleNamespace(id=2)

            async def history(self, *, limit: int):
                del limit
                if False:
                    yield None

        channel = _HistoryChannel()
        message = SimpleNamespace(
            id=215,
            author=contributor,
            content="y: give the user the final verified conclusion",
            channel=channel,
            reference=None,
            created_at=datetime.now(timezone.utc),
            attachments=[],
        )
        captured_requests: list[TicketTurnRequest] = []

        class _Executor:
            async def execute_turn(self, request: TicketTurnRequest, hooks=None):
                del hooks
                captured_requests.append(request)
                return SimpleNamespace(
                    flow_outcome=TicketAgentFlowOutcome(
                        raw_final_reply="Here is the complete verified conclusion.",
                        conversation_history=[],
                        completed_agent_key=None,
                        requires_human_handoff=False,
                    ),
                    updated_job=request.investigation_job,
                )

        ticket_owner_user_id_by_channel[channel_id] = owner_id
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "manual_stop"
        persist_ticket_state(channel_id)
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _Executor()

        sent_views: list[object | None] = []

        async def fake_send_long_message(target, text, **kwargs):
            sent_views.append(kwargs.get("view"))
            await target.send(text, **kwargs)

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _HistoryChannel),
                patch("ysupport.send_long_message", new=fake_send_long_message),
                patch("ysupport._notify_handoff") as notify,
            ):
                await bot.on_message(message)
                summon_task = pending_tasks[channel_id]
                await asyncio.wait_for(summon_task, timeout=1)

            self.assertEqual(len(captured_requests), 1)
            request = captured_requests[0]
            self.assertEqual(request.turn_source, "internal_team")
            self.assertIn(
                "remains under manual staff control", request.turn_instruction
            )
            self.assertIn("do not ask the user to reply", request.turn_instruction)
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], owner_id)
            self.assertNotIn(channel_id, conversation_threads)
            self.assertNotIn(channel_id, ticket_investigation_jobs)
            self.assertNotIn(channel_id, pending_tasks)
            self.assertEqual(
                channel.sent_messages,
                ["Here is the complete verified conclusion."],
            )
            self.assertEqual(sent_views, [None])
            notify.assert_not_called()
        finally:
            pending_tasks.pop(channel_id, None)
            clear_ticket_channel_state(channel_id, delete_persisted=True)
