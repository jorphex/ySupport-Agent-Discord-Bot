import asyncio
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agents import MaxTurnsExceeded, RunContextWrapper
import discord

from bot_behavior import OUT_OF_SCOPE_SUPPORT_MESSAGE
import config
from state import (
    BotRunContext,
    TicketInvestigationJob,
    TeamHandoffNotice,
    channel_intent_after_button,
    clear_team_handoff_notice,
    clear_ticket_investigation_job,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    is_ticket_waiting_for_team,
    last_bot_reply_ts_by_channel,
    pending_messages,
    public_conversations,
    PublicConversation,
    stopped_channels,
    team_handoff_notice_by_channel,
    ticket_investigation_jobs,
    ticket_owner_user_id_by_channel,
)
from handoff import (
    HandoffRoute,
    build_archived_handoff_notice,
    build_closed_handoff_notice,
    build_handoff_notice,
    build_pending_delivery_handoff_notice,
)
from ticket_investigation.worker import TicketInvestigationWorker
from support_agents import (
    TicketTriageDecision,
    ticket_triage_router_agent,
    triage_agent,
    yearn_bug_triage_agent,
    yearn_data_agent,
    yearn_docs_qa_agent,
)
from ticket_investigation.runtime import (
    TicketAgentFlowOutcome,
    resolve_freeform_starting_agent,
    TicketInvestigationRuntime,
    TicketTurnRequest,
)
from ysupport import (
    TicketBot,
    _notify_handoff,
)
from tests.test_ticket_intake import (
    _FakeDiscordChannel,
    _FakeOriginalMessage,
    _FakePublicChannel,
    _FakeTriggerMessage,
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
    async def test_notify_handoff_uses_model_summary_when_available(self) -> None:
        route = HandoffRoute(
            target="support_manual",
            team_label="support team",
            reason="manual follow-up needed",
        )
        notices: list[str] = []

        async def fake_summarize_handoff_summary(**kwargs) -> str | None:
            self.assertEqual(kwargs["summary"], "yes please tell them")
            self.assertTrue(
                any(
                    "withdraw button spins forever" in message
                    for message in kwargs["recent_user_messages"]
                )
            )
            self.assertIn("chain: Ethereum", kwargs["known_facts"])
            return "Withdraw button spins forever on the vault page and the user needs a manual team review."

        async def fake_send_handoff_notice(message_text: str):
            notices.append(message_text)
            return True

        with (
            patch("ysupport.summarize_handoff_summary", new=fake_summarize_handoff_summary),
            patch("ysupport.send_handoff_notice", new=fake_send_handoff_notice),
        ):
            await _notify_handoff(
                route=route,
                summary="yes please tell them",
                channel_id=1506309610192113917,
                guild_id=734804446353031319,
                source="ticket",
                recent_user_messages=[
                    "withdraw button spins forever on the vault page",
                    "yes please tell them",
                ],
                known_facts=["chain: Ethereum"],
            )

        self.assertEqual(len(notices), 1)
        self.assertIn(
            "<b>Summary</b>: Withdraw button spins forever on the vault page and the user needs a manual team review.",
            notices[0],
        )

    async def test_notify_handoff_falls_back_to_raw_summary_when_model_summary_missing(self) -> None:
        route = HandoffRoute(
            target="support_manual",
            team_label="support team",
            reason="manual follow-up needed",
        )
        notices: list[str] = []

        async def fake_summarize_handoff_summary(**kwargs) -> str | None:
            return None

        async def fake_send_handoff_notice(message_text: str):
            notices.append(message_text)
            return True

        with (
            patch("ysupport.summarize_handoff_summary", new=fake_summarize_handoff_summary),
            patch("ysupport.send_handoff_notice", new=fake_send_handoff_notice),
        ):
            await _notify_handoff(
                route=route,
                summary="yes please tell them",
                channel_id=1506309610192113917,
                guild_id=734804446353031319,
                source="ticket",
                recent_user_messages=["withdraw button spins forever on the vault page"],
                known_facts=["chain: Ethereum"],
            )

        self.assertEqual(len(notices), 1)
        self.assertIn("<b>Summary</b>: yes please tell them", notices[0])

    async def test_public_trigger_applies_outer_boundary_before_executor(self) -> None:
        original_author_id = 70
        channel_id = 71
        original_message = _FakeOriginalMessage(
            author_id=original_author_id,
            content="We want a marketing partnership with Yearn",
        )
        trigger_channel = _FakePublicChannel(channel_id, original_message)
        trigger_message = _FakeTriggerMessage(
            trigger_channel,
            reference_message_id=12345,
        )

        class _FailingExecutor:
            async def execute_turn(self, request, hooks=None):
                raise AssertionError("Boundary reply should stop before executor runs.")

        async def fake_boundary(_text: str):
            return {
                "classification": "business_boundary",
                "tripwire_triggered": True,
                "message": "Boundary reply",
            }

        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _FailingExecutor()

        with patch("ysupport._outer_support_boundary_result", new=fake_boundary):
            handled = await bot._handle_public_trigger_message(trigger_message, "y")

        self.assertTrue(handled)
        self.assertEqual(len(original_message.replies), 1)
        self.assertEqual(original_message.replies[0][0], "Boundary reply")
        self.assertFalse(original_message.replies[0][1]["mention_author"])

    async def test_public_trigger_short_circuits_moderator_access_before_executor(self) -> None:
        original_author_id = 72
        channel_id = 73
        original_message = _FakeOriginalMessage(
            author_id=original_author_id,
            content="I finished verification but still cannot access the Discord.",
        )
        trigger_channel = _FakePublicChannel(channel_id, original_message)
        trigger_message = _FakeTriggerMessage(
            trigger_channel,
            reference_message_id=12345,
        )

        class _FailingExecutor:
            async def execute_turn(self, request, hooks=None):
                raise AssertionError("Moderator access reply should stop before executor runs.")

        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _FailingExecutor()

        handled = await bot._handle_public_trigger_message(trigger_message, "y")

        self.assertTrue(handled)
        self.assertEqual(len(original_message.replies), 1)
        self.assertIn("A moderator needs to check this.", original_message.replies[0][0])
        self.assertFalse(original_message.replies[0][1]["mention_author"])

    async def test_public_trigger_max_turns_uses_configured_limit_and_replies_cleanly(self) -> None:
        original_author_id = 73
        channel_id = 74
        original_message = _FakeOriginalMessage(
            author_id=original_author_id,
            content="How is the stYFI APY calculated?",
        )
        trigger_channel = _FakePublicChannel(channel_id, original_message)
        trigger_message = _FakeTriggerMessage(
            trigger_channel,
            reference_message_id=12345,
        )

        fake_executor = _FakeInvestigationExecutor(exc=MaxTurnsExceeded("Max turns exceeded"))
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = fake_executor
        handoff_notices: list[str] = []

        async def fake_send_handoff_notice(message_text: str) -> bool:
            handoff_notices.append(message_text)
            return True

        public_conversations[original_author_id] = PublicConversation(
            history=[{"role": "assistant", "content": "Earlier context"}],
            last_interaction_time=datetime.now(timezone.utc),
        )
        try:
            with (
                patch("ysupport.send_handoff_notice", new=fake_send_handoff_notice),
            ):
                handled = await bot._handle_public_trigger_message(trigger_message, "y")
        finally:
            public_conversations.pop(original_author_id, None)

        self.assertTrue(handled)
        self.assertTrue(trigger_message.deleted)
        self.assertEqual(len(fake_executor.calls), 1)
        self.assertEqual(
            fake_executor.calls[0]["request"].current_history,
            [{"role": "assistant", "content": "Earlier context"}],
        )
        self.assertNotIn(original_author_id, public_conversations)
        self.assertEqual(len(original_message.replies), 1)
        self.assertIn("internal analysis limit", original_message.replies[0][0].lower())
        self.assertIn("I've notified the support team", original_message.replies[0][0])
        self.assertNotIn("<@", original_message.replies[0][0])
        self.assertFalse(original_message.replies[0][1]["mention_author"])
        self.assertTrue(original_message.replies[0][1]["suppress_embeds"])
        self.assertEqual(len(handoff_notices), 1)
        self.assertIn("<code>support_manual</code>", handoff_notices[0])

    async def test_public_trigger_uses_transport_executor_and_persists_public_state(self) -> None:
        original_author_id = 91
        channel_id = 92
        original_message = _FakeOriginalMessage(
            author_id=original_author_id,
            content="Where can I monitor stYFI rewards?",
        )
        trigger_channel = _FakePublicChannel(channel_id, original_message)
        trigger_message = _FakeTriggerMessage(
            trigger_channel,
            reference_message_id=12345,
        )
        updated_job = TicketInvestigationJob(channel_id=channel_id)
        updated_job.mark_waiting_for_user()
        fake_executor = _FakeInvestigationExecutor(
            result=type(
                "_Result",
                (),
                {
                    "flow_outcome": TicketAgentFlowOutcome(
                        raw_final_reply="Use the stYFI dashboard.",
                        conversation_history=[
                            {"role": "user", "content": "Where can I monitor stYFI rewards?"},
                            {"role": "assistant", "content": "Use the stYFI dashboard."},
                        ],
                        completed_agent_key=None,
                        requires_human_handoff=False,
                    ),
                    "updated_job": updated_job,
                },
            )()
        )
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = fake_executor

        try:
            handled = await bot._handle_public_trigger_message(trigger_message, "y")

            self.assertTrue(handled)
            self.assertEqual(len(fake_executor.calls), 1)
            stored_conversation = public_conversations.get(original_author_id)
            self.assertIsNotNone(stored_conversation)
            assert stored_conversation is not None
            self.assertEqual(
                stored_conversation.history[-1]["content"],
                "Use the stYFI dashboard.",
            )
            self.assertIs(stored_conversation.investigation_job, updated_job)
            self.assertEqual(trigger_channel.sent_messages, ["Use the stYFI dashboard."])
        finally:
            public_conversations.pop(original_author_id, None)

    async def test_process_ticket_message_formats_handoff_reply_and_notifies_telegram(self) -> None:
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
                            {"role": "user", "content": "PPS is flat and rewards need to be dumped."},
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

        async def fake_send_handoff_notice(message_text: str) -> bool:
            handoff_notices.append(message_text)
            return True

        try:
            with patch("ysupport.send_long_message", new=fake_send_long_message):
                with patch("ysupport.send_handoff_notice", new=fake_send_handoff_notice):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            stopped_channels.discard(channel_id)
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_channel.sent_messages), 1)
        self.assertIn("This likely needs manual review.", fake_channel.sent_messages[0])
        self.assertIn("I've notified the strategy team", fake_channel.sent_messages[0])
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, fake_channel.sent_messages[0])
        self.assertEqual(len(handoff_notices), 1)
        self.assertIn("<code>strategist_ops</code>", handoff_notices[0])

    async def test_waiting_for_team_ticket_stores_followup_and_sends_parked_ack(self) -> None:
        channel_id = 94
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)
        owner = SimpleNamespace(id=777, bot=False, name="owner")
        message = SimpleNamespace(
            author=owner,
            content="here are more details",
            channel=fake_channel,
            reference=None,
            created_at=datetime.now(timezone.utc),
            attachments=[],
        )

        bot = TicketBot(intents=discord.Intents.none())
        conversation_threads[channel_id] = [{"role": "assistant", "content": "Team notified."}]
        ticket_owner_user_id_by_channel[channel_id] = 777
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
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
        finally:
            conversation_threads.pop(channel_id, None)
            ticket_owner_user_id_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)
            last_bot_reply_ts_by_channel.pop(channel_id, None)

    async def test_telegram_handoff_reply_consumes_notice_and_posts_update(self) -> None:
        channel_id = 95
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel

        conversation_threads[channel_id] = [{"role": "user", "content": "initial issue"}]
        ticket_owner_user_id_by_channel[channel_id] = 777
        original_notice = build_handoff_notice(
            HandoffRoute(
                target="support_manual",
                team_label="support team",
                reason="manual follow-up needed",
            ),
            summary="initial issue",
            channel_id=channel_id,
            guild_id=2,
        )
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            target="support_manual",
            reason="manual follow-up needed",
            message_text=original_notice,
        )
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        channel_intent_after_button[channel_id] = "investigate_issue"
        edited_messages: list[tuple[str, int, str]] = []
        internal_turn_calls: list[dict[str, str]] = []

        update = {
            "update_id": 1,
            "message": {
                "message_id": 999,
                "chat": {"id": "123"},
                "text": "tell the user the tx is queued pending signatures",
                "reply_to_message": {"message_id": 456},
            },
        }

        async def fake_edit_handoff_notice(*, chat_id: str, message_id: int, message_text: str) -> bool:
            edited_messages.append((chat_id, message_id, message_text))
            return True

        async def fake_internal_turn(**kwargs) -> str:
            internal_turn_calls.append(
                {
                    "prompt_text": kwargs["prompt_text"],
                    "instruction_text": kwargs["instruction_text"],
                }
            )
            return "The transaction has been queued and is pending multisig signatures."

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"):
                with patch("ysupport.edit_handoff_notice", new=fake_edit_handoff_notice):
                    with patch("ysupport._run_internal_instruction_turn", new=fake_internal_turn):
                        with patch("ysupport.send_long_message", new=fake_send_long_message):
                            await bot._handle_telegram_handoff_update(update)
            self.assertEqual(
                edited_messages,
                [
                    ("123", 456, build_pending_delivery_handoff_notice(original_notice)),
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
            self.assertIn("This input is from the internal team", internal_turn_calls[0]["instruction_text"])
            self.assertIn("Write directly to the user, not back to the team", internal_turn_calls[0]["instruction_text"])
            self.assertIn("Expand shorthand like `pending sigs`", internal_turn_calls[0]["instruction_text"])
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(ticket_investigation_jobs[channel_id].mode, "waiting_for_user")
        finally:
            conversation_threads.pop(channel_id, None)
            ticket_owner_user_id_by_channel.pop(channel_id, None)
            channel_intent_after_button.pop(channel_id, None)
            clear_team_handoff_notice(channel_id)
            clear_ticket_investigation_job(channel_id)
            last_bot_reply_ts_by_channel.pop(channel_id, None)

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
            target="support_manual",
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
                with patch("ysupport.send_telegram_message", new=fake_send_telegram_message):
                    await bot._handle_telegram_handoff_update(update)
            self.assertEqual(fake_channel.sent_messages, [])
            self.assertEqual(
                telegram_feedback,
                [(
                    "123",
                    "That reply is too vague to send to the user. Reply with the exact update I should give them. Only one clear reply will be used.",
                    999,
                )],
            )
            notice = team_handoff_notice_by_channel[channel_id]
            self.assertFalse(notice.reply_consumed)
            self.assertEqual(notice.status, "open")
            self.assertIsNone(notice.pending_reply_text)
        finally:
            clear_team_handoff_notice(channel_id)
            clear_ticket_investigation_job(channel_id)
            last_bot_reply_ts_by_channel.pop(channel_id, None)

    async def test_pending_telegram_handoff_reply_resumes_and_posts_update(self) -> None:
        channel_id = 96
        fake_channel = _FakeDiscordChannel(channel_id)
        fake_channel.category = SimpleNamespace(id=1)
        fake_channel.guild = SimpleNamespace(id=2)

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel

        conversation_threads[channel_id] = [{"role": "user", "content": "initial issue"}]
        ticket_owner_user_id_by_channel[channel_id] = 777
        original_notice = build_handoff_notice(
            HandoffRoute(
                target="support_manual",
                team_label="support team",
                reason="manual follow-up needed",
            ),
            summary="initial issue",
            channel_id=channel_id,
            guild_id=2,
        )
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            target="support_manual",
            reason="manual follow-up needed",
            message_text=original_notice,
            reply_consumed=True,
            status="pending_delivery",
            pending_reply_text="tell the user the tx is queued pending signatures",
            pending_reply_message_id=999,
        )
        job = get_or_create_ticket_investigation_job(channel_id)
        job.mark_escalated_to_human()
        channel_intent_after_button[channel_id] = "investigate_issue"
        edited_messages: list[tuple[str, int, str]] = []

        async def fake_edit_handoff_notice(*, chat_id: str, message_id: int, message_text: str) -> bool:
            edited_messages.append((chat_id, message_id, message_text))
            return True

        async def fake_internal_turn(**kwargs) -> str:
            return "The transaction has been queued and is pending multisig signatures."

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch("ysupport.edit_handoff_notice", new=fake_edit_handoff_notice):
                with patch("ysupport._run_internal_instruction_turn", new=fake_internal_turn):
                    with patch("ysupport.send_long_message", new=fake_send_long_message):
                        await bot._resume_pending_telegram_handoff_replies()
            self.assertEqual(
                edited_messages,
                [
                    ("123", 456, build_pending_delivery_handoff_notice(original_notice)),
                    ("123", 456, build_closed_handoff_notice(original_notice)),
                ],
            )
            self.assertEqual(
                fake_channel.sent_messages,
                ["The transaction has been queued and is pending multisig signatures."],
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(ticket_investigation_jobs[channel_id].mode, "waiting_for_user")
        finally:
            conversation_threads.pop(channel_id, None)
            ticket_owner_user_id_by_channel.pop(channel_id, None)
            channel_intent_after_button.pop(channel_id, None)
            clear_team_handoff_notice(channel_id)
            clear_ticket_investigation_job(channel_id)
            last_bot_reply_ts_by_channel.pop(channel_id, None)

    async def test_deleted_ticket_archives_open_telegram_handoff_notice(self) -> None:
        channel_id = 188
        original_notice = build_handoff_notice(
            HandoffRoute(
                target="support_manual",
                team_label="support team",
                reason="manual follow-up needed",
            ),
            summary="need human",
            channel_id=channel_id,
            guild_id=734804446353031319,
        )
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            target="support_manual",
            reason="manual follow-up needed",
            message_text=original_notice,
        )

        bot = TicketBot(intents=discord.Intents.none())
        fake_channel = _FakeDiscordChannel(channel_id)
        edits: list[tuple[str, int, str]] = []

        async def fake_edit_handoff_notice(*, chat_id: str, message_id: int, message_text: str) -> bool:
            edits.append((chat_id, message_id, message_text))
            return True

        try:
            with patch("ysupport.discord.TextChannel", _FakeDiscordChannel), patch(
                "ysupport.edit_handoff_notice",
                side_effect=fake_edit_handoff_notice,
            ), patch("state.reset_ticket_codex_session"):
                await bot.on_guild_channel_delete(fake_channel)
            self.assertEqual(
                edits,
                [("123", 456, build_archived_handoff_notice(original_notice))],
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
        finally:
            clear_team_handoff_notice(channel_id)

    async def test_delivered_pending_close_resumes_without_resending_discord_update(self) -> None:
        channel_id = 196
        original_notice = build_handoff_notice(
            HandoffRoute(
                target="support_manual",
                team_label="support team",
                reason="manual follow-up needed",
            ),
            summary="need human",
            channel_id=channel_id,
            guild_id=734804446353031319,
        )
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            target="support_manual",
            reason="manual follow-up needed",
            message_text=original_notice,
            reply_consumed=True,
            status="delivered_pending_close",
            pending_reply_text="tell the user the tx is queued pending signatures",
            pending_reply_message_id=999,
        )

        bot = TicketBot(intents=discord.Intents.none())
        edits: list[tuple[str, int, str]] = []

        async def fake_edit_handoff_notice(*, chat_id: str, message_id: int, message_text: str) -> bool:
            edits.append((chat_id, message_id, message_text))
            return True

        async def fake_internal_turn(**kwargs) -> str:
            raise AssertionError("Delivered-pending-close should not synthesize a second Discord update.")

        async def fake_send_long_message(channel, message, **kwargs):
            raise AssertionError("Delivered-pending-close should not resend to Discord.")

        try:
            with patch("ysupport.edit_handoff_notice", new=fake_edit_handoff_notice):
                with patch("ysupport._run_internal_instruction_turn", new=fake_internal_turn):
                    with patch("ysupport.send_long_message", new=fake_send_long_message):
                        await bot._resume_pending_telegram_handoff_replies()
            self.assertEqual(
                edits,
                [("123", 456, build_closed_handoff_notice(original_notice))],
            )
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
        finally:
            clear_team_handoff_notice(channel_id)

    async def test_telegram_update_offset_persists_only_after_update_handling(self) -> None:
        bot = TicketBot(intents=discord.Intents.none())
        bot._telegram_update_offset = None
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
            with patch("ysupport.fetch_telegram_updates", new=fake_fetch_telegram_updates):
                with patch("ysupport.persist_telegram_update_offset", new=fake_persist_telegram_update_offset):
                    with patch.object(bot, "_handle_telegram_handoff_update", new=fake_handle_success):
                        with self.assertRaises(asyncio.CancelledError):
                            await bot._telegram_handoff_reply_loop()
            self.assertEqual(handled_updates, [update])
            self.assertEqual(persisted_offsets, [42])
            self.assertEqual(bot._telegram_update_offset, 42)

            bot._telegram_update_offset = None
            persisted_offsets.clear()
            call_count["fetch"] = 0

            async def fake_sleep(delay: float) -> None:
                sleep_calls.append(delay)
                raise asyncio.CancelledError

            with patch("ysupport.fetch_telegram_updates", new=fake_fetch_telegram_updates):
                with patch("ysupport.persist_telegram_update_offset", new=fake_persist_telegram_update_offset):
                    with patch.object(bot, "_handle_telegram_handoff_update", new=fake_handle_failure):
                        with patch("ysupport.asyncio.sleep", new=fake_sleep):
                            with self.assertRaises(asyncio.CancelledError):
                                await bot._telegram_handoff_reply_loop()
            self.assertEqual(persisted_offsets, [])
            self.assertIsNone(bot._telegram_update_offset)
            self.assertEqual(sleep_calls, [5])
        finally:
            await bot.close()

    async def test_resolve_freeform_starting_agent_reuses_ticket_router_for_public_lane_selection(self) -> None:
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=None,
                    last_agent=None,
                    _history=[],
                    _decision=TicketTriageDecision(
                        action="route_docs",
                        message=None,
                        reasoning="docs question",
                    ),
                ),
            ]
        )
        context = BotRunContext(channel_id=30, project_context="yearn")

        agent_key = await resolve_freeform_starting_agent(
            runner=fake_runner,
            input_list="Where do I see my stYFI position?",
            run_context=context,
            workflow_name="tests.public_route",
        )

        self.assertEqual(agent_key, "docs")
        self.assertEqual(len(fake_runner.calls), 1)
        self.assertIs(fake_runner.calls[0]["starting_agent"], ticket_triage_router_agent)

    async def test_ticket_agent_flow_routes_triage_decision_to_docs_specialist(self) -> None:
        channel_id = 31
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=None,
                    last_agent=None,
                    _history=[],
                    _decision=TicketTriageDecision(
                        action="route_docs",
                        message=None,
                        reasoning="docs question",
                    ),
                ),
                _FakeResult(
                    final_output="Open the stYFI app and check the positions page.",
                    last_agent=yearn_docs_qa_agent,
                    _history=[
                        {"role": "user", "content": "Where do I see my stYFI position?"},
                        {"role": "assistant", "content": "Open the stYFI app and check the positions page."},
                    ],
                ),
            ]
        )
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="investigate_issue",
        )
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text="I need help finding where to see my stYFI position.",
                    input_list=[{"role": "user", "content": "I need help finding where to see my stYFI position."}],
                    current_history=[],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 2)
        self.assertIs(fake_runner.calls[0]["starting_agent"], ticket_triage_router_agent)
        self.assertIs(fake_runner.calls[1]["starting_agent"], yearn_docs_qa_agent)
        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertIn("positions page", outcome.raw_final_reply.lower())

    async def test_ticket_agent_flow_remembers_single_withdrawal_target_from_data_reply(self) -> None:
        channel_id = 36
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=(
                        "**Katana Active Deposits:**\n"
                        "**Vault:** [Vault Name](https://yearn.fi/vaults/146/0x80c34BD3A3569E126e7055831036aa7b212cB159) (Symbol: yvVBUSDT)\n"
                        "  Address: `0x80c34BD3A3569E126e7055831036aa7b212cB159`\n"
                        "  Total Position: **1.000000 yvVBUSDT**"
                    ),
                    last_agent=yearn_data_agent,
                    _history=[],
                ),
            ]
        )
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="data_deposit_check",
        )
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text="0x1111111111111111111111111111111111111111",
                    input_list=[{"role": "user", "content": "0x1111111111111111111111111111111111111111"}],
                    current_history=[],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertEqual(investigation_job.evidence.withdrawal_target_chain, "katana")
        self.assertEqual(
            investigation_job.evidence.withdrawal_target_vault,
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )

    async def test_ticket_agent_flow_returns_direct_router_message_without_second_run(self) -> None:
        channel_id = 32
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=None,
                    last_agent=None,
                    _history=[],
                    _decision=TicketTriageDecision(
                        action="human_escalation",
                        message=(
                            "A moderator needs to check this. "
                            f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
                        ),
                        reasoning="discord access issue",
                    ),
                ),
            ]
        )
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text="I finished verification but still cannot access the Discord.",
                    input_list=[{"role": "user", "content": "I finished verification but still cannot access the Discord."}],
                    current_history=[],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertIsNone(outcome.completed_agent_key)
        self.assertTrue(outcome.requires_human_handoff)
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, outcome.raw_final_reply)
        self.assertEqual(
            outcome.conversation_history[-1]["content"],
            (
                "A moderator needs to check this. "
                f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
            ),
        )

    async def test_ticket_agent_flow_forces_handoff_for_explicit_human_request_with_repro_context(self) -> None:
        channel_id = 38
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=None,
                    last_agent=None,
                    _history=[],
                    _decision=TicketTriageDecision(
                        action="route_bug",
                        message=None,
                        reasoning="likely web issue",
                    ),
                ),
            ]
        )
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        prompt_text = (
            "I need a human asap. The withdraw button on "
            "https://yearn.fi/vaults/1/0xBe53A109B494E5c9f97b9Cd39Fe969BE68BF6204 "
            "just spins and never opens the wallet."
        )
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text=prompt_text,
                    input_list=[{"role": "user", "content": prompt_text}],
                    current_history=[],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertIsNone(outcome.completed_agent_key)
        self.assertTrue(outcome.requires_human_handoff)
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, outcome.raw_final_reply)

    async def test_ticket_agent_flow_keeps_bug_lane_when_human_request_lacks_repro_context(self) -> None:
        channel_id = 39
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=None,
                    last_agent=None,
                    _history=[],
                    _decision=TicketTriageDecision(
                        action="route_bug",
                        message=None,
                        reasoning="likely web issue",
                    ),
                ),
                _FakeResult(
                    final_output="Please share the exact page and what happens when you click the button.",
                    last_agent=yearn_bug_triage_agent,
                    _history=[],
                ),
            ]
        )
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        prompt_text = "I need a human asap. The button is broken."
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text=prompt_text,
                    input_list=[{"role": "user", "content": prompt_text}],
                    current_history=[],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 2)
        self.assertEqual(outcome.completed_agent_key, "bug")
        self.assertFalse(outcome.requires_human_handoff)
        self.assertIn("exact page", outcome.raw_final_reply.lower())

    async def test_ticket_agent_flow_short_circuits_bug_bounty_intake_boundary(self) -> None:
        channel_id = 37
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner([])
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        async def fake_boundary(_text: str):
            return {
                "classification": "security_process_boundary",
                "tripwire_triggered": True,
                "message": (
                    "If you are reporting a Yearn security issue and want bounty or disclosure handling, "
                    "use Yearn's official security process at https://docs.yearn.fi/developers/security. "
                    f"Human help is required beyond that path. {config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
                ),
            }
        try:
            with patch(
                "ticket_investigation.runtime.evaluate_support_boundary",
                new=fake_boundary,
            ):
                runtime = TicketInvestigationRuntime(fake_runner)
                outcome = await runtime.run_turn(
                    TicketTurnRequest(
                        aggregated_text=(
                            "Good day team, me and my team discovered an issue that should be addressed "
                            "and hope to be rewarded for our efforts"
                        ),
                        input_list=[
                            {
                                "role": "user",
                                "content": (
                                    "Good day team, me and my team discovered an issue that should be addressed "
                                    "and hope to be rewarded for our efforts"
                                ),
                            }
                        ],
                        current_history=[],
                        run_context=context,
                        investigation_job=investigation_job,
                        workflow_name="tests.ticket_flow",
                    )
                )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 0)
        self.assertIsNone(outcome.completed_agent_key)
        self.assertTrue(outcome.requires_human_handoff)
        lowered = outcome.raw_final_reply.lower()
        self.assertIn("docs.yearn.fi/developers/security", lowered)
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNotIn("browser", lowered)
        self.assertNotIn("device", lowered)

    async def test_ticket_agent_flow_uses_precomputed_boundary_without_second_model_call(self) -> None:
        channel_id = 371
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner([])
        context = BotRunContext(channel_id=channel_id, project_context="yearn")

        async def fail_boundary(_text: str):
            raise AssertionError("Precomputed boundary should bypass runtime boundary evaluation.")

        try:
            with patch(
                "ticket_investigation.runtime.evaluate_support_boundary",
                new=fail_boundary,
            ):
                runtime = TicketInvestigationRuntime(fake_runner)
                outcome = await runtime.run_turn(
                    TicketTurnRequest(
                        aggregated_text="Can you write a Python script to parse a CSV for me?",
                        input_list=[
                            {
                                "role": "user",
                                "content": "Can you write a Python script to parse a CSV for me?",
                            }
                        ],
                        current_history=[],
                        run_context=context,
                        investigation_job=investigation_job,
                        workflow_name="tests.ticket_flow",
                        precomputed_boundary={
                            "classification": "non_support_assistant",
                            "tripwire_triggered": True,
                            "message": OUT_OF_SCOPE_SUPPORT_MESSAGE,
                        },
                    )
                )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 0)
        self.assertEqual(outcome.raw_final_reply, OUT_OF_SCOPE_SUPPORT_MESSAGE)
        self.assertFalse(outcome.requires_human_handoff)

    async def test_ticket_agent_flow_marks_specialist_reply_handoff_explicitly(self) -> None:
        channel_id = 33
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=(
                        "This needs human review. "
                        f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
                    ),
                    last_agent=yearn_bug_triage_agent,
                    _history=[
                        {"role": "user", "content": "The app is broken."},
                        {
                            "role": "assistant",
                            "content": (
                                "This needs human review. "
                                f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
                            ),
                        },
                    ],
                ),
            ]
        )
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="bug_report",
        )
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text="The app is broken.",
                    input_list=[{"role": "user", "content": "The app is broken."}],
                    current_history=[],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertEqual(outcome.completed_agent_key, "bug")
        self.assertTrue(outcome.requires_human_handoff)

    async def test_ticket_agent_flow_injects_tx_followup_contract_before_specialist_run(self) -> None:
        channel_id = 34
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "data"
        investigation_job.remember_chain("katana")
        investigation_job.remember_tx_hash(
            "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        )
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output="The tx succeeded on Katana and minted 650.9147 yvWBUSDT shares.",
                    last_agent=yearn_data_agent,
                    _history=[
                        {
                            "role": "user",
                            "content": "Katana tx hash: 0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0",
                        },
                        {
                            "role": "assistant",
                            "content": "The tx succeeded on Katana and minted 650.9147 yvWBUSDT shares.",
                        },
                    ],
                ),
            ]
        )
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text="i dunno man. look into it",
                    input_list=[{"role": "user", "content": "i dunno man. look into it"}],
                    current_history=[{"role": "user", "content": "Earlier tx context"}],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertIn("650.9147", outcome.raw_final_reply)
        specialist_input = fake_runner.calls[0]["input"]
        self.assertEqual(specialist_input[-1]["role"], "system")
        self.assertIn(
            "Do not ask the user whether you should proceed",
            specialist_input[-1]["content"],
        )

    async def test_ticket_agent_flow_injects_report_pretriage_contract_before_specialist_run(self) -> None:
        channel_id = 341
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=(
                        "I checked the report, but it still needs the exact Yearn contract/path and a concrete claim."
                    ),
                    last_agent=yearn_bug_triage_agent,
                    _history=[
                        {
                            "role": "user",
                            "content": "Report: https://gist.github.com/example/abcdef1234567890",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                "I checked the report, but it still needs the exact Yearn contract/path and a concrete claim."
                            ),
                        },
                    ],
                ),
            ]
        )
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="bug_report",
        )
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text="Report: https://gist.github.com/example/abcdef1234567890",
                    input_list=[
                        {
                            "role": "user",
                            "content": "Report: https://gist.github.com/example/abcdef1234567890",
                        }
                    ],
                    current_history=[],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertEqual(outcome.completed_agent_key, "bug")
        self.assertFalse(outcome.requires_human_handoff)
        self.assertIn("exact yearn contract", outcome.raw_final_reply.lower())
        specialist_input = fake_runner.calls[0]["input"]
        self.assertEqual(specialist_input[-1]["role"], "system")
        self.assertIn("Do one bounded repo/docs pre-triage pass", specialist_input[-1]["content"])

    async def test_ticket_agent_flow_switches_from_data_followup_to_bug_for_repro_issue(self) -> None:
        channel_id = 35
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "data"
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output=None,
                    last_agent=None,
                    _history=[],
                    _decision=TicketTriageDecision(
                        action="route_bug",
                        message=None,
                        reasoning="reproducible wallet/product issue",
                    ),
                ),
                _FakeResult(
                    final_output=(
                        "What exact page, wallet, and error state are you seeing when Rabby says "
                        "'transaction not ready'?"
                    ),
                    last_agent=yearn_bug_triage_agent,
                    _history=[
                        {
                            "role": "user",
                            "content": "Rabby says transaction not ready for every address when I try to withdraw.",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                "What exact page, wallet, and error state are you seeing when Rabby says "
                                "'transaction not ready'?"
                            ),
                        },
                    ],
                ),
            ]
        )
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        try:
            runtime = TicketInvestigationRuntime(fake_runner)
            outcome = await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text="Rabby says transaction not ready for every address when I try to withdraw.",
                    input_list=[
                        {
                            "role": "assistant",
                            "content": (
                                "Okay, I can help with withdrawal instructions. "
                                "Please provide your wallet address (0x...)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Rabby says transaction not ready for every address when I try to withdraw.",
                        },
                    ],
                    current_history=[
                        {
                            "role": "assistant",
                            "content": (
                                "Okay, I can help with withdrawal instructions. "
                                "Please provide your wallet address (0x...)."
                            ),
                        }
                    ],
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 2)
        self.assertIs(fake_runner.calls[0]["starting_agent"], ticket_triage_router_agent)
        self.assertIs(fake_runner.calls[1]["starting_agent"], yearn_bug_triage_agent)
        self.assertEqual(outcome.completed_agent_key, "bug")
        self.assertNotIn("wallet address", outcome.raw_final_reply.lower())


@dataclass
class _FakeRuntime:
    outcome: object
    requests: list

    async def run_turn(self, request):
        self.requests.append(request)
        return self.outcome


class TicketWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_updates_job_state_after_specialist_turn(self) -> None:
        job = TicketInvestigationJob(channel_id=90)
        runtime = _FakeRuntime(
            outcome=TicketAgentFlowOutcome(
                raw_final_reply="Done.",
                conversation_history=[],
                completed_agent_key="data",
                requires_human_handoff=False,
            ),
            requests=[],
        )
        worker = TicketInvestigationWorker(runtime)

        result = await worker.execute_turn(
            TicketTurnRequest(
                aggregated_text="help",
                input_list=[],
                current_history=[],
                run_context=BotRunContext(channel_id=90, project_context="yearn"),
                investigation_job=job,
                workflow_name="tests.worker",
            )
        )

        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(result.flow_outcome.completed_agent_key, "data")
        self.assertEqual(job.mode, "waiting_for_user")
        self.assertEqual(job.current_specialty, "data")
        self.assertEqual(job.last_specialty, "data")

    async def test_worker_marks_human_escalation_on_handoff_outcome(self) -> None:
        job = TicketInvestigationJob(channel_id=91)
        runtime = _FakeRuntime(
            outcome=TicketAgentFlowOutcome(
                raw_final_reply=f"Needs help. {config.HUMAN_HANDOFF_TAG_PLACEHOLDER}",
                conversation_history=[],
                completed_agent_key=None,
                requires_human_handoff=True,
            ),
            requests=[],
        )
        worker = TicketInvestigationWorker(runtime)

        await worker.execute_turn(
            TicketTurnRequest(
                aggregated_text="help",
                input_list=[],
                current_history=[],
                run_context=BotRunContext(channel_id=91, project_context="yearn"),
                investigation_job=job,
                workflow_name="tests.worker",
            )
        )

        self.assertEqual(job.mode, "escalated_to_human")
        self.assertIsNone(job.current_specialty)




class DynamicInstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_data_agent_system_prompt_includes_runtime_context(self) -> None:
        context = BotRunContext(
            channel_id=6,
            project_context="yearn",
            initial_button_intent="data_withdrawal_flow_start",
        )
        prompt = await yearn_data_agent.get_system_prompt(RunContextWrapper(context))

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn("initial_button_intent: data_withdrawal_flow_start", prompt)
        self.assertIn("project_context: yearn", prompt)

    async def test_triage_agent_system_prompt_includes_runtime_context(self) -> None:
        context = BotRunContext(
            channel_id=7,
            project_context="yearn",
            initial_button_intent="other_free_form",
        )
        prompt = await triage_agent.get_system_prompt(RunContextWrapper(context))

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn("initial_button_intent: other_free_form", prompt)
        self.assertIn("is_public_trigger: false", prompt)

    async def test_docs_agent_system_prompt_includes_compact_mechanics_answer_rules(self) -> None:
        context = BotRunContext(
            channel_id=71,
            project_context="yearn",
            is_public_trigger=True,
        )
        prompt = await yearn_docs_qa_agent.get_system_prompt(RunContextWrapper(context))

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn("Synthesize Across Official Sources", prompt)
        self.assertIn("Question-Order Answers", prompt)
        self.assertIn("No Add-On Components", prompt)
        self.assertIn("Do not default to a general walkthrough", prompt)
        self.assertIn("closest supported mechanism in one sentence", prompt)

    async def test_bug_agent_system_prompt_keeps_handoff_placeholder(self) -> None:
        context = BotRunContext(
            channel_id=8,
            project_context="yearn",
            initial_button_intent="bug_report",
        )
        prompt = await yearn_bug_triage_agent.get_system_prompt(RunContextWrapper(context))

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, prompt)
        self.assertIn("initial_button_intent: bug_report", prompt)
