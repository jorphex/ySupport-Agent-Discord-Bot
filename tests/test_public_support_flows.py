import tests as _test_environment  # noqa: F401

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch


from agents import MaxTurnsExceeded
import discord

import config
from state import (
    TicketInvestigationJob,
    clear_public_conversation,
    public_conversations,
    PublicConversation,
)
from handoff import (
    TelegramSentMessage,
)
from ticket_investigation.runtime import (
    TicketAgentFlowOutcome,
)
from ysupport import (
    TicketBot,
)
from tests.ticket_flow_test_support import (
    FakeInvestigationExecutor as _FakeInvestigationExecutor,
    TicketFlowTestCase,
)
from tests.test_ticket_intake import (
    _FakeOriginalMessage,
    _FakePublicChannel,
    _FakeTriggerMessage,
)


class TicketFlowTests(TicketFlowTestCase):
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

    async def test_public_trigger_outer_setup_failure_replies_and_clears_state(
        self,
    ) -> None:
        original_author_id = 201
        original_message = _FakeOriginalMessage(
            author_id=original_author_id,
            content="Can you check my Yearn vault?",
        )
        trigger_channel = _FakePublicChannel(202, original_message)
        trigger_message = _FakeTriggerMessage(
            trigger_channel,
            reference_message_id=12345,
        )
        public_conversations[original_author_id] = PublicConversation(
            history=[{"role": "assistant", "content": "stale context"}],
            last_interaction_time=datetime.now(timezone.utc),
        )
        bot = TicketBot(intents=discord.Intents.none())

        try:
            with patch(
                "ysupport._outer_support_boundary_result",
                side_effect=RuntimeError("classifier unavailable"),
            ):
                handled = await bot._handle_public_trigger_message(trigger_message, "y")

            self.assertTrue(handled)
            self.assertNotIn(original_author_id, public_conversations)
            self.assertEqual(len(original_message.replies), 1)
            self.assertIn("preparing that request", original_message.replies[0][0])
            self.assertIn("Please try again", original_message.replies[0][0])
            self.assertFalse(original_message.replies[0][1]["mention_author"])
            self.assertTrue(original_message.replies[0][1]["suppress_embeds"])
        finally:
            clear_public_conversation(original_author_id)

    async def test_public_access_question_runs_support_before_handoff(self) -> None:
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

        updated_job = TicketInvestigationJob(channel_id=channel_id)
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _FakeInvestigationExecutor(
            result=SimpleNamespace(
                flow_outcome=TicketAgentFlowOutcome(
                    raw_final_reply=(
                        "Please confirm you completed the server verification steps "
                        "and reopen Discord before a moderator check is needed."
                    ),
                    conversation_history=[],
                    completed_agent_key=None,
                    requires_human_handoff=False,
                ),
                updated_job=updated_job,
            )
        )

        try:
            with patch("ysupport._notify_handoff") as mock_notify:
                handled = await bot._handle_public_trigger_message(trigger_message, "y")

            self.assertTrue(handled)
            self.assertEqual(len(bot.investigation_executor.calls), 1)
            mock_notify.assert_not_called()
            self.assertEqual(len(trigger_channel.sent_messages), 1)
            self.assertIn("verification steps", trigger_channel.sent_messages[0])
            self.assertIn(original_author_id, public_conversations)
        finally:
            clear_public_conversation(original_author_id)

    async def test_public_trigger_max_turns_uses_configured_limit_and_replies_cleanly(
        self,
    ) -> None:
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

        fake_executor = _FakeInvestigationExecutor(
            exc=MaxTurnsExceeded("Max turns exceeded")
        )
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = fake_executor
        handoff_notices: list[str] = []

        async def fake_send_handoff_notice(
            message_text: str,
            *,
            dismiss_enabled: bool = False,
        ) -> TelegramSentMessage:
            self.assertFalse(dismiss_enabled)
            handoff_notices.append(message_text)
            return TelegramSentMessage(
                chat_id="123",
                message_id=456,
                message_text=message_text,
            )

        public_conversations[original_author_id] = PublicConversation(
            history=[{"role": "assistant", "content": "Earlier context"}],
            last_interaction_time=datetime.now(timezone.utc),
        )
        try:
            with (
                patch(
                    "discord_support_runtime.send_handoff_notice",
                    new=fake_send_handoff_notice,
                ),
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
        self.assertIn("Please try again", original_message.replies[0][0])
        self.assertNotIn("I've notified", original_message.replies[0][0])
        self.assertNotIn("<@", original_message.replies[0][0])
        self.assertFalse(original_message.replies[0][1]["mention_author"])
        self.assertTrue(original_message.replies[0][1]["suppress_embeds"])
        self.assertEqual(handoff_notices, [])

    async def test_public_trigger_runtime_error_does_not_create_handoff(self) -> None:
        original_author_id = 196
        original_message = _FakeOriginalMessage(
            author_id=original_author_id,
            content="Check my Yearn position",
        )
        trigger_channel = _FakePublicChannel(197, original_message)
        trigger_message = _FakeTriggerMessage(
            trigger_channel,
            reference_message_id=12345,
        )
        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _FakeInvestigationExecutor(
            exc=RuntimeError("boom")
        )

        with (
            patch(
                "ysupport._outer_support_boundary_result",
                return_value={"tripwire_triggered": False},
            ),
            patch("ysupport._notify_handoff") as mock_notify,
        ):
            handled = await bot._handle_public_trigger_message(trigger_message, "y")

        self.assertTrue(handled)
        mock_notify.assert_not_called()
        self.assertNotIn(original_author_id, public_conversations)
        self.assertIn("Please try again", original_message.replies[0][0])
        self.assertNotIn("I've notified", original_message.replies[0][0])

    async def test_public_handoff_copy_and_state_follow_actual_delivery(self) -> None:
        async def run_case(
            *,
            original_author_id: int,
            notice: TelegramSentMessage | None,
        ) -> tuple[str, list[str]]:
            original_message = _FakeOriginalMessage(
                author_id=original_author_id,
                content="The Yearn app is broken and needs a person",
            )
            trigger_channel = _FakePublicChannel(
                original_author_id + 1,
                original_message,
            )
            trigger_message = _FakeTriggerMessage(
                trigger_channel,
                reference_message_id=12345,
            )
            updated_job = TicketInvestigationJob(channel_id=trigger_channel.id)
            executor = _FakeInvestigationExecutor(
                result=type(
                    "_Result",
                    (),
                    {
                        "flow_outcome": TicketAgentFlowOutcome(
                            raw_final_reply=(
                                "A person needs to review this. "
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
            bot.investigation_executor = executor
            notices: list[str] = []

            async def fake_send_handoff_notice(
                message_text: str,
                *,
                dismiss_enabled: bool = False,
            ):
                self.assertFalse(dismiss_enabled)
                notices.append(message_text)
                return notice

            with (
                patch(
                    "ysupport._outer_support_boundary_result",
                    return_value={"tripwire_triggered": False},
                ),
                patch(
                    "discord_support_runtime.send_handoff_notice",
                    new=fake_send_handoff_notice,
                ),
            ):
                await bot._handle_public_trigger_message(trigger_message, "y")
            return trigger_channel.sent_messages[0], notices

        failed_reply, failed_notices = await run_case(
            original_author_id=198,
            notice=None,
        )
        self.assertIn("couldn't send", failed_reply)
        self.assertIn(198, public_conversations)
        self.assertNotIn("Reply to this message", failed_notices[0])
        self.assertIn("Alert only", failed_notices[0])

        sent_reply, sent_notices = await run_case(
            original_author_id=200,
            notice=TelegramSentMessage(chat_id="123", message_id=456),
        )
        self.assertIn("I've notified", sent_reply)
        self.assertNotIn(200, public_conversations)
        self.assertIn("Alert only", sent_notices[0])

        clear_public_conversation(198)

    async def test_public_trigger_uses_transport_executor_and_persists_public_state(
        self,
    ) -> None:
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
                            {
                                "role": "user",
                                "content": "Where can I monitor stYFI rewards?",
                            },
                            {
                                "role": "assistant",
                                "content": "Use the stYFI dashboard.",
                            },
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
            self.assertEqual(
                trigger_channel.sent_messages, ["Use the stYFI dashboard."]
            )
        finally:
            public_conversations.pop(original_author_id, None)
