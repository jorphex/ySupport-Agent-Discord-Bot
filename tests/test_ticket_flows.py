import tests as _test_environment  # noqa: F401

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
import handoff
import state
from state import (
    active_ticket_payloads,
    BotRunContext,
    TicketInvestigationJob,
    TeamHandoffNotice,
    channel_intent_after_button,
    clear_public_conversation,
    clear_team_handoff_notice,
    clear_ticket_channel_state,
    clear_ticket_investigation_job,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    is_ticket_waiting_for_team,
    last_bot_reply_ts_by_channel,
    pending_attachments_by_channel,
    pending_messages,
    pending_tasks,
    persist_ticket_state,
    public_conversations,
    PublicConversation,
    stopped_channels,
    stop_reasons_by_channel,
    team_handoff_notice_by_channel,
    ticket_investigation_jobs,
    ticket_owner_user_id_by_channel,
)
from handoff import (
    DISMISS_HANDOFF_CALLBACK_DATA,
    build_archived_handoff_notice,
    build_closed_handoff_notice,
    build_dismissed_handoff_notice,
    build_failed_delivery_handoff_notice,
    build_handoff_notice,
    build_pending_delivery_handoff_notice,
    TelegramApiError,
    TelegramSentMessage,
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
from views import InitialInquiryView, StopBotView
from ysupport import (
    TicketBot,
    _build_discord_intents,
    _build_staff_summon_history,
    _refresh_discord_attachment_urls,
    _reload_runtime_env_and_config,
    _run_ticket_bot_with_fatal_startup_backoff,
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
            "ysupport.summarize_handoff_summary",
            return_value=None,
        )
        summary_patcher.start()
        self.addCleanup(summary_patcher.stop)

    async def test_setup_hook_registers_persistent_ticket_views(self) -> None:
        bot = TicketBot(intents=discord.Intents.none())

        with patch.object(bot, "add_view") as mock_add_view:
            await bot.setup_hook()

        registered_view_types = {
            type(call.args[0]) for call in mock_add_view.call_args_list
        }
        self.assertEqual(registered_view_types, {InitialInquiryView, StopBotView})

    async def test_parked_attachment_urls_refresh_from_source_message(self) -> None:
        current_attachment = SimpleNamespace(
            id=456,
            filename="evidence.png",
            url="https://cdn.discordapp.com/attachments/1/2/refreshed.png",
            content_type="image/png",
            size=321,
        )
        source_message = SimpleNamespace(attachments=[current_attachment])

        class _RefreshChannel:
            id = 123

            async def fetch_message(self, message_id: int):
                self.fetched_message_id = message_id
                return source_message

        channel = _RefreshChannel()
        attachments = await _refresh_discord_attachment_urls(
            channel,
            [
                {
                    "attachment_id": 456,
                    "source_message_id": 789,
                    "filename": "evidence.png",
                    "url": "https://cdn.discordapp.com/attachments/1/2/expired.png",
                    "content_type": "image/png",
                    "size": 123,
                    "is_image": True,
                }
            ],
        )

        self.assertEqual(channel.fetched_message_id, 789)
        self.assertEqual(
            attachments[0]["url"],
            "https://cdn.discordapp.com/attachments/1/2/refreshed.png",
        )
        self.assertEqual(attachments[0]["attachment_id"], 456)
        self.assertEqual(attachments[0]["source_message_id"], 789)

    async def test_persistent_ticket_views_separate_intake_and_stop_authority(
        self,
    ) -> None:
        channel_id = 194

        class _FakeInteractionResponse:
            def __init__(self) -> None:
                self.messages: list[tuple[str, bool]] = []

            def is_done(self) -> bool:
                return False

            async def send_message(self, message: str, *, ephemeral: bool) -> None:
                self.messages.append((message, ephemeral))

        def interaction(
            user_id: int,
            *,
            contributor: bool = False,
            administrator: bool = False,
        ):
            return SimpleNamespace(
                channel_id=channel_id,
                user=SimpleNamespace(
                    id=user_id,
                    roles=(
                        [SimpleNamespace(id=config.TICKET_CONTRIBUTOR_ROLE_ID)]
                        if contributor
                        else []
                    ),
                    guild_permissions=SimpleNamespace(
                        administrator=administrator,
                    ),
                ),
                response=_FakeInteractionResponse(),
            )

        ticket_owner_user_id_by_channel[channel_id] = 777
        try:
            owner_interaction = interaction(777)
            contributor_interaction = interaction(888, contributor=True)
            administrator_interaction = interaction(999, administrator=True)
            other_interaction = interaction(111)

            self.assertTrue(
                await InitialInquiryView().interaction_check(owner_interaction)
            )
            self.assertFalse(
                await InitialInquiryView().interaction_check(
                    contributor_interaction
                )
            )
            self.assertTrue(
                await StopBotView().interaction_check(owner_interaction)
            )
            self.assertTrue(
                await StopBotView().interaction_check(contributor_interaction)
            )
            self.assertTrue(
                await StopBotView().interaction_check(administrator_interaction)
            )
            self.assertFalse(await StopBotView().interaction_check(other_interaction))
            self.assertEqual(
                other_interaction.response.messages,
                [
                    (
                        "Only the ticket owner or support team can stop the bot.",
                        True,
                    )
                ],
            )
        finally:
            ticket_owner_user_id_by_channel.pop(channel_id, None)

    async def test_persistent_ticket_views_explain_unknown_owner_recovery(self) -> None:
        channel_id = 195

        class _FakeInteractionResponse:
            def __init__(self) -> None:
                self.messages: list[tuple[str, bool]] = []

            def is_done(self) -> bool:
                return False

            async def send_message(self, message: str, *, ephemeral: bool) -> None:
                self.messages.append((message, ephemeral))

        fake_interaction = SimpleNamespace(
            channel_id=channel_id,
            user=SimpleNamespace(id=777),
            response=_FakeInteractionResponse(),
        )
        ticket_owner_user_id_by_channel.pop(channel_id, None)

        self.assertFalse(
            await InitialInquiryView().interaction_check(fake_interaction)
        )
        self.assertIn("Send one message", fake_interaction.response.messages[0][0])
        self.assertTrue(fake_interaction.response.messages[0][1])

        administrator_interaction = SimpleNamespace(
            channel_id=channel_id,
            user=SimpleNamespace(
                id=888,
                roles=[],
                guild_permissions=SimpleNamespace(administrator=True),
            ),
            response=_FakeInteractionResponse(),
        )
        self.assertTrue(
            await StopBotView().interaction_check(administrator_interaction)
        )

    async def test_stop_button_does_not_claim_durable_success_when_save_fails(
        self,
    ) -> None:
        channel_id = 196

        class _FakeResponse:
            def __init__(self) -> None:
                self.deferred = False

            def is_done(self) -> bool:
                return False

            async def defer(self) -> None:
                self.deferred = True

        class _FakeFollowup:
            def __init__(self) -> None:
                self.messages: list[tuple[str, bool]] = []

            async def send(
                self,
                message: str,
                *,
                ephemeral: bool,
                suppress_embeds: bool,
            ) -> None:
                self.messages.append((message, ephemeral))

        async def active_ticket_turn() -> None:
            await asyncio.Event().wait()

        active_task = asyncio.create_task(active_ticket_turn())
        pending_tasks[channel_id] = active_task
        response = _FakeResponse()
        followup = _FakeFollowup()
        interaction = SimpleNamespace(
            channel=SimpleNamespace(id=channel_id),
            user=SimpleNamespace(id=777, name="owner"),
            response=response,
            followup=followup,
            message=None,
        )

        try:
            with (
                patch("views.stop_ticket_channel", return_value=False),
                patch("views.logging.error") as log_error,
            ):
                await StopBotView().children[0].callback(interaction)

            self.assertTrue(response.deferred)
            self.assertEqual(
                followup.messages,
                [
                    (
                        "The bot stopped, but I couldn't save that setting for a "
                        "restart. Please try Stop Bot again.",
                        True,
                    )
                ],
            )
            self.assertNotIn(channel_id, pending_tasks)
            log_error.assert_called_once()
            with self.assertRaises(asyncio.CancelledError):
                await active_task
        finally:
            pending_tasks.pop(channel_id, None)
            if not active_task.done():
                active_task.cancel()

    async def test_stop_button_retires_active_telegram_handoff(self) -> None:
        channel_id = 197

        class _FakeResponse:
            def is_done(self) -> bool:
                return False

            async def defer(self) -> None:
                return None

        class _FakeFollowup:
            def __init__(self) -> None:
                self.messages: list[tuple[str, bool]] = []

            async def send(
                self,
                message: str,
                *,
                ephemeral: bool,
                suppress_embeds: bool,
            ) -> None:
                self.messages.append((message, ephemeral))

        original_notice = build_handoff_notice(
            reason="manual follow-up needed",
            summary="initial issue",
            channel_id=channel_id,
            guild_id=2,
        )
        ticket_owner_user_id_by_channel[channel_id] = 777
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text=original_notice,
        )
        followup = _FakeFollowup()
        interaction = SimpleNamespace(
            channel=SimpleNamespace(id=channel_id),
            user=SimpleNamespace(id=777, name="owner"),
            response=_FakeResponse(),
            followup=followup,
            message=None,
        )

        try:
            with patch(
                "views.retire_handoff_notice",
                return_value=True,
            ) as retire_notice:
                await StopBotView().children[0].callback(interaction)

            retire_notice.assert_awaited_once_with(
                chat_id="123",
                message_id=456,
                fallback_message_text=build_dismissed_handoff_notice(
                    original_notice
                ),
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(
                followup.messages,
                [(
                    "Support bot stopped for this channel. "
                    "ySupport contributors are available for further inquiries.",
                    False,
                )],
            )
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_stop_button_persists_handoff_when_telegram_cleanup_fails(
        self,
    ) -> None:
        channel_id = 198

        class _FakeResponse:
            def is_done(self) -> bool:
                return False

            async def defer(self) -> None:
                return None

        class _FakeFollowup:
            async def send(
                self,
                _message: str,
                *,
                ephemeral: bool,
                suppress_embeds: bool,
            ) -> None:
                return None

        notice = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text="initial notice",
        )
        ticket_owner_user_id_by_channel[channel_id] = 777
        team_handoff_notice_by_channel[channel_id] = notice
        interaction = SimpleNamespace(
            channel=SimpleNamespace(id=channel_id),
            user=SimpleNamespace(id=777, name="owner"),
            response=_FakeResponse(),
            followup=_FakeFollowup(),
            message=None,
        )

        try:
            with (
                patch(
                    "views.retire_handoff_notice",
                    return_value=False,
                ) as retire_notice,
                patch("views.logging.warning") as warning,
            ):
                await StopBotView().children[0].callback(interaction)

            retire_notice.assert_awaited_once()
            self.assertIn(channel_id, stopped_channels)
            self.assertIs(team_handoff_notice_by_channel[channel_id], notice)
            persisted = state._read_json(
                state._TICKET_STATE_DIR / f"{channel_id}.json"
            )
            self.assertIsNotNone(persisted)
            self.assertTrue(persisted["stopped"])
            self.assertEqual(
                persisted["team_handoff_notice"]["telegram_message_id"],
                456,
            )
            warning.assert_called_once()
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    def test_build_discord_intents_enables_expected_intents(self) -> None:
        intents = _build_discord_intents()
        self.assertTrue(intents.message_content)
        self.assertTrue(intents.guilds)
        self.assertTrue(intents.messages)

    def test_reload_runtime_env_and_config_overrides_env_from_dotenv(self) -> None:
        original_value = config.DISCORD_BOT_TOKEN
        with patch("ysupport.load_dotenv") as mock_load, patch(
            "ysupport.importlib.reload",
            side_effect=lambda module: module,
        ) as mock_reload:
            _reload_runtime_env_and_config()
        mock_load.assert_called_once_with(config.BASE_DIR / ".env", override=True)
        mock_reload.assert_called_once_with(config)
        self.assertEqual(config.DISCORD_BOT_TOKEN, original_value)

    def test_run_ticket_bot_with_fatal_startup_backoff_retries_after_login_failure(self) -> None:
        attempts = {"count": 0}

        def fake_run_once():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise discord.errors.LoginFailure("bad token")
            return None

        with patch("ysupport._reload_runtime_env_and_config") as mock_reload, patch(
            "ysupport._run_ticket_bot_once",
            side_effect=fake_run_once,
        ) as mock_run_once, patch("ysupport.time.sleep") as mock_sleep:
            with patch.object(config, "DISCORD_FATAL_STARTUP_BACKOFF_SECONDS", 123.0):
                _run_ticket_bot_with_fatal_startup_backoff()

        self.assertEqual(mock_reload.call_count, 2)
        self.assertEqual(mock_run_once.call_count, 2)
        mock_sleep.assert_called_once_with(123.0)

    def test_run_ticket_bot_with_fatal_startup_backoff_clamps_minimum_sleep(self) -> None:
        attempts = {"count": 0}

        def fake_run_once():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise discord.errors.PrivilegedIntentsRequired(shard_id=None)
            return None

        with patch("ysupport._reload_runtime_env_and_config"), patch(
            "ysupport._run_ticket_bot_once",
            side_effect=fake_run_once,
        ), patch("ysupport.time.sleep") as mock_sleep:
            with patch.object(config, "DISCORD_FATAL_STARTUP_BACKOFF_SECONDS", 5.0):
                _run_ticket_bot_with_fatal_startup_backoff()

        mock_sleep.assert_called_once_with(60.0)

    async def test_notify_handoff_uses_model_summary_when_available(self) -> None:
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

        async def fake_send_handoff_notice(
            message_text: str,
            *,
            dismiss_enabled: bool = False,
        ):
            self.assertTrue(dismiss_enabled)
            notices.append(message_text)
            return True

        with (
            patch("ysupport.summarize_handoff_summary", new=fake_summarize_handoff_summary),
            patch("ysupport.send_handoff_notice", new=fake_send_handoff_notice),
        ):
            await _notify_handoff(
                reason="manual follow-up needed",
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
        notices: list[str] = []

        async def fake_summarize_handoff_summary(**kwargs) -> str | None:
            return None

        async def fake_send_handoff_notice(
            message_text: str,
            *,
            dismiss_enabled: bool = False,
        ):
            self.assertTrue(dismiss_enabled)
            notices.append(message_text)
            return True

        with (
            patch("ysupport.summarize_handoff_summary", new=fake_summarize_handoff_summary),
            patch("ysupport.send_handoff_notice", new=fake_send_handoff_notice),
        ):
            await _notify_handoff(
                reason="manual follow-up needed",
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

    async def test_public_trigger_outer_setup_failure_replies_and_clears_state(self) -> None:
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
            self.assertIn("remains under manual staff control", channel.sent_messages[0])
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

    async def test_plain_staff_message_takes_over_and_retires_handoff(self) -> None:
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
            content="Please add more ETH for gas and try again.",
            channel=channel,
            reference=None,
            created_at=datetime.now(timezone.utc),
            attachments=[],
        )

        async def active_turn() -> None:
            await asyncio.Event().wait()

        active_task = asyncio.create_task(active_turn())
        pending_tasks[channel_id] = active_task
        ticket_owner_user_id_by_channel[channel_id] = owner_id
        team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
            telegram_chat_id="123",
            telegram_message_id=456,
            reason="manual follow-up needed",
            message_text="Team review needed.",
        )
        bot = TicketBot(intents=discord.Intents.none())

        try:
            with (
                patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
                patch("views.retire_handoff_notice", return_value=True) as retire,
            ):
                await bot.on_message(message)

            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], owner_id)
            self.assertNotIn(channel_id, pending_tasks)
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(channel.sent_messages, [])
            retire.assert_awaited_once()
            with self.assertRaises(asyncio.CancelledError):
                await active_task
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
            self.assertIn("remains under manual staff control", request.turn_instruction)
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
                    "ysupport.send_handoff_notice",
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
                with patch("ysupport.send_handoff_notice", new=fake_send_handoff_notice):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

        self.assertEqual(len(fake_channel.sent_messages), 1)
        self.assertIn("This likely needs manual review.", fake_channel.sent_messages[0])
        self.assertIn("I've notified the support team", fake_channel.sent_messages[0])
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, fake_channel.sent_messages[0])
        self.assertEqual(len(handoff_notices), 1)
        self.assertIn(
            "<b>Reason</b>: manual strategy action is required to process rewards",
            handoff_notices[0],
        )

    async def test_failed_ticket_handoff_notification_does_not_park_or_claim_success(self) -> None:
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
                patch("ysupport.send_handoff_notice", return_value=None),
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

    async def test_waiting_for_team_ticket_stores_followup_and_sends_parked_ack(self) -> None:
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
        conversation_threads[channel_id] = [{"role": "assistant", "content": "Team notified."}]
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

    async def test_debounce_followup_is_added_without_losing_original_payload(
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
            self.assertEqual(
                fake_channel.sent_messages[0],
                "Got it. I’ve added your follow-up to your previous request for context, "
                "and I’m continuing to work on it now. Please wait for my response. "
                "There’s no need to resend anything.",
            )
            self.assertNotIn(channel_id, pending_messages)
            self.assertNotIn(channel_id, pending_attachments_by_channel)
            self.assertNotIn(channel_id, active_ticket_payloads)
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
            async def execute_turn(self, request: TicketTurnRequest, hooks=None):
                captured_requests.append(request)
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
                fake_channel.sent_messages[0],
                "Got it. I’ve added your follow-up to your previous request for context, "
                "and I’m continuing to work on it now. Please wait for my response. "
                "There’s no need to resend anything.",
            )
            self.assertNotIn(channel_id, pending_messages)
            self.assertNotIn(channel_id, active_ticket_payloads)
            self.assertNotIn(channel_id, pending_tasks)
        finally:
            task = pending_tasks.pop(channel_id, None)
            if task is not None:
                task.cancel()
            clear_ticket_channel_state(channel_id, delete_persisted=True)

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

        async def fake_edit_handoff_notice(*, chat_id: str, message_id: int, message_text: str) -> bool:
            edited_messages.append((chat_id, message_id, message_text))
            return True

        async def fake_internal_turn(**kwargs) -> str:
            internal_turn_calls.append(
                {
                    "prompt_text": kwargs["prompt_text"],
                    "instruction_text": kwargs["instruction_text"],
                    "attachments": kwargs["attachments"],
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
            followup_attachments=[
                {"url": "https://cdn.example/parked.png"}
            ],
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
            persisted = state._read_json(
                state._TICKET_STATE_DIR / f"{channel_id}.json"
            )
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertTrue(persisted["stopped"])
            self.assertEqual(persisted["stop_reason"], "manual_stop")
            self.assertIsNone(persisted["team_handoff_notice"])
            retired_messages.append(
                (chat_id, message_id, fallback_message_text)
            )
            return True

        bot = TicketBot(intents=discord.Intents.none())
        try:
            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch(
                    "ysupport.answer_telegram_callback_query",
                    new=fake_answer_callback,
                ),
                patch("ysupport.retire_handoff_notice", new=fake_retire_notice),
                patch("state.reset_ticket_codex_session") as reset_session,
                patch("ysupport._run_internal_instruction_turn") as internal_turn,
                patch("ysupport.send_long_message") as send_message,
            ):
                await bot._handle_telegram_handoff_update(update)

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

                await bot._handle_telegram_handoff_update(update)

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
                [(
                    "123",
                    456,
                    build_dismissed_handoff_notice(
                        "Dismissed. Handle in Discord."
                    ),
                )],
            )
            self.assertNotIn(channel_id, pending_tasks)
            self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 777)
            reset_session.assert_called_once_with(channel_id)
            internal_turn.assert_not_called()
            send_message.assert_not_called()

            payload = state._read_json(
                state._TICKET_STATE_DIR / f"{channel_id}.json"
            )
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
                    "ysupport.answer_telegram_callback_query",
                    new=fake_answer_callback,
                ),
                patch("ysupport.retire_handoff_notice") as retire_notice,
                patch("state.reset_ticket_codex_session") as reset_session,
            ):
                await bot._handle_telegram_handoff_update(
                    callback_update(
                        data="other_action",
                        chat_id="123",
                        message_id=456,
                    )
                )
                await bot._handle_telegram_handoff_update(
                    callback_update(
                        data=DISMISS_HANDOFF_CALLBACK_DATA,
                        chat_id="999",
                        message_id=456,
                    )
                )
                await bot._handle_telegram_handoff_update(
                    callback_update(
                        data=DISMISS_HANDOFF_CALLBACK_DATA,
                        chat_id="123",
                        message_id=999,
                    )
                )
                notice.status = "pending_delivery"
                await bot._handle_telegram_handoff_update(
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
        get_or_create_ticket_investigation_job(
            channel_id
        ).mark_escalated_to_human()
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
                    "ysupport.answer_telegram_callback_query",
                    return_value=True,
                ),
                patch(
                    "ysupport.retire_handoff_notice",
                    return_value=False,
                ) as retire_notice,
                patch("ysupport.logging.warning") as warning,
            ):
                await bot._handle_telegram_handoff_update(update)

            retire_notice.assert_awaited_once_with(
                chat_id="123",
                message_id=456,
                fallback_message_text=build_dismissed_handoff_notice(
                    original_notice
                ),
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "manual_stop")
            self.assertIn(channel_id, team_handoff_notice_by_channel)
            self.assertNotIn(channel_id, conversation_threads)
            self.assertNotIn(channel_id, ticket_investigation_jobs)
            persisted = state._read_json(
                state._TICKET_STATE_DIR / f"{channel_id}.json"
            )
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
                    "ysupport.answer_telegram_callback_query",
                    return_value=True,
                ),
                patch(
                    "ysupport.retire_handoff_notice",
                    return_value=True,
                ) as retry_retire,
            ):
                await bot._handle_telegram_handoff_update(update)

            retry_retire.assert_awaited_once_with(
                chat_id="123",
                message_id=456,
                fallback_message_text=build_dismissed_handoff_notice(
                    original_notice
                ),
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
        get_or_create_ticket_investigation_job(
            channel_id
        ).mark_escalated_to_human()
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
                    "ysupport.answer_telegram_callback_query",
                    return_value=True,
                ) as answer_callback,
                patch("ysupport.retire_handoff_notice") as retire_notice,
                patch("ysupport.logging.error") as log_error,
            ):
                await bot._handle_telegram_handoff_update(update)

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
                    bot,
                    "_deliver_telegram_handoff_reply",
                ) as deliver_reply,
            ):
                await bot._handle_telegram_handoff_update(reply_update)
            deliver_reply.assert_not_awaited()

            with (
                patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
                patch(
                    "ysupport.answer_telegram_callback_query",
                    return_value=True,
                ) as retry_answer,
                patch(
                    "ysupport.retire_handoff_notice",
                    return_value=True,
                ) as retry_retire,
            ):
                await bot._handle_telegram_handoff_update(update)

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
            persisted = state._read_json(
                state._TICKET_STATE_DIR / f"{channel_id}.json"
            )
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertTrue(persisted["stopped"])
            self.assertIsNone(persisted["team_handoff_notice"])
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

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
                patch("ysupport.edit_handoff_notice", return_value=True),
                patch(
                    "ysupport._run_internal_instruction_turn",
                    new=fail_synthesis,
                ),
                patch("ysupport.send_long_message") as send_message,
            ):
                delivered = await bot._deliver_telegram_handoff_reply(
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
        get_or_create_ticket_investigation_job(
            channel_id
        ).mark_escalated_to_human()
        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel

        async def stop_during_synthesis(**kwargs) -> str:
            self.assertTrue(state.stop_ticket_channel(channel_id))
            return "This reply must not be posted."

        try:
            with (
                patch("ysupport.edit_handoff_notice", return_value=True),
                patch(
                    "ysupport._run_internal_instruction_turn",
                    new=stop_during_synthesis,
                ),
                patch(
                    "ysupport.reset_ticket_codex_session",
                ) as reset_session,
                patch("ysupport.send_long_message") as send_message,
            ):
                delivered = await bot._deliver_telegram_handoff_reply(
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
                    bot,
                    "_resume_pending_telegram_handoff_replies",
                    new=fake_recover_pending,
                ),
                patch(
                    "ysupport.fetch_telegram_updates",
                    new=fake_fetch_updates,
                ),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await bot._telegram_handoff_reply_loop()

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
                    bot,
                    "_deliver_telegram_handoff_reply",
                    new=fail_delivery,
                ),
                patch(
                    "ysupport.edit_handoff_notice",
                    new=edit_failed_status,
                ),
            ):
                await bot._resume_pending_telegram_handoff_replies()
                self.assertEqual(delivery_calls, 0)

                notice.status = "pending_delivery"
                notice.pending_reply_text = "Tell the user the transaction is queued."
                await bot._resume_pending_telegram_handoff_replies()
                await bot._resume_pending_telegram_handoff_replies()
                self.assertEqual(delivery_calls, 1)

                notice.telegram_message_id = 457
                await bot._resume_pending_telegram_handoff_replies()
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
            raise TelegramApiError("Telegram API call getUpdates failed: read timed out")

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            raise asyncio.CancelledError

        try:
            with (
                patch(
                    "ysupport.fetch_telegram_updates",
                    new=fake_fetch_telegram_updates,
                ),
                patch("ysupport.asyncio.sleep", new=fake_sleep),
                patch("ysupport.logging.warning") as mock_warning,
                patch("ysupport.logging.error") as mock_error,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await bot._telegram_handoff_reply_loop()

            self.assertEqual(sleep_calls, [5])
            mock_warning.assert_called_once()
            self.assertIn(
                "Telegram polling temporarily unavailable",
                mock_warning.call_args.args[0],
            )
            mock_error.assert_not_called()
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
