import tests as _test_environment  # noqa: F401

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, PropertyMock


import discord

import config
import state
from state import (
    channels_awaiting_initial_button_press,
    TeamHandoffNotice,
    clear_ticket_channel_state,
    pending_tasks,
    stopped_channels,
    team_handoff_notice_by_channel,
    ticket_owner_user_id_by_channel,
)
from handoff import (
    build_dismissed_handoff_notice,
    build_handoff_notice,
)
from views import InitialInquiryView, StopBotView
from discord_support_runtime import (
    _notify_handoff,
    _refresh_discord_attachment_urls,
)
from ysupport import (
    TicketBot,
    _build_discord_intents,
    _reload_runtime_env_and_config,
    _run_ticket_bot_with_fatal_startup_backoff,
)
from ticket_channel_lifecycle import (
    initialize_ticket_channel,
)

from tests.ticket_flow_test_support import TicketFlowTestCase


class TicketFlowTests(TicketFlowTestCase):
    async def test_ticket_channel_initialization_records_owner_and_sends_intake(
        self,
    ) -> None:
        channel_id = 91001

        class _TextChannel:
            def __init__(self) -> None:
                self.id = channel_id
                self.name = "ticket-91001"
                self.category = SimpleNamespace(id=92001)
                self.sent = []

            async def send(self, content, **kwargs):
                self.sent.append((content, kwargs))

        channel = _TextChannel()

        async def pending_turn() -> None:
            await asyncio.Event().wait()

        old_task = asyncio.create_task(pending_turn())
        pending_tasks[channel_id] = old_task
        try:
            with (
                patch("ticket_channel_lifecycle.discord.TextChannel", _TextChannel),
                patch.object(config, "CATEGORY_CONTEXT_MAP", {92001: "yearn"}),
                patch(
                    "ticket_channel_lifecycle.asyncio.sleep",
                    new=AsyncMock(),
                ) as sleep,
                patch(
                    "ticket_channel_lifecycle._detect_ticket_owner_user_id",
                    new=AsyncMock(return_value=93001),
                ) as detect_owner,
            ):
                await initialize_ticket_channel(channel)

            self.assertTrue(old_task.cancelled)
            self.assertNotIn(channel_id, pending_tasks)
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 93001)
            self.assertIn(channel_id, channels_awaiting_initial_button_press)
            sleep.assert_awaited_once_with(1.5)
            detect_owner.assert_awaited_once_with(channel)
            self.assertEqual(len(channel.sent), 1)
            welcome, send_kwargs = channel.sent[0]
            self.assertIn("Welcome to Yearn Support", welcome)
            self.assertIsInstance(send_kwargs["view"], InitialInquiryView)
            self.assertTrue(send_kwargs["suppress_embeds"])
        finally:
            clear_ticket_channel_state(channel_id, delete_persisted=True)

    async def test_on_ready_starts_services_and_close_cancels_cleanup(self) -> None:
        bot = TicketBot(intents=discord.Intents.none())
        cleanup_started = asyncio.Event()
        ticket_task_started = asyncio.Event()

        async def cleanup_loop() -> None:
            cleanup_started.set()
            await asyncio.Event().wait()

        async def ticket_turn() -> None:
            ticket_task_started.set()
            await asyncio.Event().wait()

        ticket_task = asyncio.create_task(ticket_turn())
        pending_tasks[95002] = ticket_task

        with (
            patch.object(
                type(bot),
                "user",
                new_callable=PropertyMock,
                return_value=SimpleNamespace(id=95001),
            ),
            patch(
                "ysupport.hydrate_persisted_team_handoff_states",
                return_value=2,
            ) as hydrate,
            patch.object(bot.telegram_handoffs, "start", return_value=True) as start,
            patch.object(
                bot.telegram_handoffs,
                "close",
                new=AsyncMock(),
            ) as controller_close,
            patch.object(bot, "_state_cleanup_loop", new=cleanup_loop),
            patch.object(discord.Client, "close", new=AsyncMock()) as client_close,
        ):
            await bot.on_ready()
            cleanup_task = bot._state_cleanup_task
            self.assertIsNotNone(cleanup_task)
            await cleanup_started.wait()
            await ticket_task_started.wait()
            await bot.close()

        hydrate.assert_called_once_with()
        start.assert_called_once_with()
        controller_close.assert_awaited_once_with()
        client_close.assert_awaited_once_with()
        self.assertIsNone(bot._state_cleanup_task)
        assert cleanup_task is not None
        with self.assertRaises(asyncio.CancelledError):
            await cleanup_task
        self.assertTrue(ticket_task.cancelled())
        self.assertNotIn(95002, pending_tasks)

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
                await InitialInquiryView().interaction_check(contributor_interaction)
            )
            self.assertTrue(await StopBotView().interaction_check(owner_interaction))
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

        self.assertFalse(await InitialInquiryView().interaction_check(fake_interaction))
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
                fallback_message_text=build_dismissed_handoff_notice(original_notice),
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(
                followup.messages,
                [
                    (
                        "Support bot stopped for this channel. "
                        "ySupport contributors are available for further inquiries.",
                        False,
                    )
                ],
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
            persisted = state._read_json(state._TICKET_STATE_DIR / f"{channel_id}.json")
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
        with (
            patch("ysupport.load_dotenv") as mock_load,
            patch(
                "ysupport.importlib.reload",
                side_effect=lambda module: module,
            ) as mock_reload,
        ):
            _reload_runtime_env_and_config()
        mock_load.assert_called_once_with(config.BASE_DIR / ".env", override=True)
        mock_reload.assert_called_once_with(config)
        self.assertEqual(config.DISCORD_BOT_TOKEN, original_value)

    def test_run_ticket_bot_with_fatal_startup_backoff_retries_after_login_failure(
        self,
    ) -> None:
        attempts = {"count": 0}

        def fake_run_once():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise discord.errors.LoginFailure("bad token")
            return None

        with (
            patch("ysupport._reload_runtime_env_and_config") as mock_reload,
            patch(
                "ysupport._run_ticket_bot_once",
                side_effect=fake_run_once,
            ) as mock_run_once,
            patch("ysupport.time.sleep") as mock_sleep,
        ):
            with patch.object(config, "DISCORD_FATAL_STARTUP_BACKOFF_SECONDS", 123.0):
                _run_ticket_bot_with_fatal_startup_backoff()

        self.assertEqual(mock_reload.call_count, 2)
        self.assertEqual(mock_run_once.call_count, 2)
        mock_sleep.assert_called_once_with(123.0)

    def test_run_ticket_bot_with_fatal_startup_backoff_clamps_minimum_sleep(
        self,
    ) -> None:
        attempts = {"count": 0}

        def fake_run_once():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise discord.errors.PrivilegedIntentsRequired(shard_id=None)
            return None

        with (
            patch("ysupport._reload_runtime_env_and_config"),
            patch(
                "ysupport._run_ticket_bot_once",
                side_effect=fake_run_once,
            ),
            patch("ysupport.time.sleep") as mock_sleep,
        ):
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
            patch(
                "discord_support_runtime.summarize_handoff_summary",
                new=fake_summarize_handoff_summary,
            ),
            patch(
                "discord_support_runtime.send_handoff_notice",
                new=fake_send_handoff_notice,
            ),
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

    async def test_notify_handoff_falls_back_to_raw_summary_when_model_summary_missing(
        self,
    ) -> None:
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
            patch(
                "discord_support_runtime.summarize_handoff_summary",
                new=fake_summarize_handoff_summary,
            ),
            patch(
                "discord_support_runtime.send_handoff_notice",
                new=fake_send_handoff_notice,
            ),
        ):
            await _notify_handoff(
                reason="manual follow-up needed",
                summary="yes please tell them",
                channel_id=1506309610192113917,
                guild_id=734804446353031319,
                source="ticket",
                recent_user_messages=[
                    "withdraw button spins forever on the vault page"
                ],
                known_facts=["chain: Ethereum"],
            )

        self.assertEqual(len(notices), 1)
        self.assertIn("<b>Summary</b>: yes please tell them", notices[0])
