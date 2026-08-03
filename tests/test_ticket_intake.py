import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import patch


import discord

import config
from handoff import TelegramSentMessage
import yearn_targets
from state import (
    BotRunContext,
    TicketInvestigationJob,
    clear_team_handoff_notice,
    clear_ticket_investigation_job,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    last_bot_reply_ts_by_channel,
    last_wallet_by_channel,
    pending_messages,
    pending_wallet_confirmation_by_channel,
    stopped_channels,
    stop_reasons_by_channel,
    team_handoff_notice_by_channel,
    ticket_investigation_jobs,
    ticket_owner_user_id_by_channel,
)
from ticket_intake import (
    canonicalize_current_user_message as _canonicalize_current_user_message,
    prepare_ticket_turn_input,
)
from ticket_investigation.runtime import (
    TicketAgentFlowOutcome,
    TicketTurnRequest,
)
from ysupport import (
    TicketBot,
)

class WalletCanonicalizationTests(unittest.TestCase):
    def test_canonicalize_current_user_message_uses_resolved_wallet_for_data_turn(self) -> None:
        context = BotRunContext(
            channel_id=61,
            project_context="yearn",
            initial_button_intent="data_deposit_check",
        )

        self.assertEqual(
            _canonicalize_current_user_message(
                "0xabcDEFabcDEFabcDEFabcDEFabcDEFabcDEFabcd",
                context,
                "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD",
            ),
            "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD",
        )

    def test_canonicalize_current_user_message_uses_resolved_wallet_for_wallet_labeled_followup(self) -> None:
        context = BotRunContext(channel_id=62, project_context="yearn")

        self.assertEqual(
            _canonicalize_current_user_message(
                "wallet 0xabcDEFabcDEFabcDEFabcDEFabcDEFabcDEFabcd",
                context,
                "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD",
            ),
            "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD",
        )

    def test_canonicalize_current_user_message_preserves_non_address_text(self) -> None:
        context = BotRunContext(channel_id=63, project_context="yearn")

        self.assertEqual(
            _canonicalize_current_user_message(
                "How do I withdraw from this vault?",
                context,
                "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD",
            ),
            "How do I withdraw from this vault?",
        )


class WalletConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_replacement_wallet_updates_structured_ticket_evidence(
        self,
    ) -> None:
        channel_id = 64
        old_wallet = "0x1111111111111111111111111111111111111111"
        new_wallet = "0x2222222222222222222222222222222222222222"
        job = TicketInvestigationJob(channel_id=channel_id)
        job.remember_wallet(old_wallet)
        last_wallet_by_channel[channel_id] = old_wallet
        pending_wallet_confirmation_by_channel[channel_id] = new_wallet
        try:
            preparation = await prepare_ticket_turn_input(
                channel_id=channel_id,
                run_context=BotRunContext(
                    channel_id=channel_id,
                    project_context="yearn",
                ),
                investigation_job=job,
                aggregated_text="yes",
            )
        finally:
            last_wallet_by_channel.pop(channel_id, None)
            pending_wallet_confirmation_by_channel.pop(channel_id, None)

        self.assertEqual(job.evidence.wallet, new_wallet)
        self.assertEqual(preparation.system_hints, [])


class _FakeTypingContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeDiscordChannel:
    def __init__(self, channel_id: int):
        self.id = channel_id
        self.sent_messages: list[str] = []
        self.category = None
        self.guild = None

    async def send(self, message: str, *args, **kwargs):
        self.sent_messages.append(message)

    def typing(self):
        return _FakeTypingContext()


class _FakeOriginalAuthor:
    def __init__(self, author_id: int, *, bot: bool = False):
        self.id = author_id
        self.bot = bot


class _FakeOriginalMessage:
    def __init__(self, author_id: int, content: str):
        self.author = _FakeOriginalAuthor(author_id)
        self.content = content
        self.replies: list[tuple[str, dict]] = []

    async def reply(self, content: str, **kwargs):
        self.replies.append((content, kwargs))


class _FakePublicChannel(_FakeDiscordChannel):
    def __init__(self, channel_id: int, original_message: _FakeOriginalMessage):
        super().__init__(channel_id)
        self._original_message = original_message

    async def fetch_message(self, message_id: int):
        return self._original_message


class _FakeTriggerReference:
    def __init__(self, message_id: int):
        self.message_id = message_id


class _FakeTriggerAuthor:
    def __init__(self, name: str):
        self.name = name


class _FakeTriggerMessage:
    def __init__(self, channel: _FakePublicChannel, *, reference_message_id: int):
        self.id = 999
        self.channel = channel
        self.reference = _FakeTriggerReference(reference_message_id)
        self.author = _FakeTriggerAuthor("tester")
        self.deleted = False

    async def delete(self):
        self.deleted = True


class TicketBotWalletFlowTests(unittest.IsolatedAsyncioTestCase):
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
        summary_patcher = patch(
            "discord_support_runtime.summarize_handoff_summary",
            return_value=None,
        )
        summary_patcher.start()
        self.addCleanup(summary_patcher.stop)

    def test_extract_yearn_vault_url_target_parses_supported_urls(self) -> None:
        self.assertEqual(
            yearn_targets.extract_yearn_vault_url_target(
                "https://yearn.fi/vaults/1/0xBe53A109B494E5c9f97b9Cd39Fe969BE68BF6204"
            ),
            ("ethereum", "0xBe53A109B494E5c9f97b9Cd39Fe969BE68BF6204"),
        )
        self.assertEqual(
            yearn_targets.extract_yearn_vault_url_target(
                "https://legacy.yearn.fi/v3/747474/0x80c34BD3A3569E126e7055831036aa7b212cB159"
            ),
            ("katana", "0x80c34BD3A3569E126e7055831036aa7b212cB159"),
        )

    def test_extract_address_or_ens_accepts_short_valid_labels(self) -> None:
        self.assertEqual(yearn_targets.extract_address_or_ens("wallet a.eth"), "a.eth")
        self.assertEqual(
            yearn_targets.extract_address_or_ens("wallet ab.eth"), "ab.eth"
        )

    def test_extract_address_or_ens_rejects_trailing_hyphen_label(self) -> None:
        self.assertIsNone(yearn_targets.extract_address_or_ens("wallet a-.eth"))

    async def test_resolve_yearn_address_target_prefers_known_strategy_match(self) -> None:
        address = "0x1111111111111111111111111111111111111111"
        with patch(
            "yearn_targets._get_ydaemon_address_index",
            return_value={
                "0x1111111111111111111111111111111111111111": [
                    {
                        "kind": "strategy",
                        "chain": "ethereum",
                        "label": "Morpho Looper",
                        "vault_address": "0x2222222222222222222222222222222222222222",
                        "vault_label": "yvUSD",
                    }
                ]
            },
        ):
            resolved = await yearn_targets.resolve_yearn_address_target(address, chain_hint="ethereum")

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "strategy")
        self.assertEqual(resolved.chain, "ethereum")
        self.assertEqual(resolved.label, "Morpho Looper")
        self.assertEqual(resolved.vault_label, "yvUSD")

    async def test_resolve_yearn_address_target_falls_back_to_contract_profile(self) -> None:
        address = "0x3333333333333333333333333333333333333333"
        with (
            patch("yearn_targets._get_ydaemon_address_index", return_value={}),
            patch("yearn_targets.get_web3_instance", return_value=object()),
            patch(
                "yearn_targets.inspect_contract_profile",
                return_value={"kind": "erc20_like", "has_code": True, "symbol": "TEST", "name": "Test Contract"},
            ),
        ):
            resolved = await yearn_targets.resolve_yearn_address_target(address, chain_hint="ethereum")

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.kind, "contract_unknown")
        self.assertTrue(resolved.is_contract)
        self.assertEqual(resolved.contract_profile_kind, "erc20_like")

    async def test_process_ticket_message_uses_same_resolved_wallet_for_ack_and_turn_input(self) -> None:
        channel_id = 64
        fake_channel = _FakeDiscordChannel(channel_id)
        captured_requests: list[TicketTurnRequest] = []

        class _FakeExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                captured_requests.append(request)
                request.investigation_job.complete_specialist_turn("data")
                request.investigation_job.mark_waiting_for_user()
                return type(
                    "_FakeWorkerResult",
                    (),
                    {
                        "flow_outcome": TicketAgentFlowOutcome(
                            raw_final_reply="No deposits found.",
                            conversation_history=[],
                            completed_agent_key="data",
                            requires_human_handoff=False,
                        ),
                        "updated_job": request.investigation_job,
                    },
                )()

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FakeExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="data_deposit_check",
        )

        pending_messages[channel_id] = "0xabcDEFabcDEFabcDEFabcDEFabcDEFabcDEFabcd"
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        last_wallet_by_channel.pop(channel_id, None)
        pending_wallet_confirmation_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        resolved_wallet = "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD"
        try:
            with patch("chain_access.resolve_ens_name", return_value=resolved_wallet):
                with patch(
                    "ticket_intake.yearn_targets.resolve_yearn_address_target",
                    return_value=yearn_targets.ResolvedYearnAddressTarget(
                        address=resolved_wallet,
                        kind="eoa_or_missing_code",
                        chain=None,
                        is_contract=False,
                    ),
                ):
                    with patch("ysupport.send_long_message", new=fake_channel.send):
                        with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                            await bot.process_ticket_message(channel_id, run_context)
            observed_last_wallet = last_wallet_by_channel.get(channel_id)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_wallet_by_channel.pop(channel_id, None)
            pending_wallet_confirmation_by_channel.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(fake_channel.sent_messages[0], f"Thank you. I've received the address `{resolved_wallet}` and am looking up the information now. This may take a moment...")
        self.assertEqual(captured_requests[0].input_list[-1]["content"], resolved_wallet)
        self.assertTrue(
            any(
                item.get("role") == "system"
                and resolved_wallet in item.get("content", "")
                for item in captured_requests[0].input_list
            )
        )
        self.assertEqual(observed_last_wallet, resolved_wallet)

    async def test_process_ticket_message_treats_known_vault_address_as_target_not_wallet(self) -> None:
        channel_id = 164
        fake_channel = _FakeDiscordChannel(channel_id)
        captured_requests: list[TicketTurnRequest] = []

        class _FakeExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                captured_requests.append(request)
                request.investigation_job.complete_specialist_turn("data")
                request.investigation_job.mark_waiting_for_user()
                return type(
                    "_FakeWorkerResult",
                    (),
                    {
                        "flow_outcome": TicketAgentFlowOutcome(
                            raw_final_reply="Vault info.",
                            conversation_history=[],
                            completed_agent_key="data",
                            requires_human_handoff=False,
                        ),
                        "updated_job": request.investigation_job,
                    },
                )()

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FakeExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="data_vault_search",
        )

        vault_address = "0xBe53A109B494E5c9f97b9Cd39Fe969BE68BF6204"
        pending_messages[channel_id] = vault_address
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        last_wallet_by_channel.pop(channel_id, None)
        pending_wallet_confirmation_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        try:
            with patch("chain_access.resolve_ens_name", return_value=vault_address):
                with patch(
                    "ticket_intake.yearn_targets.resolve_yearn_address_target",
                    return_value=yearn_targets.ResolvedYearnAddressTarget(
                        address=vault_address,
                        kind="vault",
                        chain="ethereum",
                        label="USDC V3 Vault",
                        vault_address=vault_address,
                        vault_label="USDC V3 Vault",
                        is_contract=True,
                    ),
                ):
                    with patch("ysupport.send_long_message", new=fake_channel.send):
                        with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                            await bot.process_ticket_message(channel_id, run_context)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_wallet_by_channel.pop(channel_id, None)
            pending_wallet_confirmation_by_channel.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(captured_requests), 1)
        self.assertIn("identified `USDC V3 Vault` as a Yearn vault", fake_channel.sent_messages[0])
        self.assertIsNone(last_wallet_by_channel.get(channel_id))
        self.assertTrue(
            any(
                item.get("role") == "system"
                and "known Yearn vault" in item.get("content", "")
                for item in captured_requests[0].input_list
            )
        )

    async def test_process_ticket_message_does_not_assume_unknown_contract_is_wallet(self) -> None:
        channel_id = 264
        fake_channel = _FakeDiscordChannel(channel_id)
        captured_requests: list[TicketTurnRequest] = []

        class _FakeExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                captured_requests.append(request)
                request.investigation_job.complete_specialist_turn("data")
                request.investigation_job.mark_waiting_for_user()
                return type(
                    "_FakeWorkerResult",
                    (),
                    {
                        "flow_outcome": TicketAgentFlowOutcome(
                            raw_final_reply="Need clarification.",
                            conversation_history=[],
                            completed_agent_key="data",
                            requires_human_handoff=False,
                        ),
                        "updated_job": request.investigation_job,
                    },
                )()

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FakeExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="data_deposit_check",
        )

        contract_address = "0x4444444444444444444444444444444444444444"
        pending_messages[channel_id] = contract_address
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        last_wallet_by_channel.pop(channel_id, None)
        pending_wallet_confirmation_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        try:
            with patch("chain_access.resolve_ens_name", return_value=contract_address):
                with patch(
                    "ticket_intake.yearn_targets.resolve_yearn_address_target",
                    return_value=yearn_targets.ResolvedYearnAddressTarget(
                        address=contract_address,
                        kind="contract_unknown",
                        chain="ethereum",
                        label="Unknown Contract",
                        is_contract=True,
                    ),
                ):
                    with patch("ysupport.send_long_message", new=fake_channel.send):
                        with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                            await bot.process_ticket_message(channel_id, run_context)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_wallet_by_channel.pop(channel_id, None)
            pending_wallet_confirmation_by_channel.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(captured_requests), 1)
        self.assertIsNone(last_wallet_by_channel.get(channel_id))
        self.assertFalse(
            any(
                isinstance(message, str) and "I've received the address" in message
                for message in fake_channel.sent_messages
            )
        )
        self.assertTrue(
            any(
                item.get("role") == "system"
                and "Do not assume it is the user's wallet" in item.get("content", "")
                for item in captured_requests[0].input_list
            )
        )

    async def test_process_ticket_message_tx_hash_keeps_message_context_over_wallet_canonicalization(self) -> None:
        channel_id = 364
        fake_channel = _FakeDiscordChannel(channel_id)
        captured_requests: list[TicketTurnRequest] = []

        class _FakeExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                captured_requests.append(request)
                request.investigation_job.complete_specialist_turn("data")
                request.investigation_job.mark_waiting_for_user()
                return type(
                    "_FakeWorkerResult",
                    (),
                    {
                        "flow_outcome": TicketAgentFlowOutcome(
                            raw_final_reply="Checking the transaction.",
                            conversation_history=[],
                            completed_agent_key="data",
                            requires_human_handoff=False,
                        ),
                        "updated_job": request.investigation_job,
                    },
                )()

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FakeExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="data_deposit_check",
        )

        resolved_wallet = "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD"
        tx_hash = "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        prompt_text = (
            f"I transferred USDC from {resolved_wallet} in tx {tx_hash} and it did not show up."
        )
        pending_messages[channel_id] = prompt_text
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        last_wallet_by_channel.pop(channel_id, None)
        pending_wallet_confirmation_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        try:
            with patch("chain_access.resolve_ens_name", return_value=resolved_wallet):
                with patch(
                    "ticket_intake.yearn_targets.resolve_yearn_address_target",
                    return_value=yearn_targets.ResolvedYearnAddressTarget(
                        address=resolved_wallet,
                        kind="eoa_or_missing_code",
                        chain="ethereum",
                        is_contract=False,
                    ),
                ):
                    with patch("ysupport.send_long_message", new=fake_channel.send):
                        with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                            await bot.process_ticket_message(channel_id, run_context)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_wallet_by_channel.pop(channel_id, None)
            pending_wallet_confirmation_by_channel.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(captured_requests[0].input_list[-1]["content"], prompt_text)
        self.assertIsNone(last_wallet_by_channel.get(channel_id))
        self.assertFalse(
            any(
                isinstance(message, str) and "I've received the address" in message
                for message in fake_channel.sent_messages
            )
        )
        self.assertTrue(
            any(
                item.get("role") == "system"
                and "Investigate the transaction directly before falling back to wallet-position lookup" in item.get("content", "")
                for item in captured_requests[0].input_list
            )
        )

    async def test_process_ticket_message_uses_vault_url_as_target_not_wallet(self) -> None:
        channel_id = 464
        fake_channel = _FakeDiscordChannel(channel_id)
        captured_requests: list[TicketTurnRequest] = []

        class _FakeExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                captured_requests.append(request)
                request.investigation_job.complete_specialist_turn("data")
                request.investigation_job.mark_waiting_for_user()
                return type(
                    "_FakeWorkerResult",
                    (),
                    {
                        "flow_outcome": TicketAgentFlowOutcome(
                            raw_final_reply="Vault info.",
                            conversation_history=[],
                            completed_agent_key="data",
                            requires_human_handoff=False,
                        ),
                        "updated_job": request.investigation_job,
                    },
                )()

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FakeExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="data_vault_search",
        )

        vault_address = "0xBe53A109B494E5c9f97b9Cd39Fe969BE68BF6204"
        prompt_text = f"https://yearn.fi/vaults/1/{vault_address}"
        pending_messages[channel_id] = prompt_text
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        last_wallet_by_channel.pop(channel_id, None)
        pending_wallet_confirmation_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        try:
            with patch("ticket_intake.yearn_targets.resolve_yearn_address_target") as mock_resolve_target:
                mock_resolve_target.return_value = yearn_targets.ResolvedYearnAddressTarget(
                    address=vault_address,
                    kind="vault",
                    chain="ethereum",
                    label="USDC V3 Vault",
                    vault_address=vault_address,
                    vault_label="USDC V3 Vault",
                    is_contract=True,
                )
                with patch("ysupport.send_long_message", new=fake_channel.send):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_wallet_by_channel.pop(channel_id, None)
            pending_wallet_confirmation_by_channel.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(captured_requests), 1)
        self.assertIsNone(last_wallet_by_channel.get(channel_id))
        self.assertTrue(
            any(
                item.get("role") == "system"
                and "User referenced a Yearn vault page URL" in item.get("content", "")
                for item in captured_requests[0].input_list
            )
        )

    async def test_process_ticket_message_applies_outer_boundary_before_executor(self) -> None:
        channel_id = 65
        fake_channel = _FakeDiscordChannel(channel_id)

        class _FailingExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                raise AssertionError("Boundary reply should stop before executor runs.")

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FailingExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="other_free_form",
        )

        pending_messages[channel_id] = "We want a marketing partnership with Yearn"
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        async def fake_boundary(_text: str):
            return {
                "classification": "business_boundary",
                "tripwire_triggered": True,
                "message": "Boundary reply",
            }

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch("ysupport._outer_support_boundary_result", new=fake_boundary):
                with patch("ysupport.send_long_message", new=fake_send_long_message):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
            self.assertEqual(fake_channel.sent_messages, ["Boundary reply"])
            self.assertIn(channel_id, stopped_channels)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            stopped_channels.discard(channel_id)
            clear_ticket_investigation_job(channel_id)

    async def test_access_handoff_runs_executor_and_retains_reply_bridge(self) -> None:
        channel_id = 66
        fake_channel = _FakeDiscordChannel(channel_id)

        updated_job = TicketInvestigationJob(channel_id=channel_id)
        updated_job.mark_escalated_to_human()

        class _AccessExecutor:
            def __init__(self) -> None:
                self.calls = []

            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                self_inner.calls.append(request)
                return type(
                    "_Result",
                    (),
                    {
                        "flow_outcome": TicketAgentFlowOutcome(
                            raw_final_reply=(
                                "The documented verification steps are complete, "
                                "so a moderator must inspect the remaining access state."
                            ),
                            conversation_history=[],
                            completed_agent_key=None,
                            requires_human_handoff=True,
                        ),
                        "updated_job": updated_job,
                    },
                )()

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _AccessExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="other_free_form",
        )

        pending_messages[channel_id] = "I finished verification but still cannot access the Discord."
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        clear_ticket_investigation_job(channel_id)

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with (
                patch("ysupport.send_long_message", new=fake_send_long_message),
                patch(
                    "discord_support_runtime.send_handoff_notice",
                    return_value=TelegramSentMessage(
                        chat_id="123",
                        message_id=456,
                    ),
                ),
                patch("ysupport.discord.TextChannel", _FakeDiscordChannel),
            ):
                await bot.process_ticket_message(channel_id, run_context)

            self.assertEqual(len(bot.investigation_executor.calls), 1)
            self.assertEqual(len(fake_channel.sent_messages), 1)
            self.assertIn("moderator", fake_channel.sent_messages[0])
            self.assertIn("notified", fake_channel.sent_messages[0])
            self.assertNotIn(channel_id, stopped_channels)
            self.assertIn(channel_id, team_handoff_notice_by_channel)
            self.assertEqual(
                team_handoff_notice_by_channel[channel_id].telegram_message_id,
                456,
            )
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            stopped_channels.discard(channel_id)
            clear_team_handoff_notice(channel_id)
            clear_ticket_investigation_job(channel_id)

    async def test_process_ticket_message_keeps_security_process_boundary_ticket_active(self) -> None:
        channel_id = 67
        fake_channel = _FakeDiscordChannel(channel_id)

        class _FailingExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                raise AssertionError("Security-process boundary should stop before executor runs.")

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FailingExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="other_free_form",
        )

        pending_messages[channel_id] = "I want to report a bug bounty issue."
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        stopped_channels.discard(channel_id)
        clear_ticket_investigation_job(channel_id)

        async def fake_boundary(_text: str):
            return {
                "classification": "security_process_boundary",
                "tripwire_triggered": True,
                "message": "Please use the Yearn security process.",
            }

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch("ysupport._outer_support_boundary_result", new=fake_boundary):
                with patch("ysupport.send_long_message", new=fake_send_long_message):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
            self.assertEqual(
                fake_channel.sent_messages,
                ["Please use the Yearn security process."],
            )
            self.assertNotIn(channel_id, stopped_channels)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            stopped_channels.discard(channel_id)
            clear_ticket_investigation_job(channel_id)

    async def test_process_ticket_message_security_exception_does_not_notify_telegram(self) -> None:
        channel_id = 167
        fake_channel = _FakeDiscordChannel(channel_id)

        class _FailingExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                raise AssertionError("Security-process boundary should stop before executor runs.")

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FailingExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="other_free_form",
        )

        pending_messages[channel_id] = (
            "I need to report a vulnerability, but Immunefi KYC is blocked and zkPassport is not working."
        )
        conversation_threads[channel_id] = []
        last_bot_reply_ts_by_channel.pop(channel_id, None)
        stopped_channels.discard(channel_id)
        clear_ticket_investigation_job(channel_id)
        clear_team_handoff_notice(channel_id)

        async def fake_boundary(_text: str):
            return {
                "classification": "security_process_boundary",
                "tripwire_triggered": True,
                "message": config.SECURITY_ALTERNATE_CONTACT_MESSAGE,
            }

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch("ysupport._outer_support_boundary_result", new=fake_boundary):
                with patch("ysupport.send_long_message", new=fake_send_long_message):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
            self.assertEqual(fake_channel.sent_messages, [config.SECURITY_ALTERNATE_CONTACT_MESSAGE])
            self.assertNotIn(channel_id, stopped_channels)
            self.assertIsNone(team_handoff_notice_by_channel.get(channel_id))
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            stopped_channels.discard(channel_id)
            clear_team_handoff_notice(channel_id)
            clear_ticket_investigation_job(channel_id)

    async def test_process_ticket_message_runtime_error_preserves_ticket_context(self) -> None:
        channel_id = 68
        fake_channel = _FakeDiscordChannel(channel_id)

        class _FailingExecutor:
            async def execute_turn(self_inner, request: TicketTurnRequest, hooks=None):
                raise RuntimeError("boom")

        bot = TicketBot(intents=discord.Intents.none())
        bot.get_channel = lambda _channel_id: fake_channel
        bot.investigation_executor = _FailingExecutor()

        run_context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="investigate_issue",
        )

        pending_messages[channel_id] = "Something failed."
        conversation_threads[channel_id] = [{"role": "user", "content": "Earlier context"}]
        last_wallet_by_channel[channel_id] = "0xabc"
        ticket_owner_user_id_by_channel[channel_id] = 123
        job = get_or_create_ticket_investigation_job(channel_id)
        job.record_requested_intent("investigate_issue")
        last_bot_reply_ts_by_channel.pop(channel_id, None)

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch("ysupport.send_long_message", new=fake_send_long_message):
                with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                    await bot.process_ticket_message(channel_id, run_context)

            self.assertEqual(fake_channel.sent_messages, ["An unexpected error occurred."])
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "runtime_error")
            self.assertEqual(ticket_owner_user_id_by_channel[channel_id], 123)
            self.assertEqual(last_wallet_by_channel[channel_id], "0xabc")
            self.assertIn(channel_id, ticket_investigation_jobs)
            self.assertIsNone(team_handoff_notice_by_channel.get(channel_id))
            self.assertEqual(
                conversation_threads[channel_id],
                [{"role": "user", "content": "Earlier context"}],
            )
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_wallet_by_channel.pop(channel_id, None)
            ticket_owner_user_id_by_channel.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)
