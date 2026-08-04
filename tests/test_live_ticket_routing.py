import tests as _test_environment  # noqa: F401

import unittest
from types import SimpleNamespace
from unittest.mock import patch


import config
from router import select_starting_agent
from state import (
    BotRunContext,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
    stopped_channels,
    stop_reasons_by_channel,
)
from handoff import (
    build_handoff_notice,
    strip_handoff_placeholder,
)
from ticket_investigation.runtime import (
    _merge_explicit_evidence_into_job,
    _select_ticket_starting_agent,
    TicketInvestigationRuntime,
)
from discord_support_runtime import (
    _classify_ticket_message_action,
    _extract_ticket_owner_user_id_from_messages,
    _guardrail_tripwire_reply,
    _maybe_recover_runtime_stopped_ticket_for_message,
    _normalize_staff_summon_prompt,
)


class RoutingTests(unittest.TestCase):
    def test_normalize_staff_summon_prompt_strips_prefix(self) -> None:
        self.assertEqual(
            _normalize_staff_summon_prompt("y: find the apy logic"),
            "find the apy logic",
        )
        self.assertEqual(
            _normalize_staff_summon_prompt("Y:find the apy logic"),
            "find the apy logic",
        )
        self.assertIsNone(_normalize_staff_summon_prompt("find the apy logic"))
        self.assertIsNone(_normalize_staff_summon_prompt("y:   "))

    def test_extract_ticket_owner_user_id_from_messages_prefers_bot_opener_mention(
        self,
    ) -> None:
        human_message = SimpleNamespace(
            author=SimpleNamespace(bot=False),
            mentions=[SimpleNamespace(id=999, bot=False)],
        )
        bot_opener = SimpleNamespace(
            author=SimpleNamespace(bot=True),
            mentions=[
                SimpleNamespace(id=111, bot=True),
                SimpleNamespace(id=222, bot=False),
            ],
        )

        self.assertEqual(
            _extract_ticket_owner_user_id_from_messages([human_message, bot_opener]),
            222,
        )

    @patch("discord_support_runtime._is_support_staff_member")
    def test_classify_ticket_message_action_active_ticket_separates_owner_and_staff(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.side_effect = lambda author: author.id == 2
        owner = SimpleNamespace(id=1)
        staff = SimpleNamespace(id=2)
        stranger = SimpleNamespace(id=3)

        self.assertEqual(
            _classify_ticket_message_action(
                author=owner,
                content="help",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "process",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=owner,
                content="y: this is ordinary owner text",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "process",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="I have thoughts",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "staff_takeover",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="Y: explain the gas requirement",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "staff_summon",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=stranger,
                content="y: same issue",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "ignore",
        )

    @patch("discord_support_runtime._is_support_staff_member")
    def test_classify_ticket_message_action_stopped_ticket_supports_staff_commands(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.side_effect = lambda author: author.id == 2
        owner = SimpleNamespace(id=1)
        staff = SimpleNamespace(id=2)

        self.assertEqual(
            _classify_ticket_message_action(
                author=owner,
                content="y: try again",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "ignore",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="please retry",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "staff_takeover",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="y: retry with vault evidence",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "staff_summon",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="y:   ",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "staff_summon_usage",
        )

    def test_build_handoff_notice_explains_reply_and_discord_takeover(self) -> None:
        notice = build_handoff_notice(
            reason="manual follow-up needed",
            summary="i need a human asap",
            channel_id=1506309610192113917,
            guild_id=734804446353031319,
        )
        self.assertIn(
            "Reply to this message with what I should tell the user or do next. "
            "Your reply will be used for the next ticket update. "
            "To dismiss the handoff and handle the ticket in Discord, click the "
            "button below. The first reply or button action closes the handoff.",
            notice,
        )
        self.assertIn(
            "discord.com/channels/734804446353031319/1506309610192113917",
            notice,
        )

    def test_build_handoff_notice_bounds_model_generated_reason(self) -> None:
        notice = build_handoff_notice(
            reason="x" * 400,
            summary="manual action needed",
            channel_id=1506309610192113917,
            guild_id=734804446353031319,
        )

        reason_line = next(
            line for line in notice.splitlines() if line.startswith("<b>Reason</b>:")
        )
        self.assertLessEqual(len(reason_line), len("<b>Reason</b>: ") + 260)
        self.assertTrue(reason_line.endswith("..."))

    def test_strip_handoff_placeholder_preserves_newlines(self) -> None:
        text = (
            "Main difference from Aave:\n"
            "- Flex uses fixed rates.\n"
            "- Aave uses variable rates.\n\n"
            f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
        )
        self.assertEqual(
            strip_handoff_placeholder(text),
            "Main difference from Aave:\n- Flex uses fixed rates.\n- Aave uses variable rates.",
        )

    @patch("discord_support_runtime._is_support_staff_member")
    def test_runtime_stopped_ticket_recovers_for_owner_message(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = False
        channel_id = 77
        owner = SimpleNamespace(id=1, name="owner")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "runtime_error"
        try:
            self.assertTrue(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=owner,
                    ticket_owner_user_id=1,
                )
            )
            self.assertNotIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, stop_reasons_by_channel)
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)

    @patch("discord_support_runtime._is_support_staff_member")
    def test_runtime_stopped_ticket_recovers_for_known_owner_with_contributor_role(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = True
        channel_id = 177
        owner = SimpleNamespace(id=1, name="owner")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "runtime_error"
        try:
            self.assertTrue(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=owner,
                    ticket_owner_user_id=1,
                )
            )
            self.assertNotIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, stop_reasons_by_channel)
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)

    @patch("discord_support_runtime._is_support_staff_member")
    def test_runtime_stopped_ticket_does_not_recover_for_non_owner_contributor(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = True
        channel_id = 277
        contributor = SimpleNamespace(id=2, name="contributor")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "runtime_error"
        try:
            self.assertFalse(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=contributor,
                    ticket_owner_user_id=1,
                )
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "runtime_error")
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)

    @patch("discord_support_runtime._is_support_staff_member")
    def test_boundary_stopped_ticket_does_not_recover_for_owner_message(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = False
        channel_id = 78
        owner = SimpleNamespace(id=1, name="owner")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "boundary_stop"
        try:
            self.assertFalse(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=owner,
                    ticket_owner_user_id=1,
                )
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "boundary_stop")
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)

    def test_guardrail_tripwire_reply_prefers_guardrail_message(self) -> None:
        exc = type(
            "FakeTripwire",
            (),
            {
                "guardrail_result": type(
                    "FakeResult",
                    (),
                    {
                        "output": type(
                            "FakeOutput",
                            (),
                            {
                                "output_info": {
                                    "message": "Please use the BD contact path."
                                }
                            },
                        )()
                    },
                )()
            },
        )()

        self.assertEqual(
            _guardrail_tripwire_reply(exc),
            "Please use the BD contact path.",
        )

    def test_guardrail_tripwire_reply_falls_back_when_message_missing(self) -> None:
        exc = type(
            "FakeTripwire",
            (),
            {
                "guardrail_result": type(
                    "FakeResult",
                    (),
                    {
                        "output": type(
                            "FakeOutput",
                            (),
                            {
                                "output_info": {
                                    "classification": {"request_type": "partnership"}
                                }
                            },
                        )()
                    },
                )()
            },
        )()

        self.assertEqual(
            _guardrail_tripwire_reply(exc),
            "Your request could not be processed due to input checks.",
        )

    def test_select_starting_agent_uses_data_button_intent(self) -> None:
        context = BotRunContext(
            channel_id=1,
            project_context="yearn",
            initial_button_intent="data_deposit_check",
        )
        self.assertEqual(select_starting_agent("0x1234", context), "data")

    def test_select_starting_agent_uses_bug_button_intent(self) -> None:
        context = BotRunContext(
            channel_id=2,
            project_context="yearn",
            initial_button_intent="bug_report",
        )
        self.assertEqual(
            select_starting_agent("the claim button is missing", context), "bug"
        )

    def test_select_starting_agent_uses_investigate_issue_button_intent(self) -> None:
        context = BotRunContext(
            channel_id=21,
            project_context="yearn",
            initial_button_intent="investigate_issue",
        )
        self.assertEqual(
            select_starting_agent("the claim button is missing", context), "triage"
        )

    def test_select_starting_agent_keeps_styfi_question_in_triage(self) -> None:
        context = BotRunContext(channel_id=3, project_context="yearn")
        self.assertEqual(
            select_starting_agent("Where do I see my stYFI position?", context),
            "triage",
        )

    def test_select_starting_agent_keeps_styfi_contract_address_question_in_triage(
        self,
    ) -> None:
        context = BotRunContext(channel_id=25, project_context="yearn")
        self.assertEqual(
            select_starting_agent(
                "what's the contract address for styfi? it just launched today", context
            ),
            "triage",
        )

    def test_select_starting_agent_keeps_yeth_recovery_question_in_triage(self) -> None:
        context = BotRunContext(channel_id=23, project_context="yearn")
        message = (
            "I am a yETH holder associated with wallet 0x0ae6395e62c85b7b5d08c5e7918b60c1eac66680. "
            "Does that mean I can reclaim my lost ETH 1:1, and why would someone stay in the recovery vault?"
        )
        self.assertEqual(select_starting_agent(message, context), "triage")

    def test_select_starting_agent_keeps_harvest_family_question_in_triage(
        self,
    ) -> None:
        context = BotRunContext(channel_id=24, project_context="yearn")
        message = (
            "Can I make harvests myself on this Pool v2, and where should rewards appear afterward "
            "if they do not show up immediately?"
        )
        self.assertEqual(select_starting_agent(message, context), "triage")

    def test_select_starting_agent_keeps_apy_mechanics_question_in_triage(self) -> None:
        context = BotRunContext(channel_id=28, project_context="yearn")
        message = (
            "How is the AUSD vault in katana earning 10%+ native APY for almost a month "
            "if the strategies inside are earning less than 1%?"
        )
        self.assertEqual(select_starting_agent(message, context), "triage")

    def test_select_starting_agent_keeps_free_form_withdrawal_in_triage(self) -> None:
        context = BotRunContext(channel_id=4, project_context="yearn")
        message = (
            "How do I withdraw from vault "
            "0x2222222222222222222222222222222222222222 "
            "on ethereum using wallet 0x1111111111111111111111111111111111111111?"
        )
        self.assertEqual(select_starting_agent(message, context), "triage")

    def test_follow_up_routing_reuses_last_specialist_for_structured_tx_followup(
        self,
    ) -> None:
        channel_id = 9
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "data"
        investigation_job.remember_tx_hash(
            "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        )
        try:
            agent_key = _select_ticket_starting_agent(
                "look into it",
                context,
                current_history=[{"role": "user", "content": "Previous issue context"}],
                investigation_job=investigation_job,
            )
            self.assertEqual(agent_key, "data")
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_follow_up_routing_does_not_force_data_reuse_for_ui_issue(self) -> None:
        channel_id = 26
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "data"
        try:
            agent_key = _select_ticket_starting_agent(
                "Rabby says 'transaction not ready' for every address when I try to withdraw.",
                context,
                current_history=[
                    {
                        "role": "assistant",
                        "content": (
                            "Okay, I can help with withdrawal instructions. "
                            "Please provide your wallet address (0x...)."
                        ),
                    }
                ],
                investigation_job=investigation_job,
            )
            self.assertEqual(agent_key, "triage")
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_follow_up_routing_switches_data_turn_to_docs_for_veyfi_migration(
        self,
    ) -> None:
        channel_id = 27
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "data"
        try:
            agent_key = _select_ticket_starting_agent(
                "I have 1.0893 veYFI. They are unlocked now since yesterday, but I am unable to migrate.",
                context,
                current_history=[
                    {
                        "role": "assistant",
                        "content": (
                            "**Active Deposits:**\n"
                            "**Vault:** [Ethereum Vault](https://yearn.fi/vaults/1/0x6dfb4ab47a5d2947c4f0f6ea20f92955295c5f5e) (Symbol: yvUSDC-1)\n"
                            "  Address: `0x6dfb4ab47a5d2947c4f0f6ea20f92955295c5f5e`\n"
                            "  Total Position: **1.000000 yvUSDC-1**"
                        ),
                    }
                ],
                investigation_job=investigation_job,
            )
            self.assertEqual(agent_key, "docs")
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_follow_up_routing_does_not_keep_bug_lane_sticky_without_structured_state(
        self,
    ) -> None:
        channel_id = 29
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "bug"
        try:
            agent_key = _select_ticket_starting_agent(
                "where do i see my styfi position?",
                context,
                current_history=[
                    {
                        "role": "assistant",
                        "content": "Please share the page and button state.",
                    },
                ],
                investigation_job=investigation_job,
            )
            self.assertEqual(agent_key, "triage")
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_merge_explicit_evidence_into_job_tracks_chain_and_tx_hash(self) -> None:
        channel_id = 22
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        try:
            _merge_explicit_evidence_into_job(
                investigation_job,
                "Katana tx hash: 0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0",
            )
            self.assertEqual(investigation_job.evidence.chain, "katana")
            self.assertEqual(
                investigation_job.evidence.tx_hashes,
                ["0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"],
            )
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_merge_explicit_evidence_does_not_infer_chain_from_substring(self) -> None:
        channel_id = 30
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        try:
            _merge_explicit_evidence_into_job(
                investigation_job,
                "The portfolio database is showing the wrong balance.",
            )

            self.assertIsNone(investigation_job.evidence.chain)
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_build_contextual_hints_reuses_known_chain_and_tx_hash(self) -> None:
        channel_id = 23
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        try:
            investigation_job.remember_chain("katana")
            investigation_job.remember_tx_hash(
                "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
            )

            hints = TicketInvestigationRuntime.build_contextual_hints(
                investigation_job,
                "i dunno man. look into it",
            )

            self.assertEqual(len(hints), 2)
            self.assertIn("use chain 'katana'", hints[0].lower())
            self.assertIn("Do not substitute a different chain", hints[0])
            self.assertIn(
                "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0",
                hints[0],
            )
            self.assertIn(
                "follow-up to an existing transaction investigation", hints[1].lower()
            )
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_build_contextual_hints_reuses_single_listed_deposit_context_for_withdrawal(
        self,
    ) -> None:
        channel_id = 24
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        try:
            investigation_job.remember_wallet(
                "0x1111111111111111111111111111111111111111"
            )
            investigation_job.remember_withdrawal_target(
                "katana",
                "0x80c34BD3A3569E126e7055831036aa7b212cB159",
            )
            hints = TicketInvestigationRuntime.build_contextual_hints(
                investigation_job,
                "I'd like support withdrawing",
            )

            self.assertEqual(len(hints), 1)
            self.assertIn("already has the needed details", hints[0].lower())
            self.assertIn("0x1111111111111111111111111111111111111111", hints[0])
            self.assertIn("0x80c34BD3A3569E126e7055831036aa7b212cB159", hints[0])
            self.assertIn("katana", hints[0].lower())
            self.assertIn("do not re-check deposits", hints[0].lower())
            self.assertIn("do not ask which vault", hints[0].lower())
        finally:
            clear_ticket_investigation_job(channel_id)
