import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import Runner

from bot_behavior import OUT_OF_SCOPE_SUPPORT_MESSAGE, SECURITY_VENDOR_BOUNDARY_MESSAGE
import config
from router import select_starting_agent
from state import (
    BotRunContext,
    TicketInvestigationJob,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
    stopped_channels,
    stop_reasons_by_channel,
)
from handoff import (
    HandoffRoute,
    build_handoff_notice,
)
from support_agents import (
    SupportBoundaryCheckOutput,
    evaluate_support_boundary,
    support_boundary_guardrail,
    TicketTriageDecision,
)
from support_tools import _extract_artifact_refs, _repo_search_block_message
from ticket_investigation.runtime import (
    _build_specialist_turn_input,
    _contains_report_artifact_evidence,
    _merge_explicit_evidence_into_job,
    _normalize_ticket_triage_decision,
    _reply_requests_human_handoff,
    _select_ticket_starting_agent,
    TicketInvestigationRuntime,
    TicketTurnRequest,
)
from ysupport import (
    _classify_ticket_message_action,
    _extract_ticket_owner_user_id_from_messages,
    _guardrail_tripwire_reply,
    _maybe_recover_runtime_stopped_ticket_for_message,
    _normalize_contributor_override_prompt,
    _outer_moderator_access_reply,
    _outer_support_boundary_reply,
)

class RoutingTests(unittest.TestCase):
    def test_normalize_contributor_override_prompt_strips_prefix(self) -> None:
        self.assertEqual(
            _normalize_contributor_override_prompt("[y] find the apy logic"),
            "find the apy logic",
        )
        self.assertIsNone(_normalize_contributor_override_prompt("find the apy logic"))
        self.assertIsNone(_normalize_contributor_override_prompt("[y]   "))

    def test_extract_ticket_owner_user_id_from_messages_prefers_bot_opener_mention(self) -> None:
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

    @patch("ysupport._is_contributor_member")
    def test_classify_ticket_message_action_active_ticket_only_allows_owner(
        self,
        mock_is_contributor_member,
    ) -> None:
        mock_is_contributor_member.side_effect = lambda author: author.id == 2
        owner = SimpleNamespace(id=1)
        contributor = SimpleNamespace(id=2)
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
                author=contributor,
                content="I have thoughts",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "ignore",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=stranger,
                content="same issue",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "ignore",
        )

    @patch("ysupport._is_contributor_member")
    def test_classify_ticket_message_action_stopped_ticket_allows_only_prefixed_contributor(
        self,
        mock_is_contributor_member,
    ) -> None:
        mock_is_contributor_member.side_effect = lambda author: author.id == 2
        owner = SimpleNamespace(id=1)
        contributor = SimpleNamespace(id=2)

        self.assertEqual(
            _classify_ticket_message_action(
                author=owner,
                content="[y] try again",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "ignore",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=contributor,
                content="please retry",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "ignore",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=contributor,
                content="[y] retry with vault evidence",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "contributor_override",
        )

    def test_build_handoff_notice_includes_one_reply_instruction(self) -> None:
        notice = build_handoff_notice(
            HandoffRoute(
                target="support_manual",
                team_label="support team",
                reason="manual follow-up needed",
            ),
            summary="i need a human asap",
            channel_id=1506309610192113917,
            guild_id=734804446353031319,
        )
        self.assertIn("Only one reply is accepted.", notice)
        self.assertIn(
            "Reply to this message with what I should tell the user or do next.",
            notice,
        )
        self.assertIn(
            "discord.com/channels/734804446353031319/1506309610192113917",
            notice,
        )

    @patch("ysupport._is_contributor_member")
    def test_runtime_stopped_ticket_recovers_for_owner_message(
        self,
        mock_is_contributor_member,
    ) -> None:
        mock_is_contributor_member.return_value = False
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

    @patch("ysupport._is_contributor_member")
    def test_boundary_stopped_ticket_does_not_recover_for_owner_message(
        self,
        mock_is_contributor_member,
    ) -> None:
        mock_is_contributor_member.return_value = False
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
                            {"output_info": {"message": "Please use the BD contact path."}},
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
                            {"output_info": {"classification": {"request_type": "partnership"}}},
                        )()
                    },
                )()
            },
        )()

        self.assertEqual(
            _guardrail_tripwire_reply(exc),
            "Your request could not be processed due to input checks.",
        )

    def test_outer_moderator_access_reply_detects_discord_access_issue(self) -> None:
        reply = _outer_moderator_access_reply(
            "I finished verification but still cannot access the Discord."
        )
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("A moderator needs to check this.", reply)
        self.assertIn("I've notified the support team", reply)
        self.assertNotIn("<@", reply)

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
        self.assertEqual(select_starting_agent("the claim button is missing", context), "bug")

    def test_select_starting_agent_uses_investigate_issue_button_intent(self) -> None:
        context = BotRunContext(
            channel_id=21,
            project_context="yearn",
            initial_button_intent="investigate_issue",
        )
        self.assertEqual(select_starting_agent("the claim button is missing", context), "triage")

    def test_select_starting_agent_keeps_styfi_question_in_triage(self) -> None:
        context = BotRunContext(channel_id=3, project_context="yearn")
        self.assertEqual(select_starting_agent("Where do I see my stYFI position?", context), "triage")

    def test_select_starting_agent_keeps_styfi_contract_address_question_in_triage(self) -> None:
        context = BotRunContext(channel_id=25, project_context="yearn")
        self.assertEqual(
            select_starting_agent("what's the contract address for styfi? it just launched today", context),
            "triage",
        )

    def test_select_starting_agent_keeps_yeth_recovery_question_in_triage(self) -> None:
        context = BotRunContext(channel_id=23, project_context="yearn")
        message = (
            "I am a yETH holder associated with wallet 0x0ae6395e62c85b7b5d08c5e7918b60c1eac66680. "
            "Does that mean I can reclaim my lost ETH 1:1, and why would someone stay in the recovery vault?"
        )
        self.assertEqual(select_starting_agent(message, context), "triage")

    def test_select_starting_agent_keeps_harvest_family_question_in_triage(self) -> None:
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

    def test_follow_up_routing_reuses_last_specialist_for_structured_tx_followup(self) -> None:
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

    def test_follow_up_routing_switches_data_turn_to_docs_for_veyfi_migration(self) -> None:
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

    def test_follow_up_routing_does_not_keep_bug_lane_sticky_without_structured_state(self) -> None:
        channel_id = 29
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "bug"
        try:
            agent_key = _select_ticket_starting_agent(
                "where do i see my styfi position?",
                context,
                current_history=[
                    {"role": "assistant", "content": "Please share the page and button state."},
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
            self.assertIn("0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0", hints[0])
            self.assertIn("follow-up to an existing transaction investigation", hints[1].lower())
        finally:
            clear_ticket_investigation_job(channel_id)

    def test_build_contextual_hints_reuses_single_listed_deposit_context_for_withdrawal(self) -> None:
        channel_id = 24
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        try:
            investigation_job.remember_wallet("0x1111111111111111111111111111111111111111")
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


class BDPriorityGuardrailTests(unittest.IsolatedAsyncioTestCase):
    async def test_vendor_security_boundary_uses_firm_message(self) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="business_boundary",
                    business_subtype="vendor_security",
                    reasoning="security vendor outreach",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        with patch.object(Runner, "run", new=fake_run):
            result = await support_boundary_guardrail.guardrail_function(
                None,
                None,
                "phishing vendor outreach",
            )

        self.assertTrue(result.tripwire_triggered)
        self.assertEqual(
            result.output_info["message"],
            SECURITY_VENDOR_BOUNDARY_MESSAGE,
        )

    async def test_evaluate_support_boundary_returns_reusable_business_tripwire_payload(self) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="business_boundary",
                    business_subtype="job_inquiry",
                    reasoning="asks how to work for yearn",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary("How can I work for Yearn?")

        self.assertTrue(result["tripwire_triggered"])
        self.assertEqual(result["business_subtype"], "job_inquiry")
        self.assertIn("message", result)

    async def test_evaluate_support_boundary_defaults_to_yearn_support_when_model_says_so(self) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="yearn_support",
                    business_subtype=None,
                    reasoning="normal yearn support",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary("How is stYFI APY calculated?")

        self.assertFalse(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "yearn_support")

    async def test_evaluate_support_boundary_clears_business_subtype_outside_business_class(
        self,
    ) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="yearn_support",
                    business_subtype="general_bd",
                    reasoning="normal yearn support",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary("Where can I monitor stYFI rewards?")

        self.assertFalse(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "yearn_support")
        self.assertIsNone(result["business_subtype"])

    async def test_concrete_security_disclosure_overrides_vendor_security_false_positive(
        self,
    ) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="security_process_boundary",
                    business_subtype=None,
                    reasoning="mentions security team and secure report path",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        disclosure = (
            "Hello Yearn Finance Security Team, I would like to responsibly disclose a critical "
            "vulnerability affecting the stYFI contract related to stream state handling during "
            "instant withdrawals. Immunefi identity verification is blocked because zkPassport is "
            "not functioning. Please let me know the preferred secure way to submit the full "
            "technical report. Impact: permanent user fund loss. Component: stYFI contract "
            "(_withdraw / _unstake interaction)."
        )

        with patch.object(Runner, "run", new=fake_run):
            result = await support_boundary_guardrail.guardrail_function(
                None,
                None,
                disclosure,
            )

        self.assertTrue(result.tripwire_triggered)
        self.assertEqual(
            result.output_info["classification"],
            "security_process_boundary",
        )

    async def test_security_process_exception_uses_alternate_contact_message(self) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="security_process_boundary",
                    business_subtype=None,
                    reasoning="immunefi unavailable for reporter",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        disclosure = (
            "I need to report a Yearn vulnerability, but Immunefi KYC is blocked and zkPassport is not working for me."
        )

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(disclosure)

        self.assertTrue(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "security_process_boundary")
        self.assertIn("security@yearn.finance", result["message"].lower())

    async def test_outer_support_boundary_reply_prefers_bug_bounty_boundary(self) -> None:
        async def fake_boundary(_text: str):
            return {
                "classification": "security_process_boundary",
                "tripwire_triggered": True,
                "message": "Security process reply",
            }

        with patch("ysupport.evaluate_support_boundary", new=fake_boundary):
            reply = await _outer_support_boundary_reply(
                "Good day team, me and my team discovered an issue that should be addressed "
                "and hope to be rewarded for our efforts"
            )

        self.assertIsNotNone(reply)
        self.assertEqual(reply, "Security process reply")

    async def test_outer_support_boundary_reply_uses_bd_guardrail_message(self) -> None:
        async def fake_boundary(_text: str):
            return {
                "classification": "business_boundary",
                "business_subtype": "general_bd",
                "message": "Boundary reply",
                "tripwire_triggered": True,
            }

        with patch("ysupport.evaluate_support_boundary", new=fake_boundary):
            reply = await _outer_support_boundary_reply("We want a marketing partnership")

        self.assertEqual(reply, "Boundary reply")

    async def test_outer_support_boundary_reply_blocks_off_topic_coding_help(self) -> None:
        async def fake_scope(_text: str):
            return {
                "classification": "non_support_assistant",
                "tripwire_triggered": True,
                "message": OUT_OF_SCOPE_SUPPORT_MESSAGE,
            }

        with patch("ysupport.evaluate_support_boundary", new=fake_scope):
            reply = await _outer_support_boundary_reply(
                "Can you write a Python script to parse a CSV for me?"
            )

        self.assertEqual(reply, OUT_OF_SCOPE_SUPPORT_MESSAGE)

    async def test_outer_support_boundary_reply_allows_normal_yearn_support(self) -> None:
        async def fake_scope(_text: str):
            return {
                "classification": "yearn_support",
                "tripwire_triggered": False,
            }

        with patch("ysupport.evaluate_support_boundary", new=fake_scope):
            reply = await _outer_support_boundary_reply(
                "Where can I monitor stYFI rewards?"
            )

        self.assertIsNone(reply)

    async def test_evaluate_support_boundary_returns_non_support_tripwire(self) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="non_support_assistant",
                    business_subtype=None,
                    reasoning="generic coding help unrelated to Yearn",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(
                "Can you write a Python script to parse a CSV for me?"
            )

        self.assertTrue(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "non_support_assistant")

    async def test_evaluate_support_boundary_keeps_yearn_dev_question_in_scope(self) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="yearn_support",
                    business_subtype=None,
                    reasoning="question is about Yearn contract behavior",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(
                "Can you explain Yearn VaultV3 process_report behavior?"
            )

        self.assertFalse(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "yearn_support")

    async def test_evaluate_support_boundary_skips_model_for_bare_address(self) -> None:
        async def fake_run(self, *, starting_agent, input, run_config):
            raise AssertionError("Bare address support primitives should not invoke the scope model.")

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(
                "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD"
            )

        self.assertFalse(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "yearn_support")

    async def test_evaluate_support_boundary_skips_model_for_bare_tx_hash(self) -> None:
        async def fake_run(self, *, starting_agent, input, run_config):
            raise AssertionError("Bare tx-hash support primitives should not invoke the scope model.")

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(
                "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
            )

        self.assertFalse(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "yearn_support")


class InvestigationJobTests(unittest.TestCase):
    def test_job_tracks_lifecycle_and_evidence(self) -> None:
        job = TicketInvestigationJob(channel_id=77)

        job.begin_collecting("investigate_issue")
        job.remember_wallet("0x1111111111111111111111111111111111111111")
        job.remember_chain("katana")
        job.remember_tx_hash("0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0")
        job.remember_withdrawal_target(
            "katana",
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )
        job.begin_investigating()
        job.complete_specialist_turn("data")
        job.mark_waiting_for_user()
        job.mark_escalated_to_human()

        self.assertEqual(job.requested_intent, "investigate_issue")
        self.assertEqual(job.mode, "escalated_to_human")
        self.assertEqual(job.current_specialty, "data")
        self.assertEqual(job.last_specialty, "data")
        self.assertEqual(job.evidence.wallet, "0x1111111111111111111111111111111111111111")
        self.assertEqual(job.evidence.chain, "katana")
        self.assertEqual(
            job.evidence.tx_hashes,
            ["0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"],
        )
        self.assertEqual(job.evidence.withdrawal_target_chain, "katana")
        self.assertEqual(
            job.evidence.withdrawal_target_vault,
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )

    def test_job_clears_current_specialty_when_non_specialist_turn_completes(self) -> None:
        job = TicketInvestigationJob(channel_id=78, current_specialty="bug", last_specialty="bug")
        job.remember_withdrawal_target(
            "katana",
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )

        job.complete_specialist_turn(None)

        self.assertIsNone(job.current_specialty)
        self.assertEqual(job.last_specialty, "bug")
        self.assertIsNone(job.evidence.withdrawal_target_chain)
        self.assertIsNone(job.evidence.withdrawal_target_vault)

    def test_job_clears_withdrawal_target_when_turn_leaves_data_lane(self) -> None:
        job = TicketInvestigationJob(channel_id=79)
        job.remember_withdrawal_target(
            "katana",
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )

        job.complete_specialist_turn("docs")

        self.assertEqual(job.current_specialty, "docs")
        self.assertIsNone(job.evidence.withdrawal_target_chain)
        self.assertIsNone(job.evidence.withdrawal_target_vault)


class RepoHelperTests(unittest.TestCase):
    def test_extract_artifact_refs_deduplicates_and_preserves_order(self) -> None:
        text = "segment:12 fact:4 segment:12 fact:7 fact:4"
        self.assertEqual(
            _extract_artifact_refs(text),
            ["segment:12", "fact:4", "fact:7"],
        )

    def test_repo_search_block_message_points_to_fetch(self) -> None:
        context = BotRunContext(
            channel_id=5,
            project_context="yearn",
        )
        context.repo_last_search_artifact_refs = ["segment:12", "fact:4"]
        message = _repo_search_block_message(context)

        self.assertIn("fetch_repo_artifacts_tool", message)
        self.assertIn("segment:12, fact:4", message)


class TriageDecisionTests(unittest.TestCase):
    def test_normalize_human_escalation_adds_handoff_tag(self) -> None:
        decision = TicketTriageDecision(
            action="human_escalation",
            message="This needs human review.",
            reasoning="sensitive report receipt confirmation",
        )

        normalized = _normalize_ticket_triage_decision(decision)

        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, normalized.message or "")

    def test_normalize_clarifying_question_fills_empty_message(self) -> None:
        decision = TicketTriageDecision(
            action="ask_clarifying",
            message="",
            reasoning="missing routing detail",
        )

        normalized = _normalize_ticket_triage_decision(decision)

        self.assertEqual(normalized.message, "Can you clarify what you need help with?")

    def test_normalize_route_action_drops_message(self) -> None:
        decision = TicketTriageDecision(
            action="route_data",
            message="this should not survive normalization",
            reasoning="wallet issue",
        )

        normalized = _normalize_ticket_triage_decision(decision)

        self.assertIsNone(normalized.message)

    def test_reply_requests_human_handoff_checks_placeholder(self) -> None:
        self.assertTrue(
            _reply_requests_human_handoff(
                f"Human help is needed. {config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
            )
        )
        self.assertFalse(_reply_requests_human_handoff("This is a direct answer."))

    def test_build_specialist_turn_input_adds_tx_followup_contract_up_front(self) -> None:
        channel_id = 77
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.remember_tx_hash(
            "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        )
        request = TicketTurnRequest(
            aggregated_text="look into it",
            input_list=[{"role": "user", "content": "look into it"}],
            current_history=[],
            run_context=BotRunContext(channel_id=channel_id, project_context="yearn"),
            investigation_job=investigation_job,
            workflow_name="tests.runtime",
        )
        try:
            specialist_input = _build_specialist_turn_input(request)
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(specialist_input[-1]["role"], "system")
        self.assertIn(
            "Do not ask the user whether you should proceed",
            specialist_input[-1]["content"],
        )

    def test_build_specialist_turn_input_adds_report_pretriage_contract_up_front(self) -> None:
        channel_id = 78
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        request = TicketTurnRequest(
            aggregated_text="Report: https://gist.github.com/example/abcdef1234567890",
            input_list=[{"role": "user", "content": "Report: https://gist.github.com/example/abcdef1234567890"}],
            current_history=[],
            run_context=BotRunContext(channel_id=channel_id, project_context="yearn"),
            investigation_job=investigation_job,
            workflow_name="tests.runtime",
        )
        try:
            specialist_input = _build_specialist_turn_input(request)
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(specialist_input[-1]["role"], "system")
        self.assertIn(
            "Do one bounded repo/docs pre-triage pass",
            specialist_input[-1]["content"],
        )

    def test_contains_report_artifact_evidence_detects_supported_hosts_and_code_blocks(self) -> None:
        self.assertTrue(
            _contains_report_artifact_evidence(
                "Report: https://gist.github.com/example/abcdef1234567890"
            )
        )
        self.assertTrue(
            _contains_report_artifact_evidence(
                "See https://raw.githubusercontent.com/yearn/yearn-security/master/SECURITY.md"
            )
        )
        self.assertTrue(
            _contains_report_artifact_evidence(
                "```solidity\nfunction test_case() external {}\n```"
            )
        )
        self.assertFalse(_contains_report_artifact_evidence("No artifact URL here."))

