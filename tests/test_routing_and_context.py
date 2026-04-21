import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch

from agents import MaxTurnsExceeded, RunContextWrapper, Runner
import discord

from bot_behavior import OUT_OF_SCOPE_SUPPORT_MESSAGE, SECURITY_VENDOR_BOUNDARY_MESSAGE
import config
from router import select_starting_agent
from state import (
    BotRunContext,
    TicketInvestigationJob,
    clear_ticket_investigation_job,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    last_bot_reply_ts_by_channel,
    last_wallet_by_channel,
    pending_messages,
    pending_wallet_confirmation_by_channel,
    public_conversations,
    PublicConversation,
)
from ticket_investigation_worker import TicketInvestigationWorker
from support_agents import (
    SupportBoundaryCheckOutput,
    bd_priority_guardrail,
    evaluate_support_boundary,
    TicketTriageDecision,
    ticket_triage_router_agent,
    triage_agent,
    yearn_bug_triage_agent,
    yearn_data_agent,
    yearn_docs_qa_agent,
)
from support_tools import _extract_artifact_refs, _repo_search_block_message
from ticket_investigation_runtime import (
    _build_specialist_turn_input,
    TicketAgentFlowOutcome,
    _looks_like_bug_bounty_intake_boundary_case,
    _contains_report_artifact_evidence,
    _merge_explicit_evidence_into_job,
    _normalize_ticket_triage_decision,
    _reply_requests_human_handoff,
    _select_ticket_starting_agent,
    bug_bounty_intake_boundary_reply,
    resolve_freeform_starting_agent,
    TicketInvestigationRuntime,
    TicketTurnRequest,
)
from ysupport import (
    TicketBot,
    _canonicalize_current_user_message,
    _guardrail_tripwire_reply,
    _outer_support_boundary_reply,
)


class RoutingTests(unittest.TestCase):
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

    def test_bug_bounty_intake_boundary_helper_matches_opening_request(self) -> None:
        text = (
            "Good day team, me and my team discovered an issue that should be addressed "
            "and hope to be rewarded for our efforts"
        )

        self.assertTrue(_looks_like_bug_bounty_intake_boundary_case(text))

    def test_bug_bounty_intake_boundary_reply_points_to_security_process(self) -> None:
        reply = bug_bounty_intake_boundary_reply(
            "Good day team, me and my team discovered an issue that should be addressed "
            "and hope to be rewarded for our efforts"
        )

        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("docs.yearn.fi/developers/security", reply)
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, reply)

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
            result = await bd_priority_guardrail.guardrail_function(
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
            result = await bd_priority_guardrail.guardrail_function(
                None,
                None,
                disclosure,
            )

        self.assertTrue(result.tripwire_triggered)
        self.assertEqual(
            result.output_info["classification"],
            "security_process_boundary",
        )

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


class _FakeTypingContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeDiscordChannel:
    def __init__(self, channel_id: int):
        self.id = channel_id
        self.sent_messages: list[str] = []

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
            with patch("ysupport.tools_lib.resolve_ens", return_value=resolved_wallet):
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

        async def fake_boundary(_text: str) -> str | None:
            return "Boundary reply"

        async def fake_send_long_message(channel, message, **kwargs):
            await channel.send(message, **kwargs)

        try:
            with patch("ysupport._outer_support_boundary_reply", new=fake_boundary):
                with patch("ysupport.send_long_message", new=fake_send_long_message):
                    with patch("ysupport.discord.TextChannel", _FakeDiscordChannel):
                        await bot.process_ticket_message(channel_id, run_context)
        finally:
            pending_messages.pop(channel_id, None)
            conversation_threads.pop(channel_id, None)
            last_bot_reply_ts_by_channel.pop(channel_id, None)
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(fake_channel.sent_messages, ["Boundary reply"])


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

        async def fake_boundary(_text: str) -> str | None:
            return "Boundary reply"

        bot = TicketBot(intents=discord.Intents.none())
        bot.investigation_executor = _FailingExecutor()

        with patch("ysupport._outer_support_boundary_reply", new=fake_boundary):
            handled = await bot._handle_public_trigger_message(trigger_message, "y")

        self.assertTrue(handled)
        self.assertEqual(len(original_message.replies), 1)
        self.assertEqual(original_message.replies[0][0], "Boundary reply")
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

        public_conversations[original_author_id] = PublicConversation(
            history=[{"role": "assistant", "content": "Earlier context"}],
            last_interaction_time=datetime.now(timezone.utc),
        )
        try:
            with (
                patch.object(config, "HUMAN_HANDOFF_TARGET_USER_ID", "999"),
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
        self.assertIn("<@999>", original_message.replies[0][0])
        self.assertFalse(original_message.replies[0][1]["mention_author"])
        self.assertTrue(original_message.replies[0][1]["suppress_embeds"])

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

    async def test_ticket_agent_flow_short_circuits_bug_bounty_intake_boundary(self) -> None:
        channel_id = 37
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner([])
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        try:
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
