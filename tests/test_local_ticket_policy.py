import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import patch


from agents import Runner

from agent_prompts import SUPPORT_BOUNDARY_GUARDRAIL_INSTRUCTIONS
from bot_behavior import (
    OUT_OF_SCOPE_SUPPORT_MESSAGE,
    SECURITY_PROCESS_URL,
    SECURITY_VENDOR_BOUNDARY_MESSAGE,
)
import config
from state import (
    BotRunContext,
    TicketInvestigationJob,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
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
    _normalize_ticket_triage_decision,
    _reply_requests_human_handoff,
    TicketTurnRequest,
)
from router import is_wallet_confirmation
from discord_support_runtime import (
    _outer_support_boundary_reply,
)


class WalletConfirmationPolicyTests(unittest.TestCase):
    def test_confirmation_requires_an_explicit_complete_reply(self) -> None:
        self.assertTrue(is_wallet_confirmation("Yes, that's correct."))
        self.assertTrue(is_wallet_confirmation("use it"))
        self.assertFalse(is_wallet_confirmation("Yesterday the vault worked."))
        self.assertFalse(is_wallet_confirmation("That is the incorrect address."))
        self.assertFalse(is_wallet_confirmation("I did not confirm that address."))


class BDPriorityGuardrailTests(unittest.IsolatedAsyncioTestCase):
    def test_boundary_prompt_is_compact_and_preserves_taxonomy(self) -> None:
        prompt = SUPPORT_BOUNDARY_GUARDRAIL_INSTRUCTIONS

        self.assertLessEqual(len(prompt.split()), 250)
        for classification in (
            "yearn_support",
            "business_boundary",
            "security_process_boundary",
            "non_support_assistant",
            "uncertain",
        ):
            self.assertIn(f"`{classification}`", prompt)
        for subtype in ("listing", "general_bd", "vendor_security", "job_inquiry"):
            self.assertIn(f"`{subtype}`", prompt)
        self.assertIn("Immunefi or zkPassport submission blocker", prompt)
        self.assertIn("Yearn code, repository, PoC, exploit, and contract", prompt)
        self.assertIn("Greetings and neutral support openers", prompt)
        self.assertIn("choose `yearn_support`", prompt)

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

    async def test_evaluate_support_boundary_returns_reusable_business_tripwire_payload(
        self,
    ) -> None:
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

    async def test_evaluate_support_boundary_defaults_to_yearn_support_when_model_says_so(
        self,
    ) -> None:
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
            result = await evaluate_support_boundary(
                "Where can I monitor stYFI rewards?"
            )

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

    async def test_security_process_exception_uses_alternate_contact_message(
        self,
    ) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="security_process_boundary",
                    business_subtype=None,
                    reasoning="immunefi unavailable for reporter",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        disclosure = "I need to report a Yearn vulnerability, but Immunefi KYC is blocked and zkPassport is not working for me."

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(disclosure)

        self.assertTrue(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "security_process_boundary")
        self.assertIn("contact-information", result["message"].lower())

    async def test_security_process_boundary_uses_direct_disclosure_guidelines(
        self,
    ) -> None:
        class FakeResult:
            def final_output_as(self, _output_type):
                return SupportBoundaryCheckOutput(
                    classification="security_process_boundary",
                    business_subtype=None,
                    reasoning="security disclosure request",
                )

        async def fake_run(self, *, starting_agent, input, run_config):
            return FakeResult()

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(
                "I found a concrete Yearn vulnerability and want to disclose it."
            )

        self.assertEqual(result["classification"], "security_process_boundary")
        self.assertIn(SECURITY_PROCESS_URL, result["message"])

    async def test_outer_support_boundary_reply_prefers_bug_bounty_boundary(
        self,
    ) -> None:
        async def fake_boundary(_text: str):
            return {
                "classification": "security_process_boundary",
                "tripwire_triggered": True,
                "message": "Security process reply",
            }

        with patch(
            "discord_support_runtime.evaluate_support_boundary", new=fake_boundary
        ):
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

        with patch(
            "discord_support_runtime.evaluate_support_boundary", new=fake_boundary
        ):
            reply = await _outer_support_boundary_reply(
                "We want a marketing partnership"
            )

        self.assertEqual(reply, "Boundary reply")

    async def test_outer_support_boundary_reply_blocks_off_topic_coding_help(
        self,
    ) -> None:
        async def fake_scope(_text: str):
            return {
                "classification": "non_support_assistant",
                "tripwire_triggered": True,
                "message": OUT_OF_SCOPE_SUPPORT_MESSAGE,
            }

        with patch("discord_support_runtime.evaluate_support_boundary", new=fake_scope):
            reply = await _outer_support_boundary_reply(
                "Can you write a Python script to parse a CSV for me?"
            )

        self.assertEqual(reply, OUT_OF_SCOPE_SUPPORT_MESSAGE)

    async def test_outer_support_boundary_reply_allows_normal_yearn_support(
        self,
    ) -> None:
        async def fake_scope(_text: str):
            return {
                "classification": "yearn_support",
                "tripwire_triggered": False,
            }

        with patch("discord_support_runtime.evaluate_support_boundary", new=fake_scope):
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

    async def test_evaluate_support_boundary_keeps_yearn_dev_question_in_scope(
        self,
    ) -> None:
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
            raise AssertionError(
                "Bare address support primitives should not invoke the scope model."
            )

        with patch.object(Runner, "run", new=fake_run):
            result = await evaluate_support_boundary(
                "0xAbcdefABcdefABcdefABcdefABcdefABcdefABCD"
            )

        self.assertFalse(result["tripwire_triggered"])
        self.assertEqual(result["classification"], "yearn_support")

    async def test_evaluate_support_boundary_skips_model_for_bare_tx_hash(self) -> None:
        async def fake_run(self, *, starting_agent, input, run_config):
            raise AssertionError(
                "Bare tx-hash support primitives should not invoke the scope model."
            )

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
        job.remember_tx_hash(
            "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        )
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
        self.assertEqual(
            job.evidence.wallet, "0x1111111111111111111111111111111111111111"
        )
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

    def test_job_clears_current_specialty_when_non_specialist_turn_completes(
        self,
    ) -> None:
        job = TicketInvestigationJob(
            channel_id=78, current_specialty="bug", last_specialty="bug"
        )
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

    def test_build_specialist_turn_input_adds_tx_followup_contract_up_front(
        self,
    ) -> None:
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

    def test_build_specialist_turn_input_adds_report_pretriage_contract_up_front(
        self,
    ) -> None:
        channel_id = 78
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        request = TicketTurnRequest(
            aggregated_text="Report: https://gist.github.com/example/abcdef1234567890",
            input_list=[
                {
                    "role": "user",
                    "content": "Report: https://gist.github.com/example/abcdef1234567890",
                }
            ],
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

    def test_contains_report_artifact_evidence_detects_supported_hosts_and_code_blocks(
        self,
    ) -> None:
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
