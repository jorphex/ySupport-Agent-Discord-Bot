import tests as _test_environment  # noqa: F401

import unittest
from dataclasses import dataclass
from unittest.mock import patch


from agents import InputGuardrailTripwireTriggered, RunContextWrapper

from bot_behavior import OUT_OF_SCOPE_SUPPORT_MESSAGE, SECURITY_PROCESS_URL
import config
from state import (
    BotRunContext,
    TicketInvestigationJob,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
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
from tests.ticket_flow_test_support import (
    FakeResult as _FakeResult,
    FakeRunner as _FakeRunner,
    TicketFlowTestCase,
)
from ticket_investigation.runtime import (
    TicketAgentFlowOutcome,
    resolve_freeform_starting_agent,
    TicketInvestigationRuntime,
    TicketTurnRequest,
)


class TicketFlowTests(TicketFlowTestCase):
    async def test_resolve_freeform_starting_agent_reuses_ticket_router_for_public_lane_selection(
        self,
    ) -> None:
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

        with patch(
            "support_agents.evaluate_support_boundary",
            return_value={
                "classification": "yearn_support",
                "tripwire_triggered": False,
            },
        ) as boundary_check:
            agent_key = await resolve_freeform_starting_agent(
                runner=fake_runner,
                input_list="Where do I see my stYFI position?",
                run_context=context,
                workflow_name="tests.public_route",
            )

        self.assertEqual(agent_key, "docs")
        boundary_check.assert_awaited_once_with("Where do I see my stYFI position?")
        self.assertEqual(len(fake_runner.calls), 1)
        self.assertIs(
            fake_runner.calls[0]["starting_agent"], ticket_triage_router_agent
        )

    async def test_ticket_agent_flow_routes_triage_decision_to_docs_specialist(
        self,
    ) -> None:
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
                        {
                            "role": "user",
                            "content": "Where do I see my stYFI position?",
                        },
                        {
                            "role": "assistant",
                            "content": "Open the stYFI app and check the positions page.",
                        },
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
                    input_list=[
                        {
                            "role": "user",
                            "content": "I need help finding where to see my stYFI position.",
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

        self.assertEqual(len(fake_runner.calls), 2)
        self.assertIs(
            fake_runner.calls[0]["starting_agent"], ticket_triage_router_agent
        )
        self.assertIs(fake_runner.calls[1]["starting_agent"], yearn_docs_qa_agent)
        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertIn("positions page", outcome.raw_final_reply.lower())

    async def test_ticket_agent_flow_remembers_single_withdrawal_target_from_data_reply(
        self,
    ) -> None:
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
                    input_list=[
                        {
                            "role": "user",
                            "content": "0x1111111111111111111111111111111111111111",
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

        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertEqual(investigation_job.evidence.withdrawal_target_chain, "katana")
        self.assertEqual(
            investigation_job.evidence.withdrawal_target_vault,
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )

    async def test_ticket_agent_flow_returns_direct_router_message_without_second_run(
        self,
    ) -> None:
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
                    input_list=[
                        {
                            "role": "user",
                            "content": "I finished verification but still cannot access the Discord.",
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

    async def test_ticket_agent_flow_investigates_before_handoff_for_explicit_human_request_with_repro_context(
        self,
    ) -> None:
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
                _FakeResult(
                    final_output="I checked the blocked withdrawal flow and need one exact error message.",
                    last_agent=yearn_bug_triage_agent,
                    _history=[],
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

        self.assertEqual(len(fake_runner.calls), 2)
        self.assertIs(fake_runner.calls[1]["starting_agent"], yearn_bug_triage_agent)
        self.assertEqual(outcome.completed_agent_key, "bug")
        self.assertFalse(outcome.requires_human_handoff)
        self.assertIn("blocked withdrawal flow", outcome.raw_final_reply)

    async def test_freeform_router_preserves_boundary_tripwire_before_lane_selection(
        self,
    ) -> None:
        fake_runner = _FakeRunner([])
        context = BotRunContext(channel_id=381, project_context="yearn")

        async def fake_boundary(_text: str):
            return {
                "classification": "business_boundary",
                "tripwire_triggered": True,
                "message": "Business boundary",
            }

        with (
            patch(
                "support_agents.evaluate_support_boundary",
                new=fake_boundary,
            ),
            self.assertRaises(InputGuardrailTripwireTriggered) as exc_info,
        ):
            await resolve_freeform_starting_agent(
                runner=fake_runner,
                input_list="We want a marketing partnership",
                run_context=context,
                workflow_name="tests.public_route",
            )

        self.assertEqual(fake_runner.calls, [])
        self.assertEqual(
            exc_info.exception.guardrail_result.output.output_info["message"],
            "Business boundary",
        )
        self.assertEqual(ticket_triage_router_agent.input_guardrails, [])

    async def test_ticket_agent_flow_keeps_bug_lane_when_human_request_lacks_repro_context(
        self,
    ) -> None:
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

    async def test_ticket_agent_flow_short_circuits_bug_bounty_intake_boundary(
        self,
    ) -> None:
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
                    f"use Yearn's official security process at {SECURITY_PROCESS_URL}. "
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
        self.assertIn(SECURITY_PROCESS_URL.lower(), lowered)
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNotIn("browser", lowered)
        self.assertNotIn("device", lowered)

    async def test_ticket_agent_flow_uses_precomputed_boundary_without_second_model_call(
        self,
    ) -> None:
        channel_id = 371
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner([])
        context = BotRunContext(channel_id=channel_id, project_context="yearn")

        async def fail_boundary(_text: str):
            raise AssertionError(
                "Precomputed boundary should bypass runtime boundary evaluation."
            )

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

    async def test_ticket_agent_flow_skips_outer_boundary_for_internal_team_turn(
        self,
    ) -> None:
        channel_id = 372
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        fake_runner = _FakeRunner(
            [
                _FakeResult(
                    final_output="The vault remains available through the Yearn app.",
                    last_agent=yearn_docs_qa_agent,
                    _history=[],
                )
            ]
        )
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent="docs_qa",
        )

        async def fail_boundary(_text: str):
            raise AssertionError(
                "Authorized internal-team turns must bypass the outer user boundary."
            )

        try:
            with patch(
                "ticket_investigation.runtime.evaluate_support_boundary",
                new=fail_boundary,
            ):
                runtime = TicketInvestigationRuntime(fake_runner)
                outcome = await runtime.run_turn(
                    TicketTurnRequest(
                        aggregated_text="Tell the user this vault is still available.",
                        input_list=[
                            {
                                "role": "user",
                                "content": "Tell the user this vault is still available.",
                            }
                        ],
                        current_history=[],
                        run_context=context,
                        investigation_job=investigation_job,
                        workflow_name="tests.internal_team_turn",
                        turn_source="internal_team",
                    )
                )
        finally:
            clear_ticket_investigation_job(channel_id)

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertIs(fake_runner.calls[0]["starting_agent"], yearn_docs_qa_agent)
        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertEqual(
            outcome.raw_final_reply,
            "The vault remains available through the Yearn app.",
        )

    async def test_ticket_agent_flow_marks_specialist_reply_handoff_explicitly(
        self,
    ) -> None:
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

    async def test_ticket_agent_flow_injects_tx_followup_contract_before_specialist_run(
        self,
    ) -> None:
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
                    input_list=[
                        {"role": "user", "content": "i dunno man. look into it"}
                    ],
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

    async def test_ticket_agent_flow_injects_report_pretriage_contract_before_specialist_run(
        self,
    ) -> None:
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
        self.assertIn(
            "Do one bounded repo/docs pre-triage pass", specialist_input[-1]["content"]
        )

    async def test_ticket_agent_flow_switches_from_data_followup_to_bug_for_repro_issue(
        self,
    ) -> None:
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
        self.assertIs(
            fake_runner.calls[0]["starting_agent"], ticket_triage_router_agent
        )
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

    async def test_docs_agent_system_prompt_includes_compact_mechanics_answer_rules(
        self,
    ) -> None:
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
        prompt = await yearn_bug_triage_agent.get_system_prompt(
            RunContextWrapper(context)
        )

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, prompt)
        self.assertIn(SECURITY_PROCESS_URL, prompt)
        self.assertIn("initial_button_intent: bug_report", prompt)
