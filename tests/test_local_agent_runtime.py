import tests as _test_environment  # noqa: F401

import unittest
from dataclasses import dataclass
from unittest.mock import patch


from agents import InputGuardrailTripwireTriggered

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
