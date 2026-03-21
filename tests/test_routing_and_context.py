import unittest
from dataclasses import dataclass
import os
import sys
import tempfile

from agents import RunContextWrapper

import config
from router import select_starting_agent
from state import (
    BotRunContext,
    TicketInvestigationJob,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
)
from ticket_investigation_json_endpoint import (
    ExecutorBackedTicketExecutionJsonEndpoint,
    JsonEndpointTicketExecutionTransport,
    build_ticket_execution_json_endpoint,
)
from ticket_investigation_codex_bundle import (
    DEFAULT_CODEX_EXEC_COMMAND,
    build_codex_ticket_execution_bundle,
)
from ticket_investigation_subprocess_endpoint import (
    SubprocessTicketExecutionJsonEndpoint,
)
from ticket_investigation_executor import (
    LocalTicketInvestigationExecutor,
    LoopbackTicketExecutionTransport,
    LoopbackTransportTicketInvestigationExecutor,
    TicketExecutionTransport,
    TicketExecutionHooks,
    TransportTicketInvestigationExecutor,
)
from ticket_investigation_transport import (
    TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA,
    TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA,
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)
from ticket_investigation_worker import TicketInvestigationWorker, TicketWorkerResult
from support_agents import (
    TicketTriageDecision,
    ticket_triage_router_agent,
    triage_agent,
    yearn_bug_triage_agent,
    yearn_data_agent,
    yearn_docs_qa_agent,
)
from support_tools import _extract_artifact_refs, _repo_search_block_message
from ticket_investigation_runtime import (
    TicketAgentFlowOutcome,
    _merge_explicit_evidence_into_job,
    _normalize_ticket_triage_decision,
    _reply_requests_human_handoff,
    _select_ticket_starting_agent,
    TicketInvestigationRuntime,
    TicketTurnRequest,
)


class RoutingTests(unittest.TestCase):
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

    def test_select_starting_agent_routes_styfi_question_to_docs(self) -> None:
        context = BotRunContext(channel_id=3, project_context="yearn")
        self.assertEqual(select_starting_agent("Where do I see my stYFI position?", context), "docs")

    def test_select_starting_agent_keeps_free_form_withdrawal_in_triage(self) -> None:
        context = BotRunContext(channel_id=4, project_context="yearn")
        message = (
            "How do I withdraw from vault "
            "0x2222222222222222222222222222222222222222 "
            "on ethereum using wallet 0x1111111111111111111111111111111111111111?"
        )
        self.assertEqual(select_starting_agent(message, context), "triage")

    def test_follow_up_routing_reuses_last_specialist(self) -> None:
        channel_id = 9
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = "data"
        try:
            agent_key = _select_ticket_starting_agent(
                "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0",
                context,
                current_history=[{"role": "user", "content": "Previous issue context"}],
                investigation_job=investigation_job,
            )
            self.assertEqual(agent_key, "data")
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


class InvestigationJobTests(unittest.TestCase):
    def test_job_tracks_lifecycle_and_evidence(self) -> None:
        job = TicketInvestigationJob(channel_id=77)

        job.begin_collecting("investigate_issue")
        job.remember_wallet("0x1111111111111111111111111111111111111111")
        job.remember_chain("katana")
        job.remember_tx_hash("0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0")
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

    def test_job_clears_current_specialty_when_non_specialist_turn_completes(self) -> None:
        job = TicketInvestigationJob(channel_id=78, current_specialty="bug", last_specialty="bug")

        job.complete_specialist_turn(None)

        self.assertIsNone(job.current_specialty)
        self.assertEqual(job.last_specialty, "bug")


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


class TicketFlowTests(unittest.IsolatedAsyncioTestCase):
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


@dataclass
class _FakeWorker:
    requests: list

    async def execute_turn(self, request):
        self.requests.append(request)
        request.investigation_job.begin_investigating()
        request.investigation_job.complete_specialist_turn("docs")
        return TicketWorkerResult(
            flow_outcome=TicketAgentFlowOutcome(
                raw_final_reply="ok",
                conversation_history=[],
                completed_agent_key="docs",
                requires_human_handoff=False,
            )
        )


class _FakeExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Factory tests should not execute the delegate.")


class TicketExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_executor_injects_hooks_into_request(self) -> None:
        worker = _FakeWorker(requests=[])
        executor = LocalTicketInvestigationExecutor(worker)

        async def fake_send_bug_review_status() -> None:
            return None

        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context=BotRunContext(channel_id=92, project_context="yearn"),
            investigation_job=TicketInvestigationJob(channel_id=92),
            workflow_name="tests.executor",
        )
        result = await executor.execute_turn(
            request,
            hooks=TicketExecutionHooks(
                send_bug_review_status=fake_send_bug_review_status,
            ),
        )

        self.assertEqual(len(worker.requests), 1)
        self.assertIs(worker.requests[0].send_bug_review_status, fake_send_bug_review_status)
        self.assertIsNone(request.send_bug_review_status)
        self.assertEqual(request.investigation_job.mode, "idle")
        self.assertIsNone(request.investigation_job.current_specialty)
        self.assertEqual(result.updated_job.mode, "investigating")
        self.assertEqual(result.updated_job.current_specialty, "docs")

    async def test_loopback_transport_executor_round_trips_request_and_result(self) -> None:
        worker = _FakeWorker(requests=[])
        executor = LoopbackTransportTicketInvestigationExecutor(
            LocalTicketInvestigationExecutor(worker)
        )
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context=BotRunContext(channel_id=93, project_context="yearn"),
            investigation_job=TicketInvestigationJob(channel_id=93),
            workflow_name="tests.executor",
        )

        result = await executor.execute_turn(request)

        self.assertEqual(result.flow_outcome.raw_final_reply, "ok")
        self.assertEqual(result.updated_job.current_specialty, "docs")
        self.assertEqual(result.updated_job.mode, "investigating")

    async def test_transport_executor_uses_transport_boundary(self) -> None:
        @dataclass
        class _FakeTransport:
            requests: list
            hooks: list

            async def execute_transport_turn(self, request, hooks=None):
                self.requests.append(request)
                self.hooks.append(hooks)
                updated_job = TicketInvestigationJob(
                    channel_id=request.investigation_job["channel_id"]
                )
                updated_job.begin_investigating()
                updated_job.complete_specialist_turn("bug")
                return TicketExecutionTransportResult.from_execution_parts(
                    TicketAgentFlowOutcome(
                        raw_final_reply="transport-ok",
                        conversation_history=[],
                        completed_agent_key="bug",
                        requires_human_handoff=False,
                    ),
                    updated_job,
                )

        async def fake_send_bug_review_status() -> None:
            return None

        transport: TicketExecutionTransport = _FakeTransport(requests=[], hooks=[])
        executor = TransportTicketInvestigationExecutor(transport)
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context=BotRunContext(channel_id=95, project_context="yearn"),
            investigation_job=TicketInvestigationJob(channel_id=95),
            workflow_name="tests.executor.transport",
        )

        result = await executor.execute_turn(
            request,
            hooks=TicketExecutionHooks(
                send_bug_review_status=fake_send_bug_review_status,
            ),
        )

        self.assertEqual(len(transport.requests), 1)
        self.assertTrue(transport.requests[0].wants_bug_review_status)
        self.assertIs(transport.hooks[0].send_bug_review_status, fake_send_bug_review_status)
        self.assertEqual(result.flow_outcome.raw_final_reply, "transport-ok")
        self.assertEqual(result.updated_job.current_specialty, "bug")
        self.assertEqual(result.updated_job.mode, "investigating")

    async def test_loopback_transport_rehydrates_request_for_delegate(self) -> None:
        worker = _FakeWorker(requests=[])
        transport = LoopbackTicketExecutionTransport(
            LocalTicketInvestigationExecutor(worker)
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 96,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 96,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.transport.loopback",
            wants_bug_review_status=False,
        )

        result = await transport.execute_transport_turn(request)
        flow_outcome, updated_job = result.to_execution_parts()

        self.assertEqual(len(worker.requests), 1)
        self.assertEqual(worker.requests[0].run_context.channel_id, 96)
        self.assertEqual(flow_outcome.raw_final_reply, "ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_executor_backed_json_endpoint_round_trips_request_and_result(self) -> None:
        worker = _FakeWorker(requests=[])
        endpoint = ExecutorBackedTicketExecutionJsonEndpoint(
            LocalTicketInvestigationExecutor(worker)
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 97,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 97,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.json",
            wants_bug_review_status=False,
        )

        response_json = await endpoint.execute_json_turn(request.to_json())
        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()

        self.assertEqual(len(worker.requests), 1)
        self.assertEqual(worker.requests[0].run_context.channel_id, 97)
        self.assertEqual(flow_outcome.raw_final_reply, "ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_json_endpoint_transport_uses_json_boundary(self) -> None:
        @dataclass
        class _FakeJsonEndpoint:
            requests: list
            hooks: list

            async def execute_json_turn(self, request_json, hooks=None):
                self.requests.append(request_json)
                self.hooks.append(hooks)
                request = TicketExecutionTransportRequest.from_json(request_json)
                updated_job = TicketInvestigationJob(
                    channel_id=request.investigation_job["channel_id"]
                )
                updated_job.begin_investigating()
                updated_job.complete_specialist_turn("docs")
                return TicketExecutionTransportResult.from_execution_parts(
                    TicketAgentFlowOutcome(
                        raw_final_reply="json-ok",
                        conversation_history=[],
                        completed_agent_key="docs",
                        requires_human_handoff=False,
                    ),
                    updated_job,
                ).to_json()

        transport = JsonEndpointTicketExecutionTransport(
            _FakeJsonEndpoint(requests=[], hooks=[])
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 98,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 98,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.transport",
            wants_bug_review_status=False,
        )

        result = await transport.execute_transport_turn(request)
        flow_outcome, updated_job = result.to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "json-ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_subprocess_json_endpoint_round_trips_response(self) -> None:
        endpoint = SubprocessTicketExecutionJsonEndpoint(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "request=json.loads(sys.stdin.read()); "
                    "response={"
                    "'flow_outcome':{"
                    "'raw_final_reply':'subprocess-ok',"
                    "'conversation_history':[],"
                    "'completed_agent_key':'docs',"
                    "'requires_human_handoff':False"
                    "},"
                    "'updated_job':{"
                    "'channel_id':request['investigation_job']['channel_id'],"
                    "'requested_intent':request['investigation_job'].get('requested_intent'),"
                    "'mode':'investigating',"
                    "'current_specialty':'docs',"
                    "'last_specialty':'docs',"
                    "'evidence':request['investigation_job'].get('evidence',{})"
                    "}"
                    "}; "
                    "sys.stdout.write(json.dumps(response))"
                ),
            ]
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 99,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 99,
                "requested_intent": "investigate_issue",
                "mode": "collecting",
                "evidence": {"wallet": None, "chain": "katana", "tx_hashes": []},
            },
            workflow_name="tests.endpoint.subprocess",
            wants_bug_review_status=False,
        )

        response_json = await endpoint.execute_json_turn(request.to_json())
        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "subprocess-ok")
        self.assertEqual(updated_job.channel_id, 99)
        self.assertEqual(updated_job.current_specialty, "docs")
        self.assertEqual(updated_job.evidence.chain, "katana")

    async def test_subprocess_json_endpoint_sends_bug_review_hook_before_spawn(self) -> None:
        hook_calls: list[str] = []

        async def fake_send_bug_review_status() -> None:
            hook_calls.append("sent")

        endpoint = SubprocessTicketExecutionJsonEndpoint(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "request=json.loads(sys.stdin.read()); "
                    "response={"
                    "'flow_outcome':{"
                    "'raw_final_reply':'hook-ok',"
                    "'conversation_history':[],"
                    "'completed_agent_key':'bug',"
                    "'requires_human_handoff':False"
                    "},"
                    "'updated_job':{"
                    "'channel_id':request['investigation_job']['channel_id'],"
                    "'mode':'investigating',"
                    "'current_specialty':'bug',"
                    "'last_specialty':'bug',"
                    "'evidence':request['investigation_job'].get('evidence',{})"
                    "}"
                    "}; "
                    "sys.stdout.write(json.dumps(response))"
                ),
            ]
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 100,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 100,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.subprocess_hook",
            wants_bug_review_status=True,
        )

        response_json = await endpoint.execute_json_turn(
            request.to_json(),
            hooks=TicketExecutionHooks(
                send_bug_review_status=fake_send_bug_review_status,
            ),
        )
        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()

        self.assertEqual(hook_calls, ["sent"])
        self.assertEqual(flow_outcome.raw_final_reply, "hook-ok")
        self.assertEqual(updated_job.current_specialty, "bug")

    def test_subprocess_json_endpoint_rejects_disallowed_command(self) -> None:
        with self.assertRaises(ValueError):
            SubprocessTicketExecutionJsonEndpoint(
                [sys.executable, "-c", "print('nope')"],
                allowed_command_prefixes=[["codex", "exec"]],
            )

    async def test_subprocess_json_endpoint_rejects_oversized_stdout(self) -> None:
        endpoint = SubprocessTicketExecutionJsonEndpoint(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('x' * 50)",
            ],
            max_output_chars=10,
        )

        with self.assertRaises(RuntimeError):
            await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 101,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 101,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_oversized",
                    wants_bug_review_status=False,
                ).to_json()
            )

    async def test_subprocess_json_endpoint_uses_explicit_env_without_parent_inheritance(self) -> None:
        original_blocked = os.environ.get("BLOCKED_TEST_ENV")
        os.environ["BLOCKED_TEST_ENV"] = "blocked"
        try:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,os,sys; "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':f\"{os.getenv('ALLOWED_TEST_ENV')}:{os.getenv('BLOCKED_TEST_ENV')}\","
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':102,"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':{'tx_hashes':[]}"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                env={"ALLOWED_TEST_ENV": "allowed"},
                inherit_parent_env=False,
            )

            response_json = await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 102,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 102,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_env",
                    wants_bug_review_status=False,
                ).to_json()
            )
        finally:
            if original_blocked is None:
                os.environ.pop("BLOCKED_TEST_ENV", None)
            else:
                os.environ["BLOCKED_TEST_ENV"] = original_blocked

        flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "allowed:None")

    async def test_subprocess_json_endpoint_uses_configured_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,os,sys; "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':os.getcwd(),"
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':103,"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':{'tx_hashes':[]}"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                cwd=temp_dir,
            )

            response_json = await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 103,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 103,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_cwd",
                    wants_bug_review_status=False,
                ).to_json()
            )

        flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, temp_dir)

    async def test_subprocess_json_endpoint_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as artifact_dir:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,sys; "
                        "print('worker-stderr', file=sys.stderr); "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':'artifact-ok',"
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':104,"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':{'tx_hashes':[]}"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                artifact_dir=artifact_dir,
            )

            response_json = await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 104,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 104,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_artifacts",
                    wants_bug_review_status=False,
                ).to_json()
            )

            artifact_entries = os.listdir(artifact_dir)
            self.assertEqual(len(artifact_entries), 1)
            run_dir = os.path.join(artifact_dir, artifact_entries[0])
            self.assertTrue(os.path.exists(os.path.join(run_dir, "request.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "request_schema.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "response_schema.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "stdout.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "stderr.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "metadata.json")))
            with open(os.path.join(run_dir, "stdout.txt"), encoding="utf-8") as stdout_file:
                stdout_text = stdout_file.read()
            with open(os.path.join(run_dir, "stderr.txt"), encoding="utf-8") as stderr_file:
                stderr_text = stderr_file.read()
            self.assertIn(
                "artifact-ok",
                stdout_text,
            )
            self.assertIn(
                "worker-stderr",
                stderr_text,
            )

        flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "artifact-ok")

    async def test_subprocess_json_endpoint_exports_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as run_root:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,os,sys; "
                        "run_dir=os.getenv('TICKET_EXECUTION_RUN_DIR'); "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':run_dir,"
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':105,"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':{'tx_hashes':[]}"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                run_dir_root=run_root,
            )

            response_json = await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 105,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 105,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_run_dir",
                    wants_bug_review_status=False,
                ).to_json()
            )

            flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
                response_json
            ).to_execution_parts()
            self.assertIsNotNone(flow_outcome.raw_final_reply)
            assert flow_outcome.raw_final_reply is not None
            self.assertTrue(flow_outcome.raw_final_reply.startswith(run_root))
            self.assertTrue(os.path.isdir(flow_outcome.raw_final_reply))


class TicketExecutionEndpointFactoryTests(unittest.TestCase):
    def test_build_endpoint_returns_local_endpoint_by_default(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        try:
            config.TICKET_EXECUTION_ENDPOINT = "local"
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command

        self.assertIsInstance(endpoint, ExecutorBackedTicketExecutionJsonEndpoint)

    def test_build_endpoint_returns_subprocess_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = []
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

        self.assertIsInstance(endpoint, SubprocessTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.command[1:], ["-m", "ticket_investigation_worker_cli"])

    def test_build_endpoint_allows_configured_subprocess_prefix(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

        self.assertIsInstance(endpoint, SubprocessTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.command[:2], ["codex", "exec"])

    def test_build_endpoint_rejects_unknown_mode(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "invalid"
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = []
            with self.assertRaises(ValueError):
                build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes


class TicketTransportTests(unittest.TestCase):
    def test_transport_schemas_cover_required_top_level_fields(self) -> None:
        self.assertEqual(
            TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA["required"],
            [
                "aggregated_text",
                "input_list",
                "current_history",
                "run_context",
                "investigation_job",
                "workflow_name",
                "wants_bug_review_status",
            ],
        )
        self.assertEqual(
            TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA["required"],
            ["flow_outcome", "updated_job"],
        )

    def test_transport_request_round_trip_preserves_job_and_context(self) -> None:
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[{"role": "assistant", "content": "context"}],
            run_context=BotRunContext(
                channel_id=94,
                category_id=12,
                project_context="yearn",
                initial_button_intent="investigate_issue",
            ),
            investigation_job=TicketInvestigationJob(channel_id=94),
            workflow_name="tests.transport",
        )
        request.investigation_job.begin_collecting("investigate_issue")
        request.investigation_job.remember_chain("katana")
        request.investigation_job.remember_tx_hash(
            "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        )

        transport = TicketExecutionTransportRequest.from_turn_request(
            request,
            wants_bug_review_status=True,
        )
        hydrated = transport.to_turn_request()

        self.assertTrue(transport.wants_bug_review_status)
        self.assertEqual(hydrated.run_context.channel_id, 94)
        self.assertEqual(hydrated.run_context.initial_button_intent, "investigate_issue")
        self.assertEqual(hydrated.investigation_job.mode, "collecting")
        self.assertEqual(hydrated.investigation_job.evidence.chain, "katana")
        self.assertEqual(
            hydrated.investigation_job.evidence.tx_hashes,
            ["0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"],
        )

    def test_transport_request_json_round_trip_preserves_job_and_context(self) -> None:
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[{"role": "assistant", "content": "context"}],
            run_context=BotRunContext(
                channel_id=97,
                category_id=12,
                project_context="yearn",
                initial_button_intent="investigate_issue",
            ),
            investigation_job=TicketInvestigationJob(channel_id=97),
            workflow_name="tests.transport.json",
        )
        request.investigation_job.begin_collecting("investigate_issue")
        request.investigation_job.remember_chain("katana")
        transport = TicketExecutionTransportRequest.from_turn_request(
            request,
            wants_bug_review_status=True,
        )

        hydrated = TicketExecutionTransportRequest.from_json(
            transport.to_json()
        ).to_turn_request()

        self.assertEqual(hydrated.run_context.channel_id, 97)
        self.assertEqual(hydrated.investigation_job.mode, "collecting")
        self.assertEqual(hydrated.investigation_job.evidence.chain, "katana")

    def test_transport_result_round_trip_preserves_flow_and_job(self) -> None:
        job = TicketInvestigationJob(channel_id=95)
        job.begin_investigating()
        job.complete_specialist_turn("bug")
        result = TicketExecutionTransportResult.from_execution_parts(
            TicketAgentFlowOutcome(
                raw_final_reply="ok",
                conversation_history=[{"role": "assistant", "content": "ok"}],
                completed_agent_key="bug",
                requires_human_handoff=False,
            ),
            job,
        )

        flow_outcome, updated_job = result.to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "ok")
        self.assertEqual(flow_outcome.completed_agent_key, "bug")
        self.assertEqual(updated_job.mode, "investigating")
        self.assertEqual(updated_job.current_specialty, "bug")


class CodexBundleTests(unittest.TestCase):
    def test_build_codex_ticket_execution_bundle_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = build_codex_ticket_execution_bundle(
                request_json='{"example":"request"}',
                run_dir=temp_dir,
                repo_root="/root/bots/discord/ysupport",
            )

            self.assertEqual(bundle.command[: len(DEFAULT_CODEX_EXEC_COMMAND)], DEFAULT_CODEX_EXEC_COMMAND)
            self.assertTrue(bundle.request_path.exists())
            self.assertTrue(bundle.response_schema_path.exists())
            self.assertTrue(bundle.prompt_path.exists())
            self.assertIn("request.json", bundle.prompt_text)
            self.assertIn("response_schema.json", bundle.prompt_text)
            self.assertIn("--output-schema", bundle.command)
            self.assertIn("--add-dir", bundle.command)

    def test_build_codex_ticket_execution_bundle_appends_model_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = build_codex_ticket_execution_bundle(
                request_json='{"example":"request"}',
                run_dir=temp_dir,
                repo_root="/root/bots/discord/ysupport",
                model="gpt-5.4",
            )

            self.assertIn("-m", bundle.command)
            self.assertIn("gpt-5.4", bundle.command)

    def test_transport_result_json_round_trip_preserves_flow_and_job(self) -> None:
        job = TicketInvestigationJob(channel_id=98)
        job.begin_investigating()
        job.complete_specialist_turn("data")
        transport_result = TicketExecutionTransportResult.from_execution_parts(
            TicketAgentFlowOutcome(
                raw_final_reply="answer",
                conversation_history=[{"role": "assistant", "content": "answer"}],
                completed_agent_key="data",
                requires_human_handoff=False,
            ),
            job,
        )

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            transport_result.to_json()
        ).to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "answer")
        self.assertEqual(flow_outcome.completed_agent_key, "data")
        self.assertEqual(updated_job.mode, "investigating")
        self.assertEqual(updated_job.current_specialty, "data")


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
