import unittest


from ysupport import config
from ysupport.bot.state import BotRunContext, TicketInvestigationJob
from ysupport.execution.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ysupport.execution.json_endpoint import (
    FailoverTicketExecutionJsonEndpoint,
    build_ticket_execution_json_endpoint,
)
from ysupport.execution.subprocess_endpoint import SubprocessTicketExecutionJsonEndpoint
from ysupport.execution.contracts import TicketTurnRequest
from ysupport.execution.transport import (
    TicketExecutionTransportRequest,
)


class TicketExecutionEndpointFactoryTests(unittest.TestCase):
    def test_build_endpoint_wraps_primary_with_fallback_when_configured(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_subprocess_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = ["external-worker"]
            config.TICKET_EXECUTION_CODEX_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            endpoint = build_ticket_execution_json_endpoint()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_subprocess_command
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIsInstance(endpoint, FailoverTicketExecutionJsonEndpoint)
        self.assertIsInstance(endpoint.primary, CodexSupportTicketExecutionJsonEndpoint)
        self.assertIsInstance(
            endpoint.fallback, SubprocessTicketExecutionJsonEndpoint
        )


class TicketTransportTests(unittest.TestCase):
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
            precomputed_boundary={
                "classification": "business_boundary",
                "business_subtype": "job_inquiry",
                "tripwire_triggered": True,
                "message": "Boundary reply",
            },
        )
        request.investigation_job.begin_collecting("investigate_issue")
        request.investigation_job.remember_chain("katana")
        request.investigation_job.remember_tx_hash(
            "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        )
        transport = TicketExecutionTransportRequest.from_turn_request(request)

        hydrated = TicketExecutionTransportRequest.from_json(
            transport.to_json()
        ).to_turn_request()

        self.assertEqual(hydrated.run_context.channel_id, 97)
        self.assertEqual(hydrated.run_context.initial_button_intent, "investigate_issue")
        self.assertEqual(
            hydrated.investigation_job.evidence.tx_hashes,
            ["0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"],
        )
        self.assertEqual(hydrated.investigation_job.mode, "collecting")
        self.assertEqual(hydrated.investigation_job.evidence.chain, "katana")
        self.assertEqual(
            hydrated.precomputed_boundary,
            {
                "classification": "business_boundary",
                "business_subtype": "job_inquiry",
                "tripwire_triggered": True,
                "message": "Boundary reply",
            },
        )
        self.assertIsNone(
            TicketExecutionTransportRequest.from_json(transport.to_json()).smoke_mode
        )
