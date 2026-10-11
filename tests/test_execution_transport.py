import unittest


import config
from state import BotRunContext, TicketInvestigationJob
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ticket_investigation.json_endpoint import (
    ExecutorBackedTicketExecutionJsonEndpoint,
    FailoverTicketExecutionJsonEndpoint,
    build_ticket_execution_json_endpoint,
)
from ticket_investigation.runtime import TicketTurnRequest
from ticket_investigation.subprocess_endpoint import (
    SubprocessTicketExecutionJsonEndpoint,
)
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
)


class _FakeExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Factory tests should not execute the delegate.")


class TicketExecutionEndpointFactoryTests(unittest.TestCase):
    def test_build_endpoint_returns_local_endpoint_by_default(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        try:
            config.TICKET_EXECUTION_ENDPOINT = "local"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command

        self.assertIsInstance(endpoint, ExecutorBackedTicketExecutionJsonEndpoint)

    def test_build_endpoint_returns_subprocess_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = []
            endpoint = build_ticket_execution_json_endpoint()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

        self.assertIsInstance(endpoint, SubprocessTicketExecutionJsonEndpoint)
        self.assertEqual(
            endpoint.command[1:], ["-m", "ticket_investigation.worker_cli"]
        )

    def test_build_endpoint_allows_configured_subprocess_prefix(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "subprocess"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

        self.assertIsInstance(endpoint, SubprocessTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.command[:2], ["codex", "exec"])

    def test_build_endpoint_rejects_unknown_mode(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_SUBPROCESS_COMMAND
        original_codex_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        try:
            config.TICKET_EXECUTION_ENDPOINT = "invalid"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = []
            config.TICKET_EXECUTION_CODEX_COMMAND = []
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = []
            with self.assertRaises(ValueError):
                build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_SUBPROCESS_COMMAND = original_command
            config.TICKET_EXECUTION_CODEX_COMMAND = original_codex_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes

    def test_build_endpoint_returns_codex_support_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_model = config.TICKET_EXECUTION_CODEX_MODEL
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_CODEX_MODEL = "gpt-5.6-sol"
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            endpoint = build_ticket_execution_json_endpoint()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_CODEX_MODEL = original_model
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIsInstance(endpoint, CodexSupportTicketExecutionJsonEndpoint)
        self.assertEqual(endpoint.codex_command[:2], ["codex", "exec"])
        self.assertEqual(endpoint.model, "gpt-5.6-sol")

    def test_build_local_endpoint_requires_explicit_executor(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        try:
            config.TICKET_EXECUTION_ENDPOINT = "local"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            with self.assertRaisesRegex(
                ValueError, "requires an explicit local executor"
            ):
                build_ticket_execution_json_endpoint()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode

    def test_build_endpoint_wraps_primary_with_fallback_when_configured(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
            config.TICKET_EXECUTION_CODEX_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertIsInstance(endpoint, FailoverTicketExecutionJsonEndpoint)
        self.assertIsInstance(endpoint.primary, CodexSupportTicketExecutionJsonEndpoint)
        self.assertIsInstance(
            endpoint.fallback, ExecutorBackedTicketExecutionJsonEndpoint
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
