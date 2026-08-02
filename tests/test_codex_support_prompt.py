import tests as _test_environment  # noqa: F401

from pathlib import Path
import unittest

import config
from codex_support_contract import (
    SupportTurnRequest,
)
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
    _codex_support_prompt,
    _codex_support_transaction_safety_rewrite_prompt,
)
from ticket_investigation.json_endpoint import build_ticket_execution_json_endpoint
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
)


class _FakeExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Factory test should not execute the delegate.")


EXAMPLE_YSUPPORT_MCP_URL = "http://ysupport-mcp.example.test/mcp"
_SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION = "0xf8cb" + ("ab" * 203)
_SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION = "0x02f8" + ("ab" * 120)
_SYNTHETIC_HIGH_TYPE_RAW_SIGNED_TRANSACTION = "0x7afa" + ("ab" * 120)
_SHORT_LEGACY_RAW_SIGNED_TRANSACTION = (
    "0xf85f8001825208940000000000000000000000000000000000000000808025"
    "a05a420b0a542873e0f1a0a6bcf149ab3d26204c0fe61ebcb30dad82a8e7e9a370"
    "a057d3f4e87c966ab79a22aa4fbfcffd48f586a65190125aac497aacafab2a7a6f"
)


def _transaction_safety_support_request() -> SupportTurnRequest:
    return SupportTurnRequest(
        current_user_message="toujours pas",
        recent_transcript=[],
        channel_type="ticket",
        channel_id=1,
        project_context="yearn",
        workflow_name="tests.verify",
        initial_button_intent="investigate_issue",
        requested_intent="investigate_issue",
        evidence={},
        support_state={},
        constraints={"allowed_tools": ["shell"]},
    )


def _transaction_safety_transport_request(
    *,
    history: list[dict[str, str]] | None = None,
) -> TicketExecutionTransportRequest:
    return TicketExecutionTransportRequest(
        aggregated_text="toujours pas",
        input_list=[],
        current_history=history or [],
        run_context={
            "channel_id": 109,
            "project_context": "yearn",
            "initial_button_intent": "investigate_issue",
            "repo_last_search_artifact_refs": [],
        },
        investigation_job={
            "channel_id": 109,
            "requested_intent": "investigate_issue",
            "mode": "collecting",
            "evidence": {"wallet": None, "chain": "katana", "tx_hashes": []},
        },
        workflow_name="tests.endpoint.codex_support_exec",
        wants_bug_review_status=False,
    )


class CodexSupportEndpointTests(unittest.IsolatedAsyncioTestCase):
    def test_codex_support_prompt_requests_fuller_prose_for_investigations(
        self,
    ) -> None:
        request_path = Path("support_request.json")
        schema_path = Path("support_response_schema.json")
        prompt_text = _codex_support_prompt(
            support_request_path=request_path,
            response_schema_path=schema_path,
        )

        self.assertIn(str(request_path.resolve()), prompt_text)
        self.assertIn(str(schema_path.resolve()), prompt_text)
        self.assertIn("Routine support: concise.", prompt_text)
        self.assertIn("Investigations and report triage: enough prose", prompt_text)
        self.assertIn(
            "Do not mention handoff if public evidence already answers the main question.",
            prompt_text,
        )
        self.assertIn(
            "exhaust the relevant available documentation, live-data, repository, web, and image evidence",
            prompt_text,
        )
        self.assertIn(
            "does not by itself justify handoff",
            prompt_text,
        )
        self.assertIn(
            "Never describe a required human or team action while returning requires_human_handoff=false.",
            prompt_text,
        )
        self.assertIn(
            "access_or_permission_action, fund_or_account_recovery, security_process, manual_strategy_action, private_internal_fact, or human_decision",
            prompt_text,
        )
        self.assertIn(
            "do not claim that you have escalated, handed off, or notified anyone",
            prompt_text,
        )
        self.assertIn("Use `current_turn_source`", prompt_text)
        self.assertIn("If `current_turn_source` is `internal_team`", prompt_text)
        self.assertIn(
            "synthesize a concise direct answer from the Yearn documentation excerpts",
            prompt_text,
        )
        self.assertIn(
            "Do not expose retrieval metadata",
            prompt_text,
        )
        rewrite_prompt_text = _codex_support_transaction_safety_rewrite_prompt(
            response_schema_path=Path("support_response_schema.json"),
        )
        for rendered_prompt in (prompt_text, rewrite_prompt_text):
            self.assertIn(
                "Never ask for, retrieve, retain, reconstruct, quote, display, submit, broadcast, or recommend manually broadcasting a raw signed transaction.",
                rendered_prompt,
            )
            self.assertIn(
                "Reaching this safety boundary does not by itself justify human handoff.",
                rendered_prompt,
            )
        self.assertNotIn(
            "documentation tool already returns a complete answer",
            prompt_text,
        )

    def test_codex_support_prompt_leads_ambiguous_bug_intake_with_security_path(
        self,
    ) -> None:
        prompt_text = _codex_support_prompt(
            support_request_path=Path("support_request.json"),
            response_schema_path=Path("support_response_schema.json"),
        )

        security_path_index = prompt_text.index(
            "begin the reply with https://github.com/yearn/yearn-security/blob/master/SECURITY.md"
        )
        product_intake_index = prompt_text.index(
            "Only after that, offer to accept ordinary product-bug details."
        )
        self.assertLess(security_path_index, product_intake_index)
        self.assertIn(
            "Do not stop or request human handoff solely because the user used generic bug-report wording.",
            prompt_text,
        )

    def test_codex_support_prompt_requires_complete_gas_sufficiency_evidence(
        self,
    ) -> None:
        prompt_text = _codex_support_prompt(
            support_request_path=Path("support_request.json"),
            response_schema_path=Path("support_response_schema.json"),
        )
        rewrite_prompt_text = _codex_support_transaction_safety_rewrite_prompt(
            response_schema_path=Path("support_response_schema.json"),
        )

        for rendered_prompt in (prompt_text, rewrite_prompt_text):
            self.assertIn(
                "transaction's native-token value plus its maximum gas cost",
                rendered_prompt,
            )
            self.assertIn(
                "gas limit multiplied by maximum fee per gas, or by legacy gas price",
                rendered_prompt,
            )
            self.assertIn("Retain a conservative buffer", rendered_prompt)
            self.assertIn(
                "gas and native-token value committed by pending or wallet-queued transactions",
                rendered_prompt,
            )
            self.assertIn(
                "Never claim the wallet definitely has enough gas from its current balance alone.",
                rendered_prompt,
            )
            self.assertIn(
                "transaction-value, fee, or queue evidence is unknown",
                rendered_prompt,
            )
            self.assertIn(
                "sufficiency is conditional and name the missing check",
                rendered_prompt,
            )

    def test_codex_support_prompt_requires_economic_claim_reconciliation(self) -> None:
        prompt_text = _codex_support_prompt(
            support_request_path=Path("support_request.json"),
            response_schema_path=Path("support_response_schema.json"),
        )

        self.assertIn(
            "reconcile displayed balances with presently realizable wallet positions and available history",
            prompt_text,
        )
        self.assertIn(
            "whether the value was already redeemed, migrated, distributed, or represented elsewhere",
            prompt_text,
        )
        self.assertIn(
            "does not prove a current economic claim",
            prompt_text,
        )
        self.assertIn(
            "do not claim the funds are safe, stuck, claimable, awaiting liquidity or operator action, or will later become redeemable",
            prompt_text,
        )
        for temporary_product_term in ("kpdUSDC", "yvvbUSDC", "Katana pre-deposit"):
            self.assertNotIn(temporary_product_term, prompt_text)

    def test_codex_support_runtime_validation_requires_dedicated_home(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_home = config.TICKET_EXECUTION_CODEX_HOME
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_api_key = config.MCP_SERVER_API_KEY
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_HOME = None
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            config.MCP_SERVER_API_KEY = "secret-key"
            with self.assertRaises(ValueError):
                config.validate_ticket_execution_runtime_config()
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_HOME = original_home
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
            config.MCP_SERVER_API_KEY = original_api_key

    def test_endpoint_factory_builds_codex_support_endpoint(self) -> None:
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        original_home = config.TICKET_EXECUTION_CODEX_HOME
        original_api_key = config.MCP_SERVER_API_KEY
        try:
            config.TICKET_EXECUTION_ENDPOINT = "codex_support_exec"
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
            config.TICKET_EXECUTION_CODEX_COMMAND = ["codex", "exec", "--json"]
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [["codex", "exec"]]
            config.TICKET_EXECUTION_ARTIFACT_DIR = "/tmp/ticket-artifacts"
            config.TICKET_EXECUTION_CODEX_HOME = "/tmp/ysupport-codex-home"
            config.MCP_SERVER_API_KEY = "secret-key"
            endpoint = build_ticket_execution_json_endpoint(_FakeExecutor())
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_COMMAND = original_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir
            config.TICKET_EXECUTION_CODEX_HOME = original_home
            config.MCP_SERVER_API_KEY = original_api_key

        self.assertIsInstance(endpoint, CodexSupportTicketExecutionJsonEndpoint)
