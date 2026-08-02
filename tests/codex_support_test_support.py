from codex_support_contract import SupportTurnRequest
from ticket_investigation.transport import TicketExecutionTransportRequest


class FakeExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Factory test should not execute the delegate.")


EXAMPLE_YSUPPORT_MCP_URL = "http://ysupport-mcp.example.test/mcp"
SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION = "0xf8cb" + ("ab" * 203)
SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION = "0x02f8" + ("ab" * 120)
SYNTHETIC_HIGH_TYPE_RAW_SIGNED_TRANSACTION = "0x7afa" + ("ab" * 120)
SHORT_LEGACY_RAW_SIGNED_TRANSACTION = (
    "0xf85f8001825208940000000000000000000000000000000000000000808025"
    "a05a420b0a542873e0f1a0a6bcf149ab3d26204c0fe61ebcb30dad82a8e7e9a370"
    "a057d3f4e87c966ab79a22aa4fbfcffd48f586a65190125aac497aacafab2a7a6f"
)


def transaction_safety_support_request() -> SupportTurnRequest:
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


def transaction_safety_transport_request(
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
