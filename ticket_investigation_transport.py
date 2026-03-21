from dataclasses import dataclass
import json
from typing import Any

from agents import TResponseInputItem

from state import BotRunContext, InvestigationEvidence, TicketInvestigationJob
from ticket_investigation_runtime import TicketAgentFlowOutcome, TicketTurnRequest


TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "aggregated_text",
        "input_list",
        "current_history",
        "run_context",
        "investigation_job",
        "workflow_name",
        "wants_bug_review_status",
    ],
    "properties": {
        "aggregated_text": {"type": "string"},
        "input_list": {"type": "array"},
        "current_history": {"type": "array"},
        "run_context": {"type": "object"},
        "investigation_job": {"type": "object"},
        "workflow_name": {"type": "string"},
        "wants_bug_review_status": {"type": "boolean"},
        "smoke_mode": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["flow_outcome", "updated_job"],
    "properties": {
        "flow_outcome": {
            "type": "object",
            "required": [
                "raw_final_reply",
                "conversation_history",
                "completed_agent_key",
                "requires_human_handoff",
            ],
            "properties": {
                "raw_final_reply": {"type": "string"},
                "conversation_history": {"type": "array"},
                "completed_agent_key": {"type": ["string", "null"]},
                "requires_human_handoff": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "updated_job": {"type": "object"},
    },
    "additionalProperties": False,
}


@dataclass
class TicketExecutionTransportRequest:
    aggregated_text: str
    input_list: list[TResponseInputItem]
    current_history: list[TResponseInputItem]
    run_context: dict[str, Any]
    investigation_job: dict[str, Any]
    workflow_name: str
    wants_bug_review_status: bool = False
    smoke_mode: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "aggregated_text": self.aggregated_text,
            "input_list": list(self.input_list),
            "current_history": list(self.current_history),
            "run_context": dict(self.run_context),
            "investigation_job": dict(self.investigation_job),
            "workflow_name": self.workflow_name,
            "wants_bug_review_status": self.wants_bug_review_status,
            "smoke_mode": self.smoke_mode,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> "TicketExecutionTransportRequest":
        return cls(
            aggregated_text=payload["aggregated_text"],
            input_list=list(payload.get("input_list", [])),
            current_history=list(payload.get("current_history", [])),
            run_context=dict(payload.get("run_context", {})),
            investigation_job=dict(payload.get("investigation_job", {})),
            workflow_name=payload["workflow_name"],
            wants_bug_review_status=payload.get("wants_bug_review_status", False),
            smoke_mode=payload.get("smoke_mode"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_payload())

    @classmethod
    def from_json(cls, raw_json: str) -> "TicketExecutionTransportRequest":
        return cls.from_payload(json.loads(raw_json))

    @classmethod
    def from_turn_request(
        cls,
        request: TicketTurnRequest,
        *,
        wants_bug_review_status: bool = False,
    ) -> "TicketExecutionTransportRequest":
        return cls(
            aggregated_text=request.aggregated_text,
            input_list=list(request.input_list),
            current_history=list(request.current_history),
            run_context={
                "channel_id": request.run_context.channel_id,
                "category_id": request.run_context.category_id,
                "is_public_trigger": request.run_context.is_public_trigger,
                "project_context": request.run_context.project_context,
                "initial_button_intent": request.run_context.initial_button_intent,
                "repo_search_calls": request.run_context.repo_search_calls,
                "repo_fetch_calls": request.run_context.repo_fetch_calls,
                "repo_searches_without_fetch": request.run_context.repo_searches_without_fetch,
                "repo_last_search_query": request.run_context.repo_last_search_query,
                "repo_last_search_artifact_refs": list(request.run_context.repo_last_search_artifact_refs),
            },
            investigation_job={
                "channel_id": request.investigation_job.channel_id,
                "requested_intent": request.investigation_job.requested_intent,
                "mode": request.investigation_job.mode,
                "current_specialty": request.investigation_job.current_specialty,
                "last_specialty": request.investigation_job.last_specialty,
                "evidence": {
                    "wallet": request.investigation_job.evidence.wallet,
                    "chain": request.investigation_job.evidence.chain,
                    "tx_hashes": list(request.investigation_job.evidence.tx_hashes),
                },
            },
            workflow_name=request.workflow_name,
            wants_bug_review_status=wants_bug_review_status,
            smoke_mode=None,
        )

    def to_turn_request(self) -> TicketTurnRequest:
        return TicketTurnRequest(
            aggregated_text=self.aggregated_text,
            input_list=list(self.input_list),
            current_history=list(self.current_history),
            run_context=BotRunContext(
                channel_id=self.run_context["channel_id"],
                category_id=self.run_context.get("category_id"),
                is_public_trigger=self.run_context.get("is_public_trigger", False),
                project_context=self.run_context.get("project_context", "unknown"),
                initial_button_intent=self.run_context.get("initial_button_intent"),
                repo_search_calls=self.run_context.get("repo_search_calls", 0),
                repo_fetch_calls=self.run_context.get("repo_fetch_calls", 0),
                repo_searches_without_fetch=self.run_context.get("repo_searches_without_fetch", 0),
                repo_last_search_query=self.run_context.get("repo_last_search_query"),
                repo_last_search_artifact_refs=list(self.run_context.get("repo_last_search_artifact_refs", [])),
            ),
            investigation_job=TicketInvestigationJob(
                channel_id=self.investigation_job["channel_id"],
                requested_intent=self.investigation_job.get("requested_intent"),
                mode=self.investigation_job.get("mode", "idle"),
                current_specialty=self.investigation_job.get("current_specialty"),
                last_specialty=self.investigation_job.get("last_specialty"),
                evidence=InvestigationEvidence(
                    wallet=self.investigation_job.get("evidence", {}).get("wallet"),
                    chain=self.investigation_job.get("evidence", {}).get("chain"),
                    tx_hashes=list(self.investigation_job.get("evidence", {}).get("tx_hashes", [])),
                ),
            ),
            workflow_name=self.workflow_name,
        )


@dataclass
class TicketExecutionTransportResult:
    flow_outcome: dict[str, Any]
    updated_job: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "flow_outcome": dict(self.flow_outcome),
            "updated_job": dict(self.updated_job),
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> "TicketExecutionTransportResult":
        return cls(
            flow_outcome=dict(payload.get("flow_outcome", {})),
            updated_job=dict(payload.get("updated_job", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_payload())

    @classmethod
    def from_json(cls, raw_json: str) -> "TicketExecutionTransportResult":
        return cls.from_payload(json.loads(raw_json))

    @classmethod
    def from_execution_parts(
        cls,
        flow_outcome: TicketAgentFlowOutcome,
        updated_job: TicketInvestigationJob,
    ) -> "TicketExecutionTransportResult":
        return cls(
            flow_outcome={
                "raw_final_reply": flow_outcome.raw_final_reply,
                "conversation_history": list(flow_outcome.conversation_history),
                "completed_agent_key": flow_outcome.completed_agent_key,
                "requires_human_handoff": flow_outcome.requires_human_handoff,
            },
            updated_job={
                "channel_id": updated_job.channel_id,
                "requested_intent": updated_job.requested_intent,
                "mode": updated_job.mode,
                "current_specialty": updated_job.current_specialty,
                "last_specialty": updated_job.last_specialty,
                "evidence": {
                    "wallet": updated_job.evidence.wallet,
                    "chain": updated_job.evidence.chain,
                    "tx_hashes": list(updated_job.evidence.tx_hashes),
                },
            },
        )

    def to_execution_parts(self) -> tuple[TicketAgentFlowOutcome, TicketInvestigationJob]:
        return (
            TicketAgentFlowOutcome(
                raw_final_reply=self.flow_outcome["raw_final_reply"],
                conversation_history=list(self.flow_outcome["conversation_history"]),
                completed_agent_key=self.flow_outcome.get("completed_agent_key"),
                requires_human_handoff=self.flow_outcome.get("requires_human_handoff", False),
            ),
            TicketInvestigationJob(
                channel_id=self.updated_job["channel_id"],
                requested_intent=self.updated_job.get("requested_intent"),
                mode=self.updated_job.get("mode", "idle"),
                current_specialty=self.updated_job.get("current_specialty"),
                last_specialty=self.updated_job.get("last_specialty"),
                evidence=InvestigationEvidence(
                    wallet=self.updated_job.get("evidence", {}).get("wallet"),
                    chain=self.updated_job.get("evidence", {}).get("chain"),
                    tx_hashes=list(self.updated_job.get("evidence", {}).get("tx_hashes", [])),
                ),
            ),
        )


def build_smoke_transport_result(
    request: TicketExecutionTransportRequest,
    *,
    endpoint_mode: str,
) -> TicketExecutionTransportResult:
    return TicketExecutionTransportResult(
        flow_outcome={
            "raw_final_reply": f"ticket_execution_smoke_ok:{endpoint_mode}",
            "conversation_history": [],
            "completed_agent_key": None,
            "requires_human_handoff": False,
        },
        updated_job=dict(request.investigation_job),
    )
