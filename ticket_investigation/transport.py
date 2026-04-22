from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any

try:
    from agents import TResponseInputItem
except ModuleNotFoundError:
    TResponseInputItem = dict[str, Any]

from state import BotRunContext, InvestigationEvidence, TicketInvestigationJob

if TYPE_CHECKING:
    from ticket_investigation.runtime import TicketAgentFlowOutcome, TicketTurnRequest


TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "aggregated_text",
        "input_list",
        "current_history",
        "attachments",
        "run_context",
        "investigation_job",
        "workflow_name",
        "wants_bug_review_status",
    ],
    "properties": {
        "aggregated_text": {"type": "string"},
        "input_list": {"type": "array"},
        "current_history": {"type": "array"},
        "attachments": {"type": "array"},
        "run_context": {"type": "object"},
        "investigation_job": {"type": "object"},
        "workflow_name": {"type": "string"},
        "wants_bug_review_status": {"type": "boolean"},
        "precomputed_boundary": {"type": ["object", "null"]},
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


def serialize_run_context(run_context: BotRunContext) -> dict[str, Any]:
    return {
        "channel_id": run_context.channel_id,
        "category_id": run_context.category_id,
        "is_public_trigger": run_context.is_public_trigger,
        "conversation_owner_id": run_context.conversation_owner_id,
        "project_context": run_context.project_context,
        "initial_button_intent": run_context.initial_button_intent,
        "repo_search_calls": run_context.repo_search_calls,
        "repo_fetch_calls": run_context.repo_fetch_calls,
        "repo_searches_without_fetch": run_context.repo_searches_without_fetch,
        "repo_last_search_query": run_context.repo_last_search_query,
        "repo_last_search_artifact_refs": list(run_context.repo_last_search_artifact_refs),
    }


def deserialize_run_context(payload: dict[str, Any]) -> BotRunContext:
    return BotRunContext(
        channel_id=payload["channel_id"],
        category_id=payload.get("category_id"),
        is_public_trigger=payload.get("is_public_trigger", False),
        conversation_owner_id=payload.get("conversation_owner_id"),
        project_context=payload.get("project_context", "unknown"),
        initial_button_intent=payload.get("initial_button_intent"),
        repo_search_calls=payload.get("repo_search_calls", 0),
        repo_fetch_calls=payload.get("repo_fetch_calls", 0),
        repo_searches_without_fetch=payload.get("repo_searches_without_fetch", 0),
        repo_last_search_query=payload.get("repo_last_search_query"),
        repo_last_search_artifact_refs=list(payload.get("repo_last_search_artifact_refs", [])),
    )


def serialize_investigation_job(
    investigation_job: TicketInvestigationJob,
) -> dict[str, Any]:
    return {
        "channel_id": investigation_job.channel_id,
        "requested_intent": investigation_job.requested_intent,
        "mode": investigation_job.mode,
        "current_specialty": investigation_job.current_specialty,
        "last_specialty": investigation_job.last_specialty,
        "evidence": serialize_investigation_evidence(investigation_job.evidence),
    }


def deserialize_investigation_job(payload: dict[str, Any]) -> TicketInvestigationJob:
    return TicketInvestigationJob(
        channel_id=payload["channel_id"],
        requested_intent=payload.get("requested_intent"),
        mode=payload.get("mode", "idle"),
        current_specialty=payload.get("current_specialty"),
        last_specialty=payload.get("last_specialty"),
        evidence=deserialize_investigation_evidence(payload.get("evidence", {})),
    )


def serialize_investigation_evidence(
    evidence: InvestigationEvidence,
) -> dict[str, Any]:
    return {
        "wallet": evidence.wallet,
        "chain": evidence.chain,
        "tx_hashes": list(evidence.tx_hashes),
        "withdrawal_target_chain": evidence.withdrawal_target_chain,
        "withdrawal_target_vault": evidence.withdrawal_target_vault,
    }


def deserialize_investigation_evidence(
    payload: dict[str, Any],
) -> InvestigationEvidence:
    return InvestigationEvidence(
        wallet=payload.get("wallet"),
        chain=payload.get("chain"),
        tx_hashes=list(payload.get("tx_hashes", [])),
        withdrawal_target_chain=payload.get("withdrawal_target_chain"),
        withdrawal_target_vault=payload.get("withdrawal_target_vault"),
    )


def serialize_flow_outcome(flow_outcome: TicketAgentFlowOutcome) -> dict[str, Any]:
    return {
        "raw_final_reply": flow_outcome.raw_final_reply,
        "conversation_history": list(flow_outcome.conversation_history),
        "completed_agent_key": flow_outcome.completed_agent_key,
        "requires_human_handoff": flow_outcome.requires_human_handoff,
    }


def deserialize_flow_outcome(payload: dict[str, Any]) -> TicketAgentFlowOutcome:
    from ticket_investigation.runtime import TicketAgentFlowOutcome

    return TicketAgentFlowOutcome(
        raw_final_reply=payload["raw_final_reply"],
        conversation_history=list(payload["conversation_history"]),
        completed_agent_key=payload.get("completed_agent_key"),
        requires_human_handoff=payload.get("requires_human_handoff", False),
    )


def build_smoke_transport_request() -> "TicketExecutionTransportRequest":
    return TicketExecutionTransportRequest(
        aggregated_text="ticket execution smoke probe",
        input_list=[],
        current_history=[],
        attachments=[],
        run_context={
            "channel_id": 0,
            "category_id": None,
            "is_public_trigger": False,
            "conversation_owner_id": None,
            "project_context": "yearn",
            "initial_button_intent": None,
            "repo_search_calls": 0,
            "repo_fetch_calls": 0,
            "repo_searches_without_fetch": 0,
            "repo_last_search_query": None,
            "repo_last_search_artifact_refs": [],
        },
        investigation_job={
            "channel_id": 0,
            "requested_intent": "smoke_probe",
            "mode": "idle",
            "current_specialty": None,
            "last_specialty": None,
            "evidence": {
                "wallet": None,
                "chain": None,
                "tx_hashes": [],
                "withdrawal_target_chain": None,
                "withdrawal_target_vault": None,
            },
        },
        workflow_name="ticket_execution_status.smoke_probe",
        wants_bug_review_status=False,
        precomputed_boundary=None,
        smoke_mode="ping",
    )


@dataclass
class TicketExecutionTransportRequest:
    aggregated_text: str
    input_list: list[TResponseInputItem]
    current_history: list[TResponseInputItem]
    run_context: dict[str, Any]
    investigation_job: dict[str, Any]
    workflow_name: str
    wants_bug_review_status: bool = False
    precomputed_boundary: dict[str, Any] | None = None
    smoke_mode: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "aggregated_text": self.aggregated_text,
            "input_list": list(self.input_list),
            "current_history": list(self.current_history),
            "attachments": list(self.attachments),
            "run_context": dict(self.run_context),
            "investigation_job": dict(self.investigation_job),
            "workflow_name": self.workflow_name,
            "wants_bug_review_status": self.wants_bug_review_status,
            "precomputed_boundary": (
                dict(self.precomputed_boundary)
                if self.precomputed_boundary is not None
                else None
            ),
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
            attachments=list(payload.get("attachments", [])),
            run_context=dict(payload.get("run_context", {})),
            investigation_job=dict(payload.get("investigation_job", {})),
            workflow_name=payload["workflow_name"],
            wants_bug_review_status=payload.get("wants_bug_review_status", False),
            precomputed_boundary=(
                dict(payload["precomputed_boundary"])
                if payload.get("precomputed_boundary") is not None
                else None
            ),
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
            attachments=list(request.attachments),
            run_context=serialize_run_context(request.run_context),
            investigation_job=serialize_investigation_job(request.investigation_job),
            workflow_name=request.workflow_name,
            wants_bug_review_status=wants_bug_review_status,
            precomputed_boundary=(
                dict(request.precomputed_boundary)
                if request.precomputed_boundary is not None
                else None
            ),
            smoke_mode=None,
        )

    def to_turn_request(self) -> TicketTurnRequest:
        from ticket_investigation.runtime import TicketTurnRequest

        return TicketTurnRequest(
            aggregated_text=self.aggregated_text,
            input_list=list(self.input_list),
            current_history=list(self.current_history),
            attachments=list(self.attachments),
            run_context=deserialize_run_context(self.run_context),
            investigation_job=deserialize_investigation_job(self.investigation_job),
            workflow_name=self.workflow_name,
            precomputed_boundary=(
                dict(self.precomputed_boundary)
                if self.precomputed_boundary is not None
                else None
            ),
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
            flow_outcome=serialize_flow_outcome(flow_outcome),
            updated_job=serialize_investigation_job(updated_job),
        )

    def to_execution_parts(self) -> tuple[TicketAgentFlowOutcome, TicketInvestigationJob]:
        return (
            deserialize_flow_outcome(self.flow_outcome),
            deserialize_investigation_job(self.updated_job),
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
