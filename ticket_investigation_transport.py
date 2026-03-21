from dataclasses import dataclass
from typing import Any

from agents import TResponseInputItem

from state import BotRunContext, InvestigationEvidence, TicketInvestigationJob
from ticket_investigation_executor import TicketExecutionResult
from ticket_investigation_runtime import TicketAgentFlowOutcome, TicketTurnRequest


@dataclass
class TicketExecutionTransportRequest:
    aggregated_text: str
    input_list: list[TResponseInputItem]
    current_history: list[TResponseInputItem]
    run_context: dict[str, Any]
    investigation_job: dict[str, Any]
    workflow_name: str
    wants_bug_review_status: bool = False

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

    @classmethod
    def from_execution_result(
        cls,
        result: TicketExecutionResult,
    ) -> "TicketExecutionTransportResult":
        return cls(
            flow_outcome={
                "raw_final_reply": result.flow_outcome.raw_final_reply,
                "conversation_history": list(result.flow_outcome.conversation_history),
                "completed_agent_key": result.flow_outcome.completed_agent_key,
                "requires_human_handoff": result.flow_outcome.requires_human_handoff,
            },
            updated_job={
                "channel_id": result.updated_job.channel_id,
                "requested_intent": result.updated_job.requested_intent,
                "mode": result.updated_job.mode,
                "current_specialty": result.updated_job.current_specialty,
                "last_specialty": result.updated_job.last_specialty,
                "evidence": {
                    "wallet": result.updated_job.evidence.wallet,
                    "chain": result.updated_job.evidence.chain,
                    "tx_hashes": list(result.updated_job.evidence.tx_hashes),
                },
            },
        )

    def to_execution_result(self) -> TicketExecutionResult:
        return TicketExecutionResult(
            flow_outcome=TicketAgentFlowOutcome(
                raw_final_reply=self.flow_outcome["raw_final_reply"],
                conversation_history=list(self.flow_outcome["conversation_history"]),
                completed_agent_key=self.flow_outcome.get("completed_agent_key"),
                requires_human_handoff=self.flow_outcome.get("requires_human_handoff", False),
            ),
            updated_job=TicketInvestigationJob(
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
