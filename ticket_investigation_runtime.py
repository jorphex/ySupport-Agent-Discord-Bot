import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from agents import RunConfig, RunResult, TResponseInputItem

import config
from router import select_starting_agent
from state import BotRunContext, TicketInvestigationJob
from support_agents import (
    TicketTriageDecision,
    ticket_triage_router_agent,
    triage_agent,
    yearn_bug_triage_agent,
    yearn_data_agent,
    yearn_docs_qa_agent,
)


TX_HASH_RE = re.compile(r"\b0x[a-fA-F0-9]{64}\b")
CHAIN_NAMES = ("ethereum", "base", "arbitrum", "optimism", "polygon", "sonic", "katana")
ROUTE_ACTION_TO_AGENT_KEY = {
    "route_data": "data",
    "route_docs": "docs",
    "route_bug": "bug",
}


@dataclass
class TicketAgentFlowOutcome:
    raw_final_reply: str
    conversation_history: List[TResponseInputItem]
    completed_agent_key: Optional[str]
    requires_human_handoff: bool


@dataclass
class TicketTurnRequest:
    aggregated_text: str
    input_list: List[TResponseInputItem]
    current_history: List[TResponseInputItem]
    run_context: BotRunContext
    investigation_job: TicketInvestigationJob
    workflow_name: str
    send_bug_review_status: Optional[Callable[[], Awaitable[None]]] = None


def _resolve_agent(agent_key: str):
    if agent_key == "data":
        return yearn_data_agent
    if agent_key == "docs":
        return yearn_docs_qa_agent
    if agent_key == "bug":
        return yearn_bug_triage_agent
    return triage_agent


def _select_ticket_starting_agent(
    aggregated_text: str,
    run_context: BotRunContext,
    current_history: List[TResponseInputItem],
    investigation_job: TicketInvestigationJob,
) -> str:
    prior_specialist = investigation_job.last_specialty
    if (
        run_context.initial_button_intent is None
        and current_history
        and prior_specialist in {"data", "docs", "bug"}
    ):
        logging.info(
            "Reusing prior specialist '%s' for follow-up in channel %s",
            prior_specialist,
            investigation_job.channel_id,
        )
        return prior_specialist
    return select_starting_agent(aggregated_text, run_context)


def _agent_to_key(agent) -> str | None:
    if agent is yearn_data_agent:
        return "data"
    if agent is yearn_docs_qa_agent:
        return "docs"
    if agent is yearn_bug_triage_agent:
        return "bug"
    if agent is triage_agent:
        return "triage"
    return None


def _append_ticket_reply_to_history(
    current_history: List[TResponseInputItem],
    aggregated_text: str,
    reply_text: str,
) -> List[TResponseInputItem]:
    return current_history + [
        {"role": "user", "content": aggregated_text},
        {"role": "assistant", "content": reply_text},
    ]


def _normalize_ticket_triage_decision(
    decision: TicketTriageDecision,
) -> TicketTriageDecision:
    if decision.action in ROUTE_ACTION_TO_AGENT_KEY:
        return TicketTriageDecision(
            action=decision.action,
            message=None,
            reasoning=decision.reasoning,
        )

    message = (decision.message or "").strip()
    if not message:
        if decision.action == "ask_clarifying":
            message = "Can you clarify what you need help with?"
        elif decision.action == "respond_directly":
            message = "Can you share a bit more detail about what you need?"
        else:
            message = (
                "This needs human review. "
                f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
            )

    if (
        decision.action == "human_escalation"
        and config.HUMAN_HANDOFF_TAG_PLACEHOLDER not in message
    ):
        message = f"{message} {config.HUMAN_HANDOFF_TAG_PLACEHOLDER}".strip()

    return TicketTriageDecision(
        action=decision.action,
        message=message,
        reasoning=decision.reasoning,
    )


def _reply_requests_human_handoff(reply_text: str) -> bool:
    return config.HUMAN_HANDOFF_TAG_PLACEHOLDER in reply_text


def _merge_explicit_evidence_into_job(
    investigation_job: TicketInvestigationJob,
    text: str,
) -> None:
    TicketInvestigationRuntime.merge_explicit_evidence(investigation_job, text)


class TicketInvestigationRuntime:
    def __init__(self, runner) -> None:
        self.runner = runner

    @staticmethod
    def contains_tx_hash(text: str) -> bool:
        return bool(TX_HASH_RE.search(text))

    @staticmethod
    def merge_explicit_evidence(investigation_job: TicketInvestigationJob, text: str) -> None:
        if not text:
            return

        tx_hashes = TX_HASH_RE.findall(text)
        for tx_hash in tx_hashes:
            investigation_job.remember_tx_hash(tx_hash)

        lowered = text.lower()
        for chain_name in CHAIN_NAMES:
            if chain_name in lowered:
                investigation_job.remember_chain(chain_name)
                break

    @staticmethod
    def build_contextual_hints(
        investigation_job: TicketInvestigationJob,
        aggregated_text: str,
    ) -> List[str]:
        hints: List[str] = []
        if (
            investigation_job.evidence.chain
            and TicketInvestigationRuntime.contains_tx_hash(aggregated_text)
        ):
            hints.append(
                f"Known chain for this ticket is {investigation_job.evidence.chain}. "
                "Use that chain for tx investigation unless the user explicitly corrects it."
            )
        if (
            investigation_job.evidence.wallet
            and investigation_job.evidence.wallet not in aggregated_text
        ):
            hints.append(
                f"Known wallet for this ticket is {investigation_job.evidence.wallet}. "
                "Use it if needed; do not ask again unless the user corrects it."
            )
        return hints

    async def run_turn(self, request: TicketTurnRequest) -> TicketAgentFlowOutcome:
        agent_key = _select_ticket_starting_agent(
            request.aggregated_text,
            request.run_context,
            request.current_history,
            request.investigation_job,
        )
        logging.info(
            "Selected starting agent '%s' for channel %s (intent=%s)",
            agent_key,
            request.investigation_job.channel_id,
            request.run_context.initial_button_intent,
        )

        if agent_key == "triage":
            router_result: RunResult = await self.runner.run(
                starting_agent=ticket_triage_router_agent,
                input=request.input_list,
                max_turns=4,
                run_config=RunConfig(
                    workflow_name=f"{request.workflow_name} / ticket-triage-router",
                    group_id=str(request.run_context.channel_id),
                ),
                context=request.run_context,
            )
            decision = _normalize_ticket_triage_decision(
                router_result.final_output_as(TicketTriageDecision)
            )
            logging.info(
                "Ticket triage router decision for channel %s: action=%s reasoning=%s",
                request.investigation_job.channel_id,
                decision.action,
                decision.reasoning,
            )

            routed_agent_key = ROUTE_ACTION_TO_AGENT_KEY.get(decision.action)
            if routed_agent_key is None:
                return TicketAgentFlowOutcome(
                    raw_final_reply=decision.message or "I need a bit more detail to help with this.",
                    conversation_history=_append_ticket_reply_to_history(
                        request.current_history,
                        request.aggregated_text,
                        decision.message or "I need a bit more detail to help with this.",
                    ),
                    completed_agent_key=None,
                    requires_human_handoff=decision.action == "human_escalation",
                )

            agent_key = routed_agent_key

        if agent_key == "bug" and request.send_bug_review_status is not None:
            await request.send_bug_review_status()

        result: RunResult = await self.runner.run(
            starting_agent=_resolve_agent(agent_key),
            input=request.input_list,
            max_turns=config.MAX_TICKET_CONVERSATION_TURNS,
            run_config=RunConfig(
                workflow_name=request.workflow_name,
                group_id=str(request.run_context.channel_id),
            ),
            context=request.run_context,
        )
        raw_final_reply = result.final_output or "I'm not sure how to respond to that."
        return TicketAgentFlowOutcome(
            raw_final_reply=raw_final_reply,
            conversation_history=result.to_input_list(),
            completed_agent_key=_agent_to_key(result.last_agent),
            requires_human_handoff=_reply_requests_human_handoff(raw_final_reply),
        )
