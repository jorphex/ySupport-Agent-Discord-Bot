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
ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
ACTIVE_DEPOSITS_HEADER_RE = re.compile(r"\*\*(?P<chain>[A-Za-z0-9 _/-]+) Active Deposits:\*\*")
CHAIN_NAMES = ("ethereum", "base", "arbitrum", "optimism", "polygon", "sonic", "katana")
SECURITY_PROCESS_URL = "https://docs.yearn.fi/developers/security"
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
    prior_specialist = investigation_job.current_specialty or investigation_job.last_specialty
    if (
        run_context.initial_button_intent is None
        and current_history
        and prior_specialist == "data"
        and _is_documented_governance_migration_followup(aggregated_text)
    ):
        logging.info(
            "Switching follow-up from prior specialist '%s' to docs for documented governance migration in channel %s",
            prior_specialist,
            investigation_job.channel_id,
        )
        return "docs"
    if (
        run_context.initial_button_intent is None
        and current_history
        and prior_specialist == "data"
        and (
            investigation_job.evidence.tx_hashes
            or (
                _is_withdrawal_followup(aggregated_text)
                and investigation_job.evidence.withdrawal_target_chain is not None
                and investigation_job.evidence.withdrawal_target_vault is not None
            )
        )
    ):
        logging.info(
            "Reusing prior specialist '%s' for structured data follow-up in channel %s",
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


def _looks_like_bug_bounty_intake_boundary_case(text: str) -> bool:
    lowered = (text or "").lower()
    reward_terms = (
        "reward",
        "rewarded",
        "bounty",
        "compensat",
        "paid for",
        "our efforts",
    )
    report_terms = (
        "issue",
        "bug",
        "security issue",
        "vulnerability",
        "exploit",
        "report",
        "disclos",
    )
    concrete_terms = (
        "contract",
        "strategy",
        "vault",
        "router",
        "component:",
        "impact:",
        "proof of concept",
        "poc",
        "foundry",
        "steps to reproduce",
        "reproduce",
        "vaultv3",
        "_redeem",
        "_update_debt",
        "_withdraw",
        "_unstake",
    )

    if not any(term in lowered for term in reward_terms):
        return False
    if not any(term in lowered for term in report_terms):
        return False
    if _contains_report_artifact_evidence(text):
        return False
    if TX_HASH_RE.search(text or "") or ADDRESS_RE.search(text or ""):
        return False
    if any(term in lowered for term in concrete_terms):
        return False
    return True


def _bug_bounty_intake_boundary_reply(text: str) -> str | None:
    if not _looks_like_bug_bounty_intake_boundary_case(text):
        return None
    return (
        "If you are reporting a Yearn security issue and want bounty or disclosure handling, "
        f"use Yearn's official security process at {SECURITY_PROCESS_URL}. "
        f"Human help is required beyond that path. {config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
    )


async def resolve_freeform_starting_agent(
    *,
    runner,
    input_list: str | List[TResponseInputItem],
    run_context: BotRunContext,
    workflow_name: str,
) -> str:
    """
    Public free-form routing should use the same structured router that ticket
    flow uses for docs/data/bug lane selection. If the router decides the turn
    can be handled directly, needs clarification, or needs human escalation,
    start with triage instead.
    """
    router_result: RunResult = await runner.run(
        starting_agent=ticket_triage_router_agent,
        input=input_list,
        max_turns=4,
        run_config=RunConfig(
            workflow_name=f"{workflow_name} / ticket-triage-router",
            group_id=str(run_context.channel_id),
        ),
        context=run_context,
    )
    decision = _normalize_ticket_triage_decision(
        router_result.final_output_as(TicketTriageDecision)
    )
    return ROUTE_ACTION_TO_AGENT_KEY.get(decision.action, "triage")


def _reply_requests_human_handoff(reply_text: str) -> bool:
    return config.HUMAN_HANDOFF_TAG_PLACEHOLDER in reply_text


def _contains_report_artifact_evidence(text: str) -> bool:
    if bool(
        re.search(
            r"https://(?:gist\.github\.com/[^/\s]+/[0-9a-fA-F]+|"
            r"gist\.githubusercontent\.com/[^/\s]+/[0-9a-fA-F]+/[^)\]\s]+|"
            r"raw\.githubusercontent\.com/[^)\]\s]+|"
            r"github\.com/[^/\s]+/[^/\s]+/(?:blob|raw)/[^)\]\s]+)",
            text or "",
        )
    ):
        return True

    # A pasted code fence is also a review artifact and deserves one bounded
    # repo/docs pre-triage pass before we allow an immediate handoff.
    return (text or "").count("```") >= 2


def _is_withdrawal_followup(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("withdraw", "withdrawing", "redeem", "redeeming"))


def _is_documented_governance_migration_followup(text: str) -> bool:
    lowered = text.lower()
    governance_markers = ("veyfi", "ve yfi", "styfi", "st yfi")
    destination_markers = ("migrate", "migration", "unlocked", "expired", "legacy", "manage")
    concrete_failure_markers = (
        "broken",
        "404",
        "button",
        "error code",
        "not loading",
        "doesn't load",
        "doesnt load",
        "page error",
        "transaction failed",
    )
    return (
        any(marker in lowered for marker in governance_markers)
        and any(marker in lowered for marker in destination_markers)
        and not any(marker in lowered for marker in concrete_failure_markers)
    )


def _extract_single_listed_deposit_context(
    reply_text: str,
) -> tuple[str, str] | None:
    if "Active Deposits:" not in reply_text:
        return None

    listed_deposits: list[tuple[str, str]] = []
    current_chain: str | None = None
    for line in reply_text.splitlines():
        header_match = ACTIVE_DEPOSITS_HEADER_RE.search(line)
        if header_match:
            current_chain = header_match.group("chain").strip().lower()
            continue
        if "Address:" not in line:
            continue
        address_match = ADDRESS_RE.search(line)
        if address_match and current_chain:
            listed_deposits.append((current_chain, address_match.group(0)))

    if len(listed_deposits) == 1:
        return listed_deposits[0]
    if listed_deposits:
        return None
    return None


def _merge_explicit_evidence_into_job(
    investigation_job: TicketInvestigationJob,
    text: str,
) -> None:
    TicketInvestigationRuntime.merge_explicit_evidence(investigation_job, text)


def _build_specialist_turn_input(request: TicketTurnRequest) -> List[TResponseInputItem]:
    input_list = list(request.input_list)
    extra_system_prompts: list[str] = []

    known_tx_hashes = request.investigation_job.evidence.tx_hashes
    if known_tx_hashes and not any(tx_hash in request.aggregated_text for tx_hash in known_tx_hashes):
        extra_system_prompts.append(
            "Continue the current transaction investigation yourself. "
            "Do not ask the user whether you should proceed or which internal investigation branch to take. "
            "Do not say 'I can', 'would you like', 'which would you like', or present an option menu. "
            "Choose the single next investigation step yourself, run it, and give the next grounded finding. "
            "Only escalate to a human if a real blocker remains after that step."
        )

    if _contains_report_artifact_evidence(request.aggregated_text):
        extra_system_prompts.append(
            "This turn includes a submitted report artifact or pasted PoC code. "
            "Do one bounded repo/docs pre-triage pass before escalating. "
            "If repo/docs clearly explain or contradict the claim, answer directly and do not escalate. "
            "If the artifact lacks a concrete Yearn-specific claim, ask one focused follow-up for the exact Yearn contract/path, concrete claim, and observed impact. "
            "Only escalate if a plausible unresolved security issue remains after that pre-triage."
        )

    if extra_system_prompts:
        input_list.append({"role": "system", "content": " ".join(extra_system_prompts)})
    return input_list


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
        current_history: Optional[List[TResponseInputItem]] = None,
    ) -> List[str]:
        hints: List[str] = []
        lowered_text = aggregated_text.lower()
        known_wallet = investigation_job.evidence.wallet
        known_chain = investigation_job.evidence.chain
        known_tx_hashes = investigation_job.evidence.tx_hashes
        combined_chain_and_tx_hint_added = False
        if (
            known_chain
            and known_tx_hashes
        ):
            recent_hashes = ", ".join(known_tx_hashes[-2:])
            hints.append(
                f"For onchain investigation on this ticket, use chain '{known_chain}' and transaction hash(es) {recent_hashes}. "
                "Do not substitute a different chain or transaction unless the user explicitly corrects you."
            )
            combined_chain_and_tx_hint_added = True
        elif known_chain and known_chain not in lowered_text:
            hints.append(
                f"Known chain for this ticket is {known_chain}. "
                "Use that chain for continued investigation unless the user explicitly corrects it."
            )
        if (
            known_tx_hashes
            and not combined_chain_and_tx_hint_added
            and not any(tx_hash in aggregated_text for tx_hash in known_tx_hashes)
        ):
            recent_hashes = ", ".join(known_tx_hashes[-2:])
            hints.append(
                f"Known transaction hashes for this ticket: {recent_hashes}. "
                "Reuse them for continued onchain investigation unless the user provides a different transaction."
            )
        if known_tx_hashes and not any(tx_hash in aggregated_text for tx_hash in known_tx_hashes):
            hints.append(
                "This is a follow-up to an existing transaction investigation. "
                "Continue the onchain investigation yourself and do not ask the user to choose between receipt decoding, log inspection, or contract calls."
            )
        wallet_hint_added = False
        if known_wallet and known_wallet not in aggregated_text:
            hints.append(
                f"Known wallet for this ticket is {known_wallet}. "
                "Use it if needed; do not ask again unless the user corrects it."
            )
            wallet_hint_added = True
        deposit_chain = investigation_job.evidence.withdrawal_target_chain
        deposit_vault = investigation_job.evidence.withdrawal_target_vault
        if deposit_chain and deposit_vault and _is_withdrawal_followup(aggregated_text):
            if known_wallet:
                if wallet_hint_added:
                    hints.pop()
                hints.append(
                    f"This withdrawal follow-up already has the needed details from earlier in the ticket: wallet {known_wallet}, "
                    f"vault {deposit_vault}, and chain {deposit_chain}. Use those exact values for withdrawal instructions unless the user explicitly corrects them. "
                    "Do not re-check deposits, do not ask which vault they mean, and do not default to ethereum."
                )
            else:
                hints.append(
                    f"The current ticket already has a single withdrawal target: {deposit_vault} on {deposit_chain}. "
                    "For this withdrawal follow-up, use that vault and chain unless the user explicitly corrects them. "
                    "Do not re-check deposits or default to ethereum if the ticket already identified the target vault."
                )
        return hints

    async def run_turn(self, request: TicketTurnRequest) -> TicketAgentFlowOutcome:
        # Safety boundary: do not let bounty-seeking security openings fall into
        # generic bug-intake prompts before the reporter reaches the official
        # security path.
        direct_boundary_reply = _bug_bounty_intake_boundary_reply(
            request.aggregated_text
        )
        if direct_boundary_reply is not None:
            return TicketAgentFlowOutcome(
                raw_final_reply=direct_boundary_reply,
                conversation_history=_append_ticket_reply_to_history(
                    request.current_history,
                    request.aggregated_text,
                    direct_boundary_reply,
                ),
                completed_agent_key=None,
                requires_human_handoff=True,
            )

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

        specialist_input = _build_specialist_turn_input(request)
        result: RunResult = await self.runner.run(
            starting_agent=_resolve_agent(agent_key),
            input=specialist_input,
            max_turns=config.MAX_TICKET_CONVERSATION_TURNS,
            run_config=RunConfig(
                workflow_name=request.workflow_name,
                group_id=str(request.run_context.channel_id),
            ),
            context=request.run_context,
        )
        raw_final_reply = result.final_output or "I'm not sure how to respond to that."
        conversation_history = result.to_input_list()
        completed_agent_key = _agent_to_key(result.last_agent)
        if completed_agent_key == "data":
            single_deposit_context = _extract_single_listed_deposit_context(raw_final_reply)
            if single_deposit_context is not None:
                deposit_chain, deposit_vault = single_deposit_context
                request.investigation_job.remember_withdrawal_target(
                    deposit_chain,
                    deposit_vault,
                )
            else:
                request.investigation_job.clear_withdrawal_target()
        return TicketAgentFlowOutcome(
            raw_final_reply=raw_final_reply,
            conversation_history=conversation_history,
            completed_agent_key=completed_agent_key,
            requires_human_handoff=_reply_requests_human_handoff(raw_final_reply),
        )
