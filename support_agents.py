import logging
from typing import Any, List, Optional, Union, Literal

from pydantic import BaseModel, Field

import config
from agents import (
    Agent, Runner, RunContextWrapper,
    ModelSettings, handoff,
    input_guardrail, GuardrailFunctionOutput,
    RunConfig,
    TResponseInputItem,
)
from agents.model_settings import Reasoning

from agent_prompts import (
    YEARn_DATA_AGENT_INSTRUCTIONS,
    YEARn_DOCS_QA_AGENT_INSTRUCTIONS,
    YEARn_BUG_TRIAGE_AGENT_INSTRUCTIONS,
    BD_PRIORITY_GUARDRAIL_INSTRUCTIONS,
    TRIAGE_AGENT_INSTRUCTIONS,
    TICKET_TRIAGE_ROUTER_INSTRUCTIONS,
)
from bot_behavior import (
    LISTING_DENIAL_MESSAGE,
    STANDARD_REDIRECT_MESSAGE,
    SECURITY_VENDOR_BOUNDARY_MESSAGE,
    JOB_INQUIRY_REDIRECT_MESSAGE,
)
from state import BotRunContext
from support_tools import (
    search_vaults_tool,
    check_all_deposits_tool,
    get_withdrawal_instructions_tool,
    inspect_onchain_tool,
    answer_from_docs_tool,
    pretriage_repo_claim_tool,
    search_repo_context_tool,
    fetch_repo_artifacts_tool,
    fetch_report_artifact_tool,
    repo_context_status_tool,
)


class BDPriorityCheckOutput(BaseModel):
    request_type: Literal["listing", "partnership", "marketing", "vendor_security", "other_bd", "job_inquiry", "not_bd_pr"] = Field(..., description="Classify the user's primary intent: 'listing' (requesting Yearn list their token), 'partnership' (proposing integration/collaboration), 'marketing' (joint marketing/promotion), 'vendor_security' (security or phishing vendor/service outreach), 'other_bd' (other business development), 'job_inquiry' (asking to work for/contribute to Yearn, grant requests), or 'not_bd_pr' (standard support request or unrelated).")
    reasoning: str = Field(..., description="Brief explanation for the classification.")


class TicketTriageDecision(BaseModel):
    action: Literal[
        "route_data",
        "route_docs",
        "route_bug",
        "ask_clarifying",
        "respond_directly",
        "human_escalation",
    ] = Field(..., description="The next runtime action for this ticket turn.")
    message: Optional[str] = Field(
        default=None,
        description="User-facing message for ask_clarifying, respond_directly, or human_escalation. Leave empty for route_* actions.",
    )
    reasoning: str = Field(..., description="Brief explanation for the routing decision.")


def _looks_like_vendor_security_outreach(text: str) -> bool:
    lowered = (text or "").lower()
    vendor_terms = (
        "threat intelligence",
        "analysts",
        "vendor",
        "service",
        "services",
        "monitoring",
        "scan",
        "scans",
        "broader scan",
        "trial",
        "free trial",
        "takedown",
        "takedowns",
        "brand protection",
        "impersonation",
        "fake domains",
        "malicious domains",
        "phishing domains",
        "free of charge",
    )
    return any(term in lowered for term in vendor_terms)


def _looks_like_concrete_security_disclosure(text: str) -> bool:
    lowered = (text or "").lower()
    disclosure_terms = (
        "vulnerability",
        "security issue",
        "security report",
        "bug report",
        "exploit",
        "responsibly disclose",
        "responsible disclosure",
        "proof of concept",
        "poc",
        "full technical report",
        "technical report",
    )
    component_terms = (
        "contract",
        "strategy",
        "vault",
        "router",
        "styfi",
        "veyfi",
        "_withdraw",
        "_unstake",
        "component:",
        "issue:",
    )
    impact_terms = (
        "impact:",
        "severity",
        "loss of funds",
        "fund loss",
        "freezing of user funds",
        "freeze of user funds",
        "permanent user fund loss",
        "permanent freezing",
        "permanent loss",
    )
    submission_blocker_terms = (
        "immunefi",
        "zkpassport",
        "identity verification",
        "secure way to submit",
        "preferred secure way",
        "submit the full report",
    )

    has_disclosure_term = any(term in lowered for term in disclosure_terms)
    has_component_term = any(term in lowered for term in component_terms)
    has_impact_term = any(term in lowered for term in impact_terms)
    has_submission_blocker = any(term in lowered for term in submission_blocker_terms)

    return has_disclosure_term and (
        has_submission_blocker or (has_component_term and has_impact_term)
    )


def _looks_like_bd_priority_candidate(text: str) -> bool:
    lowered = (text or "").lower()
    candidate_terms = (
        "list ",
        "listing",
        "exchange",
        "partnership",
        "partner ",
        "collab",
        "collaborat",
        "marketing",
        "business development",
        "proposal",
        "integration opportunity",
        "vendor",
        "phishing",
        "takedown",
        "brand protection",
        "security service",
        "security services",
        "threat intelligence",
        "trial",
        "responsibly disclose",
        "responsible disclosure",
        "security issue",
        "vulnerability",
        "technical report",
        "immunefi",
        "security team",
        "job",
        "hiring",
        "work for yearn",
        "work with yearn",
        "grant",
        "bounty",
    )
    return any(term in lowered for term in candidate_terms)


def _gpt5_model_settings(
    *,
    effort: str,
    verbosity: Literal["low", "medium", "high"],
) -> ModelSettings:
    return ModelSettings(
        reasoning=Reasoning(effort=effort),
        verbosity=verbosity,
    )


def _with_runtime_context(base_instructions: str):
    def _instructions(
        ctx: RunContextWrapper[BotRunContext],
        agent: Agent[BotRunContext],
    ) -> str:
        run_context = ctx.context
        runtime_context = (
            "# Runtime Context\n"
            f"- project_context: {run_context.project_context}\n"
            f"- initial_button_intent: {run_context.initial_button_intent or 'none'}\n"
            f"- is_public_trigger: {str(run_context.is_public_trigger).lower()}\n"
        )
        return f"{base_instructions}\n\n{runtime_context}"

    return _instructions


bd_priority_guardrail_agent = Agent[BotRunContext](
    name="BD/PR/Listing Guardrail Check",
    instructions=BD_PRIORITY_GUARDRAIL_INSTRUCTIONS,
    output_type=BDPriorityCheckOutput,
    model=config.LLM_GUARDRAIL_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_GUARDRAIL_REASONING_EFFORT,
        verbosity=config.LLM_GUARDRAIL_VERBOSITY,
    ),
)


@input_guardrail(name="BD/PR/Listing/Job Guardrail")
async def bd_priority_guardrail(
    ctx: RunContextWrapper[BotRunContext],
    agent: Agent,
    input_data: Union[str, List[TResponseInputItem]]
) -> GuardrailFunctionOutput:
    """
    Checks if the initial user input is BD/PR/Listing.
    If it is, returns GuardrailFunctionOutput with tripwire_triggered=True
    and the appropriate response message stored in output_info.
    Otherwise, returns with tripwire_triggered=False.
    """
    if isinstance(input_data, str):
        text_input = input_data
    elif isinstance(input_data, list):
        text_input = ""
        for item in reversed(input_data):
             if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
                 text_input = item["content"]
                 break
    else:
        text_input = ""
    if not text_input:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    output_info = await evaluate_bd_priority_boundary(text_input)
    if not output_info:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=bool(output_info.get("tripwire_triggered")),
    )


async def evaluate_bd_priority_boundary(text_input: str) -> dict[str, Any]:
    if not text_input.strip():
        return {"tripwire_triggered": False}
    if not _looks_like_bd_priority_candidate(text_input):
        return {"tripwire_triggered": False}
    logging.info(f"[Guardrail:BD/Priority] Analyzing input: '{text_input[:100]}...'")

    try:
        guardrail_runner = Runner()
        run_config_guardrail = RunConfig(
            workflow_name="BD/Priority Guardrail Check",
            tracing_disabled=True,
        )
        result = await guardrail_runner.run(
            starting_agent=bd_priority_guardrail_agent,
            input=text_input,
            run_config=run_config_guardrail,
        )
        check_output = result.final_output_as(BDPriorityCheckOutput)
        effective_request_type = check_output.request_type

        if (
            effective_request_type == "vendor_security"
            and _looks_like_concrete_security_disclosure(text_input)
            and not _looks_like_vendor_security_outreach(text_input)
        ):
            logging.info(
                "[Guardrail:BD/Priority] Overriding vendor_security to not_bd_pr for concrete security disclosure."
            )
            effective_request_type = "not_bd_pr"

        logging.info(
            "[Guardrail:BD/Priority] Check result: raw_type=%s effective_type=%s Reasoning: %s",
            check_output.request_type,
            effective_request_type,
            check_output.reasoning,
        )

        message_to_send = None
        if effective_request_type == "listing":
            message_to_send = LISTING_DENIAL_MESSAGE
        elif effective_request_type == "vendor_security":
            message_to_send = SECURITY_VENDOR_BOUNDARY_MESSAGE
        elif effective_request_type in ["partnership", "marketing", "other_bd"]:
            message_to_send = STANDARD_REDIRECT_MESSAGE
        elif effective_request_type == "job_inquiry":
            message_to_send = JOB_INQUIRY_REDIRECT_MESSAGE

        output_info: dict[str, Any] = {
            "classification": check_output.model_dump(),
            "effective_request_type": effective_request_type,
            "tripwire_triggered": bool(message_to_send),
        }
        if message_to_send:
            output_info["message"] = message_to_send
        return output_info
    except Exception as e:
        logging.error(f"[Guardrail:BD/Priority] Error during check: {e}", exc_info=True)
        return {"error": str(e), "tripwire_triggered": False}


yearn_data_agent = Agent[BotRunContext](
    name="Yearn Data Specialist",
    instructions=_with_runtime_context(YEARn_DATA_AGENT_INSTRUCTIONS),
    tools=[
        search_vaults_tool,
        check_all_deposits_tool,
        get_withdrawal_instructions_tool,
        inspect_onchain_tool,
    ],
    model=config.LLM_DATA_AGENT_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_DATA_AGENT_REASONING_EFFORT,
        verbosity=config.LLM_DATA_AGENT_VERBOSITY,
    ),
)

yearn_docs_qa_agent = Agent[BotRunContext](
    name="Yearn Docs QA Specialist",
    instructions=_with_runtime_context(YEARn_DOCS_QA_AGENT_INSTRUCTIONS),
    tools=[
        answer_from_docs_tool,
        pretriage_repo_claim_tool,
        search_repo_context_tool,
        fetch_repo_artifacts_tool,
        repo_context_status_tool,
    ],
    model=config.LLM_DOCS_AGENT_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_DOCS_AGENT_REASONING_EFFORT,
        verbosity=config.LLM_DOCS_AGENT_VERBOSITY,
    ),
)

yearn_bug_triage_agent = Agent[BotRunContext](
    name="Yearn Bug Triage Specialist",
    instructions=_with_runtime_context(YEARn_BUG_TRIAGE_AGENT_INSTRUCTIONS),
    tools=[
        pretriage_repo_claim_tool,
        search_repo_context_tool,
        fetch_repo_artifacts_tool,
        fetch_report_artifact_tool,
        inspect_onchain_tool,
        answer_from_docs_tool,
        repo_context_status_tool,
    ],
    model=config.LLM_BUG_AGENT_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_BUG_AGENT_REASONING_EFFORT,
        verbosity=config.LLM_BUG_AGENT_VERBOSITY,
    ),
)

triage_agent = Agent[BotRunContext](
    name="Support Triage Agent",
    instructions=_with_runtime_context(TRIAGE_AGENT_INSTRUCTIONS),
    handoffs=[
        handoff(yearn_data_agent, tool_name_override="transfer_to_yearn_data_specialist", tool_description_override="Handoff for specific YEARN data (vaults, deposits, APR, TVL, balances, withdrawal instructions)."),
        handoff(yearn_docs_qa_agent, tool_name_override="transfer_to_yearn_docs_qa_specialist", tool_description_override="Handoff for general questions about YEARN concepts, documentation, risks."),
        handoff(yearn_bug_triage_agent, tool_name_override="transfer_to_yearn_bug_triage_specialist", tool_description_override="Handoff for YEARN bug reports, UI issues, migration issues, and protocol behavior claims that should be checked against docs and repo context before human escalation."),
    ],
    input_guardrails=[bd_priority_guardrail],
    model=config.LLM_TRIAGE_AGENT_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_TRIAGE_AGENT_REASONING_EFFORT,
        verbosity=config.LLM_TRIAGE_AGENT_VERBOSITY,
    ),
)

ticket_triage_router_agent = Agent[BotRunContext](
    name="Ticket Triage Router",
    instructions=_with_runtime_context(TICKET_TRIAGE_ROUTER_INSTRUCTIONS),
    input_guardrails=[bd_priority_guardrail],
    output_type=TicketTriageDecision,
    model=config.LLM_TRIAGE_AGENT_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_TRIAGE_AGENT_REASONING_EFFORT,
        verbosity=config.LLM_TRIAGE_AGENT_VERBOSITY,
    ),
)
