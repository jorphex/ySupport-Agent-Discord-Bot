import logging
import re
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
    SUPPORT_BOUNDARY_GUARDRAIL_INSTRUCTIONS,
    TRIAGE_AGENT_INSTRUCTIONS,
    TICKET_TRIAGE_ROUTER_INSTRUCTIONS,
)
from bot_behavior import (
    OUT_OF_SCOPE_SUPPORT_MESSAGE,
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


_SUPPORT_SCOPE_TX_HASH_RE = re.compile(r"(?:[a-z]+:)?0x[a-fA-F0-9]{64}")
_SUPPORT_SCOPE_ADDRESS_RE = re.compile(r"(?:[a-z]+:)?0x[a-fA-F0-9]{40}")
_SECURITY_PROCESS_URL = "https://docs.yearn.fi/developers/security"


class SupportBoundaryCheckOutput(BaseModel):
    classification: Literal[
        "yearn_support",
        "business_boundary",
        "security_process_boundary",
        "non_support_assistant",
        "uncertain",
    ] = Field(..., description="Top-level outer boundary classification for the user message.")
    business_subtype: Optional[
        Literal["listing", "general_bd", "vendor_security", "job_inquiry"]
    ] = Field(
        default=None,
        description="Subtype only when classification is business_boundary.",
    )
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


def _looks_like_support_scope_primitive(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(
        _SUPPORT_SCOPE_TX_HASH_RE.fullmatch(stripped)
        or _SUPPORT_SCOPE_ADDRESS_RE.fullmatch(stripped)
    )


def _security_process_boundary_message() -> str:
    return (
        "If you are reporting a Yearn security issue and want bounty or disclosure handling, "
        f"use Yearn's official security process at {_SECURITY_PROCESS_URL}. "
        "Human help is required beyond that path."
    )


def _message_for_support_boundary(
    classification: str,
    business_subtype: str | None,
) -> str | None:
    if classification == "business_boundary":
        if business_subtype == "listing":
            return LISTING_DENIAL_MESSAGE
        if business_subtype == "vendor_security":
            return SECURITY_VENDOR_BOUNDARY_MESSAGE
        if business_subtype == "job_inquiry":
            return JOB_INQUIRY_REDIRECT_MESSAGE
        return STANDARD_REDIRECT_MESSAGE
    if classification == "security_process_boundary":
        return _security_process_boundary_message()
    if classification == "non_support_assistant":
        return OUT_OF_SCOPE_SUPPORT_MESSAGE
    return None


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


support_boundary_guardrail_agent = Agent[BotRunContext](
    name="Support Boundary Guardrail Check",
    instructions=SUPPORT_BOUNDARY_GUARDRAIL_INSTRUCTIONS,
    output_type=SupportBoundaryCheckOutput,
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

    output_info = await evaluate_support_boundary(text_input)
    if not output_info:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=bool(output_info.get("tripwire_triggered")),
    )


async def evaluate_support_boundary(text_input: str) -> dict[str, Any]:
    if not text_input.strip():
        return {
            "classification": "yearn_support",
            "business_subtype": None,
            "tripwire_triggered": False,
        }
    if _looks_like_support_scope_primitive(text_input):
        return {
            "classification": "yearn_support",
            "business_subtype": None,
            "reasoning": "Explicit support primitive such as a bare address or tx hash.",
            "tripwire_triggered": False,
        }
    logging.info(f"[Guardrail:Boundary] Analyzing input: '{text_input[:100]}...'")

    try:
        guardrail_runner = Runner()
        run_config_guardrail = RunConfig(
            workflow_name="Yearn Support Boundary Guardrail Check",
            tracing_disabled=True,
        )
        result = await guardrail_runner.run(
            starting_agent=support_boundary_guardrail_agent,
            input=text_input,
            run_config=run_config_guardrail,
        )
        check_output = result.final_output_as(SupportBoundaryCheckOutput)
        message_to_send = _message_for_support_boundary(
            check_output.classification,
            check_output.business_subtype,
        )
        logging.info(
            "[Guardrail:Boundary] Check result: classification=%s subtype=%s reasoning=%s",
            check_output.classification,
            check_output.business_subtype,
            check_output.reasoning,
        )
        output_info: dict[str, Any] = {
            "classification": check_output.classification,
            "business_subtype": check_output.business_subtype,
            "reasoning": check_output.reasoning,
            "tripwire_triggered": bool(message_to_send),
        }
        if message_to_send:
            output_info["message"] = message_to_send
        return output_info
    except Exception as e:
        logging.error(f"[Guardrail:Boundary] Error during check: {e}", exc_info=True)
        return {
            "classification": "yearn_support",
            "business_subtype": None,
            "error": str(e),
            "tripwire_triggered": False,
        }


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
