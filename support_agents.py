import logging
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field

import config
from agents import (
    Agent, Runner, RunContextWrapper,
    ModelSettings, handoff,
    input_guardrail, GuardrailFunctionOutput,
    RunConfig,
    TResponseInputItem,
    AgentsException,
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


class GuardrailResponseMessageException(AgentsException):
    def __init__(self, message: str, guardrail_output: Optional[BDPriorityCheckOutput] = None):
        super().__init__(message)
        self.message = message
        self.guardrail_output = guardrail_output


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

    logging.info(f"[Guardrail:BD/Priority] Analyzing input: '{text_input[:100]}...'")

    try:
        guardrail_runner = Runner()
        run_config_guardrail = RunConfig(workflow_name="BD/Priority Guardrail Check", tracing_disabled=True)
        result = await guardrail_runner.run(
            starting_agent=bd_priority_guardrail_agent,
            input=text_input,
            run_config=run_config_guardrail
        )
        check_output = result.final_output_as(BDPriorityCheckOutput)

        logging.info(f"[Guardrail:BD/Priority] Check result: type={check_output.request_type}, Reasoning: {check_output.reasoning}")

        message_to_send = None
        should_trigger = False

        if check_output.request_type == "listing":
            message_to_send = LISTING_DENIAL_MESSAGE
            should_trigger = True
        elif check_output.request_type == "vendor_security":
            message_to_send = SECURITY_VENDOR_BOUNDARY_MESSAGE
            should_trigger = True
        elif check_output.request_type in ["partnership", "marketing", "other_bd"]:
            message_to_send = STANDARD_REDIRECT_MESSAGE
            should_trigger = True
        elif check_output.request_type == "job_inquiry":
            message_to_send = JOB_INQUIRY_REDIRECT_MESSAGE
            should_trigger = True

        output_info_dict = {
            "classification": check_output.model_dump()
        }
        if should_trigger and message_to_send:
            output_info_dict["message"] = message_to_send
            logging.info(f"[Guardrail:BD/Priority] Returning tripwire=True. Type: {check_output.request_type}.")
            return GuardrailFunctionOutput(
                output_info=output_info_dict,
                tripwire_triggered=True
            )
        logging.info("[Guardrail:BD/Priority] Returning tripwire=False.")
        return GuardrailFunctionOutput(
            output_info=output_info_dict,
            tripwire_triggered=False
        )

    except Exception as e:
        logging.error(f"[Guardrail:BD/Priority] Error during check: {e}", exc_info=True)
        return GuardrailFunctionOutput(output_info={"error": str(e)}, tripwire_triggered=False)


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
