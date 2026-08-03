"""Legacy local support-agent graph used by explicit local replay and LLM evals.

The live Discord backend is Codex. Production boundary classification lives in
``support_boundary`` so importing the live shell does not construct this graph.
"""

from typing import Literal

from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    RunContextWrapper,
    TResponseInputItem,
    handoff,
    input_guardrail,
)
from agents.model_settings import Reasoning
from pydantic import BaseModel, Field

import config
from agent_prompts import (
    TICKET_TRIAGE_ROUTER_INSTRUCTIONS,
    TRIAGE_AGENT_INSTRUCTIONS,
    YEARn_BUG_TRIAGE_AGENT_INSTRUCTIONS,
    YEARn_DATA_AGENT_INSTRUCTIONS,
    YEARn_DOCS_QA_AGENT_INSTRUCTIONS,
)
from state import BotRunContext
from support_boundary import (
    SupportBoundaryCheckOutput,
    evaluate_support_boundary,
    is_security_process_exception_request,
    support_boundary_guardrail_agent,
)
from support_tools import (
    answer_from_docs_tool,
    check_all_deposits_tool,
    fetch_repo_artifacts_tool,
    fetch_report_artifact_tool,
    get_withdrawal_instructions_tool,
    inspect_onchain_tool,
    pretriage_repo_claim_tool,
    repo_context_status_tool,
    search_repo_context_tool,
    search_vaults_tool,
)


class TicketTriageDecision(BaseModel):
    action: Literal[
        "route_data",
        "route_docs",
        "route_bug",
        "ask_clarifying",
        "respond_directly",
        "human_escalation",
    ] = Field(..., description="The next runtime action for this ticket turn.")
    message: str | None = Field(
        default=None,
        description="User-facing message for ask_clarifying, respond_directly, or human_escalation. Leave empty for route_* actions.",
    )
    reasoning: str = Field(..., description="Brief explanation for the routing decision.")


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
        del agent
        run_context = ctx.context
        runtime_context = (
            "# Runtime Context\n"
            f"- project_context: {run_context.project_context}\n"
            f"- initial_button_intent: {run_context.initial_button_intent or 'none'}\n"
            f"- is_public_trigger: {str(run_context.is_public_trigger).lower()}\n"
        )
        return f"{base_instructions}\n\n{runtime_context}"

    return _instructions


@input_guardrail(name="Support Boundary Guardrail")
async def support_boundary_guardrail(
    ctx: RunContextWrapper[BotRunContext],
    agent: Agent,
    input_data: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    del ctx, agent
    if isinstance(input_data, str):
        text_input = input_data
    else:
        text_input = ""
        for item in reversed(input_data):
            if (
                isinstance(item, dict)
                and item.get("role") == "user"
                and isinstance(item.get("content"), str)
            ):
                text_input = item["content"]
                break
    if not text_input:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    output_info = await evaluate_support_boundary(text_input)
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=bool(output_info.get("tripwire_triggered")),
    )


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
        handoff(
            yearn_data_agent,
            tool_name_override="transfer_to_yearn_data_specialist",
            tool_description_override="Handoff for specific YEARN data (vaults, deposits, APR, TVL, balances, withdrawal instructions).",
        ),
        handoff(
            yearn_docs_qa_agent,
            tool_name_override="transfer_to_yearn_docs_qa_specialist",
            tool_description_override="Handoff for general questions about YEARN concepts, documentation, risks.",
        ),
        handoff(
            yearn_bug_triage_agent,
            tool_name_override="transfer_to_yearn_bug_triage_specialist",
            tool_description_override="Handoff for YEARN bug reports, UI issues, migration issues, and protocol behavior claims that should be checked against docs and repo context before human escalation.",
        ),
    ],
    input_guardrails=[support_boundary_guardrail],
    model=config.LLM_TRIAGE_AGENT_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_TRIAGE_AGENT_REASONING_EFFORT,
        verbosity=config.LLM_TRIAGE_AGENT_VERBOSITY,
    ),
)

ticket_triage_router_agent = Agent[BotRunContext](
    name="Ticket Triage Router",
    instructions=_with_runtime_context(TICKET_TRIAGE_ROUTER_INSTRUCTIONS),
    output_type=TicketTriageDecision,
    model=config.LLM_TRIAGE_AGENT_MODEL,
    model_settings=_gpt5_model_settings(
        effort=config.LLM_TRIAGE_AGENT_REASONING_EFFORT,
        verbosity=config.LLM_TRIAGE_AGENT_VERBOSITY,
    ),
)


__all__ = [
    "SupportBoundaryCheckOutput",
    "TicketTriageDecision",
    "evaluate_support_boundary",
    "is_security_process_exception_request",
    "support_boundary_guardrail",
    "support_boundary_guardrail_agent",
    "ticket_triage_router_agent",
    "triage_agent",
    "yearn_bug_triage_agent",
    "yearn_data_agent",
    "yearn_docs_qa_agent",
]
