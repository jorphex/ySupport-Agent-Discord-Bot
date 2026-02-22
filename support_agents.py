import logging
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, RunContextWrapper,
    ModelSettings, handoff,
    input_guardrail, GuardrailFunctionOutput,
    RunConfig,
    TResponseInputItem,
    AgentsException,
)

from agent_prompts import (
    YEARn_DATA_AGENT_INSTRUCTIONS,
    YEARn_DOCS_QA_AGENT_INSTRUCTIONS,
    BD_PRIORITY_GUARDRAIL_INSTRUCTIONS,
    TRIAGE_AGENT_INSTRUCTIONS,
)
from bot_behavior import (
    LISTING_DENIAL_MESSAGE,
    STANDARD_REDIRECT_MESSAGE,
    JOB_INQUIRY_REDIRECT_MESSAGE,
)
from state import BotRunContext
from support_tools import (
    search_vaults_tool,
    check_all_deposits_tool,
    get_withdrawal_instructions_tool,
    answer_from_docs_tool,
)


class BDPriorityCheckOutput(BaseModel):
    request_type: Literal["listing", "partnership", "marketing", "other_bd", "job_inquiry", "not_bd_pr"] = Field(..., description="Classify the user's primary intent: 'listing' (requesting Yearn list their token), 'partnership' (proposing integration/collaboration), 'marketing' (joint marketing/promotion), 'other_bd' (other business development), 'job_inquiry' (asking to work for/contribute to Yearn, grant requests), or 'not_bd_pr' (standard support request or unrelated).")
    reasoning: str = Field(..., description="Brief explanation for the classification.")


class GuardrailResponseMessageException(AgentsException):
    def __init__(self, message: str, guardrail_output: Optional[BDPriorityCheckOutput] = None):
        super().__init__(message)
        self.message = message
        self.guardrail_output = guardrail_output


bd_priority_guardrail_agent = Agent[BotRunContext](
    name="BD/PR/Listing Guardrail Check",
    instructions=BD_PRIORITY_GUARDRAIL_INSTRUCTIONS,
    output_type=BDPriorityCheckOutput,
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.1)
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
    instructions=YEARn_DATA_AGENT_INSTRUCTIONS,
    tools=[
        search_vaults_tool,
        check_all_deposits_tool,
        get_withdrawal_instructions_tool,
    ],
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.0)
)

yearn_docs_qa_agent = Agent[BotRunContext](
    name="Yearn Docs QA Specialist",
    instructions=YEARn_DOCS_QA_AGENT_INSTRUCTIONS,
    tools=[answer_from_docs_tool],
    model="gpt-4.1-mini",
    tool_use_behavior="stop_on_first_tool",
    model_settings=ModelSettings(temperature=0.2)
)

triage_agent = Agent[BotRunContext](
    name="Support Triage Agent",
    instructions=TRIAGE_AGENT_INSTRUCTIONS,
    handoffs=[
        handoff(yearn_data_agent, tool_name_override="transfer_to_yearn_data_specialist", tool_description_override="Handoff for specific YEARN data (vaults, deposits, APR, TVL, balances, withdrawal instructions)."),
        handoff(yearn_docs_qa_agent, tool_name_override="transfer_to_yearn_docs_qa_specialist", tool_description_override="Handoff for general questions about YEARN concepts, documentation, risks."),
    ],
    input_guardrails=[bd_priority_guardrail],
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.1)
)
