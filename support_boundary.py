"""Live outer support-boundary classification shared by Discord and local replay."""

import logging
import re
from typing import Any, Literal

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.model_settings import Reasoning
from pydantic import BaseModel, Field

import config
from agent_prompts import SUPPORT_BOUNDARY_GUARDRAIL_INSTRUCTIONS
from bot_behavior import (
    JOB_INQUIRY_REDIRECT_MESSAGE,
    LISTING_DENIAL_MESSAGE,
    OUT_OF_SCOPE_SUPPORT_MESSAGE,
    SECURITY_VENDOR_BOUNDARY_MESSAGE,
    SECURITY_PROCESS_URL,
    STANDARD_REDIRECT_MESSAGE,
)


_SUPPORT_SCOPE_TX_HASH_RE = re.compile(r"(?:[a-z]+:)?0x[a-fA-F0-9]{64}")
_SUPPORT_SCOPE_ADDRESS_RE = re.compile(r"(?:[a-z]+:)?0x[a-fA-F0-9]{40}")
_SECURITY_PROCESS_EXCEPTION_REQUIRED_TOKENS = (
    "immunefi",
    "zkpassport",
    "kyc",
    "jurisdiction",
)
_SECURITY_PROCESS_EXCEPTION_BLOCKERS = (
    "blocked",
    "block",
    "cannot use",
    "can't use",
    "cant use",
    "unable to use",
    "cannot submit",
    "can't submit",
    "cant submit",
    "unable to submit",
    "not working",
    "isn't working",
    "isnt working",
    "unavailable",
    "restriction",
    "restricted",
)


class SupportBoundaryCheckOutput(BaseModel):
    classification: Literal[
        "yearn_support",
        "business_boundary",
        "security_process_boundary",
        "non_support_assistant",
        "uncertain",
    ] = Field(..., description="Top-level outer boundary classification for the user message.")
    business_subtype: Literal[
        "listing", "general_bd", "vendor_security", "job_inquiry"
    ] | None = Field(
        default=None,
        description="Subtype only when classification is business_boundary.",
    )
    reasoning: str = Field(..., description="Brief explanation for the classification.")


def _looks_like_support_scope_primitive(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(
        stripped
        and (
            _SUPPORT_SCOPE_TX_HASH_RE.fullmatch(stripped)
            or _SUPPORT_SCOPE_ADDRESS_RE.fullmatch(stripped)
        )
    )


def is_security_process_exception_request(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(
        lowered
        and any(
            token in lowered
            for token in _SECURITY_PROCESS_EXCEPTION_REQUIRED_TOKENS
        )
        and any(token in lowered for token in _SECURITY_PROCESS_EXCEPTION_BLOCKERS)
    )


def _message_for_support_boundary(
    text_input: str,
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
        if is_security_process_exception_request(text_input):
            return config.SECURITY_ALTERNATE_CONTACT_MESSAGE
        return (
            "If you are reporting a Yearn security issue and want bounty or disclosure handling, "
            f"use Yearn's official security process at {SECURITY_PROCESS_URL}. "
            "Human help is required beyond that path."
        )
    if classification == "non_support_assistant":
        return OUT_OF_SCOPE_SUPPORT_MESSAGE
    return None


support_boundary_guardrail_agent = Agent(
    name="Support Boundary Guardrail Check",
    instructions=SUPPORT_BOUNDARY_GUARDRAIL_INSTRUCTIONS,
    output_type=SupportBoundaryCheckOutput,
    model=config.LLM_GUARDRAIL_MODEL,
    model_settings=ModelSettings(
        reasoning=Reasoning(effort=config.LLM_GUARDRAIL_REASONING_EFFORT),
        verbosity=config.LLM_GUARDRAIL_VERBOSITY,
    ),
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
    logging.info("[Guardrail:Boundary] Analyzing input: '%s...'", text_input[:100])

    try:
        result = await Runner().run(
            starting_agent=support_boundary_guardrail_agent,
            input=text_input,
            run_config=RunConfig(
                workflow_name="Yearn Support Boundary Guardrail Check",
                tracing_disabled=True,
            ),
        )
        check_output = result.final_output_as(SupportBoundaryCheckOutput)
        business_subtype = (
            check_output.business_subtype
            if check_output.classification == "business_boundary"
            else None
        )
        message_to_send = _message_for_support_boundary(
            text_input,
            check_output.classification,
            business_subtype,
        )
        logging.info(
            "[Guardrail:Boundary] Check result: classification=%s subtype=%s reasoning=%s",
            check_output.classification,
            business_subtype,
            check_output.reasoning,
        )
        output_info: dict[str, Any] = {
            "classification": check_output.classification,
            "business_subtype": business_subtype,
            "reasoning": check_output.reasoning,
            "tripwire_triggered": bool(message_to_send),
        }
        if message_to_send:
            output_info["message"] = message_to_send
        return output_info
    except Exception as exc:
        logging.error(
            "[Guardrail:Boundary] Error during check: %s",
            exc,
            exc_info=True,
        )
        return {
            "classification": "yearn_support",
            "business_subtype": None,
            "error": str(exc),
            "tripwire_triggered": False,
        }
