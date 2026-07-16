from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agents import TResponseInputItem

from state import BotRunContext, TicketInvestigationJob


@dataclass
class TicketAgentFlowOutcome:
    raw_final_reply: str
    conversation_history: list[TResponseInputItem]
    completed_agent_key: str | None
    requires_human_handoff: bool


@dataclass
class TicketTurnRequest:
    aggregated_text: str
    input_list: list[TResponseInputItem]
    current_history: list[TResponseInputItem]
    run_context: BotRunContext
    investigation_job: TicketInvestigationJob
    workflow_name: str
    precomputed_boundary: dict[str, Any] | None = None
    send_bug_review_status: Callable[[], Awaitable[None]] | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    turn_source: str = "user"
    turn_instruction: str | None = None
