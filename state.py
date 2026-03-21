from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Literal

from agents import TResponseInputItem


SpecialtyKey = Literal["data", "docs", "bug"]
InvestigationMode = Literal[
    "idle",
    "collecting",
    "investigating",
    "waiting_for_user",
    "escalated_to_human",
]


# State & Context Definitions
@dataclass
class BotRunContext:
    channel_id: int
    category_id: Optional[int] = None
    is_public_trigger: bool = False
    project_context: Literal["yearn", "unknown"] = "unknown"
    initial_button_intent: Optional[str] = None
    repo_search_calls: int = 0
    repo_fetch_calls: int = 0
    repo_searches_without_fetch: int = 0
    repo_last_search_query: Optional[str] = None
    repo_last_search_artifact_refs: List[str] = field(default_factory=list)


# State Management for Public Conversations
@dataclass
class PublicConversation:
    """Stores the state for a temporary public conversation."""
    history: List[TResponseInputItem]
    last_interaction_time: datetime


@dataclass
class TicketInvestigationState:
    channel_id: int
    requested_intent: Optional[str] = None
    active_mode: InvestigationMode = "idle"
    current_specialty: Optional[SpecialtyKey] = None
    last_specialty: Optional[SpecialtyKey] = None
    known_wallet: Optional[str] = None
    known_chain: Optional[str] = None
    known_tx_hashes: List[str] = field(default_factory=list)


# Globals
stopped_channels: set[int] = set()
conversation_threads: Dict[int, List[TResponseInputItem]] = {}
pending_messages: Dict[int, str] = {}
pending_tasks: Dict[int, asyncio.Task] = {}
monitored_new_channels: set[int] = set()

channels_awaiting_initial_button_press: set[int] = set()
channel_intent_after_button: Dict[int, str] = {}
bug_report_debounce_channels: set[int] = set()

public_conversations: Dict[int, PublicConversation] = {}

ticket_investigation_states: Dict[int, TicketInvestigationState] = {}

# Last known wallet address per channel (checksummed)
last_wallet_by_channel: Dict[int, str] = {}

# Pending confirmation for a new wallet address per channel
pending_wallet_confirmation_by_channel: Dict[int, str] = {}

# Timestamp of the last bot reply per channel
last_bot_reply_ts_by_channel: Dict[int, datetime] = {}


def get_or_create_ticket_investigation_state(channel_id: int) -> TicketInvestigationState:
    state = ticket_investigation_states.get(channel_id)
    if state is None:
        state = TicketInvestigationState(channel_id=channel_id)
        ticket_investigation_states[channel_id] = state
    return state
