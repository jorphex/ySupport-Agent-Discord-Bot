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
class InvestigationEvidence:
    wallet: Optional[str] = None
    chain: Optional[str] = None
    tx_hashes: List[str] = field(default_factory=list)
    withdrawal_target_chain: Optional[str] = None
    withdrawal_target_vault: Optional[str] = None


@dataclass
class TicketInvestigationJob:
    channel_id: int
    requested_intent: Optional[str] = None
    mode: InvestigationMode = "idle"
    current_specialty: Optional[SpecialtyKey] = None
    last_specialty: Optional[SpecialtyKey] = None
    evidence: InvestigationEvidence = field(default_factory=InvestigationEvidence)

    def record_requested_intent(self, intent: Optional[str]) -> None:
        self.requested_intent = intent

    def begin_collecting(self, intent: Optional[str] = None) -> None:
        if intent is not None:
            self.record_requested_intent(intent)
        self.mode = "collecting"

    def begin_investigating(self) -> None:
        self.mode = "investigating"

    def mark_waiting_for_user(self) -> None:
        self.mode = "waiting_for_user"

    def mark_escalated_to_human(self) -> None:
        self.mode = "escalated_to_human"

    def clear_current_specialty(self) -> None:
        self.current_specialty = None

    def complete_specialist_turn(self, specialty: Optional[SpecialtyKey]) -> None:
        if specialty in {"data", "docs", "bug"}:
            self.last_specialty = specialty
            self.current_specialty = specialty
            if specialty != "data":
                self.clear_withdrawal_target()
            return
        self.clear_withdrawal_target()
        self.clear_current_specialty()

    def remember_wallet(self, wallet: str) -> None:
        self.evidence.wallet = wallet

    def remember_chain(self, chain: str) -> None:
        self.evidence.chain = chain

    def remember_tx_hash(self, tx_hash: str) -> None:
        if tx_hash not in self.evidence.tx_hashes:
            self.evidence.tx_hashes.append(tx_hash)

    def remember_withdrawal_target(self, chain: str, vault: str) -> None:
        self.evidence.withdrawal_target_chain = chain
        self.evidence.withdrawal_target_vault = vault

    def clear_withdrawal_target(self) -> None:
        self.evidence.withdrawal_target_chain = None
        self.evidence.withdrawal_target_vault = None

    def apply_snapshot(self, other: "TicketInvestigationJob") -> None:
        self.requested_intent = other.requested_intent
        self.mode = other.mode
        self.current_specialty = other.current_specialty
        self.last_specialty = other.last_specialty
        self.evidence.wallet = other.evidence.wallet
        self.evidence.chain = other.evidence.chain
        self.evidence.tx_hashes = list(other.evidence.tx_hashes)
        self.evidence.withdrawal_target_chain = other.evidence.withdrawal_target_chain
        self.evidence.withdrawal_target_vault = other.evidence.withdrawal_target_vault


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

ticket_investigation_jobs: Dict[int, TicketInvestigationJob] = {}

# Last known wallet address per channel (checksummed)
last_wallet_by_channel: Dict[int, str] = {}

# Pending confirmation for a new wallet address per channel
pending_wallet_confirmation_by_channel: Dict[int, str] = {}

# Timestamp of the last bot reply per channel
last_bot_reply_ts_by_channel: Dict[int, datetime] = {}


def get_or_create_ticket_investigation_job(channel_id: int) -> TicketInvestigationJob:
    job = ticket_investigation_jobs.get(channel_id)
    if job is None:
        job = TicketInvestigationJob(channel_id=channel_id)
        ticket_investigation_jobs[channel_id] = job
    return job


def clear_ticket_investigation_job(channel_id: int) -> None:
    ticket_investigation_jobs.pop(channel_id, None)
