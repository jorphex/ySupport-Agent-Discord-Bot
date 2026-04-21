from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import config
from codex_support_sessions import CodexSupportSessionManager

try:
    from agents import TResponseInputItem
except ModuleNotFoundError:
    TResponseInputItem = dict[str, Any]


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
    conversation_owner_id: Optional[int] = None
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
    investigation_job: "TicketInvestigationJob | None" = None


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

_DISCORD_STATE_ROOT = Path(config.TICKET_EXECUTION_STATE_ROOT) / "discord_state"
_TICKET_STATE_DIR = _DISCORD_STATE_ROOT / "tickets"
_PUBLIC_STATE_DIR = _DISCORD_STATE_ROOT / "public"


def get_or_create_ticket_investigation_job(channel_id: int) -> TicketInvestigationJob:
    job = ticket_investigation_jobs.get(channel_id)
    if job is None:
        job = TicketInvestigationJob(channel_id=channel_id)
        ticket_investigation_jobs[channel_id] = job
    return job


def clear_ticket_investigation_job(channel_id: int) -> None:
    ticket_investigation_jobs.pop(channel_id, None)


def hydrate_ticket_state(channel_id: int) -> None:
    if _ticket_state_is_loaded(channel_id):
        return
    payload = _read_json(_TICKET_STATE_DIR / f"{channel_id}.json")
    if payload is None:
        return
    conversation_threads[channel_id] = list(payload.get("history", []))
    job_payload = payload.get("investigation_job")
    if isinstance(job_payload, dict):
        ticket_investigation_jobs[channel_id] = _deserialize_investigation_job(job_payload)
    wallet = payload.get("last_wallet")
    if wallet:
        last_wallet_by_channel[channel_id] = wallet
    pending_wallet = payload.get("pending_wallet_confirmation")
    if pending_wallet:
        pending_wallet_confirmation_by_channel[channel_id] = pending_wallet
    last_reply_at = _parse_datetime(payload.get("last_bot_reply_at"))
    if last_reply_at is not None:
        last_bot_reply_ts_by_channel[channel_id] = last_reply_at
    if payload.get("stopped"):
        stopped_channels.add(channel_id)
    if payload.get("awaiting_initial_button_press"):
        channels_awaiting_initial_button_press.add(channel_id)
    intent = payload.get("channel_intent_after_button")
    if intent:
        channel_intent_after_button[channel_id] = intent
    if payload.get("bug_report_debounce"):
        bug_report_debounce_channels.add(channel_id)
    monitored_new_channels.add(channel_id)


def persist_ticket_state(channel_id: int) -> None:
    if not _ensure_state_dirs():
        return
    payload = {
        "channel_id": channel_id,
        "history": list(conversation_threads.get(channel_id, [])),
        "investigation_job": _serialize_investigation_job(
            ticket_investigation_jobs.get(channel_id)
        ),
        "last_wallet": last_wallet_by_channel.get(channel_id),
        "pending_wallet_confirmation": pending_wallet_confirmation_by_channel.get(channel_id),
        "last_bot_reply_at": _format_datetime(last_bot_reply_ts_by_channel.get(channel_id)),
        "stopped": channel_id in stopped_channels,
        "awaiting_initial_button_press": channel_id in channels_awaiting_initial_button_press,
        "channel_intent_after_button": channel_intent_after_button.get(channel_id),
        "bug_report_debounce": channel_id in bug_report_debounce_channels,
    }
    _write_json(_TICKET_STATE_DIR / f"{channel_id}.json", payload)


def clear_ticket_channel_state(
    channel_id: int,
    *,
    keep_stopped: bool = False,
    delete_persisted: bool = False,
) -> None:
    conversation_threads.pop(channel_id, None)
    ticket_investigation_jobs.pop(channel_id, None)
    pending_messages.pop(channel_id, None)
    last_wallet_by_channel.pop(channel_id, None)
    pending_wallet_confirmation_by_channel.pop(channel_id, None)
    last_bot_reply_ts_by_channel.pop(channel_id, None)
    channels_awaiting_initial_button_press.discard(channel_id)
    channel_intent_after_button.pop(channel_id, None)
    bug_report_debounce_channels.discard(channel_id)
    if keep_stopped:
        stopped_channels.add(channel_id)
    else:
        stopped_channels.discard(channel_id)
        monitored_new_channels.discard(channel_id)
    reset_ticket_codex_session(channel_id)
    if delete_persisted:
        _safe_unlink(_TICKET_STATE_DIR / f"{channel_id}.json")
        return
    persist_ticket_state(channel_id)


def hydrate_public_conversation(author_id: int) -> None:
    if author_id in public_conversations:
        return
    payload = _read_json(_PUBLIC_STATE_DIR / f"{author_id}.json")
    if payload is None:
        return
    public_conversations[author_id] = PublicConversation(
        history=list(payload.get("history", [])),
        last_interaction_time=_parse_datetime(payload.get("last_interaction_time"))
        or datetime.now(timezone.utc),
        investigation_job=_deserialize_investigation_job(payload["investigation_job"])
        if isinstance(payload.get("investigation_job"), dict)
        else None,
    )


def persist_public_conversation(author_id: int) -> None:
    conversation = public_conversations.get(author_id)
    if conversation is None:
        return
    if not _ensure_state_dirs():
        return
    payload = {
        "author_id": author_id,
        "history": list(conversation.history),
        "last_interaction_time": _format_datetime(conversation.last_interaction_time),
        "investigation_job": _serialize_investigation_job(conversation.investigation_job),
    }
    _write_json(_PUBLIC_STATE_DIR / f"{author_id}.json", payload)


def clear_public_conversation(author_id: int) -> None:
    public_conversations.pop(author_id, None)
    _safe_unlink(_PUBLIC_STATE_DIR / f"{author_id}.json")
    reset_public_codex_session(author_id)


def reset_ticket_codex_session(channel_id: int) -> None:
    _reset_codex_session(f"ticket:{channel_id}")


def reset_public_codex_session(author_id: int) -> None:
    _reset_codex_session(f"public_user:{author_id}")


def _reset_codex_session(conversation_key: str) -> None:
    if not config.TICKET_EXECUTION_CODEX_SESSION_DIR:
        return
    manager = CodexSupportSessionManager(
        config.TICKET_EXECUTION_CODEX_SESSION_DIR,
        max_age_hours=config.TICKET_EXECUTION_CODEX_SESSION_MAX_AGE_HOURS,
    )
    manager.reset(conversation_key)


def _ticket_state_is_loaded(channel_id: int) -> bool:
    return any(
        (
            channel_id in conversation_threads,
            channel_id in ticket_investigation_jobs,
            channel_id in last_wallet_by_channel,
            channel_id in pending_wallet_confirmation_by_channel,
            channel_id in last_bot_reply_ts_by_channel,
            channel_id in stopped_channels,
            channel_id in channels_awaiting_initial_button_press,
            channel_id in channel_intent_after_button,
            channel_id in bug_report_debounce_channels,
        )
    )


def _ensure_state_dirs() -> bool:
    try:
        _TICKET_STATE_DIR.mkdir(parents=True, exist_ok=True)
        _PUBLIC_STATE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.error("Failed to create Discord state directories: %s", exc)
        return False
    return True


def _serialize_investigation_job(job: TicketInvestigationJob | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {
        "channel_id": job.channel_id,
        "requested_intent": job.requested_intent,
        "mode": job.mode,
        "current_specialty": job.current_specialty,
        "last_specialty": job.last_specialty,
        "evidence": {
            "wallet": job.evidence.wallet,
            "chain": job.evidence.chain,
            "tx_hashes": list(job.evidence.tx_hashes),
            "withdrawal_target_chain": job.evidence.withdrawal_target_chain,
            "withdrawal_target_vault": job.evidence.withdrawal_target_vault,
        },
    }


def _deserialize_investigation_job(payload: dict[str, Any]) -> TicketInvestigationJob:
    evidence = payload.get("evidence", {})
    return TicketInvestigationJob(
        channel_id=payload.get("channel_id"),
        requested_intent=payload.get("requested_intent"),
        mode=payload.get("mode", "idle"),
        current_specialty=payload.get("current_specialty"),
        last_specialty=payload.get("last_specialty"),
        evidence=InvestigationEvidence(
            wallet=evidence.get("wallet"),
            chain=evidence.get("chain"),
            tx_hashes=list(evidence.get("tx_hashes", [])),
            withdrawal_target_chain=evidence.get("withdrawal_target_chain"),
            withdrawal_target_vault=evidence.get("withdrawal_target_vault"),
        ),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError as exc:
        logging.error("Failed to write Discord state file %s: %s", path, exc)
        _safe_unlink(temp_path)
        return False
    return True


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logging.warning("Failed to remove Discord state file %s: %s", path, exc)


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _parse_datetime(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None
