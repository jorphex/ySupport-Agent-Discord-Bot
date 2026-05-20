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
TeamHandoffNoticeStatus = Literal[
    "open",
    "pending_delivery",
    "delivered_pending_close",
]
InvestigationMode = Literal[
    "idle",
    "collecting",
    "investigating",
    "waiting_for_user",
    "escalated_to_human",
]
TicketStopReason = Literal["manual_stop", "boundary_stop", "runtime_error"]


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
class TeamHandoffNotice:
    telegram_chat_id: str
    telegram_message_id: int
    target: str
    reason: str
    message_text: str | None = None
    reply_consumed: bool = False
    status: TeamHandoffNoticeStatus = "open"
    pending_reply_text: Optional[str] = None
    pending_reply_message_id: Optional[int] = None


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
stop_reasons_by_channel: Dict[int, TicketStopReason] = {}
conversation_threads: Dict[int, List[TResponseInputItem]] = {}
pending_messages: Dict[int, str] = {}
pending_attachments_by_channel: Dict[int, List[Dict[str, Any]]] = {}
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

# Ticket opener / customer associated with the ticket channel
ticket_owner_user_id_by_channel: Dict[int, int] = {}
team_handoff_notice_by_channel: Dict[int, TeamHandoffNotice] = {}

_DISCORD_STATE_ROOT = Path(config.TICKET_EXECUTION_STATE_ROOT) / "discord_state"
_TICKET_STATE_DIR = _DISCORD_STATE_ROOT / "tickets"
_PUBLIC_STATE_DIR = _DISCORD_STATE_ROOT / "public"
_TELEGRAM_STATE_PATH = _DISCORD_STATE_ROOT / "telegram_state.json"


def get_or_create_ticket_investigation_job(channel_id: int) -> TicketInvestigationJob:
    job = ticket_investigation_jobs.get(channel_id)
    if job is None:
        job = TicketInvestigationJob(channel_id=channel_id)
        ticket_investigation_jobs[channel_id] = job
    return job


def mark_ticket_awaiting_initial_button(channel_id: int) -> None:
    channels_awaiting_initial_button_press.add(channel_id)
    persist_ticket_state(channel_id)


def apply_initial_button_intent(
    channel_id: int,
    intent_category: str,
    *,
    investigation_mode: InvestigationMode,
    bug_report_debounce: bool = False,
) -> TicketInvestigationJob:
    channels_awaiting_initial_button_press.discard(channel_id)
    channel_intent_after_button[channel_id] = intent_category
    job = get_or_create_ticket_investigation_job(channel_id)
    job.record_requested_intent(intent_category)
    if investigation_mode == "collecting":
        job.begin_collecting()
    elif investigation_mode == "waiting_for_user":
        job.mark_waiting_for_user()
    elif investigation_mode == "escalated_to_human":
        job.mark_escalated_to_human()
    if bug_report_debounce:
        bug_report_debounce_channels.add(channel_id)
    else:
        bug_report_debounce_channels.discard(channel_id)
    persist_ticket_state(channel_id)
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
    ticket_owner_user_id = payload.get("ticket_owner_user_id")
    if ticket_owner_user_id is not None:
        ticket_owner_user_id_by_channel[channel_id] = int(ticket_owner_user_id)
    if payload.get("stopped"):
        stopped_channels.add(channel_id)
        stop_reason = payload.get("stop_reason")
        if isinstance(stop_reason, str) and stop_reason in {
            "manual_stop",
            "boundary_stop",
            "runtime_error",
        }:
            stop_reasons_by_channel[channel_id] = stop_reason
    if payload.get("awaiting_initial_button_press"):
        channels_awaiting_initial_button_press.add(channel_id)
    intent = payload.get("channel_intent_after_button")
    if intent:
        channel_intent_after_button[channel_id] = intent
    if payload.get("bug_report_debounce"):
        bug_report_debounce_channels.add(channel_id)
    handoff_payload = payload.get("team_handoff_notice")
    if isinstance(handoff_payload, dict):
        handoff_message_id = handoff_payload.get("telegram_message_id")
        handoff_chat_id = handoff_payload.get("telegram_chat_id")
        if handoff_message_id is not None and handoff_chat_id:
            team_handoff_notice_by_channel[channel_id] = TeamHandoffNotice(
                telegram_chat_id=str(handoff_chat_id),
                telegram_message_id=int(handoff_message_id),
                target=str(handoff_payload.get("target") or "support_manual"),
                reason=str(handoff_payload.get("reason") or ""),
                message_text=_normalize_optional_text(handoff_payload.get("message_text")),
                reply_consumed=bool(handoff_payload.get("reply_consumed")),
                status=str(handoff_payload.get("status") or "open"),
                pending_reply_text=handoff_payload.get("pending_reply_text"),
                pending_reply_message_id=(
                    int(handoff_payload["pending_reply_message_id"])
                    if handoff_payload.get("pending_reply_message_id") is not None
                    else None
                ),
            )
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
        "ticket_owner_user_id": ticket_owner_user_id_by_channel.get(channel_id),
        "stopped": channel_id in stopped_channels,
        "stop_reason": stop_reasons_by_channel.get(channel_id),
        "awaiting_initial_button_press": channel_id in channels_awaiting_initial_button_press,
        "channel_intent_after_button": channel_intent_after_button.get(channel_id),
        "bug_report_debounce": channel_id in bug_report_debounce_channels,
        "team_handoff_notice": _serialize_team_handoff_notice(
            team_handoff_notice_by_channel.get(channel_id)
        ),
    }
    _write_json(_TICKET_STATE_DIR / f"{channel_id}.json", payload)


def clear_ticket_channel_state(
    channel_id: int,
    *,
    keep_stopped: bool = False,
    stop_reason: TicketStopReason | None = None,
    preserve_ticket_owner: bool = False,
    delete_persisted: bool = False,
) -> None:
    preserved_ticket_owner_user_id = (
        ticket_owner_user_id_by_channel.get(channel_id)
        if keep_stopped or preserve_ticket_owner
        else None
    )
    preserved_stop_reason = (
        stop_reason or stop_reasons_by_channel.get(channel_id) or "manual_stop"
        if keep_stopped
        else None
    )
    conversation_threads.pop(channel_id, None)
    ticket_investigation_jobs.pop(channel_id, None)
    pending_messages.pop(channel_id, None)
    pending_attachments_by_channel.pop(channel_id, None)
    last_wallet_by_channel.pop(channel_id, None)
    pending_wallet_confirmation_by_channel.pop(channel_id, None)
    last_bot_reply_ts_by_channel.pop(channel_id, None)
    ticket_owner_user_id_by_channel.pop(channel_id, None)
    team_handoff_notice_by_channel.pop(channel_id, None)
    channels_awaiting_initial_button_press.discard(channel_id)
    channel_intent_after_button.pop(channel_id, None)
    bug_report_debounce_channels.discard(channel_id)
    if keep_stopped:
        stopped_channels.add(channel_id)
        if preserved_stop_reason is not None:
            stop_reasons_by_channel[channel_id] = preserved_stop_reason
    else:
        stopped_channels.discard(channel_id)
        stop_reasons_by_channel.pop(channel_id, None)
        monitored_new_channels.discard(channel_id)
    if preserved_ticket_owner_user_id is not None:
        ticket_owner_user_id_by_channel[channel_id] = preserved_ticket_owner_user_id
    reset_ticket_codex_session(channel_id)
    if delete_persisted:
        _safe_unlink(_TICKET_STATE_DIR / f"{channel_id}.json")
        return
    persist_ticket_state(channel_id)


def reset_ticket_channel_for_terminal_reply(channel_id: int) -> None:
    clear_ticket_channel_state(
        channel_id,
        keep_stopped=False,
        preserve_ticket_owner=True,
        delete_persisted=True,
    )


def stop_ticket_channel(
    channel_id: int,
    *,
    reason: TicketStopReason = "manual_stop",
) -> None:
    clear_ticket_channel_state(
        channel_id,
        keep_stopped=True,
        stop_reason=reason,
        delete_persisted=False,
    )


def mark_ticket_channel_stopped(
    channel_id: int,
    *,
    reason: TicketStopReason = "runtime_error",
) -> None:
    stopped_channels.add(channel_id)
    stop_reasons_by_channel[channel_id] = reason
    persist_ticket_state(channel_id)


def recover_ticket_channel_from_runtime_stop(channel_id: int) -> bool:
    if channel_id not in stopped_channels:
        return False
    if stop_reasons_by_channel.get(channel_id) != "runtime_error":
        return False
    stopped_channels.discard(channel_id)
    stop_reasons_by_channel.pop(channel_id, None)
    reset_ticket_codex_session(channel_id)
    persist_ticket_state(channel_id)
    return True


def remember_ticket_owner_user_id(channel_id: int, user_id: int) -> None:
    ticket_owner_user_id_by_channel[channel_id] = user_id
    persist_ticket_state(channel_id)


def remember_team_handoff_notice(channel_id: int, notice: TeamHandoffNotice) -> None:
    team_handoff_notice_by_channel[channel_id] = notice
    persist_ticket_state(channel_id)


def clear_team_handoff_notice(channel_id: int) -> None:
    if team_handoff_notice_by_channel.pop(channel_id, None) is not None:
        persist_ticket_state(channel_id)


def mark_team_handoff_notice_consumed(channel_id: int) -> None:
    notice = team_handoff_notice_by_channel.get(channel_id)
    if notice is None:
        return
    notice.reply_consumed = True
    persist_ticket_state(channel_id)


def mark_team_handoff_notice_pending_delivery(
    channel_id: int,
    *,
    reply_text: str,
    reply_message_id: int,
) -> None:
    notice = team_handoff_notice_by_channel.get(channel_id)
    if notice is None:
        return
    notice.reply_consumed = True
    notice.status = "pending_delivery"
    notice.pending_reply_text = reply_text
    notice.pending_reply_message_id = reply_message_id
    persist_ticket_state(channel_id)


def mark_team_handoff_notice_delivered(channel_id: int) -> None:
    notice = team_handoff_notice_by_channel.get(channel_id)
    if notice is None:
        return
    notice.reply_consumed = True
    notice.status = "delivered_pending_close"
    persist_ticket_state(channel_id)


def reopen_team_handoff_notice(channel_id: int) -> None:
    notice = team_handoff_notice_by_channel.get(channel_id)
    if notice is None:
        return
    notice.reply_consumed = False
    notice.status = "open"
    notice.pending_reply_text = None
    notice.pending_reply_message_id = None
    persist_ticket_state(channel_id)


def is_ticket_waiting_for_team(channel_id: int) -> bool:
    job = ticket_investigation_jobs.get(channel_id)
    if job is None:
        return False
    return job.mode == "escalated_to_human"


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


def load_telegram_update_offset() -> int | None:
    payload = _read_json(_TELEGRAM_STATE_PATH)
    if not isinstance(payload, dict):
        return None
    raw_offset = payload.get("update_offset")
    if isinstance(raw_offset, int):
        return raw_offset
    return None


def persist_telegram_update_offset(update_offset: int) -> None:
    if not _ensure_state_dirs():
        return
    _write_json(_TELEGRAM_STATE_PATH, {"update_offset": update_offset})


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
            channel_id in ticket_owner_user_id_by_channel,
            channel_id in team_handoff_notice_by_channel,
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


def _serialize_team_handoff_notice(
    notice: TeamHandoffNotice | None,
) -> dict[str, Any] | None:
    if notice is None:
        return None
    return {
        "telegram_chat_id": notice.telegram_chat_id,
        "telegram_message_id": notice.telegram_message_id,
        "target": notice.target,
        "reason": notice.reason,
        "message_text": notice.message_text,
        "reply_consumed": notice.reply_consumed,
        "status": notice.status,
        "pending_reply_text": notice.pending_reply_text,
        "pending_reply_message_id": notice.pending_reply_message_id,
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


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
