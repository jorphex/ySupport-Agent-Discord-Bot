import asyncio
from copy import deepcopy
import importlib
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, List, Literal

import discord
from dotenv import load_dotenv

from agents import (
    TResponseInputItem,
    InputGuardrailTripwireTriggered, MaxTurnsExceeded, AgentsException,
    set_default_openai_key,
)

import config
from handoff import (
    answer_telegram_callback_query,
    build_archived_handoff_notice,
    build_closed_handoff_notice,
    build_dismissed_handoff_notice,
    build_pending_delivery_handoff_notice,
    delete_telegram_message,
    DISMISS_HANDOFF_CALLBACK_DATA,
    TelegramApiError,
    TelegramSentMessage,
    HandoffRoute,
    build_handoff_notice,
    build_user_handoff_reply,
    build_vague_team_reply_feedback,
    edit_handoff_notice,
    fetch_telegram_updates,
    infer_handoff_route,
    is_substantive_team_reply,
    send_handoff_notice,
    send_telegram_message,
    strip_handoff_placeholder,
    summarize_handoff_summary,
)
from router import is_bug_report_query
from support_boundary import evaluate_support_boundary
from state import (
    active_ticket_payloads,
    BotRunContext,
    PublicConversation,
    TicketInvestigationJob,
    channels_awaiting_initial_button_press,
    channel_intent_after_button,
    bug_report_debounce_channels,
    clear_team_handoff_notice,
    clear_public_conversation,
    clear_ticket_channel_state,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    hydrate_public_conversation,
    hydrate_ticket_state,
    hydrate_persisted_team_handoff_states,
    is_ticket_waiting_for_team,
    last_wallet_by_channel,
    last_bot_reply_ts_by_channel,
    load_telegram_update_offset,
    mark_team_handoff_notice_delivered,
    mark_team_handoff_notice_pending_delivery,
    mark_ticket_channel_stopped,
    mark_ticket_awaiting_initial_button,
    monitored_new_channels,
    pending_messages,
    pending_attachments_by_channel,
    pending_tasks,
    persist_telegram_update_offset,
    persist_public_conversation,
    persist_ticket_state,
    prune_expired_public_conversations,
    public_conversations,
    recover_ticket_channel_from_runtime_stop,
    remember_team_handoff_notice,
    remember_team_handoff_followup_attachments,
    remember_ticket_owner_user_id,
    reset_ticket_codex_session,
    reset_ticket_channel_for_terminal_reply,
    stop_ticket_channel,
    stopped_channels,
    TeamHandoffNotice,
    team_handoff_notice_by_channel,
    ticket_owner_user_id_by_channel,
)
from ticket_intake import prepare_ticket_turn_input
from ticket_investigation.json_endpoint import (
    build_ticket_execution_json_endpoint,
    JsonEndpointTicketExecutionTransport,
    prune_codex_support_sessions,
)
from ticket_investigation.context import (
    build_contextual_hints,
    merge_explicit_evidence,
)
from ticket_investigation.contracts import TicketTurnRequest
from ticket_investigation.executor import (
    TicketExecutionHooks,
    TransportTicketInvestigationExecutor,
)
from views import InitialInquiryView, StopBotView, is_ticket_contributor
from utils import send_long_message


load_dotenv()
set_default_openai_key(config.OPENAI_API_KEY)

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)


class _DiscordProgressReporter:
    def __init__(self, channel: discord.abc.Messageable, channel_id: int) -> None:
        self.channel = channel
        self.channel_id = channel_id
        self._message: discord.Message | None = None
        self._lines: list[str] = []
        self._last_line: str | None = None
        self._last_rendered: str | None = None
        self._last_sent_at = 0.0
        self._flush_task: asyncio.Task[None] | None = None

    async def update(self, line: str) -> None:
        normalized = line.strip()
        if not normalized or normalized == self._last_line:
            return
        self._last_line = normalized
        self._lines.append(normalized)
        self._lines = self._lines[-4:]
        await self._flush()

    async def close(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        if self._message is None:
            return
        try:
            await self._message.delete()
        except Exception:
            return
        finally:
            self._message = None

    async def _flush(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._message is not None and now - self._last_sent_at < 1.5:
            if self._flush_task is None or self._flush_task.done():
                delay = 1.5 - (now - self._last_sent_at)
                self._flush_task = asyncio.create_task(self._flush_later(delay))
            return
        content = self._render()
        if not content or content == self._last_rendered:
            return
        try:
            if self._message is None:
                self._message = await self.channel.send(content, suppress_embeds=True)
            else:
                await self._message.edit(content=content)
            self._last_rendered = content
            self._last_sent_at = loop.time()
            last_bot_reply_ts_by_channel[self.channel_id] = datetime.now(timezone.utc)
        except Exception as exc:
            logging.debug(
                "Failed to update progress message for channel %s: %s",
                self.channel_id,
                exc,
            )

    async def _flush_later(self, delay_seconds: float) -> None:
        try:
            await asyncio.sleep(max(delay_seconds, 0))
            await self._flush()
        except asyncio.CancelledError:
            return
        finally:
            self._flush_task = None

    def _render(self) -> str:
        if not self._lines:
            return ""
        return "Working...\n" + "\n".join(f"- {line}" for line in self._lines)


def _ticket_debounce_seconds(channel_id: int, run_context: BotRunContext) -> int:
    if run_context.initial_button_intent == "bug_report" or channel_id in bug_report_debounce_channels:
        return config.BUG_REPORT_COOLDOWN_SECONDS
    return config.COOLDOWN_SECONDS


def _normalize_discord_attachment(
    attachment: discord.Attachment,
) -> dict[str, Any]:
    content_type = (attachment.content_type or "").strip() or None
    payload = {
        "filename": attachment.filename,
        "url": attachment.url,
        "content_type": content_type,
        "size": attachment.size,
        "is_image": bool(content_type and content_type.startswith("image/")),
    }
    attachment_id = getattr(attachment, "id", None)
    if attachment_id is not None:
        payload["attachment_id"] = attachment_id
    return payload


def _attachment_payloads_from_message(
    message: discord.Message,
) -> list[dict[str, Any]]:
    attachments = getattr(message, "attachments", None) or []
    payloads = [
        _normalize_discord_attachment(attachment)
        for attachment in attachments
        if attachment.url
    ]
    message_id = getattr(message, "id", None)
    if message_id is not None:
        for payload in payloads:
            payload["source_message_id"] = message_id
    return payloads


async def _refresh_discord_attachment_urls(
    channel: discord.TextChannel,
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    refreshed = [dict(attachment) for attachment in attachments]
    indexes_by_message: dict[int, list[int]] = {}
    for index, attachment in enumerate(refreshed):
        source_message_id = attachment.get("source_message_id")
        if not isinstance(source_message_id, int) or attachment.get("attachment_id") is None:
            continue
        indexes_by_message.setdefault(source_message_id, []).append(index)

    for source_message_id, indexes in indexes_by_message.items():
        try:
            source_message = await channel.fetch_message(source_message_id)
        except Exception:
            logging.warning(
                "Could not refresh Discord attachment URLs from message %s in channel %s.",
                source_message_id,
                channel.id,
                exc_info=True,
            )
            continue
        current_by_id = {
            str(attachment.id): attachment for attachment in source_message.attachments
        }
        for index in indexes:
            current = current_by_id.get(str(refreshed[index].get("attachment_id")))
            if current is None:
                continue
            current_payload = _normalize_discord_attachment(current)
            current_payload["source_message_id"] = source_message_id
            refreshed[index] = current_payload
    return refreshed


def _dedupe_attachment_payloads(
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for attachment in attachments:
        url = str(attachment.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(dict(attachment))
    return deduped


def _merge_pending_ticket_payload(
    channel_id: int,
    text: str,
    attachments: list[dict[str, Any]],
) -> None:
    existing_text = pending_messages.get(channel_id)
    pending_messages[channel_id] = (
        f"{existing_text}\n{text}" if existing_text else text
    )
    pending_attachments_by_channel[channel_id] = _dedupe_attachment_payloads(
        pending_attachments_by_channel.get(channel_id, []) + attachments
    )


def _restore_active_ticket_payload(channel_id: int) -> None:
    active_payload = active_ticket_payloads.pop(channel_id, None)
    if active_payload is None:
        return
    _task, text, attachments = active_payload
    queued_text = pending_messages.pop(channel_id, None)
    queued_attachments = pending_attachments_by_channel.pop(channel_id, [])
    _merge_pending_ticket_payload(channel_id, text, attachments)
    if queued_text:
        _merge_pending_ticket_payload(
            channel_id,
            queued_text,
            queued_attachments,
        )


def _record_waiting_for_team_followup(
    channel_id: int,
    message: discord.Message,
) -> None:
    content = _message_text_for_turn(message)
    if not content:
        return
    history = conversation_threads.setdefault(channel_id, [])
    history.append({"role": "user", "content": content})
    attachments = _attachment_payloads_from_message(message)
    if attachments:
        remember_team_handoff_followup_attachments(channel_id, attachments)
    else:
        persist_ticket_state(channel_id)


def _message_text_for_turn(message: discord.Message) -> str:
    content = (message.content or "").strip()
    if content:
        return content
    if getattr(message, "attachments", None):
        return "(attachment only)"
    return ""


def _guardrail_tripwire_reply(exc: InputGuardrailTripwireTriggered) -> str:
    guardrail_info = exc.guardrail_result.output.output_info
    if isinstance(guardrail_info, dict) and "message" in guardrail_info:
        return guardrail_info["message"]
    return "Your request could not be processed due to input checks."


async def _outer_support_boundary_result(text: str) -> dict[str, Any]:
    return await evaluate_support_boundary(text)


async def _outer_support_boundary_reply(text: str) -> str | None:
    output_info = await _outer_support_boundary_result(text)
    if output_info.get("tripwire_triggered") and output_info.get("message"):
        return str(output_info["message"])
    return None


def _boundary_reply_from_output(output_info: dict[str, Any]) -> str | None:
    if output_info.get("tripwire_triggered") and output_info.get("message"):
        return str(output_info["message"])
    return None


def _should_stop_for_boundary_output(output_info: dict[str, Any] | None) -> bool:
    if not isinstance(output_info, dict):
        return True
    classification = str(output_info.get("classification") or "").strip()
    if classification == "security_process_boundary":
        return False
    return True


def _render_support_reply(raw_reply: str) -> str:
    return strip_handoff_placeholder(raw_reply)


def _waiting_for_team_reply() -> str:
    return (
        "The team has already been notified and will follow up here. "
        "You can add more details in the meantime and I'll keep them in the ticket for review."
    )


def _should_ack_waiting_for_team(channel_id: int, *, cooldown_seconds: int = 180) -> bool:
    last_reply_at = last_bot_reply_ts_by_channel.get(channel_id)
    if last_reply_at is None:
        return True
    return (datetime.now(timezone.utc) - last_reply_at).total_seconds() >= cooldown_seconds


def _find_team_handoff_notice(
    *,
    chat_id: str,
    message_id: int,
) -> tuple[int | None, TeamHandoffNotice | None]:
    for channel_id, notice in team_handoff_notice_by_channel.items():
        if (
            notice.telegram_chat_id == chat_id
            and notice.telegram_message_id == message_id
        ):
            return channel_id, notice
    return None, None


def _recent_user_messages_for_handoff(
    channel_id: int,
    latest_user_text: str,
    *,
    limit: int = 8,
) -> list[str]:
    history = conversation_threads.get(channel_id, [])
    messages = [
        str(item.get("content") or "").strip()
        for item in history
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    latest_cleaned = latest_user_text.strip()
    if latest_cleaned and (not messages or messages[-1] != latest_cleaned):
        messages.append(latest_cleaned)
    return [message for message in messages if message][-limit:]


def _handoff_known_facts(investigation_job: TicketInvestigationJob) -> list[str]:
    facts: list[str] = []
    if investigation_job.evidence.chain:
        facts.append(f"chain: {investigation_job.evidence.chain}")
    if investigation_job.evidence.wallet:
        facts.append(f"wallet: {investigation_job.evidence.wallet}")
    if investigation_job.evidence.tx_hashes:
        facts.append(
            "tx hashes: " + ", ".join(investigation_job.evidence.tx_hashes[:3])
        )
    if (
        investigation_job.evidence.withdrawal_target_chain
        and investigation_job.evidence.withdrawal_target_vault
    ):
        facts.append(
            "withdrawal target: "
            f"{investigation_job.evidence.withdrawal_target_chain} "
            f"{investigation_job.evidence.withdrawal_target_vault}"
        )
    return facts


async def _notify_handoff(
    *,
    route: HandoffRoute,
    summary: str,
    channel_id: int,
    guild_id: int | None,
    source: Literal["ticket", "public"],
    recent_user_messages: list[str] | None = None,
    known_facts: list[str] | None = None,
) -> TelegramSentMessage | None:
    summarized_text = await summarize_handoff_summary(
        route=route,
        summary=summary,
        recent_user_messages=recent_user_messages,
        known_facts=known_facts,
    )
    sent = await send_handoff_notice(
        build_handoff_notice(
            route,
            summary=summarized_text or summary,
            channel_id=channel_id,
            guild_id=guild_id,
            reply_enabled=source == "ticket",
        ),
        dismiss_enabled=source == "ticket",
    )
    if isinstance(sent, TelegramSentMessage):
        logging.info(
            "Sent Telegram handoff notice for %s channel %s (%s).",
            source,
            channel_id,
            route.target,
        )
        return sent
    return None


def _remember_sent_handoff_notice(
    *,
    channel_id: int,
    route: HandoffRoute,
    notice: TelegramSentMessage | None,
) -> None:
    if notice is None:
        return
    remember_team_handoff_notice(
        channel_id,
        TeamHandoffNotice(
            telegram_chat_id=notice.chat_id,
            telegram_message_id=notice.message_id,
            target=route.target,
            reason=route.reason,
            message_text=notice.message_text,
        ),
    )


async def _notify_and_record_ticket_handoff(
    *,
    route: HandoffRoute,
    summary: str,
    channel_id: int,
    guild_id: int | None,
    investigation_job: TicketInvestigationJob,
) -> bool:
    recent_user_messages = _recent_user_messages_for_handoff(channel_id, summary)
    known_facts = _handoff_known_facts(investigation_job)
    notice = await _notify_handoff(
        route=route,
        summary=summary,
        channel_id=channel_id,
        guild_id=guild_id,
        source="ticket",
        recent_user_messages=recent_user_messages,
        known_facts=known_facts,
    )
    if notice is None:
        investigation_job.mark_waiting_for_user()
        return False
    investigation_job.mark_escalated_to_human()
    _remember_sent_handoff_notice(
        channel_id=channel_id,
        route=route,
        notice=notice,
    )
    return True


def _handoff_delivery_failure_reply(
    base_reply: str | None,
    *,
    location: Literal["ticket", "here"] = "ticket",
) -> str:
    cleaned = strip_handoff_placeholder(base_reply)
    failure_notice = "I couldn't send the internal team notification automatically."
    retry_notice = (
        "This ticket remains active."
        if location == "ticket"
        else "Please try again here later."
    )
    failure_notice = f"{failure_notice} {retry_notice}"
    if not cleaned:
        return failure_notice
    separator = " " if cleaned.endswith((".", "!", "?")) else ". "
    return f"{cleaned}{separator}{failure_notice}"


async def _run_internal_instruction_turn(
    *,
    executor: TransportTicketInvestigationExecutor,
    channel: discord.TextChannel,
    channel_id: int,
    run_context: BotRunContext,
    prompt_text: str,
    instruction_text: str,
    workflow_suffix: str,
    attachments: list[dict[str, Any]],
) -> str:
    investigation_job = deepcopy(get_or_create_ticket_investigation_job(channel_id))
    current_history = list(conversation_threads.get(channel_id, []))
    if not current_history and isinstance(channel, discord.TextChannel):
        current_history = await _build_recent_channel_history_fallback(
            channel,
            exclude_message_id=-1,
        )
    input_list: List[TResponseInputItem] = current_history + [
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": prompt_text},
    ]

    progress_reporter = _DiscordProgressReporter(channel, channel_id)
    try:
        worker_result = await executor.execute_turn(
            _build_turn_request(
                aggregated_text=prompt_text,
                input_list=input_list,
                current_history=current_history,
                attachments=attachments,
                turn_source="internal_team",
                turn_instruction=instruction_text,
                run_context=run_context,
                investigation_job=investigation_job,
                workflow_name=f"{_ticket_workflow_name(run_context)} [{workflow_suffix}]",
                precomputed_boundary=None,
            ),
            hooks=TicketExecutionHooks(
                send_progress_update=progress_reporter.update,
            ),
        )
        conversation_threads[channel_id] = worker_result.flow_outcome.conversation_history
        persist_ticket_state(channel_id)
        return _render_support_reply(worker_result.flow_outcome.raw_final_reply)
    finally:
        await progress_reporter.close()


def _build_turn_request(
    *,
    aggregated_text: str,
    input_list: List[TResponseInputItem],
    current_history: List[TResponseInputItem],
    attachments: list[dict[str, Any]],
    turn_source: str = "user",
    turn_instruction: str | None = None,
    run_context: BotRunContext,
    investigation_job: TicketInvestigationJob,
    workflow_name: str,
    precomputed_boundary: dict[str, Any] | None,
) -> TicketTurnRequest:
    return TicketTurnRequest(
        aggregated_text=aggregated_text,
        input_list=input_list,
        current_history=current_history,
        attachments=attachments,
        turn_source=turn_source,
        turn_instruction=turn_instruction,
        run_context=run_context,
        investigation_job=investigation_job,
        workflow_name=workflow_name,
        precomputed_boundary=precomputed_boundary,
    )


def _public_workflow_name(channel_id: int) -> str:
    return f"Public Stateful Trigger-{channel_id}"


def _ticket_workflow_name(run_context: BotRunContext) -> str:
    return (
        f"Ticket Channel {run_context.channel_id} ({run_context.project_context}, "
        f"Button Intent: {run_context.initial_button_intent})"
    )


def _project_context_for_category(category_id: int | None) -> str:
    if category_id is None:
        return "unknown"
    return config.CATEGORY_CONTEXT_MAP.get(category_id, "unknown")


def _build_public_run_context(
    *,
    channel_id: int,
    conversation_owner_id: int,
    trigger_char_used: str,
) -> BotRunContext:
    return BotRunContext(
        channel_id=channel_id,
        is_public_trigger=True,
        conversation_owner_id=conversation_owner_id,
        project_context=config.TRIGGER_CONTEXT_MAP.get(trigger_char_used, "unknown"),
    )


def _record_button_requested_intent(
    *,
    channel_id: int,
    investigation_job: TicketInvestigationJob,
) -> str | None:
    current_intent = channel_intent_after_button.pop(channel_id, None)
    if current_intent:
        logging.info(
            "Message in %s is a follow-up to button intent: %s",
            channel_id,
            current_intent,
        )
        investigation_job.record_requested_intent(current_intent)
    return current_intent


def _build_ticket_run_context(
    *,
    channel_id: int,
    category_id: int | None,
    initial_button_intent: str | None,
    conversation_owner_id: int | None = None,
) -> BotRunContext:
    return BotRunContext(
        channel_id=channel_id,
        category_id=category_id,
        conversation_owner_id=conversation_owner_id,
        project_context=_project_context_for_category(category_id),
        initial_button_intent=initial_button_intent,
    )


def _is_contributor_member(author: discord.abc.User) -> bool:
    return is_ticket_contributor(author)


def _normalize_contributor_override_prompt(text: str) -> str | None:
    stripped = (text or "").strip()
    prefix = config.TICKET_CONTRIBUTOR_OVERRIDE_PREFIX.strip()
    if not stripped or not prefix:
        return None
    if not stripped.lower().startswith(prefix.lower()):
        return None
    prompt = stripped[len(prefix) :].strip()
    return prompt or None


def _extract_ticket_owner_user_id_from_messages(
    messages: List[discord.Message],
) -> int | None:
    for message in messages:
        if not message.author.bot:
            continue
        for mentioned_user in message.mentions:
            if not mentioned_user.bot:
                return mentioned_user.id
    return None


async def _detect_ticket_owner_user_id(channel: discord.TextChannel) -> int | None:
    try:
        opener_messages = [
            message async for message in channel.history(limit=5, oldest_first=True)
        ]
    except Exception as exc:
        logging.warning(
            "Failed to inspect opener messages for ticket %s: %s",
            channel.id,
            exc,
        )
        return None
    return _extract_ticket_owner_user_id_from_messages(opener_messages)


TicketMessageAction = Literal["process", "ignore", "contributor_override"]


def _classify_ticket_message_action(
    *,
    author: discord.abc.User,
    content: str,
    ticket_owner_user_id: int | None,
    stopped: bool,
) -> TicketMessageAction:
    is_contributor = _is_contributor_member(author)
    if stopped:
        if is_contributor and _normalize_contributor_override_prompt(content):
            return "contributor_override"
        return "ignore"
    if ticket_owner_user_id is None:
        return "ignore" if is_contributor else "process"
    if author.id != ticket_owner_user_id:
        return "ignore"
    return "process"


def _maybe_recover_runtime_stopped_ticket_for_message(
    *,
    channel_id: int,
    author: discord.abc.User,
    ticket_owner_user_id: int | None,
) -> bool:
    if channel_id not in stopped_channels:
        return False
    if ticket_owner_user_id is not None:
        if author.id != ticket_owner_user_id:
            return False
    elif _is_contributor_member(author):
        return False
    recovered = recover_ticket_channel_from_runtime_stop(channel_id)
    if recovered:
        logging.info(
            "Recovered runtime-stopped ticket %s on new owner-side message from %s.",
            channel_id,
            getattr(author, "name", author.id),
        )
    return recovered


async def _build_recent_channel_history_fallback(
    channel: discord.TextChannel,
    *,
    exclude_message_id: int,
    limit: int = 12,
) -> List[TResponseInputItem]:
    history: List[TResponseInputItem] = []
    messages = [
        message
        async for message in channel.history(limit=limit, oldest_first=True)
    ]
    for message in messages:
        if message.id == exclude_message_id:
            continue
        content = _message_text_for_turn(message)
        if not content:
            continue
        role = "assistant" if message.author.bot else "user"
        history.append({"role": role, "content": content})
    return history


# Discord Bot Implementation
class TicketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)
        self._telegram_updates_task: asyncio.Task[None] | None = None
        self._state_cleanup_task: asyncio.Task[None] | None = None
        self._telegram_update_offset: int | None = load_telegram_update_offset()
        local_executor = None
        if "local" in {
            config.TICKET_EXECUTION_ENDPOINT,
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT,
            config.TICKET_EXECUTION_CANARY_ENDPOINT,
            config.TICKET_EXECUTION_SHADOW_ENDPOINT,
        }:
            from ticket_execution.runtime_factory import (
                build_local_ticket_investigation_executor,
            )

            local_executor = build_local_ticket_investigation_executor()
        self.investigation_json_endpoint = build_ticket_execution_json_endpoint(
            local_executor
        )
        self.investigation_transport = JsonEndpointTicketExecutionTransport(
            self.investigation_json_endpoint
        )
        self.investigation_executor = TransportTicketInvestigationExecutor(
            self.investigation_transport
        )
        logging.info(
            "Ticket execution runtime configured: %s",
            config.ticket_execution_runtime_summary(),
        )
        for warning in config.ticket_execution_runtime_warnings():
            logging.warning("Ticket execution rollout warning: %s", warning)

    async def setup_hook(self) -> None:
        self.add_view(InitialInquiryView())
        self.add_view(StopBotView())

    async def on_ready(self):
        logging.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logging.info(f"Monitoring Yearn Ticket Category ID: {config.YEARN_TICKET_CATEGORY_ID}")
        logging.info(f"Support User ID for triggers: {config.PUBLIC_TRIGGER_USER_IDS}")
        logging.info(f"Yearn Public Trigger: '{config.YEARN_PUBLIC_TRIGGER_CHAR}'")
        hydrated_handoffs = hydrate_persisted_team_handoff_states()
        logging.info(
            "Hydrated %s persisted Telegram handoff state(s) before polling.",
            hydrated_handoffs,
        )
        if (
            config.TELEGRAM_BOT_TOKEN
            and config.TELEGRAM_YSUPPORT_CHAT
            and (self._telegram_updates_task is None or self._telegram_updates_task.done())
        ):
            self._telegram_updates_task = asyncio.create_task(
                self._telegram_handoff_reply_loop()
            )
        if self._state_cleanup_task is None or self._state_cleanup_task.done():
            self._state_cleanup_task = asyncio.create_task(
                self._state_cleanup_loop()
            )
        logging.info("Telegram handoff reply loop initialized: %s", bool(self._telegram_updates_task))

    async def close(self) -> None:
        if self._telegram_updates_task is not None:
            self._telegram_updates_task.cancel()
            self._telegram_updates_task = None
        if self._state_cleanup_task is not None:
            self._state_cleanup_task.cancel()
            self._state_cleanup_task = None
        await super().close()

    async def _state_cleanup_loop(self) -> None:
        cleanup_interval_seconds = max(
            60,
            config.PUBLIC_TRIGGER_TIMEOUT_MINUTES * 60,
        )
        while not self.is_closed():
            removed = prune_expired_public_conversations()
            if removed:
                logging.info(
                    "Removed %s expired public conversation state file(s).",
                    removed,
                )
            try:
                removed_codex_sessions = await prune_codex_support_sessions(
                    self.investigation_json_endpoint
                )
            except Exception:
                logging.exception("Failed to prune expired Codex support sessions.")
            else:
                if removed_codex_sessions:
                    logging.info(
                        "Removed %s expired Codex support session(s).",
                        removed_codex_sessions,
                    )
            await asyncio.sleep(cleanup_interval_seconds)

    async def _telegram_handoff_reply_loop(self) -> None:
        while not self.is_closed():
            try:
                await self._resume_pending_telegram_handoff_replies()
                updates = await fetch_telegram_updates(self._telegram_update_offset)
                for update in updates:
                    update_id = update.get("update_id")
                    await self._handle_telegram_handoff_update(update)
                    if isinstance(update_id, int):
                        self._telegram_update_offset = update_id + 1
                        persist_telegram_update_offset(self._telegram_update_offset)
            except asyncio.CancelledError:
                raise
            except TelegramApiError as exc:
                logging.warning(
                    "Telegram polling temporarily unavailable; retrying in 5 seconds: %s",
                    exc,
                )
                await asyncio.sleep(5)
            except Exception as exc:
                logging.error("Telegram handoff reply loop failed: %s", exc, exc_info=True)
                await asyncio.sleep(5)

    async def _resume_pending_telegram_handoff_replies(self) -> None:
        for channel_id, notice in list(team_handoff_notice_by_channel.items()):
            if notice.status == "pending_delivery" and notice.pending_reply_text:
                await self._deliver_telegram_handoff_reply(
                    channel_id=channel_id,
                    notice=notice,
                    team_reply_text=notice.pending_reply_text,
                )
                continue
            if notice.status == "delivered_pending_close":
                await self._finalize_telegram_handoff_notice_close(
                    channel_id=channel_id,
                    notice=notice,
                )

    async def _handle_telegram_handoff_update(self, update: dict[str, Any]) -> None:
        callback_query = update.get("callback_query")
        if isinstance(callback_query, dict):
            await self._handle_telegram_handoff_dismissal(callback_query)
            return

        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = str(message.get("text") or "").strip()
        if not text:
            return
        chat = message.get("chat")
        if not isinstance(chat, dict):
            return
        chat_id = str(chat.get("id") or "").strip()
        if not chat_id or chat_id != config.TELEGRAM_YSUPPORT_CHAT:
            return
        reply_to_message = message.get("reply_to_message")
        if not isinstance(reply_to_message, dict):
            return
        telegram_message_id = message.get("message_id")
        reply_to_message_id = reply_to_message.get("message_id")
        if not isinstance(reply_to_message_id, int):
            return

        matched_channel_id, matched_notice = _find_team_handoff_notice(
            chat_id=chat_id,
            message_id=reply_to_message_id,
        )
        if matched_channel_id is None or matched_notice is None:
            return
        if matched_notice.reply_consumed or matched_notice.status != "open":
            return
        if matched_channel_id in stopped_channels:
            return
        if not is_substantive_team_reply(text):
            await send_telegram_message(
                chat_id=chat_id,
                message_text=build_vague_team_reply_feedback(),
                reply_to_message_id=telegram_message_id if isinstance(telegram_message_id, int) else None,
            )
            return
        if not isinstance(telegram_message_id, int):
            return

        mark_team_handoff_notice_pending_delivery(
            matched_channel_id,
            reply_text=text,
            reply_message_id=telegram_message_id,
        )
        await self._deliver_telegram_handoff_reply(
            channel_id=matched_channel_id,
            notice=matched_notice,
            team_reply_text=text,
        )

    async def _handle_telegram_handoff_dismissal(
        self,
        callback_query: dict[str, Any],
    ) -> None:
        callback_query_id = str(callback_query.get("id") or "").strip()
        if not callback_query_id:
            return
        message = callback_query.get("message")
        chat = message.get("chat") if isinstance(message, dict) else None
        chat_id = (
            str(chat.get("id") or "").strip()
            if isinstance(chat, dict)
            else ""
        )
        message_id = message.get("message_id") if isinstance(message, dict) else None
        if (
            callback_query.get("data") != DISMISS_HANDOFF_CALLBACK_DATA
            or not chat_id
            or chat_id != config.TELEGRAM_YSUPPORT_CHAT
            or not isinstance(message_id, int)
        ):
            await answer_telegram_callback_query(
                callback_query_id=callback_query_id,
                message_text="This action is no longer available.",
            )
            return

        matched_channel_id, matched_notice = _find_team_handoff_notice(
            chat_id=chat_id,
            message_id=message_id,
        )

        if (
            matched_channel_id is None
            or matched_notice is None
            or matched_notice.reply_consumed
            or matched_notice.status != "open"
        ):
            await answer_telegram_callback_query(
                callback_query_id=callback_query_id,
                message_text="This handoff is already closed.",
            )
            return

        stopped_durably = stop_ticket_channel(matched_channel_id)
        task = pending_tasks.pop(matched_channel_id, None)
        if task is not None:
            task.cancel()

        if not stopped_durably:
            team_handoff_notice_by_channel[matched_channel_id] = matched_notice
            logging.error(
                "Stopped Telegram handoff for channel %s in memory, but durable cleanup failed.",
                matched_channel_id,
            )
            await answer_telegram_callback_query(
                callback_query_id=callback_query_id,
                message_text=(
                    "The handoff could not be dismissed safely. "
                    "Please try again, or use Stop Bot in Discord."
                ),
            )
            return

        await answer_telegram_callback_query(
            callback_query_id=callback_query_id,
            message_text="Handoff dismissed. Handle this ticket in Discord.",
        )
        deleted = await delete_telegram_message(
            chat_id=chat_id,
            message_id=message_id,
        )
        if deleted:
            return

        edited = await edit_handoff_notice(
            chat_id=chat_id,
            message_id=message_id,
            message_text=build_dismissed_handoff_notice(
                matched_notice.message_text or "Dismissed. Handle in Discord."
            ),
        )
        if not edited:
            team_handoff_notice_by_channel[matched_channel_id] = matched_notice
            retry_persisted = persist_ticket_state(matched_channel_id)
            logging.warning(
                "Dismissed Telegram handoff for channel %s, but could not remove "
                "or edit notice %s; cleanup retry persisted=%s.",
                matched_channel_id,
                message_id,
                retry_persisted,
            )

    async def _deliver_telegram_handoff_reply(
        self,
        *,
        channel_id: int,
        notice: TeamHandoffNotice,
        team_reply_text: str,
    ) -> bool:
        if channel_id in stopped_channels:
            return False
        if notice.status == "pending_delivery":
            await edit_handoff_notice(
                chat_id=notice.telegram_chat_id,
                message_id=notice.telegram_message_id,
                message_text=build_pending_delivery_handoff_notice(
                    notice.message_text or "Reply received. Delivering update..."
                ),
            )
            if channel_id in stopped_channels:
                return False

        channel = self.get_channel(channel_id)
        if channel is None or not hasattr(channel, "send") or not hasattr(channel, "typing"):
            logging.warning(
                "Could not resolve Discord channel %s for Telegram handoff reply.",
                channel_id,
            )
            return False

        ticket_owner_user_id = ticket_owner_user_id_by_channel.get(channel_id)
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        current_intent = channel_intent_after_button.get(channel_id)
        run_context = _build_ticket_run_context(
            channel_id=channel_id,
            category_id=channel.category.id if channel.category else None,
            initial_button_intent=current_intent,
            conversation_owner_id=ticket_owner_user_id,
        )
        instruction_text = (
            "This input is from the internal team, not from the user. "
            "The user asked for help earlier and the team is now telling you what the next user-facing update should communicate. "
            "Use the team message plus the ticket context, including any user follow-up details that arrived while waiting, to draft the next Discord update for the user. "
            "Write directly to the user, not back to the team. "
            "Do not say thanks to the team or acknowledge the internal sender conversationally. "
            "Translate internal shorthand into clear user-facing language. "
            "If the team reports current status or action taken, lead with that status update. "
            "Expand shorthand like `pending sigs` into normal user-facing wording like `pending signatures`. "
            "Do not mention Telegram, internal notes, or handoff mechanics. "
            "Do not make stronger claims than the team message supports."
        )
        try:
            async with channel.typing():
                try:
                    attachments = await _refresh_discord_attachment_urls(
                        channel,
                        list(notice.followup_attachments),
                    )
                    if channel_id in stopped_channels:
                        return False
                    final_reply = await _run_internal_instruction_turn(
                        executor=self.investigation_executor,
                        channel=channel,
                        channel_id=channel_id,
                        run_context=run_context,
                        prompt_text=team_reply_text,
                        instruction_text=instruction_text,
                        workflow_suffix="team handoff reply",
                        attachments=attachments,
                    )
                    if channel_id in stopped_channels:
                        reset_ticket_codex_session(channel_id)
                        return False
                except Exception as exc:
                    logging.error(
                        "Failed to synthesize Telegram handoff reply for channel %s: %s",
                        channel_id,
                        exc,
                        exc_info=True,
                    )
                    return False
                await send_long_message(channel, final_reply)
        except Exception as exc:
            logging.error(
                "Failed to deliver Telegram handoff reply for channel %s: %s",
                channel_id,
                exc,
                exc_info=True,
            )
            return False

        investigation_job.mark_waiting_for_user()
        last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
        mark_team_handoff_notice_delivered(channel_id)
        if not await self._finalize_telegram_handoff_notice_close(
            channel_id=channel_id,
            notice=notice,
        ):
            return False
        persist_ticket_state(channel_id)
        return True

    async def _finalize_telegram_handoff_notice_close(
        self,
        *,
        channel_id: int,
        notice: TeamHandoffNotice,
    ) -> bool:
        edited = await edit_handoff_notice(
            chat_id=notice.telegram_chat_id,
            message_id=notice.telegram_message_id,
            message_text=build_closed_handoff_notice(
                notice.message_text or "Reply received. Replies closed."
            ),
        )
        if not edited:
            logging.warning(
                "Failed to close Telegram handoff notice for channel %s after Discord delivery.",
                channel_id,
            )
            return False
        clear_team_handoff_notice(channel_id)
        return True

    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
        if isinstance(channel, discord.TextChannel) and channel.category:
            if channel.category.id in config.CATEGORY_CONTEXT_MAP:
                project_context = config.CATEGORY_CONTEXT_MAP.get(channel.category.id, "unknown")
                logging.info(f"New {project_context.capitalize()} ticket channel created: {channel.name} (ID: {channel.id}). Initializing state.")
                conversation_threads[channel.id] = []
                clear_ticket_channel_state(
                    channel.id,
                    keep_stopped=False,
                    delete_persisted=True,
                )
                if channel.id in pending_tasks:
                    try:
                        pending_tasks.pop(channel.id).cancel()
                    except Exception:
                        pass
                monitored_new_channels.add(channel.id)
                logging.info(f"Added channel {channel.id} to monitored_new_channels set.")

                delay_seconds = 1.5
                logging.info(f"Delaying welcome message in {channel.id} by {delay_seconds} seconds.")
                await asyncio.sleep(delay_seconds)
                ticket_owner_user_id = await _detect_ticket_owner_user_id(channel)
                if ticket_owner_user_id is not None:
                    remember_ticket_owner_user_id(channel.id, ticket_owner_user_id)
                    logging.info(
                        "Detected ticket owner %s for channel %s from opener message.",
                        ticket_owner_user_id,
                        channel.id,
                    )
                else:
                    logging.info(
                        "Could not detect ticket owner from opener messages for channel %s.",
                        channel.id,
                    )

                welcome_message = (
                    f"Welcome to {project_context.capitalize()} Support!\n\n\n"
                    "Press a category button below to get started.\n"
                    "You can share more details after making a selection.\n\n\n"
                    "To process your request accurately, please wait for my response after you see the *'ySupport is typing...'* indicator before sending another message.\n\n"
                    "---\n"
                    "**IGNORE FRIEND REQUESTS**\n"
                    "**DO NOT RESPOND TO DMS**\n\n"
                    "**WE WILL NEVER ADD OR DM YOU**"
                )
                try:
                    await channel.send(welcome_message, view=InitialInquiryView(), suppress_embeds=True)
                    last_bot_reply_ts_by_channel[channel.id] = datetime.now(timezone.utc)
                    mark_ticket_awaiting_initial_button(channel.id)
                    logging.info(f"Sent initial inquiry buttons to channel {channel.id}")
                except discord.Forbidden:
                    logging.error(f"Missing permissions to send initial message with buttons in {channel.id}")
                except Exception as e:
                    logging.error(f"Error sending initial message with buttons in {channel.id}: {e}", exc_info=True)

    async def process_synthetic_button_input(self, channel: discord.TextChannel, synthetic_text: str, intent_category: str):
        """
        Processes a synthetic input generated from an initial button press.
        This will queue up a task similar to process_ticket_message but with predefined input.
        """
        channel_id = channel.id
        logging.info(f"Processing synthetic button input for ticket {channel_id}. Intent: '{intent_category}', Text: '{synthetic_text}'")

        category_id = channel.category.id if channel.category else None
        run_context = _build_ticket_run_context(
            channel_id=channel_id,
            category_id=category_id,
            initial_button_intent=intent_category,
        )

        if channel_id in pending_tasks:
            pending_tasks[channel_id].cancel()

        await self.process_ticket_message(channel_id, run_context, is_button_trigger=True, synthetic_user_message_for_log=synthetic_text)

    async def _handle_public_trigger_message(
        self,
        message: discord.Message,
        trigger_char_used: str,
    ) -> bool:
        original_message: discord.Message | None = None
        original_author_id: int | None = None
        logging.info(
            "Stateful public trigger '%s' detected by %s in channel %s",
            trigger_char_used,
            message.author.name,
            message.channel.id,
        )

        try:
            await message.delete()
        except Exception as e:
            logging.warning(f"Failed to delete trigger message {message.id}: {e}")

        try:
            original_message = await message.channel.fetch_message(message.reference.message_id)
            if not original_message or original_message.author.bot:
                return True
            original_message_text = _message_text_for_turn(original_message)
            if not original_message_text:
                return True

            original_author_id = original_message.author.id
            current_history: List[TResponseInputItem] = []
            public_investigation_job = None
            hydrate_public_conversation(original_author_id)

            conversation = public_conversations.get(original_author_id)
            if conversation:
                time_since_last = datetime.now(timezone.utc) - conversation.last_interaction_time
                if time_since_last <= timedelta(minutes=config.PUBLIC_TRIGGER_TIMEOUT_MINUTES):
                    logging.info(
                        "Continuing public conversation for user %s (last active %.1fs ago).",
                        original_author_id,
                        time_since_last.total_seconds(),
                    )
                    current_history = conversation.history
                    public_investigation_job = conversation.investigation_job
                else:
                    logging.info(
                        "Public conversation for user %s expired (%.1fs ago). Starting new context.",
                        original_author_id,
                        time_since_last.total_seconds(),
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)

            if public_investigation_job is None:
                public_investigation_job = TicketInvestigationJob(
                    channel_id=message.channel.id
                )
            public_run_context = _build_public_run_context(
                channel_id=message.channel.id,
                conversation_owner_id=original_author_id,
                trigger_char_used=trigger_char_used,
            )

            boundary_output = await _outer_support_boundary_result(original_message_text)
            boundary_reply = _boundary_reply_from_output(boundary_output)
            if boundary_reply is not None:
                public_conversations.pop(original_author_id, None)
                clear_public_conversation(original_author_id)
                await original_message.reply(
                    boundary_reply,
                    mention_author=False,
                    suppress_embeds=True,
                )
                return True

            async with message.channel.typing():
                progress_reporter = _DiscordProgressReporter(
                    message.channel,
                    message.channel.id,
                )
                try:
                    worker_result = await self.investigation_executor.execute_turn(
                        _build_turn_request(
                            aggregated_text=original_message_text,
                            input_list=current_history + [
                                {"role": "user", "content": original_message_text}
                            ],
                            current_history=current_history,
                            attachments=_attachment_payloads_from_message(original_message),
                            run_context=public_run_context,
                            investigation_job=public_investigation_job,
                            workflow_name=_public_workflow_name(message.channel.id),
                            precomputed_boundary=boundary_output,
                        ),
                        hooks=TicketExecutionHooks(
                            send_progress_update=progress_reporter.update,
                        ),
                    )
                    flow_outcome = worker_result.flow_outcome
                    new_history = flow_outcome.conversation_history
                    public_conversations[original_author_id] = PublicConversation(
                        history=new_history,
                        last_interaction_time=datetime.now(timezone.utc),
                        investigation_job=worker_result.updated_job,
                    )
                    persist_public_conversation(original_author_id)
                    logging.info(
                        "Saved updated public conversation context for user %s. History length: %s items.",
                        original_author_id,
                        len(new_history),
                    )

                    raw_reply = (
                        flow_outcome.raw_final_reply
                        if flow_outcome.raw_final_reply
                        else "I could not determine a response."
                    )
                    if flow_outcome.requires_human_handoff:
                        route = infer_handoff_route(
                            original_message_text,
                            raw_reply,
                        )
                        notice = await _notify_handoff(
                            route=route,
                            summary=original_message_text,
                            channel_id=message.channel.id,
                            guild_id=getattr(getattr(message.channel, "guild", None), "id", None),
                            source="public",
                        )
                        if notice is not None:
                            final_reply = build_user_handoff_reply(
                                raw_reply,
                                route,
                                location="here",
                            )
                            public_conversations.pop(original_author_id, None)
                            clear_public_conversation(original_author_id)
                        else:
                            final_reply = _handoff_delivery_failure_reply(
                                raw_reply,
                                location="here",
                            )
                    else:
                        final_reply = _render_support_reply(raw_reply)
                    await progress_reporter.close()
                    await send_long_message(message.channel, final_reply)
                except InputGuardrailTripwireTriggered as e:
                    await progress_reporter.close()
                    logging.warning(
                        "Input Guardrail triggered in public channel %s for user %s.",
                        message.channel.id,
                        original_author_id,
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)
                    await original_message.reply(
                        _guardrail_tripwire_reply(e),
                        mention_author=False,
                        suppress_embeds=True,
                    )
                except MaxTurnsExceeded:
                    await progress_reporter.close()
                    logging.warning(
                        "Max turns (%s) exceeded during public trigger run for user %s in channel %s.",
                        config.MAX_PUBLIC_TRIGGER_TURNS,
                        original_author_id,
                        message.channel.id,
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)
                    if public_run_context.repo_search_calls:
                        base_reply = (
                            "I hit an internal analysis limit while reviewing repo evidence for that request."
                        )
                    else:
                        base_reply = (
                            "I hit an internal analysis limit while working on that request."
                        )
                    await original_message.reply(
                        f"{base_reply} Please try again.",
                        mention_author=False,
                        suppress_embeds=True,
                    )
                except Exception as e:
                    await progress_reporter.close()
                    logging.error(
                        "Error during public trigger agent run for user %s: %s",
                        original_author_id,
                        e,
                        exc_info=True,
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)
                    await original_message.reply(
                        "Sorry, an error occurred while processing that request. Please try again.",
                        mention_author=False,
                        suppress_embeds=True,
                    )
            return True
        except discord.NotFound:
            logging.warning(f"Original message for public trigger reply {message.id} not found.")
            return True
        except discord.Forbidden:
            logging.warning(f"Missing permissions to fetch original message for public trigger reply {message.id}.")
            return True
        except Exception as e:
            logging.error(f"Error handling public trigger for message {message.id}: {e}", exc_info=True)
            if original_author_id is not None:
                public_conversations.pop(original_author_id, None)
                clear_public_conversation(original_author_id)
            if original_message is not None:
                try:
                    await original_message.reply(
                        "Sorry, an error occurred while preparing that request. Please try again.",
                        mention_author=False,
                        suppress_embeds=True,
                    )
                except Exception:
                    logging.warning(
                        "Failed to send public trigger setup error reply for message %s.",
                        message.id,
                        exc_info=True,
                    )
            return True

    async def _collect_aggregated_ticket_payload(
        self,
        channel_id: int,
        run_context: BotRunContext,
        *,
        is_button_trigger: bool,
        synthetic_user_message_for_log: str,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        if is_button_trigger:
            return synthetic_user_message_for_log.strip(), []

        debounce_seconds = _ticket_debounce_seconds(channel_id, run_context)
        try:
            await asyncio.sleep(debounce_seconds)
        except asyncio.CancelledError:
            logging.debug(f"Processing task for channel {channel_id} cancelled (new message arrived).")
            return None, []
        aggregated_text = pending_messages.pop(channel_id, None)
        attachments = pending_attachments_by_channel.pop(channel_id, [])
        current_task = asyncio.current_task()
        if aggregated_text and current_task is not None:
            active_ticket_payloads[channel_id] = (
                current_task,
                aggregated_text,
                attachments,
            )
        return aggregated_text, attachments

    async def _handle_stopped_ticket_contributor_override(
        self,
        message: discord.Message,
        run_context: BotRunContext,
        prompt_text: str,
    ) -> None:
        channel_id = message.channel.id
        investigation_job = deepcopy(get_or_create_ticket_investigation_job(channel_id))
        current_history = list(conversation_threads.get(channel_id, []))
        if not current_history and isinstance(message.channel, discord.TextChannel):
            current_history = await _build_recent_channel_history_fallback(
                message.channel,
                exclude_message_id=message.id,
            )
        input_list: List[TResponseInputItem] = current_history + [
            {
                "role": "system",
                "content": (
                    "This is a contributor override turn in a stopped ticket. "
                    "Answer the contributor request using the existing ticket context, "
                    "but do not treat this as a normal user turn or a full bot resume."
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

        logging.info(
            "Processing contributor override in stopped ticket %s from %s.",
            channel_id,
            message.author.name,
        )

        async with message.channel.typing():
            progress_reporter = _DiscordProgressReporter(message.channel, channel_id)
            try:
                worker_result = await self.investigation_executor.execute_turn(
                    _build_turn_request(
                        aggregated_text=prompt_text,
                        input_list=input_list,
                        current_history=current_history,
                        attachments=_attachment_payloads_from_message(message),
                        run_context=run_context,
                        investigation_job=investigation_job,
                        workflow_name=f"{_ticket_workflow_name(run_context)} [contributor override]",
                        precomputed_boundary=None,
                    ),
                    hooks=TicketExecutionHooks(
                        send_progress_update=progress_reporter.update,
                    ),
                )
                final_reply = _render_support_reply(
                    worker_result.flow_outcome.raw_final_reply
                )
            except Exception as exc:
                logging.error(
                    "Contributor override failed in stopped ticket %s: %s",
                    channel_id,
                    exc,
                    exc_info=True,
                )
                final_reply = (
                    "I couldn't complete that contributor override. "
                    "The ticket remains stopped so the team can handle it directly or retry the override."
                )
            finally:
                await progress_reporter.close()

        await send_long_message(message.channel, final_reply)
        last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
        persist_ticket_state(channel_id)

    async def _build_ticket_turn_input(
        self,
        *,
        channel: discord.TextChannel,
        channel_id: int,
        run_context: BotRunContext,
        investigation_job,
        aggregated_text: str,
    ) -> tuple[List[TResponseInputItem], List[TResponseInputItem]]:
        preparation = await prepare_ticket_turn_input(
            channel_id=channel_id,
            run_context=run_context,
            investigation_job=investigation_job,
            aggregated_text=aggregated_text,
        )
        for ack_message in preparation.ack_messages:
            try:
                await channel.send(ack_message, suppress_embeds=True)
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                logging.info(
                    "Sent ticket intake acknowledgement message to channel %s",
                    channel_id,
                )
            except Exception as exc:
                logging.warning(
                    "Failed to send ticket intake acknowledgement message: %s",
                    exc,
                )

        current_history = conversation_threads.get(channel_id, [])
        input_list: List[TResponseInputItem] = current_history + [
            {"role": "user", "content": preparation.current_user_content}
        ]

        if preparation.system_hints:
            input_list = input_list[:-1] + [
                {"role": "system", "content": " ".join(preparation.system_hints)}
            ] + [input_list[-1]]

        contextual_hints = build_contextual_hints(
            investigation_job,
            aggregated_text,
            current_history=current_history,
        )
        if contextual_hints:
            input_list = input_list[:-1] + [{"role": "system", "content": " ".join(contextual_hints)}] + [input_list[-1]]

        if is_bug_report_query(aggregated_text):
            last_wallet = last_wallet_by_channel.get(channel_id)
            if last_wallet and last_wallet not in aggregated_text:
                input_list = input_list[:-1] + [{
                    "role": "system",
                    "content": f"Context: User previously provided wallet address {last_wallet} for this ticket. Use it if needed; do not ask again unless they want a different address."
                }] + [input_list[-1]]

        return current_history, input_list

    async def on_message(self, message: discord.Message):
        if message.author.bot or (self.user is not None and message.author.id == self.user.id):
            return

        is_reply = message.reference is not None
        trigger_char_used = message.content.strip()
        is_valid_trigger_char = trigger_char_used in config.TRIGGER_CONTEXT_MAP
        is_trigger_user = str(message.author.id) in config.PUBLIC_TRIGGER_USER_IDS

        if is_reply and is_trigger_user and is_valid_trigger_char:
            await self._handle_public_trigger_message(message, trigger_char_used)
            return

        if not isinstance(message.channel, discord.TextChannel) or not message.channel.category:
            return
        channel_id = message.channel.id
        if message.channel.category.id not in config.CATEGORY_CONTEXT_MAP:
            return
        hydrate_ticket_state(channel_id)
        monitored_new_channels.add(channel_id)
        ticket_owner_user_id = ticket_owner_user_id_by_channel.get(channel_id)
        if ticket_owner_user_id is None and isinstance(message.channel, discord.TextChannel):
            ticket_owner_user_id = await _detect_ticket_owner_user_id(message.channel)
            if ticket_owner_user_id is not None:
                remember_ticket_owner_user_id(channel_id, ticket_owner_user_id)
                logging.info(
                    "Recovered ticket owner %s for channel %s from opener history during message processing.",
                    ticket_owner_user_id,
                    channel_id,
                )
        _maybe_recover_runtime_stopped_ticket_for_message(
            channel_id=channel_id,
            author=message.author,
            ticket_owner_user_id=ticket_owner_user_id,
        )
        action = _classify_ticket_message_action(
            author=message.author,
            content=message.content,
            ticket_owner_user_id=ticket_owner_user_id,
            stopped=channel_id in stopped_channels,
        )

        if action == "ignore":
            logging.info(
                "Ignoring ticket message in %s from %s. owner=%s stopped=%s contributor=%s",
                channel_id,
                message.author.name,
                ticket_owner_user_id,
                channel_id in stopped_channels,
                _is_contributor_member(message.author),
            )
            return

        if ticket_owner_user_id is None and not _is_contributor_member(message.author):
            ticket_owner_user_id = message.author.id
            remember_ticket_owner_user_id(channel_id, ticket_owner_user_id)
            logging.info(
                "Assigned fallback ticket owner %s for channel %s from first non-contributor message.",
                ticket_owner_user_id,
                channel_id,
            )

        if channel_id in channels_awaiting_initial_button_press:
            try:
                await message.reply(
                    "Please select an option from the buttons on my previous message to get started.",
                    delete_after=20,
                    mention_author=False,
                    suppress_embeds=True,
                )
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                persist_ticket_state(channel_id)
            except Exception:
                pass
            return

        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        if action == "process" and is_ticket_waiting_for_team(channel_id):
            _record_waiting_for_team_followup(channel_id, message)
            if _should_ack_waiting_for_team(channel_id):
                try:
                    await message.channel.send(
                        _waiting_for_team_reply(),
                        suppress_embeds=True,
                    )
                    last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                    persist_ticket_state(channel_id)
                except Exception:
                    pass
            return
        current_intent_from_map = _record_button_requested_intent(
            channel_id=channel_id,
            investigation_job=investigation_job,
        )
        ticket_run_context = _build_ticket_run_context(
            channel_id=channel_id,
            category_id=message.channel.category.id,
            initial_button_intent=current_intent_from_map,
            conversation_owner_id=ticket_owner_user_id,
        )

        if ticket_run_context.project_context == "unknown":
            return
        if action == "contributor_override":
            override_prompt = _normalize_contributor_override_prompt(message.content)
            if override_prompt is None:
                return
            await self._handle_stopped_ticket_contributor_override(
                message,
                ticket_run_context,
                override_prompt,
            )
            return

        logging.info(f"Processing ticket message in {channel_id} from {message.author.name} (Context: {ticket_run_context.project_context}, Intent: {current_intent_from_map})")

        existing_task = pending_tasks.get(channel_id)
        if existing_task and not existing_task.done():
            _restore_active_ticket_payload(channel_id)
            existing_task.cancel()
            try:
                await message.channel.send(
                    "Got it. I’ve added your follow-up to your previous request for context, "
                    "and I’m continuing to work on it now. Please wait for my response. "
                    "There’s no need to resend anything.",
                    suppress_embeds=True,
                )
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
            except Exception:
                pass

        normalized_message_text = _message_text_for_turn(message)
        _merge_pending_ticket_payload(
            channel_id,
            normalized_message_text,
            _attachment_payloads_from_message(message),
        )

        pending_tasks[channel_id] = asyncio.create_task(self.process_ticket_message(channel_id, ticket_run_context))
        debounce_seconds = _ticket_debounce_seconds(channel_id, ticket_run_context)
        logging.debug(f"Scheduled processing task for channel {channel_id} in {debounce_seconds}s")

    async def process_ticket_message(self, channel_id: int, run_context: BotRunContext, is_button_trigger: bool = False, synthetic_user_message_for_log: str = ""):
        current_task = asyncio.current_task()
        try:
            investigation_job = get_or_create_ticket_investigation_job(channel_id)
            aggregated_text, attachments = await self._collect_aggregated_ticket_payload(
                channel_id,
                run_context,
                is_button_trigger=is_button_trigger,
                synthetic_user_message_for_log=synthetic_user_message_for_log,
            )
            if not aggregated_text:
                return
            merge_explicit_evidence(investigation_job, aggregated_text)

            channel = self.get_channel(channel_id)
            if not isinstance(channel, discord.TextChannel):
                return
            guild_id = getattr(getattr(channel, "guild", None), "id", None)

            current_history, input_list = await self._build_ticket_turn_input(
                channel=channel,
                channel_id=channel_id,
                run_context=run_context,
                investigation_job=investigation_job,
                aggregated_text=aggregated_text,
            )

            logging.info(f"Processing for ticket {channel_id} (Context: {run_context.project_context}, Initial Button Intent: {run_context.initial_button_intent}): '{aggregated_text[:100]}...'")

            async with channel.typing():
                progress_reporter = _DiscordProgressReporter(channel, channel_id)
                final_reply = "An unexpected error occurred."
                should_stop_processing = False
                stop_reason = None

                boundary_output = await _outer_support_boundary_result(aggregated_text)
                boundary_reply = _boundary_reply_from_output(boundary_output)
                if boundary_reply is not None:
                    final_reply = boundary_reply
                    should_stop_processing = _should_stop_for_boundary_output(
                        boundary_output
                    )
                    if should_stop_processing:
                        stop_reason = "boundary_stop"
                        reset_ticket_channel_for_terminal_reply(channel_id)
                else:
                    try:
                        worker_result = await self.investigation_executor.execute_turn(
                            _build_turn_request(
                                aggregated_text=aggregated_text,
                                input_list=input_list,
                                current_history=current_history,
                                attachments=attachments,
                                run_context=run_context,
                                investigation_job=investigation_job,
                                workflow_name=_ticket_workflow_name(run_context),
                                precomputed_boundary=boundary_output,
                            ),
                            hooks=TicketExecutionHooks(
                                send_progress_update=progress_reporter.update,
                            ),
                        )
                        investigation_job.apply_snapshot(worker_result.updated_job)
                        flow_outcome = worker_result.flow_outcome
                        conversation_threads[channel_id] = flow_outcome.conversation_history

                        raw_final_reply = flow_outcome.raw_final_reply
                        if flow_outcome.requires_human_handoff:
                            route = infer_handoff_route(
                                aggregated_text,
                                raw_final_reply,
                            )
                            handoff_sent = await _notify_and_record_ticket_handoff(
                                route=route,
                                summary=aggregated_text,
                                channel_id=channel_id,
                                guild_id=guild_id,
                                investigation_job=investigation_job,
                            )
                            final_reply = (
                                build_user_handoff_reply(raw_final_reply, route)
                                if handoff_sent
                                else _handoff_delivery_failure_reply(raw_final_reply)
                            )
                        else:
                            final_reply = _render_support_reply(raw_final_reply)
                        if flow_outcome.requires_human_handoff:
                            logging.info(
                                "Human handoff tag detected in response for channel %s. Leaving channel active for follow-up.",
                                channel_id,
                            )
                        persist_ticket_state(channel_id)
                    except InputGuardrailTripwireTriggered as e:
                        logging.warning(f"Input Guardrail triggered in channel {channel_id}. Extracting message from output_info.")
                        final_reply = _guardrail_tripwire_reply(e)
                        guardrail_info = getattr(
                            getattr(getattr(e, "guardrail_result", None), "output", None),
                            "output_info",
                            None,
                        )
                        should_stop_processing = _should_stop_for_boundary_output(
                            guardrail_info if isinstance(guardrail_info, dict) else None
                        )
                        if should_stop_processing:
                            stop_reason = "boundary_stop"
                            reset_ticket_channel_for_terminal_reply(channel_id)
                    except MaxTurnsExceeded:
                        logging.warning(f"Max turns ({config.MAX_TICKET_CONVERSATION_TURNS}) exceeded in channel {channel_id}.")
                        if run_context.repo_search_calls:
                            final_reply = (
                                "I hit an internal analysis limit while reviewing repo evidence for this report."
                            )
                        else:
                            final_reply = (
                                "This conversation has reached its maximum length."
                            )
                        should_stop_processing = True
                        stop_reason = "runtime_error"
                    except AgentsException as e:
                        logging.error(f"Agent SDK error during ticket processing for channel {channel_id}: {e}")
                        final_reply = (
                            f"Sorry, an error occurred while processing the request ({type(e).__name__}). Please try again."
                        )
                        should_stop_processing = True
                        stop_reason = "runtime_error"
                    except Exception as e:
                        logging.error(f"Unexpected error during ticket processing for channel {channel_id}: {e}", exc_info=True)
                        final_reply = "An unexpected error occurred."
                        should_stop_processing = True
                        stop_reason = "runtime_error"
                    finally:
                        await progress_reporter.close()

                try:
                    reply_view = StopBotView() if not should_stop_processing else None
                    await send_long_message(channel, final_reply, view=reply_view)
                    if channel_id in bug_report_debounce_channels:
                        bug_report_debounce_channels.discard(channel_id)
                        logging.info("Cleared bug-report debounce flag for channel %s", channel_id)
                    last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                    logging.info(f"Sent ticket reply/replies in channel {channel_id}. Stop processing flag: {should_stop_processing}")
                    if should_stop_processing and channel_id not in stopped_channels:
                        mark_ticket_channel_stopped(
                            channel_id,
                            reason=stop_reason or "runtime_error",
                        )
                        logging.info(f"Added channel {channel_id} to stopped channels due to error/handoff tag.")
                    elif not should_stop_processing:
                        persist_ticket_state(channel_id)
                except discord.Forbidden:
                    logging.error(f"Missing permissions to send message in channel {channel_id}")
                    mark_ticket_channel_stopped(
                        channel_id,
                        reason="runtime_error",
                    )
                except Exception as e:
                    logging.error(f"Unexpected error occurred during or after calling send_long_message for channel {channel_id}: {e}", exc_info=True)
        except asyncio.CancelledError:
            logging.info(f"Processing task for channel {channel_id} cancelled mid-run.")
            return
        finally:
            active_payload = active_ticket_payloads.get(channel_id)
            if active_payload is not None and active_payload[0] is current_task:
                active_ticket_payloads.pop(channel_id, None)
            if pending_tasks.get(channel_id) is current_task:
                pending_tasks.pop(channel_id, None)

    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel):
        if not isinstance(channel, discord.TextChannel):
            return
        notice = team_handoff_notice_by_channel.get(channel.id)
        if notice is not None:
            try:
                await edit_handoff_notice(
                    chat_id=notice.telegram_chat_id,
                    message_id=notice.telegram_message_id,
                    message_text=build_archived_handoff_notice(
                        notice.message_text or "Ticket closed. Replies disabled."
                    ),
                )
            except Exception:
                logging.warning(
                    "Failed to archive Telegram handoff notice for deleted channel %s.",
                    channel.id,
                    exc_info=True,
                )
        clear_ticket_channel_state(channel.id, keep_stopped=False, delete_persisted=True)


def _build_discord_intents() -> discord.Intents:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    return intents


def _reload_runtime_env_and_config() -> None:
    load_dotenv(config.BASE_DIR / ".env", override=True)
    importlib.reload(config)


def _run_ticket_bot_once() -> None:
    config.validate_runtime_environment_config()
    config.validate_ticket_execution_runtime_config()
    client = TicketBot(intents=_build_discord_intents())
    client.run(config.DISCORD_BOT_TOKEN)


def _run_ticket_bot_with_fatal_startup_backoff() -> None:
    while True:
        _reload_runtime_env_and_config()
        try:
            _run_ticket_bot_once()
            return
        except (
            discord.errors.LoginFailure,
            discord.errors.PrivilegedIntentsRequired,
        ):
            backoff_seconds = max(
                float(config.DISCORD_FATAL_STARTUP_BACKOFF_SECONDS or 0.0),
                60.0,
            )
            logging.critical(
                "Fatal Discord startup failure. Backing off for %.0f seconds before retrying.",
                backoff_seconds,
                exc_info=True,
            )
            time.sleep(backoff_seconds)


# Run the Bot
if __name__ == "__main__":
    _run_ticket_bot_with_fatal_startup_backoff()
