from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, List, Literal

import discord
from agents import (
    InputGuardrailTripwireTriggered,
    TResponseInputItem,
)

import config
from handoff import (
    TelegramSentMessage,
    build_handoff_notice,
    send_handoff_notice,
    strip_handoff_placeholder,
    summarize_handoff_summary,
)
from state import (
    active_ticket_executor_tasks,
    active_ticket_payloads,
    BotRunContext,
    bug_report_debounce_channels,
    channel_intent_after_button,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    last_bot_reply_ts_by_channel,
    pending_attachments_by_channel,
    pending_messages,
    persist_ticket_state,
    recover_ticket_channel_from_runtime_stop,
    remember_team_handoff_followup_attachments,
    remember_team_handoff_notice,
    stopped_channels,
    TeamHandoffNotice,
    team_handoff_notice_by_channel,
    TicketInvestigationJob,
)
from support_boundary import evaluate_support_boundary
from ticket_investigation.contracts import TicketTurnRequest
from ticket_investigation.executor import (
    TicketExecutionHooks,
    TransportTicketInvestigationExecutor,
)
from views import is_ticket_support_staff


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
            flush_task = self._flush_task
            self._flush_task = None
            flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await flush_task
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
    if (
        run_context.initial_button_intent == "bug_report"
        or channel_id in bug_report_debounce_channels
    ):
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
        if (
            not isinstance(source_message_id, int)
            or attachment.get("attachment_id") is None
        ):
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
    pending_messages[channel_id] = f"{existing_text}\n{text}" if existing_text else text
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


def _discard_pending_ticket_payload(channel_id: int) -> None:
    pending_messages.pop(channel_id, None)
    pending_attachments_by_channel.pop(channel_id, None)
    active_ticket_payloads.pop(channel_id, None)


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


def _should_ack_waiting_for_team(
    channel_id: int, *, cooldown_seconds: int = 180
) -> bool:
    last_reply_at = last_bot_reply_ts_by_channel.get(channel_id)
    if last_reply_at is None:
        return True
    return (
        datetime.now(timezone.utc) - last_reply_at
    ).total_seconds() >= cooldown_seconds


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
    reason: str,
    summary: str,
    channel_id: int,
    guild_id: int | None,
    source: Literal["ticket", "public"],
    recent_user_messages: list[str] | None = None,
    known_facts: list[str] | None = None,
) -> TelegramSentMessage | None:
    summarized_text = await summarize_handoff_summary(
        reason=reason,
        summary=summary,
        recent_user_messages=recent_user_messages,
        known_facts=known_facts,
    )
    sent = await send_handoff_notice(
        build_handoff_notice(
            reason=reason,
            summary=summarized_text or summary,
            channel_id=channel_id,
            guild_id=guild_id,
            reply_enabled=source == "ticket",
        ),
        dismiss_enabled=source == "ticket",
    )
    if isinstance(sent, TelegramSentMessage):
        logging.info(
            "Sent Telegram handoff notice for %s channel %s.",
            source,
            channel_id,
        )
        return sent
    return None


def _remember_sent_handoff_notice(
    *,
    channel_id: int,
    reason: str,
    notice: TelegramSentMessage | None,
) -> None:
    if notice is None:
        return
    remember_team_handoff_notice(
        channel_id,
        TeamHandoffNotice(
            telegram_chat_id=notice.chat_id,
            telegram_message_id=notice.message_id,
            reason=reason,
            message_text=notice.message_text,
        ),
    )


async def _send_ticket_handoff_notice(
    *,
    reason: str,
    summary: str,
    channel_id: int,
    guild_id: int | None,
    investigation_job: TicketInvestigationJob,
) -> TelegramSentMessage | None:
    recent_user_messages = _recent_user_messages_for_handoff(channel_id, summary)
    known_facts = _handoff_known_facts(investigation_job)
    notice = await _notify_handoff(
        reason=reason,
        summary=summary,
        channel_id=channel_id,
        guild_id=guild_id,
        source="ticket",
        recent_user_messages=recent_user_messages,
        known_facts=known_facts,
    )
    return notice


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
    current_history_override: List[TResponseInputItem] | None = None,
) -> InternalInstructionTurnResult:
    investigation_job = deepcopy(get_or_create_ticket_investigation_job(channel_id))
    current_history = (
        list(current_history_override)
        if current_history_override is not None
        else list(conversation_threads.get(channel_id, []))
    )
    if (
        current_history_override is None
        and not current_history
        and isinstance(channel, discord.TextChannel)
    ):
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
        worker_result = await _execute_ticket_turn(
            executor=executor,
            channel_id=channel_id,
            request=_build_turn_request(
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
        return InternalInstructionTurnResult(
            reply=_render_support_reply(worker_result.flow_outcome.raw_final_reply),
            conversation_history=worker_result.flow_outcome.conversation_history,
            input_history=current_history,
        )
    finally:
        await progress_reporter.close()


@dataclass(frozen=True)
class InternalInstructionTurnResult:
    reply: str
    conversation_history: List[TResponseInputItem]
    input_history: List[TResponseInputItem]


async def _execute_ticket_turn(
    *,
    executor: TransportTicketInvestigationExecutor,
    channel_id: int,
    request: TicketTurnRequest,
    hooks: TicketExecutionHooks,
):
    current_task = asyncio.current_task()
    if current_task is not None:
        active_ticket_executor_tasks[channel_id] = current_task
    try:
        return await executor.execute_turn(request, hooks=hooks)
    finally:
        if active_ticket_executor_tasks.get(channel_id) is current_task:
            active_ticket_executor_tasks.pop(channel_id, None)


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


def _is_support_staff_member(author: discord.abc.User) -> bool:
    return is_ticket_support_staff(author)


def _has_staff_summon_prefix(text: str) -> bool:
    stripped = (text or "").strip()
    prefix = config.TICKET_STAFF_SUMMON_PREFIX
    return bool(stripped) and stripped.lower().startswith(prefix.lower())


def _normalize_staff_summon_prompt(text: str) -> str | None:
    if not _has_staff_summon_prefix(text):
        return None
    stripped = (text or "").strip()
    prefix = config.TICKET_STAFF_SUMMON_PREFIX
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


TicketMessageAction = Literal[
    "process",
    "ignore",
    "staff_summon",
    "staff_summon_usage",
    "staff_takeover",
]


def _classify_ticket_message_action(
    *,
    author: discord.abc.User,
    content: str,
    ticket_owner_user_id: int | None,
    stopped: bool,
) -> TicketMessageAction:
    if ticket_owner_user_id is not None and author.id == ticket_owner_user_id:
        return "ignore" if stopped else "process"
    is_staff = _is_support_staff_member(author)
    if is_staff and _has_staff_summon_prefix(content):
        return (
            "staff_summon"
            if _normalize_staff_summon_prompt(content)
            else "staff_summon_usage"
        )
    if is_staff:
        return "staff_takeover"
    if stopped:
        return "ignore"
    if ticket_owner_user_id is None:
        return "process"
    return "ignore"


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
    elif _is_support_staff_member(author):
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
        message async for message in channel.history(limit=limit, oldest_first=True)
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


async def _build_staff_summon_history(
    channel: discord.TextChannel,
    *,
    exclude_message_id: int,
    ticket_owner_user_id: int | None,
    bot_user_id: int | None,
    scan_limit: int = 30,
    history_limit: int = 12,
) -> List[TResponseInputItem]:
    messages = [message async for message in channel.history(limit=scan_limit)]
    messages.reverse()
    history: List[TResponseInputItem] = []
    for message in messages:
        if message.id == exclude_message_id:
            continue
        content = _message_text_for_turn(message)
        if not content:
            continue
        if message.author.bot:
            if bot_user_id is None or message.author.id == bot_user_id:
                history.append({"role": "assistant", "content": content})
            continue
        if _is_support_staff_member(message.author):
            history.append(
                {
                    "role": "system",
                    "content": f"Internal support staff message: {content}",
                }
            )
            continue
        if ticket_owner_user_id is None or message.author.id == ticket_owner_user_id:
            history.append({"role": "user", "content": content})
    return history[-history_limit:]


# Discord Bot Implementation
