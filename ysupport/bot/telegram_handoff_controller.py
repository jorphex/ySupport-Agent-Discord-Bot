from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import logging
from typing import Any, Protocol

import discord

from ysupport.bot.discord_support_runtime import (
    _attachment_identity,
    _attachment_payloads_from_message,
    _build_ticket_run_context,
    _dedupe_attachment_payloads,
    _find_team_handoff_notice,
    _message_text_for_turn,
    _run_internal_instruction_turn,
)
from ysupport import config, stats
from ysupport.integrations.handoff import (
    answer_telegram_callback_query,
    build_closed_handoff_notice,
    build_dismissed_handoff_notice,
    build_failed_delivery_handoff_notice,
    build_pending_delivery_handoff_notice,
    build_vague_team_reply_feedback,
    DISMISS_HANDOFF_CALLBACK_DATA,
    edit_handoff_notice,
    fetch_telegram_updates,
    is_substantive_team_reply,
    retire_handoff_notice,
    send_telegram_message,
    TelegramApiError,
)
from ysupport.bot.state import (
    cancel_handoff_delivery,
    cancel_pending_ticket_task,
    channel_intent_after_button,
    clear_team_handoff_notice,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    handoff_delivery_tasks,
    history_discord_offsets_by_channel,
    last_bot_reply_ts_by_channel,
    load_telegram_update_offset,
    mark_team_handoff_notice_delivered,
    mark_team_handoff_notice_pending_delivery,
    pending_attachments_by_channel,
    persist_telegram_update_offset,
    persist_ticket_state,
    replace_ticket_history,
    reset_ticket_codex_session,
    stop_ticket_channel,
    stop_reasons_by_channel,
    stopped_channels,
    TeamHandoffNotice,
    team_handoff_notice_by_channel,
    ticket_owner_user_id_by_channel,
)
from ysupport.bot.utils import send_long_message
from ysupport.bot.public_handoffs import PublicHandoffs, PublicHandoffRegistrationPending
from ysupport.bot.views import is_ticket_support_staff
from ysupport.execution.codex_support_attachments import attachment_is_image
from ysupport.execution.contracts import PRIVATE_TICKET_CONTEXT_LIMIT, PRIVATE_TICKET_HANDOFF_CONTEXT_LIMIT


_HANDOFF_IMAGE_LIMIT = 4


class _TelegramBotHost(Protocol):
    investigation_executor: Any
    user: discord.ClientUser | None

    def is_closed(self) -> bool: ...
    def get_channel(self, channel_id: int) -> Any: ...


class TelegramHandoffController:
    def __init__(self, bot: _TelegramBotHost) -> None:
        self.bot = bot
        self.public = PublicHandoffs(bot)
        self.task: asyncio.Task[None] | None = None
        self._delivery_tasks: list[asyncio.Task[None]] = []
        self._delivery_wakeups: list[asyncio.Event] = []
        self._delivery_channels: set[int] = set()
        self._telegram_update_offset = load_telegram_update_offset()
        self._telegram_recovery_attempts: set[tuple[int, int, str]] = set()

    def _can_deliver_handoff(self, channel_id: int, notice: TeamHandoffNotice) -> bool:
        return team_handoff_notice_by_channel.get(channel_id) is notice and (
            channel_id not in stopped_channels
            or stop_reasons_by_channel.get(channel_id) == "staff_takeover"
        )

    def start(self) -> bool:
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_YSUPPORT_CHAT:
            return False
        self.public.hydrate()
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._telegram_handoff_reply_loop())
        self._wake_delivery_worker()
        return True

    async def close(self) -> None:
        if self.task is not None:
            task = self.task
            self.task = None
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        delivery_tasks, self._delivery_tasks = self._delivery_tasks, []
        self._delivery_wakeups.clear()
        for task in delivery_tasks:
            task.cancel()
        await asyncio.gather(*delivery_tasks, return_exceptions=True)

    def _wake_delivery_worker(self) -> None:
        for index in range(config.TICKET_EXECUTION_MAX_CONCURRENT_TURNS):
            if index == len(self._delivery_tasks):
                self._delivery_wakeups.append(asyncio.Event())
                self._delivery_tasks.append(asyncio.create_task(self._delivery_loop(self._delivery_wakeups[index])))
            elif self._delivery_tasks[index].done():
                self._delivery_tasks[index] = asyncio.create_task(self._delivery_loop(self._delivery_wakeups[index]))
            self._delivery_wakeups[index].set()

    async def _delivery_loop(self, wakeup: asyncio.Event) -> None:
        while not self.bot.is_closed():
            await wakeup.wait()
            wakeup.clear()
            try:
                await self._resume_pending_telegram_handoff_replies()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Telegram handoff delivery failed; pending state retained.")

    async def _telegram_handoff_reply_loop(self) -> None:
        while not self.bot.is_closed():
            try:
                self._wake_delivery_worker()
                updates = await fetch_telegram_updates(self._telegram_update_offset)
                for update in updates:
                    update_id = update.get("update_id")
                    await self._handle_telegram_handoff_update(update)
                    if isinstance(update_id, int):
                        self._telegram_update_offset = update_id + 1
                        persist_telegram_update_offset(self._telegram_update_offset)
            except asyncio.CancelledError:
                raise
            except PublicHandoffRegistrationPending:
                await asyncio.sleep(0.1)
            except TelegramApiError as exc:
                logging.warning(
                    "Telegram polling temporarily unavailable; retrying in 5 seconds: %s",
                    exc,
                )
                await asyncio.sleep(5)
            except Exception as exc:
                logging.error(
                    "Telegram handoff reply loop failed: %s", exc, exc_info=True
                )
                await asyncio.sleep(5)

    async def _resume_pending_telegram_handoff_replies(self) -> None:
        await self.public.resume()
        for channel_id, notice in list(team_handoff_notice_by_channel.items()):
            if not self._can_deliver_handoff(channel_id, notice):
                continue
            phase = notice.status
            recovery_key = (
                channel_id,
                notice.telegram_message_id,
                phase,
            )
            if recovery_key in self._telegram_recovery_attempts or channel_id in self._delivery_channels:
                continue
            self._delivery_channels.add(channel_id)
            try:
                if phase in {"pending_delivery", "posting"} and notice.pending_reply_text:
                    self._telegram_recovery_attempts.add(recovery_key)
                    try:
                        delivered = await self._run_handoff_delivery(
                            channel_id=channel_id,
                            notice=notice,
                            team_reply_text=notice.pending_reply_text,
                        )
                    except (asyncio.TimeoutError, OSError):
                        if phase != "posting":
                            raise
                        # Retry the read on a later pass while keeping the send record.
                        self._telegram_recovery_attempts.discard(recovery_key)
                        logging.warning("Discord reconciliation for ticket %s is temporarily unavailable.", channel_id)
                        continue
                    if (delivered is False and self._can_deliver_handoff(channel_id, notice)
                        and notice.status != "delivered_pending_close"):
                        if phase == "pending_delivery" and notice.status == "posting":
                            # Discord may have accepted the send. Reconcile on the next pass.
                            continue
                        if notice.status != "review":
                            self._leave_update_for_review(channel_id, notice, "Discord delivery could not be confirmed")
                        await self._show_ticket_delivery_failure(channel_id, notice)
                    continue
                if phase == "review":
                    await self._show_ticket_delivery_failure(channel_id, notice)
                    continue
                if phase == "delivered_pending_close":
                    if not persist_ticket_state(channel_id):
                        raise RuntimeError(
                            "Could not persist the delivered Telegram handoff state."
                        )
                    await self._finalize_telegram_handoff_notice_close(
                        channel_id=channel_id,
                        notice=notice,
                    )
            finally:
                self._delivery_channels.discard(channel_id)
        active_recovery_keys = {
            (channel_id, notice.telegram_message_id, notice.status)
            for channel_id, notice in team_handoff_notice_by_channel.items()
            if notice.status in {"pending_delivery", "posting", "review"}
        }
        self._telegram_recovery_attempts.intersection_update(active_recovery_keys)

    async def _show_ticket_delivery_failure(self, channel_id, notice):
        review = notice.status == "review"
        sent = await edit_handoff_notice(
            chat_id=notice.telegram_chat_id,
            message_id=notice.telegram_message_id,
            message_text=build_failed_delivery_handoff_notice(notice.message_text or "Ticket needs team review"),
            **({"missing_ok": True} if review else {}),
        )
        if sent and review and self._can_deliver_handoff(channel_id, notice):
            self._telegram_recovery_attempts.add((channel_id, notice.telegram_message_id, "review"))

    async def _handle_telegram_handoff_update(self, update: dict[str, Any]) -> None:
        if await self.public.handle(update):
            self._wake_delivery_worker()
            return
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
        if matched_notice.status != "open":
            return
        if not self._can_deliver_handoff(matched_channel_id, matched_notice):
            return
        if not is_substantive_team_reply(text):
            await send_telegram_message(
                chat_id=chat_id,
                message_text=build_vague_team_reply_feedback(),
                reply_to_message_id=telegram_message_id
                if isinstance(telegram_message_id, int)
                else None,
            )
            return
        if not isinstance(telegram_message_id, int):
            return

        if not mark_team_handoff_notice_pending_delivery(
            matched_channel_id,
            reply_text=text,
        ):
            raise RuntimeError(
                "Could not persist the accepted Telegram handoff reply."
            )
        self._wake_delivery_worker()

    async def _handle_telegram_handoff_dismissal(
        self,
        callback_query: dict[str, Any],
    ) -> None:
        callback_query_id = str(callback_query.get("id") or "").strip()
        if not callback_query_id:
            return
        message = callback_query.get("message")
        chat = message.get("chat") if isinstance(message, dict) else None
        chat_id = str(chat.get("id") or "").strip() if isinstance(chat, dict) else ""
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
            or matched_notice.status != "open"
        ):
            await answer_telegram_callback_query(
                callback_query_id=callback_query_id,
                message_text="This handoff is already closed.",
            )
            return

        stopped_durably = stop_ticket_channel(matched_channel_id)
        await cancel_pending_ticket_task(matched_channel_id)
        await cancel_handoff_delivery(matched_channel_id)

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
        retired = await retire_handoff_notice(
            chat_id=chat_id,
            message_id=message_id,
            fallback_message_text=build_dismissed_handoff_notice(
                matched_notice.message_text or "Dismissed. Handle in Discord."
            ),
        )
        if retired:
            stats.record("ticket_handoff_dismissed")
            return

        team_handoff_notice_by_channel[matched_channel_id] = matched_notice
        retry_persisted = persist_ticket_state(matched_channel_id)
        logging.warning(
            "Dismissed Telegram handoff for channel %s, but could not remove "
            "or edit notice %s; cleanup retry persisted=%s.",
            matched_channel_id,
            message_id,
            retry_persisted,
        )

    async def _run_handoff_delivery(self, *, channel_id, notice, team_reply_text) -> bool | None:
        task = asyncio.create_task(self._deliver_telegram_handoff_reply(
            channel_id=channel_id, notice=notice, team_reply_text=team_reply_text,
        ))
        handoff_delivery_tasks[channel_id] = task
        try:
            # Child cancellation is a result; worker cancellation still propagates.
            result, = await asyncio.shield(asyncio.gather(task, return_exceptions=True))
            if isinstance(result, asyncio.CancelledError):
                return None
            if isinstance(result, BaseException):
                raise result
            return result
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if handoff_delivery_tasks.get(channel_id) is task:
                handoff_delivery_tasks.pop(channel_id, None)

    async def _ticket_snapshot(self, channel, owner_id, notice):
        messages = [message async for message in channel.history(limit=PRIVATE_TICKET_CONTEXT_LIMIT)]
        bounded = len(messages) == PRIVATE_TICKET_CONTEXT_LIMIT
        after_id = messages[0].id if messages else 0
        recorded_ids = self._recorded_input_ids(notice)
        if bounded:
            # An absent ID inside the window may be a gateway update ahead of
            # history. Only supplement inputs that actually predate the window.
            older_ids = sorted(message_id for message_id in recorded_ids if message_id < messages[-1].id)
            extra_limit = PRIVATE_TICKET_HANDOFF_CONTEXT_LIMIT - PRIVATE_TICKET_CONTEXT_LIMIT
            if 0 < len(older_ids) <= extra_limit:
                older_messages = await asyncio.gather(
                    *(channel.fetch_message(message_id) for message_id in older_ids), return_exceptions=True,
                )
                for message in reversed(older_messages):
                    if isinstance(message, (asyncio.TimeoutError, discord.HTTPException, OSError)):
                        continue
                    if isinstance(message, BaseException):
                        raise message
                    messages.append(message)
        revision, history, recorded_images, recent_images = [], {}, [], []
        for message in reversed(messages):
            if message.author.bot:
                if message.author.id == getattr(self.bot.user, "id", None) and message.content:
                    history[message.id] = {"role": "assistant", "content": message.content}
                continue
            text = _message_text_for_turn(message)
            if not text:
                continue
            is_staff = is_ticket_support_staff(message.author)
            if message.author.id == owner_id or (owner_id is None and not is_staff):
                role = "ticket_user"
                history[message.id] = {"role": "user", "content": text}
            elif is_staff:
                role = "internal_team"
                history[message.id] = {"role": "system", "content": f"Internal support staff message: {text}"}
            else:
                continue
            images = [{**item, "source_author_role": role} for item in
                      _attachment_payloads_from_message(message) if attachment_is_image(item)]
            revision.append((message.id, message.author.id, role, text, _attachment_identity(images)))
            (recorded_images if message.id in recorded_ids else recent_images).extend(images)
        slots = _HANDOFF_IMAGE_LIMIT - len(recorded_images)
        attachments = recorded_images + (recent_images[-slots:] if slots > 0 else [])
        attachments.sort(key=lambda item: item["source_message_id"])
        return revision, history, attachments, after_id, bounded

    @staticmethod
    def _retained_history_indices(saved_history, recent_history, bounded, offsets):
        if not bounded or not recent_history:
            return []
        positions = {
            message_id: offset for message_id, offset in offsets.items()
            if message_id in recent_history and offset < len(saved_history)
        }
        replaced = set(positions.values())
        rows = list(recent_history.items())
        cutoff = len(saved_history)
        # Match recorded rows by source. Only infer overlap from text where
        # provenance is unavailable and the saved row has no recorded source.
        for overlap in range(min(len(saved_history), len(rows)), 0, -1):
            start = len(saved_history) - overlap
            if all(
                index == positions[message_id] if message_id in positions
                else index not in replaced and saved_history[index] == row
                for index, (message_id, row) in enumerate(rows[:overlap], start=start)
            ):
                cutoff = start
                break
        return [index for index in range(cutoff) if index not in replaced]

    def _commit_ticket_snapshot(self, channel_id, recent_history, bounded, *, saved_count=None, reply=None):
        live = conversation_threads.get(channel_id, [])
        saved_count = len(live) if saved_count is None else saved_count
        previous_offsets = history_discord_offsets_by_channel.get(channel_id, {})
        retained = self._retained_history_indices(live[:saved_count], recent_history, bounded, previous_offsets)
        history = [live[index] for index in retained]
        rebased = {old: new for new, old in enumerate(retained)}
        new_offsets = {message_id: index for index, message_id in enumerate(recent_history, start=len(history))}
        history.extend(recent_history.values())
        if reply is not None:
            history.append({"role": "assistant", "content": reply})
        # Gateway delivery can lag the Discord read while the reply is sending.
        observed = {index for message_id, index in previous_offsets.items() if message_id in recent_history}
        late = [index for index in range(saved_count, len(live)) if index not in observed]
        rebased.update({old: new for new, old in enumerate(late, start=len(history))})
        history.extend(live[index] for index in late)
        offsets = {message_id: rebased[index] for message_id, index in previous_offsets.items() if index in rebased}
        offsets.update(new_offsets)
        replace_ticket_history(channel_id, history, discord_offsets=offsets)

    @staticmethod
    def _retain_late_handoff_attachments(channel_id, attachments):
        if attachments:
            pending = pending_attachments_by_channel.get(channel_id, [])
            urls = {image.get("url") for image in attachments}
            pending_attachments_by_channel[channel_id] = _dedupe_attachment_payloads([
                {**image, "handoff_followup": True} if image.get("url") in urls else image
                for image in [*pending, *attachments]
            ])

    @staticmethod
    def _delivery_nonce(channel_id, notice):
        identity = f"ticket:{channel_id}:{notice.telegram_chat_id}:{notice.telegram_message_id}"
        return hashlib.sha256(identity.encode()).hexdigest()[:24]

    async def _recover_ticket_update(self, channel, notice):
        if not notice.delivery_digest or type(notice.delivery_after_id) is not int or notice.delivery_after_id < 0:
            raise ValueError("Interrupted ticket send has no delivery record")
        bot_id = getattr(self.bot.user, "id", None)
        nonce = self._delivery_nonce(channel.id, notice)
        matches = [message async for message in channel.history(
            after=discord.Object(id=notice.delivery_after_id), limit=50, oldest_first=True)
            if bot_id is not None and message.author.id == bot_id
            and getattr(message, "nonce", None) in (None, nonce)
            and getattr(getattr(message, "reference", None), "message_id", None) == notice.delivery_source_id
            and hashlib.sha256(message.content.encode()).hexdigest() == notice.delivery_digest]
        if len(matches) != 1:
            raise ValueError("Interrupted ticket send could not be uniquely reconciled")
        return matches[0].content

    async def _deliver_telegram_handoff_reply(
        self,
        *,
        channel_id: int,
        notice: TeamHandoffNotice,
        team_reply_text: str,
    ) -> bool:
        if not self._can_deliver_handoff(channel_id, notice):
            return False
        if notice.status == "pending_delivery":
            await edit_handoff_notice(
                chat_id=notice.telegram_chat_id,
                message_id=notice.telegram_message_id,
                message_text=build_pending_delivery_handoff_notice(
                    notice.message_text or "Reply received. Delivering update..."
                ),
            )
            if not self._can_deliver_handoff(channel_id, notice):
                return False

        channel = self.bot.get_channel(channel_id)
        if (
            channel is None
            or not hasattr(channel, "send")
            or not hasattr(channel, "typing")
        ):
            logging.warning(
                "Could not resolve Discord channel %s for Telegram handoff reply.",
                channel_id,
            )
            return False

        ticket_owner_user_id = ticket_owner_user_id_by_channel.get(channel_id)
        if notice.status == "posting":
            try:
                delivered_text = await self._recover_ticket_update(channel, notice)
            except (ValueError, discord.HTTPException):
                logging.warning("Cannot reconcile interrupted Telegram update for ticket %s.", channel_id)
                return False
            if not self._can_deliver_handoff(channel_id, notice):
                return False
            # Recovery can follow a process exit before any of the fresh history was committed.
            try:
                revision, recent_history, _, _, bounded = await self._ticket_snapshot(channel, ticket_owner_user_id, notice)
            except (asyncio.TimeoutError, discord.HTTPException, OSError):
                revision, recent_history, bounded = [], {}, False
            if not self._can_deliver_handoff(channel_id, notice):
                return False
            if (revision and self._snapshot_covers_input(revision, notice)
                and {"role": "assistant", "content": delivered_text} in recent_history.values()):
                self._commit_ticket_snapshot(channel_id, recent_history, bounded)
            else:
                conversation_threads.setdefault(channel_id, []).append({"role": "assistant", "content": delivered_text})
            visible_input_ids = {entry[0] for entry in revision}
            late_images = [
                image for image in notice.followup_attachments
                if (image.get("source_message_id") or 0) > notice.delivery_after_id
                or image.get("source_message_id") not in visible_input_ids
            ]
            self._retain_late_handoff_attachments(channel_id, late_images)
            reset_ticket_codex_session(channel_id)
            return await self._complete_ticket_update(channel_id, notice)

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
            "Do not make stronger claims than the team message supports. "
            "Use only the latest relevant ticket details and add only what the user still needs. "
            "Keep the update under 1,800 characters for a single Discord message."
        )
        try:
            async with channel.typing():
                for _ in range(2):
                    revision, recent_history, images, _, bounded = await self._ticket_snapshot(channel, ticket_owner_user_id, notice)
                    if not self._can_deliver_handoff(channel_id, notice):
                        return False
                    if not any(entry[2] == "ticket_user" for entry in revision):
                        return self._leave_update_for_review(channel_id, notice, "no readable message from the ticket user")
                    if not self._snapshot_covers_input(revision, notice):
                        continue
                    if len(images) > _HANDOFF_IMAGE_LIMIT:
                        return self._leave_update_for_review(channel_id, notice, "too many recorded screenshots to include")
                    if not self._snapshot_covers_images(images, notice):
                        continue
                    saved_history = list(conversation_threads.get(channel_id, []))
                    handoff_attachments = list(notice.followup_attachments)
                    stop_reason = stop_reasons_by_channel.get(channel_id)
                    # A previous rollout can contain messages since edited or deleted in Discord.
                    reset_ticket_codex_session(channel_id)
                    turn_result = await _run_internal_instruction_turn(
                        executor=self.bot.investigation_executor,
                        channel=channel,
                        channel_id=channel_id,
                        run_context=run_context,
                        prompt_text=team_reply_text,
                        instruction_text=instruction_text + (
                            " Staff controls this ticket. Send this one update only. Do not "
                            "promise automatic follow-up or ask the user to reply to the bot."
                            if stop_reason == "staff_takeover" else ""
                        ),
                        workflow_suffix="team handoff reply",
                        attachments=images,
                        current_history_override=list(recent_history.values()),
                        show_progress=False,
                    )
                    fresh_revision, fresh_history, fresh_images, after_id, _ = await self._ticket_snapshot(
                        channel, ticket_owner_user_id, notice,
                    )
                    if not self._can_deliver_handoff(channel_id, notice):
                        reset_ticket_codex_session(channel_id)
                        return False
                    if (revision == fresh_revision
                        and recent_history == fresh_history
                        and _attachment_identity(images) == _attachment_identity(fresh_images)
                        and self._snapshot_covers_input(fresh_revision, notice)
                        and saved_history == conversation_threads.get(channel_id, [])
                        and _attachment_identity(handoff_attachments) == _attachment_identity(notice.followup_attachments)
                        and stop_reason == stop_reasons_by_channel.get(channel_id)):
                        break
                    reset_ticket_codex_session(channel_id)
                else:
                    return self._leave_update_for_review(channel_id, notice, "recorded ticket input was unavailable or changed")

                reply = turn_result.reply.strip()
                if not reply or len(reply) > config.MAX_DISCORD_MESSAGE_LENGTH:
                    reset_ticket_codex_session(channel_id)
                    return self._leave_update_for_review(channel_id, notice, "reply did not fit in one message")
                notice.delivery_digest = hashlib.sha256(reply.encode()).hexdigest()
                notice.delivery_after_id = after_id
                notice.delivery_source_id = next((entry[0] for entry in reversed(revision)
                                                  if entry[2] == "ticket_user"), None)
                notice.status = "posting"
                if not persist_ticket_state(channel_id):
                    notice.delivery_digest = notice.delivery_after_id = notice.delivery_source_id = None
                    notice.status = "pending_delivery"
                    raise RuntimeError("Could not persist ticket delivery intent")
                reference = (discord.MessageReference(message_id=notice.delivery_source_id, channel_id=channel_id)
                             if notice.delivery_source_id is not None else None)
                await send_long_message(channel, reply, reference=reference,
                                        nonce=self._delivery_nonce(channel_id, notice))
        except Exception as exc:
            reset_ticket_codex_session(channel_id)
            logging.error(
                "Failed to deliver Telegram handoff reply for channel %s: %s",
                channel_id,
                exc,
                exc_info=True,
            )
            return False

        if not self._can_deliver_handoff(channel_id, notice):
            reset_ticket_codex_session(channel_id)
            return False

        live_history = conversation_threads.get(channel_id, [])
        late_attachments = (
            list(notice.followup_attachments[len(handoff_attachments) :])
            if notice.followup_attachments[: len(handoff_attachments)]
            == handoff_attachments
            else []
        )

        if live_history[:len(saved_history)] == saved_history:
            self._commit_ticket_snapshot(channel_id, recent_history, bounded, saved_count=len(saved_history), reply=reply)
        else:
            conversation_threads.setdefault(channel_id, []).append({"role": "assistant", "content": reply})
        self._retain_late_handoff_attachments(channel_id, late_attachments)
        return await self._complete_ticket_update(channel_id, notice)

    @staticmethod
    def _recorded_input_ids(notice):
        required = set(notice.discord_input_ids)
        for attachment in notice.followup_attachments:
            source_id = attachment.get("source_message_id")
            if type(source_id) is int:
                required.add(source_id)
        return required

    def _snapshot_covers_input(self, revision, notice):
        return self._recorded_input_ids(notice).issubset(entry[0] for entry in revision)

    @staticmethod
    def _snapshot_covers_images(images, notice):
        visible = {(image["source_message_id"], image.get("attachment_id")) for image in images}
        visible_sources = {source_id for source_id, _ in visible}
        for image in notice.followup_attachments:
            source_id, attachment_id = image.get("source_message_id"), image.get("attachment_id")
            if attachment_is_image(image) and type(source_id) is int:
                if source_id not in visible_sources or (attachment_id is not None and (source_id, attachment_id) not in visible):
                    return False
        return True

    @staticmethod
    def _leave_update_for_review(channel_id, notice, reason):
        previous_status = notice.status
        notice.status = "review"
        if not persist_ticket_state(channel_id):
            notice.status = previous_status
        elif previous_status != "review":
            stats.record("ticket_handoff_delivery_failed")
        logging.warning("Telegram update for ticket %s needs staff review: %s.", channel_id, reason)
        return False

    async def _complete_ticket_update(self, channel_id, notice):
        get_or_create_ticket_investigation_job(channel_id).mark_waiting_for_user()
        last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
        stats.record_handoff_outcome(
            "ticket_handoff_replies_delivered", self._delivery_nonce(channel_id, notice), channel_id,
        )
        if not mark_team_handoff_notice_delivered(channel_id):
            raise RuntimeError(
                "Delivered the Telegram handoff reply, but could not persist its state."
            )
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
        if not self._can_deliver_handoff(channel_id, notice):
            return False
        stats.settle_handoff_outcome(self._delivery_nonce(channel_id, notice))
        edited = await edit_handoff_notice(
            chat_id=notice.telegram_chat_id,
            message_id=notice.telegram_message_id,
            message_text=build_closed_handoff_notice(
                notice.message_text or "Reply received. Replies closed."
            ),
            missing_ok=True,
        )
        if not edited:
            logging.warning(
                "Failed to close Telegram handoff notice for channel %s after Discord delivery.",
                channel_id,
            )
            return False
        if not self._can_deliver_handoff(channel_id, notice):
            return False
        clear_team_handoff_notice(channel_id)
        return True
