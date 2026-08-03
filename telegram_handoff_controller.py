from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import logging
from typing import Any, Protocol

from discord_support_runtime import (
    _build_ticket_run_context,
    _dedupe_attachment_payloads,
    _find_team_handoff_notice,
    _refresh_discord_attachment_urls,
    _run_internal_instruction_turn,
)
import config
from handoff import (
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
from state import (
    cancel_pending_ticket_task,
    channel_intent_after_button,
    clear_team_handoff_notice,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    last_bot_reply_ts_by_channel,
    load_telegram_update_offset,
    mark_team_handoff_notice_delivered,
    mark_team_handoff_notice_pending_delivery,
    pending_attachments_by_channel,
    persist_telegram_update_offset,
    persist_ticket_state,
    reset_ticket_codex_session,
    stop_ticket_channel,
    stopped_channels,
    TeamHandoffNotice,
    team_handoff_notice_by_channel,
    ticket_owner_user_id_by_channel,
)
from utils import send_long_message


class _TelegramBotHost(Protocol):
    investigation_executor: Any

    def is_closed(self) -> bool: ...
    def get_channel(self, channel_id: int) -> Any: ...


class TelegramHandoffController:
    def __init__(self, bot: _TelegramBotHost) -> None:
        self.bot = bot
        self.task: asyncio.Task[None] | None = None
        self._telegram_update_offset = load_telegram_update_offset()
        self._telegram_recovery_attempts: set[tuple[int, int, str]] = set()

    def start(self) -> bool:
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_YSUPPORT_CHAT:
            return False
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._telegram_handoff_reply_loop())
        return True

    async def close(self) -> None:
        if self.task is not None:
            task = self.task
            self.task = None
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _telegram_handoff_reply_loop(self) -> None:
        while not self.bot.is_closed():
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
                logging.error(
                    "Telegram handoff reply loop failed: %s", exc, exc_info=True
                )
                await asyncio.sleep(5)

    async def _resume_pending_telegram_handoff_replies(self) -> None:
        for channel_id, notice in list(team_handoff_notice_by_channel.items()):
            recovery_key = (
                channel_id,
                notice.telegram_message_id,
                notice.status,
            )
            if recovery_key in self._telegram_recovery_attempts:
                continue
            if notice.status == "pending_delivery" and notice.pending_reply_text:
                self._telegram_recovery_attempts.add(recovery_key)
                delivered = await self._deliver_telegram_handoff_reply(
                    channel_id=channel_id,
                    notice=notice,
                    team_reply_text=notice.pending_reply_text,
                )
                if (
                    not delivered
                    and channel_id not in stopped_channels
                    and team_handoff_notice_by_channel.get(channel_id) is notice
                ):
                    await edit_handoff_notice(
                        chat_id=notice.telegram_chat_id,
                        message_id=notice.telegram_message_id,
                        message_text=build_failed_delivery_handoff_notice(
                            notice.message_text
                            or "Update delivery failed. Handle in Discord."
                        ),
                    )
                continue
            if notice.status == "delivered_pending_close":
                if not persist_ticket_state(channel_id):
                    raise RuntimeError(
                        "Could not persist the delivered Telegram handoff state."
                    )
                self._telegram_recovery_attempts.add(recovery_key)
                await self._finalize_telegram_handoff_notice_close(
                    channel_id=channel_id,
                    notice=notice,
                )
        active_recovery_keys = {
            (channel_id, notice.telegram_message_id, notice.status)
            for channel_id, notice in team_handoff_notice_by_channel.items()
            if notice.status in {"pending_delivery", "delivered_pending_close"}
        }
        self._telegram_recovery_attempts.intersection_update(active_recovery_keys)

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
        if matched_notice.status != "open":
            return
        if matched_channel_id in stopped_channels:
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
                    handoff_attachments = list(notice.followup_attachments)
                    attachments = await _refresh_discord_attachment_urls(
                        channel,
                        handoff_attachments,
                    )
                    if channel_id in stopped_channels:
                        return False
                    turn_result = await _run_internal_instruction_turn(
                        executor=self.bot.investigation_executor,
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
                await send_long_message(channel, turn_result.reply)
        except Exception as exc:
            reset_ticket_codex_session(channel_id)
            logging.error(
                "Failed to deliver Telegram handoff reply for channel %s: %s",
                channel_id,
                exc,
                exc_info=True,
            )
            return False

        if channel_id in stopped_channels:
            reset_ticket_codex_session(channel_id)
            return False

        input_history = turn_result.input_history
        live_history = conversation_threads.get(channel_id, [])
        late_history = (
            list(live_history[len(input_history) :])
            if live_history[: len(input_history)] == input_history
            else []
        )
        late_attachments = (
            list(notice.followup_attachments[len(handoff_attachments) :])
            if notice.followup_attachments[: len(handoff_attachments)]
            == handoff_attachments
            else []
        )

        conversation_threads[channel_id] = (
            turn_result.conversation_history + late_history
        )
        if late_attachments:
            pending_attachments_by_channel[channel_id] = (
                _dedupe_attachment_payloads(
                    pending_attachments_by_channel.get(channel_id, [])
                    + late_attachments
                )
            )
        investigation_job.mark_waiting_for_user()
        last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
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
