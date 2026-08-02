from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Protocol

import discord

import config
from discord_support_runtime import (
    _build_ticket_run_context,
    _detect_ticket_owner_user_id,
)
from handoff import build_archived_handoff_notice, edit_handoff_notice
from state import (
    clear_ticket_channel_state,
    conversation_threads,
    last_bot_reply_ts_by_channel,
    mark_ticket_awaiting_initial_button,
    monitored_new_channels,
    pending_tasks,
    remember_ticket_owner_user_id,
    team_handoff_notice_by_channel,
)
from views import InitialInquiryView


class _TicketBotHost(Protocol):
    async def process_ticket_message(
        self,
        channel_id: int,
        run_context: Any,
        is_button_trigger: bool = False,
        synthetic_user_message_for_log: str = "",
    ) -> None: ...


async def initialize_ticket_channel(channel: discord.abc.GuildChannel) -> None:
    if not isinstance(channel, discord.TextChannel) or not channel.category:
        return
    if channel.category.id not in config.CATEGORY_CONTEXT_MAP:
        return

    project_context = config.CATEGORY_CONTEXT_MAP.get(channel.category.id, "unknown")
    logging.info(
        "New %s ticket channel created: %s (ID: %s). Initializing state.",
        project_context.capitalize(),
        channel.name,
        channel.id,
    )
    conversation_threads[channel.id] = []
    clear_ticket_channel_state(channel.id, keep_stopped=False, delete_persisted=True)
    task = pending_tasks.pop(channel.id, None)
    if task is not None:
        task.cancel()
    monitored_new_channels.add(channel.id)

    await asyncio.sleep(1.5)
    ticket_owner_user_id = await _detect_ticket_owner_user_id(channel)
    if ticket_owner_user_id is not None:
        remember_ticket_owner_user_id(channel.id, ticket_owner_user_id)
    else:
        logging.info(
            "Could not detect ticket owner from opener messages for channel %s.",
            channel.id,
        )

    welcome_message = (
        f"Welcome to {project_context.capitalize()} Support!\n\n\n"
        "Press a category button below to get started.\n"
        "You can share more details after making a selection.\n\n\n"
        "To process your request accurately, please wait for my response after "
        "you see the *'ySupport is typing...'* indicator before sending another "
        "message.\n\n"
        "---\n"
        "**IGNORE FRIEND REQUESTS**\n"
        "**DO NOT RESPOND TO DMS**\n\n"
        "**WE WILL NEVER ADD OR DM YOU**"
    )
    try:
        await channel.send(
            welcome_message,
            view=InitialInquiryView(),
            suppress_embeds=True,
        )
        last_bot_reply_ts_by_channel[channel.id] = datetime.now(timezone.utc)
        mark_ticket_awaiting_initial_button(channel.id)
    except discord.Forbidden:
        logging.error(
            "Missing permissions to send initial message with buttons in %s",
            channel.id,
        )
    except Exception as exc:
        logging.error(
            "Error sending initial message with buttons in %s: %s",
            channel.id,
            exc,
            exc_info=True,
        )


async def process_synthetic_button_input(
    bot: _TicketBotHost,
    channel: discord.TextChannel,
    synthetic_text: str,
    intent_category: str,
) -> None:
    channel_id = channel.id
    run_context = _build_ticket_run_context(
        channel_id=channel_id,
        category_id=channel.category.id if channel.category else None,
        initial_button_intent=intent_category,
    )
    task = pending_tasks.get(channel_id)
    if task is not None:
        task.cancel()
    await bot.process_ticket_message(
        channel_id,
        run_context,
        is_button_trigger=True,
        synthetic_user_message_for_log=synthetic_text,
    )


async def clear_deleted_ticket_channel(channel: discord.abc.GuildChannel) -> None:
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
