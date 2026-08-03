from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging

import discord

import config
from discord_support_runtime import _detect_ticket_owner_user_id
from handoff import build_archived_handoff_notice, edit_handoff_notice
from state import (
    cancel_pending_ticket_task,
    clear_ticket_channel_state,
    last_bot_reply_ts_by_channel,
    mark_ticket_awaiting_initial_button,
    remember_ticket_owner_user_id,
    team_handoff_notice_by_channel,
)
from views import InitialInquiryView


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
    clear_ticket_channel_state(channel.id, keep_stopped=False, delete_persisted=True)
    await cancel_pending_ticket_task(channel.id)
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

async def clear_deleted_ticket_channel(channel: discord.abc.GuildChannel) -> None:
    if not isinstance(channel, discord.TextChannel):
        return
    await cancel_pending_ticket_task(channel.id)
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
