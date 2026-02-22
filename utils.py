import asyncio
import logging
from typing import Union

import discord

import config


async def send_long_message(
    target: Union[discord.TextChannel, discord.Message],
    text: str,
    view: discord.ui.View = None
):
    """Sends a potentially long message, splitting it into chunks if necessary."""
    if len(text) <= config.MAX_DISCORD_MESSAGE_LENGTH:
        try:
            if isinstance(target, discord.Message):
                await target.reply(text, view=view)
            else:
                await target.send(text, view=view)
        except discord.HTTPException as e:
            logging.error(f"Discord API error sending message: {e}")
        return

    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(line) > config.MAX_DISCORD_MESSAGE_LENGTH:
            for i in range(0, len(line), config.MAX_DISCORD_MESSAGE_LENGTH):
                part = line[i:i + config.MAX_DISCORD_MESSAGE_LENGTH]
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.append(part)
            continue

        if len(current_chunk) + len(line) + 1 > config.MAX_DISCORD_MESSAGE_LENGTH:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = f"{current_chunk}\n{line}" if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) == 1 and len(chunks[0]) > config.MAX_DISCORD_MESSAGE_LENGTH:
        logging.warning("Message splitting resulted in a single chunk still exceeding limit. Truncating.")
        chunks = [chunks[0][:config.MAX_DISCORD_MESSAGE_LENGTH - 3] + "..."]
    elif len(chunks) == 0:
        logging.warning("Message splitting resulted in zero chunks for long message.")
        chunks.append(text[:config.MAX_DISCORD_MESSAGE_LENGTH - 3] + "...")

    first_message = True

    try:
        for chunk in chunks:
            if first_message:
                if isinstance(target, discord.Message):
                    await target.reply(chunk, view=view)
                else:
                    await target.send(chunk, view=view)
                first_message = False
            else:
                channel = target.channel if isinstance(target, discord.Message) else target
                await channel.send(chunk)
            await asyncio.sleep(0.3)
    except discord.HTTPException as e:
        logging.error(f"Discord API error sending message chunk: {e}")
        try:
            channel = target.channel if isinstance(target, discord.Message) else target
            await channel.send(f"Error: Could not send full message due to Discord API issue ({e.status}).", suppress_embeds=True)
        except Exception:
            pass
    except Exception as e:
        logging.error(f"Unexpected error in send_long_message: {e}", exc_info=True)
