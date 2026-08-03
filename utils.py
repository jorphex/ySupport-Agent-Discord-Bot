import asyncio
import logging
from typing import Union

import discord

import config


def split_long_message(text: str) -> list[str]:
    """Split text into Discord-sized chunks."""
    if len(text) <= config.MAX_DISCORD_MESSAGE_LENGTH:
        return [text]

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

        if (
            current_chunk
            and len(current_chunk) + len(line) + 1
            > config.MAX_DISCORD_MESSAGE_LENGTH
        ):
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = f"{current_chunk}\n{line}" if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) == 1 and len(chunks[0]) > config.MAX_DISCORD_MESSAGE_LENGTH:
        logging.warning("Message splitting resulted in a single chunk still exceeding limit. Truncating.")
        return [chunks[0][:config.MAX_DISCORD_MESSAGE_LENGTH - 3] + "..."]
    if not chunks:
        logging.warning("Message splitting resulted in zero chunks for long message.")
        return [text[:config.MAX_DISCORD_MESSAGE_LENGTH - 3] + "..."]
    return chunks


async def send_long_message(
    target: Union[discord.TextChannel, discord.Message],
    text: str,
    view: discord.ui.View = None
):
    """Send every Discord chunk, raising if delivery is incomplete."""
    if len(text) <= config.MAX_DISCORD_MESSAGE_LENGTH:
        if isinstance(target, discord.Message):
            await target.reply(text, view=view, suppress_embeds=True)
        else:
            await target.send(text, view=view, suppress_embeds=True)
        return

    chunks = split_long_message(text)
    first_message = True

    for chunk in chunks:
        if first_message:
            if isinstance(target, discord.Message):
                await target.reply(chunk, view=view, suppress_embeds=True)
            else:
                await target.send(chunk, view=view, suppress_embeds=True)
            first_message = False
        else:
            channel = target.channel if isinstance(target, discord.Message) else target
            await channel.send(chunk, suppress_embeds=True)
        await asyncio.sleep(0.3)
