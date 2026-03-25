import unittest
from unittest.mock import patch

from utils import send_long_message


class _FakeChannel:
    def __init__(self) -> None:
        self.sent = []

    async def send(self, text, **kwargs):
        self.sent.append((text, kwargs))


class _FakeMessage:
    def __init__(self) -> None:
        self.replies = []
        self.channel = _FakeChannel()

    async def reply(self, text, **kwargs):
        self.replies.append((text, kwargs))


class SendLongMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_long_message_suppresses_embeds_for_channel_send(self) -> None:
        channel = _FakeChannel()

        await send_long_message(channel, "See https://yearn.fi/vaults/1/0x123")

        self.assertEqual(len(channel.sent), 1)
        self.assertTrue(channel.sent[0][1]["suppress_embeds"])

    async def test_send_long_message_suppresses_embeds_for_message_reply(self) -> None:
        message = _FakeMessage()

        with patch("utils.discord.Message", _FakeMessage):
            await send_long_message(message, "See https://yearn.fi/vaults/1/0x123")

        self.assertEqual(len(message.replies), 1)
        self.assertTrue(message.replies[0][1]["suppress_embeds"])
