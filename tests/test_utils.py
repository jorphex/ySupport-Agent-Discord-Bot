import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import patch

from ysupport.bot.utils import send_long_message


class _FakeChannel:
    def __init__(self) -> None:
        self.sent = []

    async def send(self, text, **kwargs):
        self.sent.append((text, kwargs))


class _FailingChannel(_FakeChannel):
    async def send(self, text, **kwargs):
        raise RuntimeError("Discord send failed")


class _FailingAfterOneChannel(_FakeChannel):
    async def send(self, text, **kwargs):
        if self.sent:
            raise RuntimeError("Discord chunk send failed")
        await super().send(text, **kwargs)


class SendLongMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_long_message_propagates_incomplete_delivery(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Discord send failed"):
            await send_long_message(_FailingChannel(), "reply")

    async def test_send_long_message_propagates_partial_chunk_delivery(self) -> None:
        channel = _FailingAfterOneChannel()
        with (
            patch('ysupport.bot.utils.config.MAX_DISCORD_MESSAGE_LENGTH', 5),
            patch('ysupport.bot.utils.asyncio.sleep', return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "Discord chunk send failed"):
                await send_long_message(channel, "first\nsecond")

        self.assertEqual(channel.sent, [("first", {"view": None, "suppress_embeds": True})])
