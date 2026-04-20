import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from ticket_transcript_fetch import (
    TranscriptMessage,
    extract_channel_id,
    main,
    normalize_message,
    render_transcript,
)


class TicketTranscriptFetchTests(unittest.TestCase):
    def test_extract_channel_id_accepts_plain_id(self) -> None:
        self.assertEqual(extract_channel_id("1484802638158626887"), "1484802638158626887")

    def test_extract_channel_id_accepts_discord_link(self) -> None:
        self.assertEqual(
            extract_channel_id("https://discord.com/channels/734804446353031319/1484802638158626887"),
            "1484802638158626887",
        )

    def test_extract_channel_id_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            extract_channel_id("not-a-channel")

    def test_normalize_message_preserves_author_and_attachment_urls(self) -> None:
        normalized = normalize_message(
            {
                "id": "1",
                "timestamp": "2026-03-22T11:08:00.000000+00:00",
                "content": "",
                "author": {"id": "42", "username": "ySupport", "bot": True},
                "attachments": [{"url": "https://cdn.example/test.png"}],
            }
        )

        self.assertEqual(normalized.author_id, "42")
        self.assertEqual(normalized.author_name, "ySupport")
        self.assertTrue(normalized.author_is_bot)
        self.assertEqual(normalized.content, "(attachment only)")
        self.assertEqual(normalized.attachments, ("https://cdn.example/test.png",))

    def test_render_transcript_adds_speaker_roles_and_summary(self) -> None:
        transcript = render_transcript(
            [
                TranscriptMessage(
                    id="1",
                    timestamp="2026-03-22T11:08:00.000000+00:00",
                    author_id="100",
                    author_name="User",
                    author_is_bot=False,
                    content="hello",
                    attachments=(),
                ),
                TranscriptMessage(
                    id="2",
                    timestamp="2026-03-22T11:09:00.000000+00:00",
                    author_id="200",
                    author_name="ySupport",
                    author_is_bot=True,
                    content="(attachment only)",
                    attachments=("https://cdn.example/test.png",),
                ),
                TranscriptMessage(
                    id="3",
                    timestamp="2026-03-22T11:10:00.000000+00:00",
                    author_id="300",
                    author_name="Contributor",
                    author_is_bot=False,
                    content="looking into it",
                    attachments=(),
                ),
            ]
        )

        self.assertIn("Speakers:", transcript)
        self.assertIn("- ticket_user: User", transcript)
        self.assertIn("- support_bot: ySupport", transcript)
        self.assertIn("- human_contributor: Contributor", transcript)
        self.assertIn("[2026-03-22T11:08:00+00:00] ticket_user(User): hello", transcript)
        self.assertIn("[2026-03-22T11:09:00+00:00] support_bot(ySupport): (attachment only)", transcript)
        self.assertIn("[2026-03-22T11:10:00+00:00] human_contributor(Contributor): looking into it", transcript)
        self.assertIn("attachment: https://cdn.example/test.png", transcript)

    def test_main_emits_json_with_normalized_messages(self) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "ticket_transcript_fetch.fetch_channel_metadata",
                return_value=type("ChannelMetadata", (), {"id": "1484802638158626887", "name": "ticket-json"})(),
            ),
            patch(
                "ticket_transcript_fetch.fetch_channel_messages",
                return_value=[
                    {
                        "id": "1",
                        "timestamp": "2026-03-22T11:08:00.000000+00:00",
                        "content": "hello",
                        "author": {"id": "100", "username": "User", "bot": False},
                        "attachments": [],
                    }
                ],
            ),
        ):
            with redirect_stdout(stdout):
                exit_code = main(["1484802638158626887", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["channel_id"], "1484802638158626887")
        self.assertEqual(payload["channel_name"], "ticket-json")
        self.assertEqual(payload["message_count"], 1)
        self.assertEqual(payload["messages"][0]["author_id"], "100")
        self.assertEqual(payload["messages"][0]["author_name"], "User")
        self.assertEqual(payload["messages"][0]["speaker_role"], "ticket_user")

    def test_main_returns_nonzero_on_invalid_input(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = main(["not-a-channel"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Expected a Discord channel id", stderr.getvalue())
