import tests as _test_environment  # noqa: F401

import io
import unittest
from contextlib import redirect_stderr

from ticket_transcript_fetch import (
    TranscriptMessage,
    main,
    normalize_message,
    render_transcript,
)


class TicketTranscriptFetchTests(unittest.TestCase):
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

    def test_main_returns_nonzero_on_invalid_input(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = main(["not-a-channel"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Expected a Discord channel id", stderr.getvalue())
