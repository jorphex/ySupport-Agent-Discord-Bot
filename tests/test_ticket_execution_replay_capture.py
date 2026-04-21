import unittest

from ticket_execution_replay_capture import (
    build_first_turn_transport_request,
    infer_initial_button_intent,
)
from ticket_transcript_fetch import TranscriptMessage


class TicketExecutionReplayCaptureTests(unittest.TestCase):
    def test_infer_initial_button_intent_matches_known_prompts(self) -> None:
        self.assertEqual(
            infer_initial_button_intent(
                "Describe the issue in as much detail as you have. You can send multiple "
                "messages now and I will review them together. Include the product or vault "
                "involved, the exact page or URL if relevant, what you expected to happen, "
                "what actually happened, any error text, any tx hash, and your browser/device "
                "or wallet if that matters."
            ),
            "investigate_issue",
        )
        self.assertEqual(
            infer_initial_button_intent(
                "Okay, please describe your issue or question in detail below. I'll do my best "
                "to assist or find the right help for you."
            ),
            "other_free_form",
        )

    def test_build_first_turn_transport_request_aggregates_user_block_and_intent(self) -> None:
        messages = [
            TranscriptMessage(
                id="1",
                timestamp="2026-04-01T00:00:00+00:00",
                author_id="bot",
                author_name="ySupport",
                author_is_bot=True,
                content="Welcome",
                attachments=(),
            ),
            TranscriptMessage(
                id="2",
                timestamp="2026-04-01T00:00:01+00:00",
                author_id="bot",
                author_name="ySupport",
                author_is_bot=True,
                content=(
                    "Describe the issue in as much detail as you have. You can send multiple "
                    "messages now and I will review them together. Include the product or vault "
                    "involved, the exact page or URL if relevant, what you expected to happen, "
                    "what actually happened, any error text, any tx hash, and your browser/device "
                    "or wallet if that matters."
                ),
                attachments=(),
            ),
            TranscriptMessage(
                id="3",
                timestamp="2026-04-01T00:00:02+00:00",
                author_id="user",
                author_name="User",
                author_is_bot=False,
                content="vault has not updated",
                attachments=(),
            ),
            TranscriptMessage(
                id="4",
                timestamp="2026-04-01T00:00:03+00:00",
                author_id="user",
                author_name="User",
                author_is_bot=False,
                content="why?",
                attachments=(),
            ),
        ]

        request = build_first_turn_transport_request(
            channel_id="123",
            messages=messages,
        )

        self.assertEqual(request.aggregated_text, "vault has not updated\nwhy?")
        self.assertEqual(request.current_history, [])
        self.assertEqual(request.run_context["initial_button_intent"], "investigate_issue")
        self.assertEqual(request.investigation_job["requested_intent"], "investigate_issue")
        self.assertEqual(request.investigation_job["mode"], "collecting")

    def test_build_first_turn_transport_request_rejects_later_turn_target(self) -> None:
        messages = [
            TranscriptMessage(
                id="1",
                timestamp="2026-04-01T00:00:00+00:00",
                author_id="bot",
                author_name="ySupport",
                author_is_bot=True,
                content="Welcome",
                attachments=(),
            ),
            TranscriptMessage(
                id="2",
                timestamp="2026-04-01T00:00:01+00:00",
                author_id="bot",
                author_name="ySupport",
                author_is_bot=True,
                content=(
                    "Okay, please describe your issue or question in detail below. I'll do my "
                    "best to assist or find the right help for you."
                ),
                attachments=(),
            ),
            TranscriptMessage(
                id="3",
                timestamp="2026-04-01T00:00:02+00:00",
                author_id="user",
                author_name="User",
                author_is_bot=False,
                content="first turn",
                attachments=(),
            ),
            TranscriptMessage(
                id="4",
                timestamp="2026-04-01T00:00:03+00:00",
                author_id="bot",
                author_name="ySupport",
                author_is_bot=True,
                content="Got it — updating based on your latest message.",
                attachments=(),
            ),
            TranscriptMessage(
                id="5",
                timestamp="2026-04-01T00:00:04+00:00",
                author_id="user",
                author_name="User",
                author_is_bot=False,
                content="later turn",
                attachments=(),
            ),
        ]

        with self.assertRaises(ValueError):
            build_first_turn_transport_request(
                channel_id="123",
                messages=messages,
                target_message_id="5",
            )
