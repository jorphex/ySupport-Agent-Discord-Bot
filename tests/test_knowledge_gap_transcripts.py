import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import patch

from knowledge_gap_worker import (
    KnowledgeGapCandidate,
    PreparedTicketTranscript,
    _build_repo_grounding_query,
    _is_closed_ticket,
    prepare_ticket_transcript,
)


class KnowledgeGapTranscriptTests(unittest.TestCase):
    def test_build_repo_grounding_query_combines_concrete_claim_detail(self) -> None:
        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="issue_draft_candidate",
            title="VaultV3 accounting issue",
            topic="VaultV3 _redeem / _update_debt balance-snapshot accounting",
            product="VaultV3.vy",
            chain=None,
            grounding_query="Assess whether VaultV3 withdrawal accounting can be manipulated.",
            evidence_summary="Reporter cites pre/post balance snapshots around redeem() and fake profit / PPS inflation risk.",
            suggested_action="Engineering review.",
            needs_repo_context=True,
        )

        repo_query = _build_repo_grounding_query(candidate)

        self.assertIn(
            "Assess whether VaultV3 withdrawal accounting can be manipulated.",
            repo_query,
        )
        self.assertIn(
            "VaultV3 _redeem / _update_debt balance-snapshot accounting", repo_query
        )
        self.assertIn("fake profit / PPS inflation risk", repo_query)

    def test_prepare_ticket_transcript_fetches_and_renders_messages(self) -> None:
        with (
            patch(
                "knowledge_gap_worker.fetch_channel_metadata",
                return_value=type(
                    "ChannelMetadata",
                    (),
                    {"id": "1484802638158626887", "name": "ticket-styfi"},
                )(),
            ),
            patch(
                "knowledge_gap_worker.fetch_channel_messages",
                return_value=[
                    {
                        "id": "1",
                        "timestamp": "2026-03-23T12:00:00.000000+00:00",
                        "content": "Where do I see migrated stYFI?",
                        "author": {"id": "100", "username": "User", "bot": False},
                        "attachments": [],
                    },
                    {
                        "id": "2",
                        "timestamp": "2026-03-23T12:01:00.000000+00:00",
                        "content": "Check the stYFI dashboard.",
                        "author": {"id": "200", "username": "ySupport", "bot": True},
                        "attachments": [],
                    },
                    {
                        "id": "3",
                        "timestamp": "2026-03-23T12:02:00.000000+00:00",
                        "content": "I can help if needed.",
                        "author": {
                            "id": "300",
                            "username": "Contributor",
                            "bot": False,
                        },
                        "attachments": [],
                    },
                ],
            ),
        ):
            prepared = prepare_ticket_transcript("1484802638158626887", limit=10)

        self.assertEqual(prepared.channel_id, "1484802638158626887")
        self.assertEqual(prepared.channel_name, "ticket-styfi")
        self.assertEqual(prepared.message_count, 3)
        self.assertIn("Speakers:", prepared.transcript_text)
        self.assertIn("- ticket_user: User", prepared.transcript_text)
        self.assertIn("- support_bot: ySupport", prepared.transcript_text)
        self.assertIn("- human_contributor: Contributor", prepared.transcript_text)
        self.assertIn(
            "ticket_user(User): Where do I see migrated stYFI?",
            prepared.transcript_text,
        )

    def test_is_closed_ticket_uses_closed_channel_prefix(self) -> None:
        self.assertTrue(
            _is_closed_ticket(
                PreparedTicketTranscript(
                    channel_id="1",
                    channel_name="closed-1450",
                    message_count=0,
                    transcript_text="",
                )
            )
        )
        self.assertFalse(
            _is_closed_ticket(
                PreparedTicketTranscript(
                    channel_id="2",
                    channel_name="ticket-1451",
                    message_count=0,
                    transcript_text="",
                )
            )
        )
