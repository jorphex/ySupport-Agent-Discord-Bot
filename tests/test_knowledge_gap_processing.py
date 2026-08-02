import tests as _test_environment  # noqa: F401

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from knowledge_gap_worker import (
    KnowledgeGapReport,
    PreparedTicketTranscript,
    _build_report_signature,
    discover_recent_closed_ticket_channels,
    preview_recent_closed_ticket_channels,
    _main_async,
    process_ticket,
    process_tickets,
)

TEST_REPORT_CHANNEL_ID = 999999999999999999


class KnowledgeGapWorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_discover_recent_closed_ticket_channels_prefers_same_parent_and_sorts_recent_first(
        self,
    ) -> None:
        guild_channels = [
            {
                "id": "10",
                "name": "closed-old",
                "type": 0,
                "parent_id": "999",
                "last_message_id": "100",
            },
            {
                "id": "20",
                "name": "closed-newer",
                "type": 0,
                "parent_id": "123",
                "last_message_id": "300",
            },
            {
                "id": "30",
                "name": "closed-newest",
                "type": 0,
                "parent_id": "123",
                "last_message_id": "400",
            },
            {
                "id": "40",
                "name": "ticket-open",
                "type": 0,
                "parent_id": "123",
                "last_message_id": "999",
            },
        ]

        with (
            patch("knowledge_gap_worker.config.YEARN_TICKET_CATEGORY_ID", 123),
            patch(
                "knowledge_gap_worker.discord_get_json",
                side_effect=[{"guild_id": "guild-1"}, guild_channels],
            ),
        ):
            channels = discover_recent_closed_ticket_channels(2)

        self.assertEqual(channels, ["30", "20"])

    def test_preview_recent_closed_ticket_channels_resolves_channel_names(self) -> None:
        with (
            patch(
                "knowledge_gap_worker.discover_recent_closed_ticket_channels",
                return_value=["30", "20"],
            ),
            patch(
                "knowledge_gap_worker.fetch_channel_metadata",
                side_effect=[
                    type("ChannelMetadata", (), {"id": "30", "name": "closed-30"})(),
                    type("ChannelMetadata", (), {"id": "20", "name": "closed-20"})(),
                ],
            ),
        ):
            preview = preview_recent_closed_ticket_channels(2)

        self.assertEqual(
            preview,
            [
                {"channel_id": "30", "channel_name": "closed-30"},
                {"channel_id": "20", "channel_name": "closed-20"},
            ],
        )

    async def test_process_ticket_dry_run_returns_formatted_report_without_posting(
        self,
    ) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="bot_behavior_gap",
            title="Legacy link directness gap",
            topic="Legacy UI navigation",
            product="legacy UI",
            chain=None,
            evidence_summary="The bot asked for bug details instead of returning the known legacy link.",
            current_official_grounding="Official sources already provide the destination.",
            assessment="Bot directness issue, not missing docs.",
            suggested_action="Support answer policy tweak.",
            confidence="high",
        )

        with (
            patch(
                "knowledge_gap_worker.prepare_ticket_transcript",
                return_value=PreparedTicketTranscript(
                    channel_id="1484632286216454295",
                    channel_name="ticket-legacy-link",
                    message_count=5,
                    transcript_text="...",
                ),
            ),
            patch(
                "knowledge_gap_worker.analyze_transcript_for_knowledge_gap",
                return_value=report,
            ),
            patch("knowledge_gap_worker.post_report_message") as mock_post,
        ):
            result = await process_ticket(
                "1484632286216454295",
                limit=80,
                report_channel_id=TEST_REPORT_CHANNEL_ID,
                dry_run=True,
            )

        self.assertFalse(result["report_posted"])
        self.assertIn("formatted_report", result)
        mock_post.assert_not_called()

    async def test_process_ticket_skips_open_ticket_when_closed_only(self) -> None:
        with patch(
            "knowledge_gap_worker.prepare_ticket_transcript",
            return_value=PreparedTicketTranscript(
                channel_id="1484632286216454295",
                channel_name="ticket-legacy-link",
                message_count=5,
                transcript_text="...",
            ),
        ):
            result = await process_ticket(
                "1484632286216454295",
                limit=80,
                report_channel_id=TEST_REPORT_CHANNEL_ID,
                dry_run=True,
                closed_only=True,
            )

        self.assertEqual(result["skipped_reason"], "open_ticket")
        self.assertFalse(result["report_posted"])

    async def test_process_tickets_groups_matching_reports_within_run(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="faq_candidate",
            title="Legacy UI discovery gap",
            topic="Legacy positions URL",
            product="legacy UI",
            chain="arbitrum",
            evidence_summary="Users cannot find legacy positions.",
            current_official_grounding="Official destination exists.",
            assessment="This is recurring confusion.",
            suggested_action="FAQ entry.",
            confidence="high",
        )
        prepared_transcripts = [
            PreparedTicketTranscript("1", "closed-1", 3, "ticket one"),
            PreparedTicketTranscript("2", "closed-2", 4, "ticket two"),
        ]

        with (
            patch(
                "knowledge_gap_worker.prepare_ticket_transcript",
                side_effect=prepared_transcripts,
            ),
            patch(
                "knowledge_gap_worker.analyze_transcript_for_knowledge_gap",
                return_value=report,
            ),
            patch("knowledge_gap_worker.post_report_message") as mock_post,
        ):
            results = await process_tickets(
                ["1", "2"],
                limit=80,
                report_channel_id=TEST_REPORT_CHANNEL_ID,
                dry_run=False,
                closed_only=True,
                state_path=None,
                max_posts=None,
            )

        self.assertTrue(results[0]["report_posted"])
        self.assertTrue(results[1]["report_posted"])
        self.assertEqual(results[0]["group_size"], 2)
        self.assertEqual(results[1]["group_size"], 2)
        self.assertIn("<#1> (1), <#2> (2)", results[0]["formatted_report"])
        mock_post.assert_called_once()

    async def test_process_tickets_uses_state_file_for_dedupe(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="faq_candidate",
            title="Legacy UI discovery gap",
            topic="Legacy positions URL",
            product="legacy UI",
            chain="arbitrum",
            evidence_summary="Users cannot find legacy positions.",
            current_official_grounding="Official destination exists.",
            assessment="This is recurring confusion.",
            suggested_action="FAQ entry.",
            confidence="high",
        )
        expected_signature = _build_report_signature(report)

        with (
            unittest.mock.patch(
                "knowledge_gap_worker.prepare_ticket_transcript",
                return_value=PreparedTicketTranscript("1", "closed-1", 3, "ticket one"),
            ),
            unittest.mock.patch(
                "knowledge_gap_worker.analyze_transcript_for_knowledge_gap",
                return_value=report,
            ),
            unittest.mock.patch(
                "knowledge_gap_worker.post_report_message"
            ) as mock_post,
            unittest.mock.patch(
                "knowledge_gap_worker._load_reported_signatures",
                return_value={expected_signature},
            ),
            unittest.mock.patch("knowledge_gap_worker._save_reported_signatures"),
        ):
            results = await process_tickets(
                ["1"],
                limit=80,
                report_channel_id=TEST_REPORT_CHANNEL_ID,
                dry_run=False,
                closed_only=True,
                state_path="state.json",
                max_posts=None,
            )

        self.assertEqual(results[0]["skipped_reason"], "duplicate_report")
        mock_post.assert_not_called()

    async def test_process_tickets_respects_max_posts_limit(self) -> None:
        reports = [
            KnowledgeGapReport(
                should_post=True,
                category="faq_candidate",
                title="Legacy UI discovery gap",
                topic="Legacy positions URL",
                product="legacy UI",
                chain="arbitrum",
                evidence_summary="Users cannot find legacy positions.",
                current_official_grounding="Official destination exists.",
                assessment="This is recurring confusion.",
                suggested_action="FAQ entry.",
                confidence="high",
            ),
            KnowledgeGapReport(
                should_post=True,
                category="docs_gap",
                title="Missing recovery docs",
                topic="yETH recovery vault explanations",
                product="yETH",
                chain="ethereum",
                evidence_summary="Users keep asking if they can reclaim 1:1.",
                current_official_grounding="Docs are thin.",
                assessment="This is a docs gap.",
                suggested_action="Docs update.",
                confidence="medium",
            ),
        ]
        prepared_transcripts = [
            PreparedTicketTranscript("1", "closed-1", 3, "ticket one"),
            PreparedTicketTranscript("2", "closed-2", 4, "ticket two"),
        ]

        with (
            patch(
                "knowledge_gap_worker.prepare_ticket_transcript",
                side_effect=prepared_transcripts,
            ),
            patch(
                "knowledge_gap_worker.analyze_transcript_for_knowledge_gap",
                side_effect=reports,
            ),
            patch("knowledge_gap_worker.post_report_message") as mock_post,
        ):
            results = await process_tickets(
                ["1", "2"],
                limit=80,
                report_channel_id=TEST_REPORT_CHANNEL_ID,
                dry_run=False,
                closed_only=True,
                state_path=None,
                max_posts=1,
            )

        self.assertTrue(results[0]["report_posted"])
        self.assertEqual(results[1]["skipped_reason"], "max_posts_reached")
        mock_post.assert_called_once()


class KnowledgeGapWorkerCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_dry_run_emits_json(self) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "knowledge_gap_worker.process_tickets",
                return_value=[
                    {
                        "channel_id": "1484802638158626887",
                        "message_count": 1,
                        "report_posted": False,
                        "report": None,
                    }
                ],
            ) as mock_process_tickets,
            redirect_stdout(stdout),
        ):
            exit_code = await _main_async(["1484802638158626887", "--dry-run"])

        self.assertEqual(exit_code, 0)
        mock_process_tickets.assert_called_once()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["results"][0]["channel_id"], "1484802638158626887")

    async def test_main_accepts_recent_closed_discovery_without_explicit_channels(
        self,
    ) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "knowledge_gap_worker.discover_recent_closed_ticket_channels",
                return_value=["111", "222"],
            ) as mock_discover,
            patch(
                "knowledge_gap_worker.process_tickets",
                return_value=[],
            ) as mock_process_tickets,
            redirect_stdout(stdout),
        ):
            exit_code = await _main_async(["--recent-closed", "2", "--dry-run"])

        self.assertEqual(exit_code, 0)
        mock_discover.assert_called_once_with(2)
        mock_process_tickets.assert_called_once()
        self.assertEqual(json.loads(stdout.getvalue())["results"], [])

    async def test_main_preview_discovery_exits_before_processing(self) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "knowledge_gap_worker.discover_recent_closed_ticket_channels",
                return_value=["111", "222"],
            ),
            patch(
                "knowledge_gap_worker.preview_recent_closed_ticket_channels",
                return_value=[
                    {"channel_id": "111", "channel_name": "closed-111"},
                    {"channel_id": "222", "channel_name": "closed-222"},
                ],
            ) as mock_preview,
            patch("knowledge_gap_worker.process_tickets") as mock_process_tickets,
            redirect_stdout(stdout),
        ):
            exit_code = await _main_async(
                ["--recent-closed", "2", "--preview-discovery", "--dry-run"]
            )

        self.assertEqual(exit_code, 0)
        mock_preview.assert_called_once_with(2)
        mock_process_tickets.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["preview_only"])
        self.assertEqual(payload["selected_channels"], ["111", "222"])
