import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from knowledge_gap_worker import (
    KnowledgeGapReport,
    PreparedTicketTranscript,
    _build_report_signature,
    _snowflake_sort_key,
    discover_recent_closed_ticket_channels,
    _is_closed_ticket,
    _main_async,
    analyze_transcript_for_knowledge_gap,
    finalize_knowledge_gap_report,
    format_knowledge_gap_report,
    post_report_message,
    prepare_ticket_transcript,
    process_ticket,
    process_tickets,
)

TEST_REPORT_CHANNEL_ID = 999999999999999999


class KnowledgeGapWorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_prepare_ticket_transcript_fetches_and_renders_messages(self) -> None:
        with (
            patch(
                "knowledge_gap_worker.fetch_channel_metadata",
                return_value=type("ChannelMetadata", (), {"id": "1484802638158626887", "name": "ticket-styfi"})(),
            ),
            patch(
                "knowledge_gap_worker.fetch_channel_messages",
                return_value=[
                    {
                        "id": "1",
                        "timestamp": "2026-03-23T12:00:00.000000+00:00",
                        "content": "Where do I see migrated stYFI?",
                        "author": {"username": "User", "bot": False},
                        "attachments": [],
                    }
                ],
            ),
        ):
            prepared = prepare_ticket_transcript("1484802638158626887", limit=10)

        self.assertEqual(prepared.channel_id, "1484802638158626887")
        self.assertEqual(prepared.channel_name, "ticket-styfi")
        self.assertEqual(prepared.message_count, 1)
        self.assertIn("Where do I see migrated stYFI?", prepared.transcript_text)

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

    async def test_analyze_transcript_returns_none_when_candidate_is_not_reportable(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1484802638158626887",
            channel_name="ticket-thanks",
            message_count=1,
            transcript_text="[2026-03-23T12:00:00+00:00] User: thanks",
        )

        class FakeCandidate:
            reportable = False
            category = "no_action"

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            return FakeCandidate()

        with patch("knowledge_gap_worker._run_structured_agent", new=fake_run_structured_agent):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNone(result)

    async def test_analyze_transcript_builds_grounded_report(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1484802638158626887",
            channel_name="ticket-styfi",
            message_count=4,
            transcript_text="[2026-03-23T12:00:00+00:00] User: where do i see migrated styfi?",
        )

        candidate = type(
            "Candidate",
            (),
            {
                "reportable": True,
                "category": "faq_candidate",
                "grounding_query": "Where do I see migrated stYFI after veYFI migration?",
                "needs_repo_context": False,
                "model_dump_json": lambda self, indent=2: json.dumps(
                    {
                        "category": "faq_candidate",
                        "grounding_query": "Where do I see migrated stYFI after veYFI migration?",
                    },
                    indent=indent,
                ),
            },
        )()
        report = KnowledgeGapReport(
            should_post=True,
            category="faq_candidate",
            title="stYFI dashboard visibility confusion",
            topic="Where migrated stYFI balances appear",
            product="stYFI",
            chain="ethereum",
            evidence_summary="Users ask where migrated balances appear after veYFI migration.",
            current_official_grounding="Official docs point users to styfi.yearn.fi for stYFI.",
            assessment="Source exists but should be surfaced more clearly in support and docs navigation.",
            suggested_action="FAQ entry and support macro update.",
            confidence="high",
        )

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            return report

        with (
            patch("knowledge_gap_worker._run_structured_agent", new=fake_run_structured_agent),
            patch(
                "knowledge_gap_worker.tools_lib.core_answer_from_docs",
                return_value="Official docs point users to styfi.yearn.fi for stYFI.",
            ),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.title, "stYFI dashboard visibility confusion")
        self.assertEqual(result.category, "faq_candidate")

    def test_format_knowledge_gap_report_renders_internal_fields(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="docs_gap",
            title="Missing recovery docs",
            topic="yETH recovery vault explanations",
            product="yETH",
            chain="ethereum",
            evidence_summary="Users keep asking if they can reclaim 1:1 and why the recovery vault exists.",
            current_official_grounding="Official docs are thin on the exact recovery explanation.",
            assessment="Source gap likely exists.",
            suggested_action="Docs update.",
            confidence="medium",
        )

        formatted = format_knowledge_gap_report(
            report,
            affected_channels=[
                PreparedTicketTranscript(channel_id="123", channel_name="ticket-alpha", message_count=1, transcript_text=""),
                PreparedTicketTranscript(channel_id="456", channel_name="ticket-beta", message_count=1, transcript_text=""),
            ],
        )

        self.assertIn("**Knowledge-Gap Report**", formatted)
        self.assertIn("**Affected tickets:** <#123> (123), <#456> (456)", formatted)
        self.assertIn("**Suggested action**", formatted)
        self.assertIn("**Confidence:** medium", formatted)

    def test_finalize_knowledge_gap_report_uses_supported_chain_hint_from_transcript(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="product_confusion",
            title="Legacy vault visibility",
            topic="Legacy vault discovery",
            product="Yearn UI",
            chain="Avalanche",
            evidence_summary="User cited chain 42161 while reporting a missing legacy vault position.",
            current_official_grounding="Docs mention yPort Bot but not a legacy UI path.",
            assessment="The chain label should come from transcript evidence, not model guesswork.",
            suggested_action="Document supported legacy discovery paths.",
            confidence="high",
        )
        transcript = PreparedTicketTranscript(
            channel_id="1484632286216454295",
            channel_name="ticket-1450",
            message_count=3,
            transcript_text="User says the position is on chain 42161 and only visible in the legacy UI.",
        )

        with patch("knowledge_gap_worker._supported_chain_id_map", return_value={42161: "arbitrum"}):
            finalized = finalize_knowledge_gap_report(report, transcript)

        self.assertEqual(finalized.chain, "arbitrum")

    def test_finalize_knowledge_gap_report_drops_unsupported_chain_without_hint(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="docs_gap",
            title="Unsupported chain label",
            topic="Bad chain label",
            product="Yearn UI",
            chain="Avalanche",
            evidence_summary="No transcript chain evidence.",
            current_official_grounding="No official chain grounding provided.",
            assessment="Unsupported chain names should not survive finalization.",
            suggested_action="Drop unverified chain labels.",
            confidence="medium",
        )
        transcript = PreparedTicketTranscript(
            channel_id="1",
            channel_name="ticket-one",
            message_count=1,
            transcript_text="User says they cannot find a legacy vault.",
        )

        with patch("knowledge_gap_worker._supported_chain_id_map", return_value={42161: "arbitrum"}):
            finalized = finalize_knowledge_gap_report(report, transcript)

        self.assertIsNone(finalized.chain)

    def test_post_report_message_sends_all_chunks(self) -> None:
        sent_payloads = []

        def fake_post_json(url: str, payload):
            sent_payloads.append((url, payload))
            return {"id": str(len(sent_payloads))}

        long_text = "a" * 4500
        with patch("knowledge_gap_worker.discord_post_json", new=fake_post_json):
            post_report_message(TEST_REPORT_CHANNEL_ID, long_text)

        self.assertGreaterEqual(len(sent_payloads), 3)
        self.assertTrue(
            all(
                url.endswith("/channels/TEST_REPORT_CHANNEL_ID/messages") and payload.get("flags") == 4
                for url, payload in sent_payloads
            )
        )

    def test_build_report_signature_is_stable_for_equivalent_content(self) -> None:
        report_a = KnowledgeGapReport(
            should_post=True,
            category="docs_gap",
            title="Missing Recovery Docs",
            topic="yETH recovery vault explanations",
            product="yETH",
            chain="ethereum",
            evidence_summary="One wording.",
            current_official_grounding="Grounding A.",
            assessment="Assessment A.",
            suggested_action="Docs update.",
            confidence="medium",
        )
        report_b = KnowledgeGapReport(
            should_post=True,
            category="docs_gap",
            title="missing recovery docs",
            topic="YETH recovery vault explanations",
            product="yeth",
            chain="Ethereum",
            evidence_summary="Different wording.",
            current_official_grounding="Grounding B.",
            assessment="Assessment B.",
            suggested_action="docs update.",
            confidence="high",
        )

        self.assertEqual(_build_report_signature(report_a), _build_report_signature(report_b))

    def test_snowflake_sort_key_handles_missing_or_invalid_values(self) -> None:
        self.assertEqual(_snowflake_sort_key(None), 0)
        self.assertEqual(_snowflake_sort_key("not-a-snowflake"), 0)
        self.assertEqual(_snowflake_sort_key("123"), 123)

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

    async def test_process_ticket_dry_run_returns_formatted_report_without_posting(self) -> None:
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

        with unittest.mock.patch(
            "knowledge_gap_worker.prepare_ticket_transcript",
            return_value=PreparedTicketTranscript("1", "closed-1", 3, "ticket one"),
        ), unittest.mock.patch(
            "knowledge_gap_worker.analyze_transcript_for_knowledge_gap",
            return_value=report,
        ), unittest.mock.patch(
            "knowledge_gap_worker.post_report_message"
        ) as mock_post, unittest.mock.patch(
            "knowledge_gap_worker._load_reported_signatures",
            return_value={expected_signature},
        ), unittest.mock.patch(
            "knowledge_gap_worker._save_reported_signatures"
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

    async def test_main_accepts_recent_closed_discovery_without_explicit_channels(self) -> None:
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
