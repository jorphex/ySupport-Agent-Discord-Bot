TEST_REPORT_CHANNEL_ID = 999999999999999999

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from knowledge_gap_worker import (
    KnowledgeGapReport,
    PreparedTicketTranscript,
    _main_async,
    analyze_transcript_for_knowledge_gap,
    format_knowledge_gap_report,
    post_report_message,
    prepare_ticket_transcript,
    process_ticket,
)


class KnowledgeGapWorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_prepare_ticket_transcript_fetches_and_renders_messages(self) -> None:
        with patch(
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
        ):
            prepared = prepare_ticket_transcript("1484802638158626887", limit=10)

        self.assertEqual(prepared.channel_id, "1484802638158626887")
        self.assertEqual(prepared.message_count, 1)
        self.assertIn("Where do I see migrated stYFI?", prepared.transcript_text)

    async def test_analyze_transcript_returns_none_when_candidate_is_not_reportable(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1484802638158626887",
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

        formatted = format_knowledge_gap_report(report, affected_channels=["123", "456"])

        self.assertIn("Knowledge-gap report", formatted)
        self.assertIn("Affected tickets: 123, 456", formatted)
        self.assertIn("Suggested action", formatted)

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
                url.endswith("/channels/TEST_REPORT_CHANNEL_ID/messages")
                for url, _payload in sent_payloads
            )
        )

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


class KnowledgeGapWorkerCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_dry_run_emits_json(self) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "knowledge_gap_worker.process_ticket",
                return_value={
                    "channel_id": "1484802638158626887",
                    "message_count": 1,
                    "report_posted": False,
                    "report": None,
                },
            ) as mock_process_ticket,
            redirect_stdout(stdout),
        ):
            exit_code = await _main_async(["1484802638158626887", "--dry-run"])

        self.assertEqual(exit_code, 0)
        mock_process_ticket.assert_called_once()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["results"][0]["channel_id"], "1484802638158626887")
