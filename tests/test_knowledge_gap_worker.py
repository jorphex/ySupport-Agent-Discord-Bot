import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from knowledge_gap_worker import (
    BotBehaviorGapExtraction,
    DocsGapExtraction,
    FaqCandidateExtraction,
    IssueDraftExtraction,
    ProductConfusionExtraction,
    KnowledgeGapReport,
    PreparedTicketTranscript,
    KnowledgeGapCandidate,
    _build_report_signature,
    _build_repo_grounding_query,
    _should_fetch_repo_grounding,
    _snowflake_sort_key,
    discover_recent_closed_ticket_channels,
    preview_recent_closed_ticket_channels,
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

        self.assertIn("Assess whether VaultV3 withdrawal accounting can be manipulated.", repo_query)
        self.assertIn("VaultV3 _redeem / _update_debt balance-snapshot accounting", repo_query)
        self.assertIn("fake profit / PPS inflation risk", repo_query)

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
                        "author": {"id": "300", "username": "Contributor", "bot": False},
                        "attachments": [],
                    }
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
        self.assertIn("ticket_user(User): Where do I see migrated stYFI?", prepared.transcript_text)

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

        with patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent):
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
        extraction = FaqCandidateExtraction(
            user_problem="Where do migrated stYFI balances appear after veYFI migration?",
            faq_answer="Migrated balances appear on the stYFI dashboard.",
            recurrence_signal="Users repeatedly ask where migrated balances appear after migration.",
            current_workaround="Support or contributors manually point users to the stYFI dashboard.",
            recommended_owner="docs",
        )
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
            if workflow_name == "Knowledge Gap FAQ Extraction":
                return extraction
            return report

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch(
                "knowledge_gap_reporting.tools_lib.core_answer_from_docs",
                return_value="Official docs point users to styfi.yearn.fi for stYFI.",
            ),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.title, "stYFI dashboard visibility confusion")
        self.assertEqual(result.category, "faq_candidate")

    async def test_analyze_transcript_passes_candidate_topic_and_evidence_into_repo_grounding(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1479167148310925425",
            channel_name="closed-1433",
            message_count=4,
            transcript_text="[2026-03-05T17:23:55+00:00] User: VaultV3 withdrawal accounting issue.",
        )

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
        report = KnowledgeGapReport(
            should_post=True,
            category="issue_draft_candidate",
            title="Potential VaultV3 issue",
            topic="VaultV3 accounting",
            product="VaultV3.vy",
            chain=None,
            evidence_summary="Reporter shared a concrete VaultV3 accounting concern.",
            current_official_grounding="Repo and docs grounding exist for the cited path.",
            assessment="This looks plausible enough for internal review.",
            suggested_action="Open an internal engineering/security triage thread.",
            confidence="medium",
        )
        extraction = IssueDraftExtraction(
            reported_issue="Reporter claims VaultV3 withdrawal accounting may allow PPS inflation.",
            plausibility_basis="Repo grounding matches the cited accounting path and concrete mechanics.",
            blocking_unknown="Need engineering/security review to confirm exploitability in practice.",
            immediate_triage_need="Open an internal engineering/security triage thread.",
            recommended_owner="engineering",
        )
        repo_queries: list[str] = []

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Issue Draft Extraction":
                return extraction
            return report

        async def fake_pretriage_repo_claim(claim_text: str, *, include_docs: bool = True, limit=None, include_legacy=False, include_ui=False) -> str:
            repo_queries.append(claim_text)
            return "Repo grounding"

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
            patch("knowledge_gap_reporting.tools_lib.core_pretriage_repo_claim", new=fake_pretriage_repo_claim),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        self.assertEqual(len(repo_queries), 1)
        self.assertIn(candidate.grounding_query, repo_queries[0])
        self.assertIn(candidate.topic, repo_queries[0])
        self.assertIn(candidate.evidence_summary, repo_queries[0])

    async def test_analyze_transcript_fetches_repo_grounding_for_issue_draft_when_model_flag_is_false(
        self,
    ) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1479167148310925425",
            channel_name="closed-1433",
            message_count=4,
            transcript_text="[2026-03-05T17:23:55+00:00] User: VaultV3 withdrawal accounting issue.",
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="issue_draft_candidate",
            title="Potential VaultV3 issue",
            topic="VaultV3 accounting",
            product="VaultV3.vy",
            chain=None,
            grounding_query="Assess whether VaultV3 accounting is safe.",
            evidence_summary="Reporter claims PPS inflation.",
            suggested_action="Engineering review.",
            needs_repo_context=False,
        )
        report = KnowledgeGapReport(
            should_post=True,
            category="issue_draft_candidate",
            title="Potential VaultV3 issue",
            topic="VaultV3 accounting",
            product="VaultV3.vy",
            chain=None,
            evidence_summary="Reporter raised a concrete VaultV3 accounting concern.",
            current_official_grounding="Repo and docs grounding exist for the cited path.",
            assessment="This should be reviewed internally.",
            suggested_action="Open an internal engineering/security triage thread.",
            confidence="medium",
        )
        extraction = IssueDraftExtraction(
            reported_issue="Reporter claims VaultV3 accounting is unsafe.",
            plausibility_basis="The report contains a concrete accounting claim tied to VaultV3.",
            blocking_unknown="Need engineering/security review to determine practical exploitability.",
            immediate_triage_need="Open an internal engineering/security triage thread.",
            recommended_owner="engineering",
        )
        repo_queries: list[str] = []

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Issue Draft Extraction":
                return extraction
            return report

        async def fake_pretriage_repo_claim(claim_text: str, *, include_docs: bool = True, limit=None, include_legacy=False, include_ui=False) -> str:
            repo_queries.append(claim_text)
            return "Repo grounding"

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
            patch("knowledge_gap_reporting.tools_lib.core_pretriage_repo_claim", new=fake_pretriage_repo_claim),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        self.assertEqual(len(repo_queries), 1)

    async def test_analyze_transcript_runs_bot_behavior_extraction_and_preserves_structured_fields(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1489664842175615199",
            channel_name="ticket-1473",
            message_count=6,
            transcript_text=(
                "Speakers:\n"
                "- ticket_user: Astral Protocol\n"
                "- support_bot: ySupport\n"
                "- human_contributor: J\n\n"
                "[2026-04-03T16:36:43+00:00] ticket_user(Astral Protocol): why hasn't the vault updated?\n"
                "[2026-04-03T16:43:49+00:00] support_bot(ySupport): open a ticket in Yearn Discord\n"
                "[2026-04-03T16:50:49+00:00] human_contributor(J): I looked at both active strategies.\n"
            ),
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="bot_behavior_gap",
            title="Bot gave bad escalation advice",
            topic="stale PPS escalation in Discord",
            product="Yearn Vaults",
            chain=None,
            grounding_query="How should support handle a flat PPS / stale report ticket already inside Discord?",
            evidence_summary="Bot told the user to open Discord while already in Discord.",
            suggested_action="Fix the bot path.",
            needs_repo_context=False,
        )
        extraction = BotBehaviorGapExtraction(
            user_problem="User wanted to know why PPS had stayed flat and whether that meant the vault was not working.",
            bot_failure="ySupport gave generic guidance and told the user to open a Discord ticket despite already being in one.",
            human_follow_up="A contributor checked the strategies, added current context, and said strategists would be notified.",
            unresolved_risk="Future users may get the same generic redirect instead of an in-channel escalation with context.",
            docs_gap_present=False,
            recommended_owner="bot",
        )
        report = KnowledgeGapReport(
            should_post=True,
            category="bot_behavior_gap",
            title="Bot gave bad escalation advice",
            topic="stale PPS escalation in Discord",
            product="Yearn Vaults",
            chain=None,
            evidence_summary="User reported a concrete stale-PPS support problem.",
            current_official_grounding="Docs describe the mechanics, but the main miss here was the bot's escalation behavior.",
            assessment="This is primarily a bot behavior gap.",
            suggested_action="Keep the user in-channel, surface available status info, and escalate with context.",
            confidence="high",
            user_problem=extraction.user_problem,
            bot_failure=extraction.bot_failure,
            human_follow_up=extraction.human_follow_up,
            unresolved_risk=extraction.unresolved_risk,
            recommended_owner=extraction.recommended_owner,
        )
        final_report_inputs: list[str] = []

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Bot Behavior Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "bot_behavior_gap")
        self.assertEqual(result.user_problem, extraction.user_problem)
        self.assertEqual(result.bot_failure, extraction.bot_failure)
        self.assertEqual(result.human_follow_up, extraction.human_follow_up)
        self.assertEqual(result.unresolved_risk, extraction.unresolved_risk)
        self.assertEqual(result.recommended_owner, "bot")
        self.assertEqual(len(final_report_inputs), 1)
        self.assertIn('"recommended_owner": "bot"', final_report_inputs[0])
        self.assertIn("bot_behavior_gap extraction", final_report_inputs[0])

    async def test_analyze_transcript_backfills_extraction_fields_when_final_report_omits_structured_fields(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1489664842175615199",
            channel_name="ticket-1473",
            message_count=6,
            transcript_text=(
                "Speakers:\n"
                "- ticket_user: Astral Protocol\n"
                "- support_bot: ySupport\n"
                "- human_contributor: J\n\n"
                "[2026-04-03T16:36:43+00:00] ticket_user(Astral Protocol): why hasn't the vault updated?\n"
                "[2026-04-03T16:43:49+00:00] support_bot(ySupport): open a ticket in Yearn Discord\n"
                "[2026-04-03T16:50:49+00:00] human_contributor(J): I looked at both active strategies.\n"
            ),
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="bot_behavior_gap",
            title="Bot gave bad escalation advice",
            topic="stale PPS escalation in Discord",
            product="Yearn Vaults",
            chain=None,
            grounding_query="How should support handle a flat PPS / stale report ticket already inside Discord?",
            evidence_summary="Bot told the user to open Discord while already in Discord.",
            suggested_action="Fix the bot path.",
            needs_repo_context=False,
        )
        extraction = BotBehaviorGapExtraction(
            user_problem="User wanted to know why PPS had stayed flat and whether that meant the vault was not working.",
            bot_failure="ySupport gave generic guidance and told the user to open a Discord ticket despite already being in one.",
            human_follow_up="A contributor checked the strategies, added current context, and said strategists would be notified.",
            unresolved_risk="Future users may get the same generic redirect instead of an in-channel escalation with context.",
            docs_gap_present=True,
            recommended_owner="mixed",
        )
        report = KnowledgeGapReport(
            should_post=True,
            category="bot_behavior_gap",
            title="Bot gave bad escalation advice",
            topic="stale PPS escalation in Discord",
            product="Yearn Vaults",
            chain=None,
            evidence_summary="User saw stale PPS in the UI despite contributor follow-up.",
            current_official_grounding="Official docs explain the mechanics, but the main miss here was the bot's escalation behavior.",
            assessment="This is primarily a bot behavior gap.",
            suggested_action="Investigate the stale-PPS escalation path and improve the bot response.",
            confidence="high",
        )

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Bot Behavior Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "bot_behavior_gap")
        self.assertEqual(result.user_problem, extraction.user_problem)
        self.assertEqual(result.bot_failure, extraction.bot_failure)
        self.assertEqual(result.human_follow_up, extraction.human_follow_up)
        self.assertEqual(result.unresolved_risk, extraction.unresolved_risk)
        self.assertEqual(result.recommended_owner, extraction.recommended_owner)

    async def test_analyze_transcript_runs_docs_gap_extraction_and_backfills_fields(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1484802638158626887",
            channel_name="ticket-styfi",
            message_count=3,
            transcript_text=(
                "Speakers:\n"
                "- ticket_user: User\n"
                "- human_contributor: Contributor\n\n"
                "[2026-03-23T12:00:00+00:00] ticket_user(User): where do I see migrated stYFI?\n"
                "[2026-03-23T12:02:00+00:00] human_contributor(Contributor): use the stYFI dashboard.\n"
            ),
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="docs_gap",
            title="Missing migration visibility docs",
            topic="Where migrated stYFI appears",
            product="stYFI",
            chain=None,
            grounding_query="Where do migrated stYFI balances appear after veYFI migration?",
            evidence_summary="User could not find migrated balance location.",
            suggested_action="Docs update.",
            needs_repo_context=False,
        )
        extraction = DocsGapExtraction(
            user_problem="User needed to know where migrated stYFI balances appear after migration.",
            missing_or_unclear_docs="Official docs do not clearly point users from migration context to the exact dashboard/view where balances appear.",
            current_workaround="A human contributor can answer with the stYFI dashboard path.",
            unresolved_risk="Users will keep opening tickets for a navigation question that should be answered in docs.",
            recommended_owner="docs",
        )
        report = KnowledgeGapReport(
            should_post=True,
            category="docs_gap",
            title="Missing migration visibility docs",
            topic="Where migrated stYFI appears",
            product="stYFI",
            chain=None,
            evidence_summary="User could not find the migrated stYFI balance destination.",
            current_official_grounding="Official docs partially mention stYFI but not this exact discovery path.",
            assessment="This is a real docs gap.",
            suggested_action="Add a direct migration FAQ entry.",
            confidence="high",
        )
        final_report_inputs: list[str] = []

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Docs Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "docs_gap")
        self.assertEqual(result.user_problem, extraction.user_problem)
        self.assertEqual(result.missing_or_unclear_docs, extraction.missing_or_unclear_docs)
        self.assertEqual(result.current_workaround, extraction.current_workaround)
        self.assertEqual(result.unresolved_risk, extraction.unresolved_risk)
        self.assertEqual(result.recommended_owner, "docs")
        self.assertEqual(len(final_report_inputs), 1)
        self.assertIn('"recommended_owner": "docs"', final_report_inputs[0])
        self.assertIn("docs_gap extraction", final_report_inputs[0])

    async def test_analyze_transcript_runs_issue_draft_extraction_and_backfills_fields(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1479167148310925425",
            channel_name="closed-1433",
            message_count=4,
            transcript_text=(
                "Speakers:\n"
                "- ticket_user: Reporter\n\n"
                "[2026-03-05T17:23:55+00:00] ticket_user(Reporter): VaultV3 withdrawal accounting issue.\n"
            ),
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="issue_draft_candidate",
            title="Potential VaultV3 issue",
            topic="VaultV3 accounting",
            product="VaultV3.vy",
            chain=None,
            grounding_query="Assess whether VaultV3 withdrawal accounting can be manipulated.",
            evidence_summary="Reporter claims PPS inflation.",
            suggested_action="Engineering review.",
            needs_repo_context=True,
        )
        extraction = IssueDraftExtraction(
            reported_issue="Reporter claims VaultV3 withdrawal accounting may allow PPS inflation.",
            plausibility_basis="Repo grounding shows balance-diff accounting in the cited path and the report includes concrete mechanics.",
            blocking_unknown="Need engineering/security review to confirm whether the claimed sequence is exploitable in practice.",
            immediate_triage_need="Open an internal engineering/security triage thread with the grounded claim summary.",
            recommended_owner="engineering",
        )
        report = KnowledgeGapReport(
            should_post=True,
            category="issue_draft_candidate",
            title="Potential VaultV3 issue",
            topic="VaultV3 accounting",
            product="VaultV3.vy",
            chain=None,
            evidence_summary="Reporter raised a concrete VaultV3 issue.",
            current_official_grounding="Repo and docs grounding exist for the cited path.",
            assessment="This is plausible enough for internal review.",
            suggested_action="Open an internal issue draft.",
            confidence="high",
        )
        final_report_inputs: list[str] = []

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Issue Draft Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        async def fake_pretriage_repo_claim(claim_text: str, *, include_docs: bool = True, limit=None, include_legacy=False, include_ui=False) -> str:
            return "Repo grounding"

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
            patch("knowledge_gap_reporting.tools_lib.core_pretriage_repo_claim", new=fake_pretriage_repo_claim),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "issue_draft_candidate")
        self.assertEqual(result.reported_issue, extraction.reported_issue)
        self.assertEqual(result.plausibility_basis, extraction.plausibility_basis)
        self.assertEqual(result.blocking_unknown, extraction.blocking_unknown)
        self.assertEqual(result.immediate_triage_need, extraction.immediate_triage_need)
        self.assertEqual(result.recommended_owner, "engineering")
        self.assertEqual(len(final_report_inputs), 1)
        self.assertIn('"recommended_owner": "engineering"', final_report_inputs[0])
        self.assertIn("issue_draft extraction", final_report_inputs[0])

    async def test_analyze_transcript_runs_product_confusion_extraction_and_backfills_fields(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1489664842175615199",
            channel_name="ticket-1473",
            message_count=6,
            transcript_text=(
                "Speakers:\n"
                "- ticket_user: Astral Protocol\n"
                "- human_contributor: J\n\n"
                "[2026-04-03T16:36:43+00:00] ticket_user(Astral Protocol): why hasn't the vault updated?\n"
                "[2026-04-03T16:50:49+00:00] human_contributor(J): I looked at both active strategies.\n"
            ),
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="product_confusion",
            title="PPS visibility confusion",
            topic="vault PPS visibility",
            product="Yearn Vaults",
            chain=None,
            grounding_query="Why can a vault show recent harvest activity while PPS still appears flat in the UI?",
            evidence_summary="User sees stale PPS and thinks vault may be broken.",
            suggested_action="Investigate visibility gap.",
            needs_repo_context=False,
        )
        extraction = ProductConfusionExtraction(
            user_problem="User wanted to understand why the vault looked inactive even though it may still be operating.",
            confusing_behavior="The visible PPS looked stale while later contributor context suggested recent harvest activity.",
            current_workaround="A human contributor can inspect strategy activity and explain the accounting path manually.",
            unresolved_risk="Users may think the vault is broken and lose trust if the visible product state remains confusing.",
            recommended_owner="product",
        )
        report = KnowledgeGapReport(
            should_post=True,
            category="product_confusion",
            title="PPS visibility confusion",
            topic="vault PPS visibility",
            product="Yearn Vaults",
            chain=None,
            evidence_summary="User saw stale PPS and thought the vault might be broken.",
            current_official_grounding="Docs explain the accounting flow but not the visible product behavior clearly enough.",
            assessment="This looks like a real product confusion issue.",
            suggested_action="Investigate the visibility gap and improve the surface explanation.",
            confidence="high",
        )
        final_report_inputs: list[str] = []

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Product Confusion Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "product_confusion")
        self.assertEqual(result.user_problem, extraction.user_problem)
        self.assertEqual(result.confusing_behavior, extraction.confusing_behavior)
        self.assertEqual(result.current_workaround, extraction.current_workaround)
        self.assertEqual(result.unresolved_risk, extraction.unresolved_risk)
        self.assertEqual(result.recommended_owner, "product")
        self.assertEqual(len(final_report_inputs), 1)
        self.assertIn('"recommended_owner": "product"', final_report_inputs[0])
        self.assertIn("product_confusion extraction", final_report_inputs[0])

    async def test_analyze_transcript_runs_faq_candidate_extraction_and_backfills_fields(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1484802638158626887",
            channel_name="ticket-styfi",
            message_count=3,
            transcript_text=(
                "Speakers:\n"
                "- ticket_user: User\n"
                "- human_contributor: Contributor\n\n"
                "[2026-03-23T12:00:00+00:00] ticket_user(User): where do I see migrated stYFI?\n"
                "[2026-03-23T12:02:00+00:00] human_contributor(Contributor): use the stYFI dashboard.\n"
            ),
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="faq_candidate",
            title="stYFI dashboard visibility confusion",
            topic="Where migrated stYFI balances appear",
            product="stYFI",
            chain=None,
            grounding_query="Where do migrated stYFI balances appear after veYFI migration?",
            evidence_summary="Users ask where migrated balances appear after migration.",
            suggested_action="Add a reusable answer.",
            needs_repo_context=False,
        )
        extraction = FaqCandidateExtraction(
            user_problem="Where do migrated stYFI balances appear after veYFI migration?",
            faq_answer="Migrated balances appear on the stYFI dashboard.",
            recurrence_signal="This is a short navigation/discovery question that support can answer the same way each time.",
            current_workaround="A human contributor can manually point users to the stYFI dashboard.",
            recommended_owner="docs",
        )
        report = KnowledgeGapReport(
            should_post=True,
            category="faq_candidate",
            title="stYFI dashboard visibility confusion",
            topic="Where migrated stYFI balances appear",
            product="stYFI",
            chain=None,
            evidence_summary="Users ask where migrated stYFI balances appear after migration.",
            current_official_grounding="Official docs partially point to stYFI but the answer is not surfaced plainly enough.",
            assessment="This deserves a reusable FAQ answer.",
            suggested_action="Add a direct FAQ entry and macro.",
            confidence="high",
        )
        final_report_inputs: list[str] = []

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap FAQ Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "faq_candidate")
        self.assertEqual(result.user_problem, extraction.user_problem)
        self.assertEqual(result.faq_answer, extraction.faq_answer)
        self.assertEqual(result.recurrence_signal, extraction.recurrence_signal)
        self.assertEqual(result.current_workaround, extraction.current_workaround)
        self.assertEqual(result.recommended_owner, "docs")
        self.assertEqual(len(final_report_inputs), 1)
        self.assertIn('"recommended_owner": "docs"', final_report_inputs[0])
        self.assertIn("faq_candidate extraction", final_report_inputs[0])

    async def test_analyze_transcript_drops_placeholder_like_report_after_validation(self) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1484802638158626887",
            channel_name="ticket-styfi",
            message_count=3,
            transcript_text=(
                "Speakers:\n"
                "- ticket_user: User\n"
                "- human_contributor: Contributor\n\n"
                "[2026-03-23T12:00:00+00:00] ticket_user(User): where do I see migrated stYFI?\n"
                "[2026-03-23T12:02:00+00:00] human_contributor(Contributor): use the stYFI dashboard.\n"
            ),
        )

        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="faq_candidate",
            title="stYFI dashboard visibility confusion",
            topic="Where migrated stYFI balances appear",
            product="stYFI",
            chain=None,
            grounding_query="Where do migrated stYFI balances appear after veYFI migration?",
            evidence_summary="Users ask where migrated balances appear after migration.",
            suggested_action="Add a reusable answer.",
            needs_repo_context=False,
        )
        extraction = FaqCandidateExtraction(
            user_problem="Where do migrated stYFI balances appear after veYFI migration?",
            faq_answer="Migrated balances appear on the stYFI dashboard.",
            recurrence_signal="This is a short navigation question that support can answer the same way each time.",
            current_workaround="A human contributor can manually point users to the dashboard.",
            recommended_owner="docs",
        )
        invalid_report = KnowledgeGapReport(
            should_post=True,
            category="faq_candidate",
            title="stYFI dashboard visibility confusion",
            topic="Where migrated stYFI balances appear",
            product="stYFI",
            chain=None,
            evidence_summary="Evidence summary",
            current_official_grounding="Grounding",
            assessment="Assessment",
            suggested_action="Suggested action",
            confidence="high",
        )

        async def fake_run_structured_agent(agent, input_text, output_type, workflow_name):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap FAQ Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                return invalid_report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch("knowledge_gap_reporting._run_structured_agent", new=fake_run_structured_agent),
            patch("knowledge_gap_reporting.tools_lib.core_answer_from_docs", return_value="Official docs grounding"),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNone(result)

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

    def test_format_bot_behavior_gap_report_uses_structured_sections(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="bot_behavior_gap",
            title="Discord redirect failure",
            topic="Bad in-ticket escalation",
            product="Yearn Vaults",
            chain=None,
            evidence_summary="Generic evidence blob",
            current_official_grounding="Docs exist for the mechanics, but the ticket was already in Discord.",
            assessment="Primary failure was the bot's escalation behavior.",
            suggested_action="Keep the user in-channel and escalate with context.",
            confidence="high",
            user_problem="User wanted to know why PPS had stayed flat.",
            bot_failure="ySupport told them to open a Discord ticket while already in one.",
            human_follow_up="A contributor checked recent strategy activity and notified strategists.",
            unresolved_risk="The same broken escalation path may recur.",
            recommended_owner="bot",
        )

        formatted = format_knowledge_gap_report(
            report,
            affected_channels=[
                PreparedTicketTranscript(channel_id="123", channel_name="ticket-alpha", message_count=1, transcript_text="")
            ],
        )

        self.assertIn("**User issue**", formatted)
        self.assertIn("**Bot failure**", formatted)
        self.assertIn("**Human follow-up**", formatted)
        self.assertIn("**Unresolved risk**", formatted)
        self.assertIn("**Recommended owner**", formatted)
        self.assertIn("**Recommended next step**", formatted)
        self.assertNotIn("**Evidence**", formatted)
        self.assertNotIn("**Suggested action**", formatted)

    def test_format_docs_gap_report_uses_structured_sections(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="docs_gap",
            title="Missing migration docs",
            topic="stYFI migration visibility",
            product="stYFI",
            chain=None,
            evidence_summary="Generic evidence blob",
            current_official_grounding="Docs mention stYFI but not the exact balance-discovery path.",
            assessment="This is a true docs discoverability gap.",
            suggested_action="Add a direct FAQ entry.",
            confidence="high",
            user_problem="User could not find migrated stYFI.",
            missing_or_unclear_docs="The migration docs do not clearly point to the right dashboard.",
            current_workaround="A human contributor can provide the dashboard link manually.",
            unresolved_risk="Users will keep opening tickets for a simple navigation problem.",
            recommended_owner="docs",
        )

        formatted = format_knowledge_gap_report(
            report,
            affected_channels=[
                PreparedTicketTranscript(channel_id="123", channel_name="ticket-alpha", message_count=1, transcript_text="")
            ],
        )

        self.assertIn("**User issue**", formatted)
        self.assertIn("**Missing or unclear docs**", formatted)
        self.assertIn("**Current workaround**", formatted)
        self.assertIn("**Unresolved risk**", formatted)
        self.assertIn("**Recommended owner**", formatted)
        self.assertIn("**Recommended next step**", formatted)
        self.assertNotIn("**Evidence**", formatted)
        self.assertNotIn("**Suggested action**", formatted)

    def test_format_issue_draft_report_uses_structured_sections_when_present(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="issue_draft_candidate",
            title="Potential VaultV3 issue",
            topic="VaultV3 accounting",
            product="VaultV3.vy",
            chain="ethereum",
            evidence_summary="Generic evidence blob",
            current_official_grounding="Repo grounding for the relevant path.",
            assessment="This is plausible enough for internal review.",
            suggested_action="Open an internal issue draft.",
            confidence="high",
            reported_issue="Reporter claims VaultV3 withdrawal accounting may allow PPS inflation.",
            plausibility_basis="The report cites a concrete accounting path and repo grounding matches it.",
            blocking_unknown="Need engineering/security review to confirm exploitability.",
            immediate_triage_need="Open an internal engineering/security triage thread.",
            recommended_owner="engineering",
        )

        formatted = format_knowledge_gap_report(
            report,
            affected_channels=[
                PreparedTicketTranscript(channel_id="789", channel_name="closed-1433", message_count=4, transcript_text="")
            ],
        )

        self.assertIn("**Reported issue**", formatted)
        self.assertIn("**Why plausible**", formatted)
        self.assertIn("**Blocking unknown**", formatted)
        self.assertIn("**Immediate triage need**", formatted)
        self.assertIn("**Recommended owner**", formatted)

    def test_format_product_confusion_report_uses_structured_sections(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="product_confusion",
            title="PPS visibility confusion",
            topic="vault PPS visibility",
            product="Yearn Vaults",
            chain=None,
            evidence_summary="Generic evidence blob",
            current_official_grounding="Docs explain the accounting flow but not the visible product state clearly enough.",
            assessment="This is a product confusion issue.",
            suggested_action="Investigate the surface behavior and improve the explanation.",
            confidence="high",
            user_problem="User thinks the vault may be broken because PPS looks stale.",
            confusing_behavior="Visible PPS appears stale even though later contributor context suggests recent strategy activity.",
            current_workaround="A human contributor can inspect strategy activity manually.",
            unresolved_risk="Users may lose trust in the vault if the visible state remains confusing.",
            recommended_owner="product",
        )

        formatted = format_knowledge_gap_report(
            report,
            affected_channels=[
                PreparedTicketTranscript(channel_id="123", channel_name="ticket-alpha", message_count=1, transcript_text="")
            ],
        )

        self.assertIn("**User confusion**", formatted)
        self.assertIn("**Confusing product behavior**", formatted)
        self.assertIn("**Current workaround**", formatted)
        self.assertIn("**Residual product risk**", formatted)
        self.assertIn("**Recommended owner**", formatted)
        self.assertIn("**Recommended next step**", formatted)
        self.assertNotIn("**Evidence**", formatted)

    def test_format_faq_candidate_report_uses_structured_sections(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="faq_candidate",
            title="stYFI dashboard visibility confusion",
            topic="Where migrated stYFI balances appear",
            product="stYFI",
            chain=None,
            evidence_summary="Generic evidence blob",
            current_official_grounding="Docs partially point there but not plainly enough.",
            assessment="This deserves a reusable FAQ answer.",
            suggested_action="Add a direct FAQ entry and support macro.",
            confidence="high",
            user_problem="Where do migrated stYFI balances appear after veYFI migration?",
            faq_answer="Migrated balances appear on the stYFI dashboard.",
            recurrence_signal="This is a short navigation question that support can answer the same way each time.",
            current_workaround="A human contributor can manually point users to the dashboard.",
            recommended_owner="docs",
        )

        formatted = format_knowledge_gap_report(
            report,
            affected_channels=[
                PreparedTicketTranscript(channel_id="123", channel_name="ticket-alpha", message_count=1, transcript_text="")
            ],
        )

        self.assertIn("**Recurring question**", formatted)
        self.assertIn("**Reusable answer**", formatted)
        self.assertIn("**Why it recurs**", formatted)
        self.assertIn("**Current workaround**", formatted)
        self.assertIn("**Recommended owner**", formatted)
        self.assertIn("**Recommended next step**", formatted)
        self.assertNotIn("**Evidence**", formatted)

    def test_format_issue_draft_report_uses_triage_headings(self) -> None:
        report = KnowledgeGapReport(
            should_post=True,
            category="issue_draft_candidate",
            title="Potential VaultV3 accounting issue",
            topic="VaultV3 withdrawal accounting",
            product="VaultV3.vy",
            chain="ethereum",
            evidence_summary="Reporter claims PPS inflation is possible during withdrawal accounting.",
            current_official_grounding="Repo context shows balance-diff accounting in the relevant path.",
            assessment="The claim is plausible enough for internal engineering/security review.",
            suggested_action="Open an internal engineering/security triage thread with the grounded claim summary.",
            confidence="high",
        )

        formatted = format_knowledge_gap_report(
            report,
            affected_channels=[
                PreparedTicketTranscript(channel_id="789", channel_name="closed-1433", message_count=4, transcript_text="")
            ],
        )

        self.assertIn("**Reported issue**", formatted)
        self.assertIn("**Current grounding**", formatted)
        self.assertIn("**Triage assessment**", formatted)
        self.assertIn("**Recommended next step**", formatted)
        self.assertNotIn("**Current official grounding**", formatted)
        self.assertNotIn("**Suggested action**", formatted)

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

        with patch("knowledge_gap_reporting._supported_chain_id_map", return_value={42161: "arbitrum"}):
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

        with patch("knowledge_gap_reporting._supported_chain_id_map", return_value={42161: "arbitrum"}):
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
                url.endswith(f"/channels/{TEST_REPORT_CHANNEL_ID}/messages") and payload.get("flags") == 4
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

    def test_should_fetch_repo_grounding_for_issue_draft_even_if_model_flag_is_false(self) -> None:
        candidate = KnowledgeGapCandidate(
            reportable=True,
            category="issue_draft_candidate",
            title="Potential issue",
            topic="VaultV3 accounting",
            product="VaultV3.vy",
            chain=None,
            grounding_query="Assess whether this is a real issue.",
            evidence_summary="Reporter claims PPS inflation.",
            suggested_action="Engineering review.",
            needs_repo_context=False,
        )

        self.assertTrue(_should_fetch_repo_grounding(candidate))

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

    def test_preview_recent_closed_ticket_channels_resolves_channel_names(self) -> None:
        with patch(
            "knowledge_gap_worker.discover_recent_closed_ticket_channels",
            return_value=["30", "20"],
        ), patch(
            "knowledge_gap_worker.fetch_channel_metadata",
            side_effect=[
                type("ChannelMetadata", (), {"id": "30", "name": "closed-30"})(),
                type("ChannelMetadata", (), {"id": "20", "name": "closed-20"})(),
            ],
        ):
            preview = preview_recent_closed_ticket_channels(2)

        self.assertEqual(
            preview,
            [
                {"channel_id": "30", "channel_name": "closed-30"},
                {"channel_id": "20", "channel_name": "closed-20"},
            ],
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
