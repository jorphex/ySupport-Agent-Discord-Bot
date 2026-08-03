import tests as _test_environment  # noqa: F401

import json
import unittest
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
    analyze_transcript_for_knowledge_gap,
)

TEST_REPORT_CHANNEL_ID = 999999999999999999


class KnowledgeGapWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_transcript_returns_none_when_candidate_is_not_reportable(
        self,
    ) -> None:
        prepared = PreparedTicketTranscript(
            channel_id="1484802638158626887",
            channel_name="ticket-thanks",
            message_count=1,
            transcript_text="[2026-03-23T12:00:00+00:00] User: thanks",
        )

        class FakeCandidate:
            reportable = False
            category = "no_action"

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            return FakeCandidate()

        with patch(
            "knowledge_gap_reporting._run_structured_agent",
            new=fake_run_structured_agent,
        ):
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap FAQ Extraction":
                return extraction
            return report

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs point users to styfi.yearn.fi for stYFI.",
            ),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.title, "stYFI dashboard visibility confusion")
        self.assertEqual(result.category, "faq_candidate")

    async def test_analyze_transcript_passes_candidate_topic_and_evidence_into_repo_grounding(
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Issue Draft Extraction":
                return extraction
            return report

        async def fake_pretriage_repo_claim(
            claim_text: str,
            *,
            include_docs: bool = True,
            limit=None,
            include_legacy=False,
        ) -> str:
            repo_queries.append(claim_text)
            return "Repo grounding"

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_pretriage_repo_claim",
                new=fake_pretriage_repo_claim,
            ),
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Issue Draft Extraction":
                return extraction
            return report

        async def fake_pretriage_repo_claim(
            claim_text: str,
            *,
            include_docs: bool = True,
            limit=None,
            include_legacy=False,
        ) -> str:
            repo_queries.append(claim_text)
            return "Repo grounding"

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_pretriage_repo_claim",
                new=fake_pretriage_repo_claim,
            ),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        self.assertEqual(len(repo_queries), 1)

    async def test_analyze_transcript_runs_bot_behavior_extraction_and_preserves_structured_fields(
        self,
    ) -> None:
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Bot Behavior Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
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

    async def test_analyze_transcript_backfills_extraction_fields_when_final_report_omits_structured_fields(
        self,
    ) -> None:
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Bot Behavior Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
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

    async def test_analyze_transcript_runs_docs_gap_extraction_and_backfills_fields(
        self,
    ) -> None:
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Docs Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.category, "docs_gap")
        self.assertEqual(result.user_problem, extraction.user_problem)
        self.assertEqual(
            result.missing_or_unclear_docs, extraction.missing_or_unclear_docs
        )
        self.assertEqual(result.current_workaround, extraction.current_workaround)
        self.assertEqual(result.unresolved_risk, extraction.unresolved_risk)
        self.assertEqual(result.recommended_owner, "docs")
        self.assertEqual(len(final_report_inputs), 1)
        self.assertIn('"recommended_owner": "docs"', final_report_inputs[0])
        self.assertIn("docs_gap extraction", final_report_inputs[0])

    async def test_analyze_transcript_runs_issue_draft_extraction_and_backfills_fields(
        self,
    ) -> None:
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Issue Draft Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        async def fake_pretriage_repo_claim(
            claim_text: str,
            *,
            include_docs: bool = True,
            limit=None,
            include_legacy=False,
        ) -> str:
            return "Repo grounding"

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_pretriage_repo_claim",
                new=fake_pretriage_repo_claim,
            ),
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

    async def test_analyze_transcript_runs_product_confusion_extraction_and_backfills_fields(
        self,
    ) -> None:
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap Product Confusion Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
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

    async def test_analyze_transcript_runs_faq_candidate_extraction_and_backfills_fields(
        self,
    ) -> None:
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap FAQ Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                final_report_inputs.append(input_text)
                return report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
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

    async def test_analyze_transcript_drops_placeholder_like_report_after_validation(
        self,
    ) -> None:
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

        async def fake_run_structured_agent(
            agent, input_text, output_type, workflow_name
        ):
            if workflow_name == "Knowledge Gap Candidate Analysis":
                return candidate
            if workflow_name == "Knowledge Gap FAQ Extraction":
                return extraction
            if workflow_name == "Knowledge Gap Final Report":
                return invalid_report
            raise AssertionError(f"Unexpected workflow: {workflow_name}")

        with (
            patch(
                "knowledge_gap_reporting._run_structured_agent",
                new=fake_run_structured_agent,
            ),
            patch(
                "knowledge_gap_reporting.docs_repo_tools.core_answer_from_docs",
                return_value="Official docs grounding",
            ),
        ):
            result = await analyze_transcript_for_knowledge_gap(prepared)

        self.assertIsNone(result)
