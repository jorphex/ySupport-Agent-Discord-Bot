import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import patch

from knowledge_gap_worker import (
    KnowledgeGapReport,
    PreparedTicketTranscript,
    KnowledgeGapCandidate,
    _build_report_signature,
    _should_fetch_repo_grounding,
    _snowflake_sort_key,
    finalize_knowledge_gap_report,
    format_knowledge_gap_report,
    post_report_message,
)

TEST_REPORT_CHANNEL_ID = 999999999999999999


class KnowledgeGapWorkerTests(unittest.IsolatedAsyncioTestCase):
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
                PreparedTicketTranscript(
                    channel_id="123",
                    channel_name="ticket-alpha",
                    message_count=1,
                    transcript_text="",
                ),
                PreparedTicketTranscript(
                    channel_id="456",
                    channel_name="ticket-beta",
                    message_count=1,
                    transcript_text="",
                ),
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
                PreparedTicketTranscript(
                    channel_id="123",
                    channel_name="ticket-alpha",
                    message_count=1,
                    transcript_text="",
                )
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
                PreparedTicketTranscript(
                    channel_id="123",
                    channel_name="ticket-alpha",
                    message_count=1,
                    transcript_text="",
                )
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

    def test_format_issue_draft_report_uses_structured_sections_when_present(
        self,
    ) -> None:
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
                PreparedTicketTranscript(
                    channel_id="789",
                    channel_name="closed-1433",
                    message_count=4,
                    transcript_text="",
                )
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
                PreparedTicketTranscript(
                    channel_id="123",
                    channel_name="ticket-alpha",
                    message_count=1,
                    transcript_text="",
                )
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
                PreparedTicketTranscript(
                    channel_id="123",
                    channel_name="ticket-alpha",
                    message_count=1,
                    transcript_text="",
                )
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
                PreparedTicketTranscript(
                    channel_id="789",
                    channel_name="closed-1433",
                    message_count=4,
                    transcript_text="",
                )
            ],
        )

        self.assertIn("**Reported issue**", formatted)
        self.assertIn("**Current grounding**", formatted)
        self.assertIn("**Triage assessment**", formatted)
        self.assertIn("**Recommended next step**", formatted)
        self.assertNotIn("**Current official grounding**", formatted)
        self.assertNotIn("**Suggested action**", formatted)

    def test_finalize_knowledge_gap_report_uses_supported_chain_hint_from_transcript(
        self,
    ) -> None:
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

        with patch(
            "knowledge_gap_reporting._supported_chain_id_map",
            return_value={42161: "arbitrum"},
        ):
            finalized = finalize_knowledge_gap_report(report, transcript)

        self.assertEqual(finalized.chain, "arbitrum")

    def test_finalize_knowledge_gap_report_drops_unsupported_chain_without_hint(
        self,
    ) -> None:
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

        with patch(
            "knowledge_gap_reporting._supported_chain_id_map",
            return_value={42161: "arbitrum"},
        ):
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
                url.endswith(f"/channels/{TEST_REPORT_CHANNEL_ID}/messages")
                and payload.get("flags") == 4
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

        self.assertEqual(
            _build_report_signature(report_a), _build_report_signature(report_b)
        )

    def test_should_fetch_repo_grounding_for_issue_draft_even_if_model_flag_is_false(
        self,
    ) -> None:
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
