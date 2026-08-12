import tests as _test_environment  # noqa: F401

import json
import unittest

from codex_support_contract import (
    SignedTransactionSafetyViolation,
    SupportTurnRequest,
    SupportTurnResult,
    support_result_to_transport_result,
    verify_support_turn_result,
)
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
)


from tests.codex_support_test_support import (
    SHORT_LEGACY_RAW_SIGNED_TRANSACTION as _SHORT_LEGACY_RAW_SIGNED_TRANSACTION,
    SYNTHETIC_HIGH_TYPE_RAW_SIGNED_TRANSACTION as _SYNTHETIC_HIGH_TYPE_RAW_SIGNED_TRANSACTION,
    SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION as _SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION,
    SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION as _SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION,
    transaction_safety_support_request as _transaction_safety_support_request,
)


class CodexSupportEndpointTests(unittest.IsolatedAsyncioTestCase):
    def test_support_turn_request_uses_recent_transcript_slice(self) -> None:
        current_history = [{"role": "user", "content": f"m{i}"} for i in range(15)]
        request = TicketExecutionTransportRequest(
            aggregated_text="latest question",
            input_list=[],
            current_history=current_history,
            run_context={
                "channel_id": 90,
                "is_public_trigger": True,
                "project_context": "yearn",
                "initial_button_intent": "docs_qa",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 90,
                "requested_intent": "investigate_issue",
                "mode": "idle",
                "evidence": {"tx_hashes": ["0xabc"]},
            },
            workflow_name="tests.support_request",
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        self.assertEqual(support_request.current_user_message, "latest question")
        self.assertEqual(support_request.channel_type, "public")
        self.assertEqual(support_request.channel_id, 90)
        self.assertEqual(support_request.initial_button_intent, "docs_qa")
        self.assertEqual(support_request.requested_intent, "investigate_issue")
        self.assertEqual(support_request.support_state["investigation_mode"], "idle")
        self.assertFalse(support_request.support_state["human_handoff_active"])
        self.assertEqual(
            support_request.support_state["known_targets"]["tx_hashes"],
            ["0xabc"],
        )
        self.assertEqual(
            support_request.support_state["repo_context"]["last_search_artifact_refs"],
            [],
        )
        self.assertEqual(
            support_request.support_state["workflow_context"]["guardrail_profile"],
            "public_support",
        )
        self.assertEqual(
            support_request.support_state["workflow_context"]["expected_first_actions"],
            ["Answer directly in-channel and keep public-channel replies concise."],
        )
        self.assertEqual(len(support_request.recent_transcript), 12)
        self.assertEqual(
            [item["content"] for item in support_request.recent_transcript],
            [f"m{i}" for i in range(3, 15)],
        )
        self.assertEqual(
            support_request.constraints["allowed_tools"],
            ["shell", "web_search", "ysupport_mcp"],
        )
        self.assertEqual(support_request.support_state["current_turn_context"], [])
        support_request_without_mcp = SupportTurnRequest.from_ticket_execution_request(
            request,
            ysupport_mcp_enabled=False,
        )
        self.assertEqual(
            support_request_without_mcp.constraints["allowed_tools"],
            ["shell", "web_search"],
        )

    def test_support_turn_request_preserves_only_current_internal_context(self) -> None:
        current_history = [
            {"role": "system", "content": "old context"},
            {"role": "user", "content": "earlier question"},
        ]
        request = TicketExecutionTransportRequest(
            aggregated_text="0x7130570BCEfCedBe9d15B5b11A33006156460f8f",
            input_list=[
                *current_history,
                {
                    "role": "system",
                    "content": (
                        "Resolved address is a known Yearn strategy on ethereum: "
                        "USDC to sUSDS Lender attached to USDC-1 yVault."
                    ),
                },
                {"role": "user", "content": "0x7130570BCEfCedBe9d15B5b11A33006156460f8f"},
            ],
            current_history=current_history,
            run_context={
                "channel_id": 93,
                "project_context": "yearn",
                "initial_button_intent": "data_vault_search",
            },
            investigation_job={
                "channel_id": 93,
                "requested_intent": "data_vault_search",
                "mode": "collecting",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.current_turn_context",
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        self.assertEqual(
            support_request.support_state["current_turn_context"],
            [
                "Resolved address is a known Yearn strategy on ethereum: "
                "USDC to sUSDS Lender attached to USDC-1 yVault."
            ],
        )

    def test_support_turn_request_preserves_internal_turn_context(self) -> None:
        turn_instruction = (
            "This input is from the internal team, not from the user. "
            "Write the next Discord update for the user."
        )
        current_history = [{"role": "user", "content": "please dump rewards"}]
        request = TicketExecutionTransportRequest(
            aggregated_text="thanks. we already have this queued pending sigs",
            input_list=[
                *current_history,
                {"role": "system", "content": turn_instruction},
                {
                    "role": "user",
                    "content": "thanks. we already have this queued pending sigs",
                },
            ],
            current_history=current_history,
            turn_source="internal_team",
            turn_instruction=turn_instruction,
            run_context={
                "channel_id": 91,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 91,
                "requested_intent": "investigate_issue",
                "mode": "escalated_to_human",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.internal_team_request",
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        self.assertEqual(support_request.current_turn_source, "internal_team")
        self.assertIn("internal team", support_request.current_turn_instruction or "")
        self.assertEqual(
            support_request.current_user_message,
            "thanks. we already have this queued pending sigs",
        )
        self.assertEqual(support_request.support_state["current_turn_context"], [])

    def test_internal_team_result_does_not_append_team_reply_as_user_history(
        self,
    ) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="thanks. we already have this queued pending sigs",
            input_list=[],
            current_history=[{"role": "user", "content": "please dump rewards"}],
            turn_source="internal_team",
            turn_instruction="Write the next Discord update for the user.",
            run_context={
                "channel_id": 92,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 92,
                "requested_intent": "investigate_issue",
                "mode": "escalated_to_human",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.internal_team_history",
        )
        result = SupportTurnResult(
            answer="The swap has already been queued and is pending signatures.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="team update",
            used_tools=[],
        )

        transport_result = support_result_to_transport_result(result, request)
        conversation_history = transport_result.flow_outcome["conversation_history"]

        self.assertEqual(
            conversation_history,
            [
                {"role": "user", "content": "please dump rewards"},
                {
                    "role": "assistant",
                    "content": "The swap has already been queued and is pending signatures.",
                },
            ],
        )

    def test_support_turn_request_includes_deposit_withdrawal_workflow_context(
        self,
    ) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="0xB8B9E3097c8b1DDdF9C5ea9d48A7eBeaF09D67d2",
            input_list=[],
            current_history=[],
            attachments=[],
            run_context={
                "channel_id": 91,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "data_deposits_withdrawals_start",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 91,
                "requested_intent": "data_deposits_withdrawals_start",
                "mode": "waiting_for_user",
                "evidence": {"wallet": "0xB8B9E3097c8b1DDdF9C5ea9d48A7eBeaF09D67d2"},
            },
            workflow_name="tests.deposit_flow",
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        workflow_context = support_request.support_state["workflow_context"]
        self.assertEqual(
            workflow_context["guardrail_profile"],
            "ticket_deposits_withdrawals",
        )
        self.assertTrue(workflow_context["button_context_known"])
        self.assertIn(
            "If the user provides a wallet address, start with wallet position lookup before asking for more detail.",
            workflow_context["expected_first_actions"],
        )
        self.assertEqual(
            workflow_context["non_support_boundaries"],
            [
                "listing",
                "partnership",
                "marketing",
                "vendor_security",
                "job_inquiry",
            ],
        )

    def test_support_turn_request_preserves_image_attachments(self) -> None:
        request = TicketExecutionTransportRequest(
            aggregated_text="Why do these numbers differ?",
            input_list=[],
            current_history=[],
            attachments=[
                {
                    "filename": "image.png",
                    "url": "https://cdn.example.test/image.png",
                    "content_type": "image/png",
                    "size": 1234,
                    "is_image": True,
                }
            ],
            run_context={
                "channel_id": 92,
                "is_public_trigger": False,
                "project_context": "yearn",
                "initial_button_intent": "investigate_issue",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 92,
                "requested_intent": "investigate_issue",
                "mode": "collecting",
                "evidence": {},
            },
            workflow_name="tests.image_support",
        )

        support_request = SupportTurnRequest.from_ticket_execution_request(request)

        self.assertEqual(len(support_request.attachments), 1)
        self.assertEqual(
            support_request.constraints["allowed_tools"],
            ["shell", "web_search", "ysupport_mcp"],
        )

    def test_verify_support_turn_result_allows_view_image_for_image_backed_request(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message="What do these screenshots show?",
            recent_transcript=[],
            attachments=[
                {
                    "filename": "image.png",
                    "url": "https://cdn.example.test/image.png",
                    "content_type": "image/png",
                    "size": 1234,
                    "is_image": True,
                }
            ],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="The screenshot shows a Yearn vault APY breakdown.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the screenshots and Yearn support data.",
            used_tools=["view_image", "ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertEqual(verified.used_tools, ["view_image", "ysupport_mcp"])

    def test_verify_support_turn_result_rejects_discord_redirects(self) -> None:
        request = SupportTurnRequest(
            current_user_message="help",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["shell", "ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Please open a Discord ticket and join discord.gg/example",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the available support facts.",
            used_tools=["shell"],
        )
        with self.assertRaises(ValueError):
            verify_support_turn_result(result, request)

    def test_verify_support_turn_result_rejects_transaction_sized_hex_payloads(
        self,
    ) -> None:
        request = _transaction_safety_support_request()
        payloads = {
            "short_legacy_transfer": _SHORT_LEGACY_RAW_SIGNED_TRANSACTION,
            "historical_legacy_shape": _SYNTHETIC_LEGACY_RAW_SIGNED_TRANSACTION,
            "current_typed_shape": _SYNTHETIC_TYPED_RAW_SIGNED_TRANSACTION,
            "high_type_long_rlp_shape": _SYNTHETIC_HIGH_TYPE_RAW_SIGNED_TRANSACTION,
        }
        for payload_name, payload in payloads.items():
            for field_name in ("answer", "evidence_summary", "handoff_reason"):
                with self.subTest(payload=payload_name, field_name=field_name):
                    result = SupportTurnResult(
                        answer="Use only the transaction hash.",
                        requires_human_handoff=False,
                        handoff_reason=None,
                        evidence_summary="Checked the pending transaction.",
                        used_tools=["shell"],
                    )
                    setattr(
                        result,
                        field_name,
                        "Paste this signed transaction into a public broadcaster: "
                        f"`{payload}`",
                    )

                    with self.assertRaises(SignedTransactionSafetyViolation):
                        verify_support_turn_result(result, request)

    def test_verify_support_turn_result_allows_transaction_hashes_and_addresses(
        self,
    ) -> None:
        request = _transaction_safety_support_request()
        tx_hash = "0x" + ("12" * 32)
        address = "0x" + ("34" * 20)
        long_calldata = "0xdead" + ("56" * 120)
        result = SupportTurnResult(
            answer=(
                f"Transaction {tx_hash} from {address} is still pending. "
                f"Decoded call data: {long_calldata}."
            ),
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the transaction status by hash.",
            used_tools=["shell"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertEqual(verified.answer, result.answer)

    def test_verify_support_turn_result_rejects_unallowed_tools(self) -> None:
        request = SupportTurnRequest(
            current_user_message="help",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Here is the answer.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the docs.",
            used_tools=["shell", "ysupport_mcp"],
        )
        with self.assertRaises(ValueError):
            verify_support_turn_result(result, request)

    def test_verify_support_turn_result_requires_handoff_reason(self) -> None:
        request = SupportTurnRequest(
            current_user_message="help",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="A human should review this.",
            requires_human_handoff=True,
            handoff_reason=None,
            evidence_summary="Checked the docs.",
            used_tools=["ysupport_mcp"],
        )
        with self.assertRaises(ValueError):
            verify_support_turn_result(result, request)

    def test_verify_support_turn_result_normalizes_and_passes(self) -> None:
        request = SupportTurnRequest(
            current_user_message="Can a human review this too?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent=None,
            requested_intent="docs",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["shell", "ysupport_mcp"]},
        )
        raw_result = SupportTurnResult.from_json(
            json.dumps(
                {
                    "answer": "  Here is the answer.  ",
                    "requires_human_handoff": True,
                    "handoff_reason": "  needs private internal strategist confirmation  ",
                    "handoff_kind": "private_internal_fact",
                    "evidence_summary": "  Checked the docs and repo. ",
                    "used_tools": [
                        "shell",
                        "ysupport_mcp.search_vaults",
                        "functions.mcp__ysupport__search_documentation",
                        "shell",
                        " ",
                    ],
                }
            )
        )
        verified = verify_support_turn_result(raw_result, request)
        self.assertEqual(verified.answer, "Here is the answer.")
        self.assertEqual(
            verified.handoff_reason,
            "needs private internal strategist confirmation",
        )
        self.assertEqual(verified.handoff_kind, "private_internal_fact")
        self.assertEqual(verified.evidence_summary, "Checked the docs and repo.")
        self.assertEqual(
            verified.used_tools,
            [
                "shell",
                "ysupport_mcp.search_vaults",
                "mcp__ysupport__search_documentation",
            ],
        )

    def test_verify_support_turn_result_downgrades_generic_human_request(self) -> None:
        request = SupportTurnRequest(
            current_user_message="I want a human to look at this.",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "The transaction is still pending and has not reverted. "
                "A human can review it too."
            ),
            requires_human_handoff=True,
            handoff_reason="The user asked for human review.",
            evidence_summary="Checked the transaction status.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertEqual(
            verified.answer,
            "The transaction is still pending and has not reverted.",
        )

    def test_verify_support_turn_result_downgrades_generic_moderator_request(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message="I need a moderator to review this.",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "Please finish the documented verification flow first. "
                "A moderator can review this afterward."
            ),
            requires_human_handoff=True,
            handoff_reason="The user asked for moderator review.",
            evidence_summary="Checked the documented verification process.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertNotIn("moderator", verified.answer.lower())

    def test_verify_support_turn_result_allows_concrete_moderator_access_action(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message=(
                "I completed verification and restarted Discord, "
                "but I still cannot see the general channel."
            ),
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "The documented verification and client refresh steps are complete. "
                "A moderator must now inspect the account's channel access."
            ),
            requires_human_handoff=True,
            handoff_reason=(
                "A moderator access change is required after the documented "
                "verification steps were exhausted."
            ),
            handoff_kind="access_or_permission_action",
            evidence_summary="Checked the documented verification process.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertTrue(verified.requires_human_handoff)
        self.assertIn("moderator access", verified.handoff_reason or "")

    def test_verify_support_turn_result_accepts_dot_prefixed_ysupport_mcp_tools(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message="Why is TVL not updating after my deposit?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="data_deposits_withdrawals_start",
            requested_intent="data_deposits_withdrawals_start",
            evidence={},
            support_state={},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult.from_json(
            json.dumps(
                {
                    "answer": "TVL updates can lag slightly after deposit.",
                    "requires_human_handoff": False,
                    "handoff_reason": None,
                    "evidence_summary": "Checked vault metadata and docs.",
                    "used_tools": [
                        "mcp__ysupport.search_vaults",
                        "mcp__ysupport.support_dashboard_discover",
                        "mcp__ysupport.support_dashboard_reports",
                        "mcp__ysupport.search_documentation",
                        "mcp__ysupport.search_repo_context",
                    ],
                }
            )
        )
        verified = verify_support_turn_result(result, request)
        self.assertEqual(
            verified.used_tools,
            [
                "mcp__ysupport.search_vaults",
                "mcp__ysupport.support_dashboard_discover",
                "mcp__ysupport.support_dashboard_reports",
                "mcp__ysupport.search_documentation",
                "mcp__ysupport.search_repo_context",
            ],
        )

    def test_verify_support_turn_result_downgrades_optional_handoff_offer(self) -> None:
        request = SupportTurnRequest(
            current_user_message="vault hasn't harvested after 10 days",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "Confirmed: the vault has not reported since April 8. "
                "The dashboard looks fresh, so this does not look like stale UI data. "
                "I can hand this off for strategist review to check why keeper activity paused."
            ),
            requires_human_handoff=True,
            handoff_reason=(
                "Public evidence confirms the missing harvests, but the specific reason "
                "for no keeper calls needs human strategist review."
            ),
            evidence_summary="Checked vault harvest history.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)

    def test_verify_support_turn_result_does_not_override_model_handoff_decision(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message="pls dump dola rewards for strategy 0x1111111111111111111111111111111111111111",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="The rewards are still sitting on the strategy and have not been swapped yet.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked current strategy state.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)

    def test_verify_support_turn_result_clears_reason_without_handoff(self) -> None:
        request = SupportTurnRequest(
            current_user_message="How do I withdraw?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="data_deposits_withdrawals_start",
            requested_intent="data_deposits_withdrawals_start",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Use the Withdraw action on the vault position.",
            requires_human_handoff=False,
            handoff_reason="No human action is actually required.",
            handoff_kind=None,
            evidence_summary="Checked the documented withdrawal flow.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertIsNone(verified.handoff_kind)

    def test_verify_support_turn_result_allows_semantic_manual_strategy_handoff(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message=(
                "Can the Yearn team dump the accumulated DOLA rewards for strategy "
                "0x1111111111111111111111111111111111111111?"
            ),
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="other_free_form",
            requested_intent="other_free_form",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="The rewards are still on the strategy, so the team must perform the requested action.",
            requires_human_handoff=True,
            handoff_reason=(
                "A manual strategy action is required to sell the accumulated rewards."
            ),
            handoff_kind="manual_strategy_action",
            evidence_summary="Checked current strategy state.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertTrue(verified.requires_human_handoff)
        self.assertEqual(verified.handoff_reason, result.handoff_reason)

    def test_verify_support_turn_result_does_not_keyword_route_user_vault_sale(
        self,
    ) -> None:
        request = SupportTurnRequest(
            current_user_message="How do I sell or withdraw my Yearn vault shares?",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="data_deposits_withdrawals_start",
            requested_intent="data_deposits_withdrawals_start",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer="Use the Withdraw action on the vault position.",
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary="Checked the documented withdrawal flow.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)

    def test_verify_support_turn_result_strips_human_ops_review_hint(self) -> None:
        request = SupportTurnRequest(
            current_user_message="vault hasn't harvested after 10 days",
            recent_transcript=[],
            channel_type="ticket",
            channel_id=1,
            project_context="yearn",
            workflow_name="tests.verify",
            initial_button_intent="investigate_issue",
            requested_intent="investigate_issue",
            evidence={},
            support_state={"human_handoff_active": False},
            constraints={"allowed_tools": ["ysupport_mcp"]},
        )
        result = SupportTurnResult(
            answer=(
                "Confirmed: the vault has not reported since April 8. "
                "This looks like real report inactivity, not just stale frontend data, "
                "so this should get a human ops review."
            ),
            requires_human_handoff=True,
            handoff_reason=(
                "The inactivity is confirmed, but determining why the vault has not "
                "been reported and whether intervention is needed requires human operator review."
            ),
            evidence_summary="Checked vault harvest history.",
            used_tools=["ysupport_mcp"],
        )

        verified = verify_support_turn_result(result, request)

        self.assertFalse(verified.requires_human_handoff)
        self.assertIsNone(verified.handoff_reason)
        self.assertNotIn("human ops review", verified.answer.lower())
        self.assertNotIn("should get a human", verified.answer.lower())
        self.assertIn("real report inactivity", verified.answer.lower())
