import json
import unittest
from unittest.mock import patch

from agents.exceptions import InputGuardrailTripwireTriggered

import config
from support_agents import yearn_data_agent
from tests.llm_e2e_harness import LlmE2EBase, RUN_LLM_E2E


@unittest.skipUnless(
    RUN_LLM_E2E,
    "Set RUN_LLM_E2E_TESTS=1 with OPENAI_API_KEY available to run real-LLM end-to-end tests.",
)
class LlmSupportAnswerTests(LlmE2EBase):
    async def test_sensitive_report_receipt_confirmation_escalates_without_repro_questions(
        self,
    ) -> None:
        output, _, starting_agent_key = await self._run_support_turn(
            "I sent an encrypted security report earlier. Can you confirm it was received?"
        )

        self.assertEqual(starting_agent_key, "triage")
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, output)
        self.assertNoGenericBugTriage(output)

    async def test_discord_access_issue_escalates_to_human_without_generic_redirects(
        self,
    ) -> None:
        output, _, starting_agent_key = await self._run_support_turn(
            "I finished the Discord verification flow but still cannot see the general channel. "
            "I need a moderator to check my access."
        )

        self.assertEqual(starting_agent_key, "triage")
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, output)

        lowered = output.lower()
        self.assertIn("discord", lowered)
        self.assertNotIn("github", lowered)
        self.assertNotIn("connect your wallet", lowered)
        self.assertNotIn("please provide your wallet", lowered)

    async def test_discord_general_access_request_escalates_without_github_or_general_redirect(
        self,
    ) -> None:
        output, _, starting_agent_key = await self._run_support_turn(
            "i want to report an issue and i can’t view the general channel on discord and i will "
            "like to report my issues there to know if anyone who has the same issue can help me out in public please"
        )

        self.assertEqual(starting_agent_key, "triage")
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, output)

        lowered = output.lower()
        self.assertIn("human", lowered)
        self.assertNotIn("github", lowered)
        self.assertNotIn("discord.gg", lowered)

    async def test_product_navigation_question_routes_to_docs_and_answers_directly(
        self,
    ) -> None:
        tool_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            tool_calls.append(user_query)
            return (
                "Go to https://styfi.yearn.fi and open the position view for your wallet. "
                "That is where supported stYFI positions are shown."
            )

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            output, _, starting_agent_key = await self._run_support_turn(
                "Where do I see my stYFI position?"
            )

        self.assertEqual(starting_agent_key, "docs")
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertIn("stYFI position", tool_calls[0])

        lowered = output.lower()
        self.assertIn("styfi.yearn.fi", lowered)
        self.assertIn("position", lowered)

    async def test_styfi_migration_balance_view_question_uses_official_dashboard_paths(
        self,
    ) -> None:
        async def fake_answer_from_docs(user_query: str) -> str:
            self.assertIn("styfi", user_query.lower())
            return (
                "Check https://styfi.yearn.fi for current stYFI. "
                "Use https://veyfi.yearn.fi for the migration and liquid-locker dashboard. "
                "Use https://legacy-veyfi.yearn.fi for legacy veYFI lock management."
            )

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            outcome = await self._run_ticket_flow(
                "after migrations from veyfi to styfi but where can i check my styfi? "
                "I went to styfi.yearn.fi and there can't find my styfi balance.",
                channel_id=9237,
            )

        self.assertEqual(
            outcome.completed_agent_key,
            "docs",
            f"reply={outcome.raw_final_reply!r}",
        )
        self.assertFalse(outcome.requires_human_handoff, outcome.raw_final_reply)
        lowered = outcome.raw_final_reply.lower()
        self.assertIn("styfi.yearn.fi", lowered)
        self.assertIn("veyfi.yearn.fi", lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNoGenericBugTriage(lowered)

    async def test_recovery_process_question_with_wallet_routes_to_docs_not_deposit_check(
        self,
    ) -> None:
        tool_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            tool_calls.append(user_query)
            return (
                "The docs do not document a 1:1 yETH recovery guarantee or a per-holder claim formula. "
                "Recovered assets, if any, are redistributed to affected depositors, but the exact claim amount and "
                "recovery-vault tradeoffs are not documented there."
            )

        async def fake_search_repo_context(*, query: str, limit: int = 5, include_legacy=None, include_ui=None) -> str:
            return ""

        async def fake_fetch_repo_artifacts(*, artifact_refs_text: str) -> str:
            return ""

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            with patch("tools_lib.core_search_repo_context", new=fake_search_repo_context):
                with patch("tools_lib.core_fetch_repo_artifacts", new=fake_fetch_repo_artifacts):
                    outcome = await self._run_ticket_flow(
                        "Hi, I am a yETH holder associated with wallet 0x0ae6395e62c85b7b5d08c5e7918b60c1eac66680. "
                        "Does that mean I can reclaim my lost ETH 1:1? If not, what amount can I recover, "
                        "and why would someone stay in the recovery vault instead of reclaiming?",
                        channel_id=9017,
                    )

        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertIn("yeth", tool_calls[0].lower())

        lowered = outcome.raw_final_reply.lower()
        self.assertIn("the docs do not", lowered)
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("do you want me to check", lowered)
        self.assertNotIn("wallet address now", lowered)

    async def test_styfi_contract_address_question_routes_to_docs_and_answers_directly(
        self,
    ) -> None:
        tool_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            tool_calls.append(user_query)
            return (
                "The stYFI contract on Ethereum mainnet is "
                "0x42b25284E8ae427D79da78b65DFFC232aAECc016.\n"
                "Source: Yearn docs, stYFI contract addresses."
            )

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            outcome = await self._run_ticket_flow(
                "whats the contract address for styfi? it just launched today",
                channel_id=9024,
            )

        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertIn("contract address", tool_calls[0].lower())

        lowered = outcome.raw_final_reply.lower()
        self.assertIn("0x42b25284e8ae427d79da78b65dffc232aaecc016", lowered)
        self.assertNotIn("official docs list", lowered)
        self.assertNotIn("based on the docs", lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNoDataLookupDetour(lowered)

    async def test_veyfi_deposit_question_points_to_current_supported_styfi_path(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "can i still deposit in veyfi?",
            channel_id=9030,
        )

        self.assertEqual(outcome.completed_agent_key, "docs")
        lowered = outcome.raw_final_reply.lower()
        self.assertTrue("legacy" in lowered or "deprecated" in lowered)
        self.assertTrue("styfi" in lowered or "styfi.yearn.fi" in lowered)
        self.assertNotIn("yes, you can still deposit", lowered)
        self.assertNotIn("if you want, i can also", lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)

    async def test_harvest_rewards_question_routes_to_docs_not_vault_search(
        self,
    ) -> None:
        tool_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            tool_calls.append(user_query)
            return (
                "`harvest()` is permissionless, but calling harvest only claims rewards. "
                "Swapping and realizing those rewards can happen later, so not seeing rewards immediately can be normal. "
                "If you have a specific failed tx or stuck action, "
                "send that evidence and I can investigate further."
            )

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            outcome = await self._run_ticket_flow(
                "hi. can I make Harvests myself?\n"
                "This. Pool v2\n"
                "I have a position in this pool. But the rewards are not coming",
                channel_id=9018,
            )

        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertIn("harvest", tool_calls[0].lower())

        lowered = outcome.raw_final_reply.lower()
        self.assertIn("harvest", lowered)
        self.assertIn("rewards", lowered)
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("wallet address", lowered)

    async def test_apy_mechanics_question_routes_to_docs_not_vault_search(self) -> None:
        docs_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            docs_calls.append(user_query)
            return (
                "Vault APY is backward-looking from pricePerShare (PPS) change, annualized over the relevant window. "
                "On non-Ethereum chains the displayed net APY can use weekly PPS change, so it can diverge from individual "
                "strategy APR figures. The docs do not document a separate donation or TVL-outflow adjustment formula "
                "beyond the PPS-based calculation."
            )

        async def fail_search_vaults(*args, **kwargs) -> str:
            raise AssertionError("Vault search should not run for an APY mechanics explanation question.")

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            with patch("tools_lib.core_search_vaults", new=fail_search_vaults):
                outcome = await self._run_ticket_flow(
                    "How is the AUSD vault in katana earning 10%+ native APY for almost a month "
                    "if the strategies inside are earning less than 1%?",
                    channel_id=9038,
                )

        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertGreaterEqual(len(docs_calls), 1)
        self.assertIn("apy", docs_calls[0].lower())
        lowered = outcome.raw_final_reply.lower()
        self.assertIn("pricepershare", lowered)
        self.assertIn("pps", lowered)
        self.assertIn("backward-looking", lowered)
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("wallet address", lowered)

    async def test_closed_1431_value_drop_question_replays_to_docs_not_wallet_detour(
        self,
    ) -> None:
        docs_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            docs_calls.append(user_query)
            return (
                "The vault position value is not documented as a guaranteed 1:1 USD carry-through from the deposit moment. "
                "Vault share value can differ from the deposit asset's spot value because the position is represented by vault shares "
                "and the docs describe PPS-based value reporting rather than a fixed deposit-dollar display. "
                "If you want a wallet-specific reconciliation after that mechanics explanation, share the exact account context for investigation."
            )

        async def fail_search_vaults(*args, **kwargs) -> str:
            raise AssertionError("Vault search should not run for the closed-1431 value-drop replay.")

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            with patch("tools_lib.core_search_vaults", new=fail_search_vaults):
                outcome = await self._run_ticket_flow(
                    "Hi\n"
                    "I staked 0.2 ETH about a week ago on the yCVR pool:\n"
                    "https://yearn.fi/vaults/1/0x27B5739e22ad9033bcBf192059122d163b60349D\n"
                    "transaction:\n"
                    "https://etherscan.io/tx/0x720e06726fe8482bf0d297494ee4e569dd663e65c77f2d1b5c694a7af724565b\n\n"
                    "What I don't understand is that the 0.2 ETH, also a week ago was about $400 in value.\n"
                    "But my staked value on the pool is only about $278 as the transaction also states.\n\n"
                    "Did I just instantly lose about 30% upon staking or am I missing something?",
                    channel_id=9040,
                )

        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertGreaterEqual(len(docs_calls), 1)
        lowered = outcome.raw_final_reply.lower()
        self.assertTrue("pps" in lowered or "vault share" in lowered or "value" in lowered)
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("wallet address", lowered)
        self.assertNotIn("withdrawals", lowered)

    async def test_onboarding_question_uses_docs_backed_getting_started_answer_without_yfi_detour(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "I'm very normie, but Yearn comes highly recommended and I'm on board with the thesis. "
            "I already have a cold and hot wallet, but some of your vault tokens look exotic and I don't know "
            "how to get my hands on them to stake them. Would love some onboarding help.",
            channel_id=9025,
        )

        self.assertEqual(outcome.completed_agent_key, "docs")
        lowered = outcome.raw_final_reply.lower()
        self.assertIn("docs.yearn.fi", lowered)
        self.assertTrue("approve" in lowered or "deposit" in lowered)
        self.assertNotIn("must buy yfi", lowered)
        self.assertNotIn("you can acquire yearn tokens", lowered)
        self.assertNotIn("acquiring yearn tokens", lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)

    async def test_recommendation_question_routes_to_data_and_suggests_current_vaults(
        self,
    ) -> None:
        tool_calls: list[dict] = []
        tool = self._get_agent_tool(yearn_data_agent, "search_vaults_tool")

        async def fake_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            tool_calls.append(payload)
            return (
                "Found 3 Yearn vault(s) matching 'all'. Showing top 3 (sorted by TVL (Descending)) with details:\n\n"
                "---\n\n"
                "Vault: Yearn USDC (yvUSDC)\n"
                "Address: 0x1111111111111111111111111111111111111111\n"
                "Yearn UI Link: https://yearn.fi/vaults/1/0x1111111111111111111111111111111111111111\n"
                "Current Net APY (compounded): 4.20% (Type: v2:averaged)\n"
                "TVL (USD): $1000000.00\n"
                "Risk Level: 1\n\n"
                "---\n\n"
                "Vault: Yearn DAI (yvDAI)\n"
                "Address: 0x2222222222222222222222222222222222222222\n"
                "Yearn UI Link: https://yearn.fi/vaults/1/0x2222222222222222222222222222222222222222\n"
                "Current Net APY (compounded): 3.80% (Type: v2:averaged)\n"
                "TVL (USD): $850000.00\n"
                "Risk Level: 1\n\n"
                "---\n\n"
                "Vault: Yearn WETH (yvWETH)\n"
                "Address: 0x3333333333333333333333333333333333333333\n"
                "Yearn UI Link: https://yearn.fi/vaults/1/0x3333333333333333333333333333333333333333\n"
                "Current Net APY (compounded): 2.10% (Type: v2:averaged)\n"
                "TVL (USD): $780000.00\n"
                "Risk Level: 2\n"
            )

        with patch.object(tool, "on_invoke_tool", new=fake_on_invoke_tool):
            output, _, starting_agent_key = await self._run_support_turn(
                "What Yearn vaults would you recommend right now if I want a simple place to start?"
            )

        self.assertEqual(starting_agent_key, "data")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["query"], "all")
        self.assertTrue(tool_calls[0]["recommended_only"])

        lowered = output.lower()
        self.assertIn("yvusdc", lowered)
        self.assertIn("yearn.fi/vaults/1/0x1111111111111111111111111111111111111111", lowered)
        self.assertNotIn("no one can decide for you", lowered)
        self.assertNotIn("docs do not specify a single best", lowered)
        self.assertNotIn("according to the docs", lowered)

    async def test_recommendation_question_avoids_single_strategy_ys_vaults_by_default(
        self,
    ) -> None:
        tool_calls: list[dict] = []
        tool = self._get_agent_tool(yearn_data_agent, "search_vaults_tool")

        async def fake_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            tool_calls.append(payload)
            self.assertTrue(payload["recommended_only"])
            return (
                "Found 2 Yearn vault(s) matching 'all'. Showing top 2 (sorted by recommendation quality) with details:\n\n"
                "---\n\n"
                "Vault: Yearn USDC (yvUSDC)\n"
                "Address: 0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
                "Yearn UI Link: https://yearn.fi/vaults/1/0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
                "Kind: Multi Strategy\n"
                "Current Net APY (compounded): 4.20% (Type: v2:averaged)\n"
                "TVL (USD): $1000000.00\n"
                "Featuring Score: 0.95\n"
                "Risk Level: 1\n"
                "Strategies (3):\n"
                "  1. Name: Strat One (`0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb01`)\n\n"
                "---\n\n"
                "Vault: Yearn DAI (yvDAI)\n"
                "Address: 0xcccccccccccccccccccccccccccccccccccccccc\n"
                "Yearn UI Link: https://yearn.fi/vaults/1/0xcccccccccccccccccccccccccccccccccccccccc\n"
                "Kind: Multi Strategy\n"
                "Current Net APY (compounded): 3.80% (Type: v2:averaged)\n"
                "TVL (USD): $850000.00\n"
                "Featuring Score: 0.89\n"
                "Risk Level: 1\n"
                "Strategies (2):\n"
                "  1. Name: Strat One (`0xcccccccccccccccccccccccccccccccccccccc01`)\n"
            )

        with patch.object(tool, "on_invoke_tool", new=fake_on_invoke_tool):
            output, _, starting_agent_key = await self._run_support_turn(
                "What Yearn vaults would you recommend right now if I want a simple place to start?"
            )

        self.assertEqual(starting_agent_key, "data")
        self.assertEqual(len(tool_calls), 1)
        self.assertTrue(tool_calls[0]["recommended_only"])

        lowered = output.lower()
        self.assertIn("yvusdc", lowered)
        self.assertIn("yvdai", lowered)
        self.assertNotIn("ysusdc", lowered)
        self.assertNotIn("single strategy", lowered)

    async def test_charity_vault_builder_question_uses_official_setup_guide_and_steps(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "I'm part of a small community supporting charity. Is there a way to set up a vault so people can deposit "
            "funds and all the yield is sent to a charity address?",
            channel_id=9035,
        )

        self.assertEqual(outcome.completed_agent_key, "docs")
        lowered = outcome.raw_final_reply.lower()
        self.assertIn("docs.yearn.fi", lowered)
        self.assertTrue("deploy" in lowered or "vault" in lowered or "charity" in lowered)

    async def test_greeting_only_message_does_not_assume_vault_search_intent(self) -> None:
        output, _, starting_agent_key = await self._run_support_turn(
            "Hello",
            channel_id=9026,
        )
        self.assertEqual(starting_agent_key, "triage")
        lowered = output.lower()
        self.assertTrue("help" in lowered or "yearn" in lowered)
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("apy", lowered)

    async def test_manual_reward_intervention_request_escalates_in_bug_flow(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "Hello, the Mooo transaction has difficulty to execute on the strategy "
            "0x1111111111111111111111111111111111111111 from the MUSD/2Pool. "
            "The CRV tokens were sent out more than 30 hours ago, but still no LP token received.",
            initial_button_intent="bug_report",
            channel_id=9019,
        )

        self.assertTrue(outcome.requires_human_handoff)
        lowered = outcome.raw_final_reply.lower()
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNoGenericBugTriage(lowered)
        self.assertNotIn("what token, vault, or criteria", lowered)

    async def test_manual_reward_intervention_request_escalates_without_button(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "Rewards have been sitting for days. Can someone compound or reinvest them for the pool?",
            channel_id=9023,
        )

        self.assertIsNone(outcome.completed_agent_key)
        self.assertTrue(outcome.requires_human_handoff)
        lowered = outcome.raw_final_reply.lower()
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNotIn("how harvests work", lowered)
        self.assertNotIn("price per share", lowered)
        self.assertNoGenericBugTriage(lowered)

    async def test_mis_sent_treasury_funds_with_tx_hash_escalates_without_vague_clarification(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "Hey guys! I got a small issue, i just accidentally sent my funds "
            '(from binance exchange) into your "treasure wallet"\n\n'
            "https://etherscan.io/tx/0x2ad12015313462f5a39513e5df53f5a76f081ff140735da637040b19fc896ea6",
            channel_id=9027,
        )

        self.assertIsNone(outcome.completed_agent_key)
        self.assertTrue(outcome.requires_human_handoff)
        lowered = outcome.raw_final_reply.lower()
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNotIn("what do you mean", lowered)
        self.assertNotIn("yearn vault address", lowered)
        self.assertNotIn("wallet address you control", lowered)

    async def test_yyb_disabled_button_first_turn_routes_to_bug_without_deposit_prompt(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "I cant add my yYB",
            channel_id=9020,
        )

        lowered = outcome.raw_final_reply.lower()
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("withdrawals if needed", lowered)
        self.assertNotIn("provide your wallet address", lowered)
        self.assertNotIn("vault address", lowered)
        self.assertNotIn("listable on exchanges", lowered)
        self.assertNotIn("listing fees", lowered)
        self.assertNotIn("proposal is necessary", lowered)
        self.assertNotIn("what token, vault, or criteria", lowered)

    async def test_withdrawal_followup_switches_to_bug_for_reproducible_rabby_issue(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "Rabby says 'transaction not ready' for every address when I try to withdraw. "
            "I mentioned a screenshot too.",
            channel_id=9024,
            last_specialty="data",
            current_history=[
                {
                    "role": "assistant",
                    "content": (
                        "Okay, I can help with withdrawal instructions. "
                        "Please provide your wallet address (0x...). I can then check your deposits "
                        "and you can tell me which one you want to withdraw from."
                    ),
                }
            ],
        )

        self.assertEqual(outcome.completed_agent_key, "bug")
        lowered = outcome.raw_final_reply.lower()
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("which vault", lowered)
        self.assertNotIn("provide your wallet address", lowered)
        self.assertNotIn("please provide your wallet address", lowered)

    async def test_vendor_security_outreach_hits_firm_boundary_message(
        self,
    ) -> None:
        with self.assertRaises(InputGuardrailTripwireTriggered) as exc_info:
            await self._run_support_turn(
                "Hey team, urgently flagging malicious sites impersonating the Yearn brand. "
                "Our threat intelligence analysts at PhishFort detected multiple fake domains and can support "
                "a one week trial, broader scan, and one takedown free of charge if useful.",
                channel_id=9022,
            )

        lowered = exc_info.exception.guardrail_result.output.output_info["message"].lower()
        self.assertIn("yearn-security", lowered)
        self.assertNotIn("share more details", lowered)
        self.assertNotIn("partnership", lowered)
        self.assertNotIn("5 sentences", lowered)
        self.assertNotIn("tag **corn**", lowered)

    async def test_vendor_audit_sales_pitch_hits_boundary_without_call_scheduling(
        self,
    ) -> None:
        with self.assertRaises(InputGuardrailTripwireTriggered) as exc_info:
            await self._run_support_turn(
                "Hi team\n\n"
                "I was sorry to hear about the exploit. Hope the team and community is managing.\n\n"
                "We do fully automated smart contract auditing using proprietary AI agents, often at a fraction "
                "of the cost and turnaround of top-tier human auditors.\n\n"
                "Happy to jump on a call too. Would this be interesting for you?\n\n"
                "Best,\nDaniel\nCEO at Cecuro\ncecuro.ai\ndaniel@cecuro.ai",
                channel_id=9028,
            )

        lowered = exc_info.exception.guardrail_result.output.output_info["message"].lower()
        self.assertIn("yearn-security", lowered)
        self.assertNotIn("jump on a call", lowered)
        self.assertNotIn("share more details", lowered)
        self.assertIn("cannot engage", lowered)

    async def test_closed_1433_reward_seeking_issue_report_uses_security_process_boundary(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "Good day team, me and my team discovered an issue that should be addressed "
            "and hope to be rewarded for our efforts",
            channel_id=9041,
        )

        self.assertIsNone(outcome.completed_agent_key)
        self.assertTrue(outcome.requires_human_handoff)
        lowered = outcome.raw_final_reply.lower()
        self.assertTrue(
            "docs.yearn.fi/developers/security" in lowered
            or "github.com/yearn/yearn-security" in lowered
            or "security@yearn.finance" in lowered
        )
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
        self.assertNotIn("describe the bug", lowered)
        self.assertNoGenericBugTriage(lowered)

    async def test_security_report_gist_submission_with_no_concrete_claim_requests_specifics_before_handoff(
        self,
    ) -> None:
        async def fake_fetch_report_artifact(report_url: str, max_chars: int = 12000) -> str:
            self.assertIn("gist.github.com", report_url)
            return (
                "Fetched public report artifact from: https://gist.github.com/.../dc2c924b6d84727f244babc09360e16f\n\n"
                "File: report.md\n"
                "Critical vulnerability found. This may cause loss of funds."
            )

        with patch("tools_lib.core_fetch_report_artifact", new=fake_fetch_report_artifact):
            outcome = await self._run_ticket_flow(
                "Report : https://gist.github.com/AlphaN0x/dc2c924b6d84727f244babc09360e16f",
                channel_id=9031,
                initial_button_intent="bug_report",
            )

        lowered = outcome.raw_final_reply.lower()
        self.assertFalse(outcome.requires_human_handoff, outcome.raw_final_reply)
        self.assertTrue(
            any(
                marker in lowered
                for marker in (
                    "exact yearn contract",
                    "which yearn contract",
                    "affected yearn contract",
                    "contract/product/path",
                    "product/path affected",
                    "yearn product path",
                    "affected contract",
                )
            )
        )
        self.assertTrue(
            any(
                marker in lowered
                for marker in (
                    "observed impact",
                    "vulnerability claim",
                    "bug claim",
                    "actual report",
                    "exploit mechanism",
                    "specific claim",
                )
            )
        )
        self.assertNoGenericBugTriage(lowered)
        self.assertNotIn("{human_handoff_tag_placeholder}", lowered)

    async def test_ticket_1462_reward_distribution_question_replays_to_docs_not_generic_ack(
        self,
    ) -> None:
        docs_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            docs_calls.append(user_query)
            return (
                "Harvest can claim rewards before they are swapped and realized into the vault position, so delayed distribution can be normal. "
                "Thresholds or waiting periods mentioned informally are not the same thing as a guaranteed immediate distribution trigger. "
                "If you have a specific failed transaction or a stuck action, share that evidence for investigation."
            )

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            outcome = await self._run_ticket_flow(
                "Hi. I have vaults cvxcrv. Awards are not distributed. How does this happen? "
                "How long should I wait? I already asked, I was told that this happens when the rewards "
                "are more than $ 300, but already more than $ 400. Nothing happens.",
                channel_id=9042,
            )

        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertGreaterEqual(len(docs_calls), 1)
        lowered = outcome.raw_final_reply.lower()
        self.assertIn("rewards", lowered)
        self.assertTrue("delay" in lowered or "normal" in lowered)
        self.assertNoDataLookupDetour(lowered)
        self.assertNotIn("what would you like me to do with this address", lowered)
        self.assertNotIn("okay, please describe your issue", lowered)

    async def test_security_report_gist_submission_uses_repo_pretriage_before_answering(
        self,
    ) -> None:
        async def fake_fetch_report_artifact(report_url: str, max_chars: int = 12000) -> str:
            self.assertIn("gist.github.com", report_url)
            return (
                "Fetched public report artifact from: https://gist.github.com/.../dc2c924b6d84727f244babc09360e16f\n\n"
                "File: report.md\n"
                "Claim: In stYFI LiquidLockerDepositor.vy, calling unstake when the caller already has a partially vested stream "
                "resets the stream start time to block.timestamp and re-locks previously vested but unredeemed funds for 14 days."
            )

        with patch("tools_lib.core_fetch_report_artifact", new=fake_fetch_report_artifact):
            outcome = await self._run_ticket_flow(
                "Report : https://gist.github.com/AlphaN0x/dc2c924b6d84727f244babc09360e16f",
                channel_id=9032,
                initial_button_intent="bug_report",
            )

        lowered = outcome.raw_final_reply.lower()
        self.assertIn("liquidlockerdepositor.vy", lowered)
        self.assertIn("block.timestamp", lowered)
        self.assertNoGenericBugTriage(lowered)
        if outcome.requires_human_handoff:
            self.assertIn("docs.yearn.fi/developers/security", lowered)
        else:
            self.assertTrue("documented" in lowered or "expected behavior" in lowered)

    async def test_sdyfi_unstake_behavior_question_gives_grounded_repo_answer_with_clear_limit(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "Hi team,\n\n"
            "I was reviewing the sdYFI Liquid Locker contract and noticed a specific behavior with the unstake "
            "function that I wanted to clarify with you.\n"
            "When a user has an existing stream containing already-vested but unredeemed tokens, calling "
            "unstake again bundles those previously vested funds into the new stream and resets the start time "
            "to block.timestamp. This effectively re-locks or temporarily freezes the previously vested funds "
            "for another 14 days.\n"
            "Is this temporary freezing of unredeemed funds intended behavior, or is it an unintended edge "
            "case? I have a quick Foundry test demonstrating the flow if it helps.\n\n"
            "Thanks!",
            channel_id=9029,
        )

        lowered = outcome.raw_final_reply.lower()
        self.assertEqual(outcome.completed_agent_key, "bug")
        self.assertTrue("block.timestamp" in lowered or "14-day" in lowered or "re-lock" in lowered)
        self.assertTrue("liquidlockerdepositor.vy" in lowered or "liquid locker" in lowered)
        self.assertNoGenericBugTriage(lowered)

    async def test_sdyfi_pasted_poc_followup_stays_repo_grounded_without_browser_questions(
        self,
    ) -> None:
        first_outcome = await self._run_ticket_flow(
            "Hi team,\n\n"
            "I was reviewing the sdYFI Liquid Locker contract and noticed a specific behavior with the unstake "
            "function that I wanted to clarify with you.\n"
            "When a user has an existing stream containing already-vested but unredeemed tokens, calling "
            "unstake again bundles those previously vested funds into the new stream and resets the start time "
            "to block.timestamp. This effectively re-locks or temporarily freezes the previously vested funds "
            "for another 14 days.\n"
            "Is this temporary freezing of unredeemed funds intended behavior, or is it an unintended edge case?\n\n"
            "I have a quick Foundry test demonstrating the flow and can paste it below.",
            channel_id=9034,
            initial_button_intent="investigate_issue",
        )

        second_outcome = await self._run_ticket_flow(
            "```solidity\n"
            "function test_TemporaryFreezingOfVestedFunds() public {\n"
            "    locker.unstake(firstUnstake);\n"
            "    vm.warp(block.timestamp + 14 days);\n"
            "    uint256 redeemableBefore = locker.maxRedeem(alice);\n"
            "    locker.unstake(secondUnstake);\n"
            "    uint256 redeemableAfter = locker.maxRedeem(alice);\n"
            "    assertEq(redeemableAfter, 0, 'BUG');\n"
            "}\n"
            "```",
            channel_id=9034,
            current_history=first_outcome.conversation_history,
            last_specialty=first_outcome.completed_agent_key,
        )

        lowered = second_outcome.raw_final_reply.lower()
        self.assertTrue(
            "liquidlockerdepositor" in lowered or "stakedyfi" in lowered,
            second_outcome.raw_final_reply,
        )
        self.assertTrue(
            "block.timestamp" in lowered or "maxredeem" in lowered,
            second_outcome.raw_final_reply,
        )
        self.assertNoGenericBugTriage(lowered)

    async def test_legacy_positions_link_request_routes_to_docs_instead_of_bug_flow(
        self,
    ) -> None:
        tool_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            tool_calls.append(user_query)
            return (
                "Use https://legacy.yearn.fi/ to view legacy Yearn positions that may not appear in the newer UI."
            )

        with patch("tools_lib.core_answer_from_docs", new=fake_answer_from_docs):
            outcome = await self._run_ticket_flow(
                "I opened a ticket before and lost the URL to see all my positions. "
                "My wallet is 0x23d402C2058052f0B9c24Fb4E17DE8cC1E3a0cb0. "
                "I have one USDC.e vault but another one is not being shown in the UI. "
                "You sent me before a link to see the legacy vaults and I could see it from there. "
                "Can you send that URL again?",
                channel_id=9015,
            )

        self.assertEqual(outcome.completed_agent_key, "docs")
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertIn("url", tool_calls[0].lower())

        lowered = outcome.raw_final_reply.lower()
        self.assertIn("legacy.yearn.fi", lowered)
        self.assertNotIn("describe the bug", lowered)
        self.assertNoGenericBugTriage(lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)

    async def test_specific_vault_url_request_routes_to_data_lookup(
        self,
    ) -> None:
        vault_address = "0x9FA306b1F4a6a83FEC98d8eBbaBEDfF78C407f6B"
        tool_calls: list[dict] = []
        tool = self._get_agent_tool(yearn_data_agent, "search_vaults_tool")

        async def fake_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            tool_calls.append(payload)
            return (
                "Vault: USDC.e-2\n"
                f"Address: {vault_address}\n"
                "Yearn UI Link: https://legacy.yearn.fi/v3/42161/"
                f"{vault_address}\n"
            )

        with patch.object(tool, "on_invoke_tool", new=fake_on_invoke_tool):
            outcome = await self._run_ticket_flow(
                "I cannot find vault 0x9FA306b1F4a6a83FEC98d8eBbaBEDfF78C407f6B in the UI. "
                "Can you send me the Yearn URL for that vault?",
                channel_id=9016,
            )

        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["query"], vault_address)
        self.assertIn("legacy.yearn.fi", outcome.raw_final_reply.lower())

        lowered = outcome.raw_final_reply.lower()
        self.assertNotIn("describe the bug", lowered)
        self.assertNoGenericBugTriage(lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)
