import os
import json
import unittest
from unittest.mock import patch

from agents import RunConfig, Runner

import config
from router import select_starting_agent
from state import (
    BotRunContext,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
)
from support_agents import (
    triage_agent,
    yearn_bug_triage_agent,
    yearn_data_agent,
    yearn_docs_qa_agent,
)
from ticket_investigation_runtime import TicketInvestigationRuntime, TicketTurnRequest


RUN_LLM_E2E = bool(config.OPENAI_API_KEY) and os.getenv("RUN_LLM_E2E_TESTS") == "1"

AGENTS_BY_KEY = {
    "data": yearn_data_agent,
    "docs": yearn_docs_qa_agent,
    "bug": yearn_bug_triage_agent,
    "triage": triage_agent,
}


@unittest.skipUnless(
    RUN_LLM_E2E,
    "Set RUN_LLM_E2E_TESTS=1 with OPENAI_API_KEY available to run real-LLM end-to-end tests.",
)
class LlmEndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None
    wallet_address = "0x1111111111111111111111111111111111111111"
    vault_address = "0x2222222222222222222222222222222222222222"

    async def _run_support_turn(
        self,
        message: str,
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9001,
    ) -> tuple[str, BotRunContext, str]:
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        starting_agent_key = select_starting_agent(message, context)
        return await self._run_agent_turn(
            starting_agent_key,
            message,
            context=context,
        )

    async def _run_agent_turn(
        self,
        starting_agent_key: str,
        message,
        *,
        context: BotRunContext,
    ) -> tuple[str, BotRunContext, str]:
        starting_agent = AGENTS_BY_KEY[starting_agent_key]
        result = await Runner.run(
            starting_agent=starting_agent,
            input=message,
            context=context,
            max_turns=6,
            run_config=RunConfig(
                workflow_name="tests.llm_e2e",
                tracing_disabled=True,
            ),
        )
        return result.final_output or "", context, starting_agent_key

    async def _run_ticket_flow(
        self,
        message: str,
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9200,
        current_history=None,
    ):
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        history = current_history or []
        input_list = history + [{"role": "user", "content": message}]
        runtime = TicketInvestigationRuntime(Runner)
        try:
            return await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text=message,
                    input_list=input_list,
                    current_history=history,
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

    @staticmethod
    def _get_agent_tool(agent, tool_name: str):
        for tool in agent.tools:
            if tool.name == tool_name:
                return tool
        raise AssertionError(f"Tool {tool_name} not found on agent {agent.name}.")

    async def test_sensitive_report_receipt_confirmation_escalates_without_repro_questions(
        self,
    ) -> None:
        output, _, starting_agent_key = await self._run_support_turn(
            "I sent an encrypted security report earlier. Can you confirm it was received?"
        )

        self.assertEqual(starting_agent_key, "triage")
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, output)

        lowered = output.lower()
        self.assertNotIn("browser", lowered)
        self.assertNotIn("device", lowered)
        self.assertNotIn("steps to reproduce", lowered)
        self.assertNotIn("screen recording", lowered)

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
        self.assertNotIn("can you clarify", lowered)
        self.assertNotIn("what do you mean", lowered)

    async def test_tx_hash_bug_report_uses_onchain_inspection_without_wallet_prompt(
        self,
    ) -> None:
        tx_hash = "0xf328a478f4a8ac0341a4f3c470c37ab4a3e41277a7abe0f06fe88a7c48dbde10"
        tool_calls: list[dict] = []

        async def fake_inspect_onchain(**kwargs) -> str:
            tool_calls.append(kwargs)
            return (
                "Onchain inspection summary\n"
                f"tx_hash: {tx_hash}\n"
                "chain: ethereum\n"
                "status: reverted\n"
                "decoded_logs: []\n"
                "reason: allowance too low for the attempted approval flow"
            )

        with patch("tools_lib.core_inspect_onchain", new=fake_inspect_onchain):
            context = BotRunContext(
                channel_id=9008,
                project_context="yearn",
                initial_button_intent="bug_report",
            )
            output, _, starting_agent_key = await self._run_agent_turn(
                "bug",
                "My approval failed on Ethereum. Please inspect this tx hash and tell me what happened: "
                f"{tx_hash}",
                context=context,
            )

        self.assertEqual(starting_agent_key, "bug")
        self.assertGreaterEqual(len(tool_calls), 1)
        matching_calls = [
            call for call in tool_calls
            if call.get("chain") == "ethereum"
            and call.get("mode") == "receipt"
            and call.get("tx_hash") == tx_hash
        ]
        self.assertTrue(matching_calls)

        lowered = output.lower()
        self.assertIn("approval", lowered)
        self.assertIn("reverted", lowered)
        self.assertNotIn("what browser", lowered)

    async def test_data_agent_tx_hash_followup_investigates_instead_of_claiming_forward(
        self,
    ) -> None:
        tx_hash = "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        context = BotRunContext(
            channel_id=9010,
            project_context="yearn",
        )
        output, _, starting_agent_key = await self._run_agent_turn(
            "data",
            [
                {
                    "role": "user",
                    "content": (
                        "I deposited 1990.3798 frxusd into the usdt vault on katana "
                        "but was only credited with 650.9147 yvvbusdt."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Please send the transaction hash and I will check it.",
                },
                {
                    "role": "user",
                    "content": f"Katana tx hash: {tx_hash}",
                },
            ],
            context=context,
        )

        self.assertEqual(starting_agent_key, "data")

        lowered = output.lower()
        self.assertNotIn("forwarded", lowered)
        self.assertNotIn("transferred", lowered)
        self.assertTrue("transaction" in lowered or tx_hash.lower() in lowered)
        self.assertTrue("chain" in lowered or "katana" in lowered)

    async def test_deposit_button_intent_calls_check_all_deposits_tool(self) -> None:
        tool_calls: list[dict] = []
        tool = self._get_agent_tool(yearn_data_agent, "check_all_deposits_tool")

        async def fake_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            tool_calls.append(payload)
            return (
                "Deposit summary\n"
                "Vault: yvUSDC (yvUSDC)\n"
                f"User: {payload['user_address_or_ens']}\n"
                "Shares: 12.34"
            )

        with patch.object(tool, "on_invoke_tool", new=fake_on_invoke_tool):
            output, _, starting_agent_key = await self._run_support_turn(
                self.wallet_address,
                initial_button_intent="data_deposit_check",
                channel_id=9002,
            )

        self.assertEqual(starting_agent_key, "data")
        self.assertEqual(
            tool_calls,
            [{"user_address_or_ens": self.wallet_address, "token_symbol": None}],
        )
        self.assertIn("yvusdc", output.lower())

    async def test_combined_data_button_routes_vault_query_to_search_tool(self) -> None:
        tool_calls: list[dict] = []
        tool = self._get_agent_tool(yearn_data_agent, "search_vaults_tool")

        async def fake_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            tool_calls.append(payload)
            return (
                "Vault: Yearn USDC (yvUSDC)\n"
                f"Address: {self.vault_address}\n"
                "Current Net APY (compounded): 4.20%"
            )

        with patch.object(tool, "on_invoke_tool", new=fake_on_invoke_tool):
            output, _, starting_agent_key = await self._run_support_turn(
                "usdc",
                initial_button_intent="data_deposits_withdrawals_start",
                channel_id=9003,
            )

        self.assertEqual(starting_agent_key, "data")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["query"].lower(), "usdc")
        self.assertIsNone(tool_calls[0]["chain"])
        self.assertIsNone(tool_calls[0]["sort_by"])
        self.assertIn("4.20%", output)

    async def test_withdrawal_button_intent_checks_deposits_before_withdrawal_tool(
        self,
    ) -> None:
        deposit_calls: list[dict] = []
        withdrawal_calls: list[dict] = []
        deposit_tool = self._get_agent_tool(yearn_data_agent, "check_all_deposits_tool")
        withdrawal_tool = self._get_agent_tool(
            yearn_data_agent,
            "get_withdrawal_instructions_tool",
        )

        async def fake_check_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            deposit_calls.append(payload)
            return (
                "Deposit summary\n"
                "Vault: Yearn USDC (yvUSDC)\n"
                f"Address: {self.vault_address}\n"
                "Shares: 12.34"
            )

        async def fake_withdrawal_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            withdrawal_calls.append(payload)
            return "Withdrawal instructions should not be called in the first turn."

        with patch.object(deposit_tool, "on_invoke_tool", new=fake_check_on_invoke_tool):
            with patch.object(
                withdrawal_tool,
                "on_invoke_tool",
                new=fake_withdrawal_on_invoke_tool,
            ):
                output, _, starting_agent_key = await self._run_support_turn(
                    self.wallet_address,
                    initial_button_intent="data_withdrawal_flow_start",
                    channel_id=9004,
                )

        self.assertEqual(starting_agent_key, "data")
        self.assertEqual(len(deposit_calls), 1)
        self.assertEqual(withdrawal_calls, [])

        lowered = output.lower()
        self.assertIn("which", lowered)
        self.assertIn("vault", lowered)

    async def test_free_form_withdrawal_request_routes_through_ticket_router_to_data(
        self,
    ) -> None:
        tool_calls: list[dict] = []
        tool = self._get_agent_tool(yearn_data_agent, "get_withdrawal_instructions_tool")

        async def fake_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            tool_calls.append(payload)
            return (
                "Withdrawal Instructions\n"
                f"Wallet: {payload['user_address_or_ens']}\n"
                f"Vault: {payload['vault_address']}\n"
                f"Chain: {payload['chain']}"
            )

        with patch.object(tool, "on_invoke_tool", new=fake_on_invoke_tool):
            outcome = await self._run_ticket_flow(
                "How do I withdraw from vault "
                f"{self.vault_address} on ethereum using wallet {self.wallet_address}?",
                channel_id=9006,
            )

        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["user_address_or_ens"], self.wallet_address)
        self.assertEqual(tool_calls[0]["vault_address"], self.vault_address)
        self.assertEqual(tool_calls[0]["chain"], "ethereum")

        lowered = outcome.raw_final_reply.lower()
        self.assertIn("withdrawal", lowered)
        self.assertNotIn("transfer", lowered)
        self.assertNotIn("forward", lowered)

    async def test_ticket_router_escalation_returns_handoff_without_specialist_text(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "I sent an encrypted security report earlier. Can you confirm it was received?",
            channel_id=9011,
        )

        self.assertIsNone(outcome.completed_agent_key)
        self.assertTrue(outcome.requires_human_handoff)
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, outcome.raw_final_reply)

        lowered = outcome.raw_final_reply.lower()
        self.assertNotIn("transfer", lowered)
        self.assertNotIn("forward", lowered)

    async def test_data_agent_free_form_withdrawal_request_calls_withdrawal_tool(
        self,
    ) -> None:
        tool_calls: list[dict] = []
        tool = self._get_agent_tool(yearn_data_agent, "get_withdrawal_instructions_tool")

        async def fake_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            tool_calls.append(payload)
            return (
                "Withdrawal Instructions\n"
                f"Wallet: {payload['user_address_or_ens']}\n"
                f"Vault: {payload['vault_address']}\n"
                f"Chain: {payload['chain']}"
            )

        with patch.object(tool, "on_invoke_tool", new=fake_on_invoke_tool):
            context = BotRunContext(
                channel_id=9007,
                project_context="yearn",
            )
            output, _, starting_agent_key = await self._run_agent_turn(
                "data",
                "How do I withdraw from vault "
                f"{self.vault_address} on ethereum using wallet {self.wallet_address}?",
                context=context,
            )

        self.assertEqual(starting_agent_key, "data")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["user_address_or_ens"], self.wallet_address)
        self.assertEqual(tool_calls[0]["vault_address"], self.vault_address)
        self.assertEqual(tool_calls[0]["chain"], "ethereum")
        self.assertIn("withdrawal", output.lower())
