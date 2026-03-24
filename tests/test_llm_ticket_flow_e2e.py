import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import config
from state import BotRunContext
from support_agents import yearn_data_agent
from tests.llm_e2e_harness import LlmE2EBase, RUN_LLM_E2E


@unittest.skipUnless(
    RUN_LLM_E2E,
    "Set RUN_LLM_E2E_TESTS=1 with OPENAI_API_KEY available to run real-LLM end-to-end tests.",
)
class LlmTicketFlowTests(LlmE2EBase):
    async def test_tx_hash_bug_report_uses_onchain_inspection_without_wallet_prompt(
        self,
    ) -> None:
        tx_hash = "0xf328a478f4a8ac0341a4f3c470c37ab4a3e41277a7abe0f06fe88a7c48dbde10"
        tool_calls: list[dict] = []
        repo_queries: list[str] = []

        async def fake_inspect_onchain(**kwargs) -> str:
            tool_calls.append(kwargs)
            return (
                "Transaction investigation\n"
                f"tx_hash: {tx_hash}\n"
                "chain: ethereum\n"
                "status: reverted\n"
                "user_transfers_out: none\n"
                "user_transfers_in: none\n"
                "approvals:\n"
                "- token: USDC\n"
                "  owner: 0x1111111111111111111111111111111111111111\n"
                "  spender: 0x2222222222222222222222222222222222222222\n"
                "  value_formatted: 0\n"
                "contracts_profiled:\n"
                "- address: 0x1111111111111111111111111111111111111111\n"
                "  kind: erc20_like\n"
                "  symbol: USDC\n"
                "notable_findings:\n"
                "- No allowance existed for the attempted approval flow.\n"
                "- No funds moved in the reverted transaction.\n"
                "reason: allowance too low for the attempted approval flow"
            )

        async def fake_search_repo_context(*, query: str, limit: int = 5, include_legacy=None, include_ui=None) -> str:
            repo_queries.append(query)
            return (
                "Repo search results\n"
                "- repo: yearn-vaults-v3\n"
                "  path: contracts/VaultV3.vy\n"
                "  artifact_ref: segment:allowance\n"
            )

        async def fake_fetch_repo_artifacts(*, artifact_refs_text: str) -> str:
            return (
                "Repo artifact\n"
                "repo: yearn-vaults-v3\n"
                "path: contracts/VaultV3.vy\n"
                "symbol: _spend_allowance\n"
                "summary: current_allowance must be at least amount or the call reverts.\n"
            )

        with patch("tools_lib.core_inspect_onchain", new=fake_inspect_onchain):
            with patch("tools_lib.core_search_repo_context", new=fake_search_repo_context):
                with patch("tools_lib.core_fetch_repo_artifacts", new=fake_fetch_repo_artifacts):
                    outcome = await self._run_ticket_flow(
                        "My approval failed on Ethereum. Please inspect this tx hash and tell me what happened: "
                        f"{tx_hash}",
                        initial_button_intent="bug_report",
                        channel_id=9008,
                    )

        self.assertEqual(outcome.completed_agent_key, "bug")
        self.assertGreaterEqual(len(tool_calls), 1)
        matching_calls = [
            call for call in tool_calls
            if call.get("chain") == "ethereum"
            and call.get("mode") == "tx_investigate"
            and call.get("tx_hash") == tx_hash
        ]
        self.assertTrue(matching_calls)

        lowered = outcome.raw_final_reply.lower()
        self.assertIn("approval", lowered)
        self.assertTrue(
            "reverted" in lowered
            or "failed" in lowered
            or "allowance-state failure" in lowered
        )
        self.assertNotIn("what browser", lowered)

    async def test_ticket_flow_tx_hash_followup_investigates_without_branching_question(
        self,
    ) -> None:
        tx_hash = "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        tool_calls: list[dict] = []

        async def fake_inspect_onchain(**kwargs) -> str:
            tool_calls.append(kwargs)
            if kwargs.get("mode") != "tx_investigate" or sum(
                1 for call in tool_calls if call.get("mode") == "tx_investigate"
            ) > 1:
                return (
                    "Detailed transaction investigation\n"
                    f"tx_hash: {tx_hash}\n"
                    "chain: katana\n"
                    "input_token: frxUSD\n"
                    "input_amount: 1990.379783\n"
                    "output_token: yvWBUSDT\n"
                    "output_amount: 650.9147\n"
                    "explanation: the transaction minted 650.9147 yvWBUSDT shares after routing the frxUSD deposit."
                )
            return (
                "Transaction summary\n"
                f"tx_hash: {tx_hash}\n"
                "chain: katana\n"
                "status: success\n"
                "contracts_profiled:\n"
                "- address: 0xC10eE9031F2a0B84766A86B55a8D90F357910fb4\n"
                "  kind: erc4626_like\n"
                "  symbol: yvWBUSDT\n"
                "events:\n"
                "- event: Transfer\n"
                "  value_formatted: 650.9147"
            )

        with patch("tools_lib.core_inspect_onchain", new=fake_inspect_onchain):
            outcome = await self._run_ticket_flow(
                "Katana tx hash: " + tx_hash,
                channel_id=9010,
                last_specialty="data",
                current_history=[
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
                ],
            )

        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertTrue(
            any(
                call.get("chain") == "katana"
                and call.get("mode") == "tx_investigate"
                and call.get("tx_hash") == tx_hash
                for call in tool_calls
            )
        )

        lowered = outcome.raw_final_reply.lower()
        self.assertNotIn("forwarded", lowered)
        self.assertNotIn("transferred to", lowered)
        self.assertNotIn("would you like me to", lowered)
        self.assertNotIn("if you need a deeper decode", lowered)
        self.assertNotIn("tell me what to inspect next", lowered)
        self.assertNotIn("which would you like", lowered)
        self.assertNotIn("decode the logs", lowered)
        self.assertNotIn("inspect a specific vault", lowered)
        self.assertTrue("transaction" in lowered or tx_hash.lower() in lowered)

    async def test_ticket_flow_follow_up_look_into_it_reuses_known_tx_context(self) -> None:
        tx_hash = "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        tool_calls: list[dict] = []

        async def fake_inspect_onchain(**kwargs) -> str:
            tool_calls.append(kwargs)
            if kwargs.get("mode") != "tx_investigate" or sum(
                1 for call in tool_calls if call.get("mode") == "tx_investigate"
            ) > 1:
                return (
                    "Detailed transaction investigation\n"
                    f"tx_hash: {kwargs.get('tx_hash')}\n"
                    f"chain: {kwargs.get('chain')}\n"
                    "input_token: frxUSD\n"
                    "input_amount: 1990.379783\n"
                    "output_token: yvWBUSDT\n"
                    "output_amount: 650.9147\n"
                    "explanation: the transaction minted 650.9147 yvWBUSDT shares after routing the frxUSD deposit."
                )
            return (
                "Transaction summary\n"
                f"tx_hash: {kwargs.get('tx_hash')}\n"
                f"chain: {kwargs.get('chain')}\n"
                "status: success\n"
                "contracts_profiled:\n"
                "- address: 0xC10eE9031F2a0B84766A86B55a8D90F357910fb4\n"
                "  kind: erc4626_like\n"
                "  symbol: yvWBUSDT\n"
                "events:\n"
                "- event: Transfer\n"
                "  value_formatted: 650.9147"
            )

        with patch("tools_lib.core_inspect_onchain", new=fake_inspect_onchain):
            outcome = await self._run_ticket_flow(
                "i dunno man. look into it",
                channel_id=9012,
                last_specialty="data",
                current_history=[
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
                    {
                        "role": "assistant",
                        "content": "Transaction summary\nchain: katana\nstatus: success",
                    },
                ],
            )

        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertTrue(
            any(
                call.get("chain") == "katana"
                and call.get("mode") == "tx_investigate"
                and call.get("tx_hash") == tx_hash
                for call in tool_calls
            )
        )

        lowered = outcome.raw_final_reply.lower()
        self.assertNotIn("which chain", lowered)
        self.assertNotIn("please provide the tx hash", lowered)
        self.assertNotIn("would you like me to", lowered)
        self.assertNotIn("which would you like", lowered)

    async def test_katana_discrepancy_transcript_keeps_investigating_without_user_branch_selection(
        self,
    ) -> None:
        tx_hash = "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        tool_calls: list[dict] = []

        async def fake_inspect_onchain(**kwargs) -> str:
            tool_calls.append(kwargs)
            if kwargs.get("mode") != "tx_investigate" or sum(
                1 for call in tool_calls if call.get("mode") == "tx_investigate"
            ) > 1:
                return (
                    "Detailed transaction investigation\n"
                    f"tx_hash: {tx_hash}\n"
                    "chain: katana\n"
                    "input_token: frxUSD\n"
                    "input_amount: 1990.379783\n"
                    "deposit_event: yvvbUSDT deposit with 650.914712 shares\n"
                    "output_token: yvvbUSDT\n"
                    "output_amount: 650.914712\n"
                    "explanation: the tx deposited vbUSDT and minted yvvbUSDT shares for the user."
                )
            return (
                "Transaction investigation\n"
                f"tx_hash: {tx_hash}\n"
                "chain: katana\n"
                "status: success\n"
                "investigation:\n"
                "- user_transfers_out: frxUSD 1990.379783\n"
                "- deposits: yvvbUSDT shares 650.914712\n"
                "- user_transfers_in: yvvbUSDT 650.914712\n"
                "- notable_findings: explicit deposit event decoded\n"
            )

        with patch("tools_lib.core_inspect_onchain", new=fake_inspect_onchain):
            outcomes = await self._run_ticket_transcript(
                [
                    "I deposited 1990.3798 frxusd into the usdt vault on katana but was only credited with 650.9147 yvvbusdt",
                    tx_hash,
                    "i dunno man. look into it",
                ],
                channel_id=9013,
            )

        self.assertEqual(len(outcomes), 3)
        self.assertTrue(
            "transaction" in outcomes[0].raw_final_reply.lower()
            or "wallet" in outcomes[0].raw_final_reply.lower()
        )
        self.assertTrue(
            any(
                call.get("chain") == "katana"
                and call.get("mode") == "tx_investigate"
                and call.get("tx_hash") == tx_hash
                for call in tool_calls
            )
        )

        second_reply = outcomes[1].raw_final_reply.lower()
        third_reply = outcomes[2].raw_final_reply.lower()
        self.assertNotEqual(second_reply.strip(), "")
        self.assertNotEqual(third_reply.strip(), "")
        self.assertNotIn("please provide the tx hash", third_reply)
        self.assertNotIn("which chain", third_reply)
        self.assertTrue(
            "yvvbusdt" in third_reply
            or config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower() in third_reply
        )

    async def test_katana_discrepancy_transcript_investigates_comparison_tx_without_resetting(
        self,
    ) -> None:
        first_tx = "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"
        second_tx = "0x53db23a0e15f250dd30978bd2a3499289e73bcec25590e1f49063b78e3bbd3f9"
        tool_calls: list[dict] = []

        async def fake_inspect_onchain(**kwargs) -> str:
            tool_calls.append(kwargs)
            tx_hash = kwargs.get("tx_hash")
            if tx_hash == first_tx:
                return (
                    "Transaction investigation\n"
                    f"tx_hash: {first_tx}\n"
                    "chain: katana\n"
                    "output_token: yvvbUSDT\n"
                    "output_amount: 650.914712\n"
                )
            return (
                "Transaction investigation\n"
                f"tx_hash: {second_tx}\n"
                "chain: katana\n"
                "output_token: yvvbUSDT\n"
                "output_amount: 1872.000000\n"
            )

        with patch("tools_lib.core_inspect_onchain", new=fake_inspect_onchain):
            outcomes = await self._run_ticket_transcript(
                [
                    "I deposited 1990.3798 frxusd into the usdt vault on katana but was only credited with 650.9147 yvvbusdt",
                    first_tx,
                    f"This is the similar transaction from minutes earlier. {second_tx}",
                ],
                channel_id=9014,
            )

        self.assertEqual(len(outcomes), 3)
        self.assertTrue(
            any(
                call.get("tx_hash") == first_tx and call.get("mode") == "tx_investigate"
                for call in tool_calls
            )
        )
        self.assertTrue(
            any(
                call.get("tx_hash") == second_tx and call.get("mode") == "tx_investigate"
                for call in tool_calls
            )
        )

        lowered = outcomes[2].raw_final_reply.lower()
        self.assertNotIn("forwarded", lowered)
        self.assertNotIn("transferred to specialists", lowered)
        self.assertNotIn("please provide the tx hash", lowered)

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
        self.assertIn(tool_calls[0]["sort_by"], (None, "highest_apr"))
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
        self.assertIn(self.wallet_address.lower(), lowered)
        self.assertIn(self.vault_address.lower()[:20], lowered)
        self.assertIn("chain: ethereum", lowered)
        self.assertNotIn("forward", lowered)

    async def test_withdrawal_followup_reuses_single_listed_deposit_chain_and_vault(
        self,
    ) -> None:
        withdrawal_calls: list[dict] = []
        deposit_calls: list[dict] = []
        withdrawal_tool = self._get_agent_tool(
            yearn_data_agent,
            "get_withdrawal_instructions_tool",
        )
        deposit_tool = self._get_agent_tool(yearn_data_agent, "check_all_deposits_tool")
        katana_vault = "0x80c34BD3A3569E126e7055831036aa7b212cB159"

        async def fake_withdrawal_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            withdrawal_calls.append(payload)
            return (
                "Withdrawal Instructions\n"
                f"Wallet: {payload['user_address_or_ens']}\n"
                f"Vault: {payload['vault_address']}\n"
                f"Chain: {payload['chain']}"
            )

        async def fake_deposit_on_invoke_tool(ctx, input: str) -> str:
            payload = json.loads(input)
            deposit_calls.append(payload)
            return "Deposit tool should not be called for this follow-up."

        with patch.object(
            withdrawal_tool,
            "on_invoke_tool",
            new=fake_withdrawal_on_invoke_tool,
        ):
            with patch.object(
                deposit_tool,
                "on_invoke_tool",
                new=fake_deposit_on_invoke_tool,
            ):
                outcome = await self._run_ticket_flow(
                    "I'd like support withdrawing",
                    channel_id=9008,
                    last_specialty="data",
                    remembered_withdrawal_target=("katana", katana_vault),
                    current_history=[
                        {"role": "user", "content": self.wallet_address},
                        {
                            "role": "assistant",
                            "content": (
                                "**Katana Active Deposits:**\n"
                                "**Vault:** [Katana Vault](https://yearn.fi/v3/146/0x80c34BD3A3569E126e7055831036aa7b212cB159) (Symbol: yvVBUSDT)\n"
                                "  Address: `0x80c34BD3A3569E126e7055831036aa7b212cB159`\n"
                                "  Total Position: **1.000000 yvVBUSDT**\n\n"
                                "Which of these vaults would you like withdrawal instructions for?"
                            ),
                        },
                    ],
                )

        self.assertEqual(outcome.completed_agent_key, "data")
        self.assertEqual(deposit_calls, [])
        self.assertEqual(len(withdrawal_calls), 1)
        self.assertEqual(withdrawal_calls[0]["user_address_or_ens"], self.wallet_address)
        self.assertEqual(withdrawal_calls[0]["vault_address"], katana_vault)
        self.assertEqual(withdrawal_calls[0]["chain"], "katana")

        lowered = outcome.raw_final_reply.lower()
        self.assertIn(self.wallet_address.lower(), lowered)
        self.assertIn(katana_vault.lower(), lowered)
        self.assertIn("katana", lowered)
        self.assertNotIn("which vault", lowered)
        self.assertNotIn("re-check", lowered)
        self.assertNotIn("ethereum", lowered)

    async def test_veyfi_migration_followup_after_data_turn_points_directly_to_legacy_manage(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "i have 1.0893 veYFI . they are unlocked now since yesterday. "
            "but i am unable to migrate and cant find out why.",
            channel_id=9040,
            last_specialty="data",
            current_history=[
                {"role": "user", "content": self.wallet_address},
                {
                    "role": "assistant",
                    "content": (
                        "**Active Deposits:**\n"
                        "**Vault:** [Ethereum Vault](https://yearn.fi/v3/1/0x6dfb4ab47a5d2947c4f0f6ea20f92955295c5f5e) (Symbol: yvUSDC-1)\n"
                        "  Address: `0x6dfb4ab47a5d2947c4f0f6ea20f92955295c5f5e`\n"
                        "  Total Position: **1.000000 yvUSDC-1**"
                    ),
                },
            ],
        )

        self.assertEqual(outcome.completed_agent_key, "docs")
        lowered = outcome.raw_final_reply.lower()
        self.assertIn("legacy-veyfi.yearn.fi/manage", lowered)
        self.assertIn("styfi.yearn.fi", lowered)
        self.assertNotIn("browser", lowered)
        self.assertNotIn("screenshot", lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)

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

    async def test_ticket_flow_runs_through_codex_exec_endpoint_mode(self) -> None:
        fake_codex = (
            "import asyncio,pathlib,sys; "
            "from dotenv import load_dotenv; "
            "from agents import set_default_openai_key; "
            "import config; "
            "from ticket_investigation_worker_cli import execute_request_json; "
            "load_dotenv(); "
            "set_default_openai_key(config.OPENAI_API_KEY); "
            "sys.stdin.read(); "
            "args=sys.argv[1:]; "
            "run_dir=pathlib.Path(args[args.index('-C') + 1]); "
            "request_json=(run_dir/'request.json').read_text(encoding='utf-8'); "
            "sys.stdout.write(asyncio.run(execute_request_json(request_json)))"
        )
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_codex_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            with tempfile.TemporaryDirectory() as artifact_dir:
                config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
                config.TICKET_EXECUTION_FALLBACK_ENDPOINT = ""
                config.TICKET_EXECUTION_CODEX_COMMAND = [sys.executable, "-c", fake_codex]
                config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [
                    [sys.executable, "-c", fake_codex]
                ]
                config.TICKET_EXECUTION_ARTIFACT_DIR = artifact_dir

                result = await self._run_ticket_flow_through_executor(
                    "I sent an encrypted security report earlier. Can you confirm it was received?",
                    channel_id=9310,
                )

                artifact_entries = os.listdir(artifact_dir)
                self.assertEqual(len(artifact_entries), 1)
                run_dir = os.path.join(artifact_dir, artifact_entries[0])
                self.assertTrue(os.path.exists(os.path.join(run_dir, "request.json")))
                self.assertTrue(os.path.exists(os.path.join(run_dir, "codex_prompt.txt")))
                self.assertTrue(os.path.exists(os.path.join(run_dir, "stdout.txt")))
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_COMMAND = original_codex_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertTrue(result.flow_outcome.requires_human_handoff)
        self.assertIn(
            config.HUMAN_HANDOFF_TAG_PLACEHOLDER,
            result.flow_outcome.raw_final_reply,
        )

    async def test_ticket_flow_falls_back_from_codex_exec_to_local_endpoint(self) -> None:
        failing_codex = (
            "import sys; "
            "sys.stdin.read(); "
            "print('codex wrapper failed', file=sys.stderr); "
            "raise SystemExit(1)"
        )
        original_mode = config.TICKET_EXECUTION_ENDPOINT
        original_fallback = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
        original_codex_command = config.TICKET_EXECUTION_CODEX_COMMAND
        original_prefixes = config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES
        original_artifact_dir = config.TICKET_EXECUTION_ARTIFACT_DIR
        try:
            with tempfile.TemporaryDirectory() as artifact_dir:
                config.TICKET_EXECUTION_ENDPOINT = "codex_exec"
                config.TICKET_EXECUTION_FALLBACK_ENDPOINT = "local"
                config.TICKET_EXECUTION_CODEX_COMMAND = [sys.executable, "-c", failing_codex]
                config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = [
                    [sys.executable, "-c", failing_codex]
                ]
                config.TICKET_EXECUTION_ARTIFACT_DIR = artifact_dir

                result = await self._run_ticket_flow_through_executor(
                    "I sent an encrypted security report earlier. Can you confirm it was received?",
                    channel_id=9311,
                )

                artifact_entries = os.listdir(artifact_dir)
                self.assertEqual(len(artifact_entries), 1)
                run_dir = os.path.join(artifact_dir, artifact_entries[0])
                with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as file:
                    metadata = json.load(file)
                self.assertEqual(metadata["returncode"], 1)
        finally:
            config.TICKET_EXECUTION_ENDPOINT = original_mode
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback
            config.TICKET_EXECUTION_CODEX_COMMAND = original_codex_command
            config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = original_prefixes
            config.TICKET_EXECUTION_ARTIFACT_DIR = original_artifact_dir

        self.assertTrue(result.flow_outcome.requires_human_handoff)
        self.assertIn(
            config.HUMAN_HANDOFF_TAG_PLACEHOLDER,
            result.flow_outcome.raw_final_reply,
        )

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
        lowered = output.lower()
        self.assertIn("withdrawal", lowered)
