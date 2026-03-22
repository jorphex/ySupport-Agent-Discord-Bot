import json
import logging
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

from agents import RunConfig, Runner
from agents.exceptions import InputGuardrailTripwireTriggered

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
from ticket_investigation_executor import (
    LocalTicketInvestigationExecutor,
    TransportTicketInvestigationExecutor,
)
from ticket_investigation_json_endpoint import (
    JsonEndpointTicketExecutionTransport,
    build_ticket_execution_json_endpoint,
)
from ticket_investigation_runtime import TicketInvestigationRuntime, TicketTurnRequest
from ticket_investigation_worker import TicketInvestigationWorker


RUN_LLM_E2E = bool(config.OPENAI_API_KEY) and os.getenv("RUN_LLM_E2E_TESTS") == "1"

AGENTS_BY_KEY = {
    "data": yearn_data_agent,
    "docs": yearn_docs_qa_agent,
    "bug": yearn_bug_triage_agent,
    "triage": triage_agent,
}

warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"unclosed transport <_SelectorSocketTransport.*",
)
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"unclosed <socket\.socket.*",
)


class _RepoContextNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.startswith("[CoreTool:search_repo_context] Repo context unavailable.")
            or message.startswith("[CoreTool:fetch_repo_artifacts] Repo context unavailable.")
        )


async def _close_agents_http_clients() -> None:
    try:
        from agents.models import openai_provider

        if openai_provider._http_client is not None:
            await openai_provider._http_client.aclose()
            openai_provider._http_client = None
    except Exception:
        pass

    try:
        from agents.voice.models import openai_model_provider

        if openai_model_provider._http_client is not None:
            await openai_model_provider._http_client.aclose()
            openai_model_provider._http_client = None
    except Exception:
        pass


@unittest.skipUnless(
    RUN_LLM_E2E,
    "Set RUN_LLM_E2E_TESTS=1 with OPENAI_API_KEY available to run real-LLM end-to-end tests.",
)
class LlmEndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None
    wallet_address = "0x1111111111111111111111111111111111111111"
    vault_address = "0x2222222222222222222222222222222222222222"
    _repo_context_noise_filter = _RepoContextNoiseFilter()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        logging.getLogger().addFilter(cls._repo_context_noise_filter)

    @classmethod
    def tearDownClass(cls) -> None:
        logging.getLogger().removeFilter(cls._repo_context_noise_filter)
        super().tearDownClass()

    async def asyncTearDown(self) -> None:
        await _close_agents_http_clients()
        await super().asyncTearDown()

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
        last_specialty: str | None = None,
    ):
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = last_specialty
        history = current_history or []
        runtime = TicketInvestigationRuntime(Runner)
        for item in history:
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                runtime.merge_explicit_evidence(investigation_job, item["content"])
        runtime.merge_explicit_evidence(investigation_job, message)
        input_list = history + [{"role": "user", "content": message}]
        contextual_hints = runtime.build_contextual_hints(
            investigation_job,
            message,
            current_history=history,
        )
        if contextual_hints:
            input_list = input_list[:-1] + [
                {"role": "system", "content": " ".join(contextual_hints)}
            ] + [input_list[-1]]
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

    async def _run_ticket_flow_through_executor(
        self,
        message: str,
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9300,
        current_history=None,
        last_specialty: str | None = None,
    ):
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = last_specialty
        history = current_history or []
        runtime = TicketInvestigationRuntime(Runner)
        for item in history:
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                runtime.merge_explicit_evidence(investigation_job, item["content"])
        runtime.merge_explicit_evidence(investigation_job, message)
        input_list = history + [{"role": "user", "content": message}]
        contextual_hints = runtime.build_contextual_hints(
            investigation_job,
            message,
            current_history=history,
        )
        if contextual_hints:
            input_list = input_list[:-1] + [
                {"role": "system", "content": " ".join(contextual_hints)}
            ] + [input_list[-1]]
        worker = TicketInvestigationWorker(runtime)
        local_executor = LocalTicketInvestigationExecutor(worker)
        endpoint = build_ticket_execution_json_endpoint(local_executor)
        transport = JsonEndpointTicketExecutionTransport(endpoint)
        executor = TransportTicketInvestigationExecutor(transport)
        try:
            return await executor.execute_turn(
                TicketTurnRequest(
                    aggregated_text=message,
                    input_list=input_list,
                    current_history=history,
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow.executor",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

    async def _run_ticket_transcript(
        self,
        user_messages: list[str],
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9400,
        last_specialty: str | None = None,
    ) -> list:
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = last_specialty
        runtime = TicketInvestigationRuntime(Runner)
        worker = TicketInvestigationWorker(runtime)
        history = []
        outcomes = []
        try:
            for index, message in enumerate(user_messages):
                if index > 0:
                    context.initial_button_intent = None
                runtime.merge_explicit_evidence(investigation_job, message)
                input_list = history + [{"role": "user", "content": message}]
                contextual_hints = runtime.build_contextual_hints(
                    investigation_job,
                    message,
                    current_history=history,
                )
                if contextual_hints:
                    input_list = input_list[:-1] + [
                        {"role": "system", "content": " ".join(contextual_hints)}
                    ] + [input_list[-1]]
                worker_result = await worker.execute_turn(
                    TicketTurnRequest(
                        aggregated_text=message,
                        input_list=input_list,
                        current_history=history,
                        run_context=context,
                        investigation_job=investigation_job,
                        workflow_name="tests.ticket_transcript",
                    )
                )
                outcomes.append(worker_result.flow_outcome)
                history = worker_result.flow_outcome.conversation_history
        finally:
            clear_ticket_investigation_job(channel_id)

        return outcomes

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

    async def test_recovery_process_question_with_wallet_routes_to_docs_not_deposit_check(
        self,
    ) -> None:
        tool_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            tool_calls.append(user_query)
            return (
                "Official sources do not document a 1:1 yETH recovery guarantee or a per-holder claim formula. "
                "Recovered assets, if any, are redistributed to affected depositors, but the exact claim amount and "
                "recovery-vault tradeoffs are not specified in the official materials I have."
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
        self.assertIn("official", lowered)
        self.assertNotIn("check your deposits", lowered)
        self.assertNotIn("check deposits", lowered)
        self.assertNotIn("do you want me to check", lowered)
        self.assertNotIn("wallet address now", lowered)

    async def test_harvest_rewards_question_routes_to_docs_not_vault_search(
        self,
    ) -> None:
        tool_calls: list[str] = []

        async def fake_answer_from_docs(user_query: str) -> str:
            tool_calls.append(user_query)
            return (
                "Official docs for Yearn factory strategies say `harvest()` is permissionless, "
                "but calling harvest only claims rewards. Swapping and realizing those rewards can happen later, "
                "so not seeing rewards immediately can be normal. If you have a specific failed tx or stuck action, "
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
        self.assertNotIn("find vaults", lowered)
        self.assertNotIn("what token, vault, or criteria", lowered)
        self.assertNotIn("check your deposits", lowered)
        self.assertNotIn("wallet address", lowered)

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
        self.assertNotIn("what browser", lowered)
        self.assertNotIn("which browser", lowered)
        self.assertNotIn("device are you using", lowered)
        self.assertNotIn("steps to reproduce", lowered)
        self.assertNotIn("what token, vault, or criteria", lowered)

    async def test_yyb_disabled_button_first_turn_routes_to_bug_without_deposit_prompt(
        self,
    ) -> None:
        outcome = await self._run_ticket_flow(
            "I cant add my yYB",
            channel_id=9020,
        )

        lowered = outcome.raw_final_reply.lower()
        self.assertNotIn("check your deposits", lowered)
        self.assertNotIn("withdrawals if needed", lowered)
        self.assertNotIn("provide your wallet address", lowered)
        self.assertNotIn("vault address", lowered)
        self.assertNotIn("listable on exchanges", lowered)
        self.assertNotIn("listing fees", lowered)
        self.assertNotIn("proposal is necessary", lowered)
        self.assertNotIn("what token, vault, or criteria", lowered)

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
        self.assertNotIn("browser", lowered)
        self.assertNotIn("device", lowered)
        self.assertNotIn("support bot stopped", lowered)
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
        self.assertNotIn("browser", lowered)
        self.assertNotIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER.lower(), lowered)

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
        self.assertIn("reverted", lowered)
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
        self.assertTrue("chain" in lowered or "katana" in lowered)

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
        self.assertNotIn("pick one", second_reply)
        self.assertNotIn("would you like me to", second_reply)
        self.assertNotIn("tell me what to inspect next", second_reply)
        self.assertNotIn("which would you like", second_reply)
        self.assertNotIn("pick one", third_reply)
        self.assertNotIn("would you like me to", third_reply)
        self.assertNotIn("tell me what to inspect next", third_reply)
        self.assertNotIn("which would you like", third_reply)
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
        self.assertIn(self.wallet_address.lower(), lowered)
        self.assertIn(self.vault_address.lower(), lowered)
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
        self.assertIn("withdrawal", lowered)
        self.assertNotIn("which vault", lowered)
        self.assertNotIn("re-check", lowered)
        self.assertNotIn("ethereum", lowered)

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
        self.assertIn(self.wallet_address.lower(), lowered)
        self.assertIn(self.vault_address.lower(), lowered)
        self.assertIn("ethereum", lowered)
