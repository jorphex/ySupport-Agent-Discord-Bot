import unittest

from agents import RunContextWrapper

import config
from router import select_starting_agent
from state import BotRunContext, ticket_investigation_states, get_or_create_ticket_investigation_state
from support_agents import triage_agent, yearn_bug_triage_agent, yearn_data_agent
from support_tools import _extract_artifact_refs, _repo_search_block_message
from ysupport import _select_ticket_starting_agent, _merge_explicit_evidence_into_state


class RoutingTests(unittest.TestCase):
    def test_select_starting_agent_uses_data_button_intent(self) -> None:
        context = BotRunContext(
            channel_id=1,
            project_context="yearn",
            initial_button_intent="data_deposit_check",
        )
        self.assertEqual(select_starting_agent("0x1234", context), "data")

    def test_select_starting_agent_uses_bug_button_intent(self) -> None:
        context = BotRunContext(
            channel_id=2,
            project_context="yearn",
            initial_button_intent="bug_report",
        )
        self.assertEqual(select_starting_agent("the claim button is missing", context), "bug")

    def test_select_starting_agent_uses_investigate_issue_button_intent(self) -> None:
        context = BotRunContext(
            channel_id=21,
            project_context="yearn",
            initial_button_intent="investigate_issue",
        )
        self.assertEqual(select_starting_agent("the claim button is missing", context), "triage")

    def test_select_starting_agent_routes_styfi_question_to_docs(self) -> None:
        context = BotRunContext(channel_id=3, project_context="yearn")
        self.assertEqual(select_starting_agent("Where do I see my stYFI position?", context), "docs")

    def test_select_starting_agent_keeps_free_form_withdrawal_in_triage(self) -> None:
        context = BotRunContext(channel_id=4, project_context="yearn")
        message = (
            "How do I withdraw from vault "
            "0x2222222222222222222222222222222222222222 "
            "on ethereum using wallet 0x1111111111111111111111111111111111111111?"
        )
        self.assertEqual(select_starting_agent(message, context), "triage")

    def test_follow_up_routing_reuses_last_specialist(self) -> None:
        channel_id = 9
        context = BotRunContext(channel_id=channel_id, project_context="yearn")
        investigation_state = get_or_create_ticket_investigation_state(channel_id)
        investigation_state.last_specialty = "data"
        try:
            agent_key = _select_ticket_starting_agent(
                "0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0",
                context,
                current_history=[{"role": "user", "content": "Previous issue context"}],
                investigation_state=investigation_state,
            )
            self.assertEqual(agent_key, "data")
        finally:
            ticket_investigation_states.pop(channel_id, None)

    def test_merge_explicit_evidence_into_state_tracks_chain_and_tx_hash(self) -> None:
        channel_id = 22
        investigation_state = get_or_create_ticket_investigation_state(channel_id)
        try:
            _merge_explicit_evidence_into_state(
                investigation_state,
                "Katana tx hash: 0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0",
            )
            self.assertEqual(investigation_state.known_chain, "katana")
            self.assertEqual(
                investigation_state.known_tx_hashes,
                ["0x87babcb5328cf17c6edb9027a29de1e32764306d6707669cabfb0436e11474d0"],
            )
        finally:
            ticket_investigation_states.pop(channel_id, None)


class RepoHelperTests(unittest.TestCase):
    def test_extract_artifact_refs_deduplicates_and_preserves_order(self) -> None:
        text = "segment:12 fact:4 segment:12 fact:7 fact:4"
        self.assertEqual(
            _extract_artifact_refs(text),
            ["segment:12", "fact:4", "fact:7"],
        )

    def test_repo_search_block_message_points_to_fetch(self) -> None:
        context = BotRunContext(
            channel_id=5,
            project_context="yearn",
        )
        context.repo_last_search_artifact_refs = ["segment:12", "fact:4"]
        message = _repo_search_block_message(context)

        self.assertIn("fetch_repo_artifacts_tool", message)
        self.assertIn("segment:12, fact:4", message)


class DynamicInstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_data_agent_system_prompt_includes_runtime_context(self) -> None:
        context = BotRunContext(
            channel_id=6,
            project_context="yearn",
            initial_button_intent="data_withdrawal_flow_start",
        )
        prompt = await yearn_data_agent.get_system_prompt(RunContextWrapper(context))

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn("initial_button_intent: data_withdrawal_flow_start", prompt)
        self.assertIn("project_context: yearn", prompt)

    async def test_triage_agent_system_prompt_includes_runtime_context(self) -> None:
        context = BotRunContext(
            channel_id=7,
            project_context="yearn",
            initial_button_intent="other_free_form",
        )
        prompt = await triage_agent.get_system_prompt(RunContextWrapper(context))

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn("initial_button_intent: other_free_form", prompt)
        self.assertIn("is_public_trigger: false", prompt)

    async def test_bug_agent_system_prompt_keeps_handoff_placeholder(self) -> None:
        context = BotRunContext(
            channel_id=8,
            project_context="yearn",
            initial_button_intent="bug_report",
        )
        prompt = await yearn_bug_triage_agent.get_system_prompt(RunContextWrapper(context))

        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertIn(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, prompt)
        self.assertIn("initial_button_intent: bug_report", prompt)
