import tests as _test_environment  # noqa: F401

import unittest
from types import SimpleNamespace
from unittest.mock import patch


import config
from state import (
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
    stopped_channels,
    stop_reasons_by_channel,
)
from handoff import (
    strip_handoff_placeholder,
)
from ticket_investigation.context import merge_explicit_evidence
from discord_support_runtime import (
    _classify_ticket_message_action,
    _extract_ticket_owner_user_id_from_messages,
    _maybe_recover_runtime_stopped_ticket_for_message,
)


class RoutingTests(unittest.TestCase):
    def test_extract_ticket_owner_user_id_from_messages_prefers_bot_opener_mention(
        self,
    ) -> None:
        human_message = SimpleNamespace(
            author=SimpleNamespace(bot=False),
            mentions=[SimpleNamespace(id=999, bot=False)],
        )
        bot_opener = SimpleNamespace(
            author=SimpleNamespace(bot=True),
            mentions=[
                SimpleNamespace(id=111, bot=True),
                SimpleNamespace(id=222, bot=False),
            ],
        )

        self.assertEqual(
            _extract_ticket_owner_user_id_from_messages([human_message, bot_opener]),
            222,
        )

    @patch("discord_support_runtime._is_support_staff_member")
    def test_classify_ticket_message_action_active_ticket_separates_owner_and_staff(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.side_effect = lambda author: author.id == 2
        owner = SimpleNamespace(id=1)
        staff = SimpleNamespace(id=2)
        stranger = SimpleNamespace(id=3)

        self.assertEqual(
            _classify_ticket_message_action(
                author=owner,
                content="help",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "process",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=owner,
                content="y: this is ordinary owner text",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "process",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="I have thoughts",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "staff_takeover",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="Y: explain the gas requirement",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "staff_summon",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=stranger,
                content="y: same issue",
                ticket_owner_user_id=1,
                stopped=False,
            ),
            "ignore",
        )

    @patch("discord_support_runtime._is_support_staff_member")
    def test_classify_ticket_message_action_stopped_ticket_supports_staff_commands(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.side_effect = lambda author: author.id == 2
        owner = SimpleNamespace(id=1)
        staff = SimpleNamespace(id=2)

        self.assertEqual(
            _classify_ticket_message_action(
                author=owner,
                content="y: try again",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "ignore",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="please retry",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "staff_takeover",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="y: retry with vault evidence",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "staff_summon",
        )
        self.assertEqual(
            _classify_ticket_message_action(
                author=staff,
                content="y:   ",
                ticket_owner_user_id=1,
                stopped=True,
            ),
            "staff_summon_usage",
        )

    def test_strip_handoff_placeholder_preserves_newlines(self) -> None:
        text = (
            "Main difference from Aave:\n"
            "- Flex uses fixed rates.\n"
            "- Aave uses variable rates.\n\n"
            f"{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}"
        )
        self.assertEqual(
            strip_handoff_placeholder(text),
            "Main difference from Aave:\n- Flex uses fixed rates.\n- Aave uses variable rates.",
        )

    @patch("discord_support_runtime._is_support_staff_member")
    def test_runtime_stopped_ticket_recovers_for_owner_message(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = False
        channel_id = 77
        owner = SimpleNamespace(id=1, name="owner")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "runtime_error"
        try:
            self.assertTrue(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=owner,
                    ticket_owner_user_id=1,
                )
            )
            self.assertNotIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, stop_reasons_by_channel)
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)

    @patch("discord_support_runtime._is_support_staff_member")
    def test_runtime_stopped_ticket_recovers_for_known_owner_with_contributor_role(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = True
        channel_id = 177
        owner = SimpleNamespace(id=1, name="owner")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "runtime_error"
        try:
            self.assertTrue(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=owner,
                    ticket_owner_user_id=1,
                )
            )
            self.assertNotIn(channel_id, stopped_channels)
            self.assertNotIn(channel_id, stop_reasons_by_channel)
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)

    @patch("discord_support_runtime._is_support_staff_member")
    def test_runtime_stopped_ticket_does_not_recover_for_non_owner_contributor(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = True
        channel_id = 277
        contributor = SimpleNamespace(id=2, name="contributor")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "runtime_error"
        try:
            self.assertFalse(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=contributor,
                    ticket_owner_user_id=1,
                )
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "runtime_error")
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)

    @patch("discord_support_runtime._is_support_staff_member")
    def test_boundary_stopped_ticket_does_not_recover_for_owner_message(
        self,
        mock_is_support_staff_member,
    ) -> None:
        mock_is_support_staff_member.return_value = False
        channel_id = 78
        owner = SimpleNamespace(id=1, name="owner")
        stopped_channels.add(channel_id)
        stop_reasons_by_channel[channel_id] = "boundary_stop"
        try:
            self.assertFalse(
                _maybe_recover_runtime_stopped_ticket_for_message(
                    channel_id=channel_id,
                    author=owner,
                    ticket_owner_user_id=1,
                )
            )
            self.assertIn(channel_id, stopped_channels)
            self.assertEqual(stop_reasons_by_channel[channel_id], "boundary_stop")
        finally:
            stopped_channels.discard(channel_id)
            stop_reasons_by_channel.pop(channel_id, None)


    def test_merge_explicit_evidence_does_not_infer_chain_from_substring(self) -> None:
        channel_id = 30
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        try:
            merge_explicit_evidence(
                investigation_job,
                "The portfolio database is showing the wrong balance.",
            )

            self.assertIsNone(investigation_job.evidence.chain)
        finally:
            clear_ticket_investigation_job(channel_id)
