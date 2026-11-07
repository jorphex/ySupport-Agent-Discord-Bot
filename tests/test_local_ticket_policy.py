import tests as _test_environment  # noqa: F401

import unittest

from ysupport.bot.state import (
    TicketInvestigationJob,
)
from ysupport.bot.router import is_wallet_confirmation


class WalletConfirmationPolicyTests(unittest.TestCase):
    def test_confirmation_requires_an_explicit_complete_reply(self) -> None:
        self.assertTrue(is_wallet_confirmation("Yes, that's correct."))
        self.assertTrue(is_wallet_confirmation("use it"))
        self.assertFalse(is_wallet_confirmation("Yesterday the vault worked."))
        self.assertFalse(is_wallet_confirmation("That is the incorrect address."))
        self.assertFalse(is_wallet_confirmation("I did not confirm that address."))


class InvestigationJobTests(unittest.TestCase):
    def test_job_clears_current_specialty_when_non_specialist_turn_completes(
        self,
    ) -> None:
        job = TicketInvestigationJob(
            channel_id=78, current_specialty="bug", last_specialty="bug"
        )
        job.remember_withdrawal_target(
            "katana",
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )

        job.complete_specialist_turn(None)

        self.assertIsNone(job.current_specialty)
        self.assertEqual(job.last_specialty, "bug")
        self.assertIsNone(job.evidence.withdrawal_target_chain)
        self.assertIsNone(job.evidence.withdrawal_target_vault)

    def test_job_clears_withdrawal_target_when_turn_leaves_data_lane(self) -> None:
        job = TicketInvestigationJob(channel_id=79)
        job.remember_withdrawal_target(
            "katana",
            "0x80c34BD3A3569E126e7055831036aa7b212cB159",
        )

        job.complete_specialist_turn("docs")

        self.assertEqual(job.current_specialty, "docs")
        self.assertIsNone(job.evidence.withdrawal_target_chain)
        self.assertIsNone(job.evidence.withdrawal_target_vault)
