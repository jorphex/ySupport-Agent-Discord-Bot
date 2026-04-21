import unittest
from unittest.mock import patch

import tools_lib


class InspectOnchainBlockIdentifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_mode_supports_latest_minus_offset(self) -> None:
        observed_filter = {}

        class _FakeEth:
            block_number = 987654

            def get_logs(self, filter_params):
                observed_filter.update(filter_params)
                return []

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        with patch.dict(tools_lib.WEB3_INSTANCES, {"ethereum": _FakeWeb3()}, clear=True):
            result = await tools_lib.core_inspect_onchain(
                chain="ethereum",
                mode="logs",
                from_block="latest-50000",
                to_block="latest",
            )

        self.assertIn("Log query result", result)
        self.assertEqual(observed_filter["fromBlock"], 937654)
        self.assertEqual(observed_filter["toBlock"], "latest")
