import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import patch

import chain_access
import onchain_tools


class InspectOnchainBlockIdentifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_mode_preserves_genesis_block_identifier(self) -> None:
        observed = {}

        class _Call:
            def call(self, *, block_identifier=None):
                observed["block_identifier"] = block_identifier
                return 1

        class _Functions:
            def totalAssets(self):
                return _Call()

        class _Contract:
            functions = _Functions()

        class _FakeEth:
            def contract(self, *, address, abi):
                return _Contract()

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        with patch.dict(
            chain_access.WEB3_INSTANCES,
            {"ethereum": _FakeWeb3()},
            clear=True,
        ):
            result = await onchain_tools.core_inspect_onchain(
                chain="ethereum",
                mode="call",
                to_address="0x1111111111111111111111111111111111111111",
                function_signature="totalAssets()",
                output_types_json='["uint256"]',
                block_identifier="0",
            )

        self.assertEqual(observed["block_identifier"], 0)
        self.assertIn("block_identifier: 0", result)

    async def test_logs_mode_preserves_genesis_block_identifier(self) -> None:
        observed_filter = {}

        class _FakeEth:
            block_number = 987654

            def get_logs(self, filter_params):
                observed_filter.update(filter_params)
                return []

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        with patch.dict(
            chain_access.WEB3_INSTANCES,
            {"ethereum": _FakeWeb3()},
            clear=True,
        ):
            await onchain_tools.core_inspect_onchain(
                chain="ethereum",
                mode="logs",
                from_block="0",
                to_block="0x0",
            )

        self.assertEqual(observed_filter["fromBlock"], 0)
        self.assertEqual(observed_filter["toBlock"], 0)

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

        with patch.dict(
            chain_access.WEB3_INSTANCES,
            {"ethereum": _FakeWeb3()},
            clear=True,
        ):
            result = await onchain_tools.core_inspect_onchain(
                chain="ethereum",
                mode="logs",
                from_block="latest-50000",
                to_block="latest",
            )

        self.assertIn("Log query result", result)
        self.assertEqual(observed_filter["fromBlock"], 937654)
        self.assertEqual(observed_filter["toBlock"], "latest")
