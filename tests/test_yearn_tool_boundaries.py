import unittest
from unittest.mock import patch


from ysupport.tools import onchain_tools


class OnchainInspectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_tx_summary_profiles_contracts_and_formats_transfers(self) -> None:
        class _HexLike:
            def __init__(self, value: str) -> None:
                self._value = value

            def hex(self) -> str:
                return self._value

        class _FakeEth:
            def get_transaction_receipt(self, tx_hash: str) -> dict:
                return {
                    "status": 1,
                    "blockNumber": 27331428,
                    "gasUsed": 771090,
                    "logs": [
                        {
                            "address": "0x2222222222222222222222222222222222222222",
                            "logIndex": 0,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        }
                    ],
                }

            def get_transaction(self, tx_hash: str) -> dict:
                return {
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x3333333333333333333333333333333333333333",
                }

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        async def fake_inspect_contract_profile(
            web3_instance, contract_address: str, *, block_identifier=None
        ):
            self.assertEqual(block_identifier, 27331428)
            return {
                "address": contract_address,
                "symbol": "yvWBUSDT",
                "name": "Yearn Katana Vault",
                "decimals": 18,
                "asset": "0x4444444444444444444444444444444444444444",
                "asset_symbol": "USDT",
                "asset_decimals": 6,
                "kind": "erc4626_like",
                "has_code": True,
            }

        with patch.dict(
            onchain_tools.chain_access.WEB3_INSTANCES,
            {"katana": _FakeWeb3()},
            clear=True,
        ):
            with patch(
                'ysupport.tools.onchain_tools._decode_logs_with_abis',
                return_value=[
                    {
                        "event": "Transfer",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 0,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "from": "0x1111111111111111111111111111111111111111",
                            "to": "0x5555555555555555555555555555555555555555",
                            "value": 650914700000000000000,
                        },
                    }
                ],
            ):
                with patch(
                    'ysupport.tools.onchain_tools._inspect_contract_profile',
                    new=fake_inspect_contract_profile,
                ):
                    summary = await onchain_tools.core_inspect_onchain(
                        chain="katana",
                        mode="tx_summary",
                        tx_hash="0x" + "a" * 64,
                    )

        self.assertIn("Transaction summary", summary)
        self.assertIn("contracts_profiled", summary)
        self.assertIn("yvWBUSDT", summary)
        self.assertIn("650.9147", summary)
        self.assertIn("erc4626_like", summary)

    async def test_tx_investigate_summarizes_transfer_and_approval_activity(
        self,
    ) -> None:
        class _HexLike:
            def __init__(self, value: str) -> None:
                self._value = value

            def hex(self) -> str:
                return self._value

        class _FakeEth:
            def get_transaction_receipt(self, tx_hash: str) -> dict:
                return {
                    "status": 1,
                    "blockNumber": 27331428,
                    "gasUsed": 771090,
                    "logs": [
                        {
                            "address": "0x2222222222222222222222222222222222222222",
                            "logIndex": 0,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        },
                        {
                            "address": "0x3333333333333333333333333333333333333333",
                            "logIndex": 1,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        },
                    ],
                }

            def get_transaction(self, tx_hash: str) -> dict:
                return {
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x4444444444444444444444444444444444444444",
                }

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        async def fake_inspect_contract_profile(
            web3_instance, contract_address: str, *, block_identifier=None
        ):
            self.assertEqual(block_identifier, 27331428)
            if contract_address.lower() == "0x2222222222222222222222222222222222222222":
                return {
                    "address": contract_address,
                    "symbol": "frxUSD",
                    "name": "Frax USD",
                    "decimals": 18,
                    "asset": None,
                    "asset_symbol": None,
                    "asset_decimals": None,
                    "kind": "erc20_like",
                    "has_code": True,
                }
            return {
                "address": contract_address,
                "symbol": "yvWBUSDT",
                "name": "Yearn Katana Vault",
                "decimals": 18,
                "asset": "0x5555555555555555555555555555555555555555",
                "asset_symbol": "USDT",
                "asset_decimals": 6,
                "kind": "erc4626_like",
                "has_code": True,
            }

        with patch.dict(
            onchain_tools.chain_access.WEB3_INSTANCES,
            {"katana": _FakeWeb3()},
            clear=True,
        ):
            with patch(
                'ysupport.tools.onchain_tools._decode_logs_with_abis',
                return_value=[
                    {
                        "event": "Transfer",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 0,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "from": "0x1111111111111111111111111111111111111111",
                            "to": "0x3333333333333333333333333333333333333333",
                            "value": 1990379783000000000000,
                        },
                    },
                    {
                        "event": "Approval",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 1,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "owner": "0x1111111111111111111111111111111111111111",
                            "spender": "0x3333333333333333333333333333333333333333",
                            "value": 1990379783000000000000,
                        },
                    },
                ],
            ):
                with patch(
                    'ysupport.tools.onchain_tools._inspect_contract_profile',
                    new=fake_inspect_contract_profile,
                ):
                    investigation = await onchain_tools.core_inspect_onchain(
                        chain="katana",
                        mode="tx_investigate",
                        tx_hash="0x" + "a" * 64,
                    )

        self.assertIn("Transaction investigation", investigation)
        self.assertIn("user_transfers_out", investigation)
        self.assertIn("approvals", investigation)
        self.assertIn("Observed 1 transfer(s) out from the tx sender.", investigation)
        self.assertIn("Decoded 1 approval event(s).", investigation)
