import unittest
from unittest.mock import patch


import onchain_tools
import vault_search_tools


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
                "onchain_tools._decode_logs_with_abis",
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
                    "onchain_tools._inspect_contract_profile",
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
                "onchain_tools._decode_logs_with_abis",
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
                    "onchain_tools._inspect_contract_profile",
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


class SearchVaultsTests(unittest.IsolatedAsyncioTestCase):
    def test_format_single_vault_data_handles_nullable_apr_fields(self) -> None:
        formatted = vault_search_tools.format_single_vault_data_for_llm(
            {
                "name": "Nullable Vault",
                "symbol": "yvNULL",
                "address": "0x1111111111111111111111111111111111111111",
                "version": "3.0.4",
                "kind": "Multi Strategy",
                "description": "Test vault",
                "token": {
                    "name": "USD Coin",
                    "symbol": "USDC",
                    "address": "0x2222222222222222222222222222222222222222",
                },
                "tvl": {"price": None, "tvl": None},
                "apr": {
                    "netAPR": None,
                    "type": "v3:averaged",
                    "forwardAPR": {"netAPR": None, "type": "projection"},
                    "fees": {"performance": None, "management": None},
                    "points": {"weekAgo": None, "monthAgo": None, "inception": None},
                },
                "featuringScore": None,
                "info": {
                    "riskLevel": None,
                    "isRetired": False,
                    "isBoosted": False,
                    "isHighlighted": False,
                },
                "migration": {"available": False},
                "strategies": [
                    {
                        "name": "Strat One",
                        "address": "0x3333333333333333333333333333333333333333",
                        "status": "active",
                        "netAPR": None,
                        "details": {"lastReport": None},
                    }
                ],
                "staking": {
                    "available": True,
                    "source": "veYFI",
                    "address": "0x4444444444444444444444444444444444444444",
                    "rewards": [
                        {
                            "name": "Reward",
                            "symbol": "RWD",
                            "address": "0x5555555555555555555555555555555555555555",
                            "apr": None,
                            "isFinished": False,
                            "finishedAt": None,
                        }
                    ],
                },
            },
            1,
        )

        self.assertIn("Current Net APY (compounded): N/A", formatted)
        self.assertIn("Vault Fees: Performance=N/A, Management=N/A", formatted)
        self.assertIn(
            "Historical Net APY: Week Ago=N/A, Month Ago=N/A, Inception=N/A", formatted
        )
        self.assertIn("Individual APY: N/A", formatted)
        self.assertIn("APY: N/A", formatted)

    async def test_core_search_vaults_supports_all_query(self) -> None:
        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self) -> None:
                return None

            async def json(self):
                return [
                    {
                        "chainID": 1,
                        "address": "0x1111111111111111111111111111111111111111",
                        "name": "Yearn USDC",
                        "symbol": "yvUSDC",
                        "token": {
                            "name": "USD Coin",
                            "symbol": "USDC",
                            "address": "0x2222222222222222222222222222222222222222",
                            "decimals": 6,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.042,
                            "points": {
                                "weekAgo": 0.04,
                                "monthAgo": 0.041,
                                "inception": 0.05,
                            },
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 1000000},
                        "info": {
                            "riskLevel": 1,
                            "isRetired": False,
                            "isBoosted": False,
                            "isHighlighted": True,
                        },
                        "migration": {"available": False},
                        "strategies": [],
                    },
                    {
                        "chainID": 42161,
                        "address": "0x3333333333333333333333333333333333333333",
                        "name": "Yearn DAI",
                        "symbol": "yvDAI",
                        "token": {
                            "name": "Dai Stablecoin",
                            "symbol": "DAI",
                            "address": "0x4444444444444444444444444444444444444444",
                            "decimals": 18,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.031,
                            "points": {
                                "weekAgo": 0.029,
                                "monthAgo": 0.03,
                                "inception": 0.032,
                            },
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 500000},
                        "info": {
                            "riskLevel": 2,
                            "isRetired": False,
                            "isBoosted": False,
                            "isHighlighted": False,
                        },
                        "migration": {"available": False},
                        "strategies": [],
                    },
                ]

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch.object(
            vault_search_tools.aiohttp, "ClientSession", return_value=FakeSession()
        ):
            result = await vault_search_tools.core_search_vaults("all")

        self.assertIn("Found 2 Yearn vault(s) matching 'all'.", result)
        self.assertIn("Vault: Yearn USDC (yvUSDC)", result)
        self.assertIn("Vault: Yearn DAI (yvDAI)", result)

    async def test_core_search_vaults_recommended_only_filters_single_strategy_ys_vaults(
        self,
    ) -> None:
        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self) -> None:
                return None

            async def json(self):
                return [
                    {
                        "chainID": 1,
                        "address": "0x1111111111111111111111111111111111111111",
                        "name": "Yearn Saver Strategy",
                        "symbol": "ysUSDC",
                        "kind": "Single Strategy",
                        "featuringScore": 0.10,
                        "token": {
                            "name": "USD Coin",
                            "symbol": "USDC",
                            "address": "0x2222222222222222222222222222222222222222",
                            "decimals": 6,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.089,
                            "points": {
                                "weekAgo": 0.08,
                                "monthAgo": 0.085,
                                "inception": 0.09,
                            },
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 300000},
                        "info": {
                            "riskLevel": 3,
                            "isRetired": False,
                            "isBoosted": False,
                            "isHighlighted": False,
                        },
                        "migration": {"available": False},
                        "strategies": [{"name": "Only Strategy"}],
                    },
                    {
                        "chainID": 1,
                        "address": "0x3333333333333333333333333333333333333333",
                        "name": "Yearn USDC",
                        "symbol": "yvUSDC",
                        "kind": "Multi Strategy",
                        "featuringScore": 0.95,
                        "token": {
                            "name": "USD Coin",
                            "symbol": "USDC",
                            "address": "0x4444444444444444444444444444444444444444",
                            "decimals": 6,
                            "price": 1.0,
                        },
                        "apr": {
                            "netAPR": 0.042,
                            "points": {
                                "weekAgo": 0.04,
                                "monthAgo": 0.041,
                                "inception": 0.05,
                            },
                            "fees": {"performance": 0.1, "management": 0.02},
                        },
                        "tvl": {"tvl": 1000000},
                        "info": {
                            "riskLevel": 1,
                            "isRetired": False,
                            "isBoosted": False,
                            "isHighlighted": True,
                        },
                        "migration": {"available": False},
                        "strategies": [{"name": "Strat One"}, {"name": "Strat Two"}],
                    },
                ]

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch.object(
            vault_search_tools.aiohttp, "ClientSession", return_value=FakeSession()
        ):
            result = await vault_search_tools.core_search_vaults(
                "all", recommended_only=True
            )

        self.assertIn("Vault: Yearn USDC (yvUSDC)", result)
        self.assertNotIn("Vault: Yearn Saver Strategy (ysUSDC)", result)
